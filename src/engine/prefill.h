#pragma once
#include "model.h"

static void check_m8_block(Model& m, int l, int t0, const char* kind,
                           const BaaTensor* const* weights, int count,
                           const float* x, float* const* ys, const int* y_strides) {
  double num = 0, den = 0;
  size_t exact_mismatches = 0;
  int worst_m = 0, worst_i = 0;
  double worst_rel = 0;
  for (int row = 0; row < 8; row++) {
    CpuXbf4Activation a(Cfg::H);
    a.prepare(x + (size_t)row * Cfg::H, Cfg::H);
    for (int i = 0; i < count; i++) {
      int N = (int)weights[i]->shape[0];
      std::vector<float> ref(N);
      cpu_mv_pretiled_fast(*weights[i], a, ref.data(), &m.pool, true);
      double local_num = 0, local_den = 0;
      for (int n = 0; n < N; n++) {
        double d = ys[i][(size_t)row * y_strides[i] + n] - ref[n];
        num += d * d; den += (double)ref[n] * ref[n];
        local_num += d * d; local_den += (double)ref[n] * ref[n];
        exact_mismatches += d != 0;
      }
      double local_rel = sqrt(local_num / (local_den + 1e-30));
      if (local_rel > worst_rel) {
        worst_rel = local_rel; worst_m = row; worst_i = i;
      }
    }
  }
  double rel_l2 = sqrt(num / (den + 1e-30));
  fprintf(stderr, "[m8-check] layer=%d block=%d kind=%s rel_l2=%.9g exact_mismatches=%zu %s\n",
          l, t0, kind, rel_l2, exact_mismatches, rel_l2 < 1e-5 ? "PASS" : "FAIL");
  if (rel_l2 >= 1e-5) {
    fprintf(stderr, "[m8-check] worst_row=%d projection=%d\n", worst_m, worst_i);
    throw std::runtime_error("M8 projection block mismatch");
  }
}

static void gdn_prefill_chunk(Model& m, int l, GdnState& st, const float* xin, int T, float* xout) {
  using namespace Cfg;
  std::vector<float> xn(T * H), qkv(T * QKV_DIM), z(T * VAL_DIM);
  std::vector<float> beta(T * VH), alpha(T * VH);
  const auto& qw = m.T(LN(l, "linear_attn.in_proj_qkv.weight"));
  const auto& zw = m.T(LN(l, "linear_attn.in_proj_z.weight"));
  auto pt = prof_now();
  for (int t = 0; t < T; t++)
    m.nrm(m.T(LN(l, "input_layernorm.weight")), xin + t * H, H, xn.data() + t * H, true);
  const BaaTensor* weights[] = {&qw, &zw};
  const int output_strides[] = {QKV_DIM, VAL_DIM};
  bool npu_proj = env_flag("NPU_PROJ") || env_flag("NPU_M8");
  for (int t = 0; t < T;) {
    float* outputs[] = {qkv.data() + (size_t)t * QKV_DIM, z.data() + (size_t)t * VAL_DIM};
    if (t + 8 <= T && env_flag("NPU_M8") && m.npu && m.npu->ok &&
        m.npu->gemv_fused_m8(weights, 2, xn.data() + (size_t)t * H, H, outputs, output_strides)) {
      if (env_flag("NPU_M8_CHECK"))
        check_m8_block(m, l, t, "gdn", weights, 2, xn.data() + (size_t)t * H,
                       outputs, output_strides);
      for (int u = t; u < t + 8; u++) {
        m.mv(m.T(LN(l, "linear_attn.in_proj_b.weight")), xn.data() + (size_t)u * H,
             beta.data() + (size_t)u * VH);
        m.mv(m.T(LN(l, "linear_attn.in_proj_a.weight")), xn.data() + (size_t)u * H,
             alpha.data() + (size_t)u * VH);
      }
      t += 8;
      continue;
    }
    if (!npu_proj) {
      cpu_mv_pretiled_fast(qw, xn.data() + (size_t)t * H, qkv.data() + (size_t)t * QKV_DIM, &m.pool);
      cpu_mv_pretiled_fast(zw, xn.data() + (size_t)t * H, z.data() + (size_t)t * VAL_DIM, &m.pool);
    } else if (!m.npu || !m.npu->ok ||
               !m.npu->gemv_fused(weights, 2, xn.data() + (size_t)t * H, outputs)) {
      m.mv(qw, xn.data() + (size_t)t * H, qkv.data() + (size_t)t * QKV_DIM);
      m.mv(zw, xn.data() + (size_t)t * H, z.data() + (size_t)t * VAL_DIM);
    }
    m.mv(m.T(LN(l, "linear_attn.in_proj_b.weight")), xn.data() + (size_t)t * H,
         beta.data() + (size_t)t * VH);
    m.mv(m.T(LN(l, "linear_attn.in_proj_a.weight")), xn.data() + (size_t)t * H,
         alpha.data() + (size_t)t * VH);
    t++;
  }
  prof_add(g_prof.gdn_proj[l], pt); pt = prof_now();

  const float* dt_b = m.T(LN(l, "linear_attn.dt_bias")).f32();
  const float* a_log = m.T(LN(l, "linear_attn.A_log")).f32();
  const float* cw = m.T(LN(l, "linear_attn.conv1d.weight")).f32();
  const float* nw = m.T(LN(l, "linear_attn.norm.weight")).f32();
  std::vector<float> qc(QKV_DIM), o(VAL_DIM), og(VAL_DIM), dec(H), cwt(4 * QKV_DIM);
  for (int tap = 0; tap < 4; tap++)
    for (int c = 0; c < QKV_DIM; c++) cwt[tap * QKV_DIM + c] = cw[c * 4 + tap];
  for (int t = 0; t < T; t++) {
    auto st0 = prof_now();
    float* qkv_t = qkv.data() + t * QKV_DIM;
    m.pool.run([&](int id) {
      gdn_conv_step_prefill<QKV_DIM>(st.conv.data(), qkv_t, cwt.data(), qc.data(),
                                     id * QKV_DIM / 2, (id + 1) * QKV_DIM / 2);
    });

    float* q = qc.data();
    float* k = qc.data() + KEY_DIM;
    float* v = qc.data() + KEY_DIM * 2;
    m.pool.run([&](int id) {
      for (int h = id * KH; h < (id + 1) * KH; h++) {
        float* p = qc.data() + h * HD;
        float ss = 0;
        for (int i = 0; i < HD; i++) ss += p[i] * p[i];
        float inv = 1.f / sqrtf(ss / HD + EPS / HD) / sqrtf((float)HD);
        for (int i = 0; i < HD; i++) p[i] *= inv;
        if (h < KH)
          for (int i = 0; i < HD; i++) p[i] *= 1.f / sqrtf((float)HD);
      }
    });
    prof_add(g_prof.gdn_conv[l], st0); st0 = prof_now();

    std::fill(o.begin(), o.end(), 0.f);
    auto run_heads = [&](int begin, int end) {
    for (int j = begin; j < end; j++) {
      float a = softplusf(alpha[t * VH + j] + dt_b[j]);
      float g = -a * expf(a_log[j]);
      float b = sigmoidf(beta[t * VH + j]);
      float* s = st.ssm.data() + (size_t)j * HD * HD;
      const float* kj = k + (j / (VH / KH)) * HD;
      const float* qj = q + (j / (VH / KH)) * HD;
      const float* vj = v + j * HD;
      float decay = expf(g);
      const __m512 vdecay = _mm512_set1_ps(decay);
      for (int i = 0; i < HD; i++)
        o[j * HD + i] = gdn_recur_row_prefill<HD>(s + i * HD, kj, qj, vj[i], b, vdecay);
    }
    };
    m.pool.run([&](int id) { run_heads(id * VH / 2, (id + 1) * VH / 2); });
    prof_add(g_prof.gdn_recur[l], st0); st0 = prof_now();

    m.pool.run([&](int id) {
      for (int j = id * VH / 2; j < (id + 1) * VH / 2; j++) {
        float* p = o.data() + j * HD;
        float ss = 0;
        for (int i = 0; i < HD; i++) ss += p[i] * p[i];
        float inv = 1.f / sqrtf(ss / HD + EPS);
        gdn_output_norm_prefill<HD>(p, z.data() + t * VAL_DIM + j * HD, nw, inv, og.data() + j * HD);
      }
    });
    prof_add(g_prof.gdn_norm[l], st0);
    prof_add(g_prof.gdn_state[l], pt); pt = prof_now();
    m.mv_out_prefill(m.T(LN(l, "linear_attn.out_proj.weight")), og.data(), dec.data());
    prof_add(g_prof.gdn_out[l], pt); pt = prof_now();
    for (int i = 0; i < H; i++) xout[t * H + i] = xin[t * H + i] + dec[i];
  }
}

static void gdn_prefill(Model& m, int l, GdnState& st, const float* xin, int T, float* xout) {
  for (int base = 0; base < T; base += 64) {
    int n = std::min(64, T - base);
    gdn_prefill_chunk(m, l, st, xin + (size_t)base * Cfg::H, n, xout + (size_t)base * Cfg::H);
  }
}

static void attn_prefill_chunk(Model& m, int l, AttnState& st, int pos0,
                               const float* xin, int T, float* xout) {
  using namespace Cfg;
  std::vector<float> xn(T * H), qg(T * QPROJ), kk(T * KVH * AHD), vv(T * KVH * AHD);
  const auto& qw = m.T(LN(l, "self_attn.q_proj.weight"));
  const auto& kw = m.T(LN(l, "self_attn.k_proj.weight"));
  const auto& vw = m.T(LN(l, "self_attn.v_proj.weight"));
  auto pt = prof_now();
  for (int t = 0; t < T; t++)
    m.nrm(m.T(LN(l, "input_layernorm.weight")), xin + t * H, H, xn.data() + t * H, true);
  const BaaTensor* weights[] = {&qw, &kw, &vw};
  const int output_strides[] = {QPROJ, KVH * AHD, KVH * AHD};
  bool npu_proj = env_flag("NPU_PROJ") || env_flag("NPU_M8");
  int wide = env_flag_default("NPU_WIDE_ATTN") ? 12288 : 0;
  for (int t = 0; t < T;) {
    float* outputs[] = {qg.data() + (size_t)t * QPROJ, kk.data() + (size_t)t * KVH * AHD,
                        vv.data() + (size_t)t * KVH * AHD};
    if (t + 8 <= T && wide == 12288 && env_flag("NPU_M8") && m.npu && m.npu->ok &&
        m.npu->gemv_fused_m8(weights, 3, xn.data() + (size_t)t * H, H, outputs, output_strides, wide)) {
      if (env_flag("NPU_M8_CHECK"))
        check_m8_block(m, l, t, "attn", weights, 3, xn.data() + (size_t)t * H,
                       outputs, output_strides);
      t += 8;
      continue;
    }
    if (!npu_proj) {
      cpu_mv_pretiled_fast(qw, xn.data() + (size_t)t * H, qg.data() + (size_t)t * QPROJ, &m.pool);
      cpu_mv_pretiled_fast(kw, xn.data() + (size_t)t * H, kk.data() + (size_t)t * KVH * AHD, &m.pool);
      cpu_mv_pretiled_fast(vw, xn.data() + (size_t)t * H, vv.data() + (size_t)t * KVH * AHD, &m.pool);
    } else if (!m.npu || !m.npu->ok ||
               !m.npu->gemv_fused(weights, 3, xn.data() + (size_t)t * H, outputs, wide)) {
      m.mv(qw, xn.data() + (size_t)t * H, qg.data() + (size_t)t * QPROJ);
      m.mv(kw, xn.data() + (size_t)t * H, kk.data() + (size_t)t * KVH * AHD);
      m.mv(vw, xn.data() + (size_t)t * H, vv.data() + (size_t)t * KVH * AHD);
    }
    t++;
  }
  prof_add(g_prof.attn_proj[l], pt); pt = prof_now();
  const float* qnw = m.T(LN(l, "self_attn.q_norm.weight")).f32();
  const float* knw = m.T(LN(l, "self_attn.k_norm.weight")).f32();
  std::vector<float> rope_cos(T * (ROPE_DIM / 2)), rope_sin(T * (ROPE_DIM / 2));
  for (int t = 0; t < T; t++) {
    for (int i = 0; i < ROPE_DIM / 2; i++) {
      float freq = 1.f / powf(ROPE_THETA, 2.f * i / ROPE_DIM);
      float ang = (pos0 + t) * freq;
      rope_cos[t * (ROPE_DIM / 2) + i] = cosf(ang);
      rope_sin[t * (ROPE_DIM / 2) + i] = sinf(ang);
    }
  }
  auto rope = [&](float* vec, int nheads, int t) {
    for (int h = 0; h < nheads; h++) {
      float* p = vec + h * AHD;
      for (int i = 0; i < ROPE_DIM / 2; i++) {
        float ca = rope_cos[t * (ROPE_DIM / 2) + i];
        float sa = rope_sin[t * (ROPE_DIM / 2) + i];
        float x0 = p[2 * i], x1 = p[2 * i + 1];
        p[2 * i] = x0 * ca - x1 * sa;
        p[2 * i + 1] = x0 * sa + x1 * ca;
      }
    }
  };
  st.k.reserve(st.k.size() + (size_t)T * KVH * AHD);
  st.v.reserve(st.v.size() + (size_t)T * KVH * AHD);
  for (int t = 0; t < T; t++) {
    auto st0 = prof_now();
    float* q = qg.data() + t * QPROJ;
    float* k = kk.data() + t * KVH * AHD;
    for (int h = 0; h < QH; h++) {
      float* p = q + h * 2 * AHD;
      float ss = 0; for (int i = 0; i < AHD; i++) ss += p[i] * p[i];
      float inv = 1.f / sqrtf(ss / AHD + EPS);
      for (int i = 0; i < AHD; i++) p[i] = p[i] * inv * (1.f + qnw[i]);
    }
    for (int h = 0; h < KVH; h++) {
      float* p = k + h * AHD;
      float ss = 0; for (int i = 0; i < AHD; i++) ss += p[i] * p[i];
      float inv = 1.f / sqrtf(ss / AHD + EPS);
      for (int i = 0; i < AHD; i++) p[i] = p[i] * inv * (1.f + knw[i]);
    }
    rope(q, QH, t);
    rope(k, KVH, t);
    prof_add(g_prof.attn_rope[l], st0); st0 = prof_now();
    st.k.insert(st.k.end(), k, k + KVH * AHD);
    st.v.insert(st.v.end(), vv.data() + t * KVH * AHD, vv.data() + (t + 1) * KVH * AHD);
    prof_add(g_prof.attn_cache[l], st0);
  }
  st.len += T;

  std::vector<float> attn(QH * AHD), dec(H);
  std::vector<float> sc0(pos0 + T), sc1(pos0 + T);
  double qk_ms[2]{}, softmax_ms[2]{}, pv_ms[2]{}, gate_ms[2]{};
  for (int t = 0; t < T; t++) {
    std::fill(attn.begin(), attn.end(), 0.f);
    const float* q = qg.data() + t * QPROJ;
    int len = pos0 + t + 1;
    auto run_heads = [&](int id, int begin, int end) {
      std::vector<float>& sc = id ? sc1 : sc0;
      for (int h = begin; h < end; h++) {
        const float* qh = q + h * 2 * AHD;
        int kvh = h / (QH / KVH);
        auto ht = prof_now();
        float mx = attn_qk_prefill<AHD>(qh, st.k.data() + kvh * AHD, len, KVH * AHD, sc.data());
        prof_add(qk_ms[id], ht); ht = prof_now();
        float sum = 0;
        for (int p = 0; p < len; p++) { sc[p] = expf(sc[p] - mx); sum += sc[p]; }
        prof_add(softmax_ms[id], ht); ht = prof_now();
        attn_pv_prefill<AHD>(st.v.data() + kvh * AHD, len, KVH * AHD, sc.data(), sum,
                             attn.data() + h * AHD);
        prof_add(pv_ms[id], ht); ht = prof_now();
        const float* gate = q + h * 2 * AHD + AHD;
        for (int i = 0; i < AHD; i++) attn[h * AHD + i] *= sigmoidf(gate[i]);
        prof_add(gate_ms[id], ht);
      }
    };
    m.pool.run([&](int id) { run_heads(id, id * QH / 2, (id + 1) * QH / 2); });
    prof_add(g_prof.attn_state[l], pt); pt = prof_now();
    m.mv_out_prefill(m.T(LN(l, "self_attn.o_proj.weight")), attn.data(), dec.data());
    prof_add(g_prof.attn_out[l], pt); pt = prof_now();
    for (int i = 0; i < H; i++) xout[t * H + i] = xin[t * H + i] + dec[i];
  }
  g_prof.attn_qk[l] += (qk_ms[0] + qk_ms[1]) * 0.5;
  g_prof.attn_softmax[l] += (softmax_ms[0] + softmax_ms[1]) * 0.5;
  g_prof.attn_pv[l] += (pv_ms[0] + pv_ms[1]) * 0.5;
  g_prof.attn_gate[l] += (gate_ms[0] + gate_ms[1]) * 0.5;
}

static void attn_prefill(Model& m, int l, AttnState& st, int pos,
                         const float* xin, int T, float* xout) {
  for (int base = 0; base < T; base += 64) {
    int n = std::min(64, T - base);
    attn_prefill_chunk(m, l, st, pos + base, xin + (size_t)base * Cfg::H, n,
                       xout + (size_t)base * Cfg::H);
  }
}
