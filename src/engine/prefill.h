#pragma once
#include "model.h"

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
  for (int t = 0; t < T; t++) {
    float* outputs[] = {qkv.data() + t * QKV_DIM, z.data() + t * VAL_DIM};
    bool done = false;
    if (t + 1 < T && env_flag("NPU_M2") && m.npu && m.npu->ok) {
      float* outputs1[] = {qkv.data() + (t + 1) * QKV_DIM, z.data() + (t + 1) * VAL_DIM};
      done = m.npu->gemv_fused_pair(weights, 2, xn.data() + t * H, xn.data() + (t + 1) * H,
                                    outputs, outputs1);
      if (done) {
        for (int u = t; u <= t + 1; u++) {
          m.mv(m.T(LN(l, "linear_attn.in_proj_b.weight")), xn.data() + u * H, beta.data() + u * VH);
          m.mv(m.T(LN(l, "linear_attn.in_proj_a.weight")), xn.data() + u * H, alpha.data() + u * VH);
        }
        t++;
      }
    }
    if (done) continue;
    if (!m.npu || !m.npu->ok || !m.npu->gemv_fused(weights, 2, xn.data() + t * H, outputs)) {
      m.mv(qw, xn.data() + t * H, qkv.data() + t * QKV_DIM);
      m.mv(zw, xn.data() + t * H, z.data() + t * VAL_DIM);
    }
    m.mv(m.T(LN(l, "linear_attn.in_proj_b.weight")), xn.data() + t * H, beta.data() + t * VH);
    m.mv(m.T(LN(l, "linear_attn.in_proj_a.weight")), xn.data() + t * H, alpha.data() + t * VH);
  }
  prof_add(g_prof.gdn_proj[l], pt); pt = prof_now();

  const float* dt_b = m.T(LN(l, "linear_attn.dt_bias")).f32();
  const float* a_log = m.T(LN(l, "linear_attn.A_log")).f32();
  const float* cw = m.T(LN(l, "linear_attn.conv1d.weight")).f32();
  const float* nw = m.T(LN(l, "linear_attn.norm.weight")).f32();
  std::vector<float> qc(QKV_DIM), o(VAL_DIM), og(VAL_DIM), dec(H);
  for (int t = 0; t < T; t++) {
    float* qkv_t = qkv.data() + t * QKV_DIM;
    for (int c = 0; c < QKV_DIM; c++) {
      float s = st.conv[c] * cw[c * 4]
              + st.conv[QKV_DIM + c] * cw[c * 4 + 1]
              + st.conv[2 * QKV_DIM + c] * cw[c * 4 + 2]
              + qkv_t[c] * cw[c * 4 + 3];
      qc[c] = siluf(s);
    }
    memcpy(st.conv.data(), st.conv.data() + QKV_DIM, 2 * QKV_DIM * 4);
    memcpy(st.conv.data() + 2 * QKV_DIM, qkv_t, QKV_DIM * 4);

    float* q = qc.data();
    float* k = qc.data() + KEY_DIM;
    float* v = qc.data() + KEY_DIM * 2;
    auto l2norm = [&](float* vec) {
      for (int h = 0; h < KH; h++) {
        float* p = vec + h * HD;
        float ss = 0;
        for (int i = 0; i < HD; i++) ss += p[i] * p[i];
        float inv = 1.f / sqrtf(ss / HD + EPS / HD) / sqrtf((float)HD);
        for (int i = 0; i < HD; i++) p[i] *= inv;
      }
    };
    l2norm(q); l2norm(k);
    for (int i = 0; i < KEY_DIM; i++) q[i] *= 1.f / sqrtf((float)HD);

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
      const __m512 vkj = _mm512_loadu_ps(kj);
      const __m512 vqj = _mm512_loadu_ps(qj);
      float sk[HD], d[HD];
      for (int i = 0; i < HD; i++) {
        float* sr = s + i * HD;
        __m512 acc = _mm512_setzero_ps();
        for (int mm = 0; mm < HD; mm += 16) {
          __m512 sv = _mm512_mul_ps(_mm512_loadu_ps(sr + mm), vdecay);
          _mm512_storeu_ps(sr + mm, sv);
          acc = _mm512_fmadd_ps(sv, _mm512_loadu_ps(kj + mm % HD), acc);
        }
        sk[i] = _mm512_reduce_add_ps(acc);
      }
      for (int i = 0; i < HD; i++) d[i] = (vj[i] - sk[i]) * b;
      for (int i = 0; i < HD; i++) {
        float* sr = s + i * HD;
        const __m512 vd = _mm512_set1_ps(d[i]);
        __m512 acc = _mm512_setzero_ps();
        for (int mm = 0; mm < HD; mm += 16) {
          __m512 updated = _mm512_fmadd_ps(_mm512_loadu_ps(kj + mm), vd, _mm512_loadu_ps(sr + mm));
          _mm512_storeu_ps(sr + mm, updated);
          acc = _mm512_fmadd_ps(updated, _mm512_loadu_ps(qj + mm), acc);
        }
        o[j * HD + i] = _mm512_reduce_add_ps(acc);
      }
    }
    };
    m.pool.run([&](int id) { run_heads(id * VH / 2, (id + 1) * VH / 2); });

    for (int j = 0; j < VH; j++) {
      float* p = o.data() + j * HD;
      float ss = 0;
      for (int i = 0; i < HD; i++) ss += p[i] * p[i];
      float inv = 1.f / sqrtf(ss / HD + EPS);
      for (int i = 0; i < HD; i++)
        og[j * HD + i] = p[i] * inv * nw[i] * siluf(z[t * VAL_DIM + j * HD + i]);
    }
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
  for (int t = 0; t < T; t++) {
    float* outputs[] = {qg.data() + t * QPROJ, kk.data() + t * KVH * AHD, vv.data() + t * KVH * AHD};
    bool done = false;
    int wide = env_flag_default("NPU_WIDE_ATTN") ? 12288 : 0;
    if (t + 1 < T && env_flag("NPU_M2") && m.npu && m.npu->ok) {
      float* outputs1[] = {qg.data() + (t + 1) * QPROJ, kk.data() + (t + 1) * KVH * AHD,
                           vv.data() + (t + 1) * KVH * AHD};
      done = m.npu->gemv_fused_pair(weights, 3, xn.data() + t * H, xn.data() + (t + 1) * H,
                                    outputs, outputs1, wide);
      if (done) t++;
    }
    if (done) continue;
    if (!m.npu || !m.npu->ok || !m.npu->gemv_fused(weights, 3, xn.data() + t * H, outputs, wide)) {
      m.mv(qw, xn.data() + t * H, qg.data() + t * QPROJ);
      m.mv(kw, xn.data() + t * H, kk.data() + t * KVH * AHD);
      m.mv(vw, xn.data() + t * H, vv.data() + t * KVH * AHD);
    }
  }
  prof_add(g_prof.attn_proj[l], pt); pt = prof_now();
  const float* qnw = m.T(LN(l, "self_attn.q_norm.weight")).f32();
  const float* knw = m.T(LN(l, "self_attn.k_norm.weight")).f32();
  auto rope = [&](float* vec, int nheads, int pos) {
    for (int h = 0; h < nheads; h++) {
      float* p = vec + h * AHD;
      for (int i = 0; i < ROPE_DIM / 2; i++) {
        float freq = 1.f / powf(ROPE_THETA, 2.f * i / ROPE_DIM);
        float ang = pos * freq;
        float ca = cosf(ang), sa = sinf(ang);
        float x0 = p[2 * i], x1 = p[2 * i + 1];
        p[2 * i] = x0 * ca - x1 * sa;
        p[2 * i + 1] = x0 * sa + x1 * ca;
      }
    }
  };
  for (int t = 0; t < T; t++) {
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
    rope(q, QH, pos0 + t);
    rope(k, KVH, pos0 + t);
    st.k.insert(st.k.end(), k, k + KVH * AHD);
    st.v.insert(st.v.end(), vv.data() + t * KVH * AHD, vv.data() + (t + 1) * KVH * AHD);
  }
  st.len += T;

  std::vector<float> attn(QH * AHD), dec(H);
  for (int t = 0; t < T; t++) {
    std::fill(attn.begin(), attn.end(), 0.f);
    const float* q = qg.data() + t * QPROJ;
    int len = pos0 + t + 1;
    auto run_heads = [&](int begin, int end) {
      for (int h = begin; h < end; h++) {
        const float* qh = q + h * 2 * AHD;
        int kvh = h / (QH / KVH);
        std::vector<float> sc(len);
        float mx = -1e30f;
        for (int p = 0; p < len; p++) {
          const float* kp = st.k.data() + (size_t)p * KVH * AHD + kvh * AHD;
          float s = 0;
          for (int i = 0; i < AHD; i++) s += qh[i] * kp[i];
          sc[p] = s / sqrtf((float)AHD);
          mx = std::max(mx, sc[p]);
        }
        float sum = 0;
        for (int p = 0; p < len; p++) { sc[p] = expf(sc[p] - mx); sum += sc[p]; }
        for (int p = 0; p < len; p++) {
          const float* vp = st.v.data() + (size_t)p * KVH * AHD + kvh * AHD;
          float w = sc[p] / sum;
          for (int i = 0; i < AHD; i++) attn[h * AHD + i] += w * vp[i];
        }
        const float* gate = q + h * 2 * AHD + AHD;
        for (int i = 0; i < AHD; i++) attn[h * AHD + i] *= sigmoidf(gate[i]);
      }
    };
    m.pool.run([&](int id) { run_heads(id * QH / 2, (id + 1) * QH / 2); });
    prof_add(g_prof.attn_state[l], pt); pt = prof_now();
    m.mv_out_prefill(m.T(LN(l, "self_attn.o_proj.weight")), attn.data(), dec.data());
    prof_add(g_prof.attn_out[l], pt); pt = prof_now();
    for (int i = 0; i < H; i++) xout[t * H + i] = xin[t * H + i] + dec[i];
  }
}

static void attn_prefill(Model& m, int l, AttnState& st, int pos,
                         const float* xin, int T, float* xout) {
  for (int base = 0; base < T; base += 64) {
    int n = std::min(64, T - base);
    attn_prefill_chunk(m, l, st, pos + base, xin + (size_t)base * Cfg::H, n,
                       xout + (size_t)base * Cfg::H);
  }
}
