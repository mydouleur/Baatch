// model.h — Ornith-1.5-35B-A3B（qwen35moe）层前向（decode 单 token）
#pragma once
#include "baa.h"
#include "kernels.h"
#include "kernels_xbf.h"
#include "npu.h"
#include <vector>
#include <string>

// 模型超参数（config.json text_config，硬编码）
namespace Cfg {
  constexpr int H = 2048;          // hidden
  constexpr int NLAYERS = 40;
  constexpr int FULL_INTERVAL = 4; // (i+1)%4==0 → 全注意力
  // GDN
  constexpr int KH = 16, VH = 32, HD = 128;   // key/value 头数、头维
  constexpr int KEY_DIM = 2048, VAL_DIM = 4096, QKV_DIM = 8192, CONV_K = 4;
  // 全注意力
  constexpr int QH = 16, KVH = 2, AHD = 256;
  constexpr int QPROJ = 8192;      // q(16×256) + gate(16×256) 融合
  constexpr int ROPE_DIM = 64;     // partial_rotary 0.25×256
  constexpr float ROPE_THETA = 1e7f;
  // MoE
  constexpr int NE = 256, TOPK = 8, MI = 512;
  constexpr float EPS = 1e-6f;
  constexpr int VOCAB = 248320;
}  // namespace Cfg

#include "profile.h"

static void cpu_mv_pretiled(const BaaTensor& t, const float* x, float* y);
static void cpu_prepare_pretiled(const BaaTensor& t);
static void cpu_mv_pretiled_fast(const BaaTensor& t, const float* x, float* y,
                                 Pool* pool = nullptr, bool npu_semantics = false);

struct GdnState {
  std::vector<float> conv;   // (CONV_K-1) × QKV_DIM
  std::vector<float> ssm;    // VH × HD × HD
  GdnState() : conv((Cfg::CONV_K - 1) * Cfg::QKV_DIM, 0.f),
               ssm((size_t)Cfg::VH * Cfg::HD * Cfg::HD, 0.f) {}
};

struct AttnState {  // 一个全注意力层的 KV cache
  std::vector<float> k, v;   // [pos][KVH*AHD]
  int len = 0;
};

static inline std::string LN(int l, const char* suffix) {
  return "layers." + std::to_string(l) + "." + suffix;
}

static inline bool env_flag(const char* name) {
  const char* value = getenv(name);
  return value && *value && *value != '0';
}
static inline bool env_flag_default(const char* name) {
  const char* value = getenv(name);
  return !value || *value != '0';
}

struct Model {
  Baa baa;
  Pool pool;
  NpuBackend* npu = nullptr;   // 挂载则密集预平铺张量走 NPU
  std::vector<float> xq_buf_f;                 // 激活量化暂存
  std::vector<int8_t> xq;
  std::vector<float> sumxq;
  float sx = 1.f;

  void open(const std::string& p) { baa.open(p); }
  const BaaTensor& T(const std::string& n) const { return baa.at(n); }
  bool is_gdn(int l) const { return (l + 1) % Cfg::FULL_INTERVAL != 0; }

  std::vector<int8_t> xe, xo, xq4;
  // GEMV 包装：按 scheme 分发（0=BF16 5=F32 3=XBF4 4=XBF2）
  void mv(const BaaTensor& w, const float* x, float* y) {
    mv_named(w, "", x, y);
  }
  void mv_out(const BaaTensor& w, const float* x, float* y) {
    if ((env_flag("CPU_OUT_PROJ") || !env_flag("NPU_OUT_PROJ")) && w.scheme == 3 && (w.flags & 1))
      cpu_mv_pretiled_fast(w, x, y);
    else mv(w, x, y);
  }
  void mv_out_prefill(const BaaTensor& w, const float* x, float* y) {
    if (!env_flag("NPU_OUT_PROJ") && w.scheme == 3 && (w.flags & 1)) cpu_mv_pretiled_fast(w, x, y);
    else mv(w, x, y);
  }
  void mv_named(const BaaTensor& w, const std::string& name, const float* x, float* y) {
    int R = (int)w.shape[w.ndim - 2], C = (int)w.shape[w.ndim - 1];
    bool expert_npu = npu && npu->xbf4_experts && name.find(".experts.") != std::string::npos;
    if (name.find(".experts.gate_up_proj.") != std::string::npos && env_flag("CPU_EXPERT_GATE_UP")) expert_npu = false;
    if (name.find(".experts.down_proj.") != std::string::npos && env_flag("CPU_EXPERT_DOWN")) expert_npu = false;
    if (w.scheme == 3 && ((w.flags & 1) || expert_npu) && npu && npu->ok) {
      if (npu->gemv(w, x, y)) return;
      if (w.flags & 1) {
        fprintf(stderr, "[npu] fallback CPU for %s\n", name.c_str());
        cpu_mv_pretiled(w, x, y);
        return;
      }
    }
    if (w.scheme == 3 && (w.flags & 1)) { cpu_mv_pretiled(w, x, y); return; }
    if (w.scheme == 0) { gemv_bf16(pool, w.bf16(), R, C, x, y); return; }
    if (w.scheme == 5) { gemv_f32(pool, w.f32(), R, C, x, y); return; }
    xq.resize(C); sumxq.resize(C / XBF_SB);
    sx = quant_xq(x, C, xq.data(), sumxq.data());
    if (w.scheme == 3) {
      xe.resize(C / 2); xo.resize(C / 2);
      xq_split4(xq.data(), C, xe.data(), xo.data());
      int half = (R / 2 + 15) & ~15;
      pool.run([&](int id) {
        if (id == 0) gemv_xbf4_rows(w.data, 0, half, C, xe.data(), xo.data(), sumxq.data(), sx, y);
        else gemv_xbf4_rows(w.data, half, R - half, C, xe.data(), xo.data(), sumxq.data(), sx, y);
      });
      return;
    }
    if (w.scheme == 4) {
      xq4.resize(C);
      xq_split2(xq.data(), C, xq4.data());
      const int8_t* q4[4] = {xq4.data(), xq4.data() + C/4, xq4.data() + C/2, xq4.data() + 3*C/4};
      int half = (R / 2 + 15) & ~15;
      pool.run([&](int id) {
        if (id == 0) gemv_xbf2_rows(w.data, 0, half, C, q4, sumxq.data(), sx, y);
        else gemv_xbf2_rows(w.data, half, R - half, C, q4, sumxq.data(), sx, y);
      });
      return;
    }
    throw std::runtime_error("unsupported scheme");
  }
  void mv_xbf2_quantized(const BaaTensor& w, const int8_t* xq4,
                         const float* sumxq, float sx, float* y) {
    int R = (int)w.shape[0], C = (int)w.shape[1];
    const int8_t* q4[4] = {xq4, xq4 + C/4, xq4 + C/2, xq4 + 3*C/4};
    int half = (R / 2 + 15) & ~15;
    pool.run([&](int id) {
      if (id == 0) gemv_xbf2_rows(w.data, 0, half, C, q4, sumxq, sx, y);
      else gemv_xbf2_rows(w.data, half, R - half, C, q4, sumxq, sx, y);
    });
  }
  bool npu_submit(const BaaTensor& w, const float* x, float* y, NpuBackend::Job& j) {
    return w.scheme == 3 && (w.flags & 1) && npu && npu->ok && npu->submit(w, x, y, j);
  }
  void npu_finish(const BaaTensor& w, const float* x, float* y,
                  NpuBackend::Job& j, const char* name) {
    if (npu->wait(j)) return;
    fprintf(stderr, "[npu] fallback CPU for %s\n", name);
    cpu_mv_pretiled(w, x, y);
  }
  // 专家 GEMV：v1 容器已按专家拆目录项，e 由调用方拼进名字——此接口仅兼容旧调用
  void mv_xbf3(const BaaTensor& t, int e, const float* x, float* y) { mv(t, x, y); }
  // scheme 感知 RMSNorm
  void nrm(const BaaTensor& t, const float* x, int n, float* y, bool plus1) {
    if (t.scheme == 5) rmsnorm_f32(x, t.f32(), n, y, Cfg::EPS, plus1);
    else rmsnorm(x, t.bf16(), n, y, Cfg::EPS, plus1);
  }
};

// CPU 读 TileFuse 预平铺 X4（fallback/对拍路径，非热路径）
static void cpu_mv_pretiled(const BaaTensor& t, const float* x, float* y) {
  int N = (int)t.shape[0], K = (int)t.shape[1];
  int physical_K = (t.flags & 2) ? 4096 : (t.flags & 4) ? 2048 : K;
  int physical_N = (t.flags & 2) || (t.flags & 8) ? 8192 : (t.flags & 4) ? 2048 : (N + 2047) & ~2047;
  for (int n = 0; n < N; n++) y[n] = 0.f;
  const uint8_t* tiles = t.data;
  int nlt_col = physical_N / 64, idx = 0;
  for (int cm = 0; cm < 8; cm++)
    for (int lt_col = cm; lt_col < nlt_col; lt_col += 8)
      for (int ltr = 0; ltr < physical_K / 128; ltr++, idx++) {
        if (lt_col >= N / 64) continue;
        const uint8_t* tp = tiles + (size_t)idx * 4352;
        const uint8_t* wpk = tp;
        const uint16_t* sc = (const uint16_t*)(tp + 4096);
        for (int str = 0; str < 16; str++)
          for (int stc = 0; stc < 8; stc++) {
            const uint8_t* sp = wpk + (str * 8 + stc) * 32;
            for (int i = 0; i < 8; i++)
              for (int j = 0; j < 8; j += 2) {
                uint8_t byte = sp[(i * 8 + j) / 2];
                int k0 = ltr * 128 + str * 8 + i;
                int n0 = lt_col * 64 + stc * 8 + j;
                if (k0 >= K) continue;
                float s0 = bf16_to_f32(sc[stc * 8 + j]);
                float s1 = bf16_to_f32(sc[stc * 8 + j + 1]);
                y[n0]     += ((int)(byte & 0xF) - 8) * s0 * x[k0];
                y[n0 + 1] += ((int)(byte >> 4) - 8) * s1 * x[k0];
              }
          }
      }
}

static void cpu_prepare_pretiled(const BaaTensor& t) {
  int N = (int)t.shape[0], K = (int)t.shape[1];
  int physical_K = (t.flags & 2) ? 4096 : (t.flags & 4) ? 2048 : K;
  int physical_N = (t.flags & 2) || (t.flags & 8) ? 8192 : (t.flags & 4) ? 2048 : (N + 2047) & ~2047;
  if (t.cpu_rows.empty()) {
    t.cpu_rows.resize((size_t)N * (K / 128) * 66);
    int nlt_col = physical_N / 64, idx = 0;
    for (int cm = 0; cm < 8; cm++) {
      for (int lt_col = cm; lt_col < nlt_col; lt_col += 8) {
        for (int ltr = 0; ltr < physical_K / 128; ltr++, idx++) {
          if (lt_col >= (N + 63) / 64 || ltr >= K / 128) continue;
          const uint8_t* tp = t.data + (size_t)idx * 4352;
          const uint16_t* sc = (const uint16_t*)(tp + 4096);
          alignas(64) uint8_t weights[64 * 128];
          for (int str = 0; str < 16; str++) {
            for (int i = 0; i < 8; i++) {
              for (int stc = 0; stc < 8; stc++) {
                const uint32_t packed = *reinterpret_cast<const uint32_t*>(tp + (str * 8 + stc) * 32 + i * 4);
                for (int lane = 0; lane < 4; lane++) {
                  weights[(stc * 8 + lane * 2) * 128 + str * 8 + i] = (packed >> (lane * 8)) & 0xF;
                  weights[(stc * 8 + lane * 2 + 1) * 128 + str * 8 + i] = (packed >> (lane * 8 + 4)) & 0xF;
                }
              }
            }
          }
          for (int n = 0; n < 64 && lt_col * 64 + n < N; n++) {
            uint8_t* row = t.cpu_rows.data() + ((size_t)(lt_col * 64 + n) * (K / 128) + ltr) * 66;
            row[0] = sc[n] & 0xFF;
            row[1] = sc[n] >> 8;
            for (int k = 0; k < 128; k += 2)
              row[2 + k / 2] = weights[n * 128 + k] | (weights[n * 128 + k + 1] << 4);
          }
        }
      }
    }
  }
}

static void cpu_mv_pretiled_fast(const BaaTensor& t, const float* x, float* y,
                                 Pool* pool, bool npu_semantics) {
  int N = (int)t.shape[0], K = (int)t.shape[1];
  cpu_prepare_pretiled(t);
  std::vector<int8_t> q(K), xe(K / 2), xo(K / 2);
  std::vector<float> sums(K / XBF_SB);
  float sx = quant_xq(x, K, q.data(), sums.data());
  xq_split4(q.data(), K, xe.data(), xo.data());
  std::vector<int> tile_sums(K / 128);
  for (int b = 0; b < K / 128; b++)
    for (int i = 0; i < 128; i++) tile_sums[b] += q[b * 128 + i];
  auto run_rows = [&](const uint8_t* rows, int count, float* out) {
    if (npu_semantics)
      gemv_xbf4_128_rows_npu(rows, count, K, xe.data(), xo.data(), tile_sums.data(), sx, out);
    else
      gemv_xbf4_128_rows(rows, count, K, xe.data(), xo.data(), tile_sums.data(), sx, out);
  };
  if (pool) {
    size_t row_bytes = (size_t)(K / 128) * 66;
    pool->run_all([&](int id) {
      int begin = N * id / pool->workers, end = N * (id + 1) / pool->workers;
      run_rows(t.cpu_rows.data() + (size_t)begin * row_bytes, end - begin, y + begin);
    });
  } else {
    run_rows(t.cpu_rows.data(), N, y);
  }
}

// ---------------- GDN decode（linear_attn，照 qwen35moe/delta-net decode 数学） ----------------
static void gdn_decode(Model& m, int l, GdnState& st, const float* xin, float* xout) {
  if (getenv("TRACE")) fprintf(stderr, "gdn %d start\n", l);
  using namespace Cfg;
  std::vector<float> xn(H), qkv(QKV_DIM), z(VAL_DIM), beta(VH), alpha(VH);
  auto pt = prof_now();
  m.nrm(m.T(LN(l, "input_layernorm.weight")), xin, H, xn.data(), true);
  if (getenv("TRACE")) fprintf(stderr, "gdn %d norm done\n", l);
  const auto& qw = m.T(LN(l, "linear_attn.in_proj_qkv.weight"));
  const auto& zw = m.T(LN(l, "linear_attn.in_proj_z.weight"));
  const BaaTensor* fused_weights[] = {&qw, &zw};
  float* fused_outputs[] = {qkv.data(), z.data()};
  NpuBackend::Job qjob;
  if (!env_flag("NPU_DECODE_PROJ")) {
    cpu_mv_pretiled_fast(qw, xn.data(), qkv.data(), &m.pool, true);
    cpu_mv_pretiled_fast(zw, xn.data(), z.data(), &m.pool, true);
    m.mv(m.T(LN(l, "linear_attn.in_proj_b.weight")), xn.data(), beta.data());
    m.mv(m.T(LN(l, "linear_attn.in_proj_a.weight")), xn.data(), alpha.data());
  } else if (m.npu && m.npu->ok && m.npu->gemv_fused(fused_weights, 2, xn.data(), fused_outputs)) {
    m.mv(m.T(LN(l, "linear_attn.in_proj_b.weight")), xn.data(), beta.data());
    m.mv(m.T(LN(l, "linear_attn.in_proj_a.weight")), xn.data(), alpha.data());
  } else if (m.npu_submit(qw, xn.data(), qkv.data(), qjob)) {
    m.mv(m.T(LN(l, "linear_attn.in_proj_b.weight")), xn.data(), beta.data());
    m.mv(m.T(LN(l, "linear_attn.in_proj_a.weight")), xn.data(), alpha.data());
    m.npu_finish(qw, xn.data(), qkv.data(), qjob, "in_proj_qkv");
    m.mv(zw, xn.data(), z.data());
  } else {
    m.mv(qw, xn.data(), qkv.data());
    if (getenv("TRACE")) fprintf(stderr, "gdn %d qkv done\n", l);
    m.mv(m.T(LN(l, "linear_attn.in_proj_z.weight")), xn.data(), z.data());
    if (getenv("TRACE")) fprintf(stderr, "gdn %d z done\n", l);
    m.mv(m.T(LN(l, "linear_attn.in_proj_b.weight")), xn.data(), beta.data());
    if (getenv("TRACE")) fprintf(stderr, "gdn %d b done\n", l);
    m.mv(m.T(LN(l, "linear_attn.in_proj_a.weight")), xn.data(), alpha.data());
    if (getenv("TRACE")) fprintf(stderr, "gdn %d a done\n", l);
  }
  if (getenv("TRACE")) fprintf(stderr, "gdn %d projections done\n", l);
  prof_add(g_prof.gdn_proj[l], pt); pt = prof_now();
  const float* dt_b = m.T(LN(l, "linear_attn.dt_bias")).f32();
  const float* a_log = m.T(LN(l, "linear_attn.A_log")).f32();
  float g[VH];
  for (int j = 0; j < VH; j++) {
    float a = softplusf(alpha[j] + dt_b[j]);
    g[j] = -a * expf(a_log[j]);   // ssm_a = -exp(A_log)（llama qwen35moe）
    beta[j] = sigmoidf(beta[j]);
  }
  // conv1d depthwise 因果（k=4）：state 3 个 + 当前 qkv
  const float* cw = m.T(LN(l, "linear_attn.conv1d.weight")).f32();  // [8192,1,4]
  float window[4];
  std::vector<float> qc(QKV_DIM);
  for (int c = 0; c < QKV_DIM; c++) {
    window[0] = st.conv[c]; window[1] = st.conv[QKV_DIM + c]; window[2] = st.conv[2 * QKV_DIM + c];
    window[3] = qkv[c];
    float s = 0;
    for (int t = 0; t < 4; t++) s += window[t] * cw[c * 4 + t];
    qc[c] = siluf(s);
  }
  if (getenv("TRACE")) fprintf(stderr, "gdn %d conv done\n", l);
  // conv state 左移
  memcpy(st.conv.data(), st.conv.data() + QKV_DIM, 2 * QKV_DIM * 4);
  memcpy(st.conv.data() + 2 * QKV_DIM, qkv.data(), QKV_DIM * 4);
  // 切 q/k/v + L2 norm（per 128 头：x/sqrt(mean(x²)+eps/n)/sqrt(n)）
  float* q = qc.data(); float* k = qc.data() + KEY_DIM; float* v = qc.data() + KEY_DIM * 2;
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
  { const float qs = 1.f / sqrtf((float)HD);   // llama delta-net: q 先乘 1/sqrt(S_k)
    for (int i = 0; i < KEY_DIM; i++) q[i] *= qs; }
  // delta rule（v 头 j 用 k 头 j%16 —— qwen3.5 交错序）
  std::vector<float> o(VAL_DIM, 0.f);
  auto run_heads = [&](int begin, int end) {
  for (int j = begin; j < end; j++) {
    float* s = st.ssm.data() + (size_t)j * HD * HD;
    const float* kj = k + (j / (VH / KH)) * HD;
    const float* qj = q + (j / (VH / KH)) * HD;
    const float* vj = v + j * HD;
    float decay = expf(g[j]);
    const __m512 vdecay = _mm512_set1_ps(decay);
    const __m512 vkj = _mm512_loadu_ps(kj);   // HD=128 固定
    const __m512 vqj = _mm512_loadu_ps(qj);
    float sk[HD], d[HD];
    for (int i = 0; i < HD; i++) {
      float* sr = s + i * HD;
      __m512 acc = _mm512_setzero_ps();
      for (int mm = 0; mm < HD; mm += 16) {
        __m512 v = _mm512_mul_ps(_mm512_loadu_ps(sr + mm), vdecay);
        _mm512_storeu_ps(sr + mm, v);
        acc = _mm512_fmadd_ps(v, _mm512_loadu_ps(kj + mm % HD), acc);  // 见下注
      }
      sk[i] = _mm512_reduce_add_ps(acc);
    }
    for (int i = 0; i < HD; i++) d[i] = (vj[i] - sk[i]) * beta[j];
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
  // gated norm：per-head RMSNorm(128) × w_norm × SiLU(z)
  const float* nw = m.T(LN(l, "linear_attn.norm.weight")).f32();
  std::vector<float> og(VAL_DIM);
  for (int j = 0; j < VH; j++) {
    float* p = o.data() + j * HD;
    float ss = 0;
    for (int i = 0; i < HD; i++) ss += p[i] * p[i];
    float inv = 1.f / sqrtf(ss / HD + EPS);
    for (int i = 0; i < HD; i++)
      og[j * HD + i] = p[i] * inv * nw[i] * siluf(z[j * HD + i]);
  }
  std::vector<float> dec(H);
  prof_add(g_prof.gdn_state[l], pt); pt = prof_now();
  m.mv_out(m.T(LN(l, "linear_attn.out_proj.weight")), og.data(), dec.data());
  prof_add(g_prof.gdn_out[l], pt);
  for (int i = 0; i < H; i++) xout[i] = xin[i] + dec[i];
  if (getenv("TRACE")) fprintf(stderr, "gdn %d done\n", l);
}

// ---------------- 全注意力 decode ----------------
static void attn_decode(Model& m, int l, AttnState& st, int pos, const float* xin, float* xout) {
  using namespace Cfg;
  std::vector<float> xn(H), qg(QPROJ), kk(KVH * AHD), vv(KVH * AHD);
  auto pt = prof_now();
  m.nrm(m.T(LN(l, "input_layernorm.weight")), xin, H, xn.data(), true);
  const auto& qw = m.T(LN(l, "self_attn.q_proj.weight"));
  const auto& kw = m.T(LN(l, "self_attn.k_proj.weight"));
  const auto& vw = m.T(LN(l, "self_attn.v_proj.weight"));
  bool kv_cpu = env_flag("CPU_KV") || !((kw.flags & 1) || (vw.flags & 1));
  auto run_kv = [&] {
    if (env_flag("CPU_KV")) {
      cpu_mv_pretiled(kw, xn.data(), kk.data());
      cpu_mv_pretiled(vw, xn.data(), vv.data());
    } else {
      m.mv(kw, xn.data(), kk.data());
      m.mv(vw, xn.data(), vv.data());
    }
  };
  const BaaTensor* fused_weights[] = {&qw, &kw, &vw};
  float* fused_outputs[] = {qg.data(), kk.data(), vv.data()};
  NpuBackend::Job qjob;
  if (!env_flag("NPU_DECODE_PROJ")) {
    cpu_mv_pretiled_fast(qw, xn.data(), qg.data(), &m.pool, true);
    cpu_mv_pretiled_fast(kw, xn.data(), kk.data(), &m.pool, true);
    cpu_mv_pretiled_fast(vw, xn.data(), vv.data(), &m.pool, true);
  } else if (!env_flag("CPU_KV") && m.npu && m.npu->ok &&
      m.npu->gemv_fused(fused_weights, 3, xn.data(), fused_outputs,
                        env_flag_default("NPU_WIDE_ATTN") ? 12288 : 0)) {
  } else if (kv_cpu && m.npu_submit(qw, xn.data(), qg.data(), qjob)) {
    run_kv();
    m.npu_finish(qw, xn.data(), qg.data(), qjob, "q_proj");
  } else {
    m.mv(qw, xn.data(), qg.data());
    run_kv();
  }
  prof_add(g_prof.attn_proj[l], pt); pt = prof_now();
  const float* qnw = m.T(LN(l, "self_attn.q_norm.weight")).f32();
  const float* knw = m.T(LN(l, "self_attn.k_norm.weight")).f32();
  // 逐头 RMSNorm（Q 用前半 256，gate 后半）
  for (int h = 0; h < QH; h++) {
    float* p = qg.data() + h * 2 * AHD;
    float ss = 0; for (int i = 0; i < AHD; i++) ss += p[i] * p[i];
    float inv = 1.f / sqrtf(ss / AHD + EPS);
    for (int i = 0; i < AHD; i++) p[i] = p[i] * inv * (1.f + qnw[i]);
  }
  for (int h = 0; h < KVH; h++) {
    float* p = kk.data() + h * AHD;
    float ss = 0; for (int i = 0; i < AHD; i++) ss += p[i] * p[i];
    float inv = 1.f / sqrtf(ss / AHD + EPS);
    for (int i = 0; i < AHD; i++) p[i] = p[i] * inv * (1.f + knw[i]);
  }
  // RoPE：前 ROPE_DIM 维，theta 1e7，interleaved（mrope_interleaved=true）
  float rope_cos[ROPE_DIM / 2], rope_sin[ROPE_DIM / 2];
  for (int i = 0; i < ROPE_DIM / 2; i++) {
    float freq = 1.f / powf(ROPE_THETA, 2.f * i / ROPE_DIM);
    float ang = pos * freq;
    rope_cos[i] = cosf(ang); rope_sin[i] = sinf(ang);
  }
  auto rope = [&](float* vec, int nheads) {
    for (int h = 0; h < nheads; h++) {
      float* p = vec + h * AHD;
      for (int i = 0; i < ROPE_DIM / 2; i++) {
        float ca = rope_cos[i], sa = rope_sin[i];
        float x0 = p[2 * i], x1 = p[2 * i + 1];
        p[2 * i] = x0 * ca - x1 * sa;
        p[2 * i + 1] = x0 * sa + x1 * ca;
      }
    }
  };
  rope(qg.data(), QH);       // gate 后半不旋转（qg 前 QH×256 是 q）——注意：rope 只作用 q 区
  rope(kk.data(), KVH);
  // KV 追加
  st.k.insert(st.k.end(), kk.begin(), kk.end());
  st.v.insert(st.v.end(), vv.begin(), vv.end());
  st.len++;
  // 注意力：16 q 头 → kv 头 h/8
  std::vector<float> attn(QH * AHD, 0.f), sc0(st.len), sc1(st.len);
  float kq_scale = 1.f / sqrtf((float)AHD);
  auto run_heads = [&](int begin, int end, std::vector<float>& sc) {
    for (int h = begin; h < end; h++) {
      const float* qh = qg.data() + h * 2 * AHD;
      int kvh = h / (QH / KVH);
      float mx = -1e30f;
      for (int p = 0; p < st.len; p++) {
        const float* kp = st.k.data() + (size_t)p * KVH * AHD + kvh * AHD;
        float s = 0;
        for (int i = 0; i < AHD; i++) s += qh[i] * kp[i];
        sc[p] = s * kq_scale;
        mx = std::max(mx, sc[p]);
      }
      float sum = 0;
      for (int p = 0; p < st.len; p++) { sc[p] = expf(sc[p] - mx); sum += sc[p]; }
      for (int p = 0; p < st.len; p++) {
        const float* vp = st.v.data() + (size_t)p * KVH * AHD + kvh * AHD;
        float w = sc[p] / sum;
        for (int i = 0; i < AHD; i++) attn[h * AHD + i] += w * vp[i];
      }
      // output gate：out × sigmoid(gate)
      const float* gate = qg.data() + h * 2 * AHD + AHD;
      for (int i = 0; i < AHD; i++) attn[h * AHD + i] *= sigmoidf(gate[i]);
    }
  };
  m.pool.run([&](int id) { run_heads(id * QH / 2, (id + 1) * QH / 2, id ? sc1 : sc0); });
  std::vector<float> dec(H);
  prof_add(g_prof.attn_state[l], pt); pt = prof_now();
  m.mv_out(m.T(LN(l, "self_attn.o_proj.weight")), attn.data(), dec.data());
  prof_add(g_prof.attn_out[l], pt);
  for (int i = 0; i < H; i++) xout[i] = xin[i] + dec[i];
}

// ---------------- MoE ----------------
static void moe(Model& m, int l, const float* xin, float* xout) {
  using namespace Cfg;
  std::vector<float> xn(H);
  m.nrm(m.T(LN(l, "post_attention_layernorm.weight")), xin, H, xn.data(), true);
  std::vector<float> logits(NE);
  m.mv(m.T(LN(l, "mlp.gate.weight")), xn.data(), logits.data());
  // softmax → top8 → 归一化
  float mx = *std::max_element(logits.begin(), logits.end());
  float sum = 0;
  for (int i = 0; i < NE; i++) { logits[i] = expf(logits[i] - mx); sum += logits[i]; }
  for (int i = 0; i < NE; i++) logits[i] /= sum;
  int top[TOPK]; float tw[TOPK];
  {
    std::vector<std::pair<float, int>> idx(NE);
    for (int i = 0; i < NE; i++) idx[i] = {logits[i], i};
    std::partial_sort(idx.begin(), idx.begin() + TOPK, idx.end(), std::greater<>());
    float wsum = 0;
    for (int t = 0; t < TOPK; t++) { top[t] = idx[t].second; tw[t] = idx[t].first; wsum += tw[t]; }
    for (int t = 0; t < TOPK; t++) tw[t] /= std::max(wsum, 1e-9f);
    static bool dbg = getenv("MOE_DBG") != nullptr;
    if (dbg) { fprintf(stderr, "  top8 L%d:", l); for (int t = 0; t < TOPK; t++) fprintf(stderr, " %d(%.3f)", top[t], tw[t]); fprintf(stderr, "\n"); }
  }
  std::vector<float> sg(MI), su(MI), sh(H), sgate(1);
  const auto& sgw = m.T(LN(l, "mlp.shared_expert.gate_proj.weight"));
  const auto& suw = m.T(LN(l, "mlp.shared_expert.up_proj.weight"));
  if (env_flag("CPU_SHARED_GATE_UP") || !env_flag("NPU_SHARED_GATE_UP")) {
    cpu_mv_pretiled(sgw, xn.data(), sg.data());
    cpu_mv_pretiled(suw, xn.data(), su.data());
  } else {
    m.mv(sgw, xn.data(), sg.data());
    m.mv(suw, xn.data(), su.data());
  }
  for (int i = 0; i < MI; i++) sg[i] = siluf(sg[i]) * su[i];
  const auto& sdw = m.T(LN(l, "mlp.shared_expert.down_proj.weight"));
  NpuBackend::Job sdjob;
  bool cpu_shared_down = env_flag("CPU_SHARED_DOWN") || !env_flag("NPU_SHARED_DOWN");
  bool sd_async = !cpu_shared_down && (!m.npu || !m.npu->xbf4_experts) &&
                  m.npu_submit(sdw, sg.data(), sh.data(), sdjob);
  if (cpu_shared_down) cpu_mv_pretiled(sdw, sg.data(), sh.data());
  else if (!sd_async) m.mv(sdw, sg.data(), sh.data());
  m.mv(m.T(LN(l, "mlp.shared_expert_gate.weight")), xn.data(), sgate.data());

  auto prof0 = std::chrono::steady_clock::now();
  // v1 容器：专家按粒度独立目录项（layers.N.mlp.experts.gate_up_proj.{e}），
  // scheme 逐专家读取——禁止按层假设统一位宽（docs v1 定案）
  std::vector<float> acc(H, 0.f), guv(2 * MI), hd(MI), dov(H);
  char nmbuf[160];
  for (int t = 0; t < TOPK; t++) {
    snprintf(nmbuf, 160, "layers.%d.mlp.experts.gate_up_proj.%d", l, top[t]);
    m.mv_named(m.T(nmbuf), nmbuf, xn.data(), guv.data());
    for (int i = 0; i < MI; i++) {
      float g = guv[i], u = guv[MI + i];
      hd[i] = siluf(g) * u;
    }
    snprintf(nmbuf, 160, "layers.%d.mlp.experts.down_proj.%d", l, top[t]);
    m.mv_named(m.T(nmbuf), nmbuf, hd.data(), dov.data());
    for (int i = 0; i < H; i++) acc[i] += tw[t] * dov[i];
  }
  auto prof1 = std::chrono::steady_clock::now();
  if (sd_async) m.npu_finish(sdw, sg.data(), sh.data(), sdjob, "shared_expert.down_proj");
  static bool prof = getenv("MOE_PROF") != nullptr;
  if (prof) {
    auto prof2 = std::chrono::steady_clock::now();
    fprintf(stderr, "[moe] L%d routed=%.2fms shared=%.2fms\n", l,
            std::chrono::duration<double>(prof1 - prof0).count() * 1e3,
            std::chrono::duration<double>(prof2 - prof1).count() * 1e3);
  }
  float g0 = sigmoidf(sgate[0]);
  for (int i = 0; i < H; i++) xout[i] = xin[i] + acc[i] + g0 * sh[i];
}
