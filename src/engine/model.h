// model.h — Ornith-1.5-35B-A3B（qwen35moe）层前向（decode 单 token）
#pragma once
#include "baa.h"
#include "kernels.h"
#include "kernels_xbf.h"
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

struct Model {
  Baa baa;
  Pool pool;
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
    int R = (int)w.shape[w.ndim - 2], C = (int)w.shape[w.ndim - 1];
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
  // 专家 GEMV：v1 容器已按专家拆目录项，e 由调用方拼进名字——此接口仅兼容旧调用
  void mv_xbf3(const BaaTensor& t, int e, const float* x, float* y) { mv(t, x, y); }
  // scheme 感知 RMSNorm
  void nrm(const BaaTensor& t, const float* x, int n, float* y, bool plus1) {
    if (t.scheme == 5) rmsnorm_f32(x, t.f32(), n, y, Cfg::EPS, plus1);
    else rmsnorm(x, t.bf16(), n, y, Cfg::EPS, plus1);
  }
};

// ---------------- GDN decode（linear_attn，照 qwen35moe/delta-net decode 数学） ----------------
static void gdn_decode(Model& m, int l, GdnState& st, const float* xin, float* xout) {
  using namespace Cfg;
  std::vector<float> xn(H), qkv(QKV_DIM), z(VAL_DIM), beta(VH), alpha(VH);
  m.nrm(m.T(LN(l, "input_layernorm.weight")), xin, H, xn.data(), true);
  m.mv(m.T(LN(l, "linear_attn.in_proj_qkv.weight")), xn.data(), qkv.data());
  m.mv(m.T(LN(l, "linear_attn.in_proj_z.weight")), xn.data(), z.data());
  m.mv(m.T(LN(l, "linear_attn.in_proj_b.weight")), xn.data(), beta.data());
  m.mv(m.T(LN(l, "linear_attn.in_proj_a.weight")), xn.data(), alpha.data());
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
  for (int j = 0; j < VH; j++) {
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
      for (int mm = 0; mm < HD; mm += 16)
        _mm512_storeu_ps(sr + mm, _mm512_fmadd_ps(_mm512_loadu_ps(kj + mm), vd, _mm512_loadu_ps(sr + mm)));
    }
    for (int i = 0; i < HD; i++) {
      const float* sr = s + i * HD;
      __m512 acc = _mm512_setzero_ps();
      for (int mm = 0; mm < HD; mm += 16)
        acc = _mm512_fmadd_ps(_mm512_loadu_ps(sr + mm), _mm512_loadu_ps(qj + mm % HD), acc);
      o[j * HD + i] = _mm512_reduce_add_ps(acc);
    }
  }
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
  m.mv(m.T(LN(l, "linear_attn.out_proj.weight")), og.data(), dec.data());
  for (int i = 0; i < H; i++) xout[i] = xin[i] + dec[i];
}

// ---------------- 全注意力 decode ----------------
static void attn_decode(Model& m, int l, AttnState& st, int pos, const float* xin, float* xout) {
  using namespace Cfg;
  std::vector<float> xn(H), qg(QPROJ), kk(KVH * AHD), vv(KVH * AHD);
  m.nrm(m.T(LN(l, "input_layernorm.weight")), xin, H, xn.data(), true);
  m.mv(m.T(LN(l, "self_attn.q_proj.weight")), xn.data(), qg.data());
  m.mv(m.T(LN(l, "self_attn.k_proj.weight")), xn.data(), kk.data());
  m.mv(m.T(LN(l, "self_attn.v_proj.weight")), xn.data(), vv.data());
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
  auto rope = [&](float* vec, int nheads) {
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
  rope(qg.data(), QH);       // gate 后半不旋转（qg 前 QH×256 是 q）——注意：rope 只作用 q 区
  rope(kk.data(), KVH);
  // KV 追加
  st.k.insert(st.k.end(), kk.begin(), kk.end());
  st.v.insert(st.v.end(), vv.begin(), vv.end());
  st.len++;
  // 注意力：16 q 头 → kv 头 h/8
  std::vector<float> attn(QH * AHD, 0.f);
  float kq_scale = 1.f / sqrtf((float)AHD);
  for (int h = 0; h < QH; h++) {
    const float* qh = qg.data() + h * 2 * AHD;
    int kvh = h / (QH / KVH);
    std::vector<float> sc(st.len);
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
  std::vector<float> dec(H);
  m.mv(m.T(LN(l, "self_attn.o_proj.weight")), attn.data(), dec.data());
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
  // v1 容器：专家按粒度独立目录项（layers.N.mlp.experts.gate_up_proj.{e}），
  // scheme 逐专家读取——禁止按层假设统一位宽（docs v1 定案）
  std::vector<float> acc(H, 0.f), guv(2 * MI), hd(MI), dov(H);
  char nmbuf[160];
  for (int t = 0; t < TOPK; t++) {
    snprintf(nmbuf, 160, "layers.%d.mlp.experts.gate_up_proj.%d", l, top[t]);
    m.mv(m.T(nmbuf), xn.data(), guv.data());
    for (int i = 0; i < MI; i++) {
      float g = guv[i], u = guv[MI + i];      // 前半 gate、后半 up
      hd[i] = siluf(g) * u;
    }
    snprintf(nmbuf, 160, "layers.%d.mlp.experts.down_proj.%d", l, top[t]);
    m.mv(m.T(nmbuf), hd.data(), dov.data());
    for (int i = 0; i < H; i++) acc[i] += tw[t] * dov[i];
  }
  // 共享专家（dense SwiGLU + sigmoid 门）
  std::vector<float> sg(MI), su(MI), sh(H), sgate(1);
  m.mv(m.T(LN(l, "mlp.shared_expert.gate_proj.weight")), xn.data(), sg.data());
  m.mv(m.T(LN(l, "mlp.shared_expert.up_proj.weight")), xn.data(), su.data());
  for (int i = 0; i < MI; i++) sg[i] = siluf(sg[i]) * su[i];
  m.mv(m.T(LN(l, "mlp.shared_expert.down_proj.weight")), sg.data(), sh.data());
  m.mv(m.T(LN(l, "mlp.shared_expert_gate.weight")), xn.data(), sgate.data());
  float g0 = sigmoidf(sgate[0]);
  for (int i = 0; i < H; i++) xout[i] = xin[i] + acc[i] + g0 * sh[i];
}
