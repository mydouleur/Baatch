// model.h — Ornith-1.5-35B-A3B（qwen35moe）层前向（decode 单 token）
#pragma once
#include "baa.h"
#include "kernels.h"
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

  // GEMV 包装：bf16 张量 [out,in]，y = W@x
  void mv(const BaaTensor& w, const float* x, float* y) {
    int R = (int)w.shape[w.ndim - 2], C = (int)w.shape[w.ndim - 1];
    gemv_bf16(pool, w.bf16(), R, C, x, y);
  }
  // XBF3 专家 GEMV：t 为 3D [E, out, in] 或 2D
  void mv_xbf3(const BaaTensor& t, int e, const float* x, float* y) {
    int R, C; const uint8_t* base;
    if (t.ndim == 3) {
      R = (int)t.shape[1]; C = (int)t.shape[2];
      base = t.data + (size_t)e * R * (C / XBF_SB) * 98;
    } else {
      R = (int)t.shape[0]; C = (int)t.shape[1];
      base = t.data;
    }
    xq.resize(C); sumxq.resize(C / XBF_SB);
    sx = quant_xq(x, C, xq.data(), sumxq.data());
    gemv_xbf3(pool, base, R, C, xq.data(), sumxq.data(), sx, y);
  }
};

// ---------------- GDN decode（linear_attn，照 qwen35moe/delta-net decode 数学） ----------------
static void gdn_decode(Model& m, int l, GdnState& st, const float* xin, float* xout) {
  using namespace Cfg;
  std::vector<float> xn(H), qkv(QKV_DIM), z(VAL_DIM), beta(VH), alpha(VH);
  rmsnorm(xin, m.T(LN(l, "input_layernorm.weight")).bf16(), H, xn.data());
  m.mv(m.T(LN(l, "linear_attn.in_proj_qkv.weight")), xn.data(), qkv.data());
  m.mv(m.T(LN(l, "linear_attn.in_proj_z.weight")), xn.data(), z.data());
  m.mv(m.T(LN(l, "linear_attn.in_proj_b.weight")), xn.data(), beta.data());
  m.mv(m.T(LN(l, "linear_attn.in_proj_a.weight")), xn.data(), alpha.data());
  const uint16_t* dt_b = m.T(LN(l, "linear_attn.dt_bias")).bf16();
  const uint16_t* a_log = m.T(LN(l, "linear_attn.A_log")).bf16();
  float g[VH];
  for (int j = 0; j < VH; j++) {
    float a = softplusf(alpha[j] + bf16_to_f32(dt_b[j]));
    g[j] = -a * expf(bf16_to_f32(a_log[j]));   // ssm_a = -exp(A_log)（llama qwen35moe）
    beta[j] = sigmoidf(beta[j]);
  }
  // conv1d depthwise 因果（k=4）：state 3 个 + 当前 qkv
  const uint16_t* cw = m.T(LN(l, "linear_attn.conv1d.weight")).bf16();  // [8192,1,4]
  float window[4];
  std::vector<float> qc(QKV_DIM);
  for (int c = 0; c < QKV_DIM; c++) {
    window[0] = st.conv[c]; window[1] = st.conv[QKV_DIM + c]; window[2] = st.conv[2 * QKV_DIM + c];
    window[3] = qkv[c];
    float s = 0;
    for (int t = 0; t < 4; t++) s += window[t] * bf16_to_f32(cw[c * 4 + t]);
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
    const float* kj = k + (j % KH) * HD;
    const float* qj = q + (j % KH) * HD;
    const float* vj = v + j * HD;
    float decay = expf(g[j]);
    float sk[HD], d[HD];
    for (int i = 0; i < HD; i++) {
      float* sr = s + i * HD;
      float acc = 0;
      for (int mm = 0; mm < HD; mm++) sr[mm] *= decay;
      for (int mm = 0; mm < HD; mm++) acc += sr[mm] * kj[mm];
      sk[i] = acc;
    }
    for (int i = 0; i < HD; i++) d[i] = (vj[i] - sk[i]) * beta[j];
    for (int i = 0; i < HD; i++) {
      float* sr = s + i * HD;
      for (int mm = 0; mm < HD; mm++) sr[mm] += kj[mm] * d[i];
    }
    for (int i = 0; i < HD; i++) {
      const float* sr = s + i * HD;
      float acc = 0;
      for (int mm = 0; mm < HD; mm++) acc += sr[mm] * qj[mm];
      o[j * HD + i] = acc;
    }
  }
  // gated norm：per-head RMSNorm(128) × w_norm × SiLU(z)
  const uint16_t* nw = m.T(LN(l, "linear_attn.norm.weight")).bf16();
  std::vector<float> og(VAL_DIM);
  for (int j = 0; j < VH; j++) {
    float* p = o.data() + j * HD;
    float ss = 0;
    for (int i = 0; i < HD; i++) ss += p[i] * p[i];
    float inv = 1.f / sqrtf(ss / HD + EPS);
    for (int i = 0; i < HD; i++)
      og[j * HD + i] = p[i] * inv * bf16_to_f32(nw[i]) * siluf(z[j * HD + i]);
  }
  std::vector<float> dec(H);
  m.mv(m.T(LN(l, "linear_attn.out_proj.weight")), og.data(), dec.data());
  for (int i = 0; i < H; i++) xout[i] = xin[i] + dec[i];
}

// ---------------- 全注意力 decode ----------------
static void attn_decode(Model& m, int l, AttnState& st, int pos, const float* xin, float* xout) {
  using namespace Cfg;
  std::vector<float> xn(H), qg(QPROJ), kk(KVH * AHD), vv(KVH * AHD);
  rmsnorm(xin, m.T(LN(l, "input_layernorm.weight")).bf16(), H, xn.data());
  m.mv(m.T(LN(l, "self_attn.q_proj.weight")), xn.data(), qg.data());
  m.mv(m.T(LN(l, "self_attn.k_proj.weight")), xn.data(), kk.data());
  m.mv(m.T(LN(l, "self_attn.v_proj.weight")), xn.data(), vv.data());
  const uint16_t* qnw = m.T(LN(l, "self_attn.q_norm.weight")).bf16();
  const uint16_t* knw = m.T(LN(l, "self_attn.k_norm.weight")).bf16();
  // 逐头 RMSNorm（Q 用前半 256，gate 后半）
  for (int h = 0; h < QH; h++) {
    float* p = qg.data() + h * 2 * AHD;
    float ss = 0; for (int i = 0; i < AHD; i++) ss += p[i] * p[i];
    float inv = 1.f / sqrtf(ss / AHD + EPS);
    for (int i = 0; i < AHD; i++) p[i] = p[i] * inv * bf16_to_f32(qnw[i]);
  }
  for (int h = 0; h < KVH; h++) {
    float* p = kk.data() + h * AHD;
    float ss = 0; for (int i = 0; i < AHD; i++) ss += p[i] * p[i];
    float inv = 1.f / sqrtf(ss / AHD + EPS);
    for (int i = 0; i < AHD; i++) p[i] = p[i] * inv * bf16_to_f32(knw[i]);
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
  rmsnorm(xin, m.T(LN(l, "post_attention_layernorm.weight")).bf16(), H, xn.data());
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
  }
  const BaaTensor& gu = m.T(LN(l, "mlp.experts.gate_up_proj"));
  const BaaTensor& dn = m.T(LN(l, "mlp.experts.down_proj"));
  std::vector<float> acc(H, 0.f), guv(2 * MI), hd(MI), dov(H);
  for (int t = 0; t < TOPK; t++) {
    m.mv_xbf3(gu, top[t], xn.data(), guv.data());
    for (int i = 0; i < MI; i++) {
      float g = guv[i], u = guv[MI + i];      // 前半 gate、后半 up
      hd[i] = siluf(g) * u;
    }
    m.mv_xbf3(dn, top[t], hd.data(), dov.data());
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
