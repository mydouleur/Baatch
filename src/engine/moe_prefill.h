#pragma once
#include "moe_scheduler.h"

struct MoePrefillWorkspace {
  int tokens;
  std::vector<float> xn, logits, sx, sums, sg, su, sh, sgate;
  std::vector<float> acc, guv, hd, dov, hsx, hsums, gather_sums, gather_sx;
  std::vector<int8_t> q, xe, xo, q4, hq, hxe, hxo, hq4, gather_q, gather_xe, gather_xo;
  std::vector<float> weights;
  std::array<std::vector<int>, Cfg::NE> expert_tokens;
  std::array<std::vector<int>, Cfg::NE> expert_slots;

  MoePrefillWorkspace(int T) : tokens(T) {
    xn.resize((size_t)T * Cfg::H); logits.resize((size_t)T * Cfg::NE);
    sx.resize(T); sums.resize((size_t)T * Cfg::H / XBF_SB);
    q.resize((size_t)T * Cfg::H); xe.resize((size_t)T * Cfg::H / 2);
    xo.resize((size_t)T * Cfg::H / 2); q4.resize((size_t)T * Cfg::H);
    weights.resize((size_t)T * Cfg::TOPK);
    gather_q.resize((size_t)T * Cfg::H); gather_xe.resize((size_t)T * Cfg::H / 2);
    gather_xo.resize((size_t)T * Cfg::H / 2);
    gather_sums.resize((size_t)T * Cfg::H / XBF_SB); gather_sx.resize(T);
    sg.resize((size_t)T * Cfg::MI); su.resize((size_t)T * Cfg::MI);
    sh.resize((size_t)T * Cfg::H); sgate.resize(T); acc.resize((size_t)T * Cfg::H);
    guv.resize((size_t)T * 2 * Cfg::MI); hd.resize((size_t)T * Cfg::MI);
    dov.resize((size_t)T * Cfg::H); hsx.resize(T);
    hsums.resize((size_t)T * Cfg::MI / XBF_SB);
    hq.resize((size_t)T * Cfg::MI); hxe.resize((size_t)T * Cfg::MI / 2);
    hxo.resize((size_t)T * Cfg::MI / 2); hq4.resize((size_t)T * Cfg::MI);
  }
};

static void moe_prefill(Model& m, const MoeLayerPlan& p, MoePrefillWorkspace& w,
                        const float* xin, float* xout) {
  using namespace Cfg;
  const int T = w.tokens;
  std::fill(w.acc.begin(), w.acc.end(), 0.f);
  for (auto& tokens : w.expert_tokens) tokens.clear();
  for (auto& slots : w.expert_slots) slots.clear();
  std::vector<std::pair<float, int>> order(NE);
  for (int tok = 0; tok < T; tok++) {
    const float* x = xin + (size_t)tok * H;
    float* xn = w.xn.data() + (size_t)tok * H;
    float* logits = w.logits.data() + (size_t)tok * NE;
    m.nrm(*p.norm, x, H, xn, true);
    if (p.router->scheme == 5) gemv_f32_rows(p.router->f32(), 0, NE, H, xn, logits);
    else m.mv(*p.router, xn, logits);
    float mx = *std::max_element(logits, logits + NE), sum = 0.f;
    for (int e = 0; e < NE; e++) { logits[e] = expf(logits[e] - mx); sum += logits[e]; }
    for (int e = 0; e < NE; e++) { logits[e] /= sum; order[e] = {logits[e], e}; }
    std::partial_sort(order.begin(), order.begin() + TOPK, order.end(), std::greater<>());
    float wsum = 0.f;
    for (int t = 0; t < TOPK; t++) wsum += order[t].first;
    for (int t = 0; t < TOPK; t++) {
      int e = order[t].second;
      w.weights[(size_t)tok * TOPK + t] = order[t].first / std::max(wsum, 1e-9f);
      w.expert_tokens[e].push_back(tok);
      w.expert_slots[e].push_back(t);
    }
    w.sx[tok] = quant_xq(xn, H, w.q.data() + (size_t)tok * H,
                         w.sums.data() + (size_t)tok * H / XBF_SB);
    xq_split4(w.q.data() + (size_t)tok * H, H,
              w.xe.data() + (size_t)tok * H / 2, w.xo.data() + (size_t)tok * H / 2);
    xq_split2(w.q.data() + (size_t)tok * H, H, w.q4.data() + (size_t)tok * H);
    cpu_mv_pretiled_fast(*p.shared_gate, xn, w.sg.data() + (size_t)tok * MI);
    cpu_mv_pretiled_fast(*p.shared_up, xn, w.su.data() + (size_t)tok * MI);
    float* sg = w.sg.data() + (size_t)tok * MI;
    const float* su = w.su.data() + (size_t)tok * MI;
    for (int i = 0; i < MI; i++) sg[i] = siluf(sg[i]) * su[i];
    cpu_mv_pretiled_fast(*p.shared_down, sg, w.sh.data() + (size_t)tok * H);
    if (p.shared_sigmoid->scheme == 5)
      gemv_f32_rows(p.shared_sigmoid->f32(), 0, 1, H, xn, w.sgate.data() + tok);
    else
      m.mv(*p.shared_sigmoid, xn, w.sgate.data() + tok);
  }

  for (int e = 0; e < NE; e++) {
    const auto& tokens = w.expert_tokens[e];
    int M = (int)tokens.size();
    if (!M) continue;
    const BaaTensor& gu = *p.gate_up[e];
    int R = (int)gu.shape[0], half = (R / 2 + 15) & ~15;
    auto& gx = w.gather_q;
    auto& gxe = w.gather_xe;
    auto& gxo = w.gather_xo;
    auto& gs = w.gather_sums;
    auto& gsx = w.gather_sx;
    for (int j = 0; j < M; j++) {
      int tok = tokens[j];
      memcpy(gx.data() + (size_t)j * H, w.q4.data() + (size_t)tok * H, H);
      memcpy(gxe.data() + (size_t)j * H / 2, w.xe.data() + (size_t)tok * H / 2, H / 2);
      memcpy(gxo.data() + (size_t)j * H / 2, w.xo.data() + (size_t)tok * H / 2, H / 2);
      memcpy(gs.data() + (size_t)j * H / XBF_SB,
             w.sums.data() + (size_t)tok * H / XBF_SB, H / XBF_SB * sizeof(float));
      gsx[j] = w.sx[tok];
    }
    if (gu.scheme == 3) {
      m.pool.run([&](int id) {
        int begin = id == 0 ? 0 : half, end = id == 0 ? half : R;
        gemm_xbf4_rows(gu.data, begin, end - begin, H, M, R, gxe.data(), gxo.data(),
                       gs.data(), gsx.data(), w.guv.data() + (size_t)begin);
      });
    } else if (gu.scheme == 4) {
      m.pool.run([&](int id) {
        int begin = id == 0 ? 0 : half, end = id == 0 ? half : R;
        gemm_xbf2_rows(gu.data, begin, end - begin, H, M, R, gx.data(),
                       gs.data(), gsx.data(), w.guv.data() + (size_t)begin);
      });
    } else {
      throw std::runtime_error("prefill MoE unsupported gate_up scheme");
    }
    for (int j = 0; j < M; j++) {
      float* guv = w.guv.data() + (size_t)j * R;
      float* hd = w.hd.data() + (size_t)j * MI;
      for (int i = 0; i < MI; i++) hd[i] = siluf(guv[i]) * guv[MI + i];
      w.hsx[j] = quant_xq(hd, MI, w.hq.data() + (size_t)j * MI,
                          w.hsums.data() + (size_t)j * MI / XBF_SB);
      xq_split4(w.hq.data() + (size_t)j * MI, MI,
                w.hxe.data() + (size_t)j * MI / 2, w.hxo.data() + (size_t)j * MI / 2);
      xq_split2(w.hq.data() + (size_t)j * MI, MI, w.hq4.data() + (size_t)j * MI);
    }
    const BaaTensor& down = *p.down[e];
    R = (int)down.shape[0]; half = (R / 2 + 15) & ~15;
    if (down.scheme == 3) {
      m.pool.run([&](int id) {
        int begin = id == 0 ? 0 : half, end = id == 0 ? half : R;
        gemm_xbf4_rows(down.data, begin, end - begin, MI, M, R, w.hxe.data(), w.hxo.data(),
                       w.hsums.data(), w.hsx.data(), w.dov.data() + (size_t)begin);
      });
    } else if (down.scheme == 4) {
      m.pool.run([&](int id) {
        int begin = id == 0 ? 0 : half, end = id == 0 ? half : R;
        gemm_xbf2_rows(down.data, begin, end - begin, MI, M, R, w.hq4.data(),
                       w.hsums.data(), w.hsx.data(), w.dov.data() + (size_t)begin);
      });
    } else {
      throw std::runtime_error("prefill MoE unsupported down scheme");
    }
    for (int j = 0; j < M; j++) {
      int tok = tokens[j];
      float weight = w.weights[(size_t)tok * TOPK + w.expert_slots[e][j]];
      const float* dov = w.dov.data() + (size_t)j * R;
      float* acc = w.acc.data() + (size_t)tok * H;
      for (int i = 0; i < H; i++) acc[i] += weight * dov[i];
    }
  }
  for (int tok = 0; tok < T; tok++) {
    float gate = sigmoidf(w.sgate[tok]);
    for (int i = 0; i < H; i++) {
      size_t at = (size_t)tok * H + i;
      xout[at] = xin[at] + w.acc[at] + gate * w.sh[at];
    }
  }
}
