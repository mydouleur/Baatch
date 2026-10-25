#pragma once
#include "model.h"
#include <array>

struct MoeLayerPlan {
  int layer;
  const BaaTensor* norm;
  const BaaTensor* router;
  std::array<const BaaTensor*, Cfg::NE> gate_up;
  std::array<const BaaTensor*, Cfg::NE> down;
  const BaaTensor* shared_gate;
  const BaaTensor* shared_up;
  const BaaTensor* shared_down;
  const BaaTensor* shared_sigmoid;

  MoeLayerPlan(Model& m, int l) : layer(l) {
    norm = &m.T(LN(l, "post_attention_layernorm.weight"));
    router = &m.T(LN(l, "mlp.gate.weight"));
    char name[160];
    for (int e = 0; e < Cfg::NE; e++) {
      snprintf(name, sizeof(name), "layers.%d.mlp.experts.gate_up_proj.%d", l, e);
      gate_up[e] = &m.T(name);
      snprintf(name, sizeof(name), "layers.%d.mlp.experts.down_proj.%d", l, e);
      down[e] = &m.T(name);
    }
    shared_gate = &m.T(LN(l, "mlp.shared_expert.gate_proj.weight"));
    shared_up = &m.T(LN(l, "mlp.shared_expert.up_proj.weight"));
    shared_down = &m.T(LN(l, "mlp.shared_expert.down_proj.weight"));
    shared_sigmoid = &m.T(LN(l, "mlp.shared_expert_gate.weight"));
  }
};

struct MoeWorkspace {
  std::vector<float> xn, logits;
  std::vector<std::pair<float, int>> idx;
  std::vector<float> acc, guv, hd, dov;
  std::vector<float> batch_y, batch_hd, batch_down;
  std::vector<float> sg, su, sh, sgate;
  std::vector<int16_t> npu_q;
  std::vector<int8_t> cpu_q, cpu_q4;
  std::vector<float> sumxq;
  float npu_sx = 1.f, cpu_sx = 1.f;

  MoeWorkspace()
      : xn(Cfg::H), logits(Cfg::NE), idx(Cfg::NE), acc(Cfg::H), guv(2 * Cfg::MI),
        hd(Cfg::MI), dov(Cfg::H), batch_y(Cfg::TOPK * 2048),
        batch_hd(Cfg::TOPK * Cfg::MI), batch_down(Cfg::TOPK * 2048),
        sg(Cfg::MI), su(Cfg::MI), sh(Cfg::H), sgate(1),
        npu_q(Cfg::H), cpu_q(Cfg::H), cpu_q4(Cfg::H), sumxq(Cfg::H / XBF_SB) {}
};

struct NpuMoeBatch {
  static constexpr int E = Cfg::TOPK;
  NpuBackend& npu;
  NpuBackend::Shape* single;
  std::shared_ptr<xrt::hw_context> hw;
  xrt::kernel kernel;
  xrt::bo bi, ba, bc;
  unsigned instr_words;
  int K, N, experts;
  bool parallel;
  std::map<const void*, xrt::bo> parallel_cache;

  NpuMoeBatch(NpuBackend& backend, int k = 2048, int n = 2048, int count = Cfg::TOPK,
              bool parallel = false)
      : npu(backend), single(&npu.shape_for(k, n)), K(k), N(n), experts(count), parallel(parallel) {
    std::string base = npu.dir + "/k" + std::to_string(K) + "n" + std::to_string(N) +
                       (parallel ? "p" : "e") + std::to_string(experts);
    std::ifstream f(base + ".insts.bin", std::ios::binary);
    if (!f) throw std::runtime_error("npu MoE batch shape missing: " + base);
    f.seekg(0, std::ios::end); size_t size = f.tellg(); f.seekg(0);
    std::vector<char> inst(size); f.read(inst.data(), size);
    auto xcl = xrt::xclbin(base + ".xclbin");
    auto uuid = npu.dev.register_xclbin(xcl);
    hw = std::make_shared<xrt::hw_context>(npu.dev, uuid);
    kernel = xrt::kernel(*hw, "MLIR_AIE");
    bi = xrt::bo(npu.dev, size, XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    memcpy(bi.map(), inst.data(), size);
    npu.sync_bo(bi, XCL_BO_SYNC_BO_TO_DEVICE);
    ba = xrt::bo(npu.dev, E * K * 2, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    bc = xrt::bo(npu.dev, E * N * 2, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4 + experts));
    instr_words = size / 4;
  }

  int16_t* activations() { return static_cast<int16_t*>(ba.map()); }
  const uint16_t* outputs() { return static_cast<const uint16_t*>(bc.map()); }

  xrt::bo& parallel_bo(int e, const BaaTensor& t) {
    auto it = parallel_cache.find(t.data);
    if (it != parallel_cache.end()) { npu.bo_hits++; return it->second; }
    npu.bo_misses++;
    size_t bytes = (size_t)(K / 128) * (N / 64) * 4352;
    std::vector<uint8_t> tiled(bytes);
    auto t0 = std::chrono::steady_clock::now();
    NpuBackend::pack_expert(t, tiled.data(), N, K);
    xrt::bo b(npu.dev, bytes, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4 + e));
    for (int lt = 0; lt < N / 64; lt++)
      for (int ltr = 0; ltr < K / 128; ltr++) {
        size_t src = ((size_t)(lt % 8) * (N / 512) + lt / 8) * (K / 128) + ltr;
        size_t dst = (size_t)lt * (K / 128) + ltr;
        memcpy(static_cast<uint8_t*>(b.map()) + dst * 4352, tiled.data() + src * 4352, 4352);
      }
    npu.pack_ms += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e3;
    npu.sync_bo(b, XCL_BO_SYNC_BO_TO_DEVICE);
    auto inserted = parallel_cache.emplace(t.data, std::move(b));
    return inserted.first->second;
  }

  bool run(const std::array<const BaaTensor*, E>& weights,
           const std::array<float, E>& scales, float* y) {
    if (npu.busy) throw std::runtime_error("npu MoE batch while busy");
    std::array<xrt::bo*, E> bos;
    for (int e = 0; e < experts; e++)
      bos[e] = parallel ? &parallel_bo(e, *weights[e]) : &npu.bo_for(*single, *weights[e]);
    npu.physical_weight_bytes += (size_t)experts * (K / 128) * (N / 64) * 4352;
    npu.sync_bo(ba, XCL_BO_SYNC_BO_TO_DEVICE);
    npu.busy = true;
    for (int attempt = 0; attempt < 4; attempt++) {
      memset(bc.map(), 0, E * N * 2);
      npu.sync_bo(bc, XCL_BO_SYNC_BO_TO_DEVICE);
      xrt::run run{kernel};
      run.set_arg(0, 3);
      run.set_arg(1, bi); run.set_arg(2, instr_words);
      run.set_arg(3, ba);
      for (int e = 0; e < experts; e++) run.set_arg(4 + e, *bos[e]);
      run.set_arg(4 + experts, bc);
      auto t0 = std::chrono::steady_clock::now();
      run.start();
      try { run.wait2(); }
      catch (const std::exception& e) {
        fprintf(stderr, "[npu-batch] run timeout: %s\n", e.what());
        npu.runs++; npu.ok = false; npu.busy = false;
        npu.wait_ms += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e3;
        return false;
      }
      npu.runs++;
      npu.wait_ms += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e3;
      npu.sync_bo(bc, XCL_BO_SYNC_BO_FROM_DEVICE);
      const uint16_t* c = outputs();
      bool flake = false;
      for (int e = 0; e < experts; e++) {
        bool zero = true;
        int logical_n = weights[e]->shape[0];
        for (int i = 0; i < logical_n; i++) if (c[e * N + i] != 0) { zero = false; break; }
        flake |= zero;
      }
      if (!flake) {
        for (int e = 0; e < experts; e++)
          for (int i = 0; i < N; i++) y[e * N + i] = bf16_to_f32(c[e * N + i]) * scales[e];
        npu.busy = false;
        return true;
      }
      npu.flakes++;
    }
    fprintf(stderr, "[npu-batch] flake x4\n");
    npu.busy = false;
    return false;
  }
};

static bool moe_planned(Model& m, const MoeLayerPlan& p, MoeWorkspace& w,
                        const float* xin, float* xout, NpuMoeBatch* batch = nullptr) {
  using namespace Cfg;
  m.nrm(*p.norm, xin, H, w.xn.data(), true);
  m.mv(*p.router, w.xn.data(), w.logits.data());
  float mx = *std::max_element(w.logits.begin(), w.logits.end());
  float sum = 0;
  for (int i = 0; i < NE; i++) { w.logits[i] = expf(w.logits[i] - mx); sum += w.logits[i]; }
  for (int i = 0; i < NE; i++) w.logits[i] /= sum;
  for (int i = 0; i < NE; i++) w.idx[i] = {w.logits[i], i};
  std::partial_sort(w.idx.begin(), w.idx.begin() + TOPK, w.idx.end(), std::greater<>());
  int top[TOPK]; float tw[TOPK];
  float wsum = 0;
  for (int t = 0; t < TOPK; t++) { top[t] = w.idx[t].second; tw[t] = w.idx[t].first; wsum += tw[t]; }
  for (int t = 0; t < TOPK; t++) tw[t] /= std::max(wsum, 1e-9f);

  w.npu_sx = quant_xq16(w.xn.data(), H, w.npu_q.data());
  w.cpu_sx = quant_xq(w.xn.data(), H, w.cpu_q.data(), w.sumxq.data());
  xq_split2(w.cpu_q.data(), H, w.cpu_q4.data());
  std::fill(w.acc.begin(), w.acc.end(), 0.f);
  auto npu_xn = [&](const BaaTensor& weight, float* y, bool expert = false) {
    if (((weight.flags & 1) || expert) && m.npu && m.npu->ok) {
      if (m.npu->gemv_quantized(weight, H, w.npu_q.data(), w.npu_sx, y)) return;
      if (weight.flags & 1) cpu_mv_pretiled(weight, w.xn.data(), y);
      else m.mv(weight, w.xn.data(), y);
      return;
    }
    m.mv(weight, w.xn.data(), y);
  };
  bool use_batch = batch && m.npu && m.npu->xbf4_experts;
  for (int t = 0; t < TOPK; t++) use_batch &= p.gate_up[top[t]]->scheme == 3;
  if (use_batch) {
    std::array<const BaaTensor*, NpuMoeBatch::E> weights;
    std::array<float, NpuMoeBatch::E> scales;
    for (int chunk = 0; chunk < 2; chunk++) {
      memset(batch->activations(), 0, NpuMoeBatch::E * 2048 * 2);
      for (int e = 0; e < 4; e++) {
        int t = chunk * 4 + e;
        weights[e] = p.gate_up[top[t]];
        scales[e] = w.npu_sx;
        memcpy(batch->activations() + e * 2048, w.npu_q.data(), H * 2);
      }
      if (!batch->run(weights, scales, w.batch_y.data() + chunk * 4 * 2048)) return false;
    }
    for (int t = 0; t < TOPK; t++)
      for (int i = 0; i < MI; i++)
        w.batch_hd[t * MI + i] = siluf(w.batch_y[t * 2048 + i]) * w.batch_y[t * 2048 + MI + i];
    for (int chunk = 0; chunk < 2; chunk++) {
      memset(batch->activations(), 0, NpuMoeBatch::E * 2048 * 2);
      for (int e = 0; e < 4; e++) {
        int t = chunk * 4 + e;
        weights[e] = p.down[top[t]];
        scales[e] = quant_xq16(w.batch_hd.data() + t * MI, MI, batch->activations() + e * 2048);
      }
      if (!batch->run(weights, scales, w.batch_down.data() + chunk * 4 * 2048)) return false;
    }
    for (int t = 0; t < TOPK; t++)
      for (int i = 0; i < H; i++) w.acc[i] += tw[t] * w.batch_down[t * 2048 + i];
  } else {
    for (int t = 0; t < TOPK; t++) {
      const BaaTensor& gu = *p.gate_up[top[t]];
      if (gu.scheme == 3 && m.npu && m.npu->xbf4_experts) npu_xn(gu, w.guv.data(), true);
      else if (gu.scheme == 4) m.mv_xbf2_quantized(gu, w.cpu_q4.data(), w.sumxq.data(), w.cpu_sx, w.guv.data());
      else m.mv(gu, w.xn.data(), w.guv.data());
      for (int i = 0; i < MI; i++) {
        float g = w.guv[i], u = w.guv[MI + i];
        w.hd[i] = siluf(g) * u;
      }
      const BaaTensor& down = *p.down[top[t]];
      if (down.scheme == 3) m.mv_named(down, ".experts.", w.hd.data(), w.dov.data());
      else m.mv(down, w.hd.data(), w.dov.data());
      for (int i = 0; i < H; i++) w.acc[i] += tw[t] * w.dov[i];
    }
  }

  npu_xn(*p.shared_gate, w.sg.data());
  npu_xn(*p.shared_up, w.su.data());
  for (int i = 0; i < MI; i++) w.sg[i] = siluf(w.sg[i]) * w.su[i];
  NpuBackend::Job sdjob;
  if (m.npu_submit(*p.shared_down, w.sg.data(), w.sh.data(), sdjob)) {
    m.mv(*p.shared_sigmoid, w.xn.data(), w.sgate.data());
    m.npu_finish(*p.shared_down, w.sg.data(), w.sh.data(), sdjob, "shared_expert.down_proj");
  } else {
    m.mv(*p.shared_down, w.sg.data(), w.sh.data());
    m.mv(*p.shared_sigmoid, w.xn.data(), w.sgate.data());
  }
  float g0 = sigmoidf(w.sgate[0]);
  for (int i = 0; i < H; i++) xout[i] = xin[i] + w.acc[i] + g0 * w.sh[i];
  return true;
}
