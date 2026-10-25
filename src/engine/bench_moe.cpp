#include "moe_prefill.h"
#include <chrono>
#include <random>

static uint64_t hash_output(const std::vector<float>& y) {
  uint64_t h = 1469598103934665603ull;
  for (float v : y) {
    uint32_t u;
    memcpy(&u, &v, 4);
    h = (h ^ u) * 1099511628211ull;
  }
  return h;
}

static void direct_moe_npu(Model& m, const MoeLayerPlan& p, DirectMoeWorkspace& w,
                           const float* xin, float* xout, NpuBackend& npu,
                           DirectProfile* profile, const int* top_override) {
  using namespace Cfg;
  direct_prepare(m, p, w, xin, profile, top_override);
  direct_shared(m, p, w, profile);
  std::vector<int16_t> q(H), hq(MI);
  float sx = quant_xq16(w.xn.data(), H, q.data());
  std::fill(w.acc.begin(), w.acc.end(), 0.f);
  for (int t = 0; t < TOPK; t++) {
    auto& work = w.experts[t];
    if (work.gate_up->scheme != 3 || work.down->scheme != 3)
      throw std::runtime_error("NPU routed comparison requires XBF4 experts");
    auto start = std::chrono::steady_clock::now();
    if (!npu.gemv_quantized(*work.gate_up, H, q.data(), sx, work.guv.data()))
      throw std::runtime_error("NPU routed gate_up failed");
    if (profile) {
      direct_mark(profile, profile->gemv, start);
      direct_mark(profile, profile->gate_up[t], start);
    }
    auto activation = std::chrono::steady_clock::now();
    for (int i = 0; i < MI; i++) work.hd[i] = siluf(work.guv[i]) * work.guv[MI + i];
    if (profile) direct_mark(profile, profile->activation, activation);
    auto quant = std::chrono::steady_clock::now();
    float hsx = quant_xq16(work.hd.data(), MI, hq.data());
    if (profile) direct_mark(profile, profile->expert_quant, quant);
    start = std::chrono::steady_clock::now();
    if (!npu.gemv_quantized(*work.down, MI, hq.data(), hsx, work.out.data()))
      throw std::runtime_error("NPU routed down failed");
    if (profile) {
      direct_mark(profile, profile->gemv, start);
      direct_mark(profile, profile->down[t], start);
    }
    for (int i = 0; i < H; i++) w.acc[i] += work.route_weight * work.out[i];
  }
  direct_merge(w, xin, xout, profile);
}

int main(int argc, char** argv) { try {
  std::string baa = "D:/xdna_llm/models/ornith35_xbf24_dense.baa";
  std::string npu_dir = "models/npu", mode = "current", top8 = "model";
  int layer = 0, inputs = 8, rounds = 4, prefill_tokens = 0;
  bool cpu_ref = false;
  for (int i = 1; i < argc; i++) {
    std::string s = argv[i];
    if (s == "--baa") baa = argv[++i];
    else if (s == "--npu-dir") npu_dir = argv[++i];
    else if (s == "--mode") mode = argv[++i];
    else if (s == "--layer") layer = atoi(argv[++i]);
    else if (s == "--inputs") inputs = atoi(argv[++i]);
    else if (s == "--rounds") rounds = atoi(argv[++i]);
    else if (s == "--top8") top8 = argv[++i];
    else if (s == "--prefill-tokens") prefill_tokens = atoi(argv[++i]);
    else if (s == "--cpu-ref") cpu_ref = true;
    else throw std::runtime_error("unknown argument: " + s);
  }
  if (layer < 0 || layer >= Cfg::NLAYERS || inputs <= 0 || rounds <= 0 ||
      prefill_tokens < 0 || prefill_tokens > inputs || (prefill_tokens && mode != "direct") ||
      (top8 != "model" && top8 != "fixed" && top8 != "random") ||
      (mode != "direct" && mode != "npu" && top8 != "model") ||
      (cpu_ref && (mode != "npu" || top8 != "model")) ||
      (mode != "current" && mode != "planned" && mode != "batch" && mode != "parallel" &&
       mode != "direct" && mode != "scheduled" && mode != "npu"))
    throw std::runtime_error("invalid layer, inputs, rounds, mode or top8");

  Model m;
  m.open(baa);
  NpuBackend npu;
  npu.init(npu_dir, mode == "batch" || mode == "parallel" || mode == "npu");
  if (mode != "direct") m.npu = &npu;
  std::unique_ptr<MoeLayerPlan> plan;
  std::unique_ptr<MoeScheduler> scheduler;
  std::unique_ptr<NpuMoeBatch> batch;
  MoeWorkspace workspace;
  DirectMoeWorkspace direct_workspace;
  DirectProfile direct_profile;
  if (mode != "current") plan = std::make_unique<MoeLayerPlan>(m, layer);
  if (mode == "scheduled") scheduler = std::make_unique<MoeScheduler>(m, plan.get());
  if (mode == "batch" || mode == "parallel")
    batch = std::make_unique<NpuMoeBatch>(npu, 2048, 2048, 4, mode == "parallel");
  std::mt19937 route_rng(20261025);
  auto run_moe = [&](const float* in, float* out) {
    std::array<int, Cfg::TOPK> forced{};
    const int* top_override = nullptr;
    if (top8 != "model") {
      for (int t = 0; t < Cfg::TOPK; t++) forced[t] = t;
      if (top8 == "random") {
        std::array<int, Cfg::NE> experts;
        for (int e = 0; e < Cfg::NE; e++) experts[e] = e;
        for (int t = 0; t < Cfg::TOPK; t++) {
          std::uniform_int_distribution<int> pick(t, Cfg::NE - 1);
          int j = pick(route_rng);
          std::swap(experts[t], experts[j]);
          forced[t] = experts[t];
        }
      }
      top_override = forced.data();
    }
    if (batch) {
      if (!moe_planned(m, *plan, workspace, in, out, batch.get()))
        throw std::runtime_error("MoE batch execution failed");
    } else if (mode == "direct") direct_moe(m, *plan, direct_workspace, in, out, &direct_profile, top_override);
    else if (mode == "npu") direct_moe_npu(m, *plan, direct_workspace, in, out, npu, &direct_profile, top_override);
    else if (mode == "scheduled") scheduler->run(*plan, in, out, &direct_profile);
    else if (plan) moe_planned(m, *plan, workspace, in, out);
    else moe(m, layer, in, out);
  };
  std::mt19937 rng(20261024);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  std::vector<float> x((size_t)inputs * Cfg::H), reference((size_t)inputs * Cfg::H), y(Cfg::H);
  for (float& v : x) v = dist(rng);
  if (prefill_tokens) {
    int T = prefill_tokens;
    std::vector<int> forced((size_t)T * Cfg::TOPK);
    for (int tok = 0; tok < T; tok++) {
      for (int t = 0; t < Cfg::TOPK; t++) forced[tok * Cfg::TOPK + t] = t;
      if (top8 == "random") {
        std::array<int, Cfg::NE> experts;
        for (int e = 0; e < Cfg::NE; e++) experts[e] = e;
        for (int t = 0; t < Cfg::TOPK; t++) {
          std::uniform_int_distribution<int> pick(t, Cfg::NE - 1);
          int j = pick(route_rng);
          std::swap(experts[t], experts[j]);
          forced[tok * Cfg::TOPK + t] = experts[t];
        }
      }
    }
    const int* top_override = top8 == "model" ? nullptr : forced.data();
    std::vector<float> out((size_t)T * Cfg::H), legacy((size_t)T * Cfg::H);
    auto direct_start = std::chrono::steady_clock::now();
    for (int tok = 0; tok < T; tok++)
      direct_moe(m, *plan, direct_workspace, x.data() + (size_t)tok * Cfg::H,
                 reference.data() + (size_t)tok * Cfg::H, nullptr, top_override ? top_override + tok * Cfg::TOPK : nullptr);
    double direct_ms = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - direct_start).count() * 1e3;
    MoePrefillWorkspace prefill_workspace(T);
    double total = 0, best = 1e30, legacy_total = 0;
    for (int r = 0; r < rounds; r++) {
      auto start = std::chrono::steady_clock::now();
      moe_prefill_legacy(m, *plan, prefill_workspace, x.data(), legacy.data(), top_override);
      legacy_total += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() * 1e3;
      start = std::chrono::steady_clock::now();
      moe_prefill8(m, *plan, prefill_workspace, x.data(), out.data(), top_override);
      double ms = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() * 1e3;
      total += ms; best = std::min(best, ms);
    }
    double num = 0, den = 0, max_rel = 0;
    for (size_t i = 0; i < out.size(); i++) {
      double d = out[i] - reference[i];
      num += d * d; den += (double)reference[i] * reference[i];
      max_rel = std::max(max_rel, fabs(d) / (fabs(reference[i]) + 1e-4));
    }
    uint64_t direct_hash = hash_output(reference), batch_hash = hash_output(out), legacy_hash = hash_output(legacy);
    printf("PREFILL_MOE layer=%d tokens=%d top8=%s rounds=%d direct_ms=%.3f legacy_mean_ms=%.3f batch_mean_ms=%.3f batch_min_ms=%.3f rel_l2=%.6g max_rel=%.6g\n",
           layer, T, top8.c_str(), rounds, direct_ms, legacy_total / rounds, total / rounds, best,
           sqrt(num / (den + 1e-12)), max_rel);
    printf("PREFILL_HASH direct=%016llx batch=%016llx legacy=%016llx %s\n",
           (unsigned long long)direct_hash, (unsigned long long)batch_hash, (unsigned long long)legacy_hash,
           direct_hash == batch_hash ? "PASS" : "FAIL");
    return max_rel < 0.02 && direct_hash == batch_hash ? 0 : 3;
  }

  if (const char* gate = getenv("BENCH_GATE")) {
    while (GetFileAttributesA(gate) == INVALID_FILE_ATTRIBUTES) Sleep(1);
  }
  std::vector<float> cpu_reference;
  if (cpu_ref) {
    cpu_reference.resize(x.size());
    DirectMoeWorkspace ref_workspace;
    for (int i = 0; i < inputs; i++)
      direct_moe(m, *plan, ref_workspace, x.data() + (size_t)i * Cfg::H,
                 cpu_reference.data() + (size_t)i * Cfg::H);
  }
  npu.reset_stats();
  for (int i = 0; i < inputs; i++) {
    auto t0 = std::chrono::steady_clock::now();
    run_moe(x.data() + (size_t)i * Cfg::H, y.data());
    double ms = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e3;
    memcpy(reference.data() + (size_t)i * Cfg::H, y.data(), Cfg::H * 4);
    printf("COLD layer=%d input=%d ms=%.3f hash=%016llx\n", layer, i, ms,
           (unsigned long long)hash_output(y));
    if (cpu_ref) {
      double num = 0, den = 0, max_rel = 0;
      for (int j = 0; j < Cfg::H; j++) {
        float ref = cpu_reference[(size_t)i * Cfg::H + j];
        double d = y[j] - ref;
        num += d * d; den += (double)ref * ref;
        max_rel = std::max(max_rel, fabs(d) / (fabs(ref) + 1e-4));
      }
      double rel_l2 = sqrt(num / (den + 1e-12));
      printf("CPU_REF layer=%d input=%d rel_l2=%.6g max_rel=%.6g %s\n",
             layer, i, rel_l2, max_rel, rel_l2 < 0.02 ? "PASS" : "FAIL");
      if (rel_l2 >= 0.02) return 3;
    }
  }
  printf("COLD_TOTAL layer=%d\n", layer);
  npu.print_stats();

  npu.reset_stats();
  direct_profile = DirectProfile{};
  double total = 0, best = 1e30, worst = 0;
  for (int r = 0; r < rounds; r++) {
    for (int i = 0; i < inputs; i++) {
      auto t0 = std::chrono::steady_clock::now();
      run_moe(x.data() + (size_t)i * Cfg::H, y.data());
      double ms = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e3;
      total += ms; best = std::min(best, ms); worst = std::max(worst, ms);
      if (top8 != "random" &&
          memcmp(reference.data() + (size_t)i * Cfg::H, y.data(), Cfg::H * 4) != 0) {
        fprintf(stderr, "nondeterministic output: layer=%d input=%d round=%d hash=%016llx ref=%016llx\n",
                layer, i, r, (unsigned long long)hash_output(y),
                (unsigned long long)hash_output(std::vector<float>(reference.begin() + (size_t)i * Cfg::H,
                                                                   reference.begin() + (size_t)(i + 1) * Cfg::H)));
        return 3;
      }
    }
  }
  int count = inputs * rounds;
  printf("WARM layer=%d calls=%d mean_ms=%.3f min_ms=%.3f max_ms=%.3f total_ms=%.3f\n",
         layer, count, total / count, best, worst, total);
  npu.print_stats();
  if (mode == "direct" || mode == "scheduled" || mode == "npu") {
    printf("[direct] router_ms=%.3f quant_ms=%.3f shared_ms=%.3f merge_ms=%.3f\n",
           direct_profile.router / count, direct_profile.quant / count,
           direct_profile.shared / count, direct_profile.merge / count);
    printf("[direct] calls=%d gemv_ms=%.3f pool_sync_ms=%.3f expert_quant_ms=%.3f activation_ms=%.3f\n",
           count, direct_profile.gemv / count, direct_profile.pool_sync / count,
           direct_profile.expert_quant / count, direct_profile.activation / count);
    for (int t = 0; t < Cfg::TOPK; t++)
      printf("[direct] slot=%d gate_up_ms=%.3f down_ms=%.3f\n",
             t, direct_profile.gate_up[t] / count, direct_profile.down[t] / count);
  }
  return 0;
} catch (const std::exception& e) { fprintf(stderr, "EXC: %s\n", e.what()); return 2; } }
