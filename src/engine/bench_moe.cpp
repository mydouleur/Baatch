#include "moe_scheduler.h"
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

int main(int argc, char** argv) { try {
  std::string baa = "D:/xdna_llm/models/ornith35_xbf24_dense.baa";
  std::string npu_dir = "models/npu", mode = "current", top8 = "model";
  int layer = 0, inputs = 8, rounds = 4;
  for (int i = 1; i < argc; i++) {
    std::string s = argv[i];
    if (s == "--baa") baa = argv[++i];
    else if (s == "--npu-dir") npu_dir = argv[++i];
    else if (s == "--mode") mode = argv[++i];
    else if (s == "--layer") layer = atoi(argv[++i]);
    else if (s == "--inputs") inputs = atoi(argv[++i]);
    else if (s == "--rounds") rounds = atoi(argv[++i]);
    else if (s == "--top8") top8 = argv[++i];
    else throw std::runtime_error("unknown argument: " + s);
  }
  if (layer < 0 || layer >= Cfg::NLAYERS || inputs <= 0 || rounds <= 0 ||
      (top8 != "model" && top8 != "fixed" && top8 != "random") ||
      (mode != "direct" && top8 != "model") ||
      (mode != "current" && mode != "planned" && mode != "batch" && mode != "parallel" &&
       mode != "direct" && mode != "scheduled"))
    throw std::runtime_error("invalid layer, inputs, rounds, mode or top8");

  Model m;
  m.open(baa);
  NpuBackend npu;
  npu.init(npu_dir, mode == "batch" || mode == "parallel");
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
    else if (mode == "scheduled") scheduler->run(*plan, in, out, &direct_profile);
    else if (plan) moe_planned(m, *plan, workspace, in, out);
    else moe(m, layer, in, out);
  };
  std::mt19937 rng(20261024);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  std::vector<float> x((size_t)inputs * Cfg::H), reference((size_t)inputs * Cfg::H), y(Cfg::H);
  for (float& v : x) v = dist(rng);

  npu.reset_stats();
  for (int i = 0; i < inputs; i++) {
    auto t0 = std::chrono::steady_clock::now();
    run_moe(x.data() + (size_t)i * Cfg::H, y.data());
    double ms = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e3;
    memcpy(reference.data() + (size_t)i * Cfg::H, y.data(), Cfg::H * 4);
    printf("COLD layer=%d input=%d ms=%.3f hash=%016llx\n", layer, i, ms,
           (unsigned long long)hash_output(y));
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
  if (mode == "direct" || mode == "scheduled") {
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
