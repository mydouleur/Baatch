#pragma once
#include "moe_batch.h"
#include <condition_variable>
#include <mutex>

struct DirectExpertWork {
  const BaaTensor* gate_up = nullptr;
  const BaaTensor* down = nullptr;
  float route_weight = 0.f;
  std::vector<float> guv, hd, out;
  std::vector<int8_t> q, xe, xo, q4;
  std::vector<float> sum;
  float sx = 1.f;

  DirectExpertWork() {
    guv.resize(2 * Cfg::MI); hd.resize(Cfg::MI); out.resize(Cfg::H);
    q.resize(Cfg::MI); xe.resize(Cfg::MI / 2); xo.resize(Cfg::MI / 2); q4.resize(Cfg::MI);
    sum.resize(Cfg::MI / XBF_SB);
  }
};

struct DirectMoeWorkspace {
  std::vector<float> xn, logits, acc;
  std::vector<std::pair<float, int>> idx;
  std::vector<float> sg, su, sh, sgate;
  std::vector<int8_t> q, xe, xo, q4;
  std::vector<float> sum;
  float sx = 1.f;
  std::array<DirectExpertWork, Cfg::TOPK> experts;

  DirectMoeWorkspace() {
    xn.resize(Cfg::H); logits.resize(Cfg::NE); acc.resize(Cfg::H); idx.resize(Cfg::NE);
    sg.resize(Cfg::MI); su.resize(Cfg::MI); sh.resize(Cfg::H); sgate.resize(1);
    q.resize(Cfg::H); xe.resize(Cfg::H / 2); xo.resize(Cfg::H / 2); q4.resize(Cfg::H);
    sum.resize(Cfg::H / XBF_SB);
  }
};

struct DirectProfile {
  double router = 0, quant = 0, shared = 0, merge = 0;
  double gemv = 0, pool_sync = 0, expert_quant = 0, activation = 0;
  double gate_up[Cfg::TOPK]{};
  double down[Cfg::TOPK]{};
};

static void direct_mark(DirectProfile* profile, double& dst,
                        std::chrono::steady_clock::time_point start) {
  if (profile) dst += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() * 1e3;
}

static void direct_route(Model& m, const MoeLayerPlan& p, DirectMoeWorkspace& w,
                         DirectProfile* profile, const int* top_override, bool single_thread) {
  using namespace Cfg;
  auto t0 = std::chrono::steady_clock::now();
  if (p.router->scheme == 5) {
    if (single_thread) gemv_f32_rows(p.router->f32(), 0, NE, H, w.xn.data(), w.logits.data());
    else gemv_f32(m.pool, p.router->f32(), NE, H, w.xn.data(), w.logits.data());
  } else {
    m.mv(*p.router, w.xn.data(), w.logits.data());
  }
  float mx = *std::max_element(w.logits.begin(), w.logits.end());
  float sum = 0;
  for (int i = 0; i < NE; i++) { w.logits[i] = expf(w.logits[i] - mx); sum += w.logits[i]; }
  for (int i = 0; i < NE; i++) w.logits[i] /= sum;
  for (int i = 0; i < NE; i++) w.idx[i] = {w.logits[i], i};
  std::partial_sort(w.idx.begin(), w.idx.begin() + TOPK, w.idx.end(), std::greater<>());
  float wsum = 0;
  for (int t = 0; t < TOPK; t++) {
    int e = top_override ? top_override[t] : w.idx[t].second;
    w.experts[t].gate_up = p.gate_up[e];
    w.experts[t].down = p.down[e];
    w.experts[t].route_weight = top_override ? 1.f / TOPK : w.idx[t].first;
    wsum += w.experts[t].route_weight;
  }
  for (int t = 0; t < TOPK; t++) w.experts[t].route_weight /= std::max(wsum, 1e-9f);
  if (profile) direct_mark(profile, profile->router, t0);
}

static void direct_quant(DirectMoeWorkspace& w, DirectProfile* profile) {
  using namespace Cfg;
  auto t0 = std::chrono::steady_clock::now();
  w.sx = quant_xq(w.xn.data(), H, w.q.data(), w.sum.data());
  xq_split4(w.q.data(), H, w.xe.data(), w.xo.data());
  xq_split2(w.q.data(), H, w.q4.data());
  if (profile) direct_mark(profile, profile->quant, t0);
}

static void direct_prepare(Model& m, const MoeLayerPlan& p, DirectMoeWorkspace& w,
                           const float* xin, DirectProfile* profile,
                           const int* top_override = nullptr) {
  using namespace Cfg;
  auto t0 = std::chrono::steady_clock::now();
  m.nrm(*p.norm, xin, H, w.xn.data(), true);
  if (profile) direct_mark(profile, profile->router, t0);
  direct_route(m, p, w, profile, top_override, false);
  direct_quant(w, profile);
}

static void direct_shared(Model& m, const MoeLayerPlan& p, DirectMoeWorkspace& w,
                          DirectProfile* profile) {
  using namespace Cfg;
  auto t0 = std::chrono::steady_clock::now();
  cpu_mv_pretiled_fast(*p.shared_gate, w.xn.data(), w.sg.data());
  cpu_mv_pretiled_fast(*p.shared_up, w.xn.data(), w.su.data());
  for (int i = 0; i < MI; i++) w.sg[i] = siluf(w.sg[i]) * w.su[i];
  cpu_mv_pretiled_fast(*p.shared_down, w.sg.data(), w.sh.data());
  if (p.shared_sigmoid->scheme == 5)
    gemv_f32_rows(p.shared_sigmoid->f32(), 0, 1, H, w.xn.data(), w.sgate.data());
  else
    m.mv(*p.shared_sigmoid, w.xn.data(), w.sgate.data());
  if (profile) direct_mark(profile, profile->shared, t0);
}

static void direct_routed(Model& m, DirectMoeWorkspace& w, DirectProfile* profile) {
  using namespace Cfg;
  double gemv_ms[2]{}, quant_ms[2]{}, activation_ms[2]{}, work_ms[2]{};
  auto outer = std::chrono::steady_clock::now();
  m.pool.run([&](int id) {
    auto work_start = std::chrono::steady_clock::now();
    auto run_xbf4 = [&](const BaaTensor& t, int n_in, const int8_t* xe, const int8_t* xo,
                        const float* sum, float sx, float* y) {
      auto start = std::chrono::steady_clock::now();
      gemv_xbf4_rows(t.data, 0, (int)t.shape[0], n_in, xe, xo, sum, sx, y);
      gemv_ms[id] += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() * 1e3;
    };
    auto run_xbf2 = [&](const BaaTensor& t, int n_in, const int8_t* const* q4,
                        const float* sum, float sx, float* y) {
      auto start = std::chrono::steady_clock::now();
      gemv_xbf2_rows(t.data, 0, (int)t.shape[0], n_in, q4, sum, sx, y);
      gemv_ms[id] += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() * 1e3;
    };
    for (int t = id * TOPK / 2; t < (id + 1) * TOPK / 2; t++) {
      auto& work = w.experts[t];
      const BaaTensor& gu = *work.gate_up;
      auto t0 = std::chrono::steady_clock::now();
      if (gu.scheme == 3) {
        run_xbf4(gu, H, w.xe.data(), w.xo.data(), w.sum.data(), w.sx, work.guv.data());
      } else if (gu.scheme == 4) {
        const int8_t* q4[] = {w.q4.data(), w.q4.data() + H / 4, w.q4.data() + H / 2, w.q4.data() + 3 * H / 4};
        run_xbf2(gu, H, q4, w.sum.data(), w.sx, work.guv.data());
      } else {
        throw std::runtime_error("direct MoE unsupported gate_up scheme");
      }
      if (profile) direct_mark(profile, profile->gate_up[t], t0);
      auto ta = std::chrono::steady_clock::now();
      for (int i = 0; i < MI; i++) work.hd[i] = siluf(work.guv[i]) * work.guv[MI + i];
      if (profile) activation_ms[id] += std::chrono::duration<double>(
          std::chrono::steady_clock::now() - ta).count() * 1e3;

      const BaaTensor& down = *work.down;
      t0 = std::chrono::steady_clock::now();
      work.sx = quant_xq(work.hd.data(), MI, work.q.data(), work.sum.data());
      if (down.scheme == 3) xq_split4(work.q.data(), MI, work.xe.data(), work.xo.data());
      else if (down.scheme == 4) xq_split2(work.q.data(), MI, work.q4.data());
      else throw std::runtime_error("direct MoE unsupported down scheme");
      if (profile) quant_ms[id] += std::chrono::duration<double>(
          std::chrono::steady_clock::now() - t0).count() * 1e3;
      if (down.scheme == 3) {
        run_xbf4(down, MI, work.xe.data(), work.xo.data(), work.sum.data(), work.sx, work.out.data());
      } else {
        const int8_t* q4[] = {work.q4.data(), work.q4.data() + MI / 4,
                              work.q4.data() + MI / 2, work.q4.data() + 3 * MI / 4};
        run_xbf2(down, MI, q4, work.sum.data(), work.sx, work.out.data());
      }
      if (profile) direct_mark(profile, profile->down[t], t0);
    }
    work_ms[id] = std::chrono::duration<double>(std::chrono::steady_clock::now() - work_start).count() * 1e3;
  });
  if (profile) {
    direct_mark(profile, profile->pool_sync, outer);
    profile->pool_sync -= std::max(work_ms[0], work_ms[1]);
    profile->gemv += std::max(gemv_ms[0], gemv_ms[1]);
    profile->expert_quant += quant_ms[0] + quant_ms[1];
    profile->activation += activation_ms[0] + activation_ms[1];
  }
  std::fill(w.acc.begin(), w.acc.end(), 0.f);
  for (int t = 0; t < TOPK; t++)
    for (int i = 0; i < H; i++) w.acc[i] += w.experts[t].route_weight * w.experts[t].out[i];
}

static void direct_merge(DirectMoeWorkspace& w, const float* xin, float* xout, DirectProfile* profile) {
  using namespace Cfg;
  auto t0 = std::chrono::steady_clock::now();
  float g0 = sigmoidf(w.sgate[0]);
  for (int i = 0; i < H; i++) xout[i] = xin[i] + w.acc[i] + g0 * w.sh[i];
  if (profile) direct_mark(profile, profile->merge, t0);
}

static void direct_moe(Model& m, const MoeLayerPlan& p, DirectMoeWorkspace& w,
                       const float* xin, float* xout, DirectProfile* profile = nullptr,
                       const int* top_override = nullptr) {
  direct_prepare(m, p, w, xin, profile, top_override);
  direct_shared(m, p, w, profile);
  direct_routed(m, w, profile);
  direct_merge(w, xin, xout, profile);
}

struct MoeScheduler {
  Model& m;
  const MoeLayerPlan* p;
  DirectMoeWorkspace w;
  std::thread worker;
  std::mutex mu;
  std::condition_variable cv;
  bool task = false, done = false, stop = false;
  DirectProfile* profile = nullptr;
  std::exception_ptr error;

  MoeScheduler(Model& model, const MoeLayerPlan* plan = nullptr) : m(model), p(plan) {
    SetThreadAffinityMask(GetCurrentThread(), env_flag_default("MOE_HOST_SIBLING") ? 1ull << 1 : 1ull << 0);
    worker = std::thread([this] {
      SetThreadAffinityMask(GetCurrentThread(), 1ull << 0);
      std::unique_lock<std::mutex> lock(mu);
      while (!stop) {
        cv.wait(lock, [&] { return task || stop; });
        if (stop) break;
        try {
          direct_shared(m, *p, w, profile);
        } catch (...) {
          error = std::current_exception();
        }
        task = false;
        done = true;
        cv.notify_all();
      }
    });
  }

  ~MoeScheduler() {
    {
      std::lock_guard<std::mutex> lock(mu);
      stop = true;
      cv.notify_all();
    }
    worker.join();
  }

  void run(const MoeLayerPlan& plan, const float* xin, float* xout, DirectProfile* next_profile = nullptr) {
    p = &plan;
    auto t0 = std::chrono::steady_clock::now();
    m.nrm(*p->norm, xin, Cfg::H, w.xn.data(), true);
    if (next_profile) direct_mark(next_profile, next_profile->router, t0);
    {
      std::lock_guard<std::mutex> lock(mu);
      profile = next_profile;
      error = nullptr;
      done = false;
      task = true;
      cv.notify_all();
    }
    direct_route(m, *p, w, next_profile, nullptr, true);
    direct_quant(w, next_profile);
    direct_routed(m, w, next_profile);
    {
      std::unique_lock<std::mutex> lock(mu);
      cv.wait(lock, [&] { return done; });
      if (error) std::rethrow_exception(error);
    }
    direct_merge(w, xin, xout, next_profile);
  }
};

