#pragma once
#include <chrono>

struct EngineProfile {
  bool enabled = false;
  double gdn_proj[Cfg::NLAYERS]{};
  double gdn_state[Cfg::NLAYERS]{};
  double gdn_out[Cfg::NLAYERS]{};
  double attn_proj[Cfg::NLAYERS]{};
  double attn_state[Cfg::NLAYERS]{};
  double attn_out[Cfg::NLAYERS]{};
  double moe[Cfg::NLAYERS]{};

  void reset() {
    bool keep = enabled;
    *this = EngineProfile{};
    enabled = keep;
  }
};

static EngineProfile g_prof;

static inline std::chrono::steady_clock::time_point prof_now() {
  return g_prof.enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
}

static inline void prof_add(double& dst, std::chrono::steady_clock::time_point start) {
  if (g_prof.enabled)
    dst += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() * 1e3;
}
