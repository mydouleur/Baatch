// NPU<->CPU rendezvous sync-latency probe (task 0c, Baatch turbo-mixing go/no-go).
//
// Measures the end-to-end latency of one rendezvous:
//   CPU thread finishes simulated work -> xrt run submit -> NPU executes
//   -> completion -> CPU thread wakes from blocking wait.
// Also measures the reverse direction component (submit() call cost) and
// compares interrupt-driven blocking wait vs busy-poll wake-up.
//
// Modes:
//   0 bare        : run create + set_arg + start + wait2() back to back
//   1 rendezvous  : burn cpu_us of simulated GEMV work, then create+start+wait2
//   2 poll        : like bare but wake via busy-polling run.state()
// Per-iteration CSV columns: create_ns, submit_ns (start() call), e2e_ns
// (start->wake), burn_ns (actual CPU burn).
//
// Usage: host_sync_lat.exe <xclbin> <elf> <iters> <buf_bytes> <mode> [cpu_us] [csv]
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/experimental/xrt_xclbin.h"
#include "xrt/experimental/xrt_elf.h"
#include "xrt/experimental/xrt_module.h"
#include "xrt/experimental/xrt_ext.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

using clk = std::chrono::steady_clock;
static inline double ns_since(clk::time_point t0) {
  return std::chrono::duration<double, std::nano>(clk::now() - t0).count();
}

// Simulated GEMV-ish CPU work: streaming dot products over a 4 MB fp32 matrix.
static std::vector<float> g_mat(1 << 20);
static volatile float g_sink;
static double burn_ns(double target_us) {
  auto t0 = clk::now();
  double target_ns = target_us * 1e3;
  float acc = 0.f;
  const size_t n = g_mat.size();
  do {
    for (size_t i = 0; i + 4 <= n; i += 4)
      acc += g_mat[i] * 1.0000001f + g_mat[i + 1] * 0.9999999f +
             g_mat[i + 2] * 1.0000003f + g_mat[i + 3] * 0.9999997f;
  } while (std::chrono::duration<double, std::nano>(clk::now() - t0).count() < target_ns);
  g_sink = acc;
  return ns_since(t0);
}

static void stats(const char* name, std::vector<double>& v) {
  if (v.empty()) return;
  std::sort(v.begin(), v.end());
  auto pct = [&](double p) { return v[std::min<size_t>((size_t)(p * v.size()), v.size() - 1)] / 1e3; };
  double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size() / 1e3;
  std::printf("%-10s n=%zu  min=%8.2f  p50=%8.2f  p90=%8.2f  p99=%8.2f  max=%9.2f  mean=%8.2f us\n",
              name, v.size(), v.front() / 1e3, pct(0.50), pct(0.90), pct(0.99), v.back() / 1e3, mean);
}

int main(int argc, char** argv) {
  if (argc < 6) {
    std::printf("usage: %s <xclbin> <elf> <iters> <buf_bytes> <mode 0|1|2> [cpu_us] [csv]\n", argv[0]);
    return 1;
  }
  std::string xclbin_path = argv[1], elf_path = argv[2];
  int iters = std::atoi(argv[3]);
  size_t buf_size = std::atoll(argv[4]);
  int mode = std::atoi(argv[5]);
  double cpu_us = argc > 6 ? std::atof(argv[6]) : 0.0;
  const char* csv_path = argc > 7 ? argv[7] : nullptr;

  // timer sanity
  {
    auto t0 = clk::now();
    while (clk::now() == t0) {}
    std::printf("clock tick ~%.0f ns\n", ns_since(t0));
  }
  for (auto& x : g_mat) x = 1.0f;

  try {
    xrt::device dev{0};
    std::printf("device ok\n"); std::fflush(stdout);
    xrt::xclbin xclbin{xclbin_path};
    auto kernels = xclbin.get_kernels();
    std::printf("xclbin ok, %zu kernels\n", kernels.size()); std::fflush(stdout);
    std::string kname;
    for (auto& k : kernels) if (k.get_name() == "DPU") kname = "DPU";
    if (kname.empty()) kname = kernels.empty() ? "DPU" : kernels[0].get_name();
    auto uuid = dev.register_xclbin(xclbin);
    std::printf("registered\n"); std::fflush(stdout);
    xrt::hw_context hwctx{dev, uuid};
    std::printf("hwctx ok\n"); std::fflush(stdout);
    xrt::elf elf{elf_path};
    std::printf("elf ok\n"); std::fflush(stdout);
    xrt::module mod{elf};
    std::printf("module ok\n"); std::fflush(stdout);
    xrt::ext::kernel kernel{hwctx, mod, kname};
    std::printf("kernel='%s' mode=%d iters=%d buf=%zu bytes cpu_us=%.0f\n",
                kname.c_str(), mode, iters, buf_size, cpu_us);

    xrt::ext::bo bo_ifm{dev, buf_size};
    xrt::ext::bo bo_ofm{dev, buf_size};
    {
      auto* p = bo_ifm.map<int*>();
      for (size_t i = 0; i < buf_size / 4; i++) p[i] = (int)(i * 2654435761u);
    }

    std::vector<double> v_create, v_submit, v_e2e, v_burn;
    FILE* csv = csv_path ? std::fopen(csv_path, "w") : nullptr;
    if (csv) std::fprintf(csv, "iter,create_ns,submit_ns,e2e_ns,burn_ns\n");

    for (int i = 0; i < iters; i++) {
      double burn = 0.0;
      if (mode == 1 && cpu_us > 0.0) burn = burn_ns(cpu_us);

      auto t0 = clk::now();
      xrt::run run{kernel};
      run.set_arg(0, 3);
      run.set_arg(1, 0);
      run.set_arg(2, 0);
      run.set_arg(3, bo_ifm);
      run.set_arg(5, bo_ofm);
      run.set_arg(6, 0);
      run.set_arg(7, 0);
      auto t1 = clk::now();
      run.start();
      auto t2 = clk::now();
      if (mode == 2) {
        // busy-poll wake path
        auto deadline = t2 + std::chrono::seconds(30);
        while (run.state() != ERT_CMD_STATE_COMPLETED) {
          if (clk::now() > deadline) { std::fprintf(stderr, "poll timeout iter %d\n", i); return 3; }
        }
      } else {
        run.wait2();
      }
      auto t3 = clk::now();

      double create = std::chrono::duration<double, std::nano>(t1 - t0).count();
      double submit = std::chrono::duration<double, std::nano>(t2 - t1).count();
      double e2e = std::chrono::duration<double, std::nano>(t3 - t1).count();
      v_create.push_back(create); v_submit.push_back(submit);
      v_e2e.push_back(e2e); v_burn.push_back(burn);
      if (csv) std::fprintf(csv, "%d,%.0f,%.0f,%.0f,%.0f\n", i, create, submit, e2e, burn);
    }
    if (csv) std::fclose(csv);

    // verify only the ranges the BDs actually cover:
    // per col c: [c*span, c*span+xfer) and [c*span+span/2, +xfer)
    // span = buf_size/4 (4 cols); xfer defaults to span/2 (stock full-cover layout).
    size_t span = buf_size / 4;
    size_t xfer = span / 2;
    if (const char* e = std::getenv("SYNC_XFER")) xfer = std::atoll(e);
    auto* pi = bo_ifm.map<const int*>();
    auto* po = bo_ofm.map<const int*>();
    size_t bad = 0, checked = 0;
    for (int c = 0; c < 4; c++)
      for (size_t base : {c * span, c * span + span / 2})
        for (size_t i = 0; i < xfer / 4; i++) {
          checked++;
          if (pi[base / 4 + i] != po[base / 4 + i]) {
            bad++;
            if (bad < 5) std::printf("mismatch @%zu: %d != %d\n", base / 4 + i, pi[base / 4 + i], po[base / 4 + i]);
          }
        }
    std::printf("verify: %s (%zu/%zu words mismatched, xfer=%zuB/bd)\n",
                bad ? "FAIL" : "PASS", bad, checked, xfer);

    // drop warmup (first 5) before stats
    auto warm = [](std::vector<double>& v) {
      return std::vector<double>(v.begin() + (v.size() > 10 ? 5 : 0), v.end());
    };
    auto wc = warm(v_create), ws = warm(v_submit), we = warm(v_e2e), wb = warm(v_burn);
    stats("create", wc);
    stats("submit", ws);
    stats("e2e", we);
    if (mode == 1) stats("burn", wb);
    return 0;
  }
  catch (const std::exception& e) {
    std::fprintf(stderr, "ERROR: %s\n", e.what());
    return 2;
  }
}
