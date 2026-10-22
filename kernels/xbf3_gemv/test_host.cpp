// test_host.cpp — Windows XRT host for xbf3_gemv (v1/v2).
// Loads CI-built xclbin + insts.bin, feeds wts.bin/act.bin, verifies against out_ref.bin.
// Host ABI (MLIR_AIE): arg0=opcode 3, arg1=instr bo, arg2=instr words,
// arg3=wts, arg4=act, arg5=out  (runtime sequence operand order [wts, act, out]).
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/experimental/xrt_xclbin.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>

static std::vector<char> read_file(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) { std::fprintf(stderr, "cannot open %s\n", p.c_str()); exit(1); }
  f.seekg(0, std::ios::end);
  size_t n = f.tellg();
  f.seekg(0);
  std::vector<char> d(n);
  f.read(d.data(), n);
  return d;
}

int main(int argc, char** argv) {
  if (argc < 4) {
    std::printf("usage: %s <xclbin> <insts.bin> <datadir> [iters] [--noverify]\n", argv[0]);
    return 1;
  }
  const char* xclbin_path = argv[1];
  const char* insts_path = argv[2];
  std::string dir = argv[3];
  int iters = argc > 4 ? atoi(argv[4]) : 20;
  bool verify = !(argc > 5 && std::string(argv[5]) == "--noverify");
  int ncols = 1;
  for (int i = 5; i + 1 < argc; i++) if (std::string(argv[i]) == "--cols") ncols = atoi(argv[i+1]);

  auto wts = read_file(dir + "\\wts.bin");
  auto act = read_file(dir + "\\act.bin");
  auto ref = verify ? read_file(dir + "\\out_ref.bin") : std::vector<char>();
  auto instr = read_file(insts_path);
  size_t instr_words = instr.size() / 4;
  size_t out_bytes = verify ? ref.size() : 2048 * 4;
  // v2: out size = E * 2048 * 4; derive E from wts size (401408B/expert)
  size_t n_exp = wts.size() / 401408;
  if (!verify) out_bytes = n_exp * 2048 * 4;

  try {
    xrt::device dev{0};
    xrt::xclbin xclbin{std::string(xclbin_path)};
    auto uuid = dev.register_xclbin(xclbin);
    xrt::hw_context hwctx{dev, uuid};
    xrt::kernel kernel{hwctx, "MLIR_AIE"};
    std::printf("kernel MLIR_AIE ready; experts=%zu wts=%.1fMB out=%zuKB\n",
                n_exp, wts.size() / 1048576.0, out_bytes / 1024);

    xrt::bo bo_instr(dev, instr.size(), XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    xrt::bo bo_wts(dev, wts.size(), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    xrt::bo bo_act(dev, act.size(), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
    xrt::bo bo_out(dev, out_bytes, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

    memcpy(bo_instr.map(), instr.data(), instr.size());
    memcpy(bo_wts.map(), wts.data(), wts.size());
    memcpy(bo_act.map(), act.data(), act.size());
    memset(bo_out.map(), 0, out_bytes);
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    std::vector<double> times;
    times.reserve(iters);
    for (int i = 0; i < iters; i++) {
      xrt::run run{kernel};
      run.set_arg(0, 3);
      run.set_arg(1, bo_instr);
      run.set_arg(2, (unsigned int)instr_words);
      run.set_arg(3, bo_wts);
      run.set_arg(4, bo_act);
      run.set_arg(5, bo_out);
      auto t0 = std::chrono::steady_clock::now();
      run.start();
      run.wait2();
      auto t1 = std::chrono::steady_clock::now();
      times.push_back(std::chrono::duration<double>(t1 - t0).count());
    }

    int bad = 0;
    float max_rel = 0;
    if (verify) {
      const float* yb = bo_out.map<const float*>();
      const float* yr = reinterpret_cast<const float*>(ref.data());
      size_t n = ref.size() / 4;
      // v3: out 到达为 column-major [col][E][rows_per_col]；重排回 row-major [E][OUT]
      std::vector<float> yrm(n);
      size_t rpc = 2048 / ncols, per_col = n_exp * rpc;
      for (size_t i = 0; i < n; i++) {
        size_t e = i / 2048, r = i % 2048, c = r / rpc;
        yrm[i] = yb[c * per_col + e * rpc + (r % rpc)];
      }
      const float* y = yrm.data();
      for (size_t i = 0; i < n; i++) {
        float d = std::fabs(y[i] - yr[i]);
        float rel = d / (std::fabs(yr[i]) + 1e-6f);
        if (rel > max_rel) max_rel = rel;
        if (rel > 1e-3f && d > 1e-3f) {
          if (bad < 5) std::printf("  y[%zu] = %f  ref %f\n", i, y[i], yr[i]);
          bad++;
        }
      }
      std::printf("verify: %s (bad=%d max_rel=%.2e)\n", bad ? "FAIL" : "PASS", bad, max_rel);
    }

    std::vector<double> warm(times.begin() + (times.size() > 5 ? 5 : 0), times.end());
    double mn = *std::min_element(warm.begin(), warm.end());
    double mx = *std::max_element(warm.begin(), warm.end());
    double avg = 0; for (double t : warm) avg += t; avg /= warm.size();
    double gb = (double)wts.size() / 1e9;
    std::printf("RESULT iters=%d experts=%zu  per-run ms: min=%.3f avg=%.3f max=%.3f\n",
                iters, n_exp, mn * 1e3, avg * 1e3, mx * 1e3);
    std::printf("RESULT weight-stream GB/s: min=%.2f avg=%.2f max=%.2f (warm)\n",
                gb / mx, gb / avg, gb / mn);
    return bad ? 1 : 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ERROR: %s\n", e.what());
    return 2;
  }
}
