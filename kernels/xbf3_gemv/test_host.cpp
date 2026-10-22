// test_host.cpp — Windows XRT host for xbf3_gemv (M2'-1).
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
    std::printf("usage: %s <xclbin> <insts.bin> <datadir> [iters]\n", argv[0]);
    return 1;
  }
  const char* xclbin_path = argv[1];
  const char* insts_path = argv[2];
  std::string dir = argv[3];
  int iters = argc > 4 ? atoi(argv[4]) : 20;

  auto wts = read_file(dir + "\\wts.bin");      // 401408 B
  auto act = read_file(dir + "\\act.bin");      // 2048 B
  auto ref = read_file(dir + "\\out_ref.bin");  // 8192 B
  auto instr = read_file(insts_path);
  size_t instr_words = instr.size() / 4;

  try {
    xrt::device dev{0};
    xrt::xclbin xclbin{std::string(xclbin_path)};
    auto uuid = dev.register_xclbin(xclbin);
    xrt::hw_context hwctx{dev, uuid};
    xrt::kernel kernel{hwctx, "MLIR_AIE"};
    std::printf("kernel MLIR_AIE ready\n");

    xrt::bo bo_instr(dev, instr.size(), XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    xrt::bo bo_wts(dev, wts.size(), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    xrt::bo bo_act(dev, act.size(), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
    xrt::bo bo_out(dev, ref.size(), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

    memcpy(bo_instr.map(), instr.data(), instr.size());
    memcpy(bo_wts.map(), wts.data(), wts.size());
    memcpy(bo_act.map(), act.data(), act.size());
    memset(bo_out.map(), 0, ref.size());
    // host-only/coherent BOs on NPU: no sync needed (M0 finding); instr is cacheable:
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    std::vector<double> times;
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
      if (i == 0) std::printf("iter0 %.3f ms\n", times[0] * 1e3);
    }

    // dump for offline analysis
    { auto* yp = bo_out.map<float*>();
      std::ofstream of("out_npu.bin", std::ios::binary);
      of.write((char*)yp, 2048*4); }
    // verify
    const float* y = bo_out.map<const float*>();
    const float* yr = reinterpret_cast<const float*>(ref.data());
    int bad = 0;
    float max_rel = 0;
    for (int i = 0; i < 2048; i++) {
      float d = std::fabs(y[i] - yr[i]);
      float rel = d / (std::fabs(yr[i]) + 1e-6f);
      if (rel > max_rel) max_rel = rel;
      if (rel > 1e-3f && d > 1e-3f) {
        if (bad < 5) std::printf("  y[%d] = %f  ref %f\n", i, y[i], yr[i]);
        bad++;
      }
    }
    std::printf("verify: %s (bad=%d max_rel=%.2e)\n", bad ? "FAIL" : "PASS", bad, max_rel);

    double avg = 0; for (double t : times) avg += t; avg /= times.size();
    double bw = (double)wts.size() / avg / 1e9;
    std::printf("RESULT iters=%d avg=%.3f ms  weight-stream %.2f GB/s\n",
                iters, avg * 1e3, bw);
    return bad ? 1 : 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ERROR: %s\n", e.what());
    return 2;
  }
}
