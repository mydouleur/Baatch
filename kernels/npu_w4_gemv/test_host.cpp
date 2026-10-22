// test_host.cpp — Windows XRT host for npu_w4_gemv（任务 0a）。
// Loads CI-built xclbin + insts.bin, feeds a_bf16.bin/b_packed.bin, verifies vs c_ref.bin。
// Host ABI（runtime_sequence 操作数序 [A, B, C]）：arg0=opcode 3, arg1=instr bo,
// arg2=instr 字数, arg3=A, arg4=B, arg5=C。
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

static inline float bf16_to_f32(uint16_t b) {
  uint32_t u = (uint32_t)b << 16;
  float f; memcpy(&f, &u, 4); return f;
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

  auto a_dat = read_file(dir + "\\a_bf16.bin");
  auto b_dat = read_file(dir + "\\b_packed.bin");
  auto ref = verify ? read_file(dir + "\\c_ref.bin") : std::vector<char>();
  auto instr = read_file(insts_path);
  size_t instr_words = instr.size() / 4;
  size_t n_out = verify ? ref.size() / 4 : 4096;
  size_t out_bytes = n_out * 2;  // bf16

  try {
    xrt::device dev{0};
    xrt::xclbin xclbin{std::string(xclbin_path)};
    auto uuid = dev.register_xclbin(xclbin);
    xrt::hw_context hwctx{dev, uuid};
    xrt::kernel kernel{hwctx, "MLIR_AIE"};
    std::printf("kernel MLIR_AIE ready; B=%.1fMB A=%zuKB out=%zu bf16\n",
                b_dat.size() / 1048576.0, a_dat.size() / 1024, n_out);

    xrt::bo bo_instr(dev, instr.size(), XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    xrt::bo bo_a(dev, a_dat.size(), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    xrt::bo bo_b(dev, b_dat.size(), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
    xrt::bo bo_out(dev, out_bytes, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

    memcpy(bo_instr.map(), instr.data(), instr.size());
    memcpy(bo_a.map(), a_dat.data(), a_dat.size());
    memcpy(bo_b.map(), b_dat.data(), b_dat.size());
    memset(bo_out.map(), 0, out_bytes);
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    std::vector<double> times;
    times.reserve(iters);
    for (int i = 0; i < iters; i++) {
      xrt::run run{kernel};
      run.set_arg(0, 3);
      run.set_arg(1, bo_instr);
      run.set_arg(2, (unsigned int)instr_words);
      run.set_arg(3, bo_a);
      run.set_arg(4, bo_b);
      run.set_arg(5, bo_out);
      auto t0 = std::chrono::steady_clock::now();
      run.start();
      run.wait2();
      auto t1 = std::chrono::steady_clock::now();
      times.push_back(std::chrono::duration<double>(t1 - t0).count());
    }
    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    int bad = 0;
    float max_rel = 0, max_abs = 0;
    {  // dump 输出（转 fp32）供离线分析
      auto* yp = bo_out.map<uint16_t*>();
      std::ofstream of("out_npu_w4.bin", std::ios::binary);
      for (size_t i = 0; i < n_out; i++) {
        float f = bf16_to_f32(yp[i]);
        of.write((char*)&f, 4);
      }
    }
    if (verify) {
      const uint16_t* yb = bo_out.map<const uint16_t*>();
      const float* yr = reinterpret_cast<const float*>(ref.data());
      for (size_t i = 0; i < n_out; i++) {
        float y = bf16_to_f32(yb[i]);
        float d = std::fabs(y - yr[i]);
        float rel = d / (std::fabs(yr[i]) + 1e-6f);
        if (rel > max_rel) max_rel = rel;
        if (d > max_abs) max_abs = d;
        // bf16 输出量化 ~0.4% rel；双门容差（近零元走 abs）
        if (rel > 2e-2f && d > 5e-3f) {
          if (bad < 5) std::printf("  y[%zu] = %f  ref %f\n", i, y, yr[i]);
          bad++;
        }
      }
      std::printf("verify: %s (bad=%d max_rel=%.2e max_abs=%.2e)\n",
                  bad ? "FAIL" : "PASS", bad, max_rel, max_abs);
    }

    std::vector<double> warm(times.begin() + (times.size() > 5 ? 5 : 0), times.end());
    double mn = *std::min_element(warm.begin(), warm.end());
    double mx = *std::max_element(warm.begin(), warm.end());
    double avg = 0; for (double t : warm) avg += t; avg /= warm.size();
    double gb = (double)b_dat.size() / 1e9;
    std::printf("RESULT iters=%d  per-run ms: min=%.3f avg=%.3f max=%.3f\n",
                iters, mn * 1e3, avg * 1e3, mx * 1e3);
    std::printf("RESULT weight-stream GB/s: min=%.2f avg=%.2f max=%.2f (warm)\n",
                gb / mx, gb / avg, gb / mn);
    return bad ? 1 : 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ERROR: %s\n", e.what());
    return 2;
  }
}
