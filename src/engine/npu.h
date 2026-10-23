// npu.h — NPU W4A8 GEMV 后端（M3'-b）。kernel = rai_explore/vm_mix_int4_64x8_i8.cc
//（int16 VMAC，数值全 PASS，8192 档 43.6 GB/s）。设计 = TileFuse 8列×4行（预平铺 XBF4）。
//
// 实测取证（2026-10-24）：
// - 同形状连续调用 ~0.43ms/call（8.5MB）；**xclbin 切换 ~2.6ms/次**（驱动级 PDI 重配）
//   ——bench_switch.exe 实测（same-shape 0.43 vs alternate 5.8ms/对）。
// - **insts 热换（单 xclbin + 换指令缓冲）尝试过：挂死设备**（4 形状 PDI 逐位一致，
//   理论可行，但实测把驱动拖进僵尸态，需重启恢复）——**禁止再用，留 v2 查因**。
// - flake（~10-13% 整块输出为 0，见 rai_explore/REPORT.md）：校验+重试≤3+计数。
// - run 超时/异常 = 挂起语义：不重试直接 false（上层 CPU 兜底）。
#pragma once
#include "baa.h"
#include "kernels.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/experimental/xrt_xclbin.h"
#include <map>
#include <memory>
#include <fstream>
#include <chrono>

struct NpuBackend {
  struct Shape {
    xrt::kernel kernel;
    xrt::bo bi, ba, bc;
    unsigned instr_words;
    int K, N;
    std::map<const void*, xrt::bo> bcache;   // 权重 BO 按容器数据指针缓存（加载一次）
  };
  xrt::device dev{0};
  std::map<uint64_t, std::shared_ptr<Shape>> shapes;   // key = K<<32|N
  bool ok = false;
  long runs = 0, flakes = 0;
  std::string dir;
  std::vector<int16_t> aq;

  void init(const std::string& d) { dir = d; ok = true; }

  Shape& shape_for(int K, int N) {
    uint64_t key = (uint64_t)K << 32 | (unsigned)N;
    auto it = shapes.find(key);
    if (it != shapes.end()) return *it->second;
    auto s = std::make_shared<Shape>();
    std::string base = dir + "/k" + std::to_string(K) + "n" + std::to_string(N);
    std::ifstream f(base + ".insts.bin", std::ios::binary);
    if (!f) throw std::runtime_error("npu shape missing: " + base);
    f.seekg(0, std::ios::end); size_t n = f.tellg(); f.seekg(0);
    std::vector<char> inst(n); f.read(inst.data(), n);
    auto xcl = xrt::xclbin(base + ".xclbin");
    auto uuid = dev.register_xclbin(xcl);
    xrt::hw_context* hw = new xrt::hw_context(dev, uuid);   // 常驻（泄漏可接受）
    s->kernel = xrt::kernel(*hw, "MLIR_AIE");
    s->bi = xrt::bo(dev, inst.size(), XCL_BO_FLAGS_CACHEABLE, s->kernel.group_id(1));
    memcpy(s->bi.map(), inst.data(), inst.size());
    s->bi.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    s->ba = xrt::bo(dev, K * 2, XRT_BO_FLAGS_HOST_ONLY, s->kernel.group_id(3));
    s->bc = xrt::bo(dev, N * 2, XRT_BO_FLAGS_HOST_ONLY, s->kernel.group_id(5));
    s->instr_words = inst.size() / 4; s->K = K; s->N = N;
    shapes[key] = s;
    fprintf(stderr, "[npu] shape k%d n%d ready\n", K, N);
    return *s;
  }

  xrt::bo& bo_for(Shape& s, const BaaTensor& t) {
    auto it = s.bcache.find(t.data);
    if (it != s.bcache.end()) return it->second;
    xrt::bo b(dev, t.size, XRT_BO_FLAGS_HOST_ONLY, s.kernel.group_id(4));
    memcpy(b.map(), t.data, t.size);
    b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto r = s.bcache.emplace(t.data, std::move(b));
    return r.first->second;
  }

  // y = W_pretiled(N,K) @ x × sx；flake（整块 0）重试 ≤3
  bool gemv(const BaaTensor& t, const float* x, float* y) {
    int N = (int)t.shape[0], K = (int)t.shape[1];
    Shape& s = shape_for(K, N);
    xrt::bo& bb = bo_for(s, t);
    aq.resize(K);
    float amax = 0;
    for (int i = 0; i < K; i++) amax = std::max(amax, fabsf(x[i]));
    float sx = std::max(amax / 127.f, 1e-12f);
    for (int i = 0; i < K; i++) {
      int v = (int)lroundf(x[i] / sx);
      aq[i] = (int16_t)std::max(-128, std::min(127, v));
    }
    memcpy(s.ba.map(), aq.data(), K * 2);
    s.ba.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    for (int att = 0; att < 4; att++) {
      memset(s.bc.map(), 0, N * 2);
      s.bc.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      xrt::run run{s.kernel};
      run.set_arg(0, 3);
      run.set_arg(1, s.bi); run.set_arg(2, s.instr_words);
      run.set_arg(3, s.ba); run.set_arg(4, bb); run.set_arg(5, s.bc);
      run.start();
      try { run.wait2(); }
      catch (const std::exception& e) {
        fprintf(stderr, "[npu] run 超时/挂起：%s — 不重试（看门狗语义）\n", e.what());
        runs++;
        return false;
      }
      runs++;
      s.bc.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      const uint16_t* c = (const uint16_t*)s.bc.map();
      bool allzero = true;
      for (int i = 0; i < N; i++) if (c[i] != 0) { allzero = false; break; }
      if (!allzero) {
        for (int i = 0; i < N; i++) y[i] = bf16_to_f32(c[i]) * sx;
        return true;
      }
      flakes++;
    }
    fprintf(stderr, "[npu] flake ×4（记数继续）\n");
    return false;
  }
  ~NpuBackend() {
    if (runs) fprintf(stderr, "[npu] runs=%ld flakes=%ld (%.2f%%)\n", runs, flakes, 100.0 * flakes / runs);
  }
};
