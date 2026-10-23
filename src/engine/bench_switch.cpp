// bench_switch.cpp — 测量 NPU xclbin 切换代价：同形状连跑 vs 两形状交替
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/experimental/xrt_xclbin.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

static std::vector<char> rd(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  f.seekg(0, std::ios::end); size_t n = f.tellg(); f.seekg(0);
  std::vector<char> d(n); f.read(d.data(), n); return d;
}

struct Sh {
  xrt::kernel k; xrt::bo bi, ba, bb, bc; unsigned iw;
  Sh(xrt::device& dev, const std::string& base, int K, int N) {
    auto inst = rd(base + ".insts.bin");
    auto xcl = xrt::xclbin(base + ".xclbin");
    auto uuid = dev.register_xclbin(xcl);
    auto* hw = new xrt::hw_context(dev, uuid);
    k = xrt::kernel(*hw, "MLIR_AIE");
    bi = xrt::bo(dev, inst.size(), XCL_BO_FLAGS_CACHEABLE, k.group_id(1));
    memcpy(bi.map(), inst.data(), inst.size()); bi.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    iw = inst.size() / 4;
    ba = xrt::bo(dev, K * 2, XRT_BO_FLAGS_HOST_ONLY, k.group_id(3));
    bb = xrt::bo(dev, 8912896, XRT_BO_FLAGS_HOST_ONLY, k.group_id(4));
    bc = xrt::bo(dev, N * 2, XRT_BO_FLAGS_HOST_ONLY, k.group_id(5));
    memset(ba.map(), 1, K * 2); ba.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    memset(bb.map(), 3, 8912896); bb.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }
  xrt::run mk() {
    xrt::run r{k};
    r.set_arg(0, 3); r.set_arg(1, bi); r.set_arg(2, iw);
    r.set_arg(3, ba); r.set_arg(4, bb); r.set_arg(5, bc);
    return r;
  }
  double run1() {
    auto r = mk();
    auto t0 = std::chrono::steady_clock::now();
    r.start(); r.wait2();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e3;
  }
  double run1_reuse(xrt::run& r) {
    auto t0 = std::chrono::steady_clock::now();
    r.start(); r.wait2();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e3;
  }
};

int main(int argc, char** argv) {
  std::string dir = "models/npu";
  xrt::device dev{0};
  Sh a(dev, dir + "/k2048n8192", 2048, 8192);
  Sh b(dev, dir + "/k2048n4096", 2048, 4096);
  // warm
  for (int i = 0; i < 5; i++) { a.run1(); b.run1(); }
  { auto s = std::chrono::steady_clock::now(); for (int i = 0; i < 20; i++) a.run1();
    printf("same-shape A×20: %.3f ms/run\n", std::chrono::duration<double>(std::chrono::steady_clock::now()-s).count()*50); }
  { auto s = std::chrono::steady_clock::now(); for (int i = 0; i < 20; i++) b.run1();
    printf("same-shape B×20: %.3f ms/run\n", std::chrono::duration<double>(std::chrono::steady_clock::now()-s).count()*50); }
  { auto s = std::chrono::steady_clock::now(); for (int i = 0; i < 20; i++) { a.run1(); b.run1(); }
    printf("alternate A/B ×20对: %.3f ms/对\n", std::chrono::duration<double>(std::chrono::steady_clock::now()-s).count()*50); }
  return 0;
}
