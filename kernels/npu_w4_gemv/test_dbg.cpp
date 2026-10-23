// test_dbg.cpp — dequant_dbg 最小复现 host（in → st1/st2/st3 dump）
// ABI: arg3=in(4352B), arg4=st1(8KB), arg5=st2(8KB), arg6=st3(16KB)
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/experimental/xrt_xclbin.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>

static std::vector<char> read_file(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) { fprintf(stderr, "cannot open %s\n", p.c_str()); exit(1); }
  f.seekg(0, std::ios::end); size_t n = f.tellg(); f.seekg(0);
  std::vector<char> d(n); f.read(d.data(), n); return d;
}

int main(int argc, char** argv) {
  if (argc < 4) { printf("usage: %s <xclbin> <insts> <in.bin> [outprefix]\n", argv[0]); return 1; }
  auto inp = read_file(argv[3]);
  std::string prefix = argc > 4 ? argv[4] : "dbg";
  auto instr = read_file(argv[2]);
  try {
    xrt::device dev{0};
    xrt::xclbin xclbin{std::string(argv[1])};
    auto uuid = dev.register_xclbin(xclbin);
    xrt::hw_context hwctx{dev, uuid};
    xrt::kernel kernel{hwctx, "MLIR_AIE"};
    xrt::bo bi(dev, instr.size(), XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    xrt::bo b0(dev, inp.size(), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    xrt::bo b1(dev, 8192, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
    xrt::bo b2(dev, 8192, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
    xrt::bo b3(dev, 16384, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));
    memcpy(bi.map(), instr.data(), instr.size());
    memcpy(b0.map(), inp.data(), inp.size());
    memset(b1.map(), 0xEE, 8192); memset(b2.map(), 0xEE, 8192); memset(b3.map(), 0xEE, 16384);
    bi.sync(XCL_BO_SYNC_BO_TO_DEVICE); b0.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    xrt::run run{kernel};
    run.set_arg(0, 3); run.set_arg(1, bi); run.set_arg(2, (unsigned)(instr.size() / 4));
    run.set_arg(3, b0); run.set_arg(4, b1); run.set_arg(5, b2); run.set_arg(6, b3);
    run.start(); run.wait2();
    b1.sync(XCL_BO_SYNC_BO_FROM_DEVICE); b2.sync(XCL_BO_SYNC_BO_FROM_DEVICE); b3.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    { std::ofstream f(prefix + ".st1.bin", std::ios::binary); f.write((char*)b1.map(), 8192); }
    { std::ofstream f(prefix + ".st2.bin", std::ios::binary); f.write((char*)b2.map(), 8192); }
    { std::ofstream f(prefix + ".st3.bin", std::ios::binary); f.write((char*)b3.map(), 16384); }
    printf("OK dumped %s.st{1,2,3}.bin\n", prefix.c_str());
    return 0;
  } catch (const std::exception& e) { fprintf(stderr, "ERROR: %s\n", e.what()); return 2; }
}
