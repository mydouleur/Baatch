// M0 DMA bandwidth probe host - XDNA2 NPU streaming bandwidth measurement
// Uses XRT C++ API against xrt_coreutil.dll (driver-installed XRT 2.19).
// Flow: validate xclbin (4-col DPU overlay) + df_bw ctrlcode ELF (shim DMA loopback).
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/experimental/xrt_xclbin.h"
#include "xrt/experimental/xrt_elf.h"
#include "xrt/experimental/xrt_module.h"
#include "xrt/experimental/xrt_ext.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>
#include <algorithm>

static void usage(const char* prog) {
  std::printf("usage: %s <xclbin> <elf> <iters> [bufsize_mb] [kernel_name]\n", prog);
}

int main(int argc, char** argv) {
  if (argc < 4) { usage(argv[0]); return 1; }
  std::string xclbin_path = argv[1];
  std::string elf_path = argv[2];
  int iters = std::atoi(argv[3]);
  size_t buf_mb = argc > 4 ? std::atoll(argv[4]) : 1024;
  size_t buf_size = buf_mb * 1024 * 1024;

  try {
    xrt::device dev{0};
    xrt::xclbin xclbin{xclbin_path};
    auto kernels = xclbin.get_kernels();
    std::printf("xclbin kernels:");
    for (auto& k : kernels) std::printf(" '%s'", k.get_name().c_str());
    std::printf("\n");

    std::string kname = argc > 5 ? argv[5] : (kernels.empty() ? "DPU" : kernels[0].get_name());
    auto uuid = dev.register_xclbin(xclbin);
    xrt::hw_context hwctx{dev, uuid};
    std::printf("hw_context created, kernel='%s'\n", kname.c_str());

    xrt::elf elf{elf_path};
    xrt::module mod{elf};
    xrt::ext::kernel kernel{hwctx, mod, kname};
    std::printf("ext::kernel created (elf flow)\n");

    // buffers: ext::bo host-visible
    xrt::ext::bo bo_ifm{dev, buf_size};
    xrt::ext::bo bo_ofm{dev, buf_size};
    std::printf("buffers: ifm=%p ofm=%p size=%zu MB\n",
                (void*)bo_ifm.address(), (void*)bo_ofm.address(), buf_mb);

    // fill input with pattern (host_only buffers are coherent; no sync needed)
    {
      auto* p = bo_ifm.map<int*>();
      for (size_t i = 0; i < buf_size / 4; i++) p[i] = (int)(i * 2654435761u);
    }

    std::vector<double> gbps;
    gbps.reserve(iters);
    FILE* csv = std::fopen("iters.csv", "w");
    if (csv) std::fprintf(csv, "iter,ms,gbps\n");
    for (int i = 0; i < iters; i++) {
      xrt::run run{kernel};
      // DPU arg layout (recipe): arg0=opcode 3 (txn buffer), 1,2=0,
      // arg3=ifm, arg4=unused, arg5=ofm, 6,7=0
      run.set_arg(0, 3);
      run.set_arg(1, 0);
      run.set_arg(2, 0);
      run.set_arg(3, bo_ifm);
      run.set_arg(5, bo_ofm);
      run.set_arg(6, 0);
      run.set_arg(7, 0);
      auto t0 = std::chrono::steady_clock::now();
      run.start();
      run.wait2();
      auto t1 = std::chrono::steady_clock::now();
      double sec = std::chrono::duration<double>(t1 - t0).count();
      double bw = 2.0 * buf_size / sec / 1e9; // read+write
      gbps.push_back(bw);
      if (csv) std::fprintf(csv, "%d,%.6f,%.2f\n", i, sec * 1e3, bw);
      if (i < 3 || i == iters - 1)
        std::printf("iter %d: %.3f ms  %.2f GB/s\n", i, sec * 1e3, bw);
    }

    // verify
    auto* pi = bo_ifm.map<const int*>();
    auto* po = bo_ofm.map<const int*>();
    size_t bad = 0;
    for (size_t i = 0; i < buf_size / 4; i++)
      if (pi[i] != po[i]) { bad++; if (bad < 5) std::printf("mismatch @%zu: %d != %d\n", i, pi[i], po[i]); }
    std::printf("verify: %s (%zu mismatches)\n", bad ? "FAIL" : "PASS", bad);

    if (csv) std::fclose(csv);
    // stats excluding first 5 warmup iters
    std::vector<double> warm(gbps.begin() + (gbps.size() > 10 ? 5 : 0), gbps.end());
    auto mn0 = *std::min_element(gbps.begin(), gbps.end());
    auto mx0 = *std::max_element(gbps.begin(), gbps.end());
    double avg0 = std::accumulate(gbps.begin(), gbps.end(), 0.0) / gbps.size();
    auto mn = *std::min_element(warm.begin(), warm.end());
    auto mx = *std::max_element(warm.begin(), warm.end());
    double avg = std::accumulate(warm.begin(), warm.end(), 0.0) / warm.size();
    std::printf("RESULT iters=%d buf=%zuMB  all: min=%.2f avg=%.2f max=%.2f | warm(excl first 5): min=%.2f avg=%.2f max=%.2f GB/s\n",
                iters, buf_mb, mn0, avg0, mx0, mn, avg, mx);
    return 0;
  }
  catch (const std::exception& e) {
    std::fprintf(stderr, "ERROR: %s\n", e.what());
    return 2;
  }
}
