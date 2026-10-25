// SPDX-License-Identifier: Apache-2.0
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/experimental/xrt_xclbin.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

static std::vector<char> read_file(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  f.seekg(0, std::ios::end);
  size_t size = f.tellg();
  f.seekg(0);
  std::vector<char> data(size);
  f.read(data.data(), size);
  return data;
}

static float bf16_to_f32(uint16_t value) {
  uint32_t bits = uint32_t(value) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

struct Result {
  std::vector<double> times;
  size_t calls = 0;
  size_t bad_iterations = 0;
  size_t bad_values = 0;
  size_t exact_mismatches = 0;
  float max_rel = 0;
  float max_abs = 0;
};

class Runner {
public:
  Runner(const char *xclbin_path, const char *insts_path, size_t a_bytes,
         size_t b_bytes, size_t out_bytes)
      : device(0), xclbin(std::string(xclbin_path)), uuid(device.register_xclbin(xclbin)),
        context(device, uuid), kernel(context, "MLIR_AIE"),
        instr_data(read_file(insts_path)),
        instr(device, instr_data.size(), XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1)),
        output(device, out_bytes, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5)) {
    std::memcpy(instr.map(), instr_data.data(), instr_data.size());
    instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    std::memset(output.map(), 0, out_bytes);
    (void)a_bytes;
    (void)b_bytes;
  }

  std::unique_ptr<xrt::bo> make_input(size_t bytes, int group) {
    return std::make_unique<xrt::bo>(device, bytes, XRT_BO_FLAGS_HOST_ONLY,
                                     kernel.group_id(group));
  }

  void call(xrt::bo &a, xrt::bo &b) {
    xrt::run run{kernel};
    run.set_arg(0, 3);
    run.set_arg(1, instr);
    run.set_arg(2, static_cast<unsigned int>(instr_data.size() / 4));
    run.set_arg(3, a);
    run.set_arg(4, b);
    run.set_arg(5, output);
    run.start();
    run.wait2();
    output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  }

private:
  xrt::device device;
  xrt::xclbin xclbin;
  xrt::uuid uuid;
  xrt::hw_context context;
  xrt::kernel kernel;
  std::vector<char> instr_data;
  xrt::bo instr;
public:
  xrt::bo output;
};

static size_t native_ref_index(size_t n, size_t m, size_t N, size_t M) {
  size_t transfer = n / 2048;
  size_t within = n % 2048;
  size_t row = within / 512;
  size_t col = (within % 512) / 64;
  size_t lane = n % 64;
  return (((transfer * 8 + col) * 4 + row) * M + m) * 64 + lane;
}

static void check_value(uint16_t actual_bits, float expected, size_t index,
                        Result &result, bool &iteration_bad) {
  float actual = bf16_to_f32(actual_bits);
  if (actual != expected) ++result.exact_mismatches;
  float abs = std::fabs(actual - expected);
  float rel = abs / (std::fabs(expected) + 1e-6f);
  result.max_abs = std::max(result.max_abs, abs);
  result.max_rel = std::max(result.max_rel, rel);
  if (rel > 2e-2f && abs > 5e-3f) {
    if (result.bad_values < 5)
      std::printf("  native[%zu] actual=%g expected=%g\n", index, actual, expected);
    ++result.bad_values;
    iteration_bad = true;
  }
}

static Result run_mode(const char *xclbin, const char *insts,
                       const std::vector<char> &a, const std::vector<char> &b,
                       const std::vector<float> &ref, size_t K, size_t N,
                       size_t M, int iters, bool m1_repeat) {
  size_t rows_per_call = m1_repeat ? 1 : M;
  size_t a_call_bytes = rows_per_call * K * sizeof(int16_t);
  size_t out_call_values = rows_per_call * N;
  Runner runner(xclbin, insts, a_call_bytes, b.size(),
                out_call_values * sizeof(uint16_t));
  auto bo_b = runner.make_input(b.size(), 4);
  std::memcpy(bo_b->map(), b.data(), b.size());
  bo_b->sync(XCL_BO_SYNC_BO_TO_DEVICE);

  std::vector<std::unique_ptr<xrt::bo>> inputs;
  if (m1_repeat) {
    const int16_t *packed = reinterpret_cast<const int16_t *>(a.data());
    for (size_t m = 0; m < M; ++m) {
      auto input = runner.make_input(K * sizeof(int16_t), 3);
      int16_t *dst = input->map<int16_t *>();
      for (size_t tile = 0; tile < K / 128; ++tile)
        std::memcpy(dst + tile * 128, packed + (tile * M + m) * 128,
                    128 * sizeof(int16_t));
      input->sync(XCL_BO_SYNC_BO_TO_DEVICE);
      inputs.push_back(std::move(input));
    }
  } else {
    auto input = runner.make_input(a.size(), 3);
    std::memcpy(input->map(), a.data(), a.size());
    input->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    inputs.push_back(std::move(input));
  }

  Result result;
  size_t calls_per_iter = m1_repeat ? M : 1;
  for (int iteration = 0; iteration < iters; ++iteration) {
    bool iteration_bad = false;
    double iteration_seconds = 0;
    for (size_t call = 0; call < calls_per_iter; ++call) {
      auto begin = std::chrono::steady_clock::now();
      runner.call(*inputs[call], *bo_b);
      auto end = std::chrono::steady_clock::now();
      iteration_seconds += std::chrono::duration<double>(end - begin).count();
      ++result.calls;
      const uint16_t *out = runner.output.map<const uint16_t *>();
      if (m1_repeat) {
        for (size_t n = 0; n < N; ++n) {
          size_t ref_index = native_ref_index(n, call, N, M);
          check_value(out[n], ref[ref_index], ref_index, result, iteration_bad);
        }
      } else {
        for (size_t index = 0; index < M * N; ++index)
          check_value(out[index], ref[index], index, result, iteration_bad);
      }
    }
    result.times.push_back(iteration_seconds);
    if (iteration_bad) ++result.bad_iterations;
  }
  return result;
}

static void print_result(const char *name, const Result &result, size_t M,
                         size_t weight_bytes, int iters) {
  size_t warmup = result.times.size() > 5 ? 5 : 0;
  double total = 0;
  for (size_t i = warmup; i < result.times.size(); ++i) total += result.times[i];
  size_t warm_iters = result.times.size() - warmup;
  double avg_ms = total / warm_iters * 1e3;
  double min_ms = *std::min_element(result.times.begin() + warmup, result.times.end()) * 1e3;
  double max_ms = *std::max_element(result.times.begin() + warmup, result.times.end()) * 1e3;
  size_t physical_weight_bytes = weight_bytes * (std::string(name) == "M1xM" ? M : 1);
  std::printf("%s verify=%s iterations=%d bad_iterations=%zu bad=%zu exact_mismatches=%zu max_rel=%.3e max_abs=%.3e\n",
              name, result.bad_values ? "FAIL" : "PASS", iters,
              result.bad_iterations, result.bad_values, result.exact_mismatches,
              result.max_rel, result.max_abs);
  std::printf("%s calls=%zu warm_iters=%zu batch_ms min=%.6f avg=%.6f max=%.6f per_token_ms=%.6f physical_weight_bytes=%zu effective_GBps=%.3f\n",
              name, result.calls, warm_iters, min_ms, avg_ms, max_ms,
              avg_ms / M, physical_weight_bytes,
              physical_weight_bytes / (avg_ms * 1e6));
}

int main(int argc, char **argv) {
  if (argc != 11) {
    std::printf("usage: %s <batch.xclbin> <batch.insts> <m1.xclbin> <m1.insts> <data> <K> <N> <M> <iters> <batch|m1|both>\n",
                argv[0]);
    return 2;
  }
  try {
    size_t K = std::stoull(argv[6]);
    size_t N = std::stoull(argv[7]);
    size_t M = std::stoull(argv[8]);
    int iters = std::atoi(argv[9]);
    std::string mode = argv[10];
    if (!M || !iters || (mode != "batch" && mode != "m1" && mode != "both")) return 2;
    std::string dir = argv[5];
    auto a = read_file(dir + "/a_bf16.bin");
    auto b = read_file(dir + "/b_packed.bin");
    auto ref_data = read_file(dir + "/c_ref.bin");
    if (a.size() != M * K * sizeof(int16_t) || ref_data.size() != M * N * sizeof(float))
      throw std::runtime_error("input size does not match K/N/M");
    const float *ref_ptr = reinterpret_cast<const float *>(ref_data.data());
    std::vector<float> ref(ref_ptr, ref_ptr + M * N);
    Result batch, m1;
    if (mode == "m1" || mode == "both") {
      m1 = run_mode(argv[3], argv[4], a, b, ref, K, N, M, iters, true);
      print_result("M1xM", m1, M, b.size(), iters);
      if (m1.bad_values) return 1;
    }
    if (mode == "batch" || mode == "both") {
      batch = run_mode(argv[1], argv[2], a, b, ref, K, N, M, iters, false);
      print_result("BATCH", batch, M, b.size(), iters);
      if (batch.bad_values) return 1;
    }
    if (mode == "both") {
      auto mean = [](const Result &r) {
        size_t skip = r.times.size() > 5 ? 5 : 0;
        return std::accumulate(r.times.begin() + skip, r.times.end(), 0.0) /
               (r.times.size() - skip);
      };
      double m1_avg = mean(m1), batch_avg = mean(batch);
      std::printf("COMPARISON M=%zu batch_vs_M1xM=%.3f%% token_speedup=%.3fx\n", M,
                  (batch_avg / m1_avg - 1.0) * 100.0, m1_avg / batch_avg);
    }
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "ERROR: %s\n", e.what());
    return 2;
  }
}
