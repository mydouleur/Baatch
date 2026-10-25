// SPDX-License-Identifier: Apache-2.0
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/experimental/xrt_xclbin.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
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

struct Data {
  std::vector<char> a;
  std::vector<char> b;
  std::vector<float> ref;
};

class Runner {
public:
  Runner(const char *xclbin_path, const char *insts_path, size_t out_bytes)
      : device(0), xclbin(std::string(xclbin_path)), uuid(device.register_xclbin(xclbin)),
        context(device, uuid), kernel(context, "MLIR_AIE"),
        instr_data(read_file(insts_path)),
        instr(device, instr_data.size(), XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1)),
        output(device, out_bytes, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5)) {
    std::memcpy(instr.map(), instr_data.data(), instr_data.size());
    instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    std::memset(output.map(), 0, out_bytes);
  }

  std::unique_ptr<xrt::bo> make_input(const std::vector<char> &data, int group) {
    auto bo = std::make_unique<xrt::bo>(device, data.size(), XRT_BO_FLAGS_HOST_ONLY,
                                        kernel.group_id(group));
    std::memcpy(bo->map(), data.data(), data.size());
    bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    return bo;
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

struct Stats {
  size_t calls = 0;
  size_t bad_iterations = 0;
  size_t bad_values = 0;
  size_t exact_mismatches = 0;
  float max_rel = 0;
  float max_abs = 0;
};

static void check_output(Runner &runner, const Data &data, size_t M, size_t N,
                         Stats &stats) {
  bool iteration_bad = false;
  const uint16_t *out = runner.output.map<const uint16_t *>();
  for (size_t index = 0; index < M * N; ++index) {
    float actual = bf16_to_f32(out[index]);
    float expected = data.ref[index];
    if (actual != expected) ++stats.exact_mismatches;
    float abs = std::fabs(actual - expected);
    float rel = abs / (std::fabs(expected) + 1e-6f);
    stats.max_abs = std::max(stats.max_abs, abs);
    stats.max_rel = std::max(stats.max_rel, rel);
    if (rel > 2e-2f && abs > 5e-3f) {
      if (stats.bad_values < 5)
        std::printf("  call=%zu native[%zu] actual=%g expected=%g\n",
                    stats.calls, index, actual, expected);
      ++stats.bad_values;
      iteration_bad = true;
    }
  }
  if (iteration_bad) ++stats.bad_iterations;
}

static Data load_data(const std::string &dir, size_t K, size_t N, size_t M) {
  Data data;
  data.a = read_file(dir + "/a_bf16.bin");
  data.b = read_file(dir + "/b_packed.bin");
  auto ref = read_file(dir + "/c_ref.bin");
  if (data.a.size() != M * K * sizeof(int16_t) ||
      ref.size() != M * N * sizeof(float))
    throw std::runtime_error("input size does not match K/N/M in " + dir);
  const float *ref_ptr = reinterpret_cast<const float *>(ref.data());
  data.ref.assign(ref_ptr, ref_ptr + M * N);
  return data;
}

int main(int argc, char **argv) {
  if (argc != 11 && argc != 12) {
    std::printf("usage: %s <batch.xclbin> <batch.insts> <w0a0> <w0a1> <w1a0> <w1a1> <K> <N> <M> <iters-per-phase> [phase 0..3]\n",
                argv[0]);
    return 2;
  }
  try {
    size_t K = std::stoull(argv[7]);
    size_t N = std::stoull(argv[8]);
    size_t M = std::stoull(argv[9]);
    int iters = std::atoi(argv[10]);
    int only_phase = argc == 12 ? std::atoi(argv[11]) : -1;
    if (!M || !iters || only_phase < -1 || only_phase > 3) return 2;
    Data data[2][2] = {
        {load_data(argv[3], K, N, M), load_data(argv[4], K, N, M)},
        {load_data(argv[5], K, N, M), load_data(argv[6], K, N, M)}};
    Stats total;
    const char *phase_names[] = {"reuse", "fresh_weight", "fresh_inputs", "fresh_runner"};
    for (int phase = 0; phase < 4; ++phase) {
      if (only_phase >= 0 && phase != only_phase) continue;
      Stats stats;
      std::unique_ptr<Runner> persistent;
      std::unique_ptr<xrt::bo> persistent_a[2][2];
      std::unique_ptr<xrt::bo> persistent_b[2];
      if (phase != 3) {
        persistent = std::make_unique<Runner>(argv[1], argv[2], M * N * sizeof(uint16_t));
        if (phase == 0) {
          for (int w = 0; w < 2; ++w) {
            persistent_b[w] = persistent->make_input(data[w][0].b, 4);
            for (int a = 0; a < 2; ++a)
              persistent_a[w][a] = persistent->make_input(data[w][a].a, 3);
          }
        } else if (phase == 1) {
          for (int w = 0; w < 2; ++w)
            for (int a = 0; a < 2; ++a)
              persistent_a[w][a] = persistent->make_input(data[w][a].a, 3);
        }
      }
      for (int iteration = 0; iteration < iters; ++iteration) {
        int w = iteration & 1;
        int a = (iteration >> 1) & 1;
        Runner *runner = persistent.get();
        std::unique_ptr<Runner> transient;
        if (phase == 3) {
          transient = std::make_unique<Runner>(argv[1], argv[2], M * N * sizeof(uint16_t));
          runner = transient.get();
        }
        std::unique_ptr<xrt::bo> transient_a;
        std::unique_ptr<xrt::bo> transient_b;
        xrt::bo *bo_a = nullptr;
        xrt::bo *bo_b = nullptr;
        if (phase == 0 || phase == 1) {
          bo_a = persistent_a[w][a].get();
        } else {
          transient_a = runner->make_input(data[w][a].a, 3);
          bo_a = transient_a.get();
        }
        if (phase == 0) {
          bo_b = persistent_b[w].get();
        } else {
          transient_b = runner->make_input(data[w][a].b, 4);
          bo_b = transient_b.get();
        }
        runner->call(*bo_a, *bo_b);
        ++stats.calls;
        check_output(*runner, data[w][a], M, N, stats);
      }
      std::printf("%s verify=%s calls=%zu bad_iterations=%zu bad=%zu exact_mismatches=%zu max_rel=%.3e max_abs=%.3e\n",
                  phase_names[phase], stats.bad_values ? "FAIL" : "PASS", stats.calls,
                  stats.bad_iterations, stats.bad_values, stats.exact_mismatches,
                  stats.max_rel, stats.max_abs);
      total.calls += stats.calls;
      total.bad_iterations += stats.bad_iterations;
      total.bad_values += stats.bad_values;
      total.exact_mismatches += stats.exact_mismatches;
      total.max_rel = std::max(total.max_rel, stats.max_rel);
      total.max_abs = std::max(total.max_abs, stats.max_abs);
      if (stats.bad_values) return 1;
    }
    std::printf("TOTAL verify=PASS calls=%zu bad_iterations=0 bad=0 exact_mismatches=%zu max_rel=%.3e max_abs=%.3e\n",
                total.calls, total.exact_mismatches, total.max_rel, total.max_abs);
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "ERROR: %s\n", e.what());
    return 2;
  }
}
