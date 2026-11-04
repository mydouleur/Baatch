// SPDX-License-Identifier: Apache-2.0
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/experimental/xrt_xclbin.h"
#include "../../src/engine/kernels_xbf.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
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

static double elapsed_ms(std::chrono::steady_clock::time_point begin) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count() * 1e3;
}

class Runner {
public:
  Runner(const std::string &xclbin_path, const std::string &insts_path, size_t out_bytes)
      : device(0), xclbin(xclbin_path), uuid(device.register_xclbin(xclbin)),
        context(device, uuid), kernel(context, "MLIR_AIE"),
        instr_data(read_file(insts_path)),
        instr(device, instr_data.size(), XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1)),
        output(device, out_bytes, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5)) {
    std::memcpy(instr.map(), instr_data.data(), instr_data.size());
    instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    std::memset(output.map(), 0, out_bytes);
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

struct MatrixData {
  int K, N;
  std::vector<char> tiled;
  std::vector<char> rowmajor;
};

static MatrixData load_matrix(const std::string &dir, int K, int N) {
  MatrixData data{K, N, read_file(dir + "/b_packed.bin"), read_file(dir + "/b_rowmajor.bin")};
  size_t tiled_size = (size_t)K * N / 64 * 34;
  size_t rowmajor_size = (size_t)K * N / 256 * 130;
  if (data.tiled.size() != tiled_size || data.rowmajor.size() != rowmajor_size)
    throw std::runtime_error("weight size mismatch in " + dir);
  return data;
}

struct QuantizedRows {
  std::vector<int8_t> q, xe, xo;
  std::vector<float> sums, sx;
};

static QuantizedRows quantize_rows(const float *x, int K, int M, bool split) {
  QuantizedRows out;
  out.q.resize((size_t)M * K);
  out.sums.resize((size_t)M * K / XBF_SB);
  out.sx.resize(M);
  if (split) {
    out.xe.resize((size_t)M * K / 2);
    out.xo.resize((size_t)M * K / 2);
  }
  for (int m = 0; m < M; ++m) {
    out.sx[m] = quant_xq(x + (size_t)m * K, K, out.q.data() + (size_t)m * K,
                         out.sums.data() + (size_t)m * K / XBF_SB);
    if (split)
      xq_split4(out.q.data() + (size_t)m * K, K, out.xe.data() + (size_t)m * K / 2,
                out.xo.data() + (size_t)m * K / 2);
  }
  return out;
}

static void activate_rows(const float *guv, float *hidden, int M) {
  for (int m = 0; m < M; ++m)
    for (int i = 0; i < 512; ++i) {
      float gate = guv[(size_t)m * 1024 + i];
      hidden[(size_t)m * 512 + i] = siluf(gate) * guv[(size_t)m * 1024 + 512 + i];
    }
}

static int dot_128(const uint8_t *w, const int8_t *q) {
  int sum = 0;
  for (int i = 0; i < 64; ++i) {
    sum += (w[i] & 15) * q[2 * i];
    sum += (w[i] >> 4) * q[2 * i + 1];
  }
  for (int i = 0; i < 128; ++i) sum -= 8 * q[i];
  return sum;
}

static void reference_gemm(const MatrixData &w, const QuantizedRows &x, int M,
                           bool npu_semantics, float *out) {
  int blocks = w.K / 128;
  size_t row_bytes = (size_t)w.K / 256 * 130;
  for (int m = 0; m < M; ++m) {
    const int8_t *q = x.q.data() + (size_t)m * w.K;
    for (int n = 0; n < w.N; ++n) {
      const uint8_t *row = reinterpret_cast<const uint8_t *>(w.rowmajor.data()) + n * row_bytes;
      float y = 0;
      if (npu_semantics) {
        for (int b = 0; b < blocks; ++b) {
          const uint8_t *wb = row + (b / 2) * 130 + 2 + (b % 2) * 64;
          int sum = dot_128(wb, q + b * 128);
          float scale = bf16_to_f32(row[(b / 2) * 130] | (row[(b / 2) * 130 + 1] << 8));
          float acc_bf = bf16_to_f32(f32_to_bf16((float)sum));
          y = bf16_to_f32(f32_to_bf16(y + acc_bf * scale));
        }
        out[(size_t)m * w.N + n] = y * x.sx[m];
      } else {
        for (int b = 0; b < blocks / 2; ++b) {
          const uint8_t *wb = row + b * 130 + 2;
          int sum = dot_128(wb, q + b * 256) + dot_128(wb + 64, q + b * 256 + 128);
          for (int i = 0; i < 256; ++i) sum += 8 * q[b * 256 + i];
          sum -= 8 * (int)x.sums[(size_t)m * blocks / 2 + b];
          float scale = bf16_to_f32(row[b * 130] | (row[b * 130 + 1] << 8));
          y += sum * x.sx[m] * scale;
        }
        out[(size_t)m * w.N + n] = y;
      }
    }
  }
}

static void reference_expert(const MatrixData &gate, const MatrixData &down,
                             const float *input, int M, bool npu_semantics, float *output) {
  auto x = quantize_rows(input, gate.K, M, false);
  std::vector<float> guv((size_t)M * gate.N), hidden((size_t)M * 512);
  reference_gemm(gate, x, M, npu_semantics, guv.data());
  activate_rows(guv.data(), hidden.data(), M);
  auto hq = quantize_rows(hidden.data(), down.K, M, false);
  reference_gemm(down, hq, M, npu_semantics, output);
}

class CpuExpert {
public:
  CpuExpert(const MatrixData &gate, const MatrixData &down) : gate(gate), down(down) {}

  void run(const float *input, int M, float *output) {
    auto x = quantize_rows(input, gate.K, M, true);
    std::vector<float> guv((size_t)M * gate.N), hidden((size_t)M * 512);
    int gate_half = gate.N / 2;
    pool.run([&](int id) {
      int begin = id * gate_half;
      gemm_xbf4_rows(reinterpret_cast<const uint8_t *>(gate.rowmajor.data()), begin, gate_half,
                     gate.K, M, gate.N, x.xe.data(), x.xo.data(), x.sums.data(), x.sx.data(),
                     guv.data() + begin);
    });
    activate_rows(guv.data(), hidden.data(), M);
    auto hq = quantize_rows(hidden.data(), down.K, M, true);
    int down_half = down.N / 2;
    if (M >= 4) {
      std::vector<float> transposed((size_t)M * down.N);
      pool.run([&](int id) {
        int begin = id * down_half;
        gemm_xbf4_down_m4<true>(reinterpret_cast<const uint8_t *>(down.rowmajor.data()), begin,
                               down_half, down.K, M, down.N, hq.xe.data(), hq.xo.data(),
                               hq.sums.data(), hq.sx.data(), transposed.data() + (size_t)begin * M);
      });
      for (int m = 0; m < M; ++m)
        for (int n = 0; n < down.N; ++n) output[(size_t)m * down.N + n] = transposed[(size_t)n * M + m];
    } else {
      pool.run([&](int id) {
        int begin = id * down_half;
        gemm_xbf4_rows(reinterpret_cast<const uint8_t *>(down.rowmajor.data()), begin, down_half,
                       down.K, M, down.N, hq.xe.data(), hq.xo.data(), hq.sums.data(), hq.sx.data(),
                       output + begin);
      });
    }
  }

private:
  const MatrixData &gate;
  const MatrixData &down;
  Pool pool;
};

class NpuShape {
public:
  NpuShape(const std::string &root, const std::string &prefix, const MatrixData &data,
           bool stable_single = false)
      : K(data.K), N(data.N),
        batch(root + "/" + prefix + "m8fix.xclbin", root + "/" + prefix + "m8fix.insts.bin",
              (size_t)8 * N * sizeof(uint16_t)),
        single(root + "/" + prefix + (stable_single ? "" : "m1fix") + ".xclbin",
               root + "/" + prefix + (stable_single ? "" : "m1fix") + ".insts.bin",
               (size_t)N * sizeof(uint16_t)),
        batch_weight(batch.make_input(data.tiled.size(), 4)),
        single_weight(single.make_input(data.tiled.size(), 4)),
        batch_a(batch.make_input((size_t)8 * K * sizeof(int16_t), 3)),
        single_a(single.make_input((size_t)K * sizeof(int16_t), 3)) {
    std::memcpy(batch_weight->map(), data.tiled.data(), data.tiled.size());
    std::memcpy(single_weight->map(), data.tiled.data(), data.tiled.size());
    batch_weight->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    single_weight->sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }

  void run_batch(const QuantizedRows &x, int M, float *output) {
    std::vector<int16_t> packed((size_t)M * K);
    for (int b = 0; b < K / 128; ++b)
      for (int m = 0; m < M; ++m)
        for (int i = 0; i < 128; ++i)
          packed[((size_t)b * M + m) * 128 + i] = x.q[(size_t)m * K + b * 128 + i];
    std::memcpy(batch_a->map(), packed.data(), packed.size() * sizeof(int16_t));
    batch_a->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    batch.call(*batch_a, *batch_weight);
    unpack(batch.output.map<const uint16_t *>(), x.sx.data(), M, output);
  }

  void run_single(const QuantizedRows &x, int row, float *output) {
    std::vector<int16_t> packed(K);
    for (int i = 0; i < K; ++i) packed[i] = x.q[(size_t)row * K + i];
    std::memcpy(single_a->map(), packed.data(), packed.size() * sizeof(int16_t));
    single_a->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    single.call(*single_a, *single_weight);
    const uint16_t *raw = single.output.map<const uint16_t *>();
    for (int n = 0; n < N; ++n) output[n] = bf16_to_f32(raw[n]) * x.sx[row];
  }

public:
  void unpack(const uint16_t *raw, const float *sx, int M, float *output) {
    int core_rows = N == 1024 ? 2 : 4;
    int span = core_rows * 512;
    for (int m = 0; m < M; ++m)
      for (int n = 0; n < N; ++n) {
        int tr = n / span, within = n % span;
        int row = within / 512, col = (within % 512) / 64, lane = n % 64;
        size_t index = (((size_t)(tr * 8 + col) * core_rows + row) * M + m) * 64 + lane;
        output[(size_t)m * N + n] = bf16_to_f32(raw[index]) * sx[m];
      }
  }

  int K, N;
  Runner batch, single;
  std::unique_ptr<xrt::bo> batch_weight, single_weight, batch_a, single_a;
};

class NpuExpert {
public:
  NpuExpert(const std::string &root, const MatrixData &gate_data, const MatrixData &down_data)
      : gate(root, "k2048n1024", gate_data), down(root, "k512n2048", down_data, true) {}

  void run(const float *input, int M, bool use_m8, float *output, size_t &calls) {
    auto x = quantize_rows(input, gate.K, M, false);
    std::vector<float> guv((size_t)M * gate.N), hidden((size_t)M * 512);
    run_matrix(gate, x, M, use_m8, guv.data(), calls);
    activate_rows(guv.data(), hidden.data(), M);
    auto hq = quantize_rows(hidden.data(), down.K, M, false);
    run_matrix(down, hq, M, use_m8, output, calls);
  }

private:
  void run_matrix(NpuShape &shape, const QuantizedRows &x, int M, bool use_m8,
                  float *output, size_t &calls) {
    int base = 0;
    if (use_m8) {
      for (; base + 8 <= M; base += 8) {
        QuantizedRows chunk;
        chunk.q.assign(x.q.begin() + (size_t)base * shape.K,
                       x.q.begin() + (size_t)(base + 8) * shape.K);
        chunk.sx.assign(x.sx.begin() + base, x.sx.begin() + base + 8);
        shape.run_batch(chunk, 8, output + (size_t)base * shape.N);
        ++calls;
      }
    }
    for (int m = base; m < M; ++m) {
      shape.run_single(x, m, output + (size_t)m * shape.N);
      ++calls;
    }
  }

  NpuShape gate, down;
};

struct Result {
  std::vector<double> times;
  size_t bad_iterations = 0, mismatches = 0, calls = 0;
};

template <typename Fn>
static Result measure(Fn &&fn, float *output, const float *expected, size_t values, int iters) {
  Result result;
  for (int iteration = 0; iteration < iters; ++iteration) {
    auto begin = std::chrono::steady_clock::now();
    fn();
    result.times.push_back(elapsed_ms(begin));
    size_t bad = 0;
    for (size_t i = 0; i < values; ++i) {
      uint32_t a, b;
      std::memcpy(&a, output + i, 4);
      std::memcpy(&b, expected + i, 4);
      bad += a != b;
    }
    result.mismatches += bad;
    result.bad_iterations += bad != 0;
  }
  return result;
}

static double average_ms(const Result &result) {
  size_t skip = result.times.size() > 5 ? 5 : 0;
  return std::accumulate(result.times.begin() + skip, result.times.end(), 0.0) /
         (result.times.size() - skip);
}

static void print_result(const char *name, const Result &result, int M, size_t weight_bytes) {
  double avg = average_ms(result);
  double min = *std::min_element(result.times.begin() + 5, result.times.end());
  double max = *std::max_element(result.times.begin() + 5, result.times.end());
  std::printf("%s verify=%s M=%d iterations=%zu bad_iterations=%zu exact_mismatches=%zu npu_calls=%zu\n",
              name, result.bad_iterations ? "FAIL" : "PASS", M, result.times.size(),
              result.bad_iterations, result.mismatches, result.calls);
  std::printf("%s expert_ms min=%.6f avg=%.6f max=%.6f per_row_ms=%.6f weight_bytes=%zu weight_GBps=%.3f\n",
              name, min, avg, max, avg / M, weight_bytes, weight_bytes / (avg * 1e6));
}

int main(int argc, char **argv) {
  if (argc != 5) {
    std::printf("usage: %s <asset-and-data-root> <M 1..8> <iters >=6> <cpu|m1|m8|both>\n", argv[0]);
    return 2;
  }
  try {
    SetThreadAffinityMask(GetCurrentThread(), 1ull << 0);
    std::string root = argv[1];
    int M = std::atoi(argv[2]), iters = std::atoi(argv[3]);
    std::string mode = argv[4];
    if (M < 1 || M > 8 || iters < 6 ||
        (mode != "cpu" && mode != "m1" && mode != "m8" && mode != "both")) return 2;
    auto gate = load_matrix(root + "/data_k2048n1024_expert", 2048, 1024);
    auto down = load_matrix(root + "/data_k512n2048_expert", 512, 2048);
    std::mt19937 rng(20261105);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<float> input((size_t)M * 2048);
    for (float &v : input) v = dist(rng);
    std::vector<float> cpu_expected((size_t)M * 2048), npu_expected((size_t)M * 2048);
    reference_expert(gate, down, input.data(), M, false, cpu_expected.data());
    reference_expert(gate, down, input.data(), M, true, npu_expected.data());
    size_t cpu_weight = gate.rowmajor.size() + down.rowmajor.size();
    size_t npu_weight = gate.tiled.size() + down.tiled.size();
    std::vector<float> output((size_t)M * 2048);
    Result cpu_result, m1_result, m8_result;
    if (mode == "cpu" || mode == "both") {
      CpuExpert cpu(gate, down);
      cpu_result = measure([&] { cpu.run(input.data(), M, output.data()); }, output.data(),
                           cpu_expected.data(), output.size(), iters);
      print_result("CPU", cpu_result, M, cpu_weight);
      if (cpu_result.bad_iterations) return 1;
    }
    if (mode != "cpu") {
      auto setup_begin = std::chrono::steady_clock::now();
      NpuExpert npu(root, gate, down);
      double setup_ms = elapsed_ms(setup_begin);
      if (mode == "m1" || mode == "both") {
        m1_result = measure([&] {
          size_t calls = 0;
          npu.run(input.data(), M, false, output.data(), calls);
        }, output.data(), npu_expected.data(), output.size(), iters);
        m1_result.calls = (size_t)2 * M * iters;
        print_result("NPU_M1xM", m1_result, M, npu_weight * M);
        if (m1_result.bad_iterations) return 1;
      }
      if (mode == "m8" || mode == "both") {
        m8_result = measure([&] {
          size_t calls = 0;
          npu.run(input.data(), M, true, output.data(), calls);
        }, output.data(), npu_expected.data(), output.size(), iters);
        size_t batches = M / 8 + M % 8;
        m8_result.calls = 2 * batches * iters;
        print_result("NPU_M8", m8_result, M, npu_weight * batches);
        if (m8_result.bad_iterations) return 1;
      }
      std::printf("NPU_SETUP context_and_weight_BO_ms=%.6f\n", setup_ms);
    }
    if (mode == "both") {
      double cpu = average_ms(cpu_result), m1 = average_ms(m1_result), m8 = average_ms(m8_result);
      double num = 0, den = 0;
      for (size_t i = 0; i < output.size(); ++i) {
        double d = npu_expected[i] - cpu_expected[i];
        num += d * d;
        den += (double)cpu_expected[i] * cpu_expected[i];
      }
      std::printf("COMPARISON M=%d M8_vs_CPU=%.3f%% M1xM_vs_CPU=%.3f%% M8_vs_M1xM=%.3f%% npu_vs_cpu_rel_l2=%.6g\n",
                  M, (m8 / cpu - 1) * 100, (m1 / cpu - 1) * 100, (m8 / m1 - 1) * 100,
                  std::sqrt(num / (den + 1e-30)));
    }
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "ERROR: %s\n", e.what());
    return 2;
  }
}
