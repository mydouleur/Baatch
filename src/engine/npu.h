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
  struct Weight {
    xrt::bo bo;
    size_t bytes;
    uint64_t last;
    bool pinned;
  };
  struct Shape {
    std::shared_ptr<xrt::hw_context> hw;
    xrt::kernel kernel;
    std::unique_ptr<xrt::run> run;
    xrt::bo bi, ba, bc;
    unsigned instr_words;
    int K, N, M;
    long runs = 0, switches = 0;
    double submit_ms = 0, wait_ms = 0;
    uint64_t weight_bytes = 0;
    std::map<const void*, Weight> bcache;
    std::map<std::vector<const void*>, Weight> fused_cache;
  };
  xrt::device dev{0};
  std::map<uint64_t, std::shared_ptr<Shape>> shapes;   // key = M<<56|K<<32|N
  bool ok = false;
  bool xbf4_experts = false;
  long runs = 0, flakes = 0;
  std::string dir;
  std::vector<int16_t> aq, aq2;
  size_t cached_bytes = 0, pinned_bytes = 0;
  size_t bo_budget = 3ull * 1024 * 1024 * 1024;
  uint64_t cache_clock = 0;
  long bo_hits = 0, bo_misses = 0, shape_switches = 0;
  double pack_ms = 0, submit_ms = 0, wait_ms = 0, sync_ms = 0;
  uint64_t physical_weight_bytes = 0;
  Shape* last_shape = nullptr;

  void init(const std::string& d, bool experts = false) {
    dir = d; xbf4_experts = experts; ok = true;
    if (const char* mb = getenv("NPU_BO_MB")) bo_budget = strtoull(mb, nullptr, 10) * 1024 * 1024;
  }
  void reset_stats() {
    runs = flakes = bo_hits = bo_misses = shape_switches = 0;
    pack_ms = submit_ms = wait_ms = sync_ms = 0;
    physical_weight_bytes = 0; last_shape = nullptr;
    for (auto& [key, shape] : shapes) {
      shape->runs = shape->switches = 0;
      shape->submit_ms = shape->wait_ms = 0;
      shape->weight_bytes = 0;
    }
  }
  void print_stats() const {
    printf("[npu-stats] runs=%ld flakes=%ld switches=%ld bo_hit=%ld bo_miss=%ld "
           "pack_ms=%.3f submit_ms=%.3f wait_ms=%.3f sync_ms=%.3f weight_MB=%.3f "
           "expert_cache_MB=%.3f pinned_MB=%.3f budget_MB=%.3f\n",
           runs, flakes, shape_switches, bo_hits, bo_misses, pack_ms, submit_ms, wait_ms,
           sync_ms, physical_weight_bytes / 1e6, cached_bytes / 1e6, pinned_bytes / 1e6,
           bo_budget / 1e6);
    for (const auto& [key, shape] : shapes)
      printf("[npu-shape] k=%d n=%d m=%d runs=%ld switches=%ld submit_ms=%.3f wait_ms=%.3f weight_MB=%.3f\n",
             shape->K, shape->N, shape->M, shape->runs, shape->switches,
             shape->submit_ms, shape->wait_ms, shape->weight_bytes / 1e6);
  }
  template <typename Direction>
  void sync_bo(xrt::bo& b, Direction direction) {
    auto t0 = std::chrono::steady_clock::now();
    b.sync(direction);
    sync_ms += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e3;
  }

  Shape& shape_for(int K, int N, int M = 1) {
    uint64_t key = (uint64_t)M << 56 | (uint64_t)K << 32 | (unsigned)N;
    auto it = shapes.find(key);
    if (it != shapes.end()) return *it->second;
    auto s = std::make_shared<Shape>();
    std::string base = dir + "/k" + std::to_string(K) + "n" + std::to_string(N);
    if (M == 2) base += "m2fix";
    std::ifstream f(base + ".insts.bin", std::ios::binary);
    if (!f) throw std::runtime_error("npu shape missing: " + base);
    f.seekg(0, std::ios::end); size_t n = f.tellg(); f.seekg(0);
    std::vector<char> inst(n); f.read(inst.data(), n);
    auto xcl = xrt::xclbin(base + ".xclbin");
    auto uuid = dev.register_xclbin(xcl);
    s->hw = std::make_shared<xrt::hw_context>(dev, uuid);
    s->kernel = xrt::kernel(*s->hw, "MLIR_AIE");
    s->bi = xrt::bo(dev, inst.size(), XCL_BO_FLAGS_CACHEABLE, s->kernel.group_id(1));
    memcpy(s->bi.map(), inst.data(), inst.size());
    sync_bo(s->bi, XCL_BO_SYNC_BO_TO_DEVICE);
    s->ba = xrt::bo(dev, M * K * 2, XRT_BO_FLAGS_HOST_ONLY, s->kernel.group_id(3));
    s->bc = xrt::bo(dev, M * N * 2, XRT_BO_FLAGS_HOST_ONLY, s->kernel.group_id(5));
    s->instr_words = inst.size() / 4; s->K = K; s->N = N; s->M = M;
    shapes[key] = s;
    fprintf(stderr, "[npu] shape k%d n%d m%d ready\n", K, N, M);
    return *s;
  }

  static void pack_expert_range(const BaaTensor& t, uint8_t* out, int physical_N,
                                int physical_K, int cm_begin, int cm_end) {
    int N = (int)t.shape[0], K = (int)t.shape[1];
    size_t sb_per_row = K / 256;
    size_t idx = (size_t)cm_begin * (physical_N / 512) * (physical_K / 128);
    for (int cm = cm_begin; cm < cm_end; cm++)
      for (int lt = cm; lt < physical_N / 64; lt += 8)
        for (int ltr = 0; ltr < physical_K / 128; ltr++, idx++) {
          uint8_t* tp = out + idx * 4352;
          memset(tp + 4096, 0, 128);
          memset(tp + 4224, 8, 128);
          if (ltr * 128 >= K) continue;
          for (int str = 0; str < 16; str++)
            for (int stc = 0; stc < 8; stc++) {
              uint64_t packed[4] = {};
              for (int j = 0; j < 8; j++) {
                int n = lt * 64 + stc * 8 + j;
                uint32_t w = 0x88888888;
                if (n < N) {
                  const uint8_t* src = t.data + ((size_t)n * sb_per_row + ltr / 2) * 130;
                  if (str == 0) memcpy(tp + 4096 + (stc * 8 + j) * 2, src, 2);
                  memcpy(&w, src + 2 + (ltr % 2) * 64 + str * 4, 4);
                }
                for (int b = 0; b < 4; b++) {
                  uint64_t v = (w >> (b * 8)) & 0xFF;
                  packed[b] |= (v & 0xF) << (j * 4);
                  packed[b] |= (v >> 4) << (j * 4 + 32);
                }
              }
              memcpy(tp + (str * 8 + stc) * 32, packed, sizeof(packed));
            }
        }
  }

  static void pack_expert(const BaaTensor& t, uint8_t* out, int physical_N, int physical_K = 0) {
    int K = (int)t.shape[1];
    if (!physical_K) physical_K = K;
    size_t bytes = (size_t)(physical_K / 128) * (physical_N / 64) * 4352;
    memset(out, 0x88, bytes);
    std::thread worker([&] { pack_expert_range(t, out, physical_N, physical_K, 4, 8); });
    pack_expert_range(t, out, physical_N, physical_K, 0, 4);
    worker.join();
  }

  xrt::bo& bo_for(Shape& s, const BaaTensor& t) {
    auto it = s.bcache.find(t.data);
    if (it != s.bcache.end()) {
      bo_hits++;
      it->second.last = ++cache_clock;
      return it->second.bo;
    }
    bo_misses++;
    bool pinned = (t.flags & 1) != 0;
    size_t bytes = pinned ? t.size : (size_t)(s.K / 128) * (s.N / 64) * 4352;
    if (pinned && pinned_bytes + cached_bytes + bytes > bo_budget)
      throw std::runtime_error("npu BO budget exceeded by pinned weights");
    while (!pinned && (cached_bytes + bytes > 512ull * 1024 * 1024 ||
                       pinned_bytes + cached_bytes + bytes > bo_budget)) {
      Shape* oldest_shape = nullptr;
      const void* oldest_key = nullptr;
      uint64_t oldest = ~0ull;
      for (auto& [key, shape] : shapes)
        for (auto& [ptr, w] : shape->bcache)
          if (!w.pinned && w.last < oldest) {
            oldest = w.last; oldest_shape = shape.get(); oldest_key = ptr;
          }
      if (!oldest_shape) throw std::runtime_error("npu expert cache entry too large");
      cached_bytes -= oldest_shape->bcache.at(oldest_key).bytes;
      oldest_shape->bcache.erase(oldest_key);
    }
    xrt::bo b(dev, bytes, XRT_BO_FLAGS_HOST_ONLY, s.kernel.group_id(4));
    auto t0 = std::chrono::steady_clock::now();
    if (pinned) memcpy(b.map(), t.data, t.size);
    else pack_expert(t, static_cast<uint8_t*>(b.map()), s.N, s.K);
    pack_ms += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e3;
    sync_bo(b, XCL_BO_SYNC_BO_TO_DEVICE);
    if (pinned) pinned_bytes += bytes;
    else cached_bytes += bytes;
    Weight w{std::move(b), bytes, ++cache_clock, pinned};
    auto r = s.bcache.emplace(t.data, std::move(w));
    return r.first->second.bo;
  }

  xrt::bo& bo_fused(Shape& s, const BaaTensor* const* tensors, int tensor_count,
                    const std::vector<int>& physical_ns) {
    std::vector<const void*> key;
    size_t bytes = 0;
    int K = (int)tensors[0]->shape[1];
    for (int i = 0; i < tensor_count; i++) key.push_back(tensors[i]->data);
    for (int n : physical_ns) bytes += (size_t)(K / 128) * (n / 64) * 4352;
    auto it = s.fused_cache.find(key);
    if (it != s.fused_cache.end()) {
      bo_hits++;
      it->second.last = ++cache_clock;
      return it->second.bo;
    }
    bo_misses++;
    if (pinned_bytes + cached_bytes + bytes > bo_budget)
      throw std::runtime_error("npu BO budget exceeded by fused weights");
    xrt::bo b(dev, bytes, XRT_BO_FLAGS_HOST_ONLY, s.kernel.group_id(4));
    auto t0 = std::chrono::steady_clock::now();
    uint8_t* out = static_cast<uint8_t*>(b.map());
    size_t dst = 0;
    for (int cm = 0; cm < 8; cm++) {
      for (int i = 0; i < (int)physical_ns.size(); i++) {
        int N = i < tensor_count ? (int)tensors[i]->shape[0] : 0;
        int src_n = i < tensor_count ? (N + 2047) & ~2047 : 0;
        for (int lt = cm; lt < physical_ns[i] / 64; lt += 8) {
          for (int ltr = 0; ltr < K / 128; ltr++) {
            if (lt * 64 < N) {
              size_t src = ((size_t)cm * (src_n / 512) + lt / 8) * (K / 128) + ltr;
              memcpy(out + dst, tensors[i]->data + src * 4352, 4352);
            } else {
              memset(out + dst, 0, 4352);
            }
            dst += 4352;
          }
        }
      }
    }
    if (dst != bytes) throw std::runtime_error("npu fused pack size mismatch");
    pack_ms += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e3;
    sync_bo(b, XCL_BO_SYNC_BO_TO_DEVICE);
    pinned_bytes += bytes;
    Weight w{std::move(b), bytes, ++cache_clock, true};
    auto inserted = s.fused_cache.emplace(std::move(key), std::move(w));
    return inserted.first->second.bo;
  }

  struct Job {
    Shape* shape;
    xrt::bo* weights;
    float* y;
    float sx;
    int N, physical_N;
    xrt::run* run;
    float* y2 = nullptr;
    float sx2 = 0.f;
    int M = 1;
  };
  bool busy = false;

  void launch(Job& j) {
    Shape& s = *j.shape;
    memset(s.bc.map(), 0, j.M * j.physical_N * 2);
    sync_bo(s.bc, XCL_BO_SYNC_BO_TO_DEVICE);
    if (!s.run) {
      s.run = std::make_unique<xrt::run>(s.kernel);
      s.run->set_arg(0, 3);
      s.run->set_arg(1, s.bi); s.run->set_arg(2, s.instr_words);
      s.run->set_arg(3, s.ba); s.run->set_arg(5, s.bc);
    }
    s.run->set_arg(4, *j.weights);
    j.run = s.run.get();
    j.run->start();
  }

  // 同一时刻只允许一个未完成任务，避免多个 context 并发提交。
  bool submit(const BaaTensor& t, const float* x, float* y, Job& j) {
    if (busy) throw std::runtime_error("npu submit while busy");
    auto t0 = std::chrono::steady_clock::now();
    int K = (int)t.shape[1];
    aq.resize(K);
    float sx = quant_xq16(x, K, aq.data());
    submit_quantized(t, K, aq.data(), sx, y, j);
    submit_ms += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e3;
    return true;
  }

  bool submit_quantized(const BaaTensor& t, int K, const int16_t* q, float sx, float* y, Job& j) {
    auto t0 = std::chrono::steady_clock::now();
    if (busy) throw std::runtime_error("npu submit while busy");
    int N = (int)t.shape[0];
    int physical_K = (t.flags & 2) ? 4096 : (t.flags & 4) || (!(t.flags & 1) && K == 512) ? 2048 : K;
    int physical_N = (t.flags & 2) || (t.flags & 8) ? 8192 : (t.flags & 4) ? 2048 : (N + 2047) & ~2047;
    Shape& s = shape_for(physical_K, physical_N);
    if (last_shape && last_shape != &s) { shape_switches++; s.switches++; }
    last_shape = &s;
    size_t weight_bytes = (t.flags & 1) ? t.size : (size_t)(s.K / 128) * (s.N / 64) * 4352;
    physical_weight_bytes += weight_bytes;
    s.weight_bytes += weight_bytes;
    xrt::bo& bb = bo_for(s, t);
    memset(s.ba.map(), 0, physical_K * 2);
    memcpy(s.ba.map(), q, K * 2);
    sync_bo(s.ba, XCL_BO_SYNC_BO_TO_DEVICE);
    j.shape = &s; j.weights = &bb; j.y = y; j.sx = sx;
    j.N = N; j.physical_N = physical_N;
    busy = true;
    launch(j);
    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e3;
    submit_ms += elapsed;
    s.submit_ms += elapsed;
    return true;
  }

  bool gemv_quantized(const BaaTensor& t, int K, const int16_t* q, float sx, float* y) {
    Job j;
    submit_quantized(t, K, q, sx, y, j);
    return wait(j);
  }

  bool gemv_fused(const BaaTensor* const* tensors, int count, const float* x, float* const* ys,
                  int min_physical_N = 0) {
    if (busy || count < 2) return false;
    auto t0 = std::chrono::steady_clock::now();
    int K = (int)tensors[0]->shape[1], physical_N = 0;
    size_t weight_bytes = 0;
    std::vector<int> physical_ns(count);
    for (int i = 0; i < count; i++) {
      int N = (int)tensors[i]->shape[0];
      if (tensors[i]->flags != 1 || tensors[i]->shape[1] != K) return false;
      physical_ns[i] = N <= 1024 ? 1024 : (N + 2047) & ~2047;
      physical_N += physical_ns[i];
      weight_bytes += (size_t)(K / 128) * (physical_ns[i] / 64) * 4352;
    }
    min_physical_N = (min_physical_N + 2047) & ~2047;
    if (physical_N < min_physical_N) {
      int padding = min_physical_N - physical_N;
      physical_ns.push_back(padding);
      physical_N += padding;
      weight_bytes += (size_t)(K / 128) * (padding / 64) * 4352;
    }
    Shape& s = shape_for(K, physical_N);
    if (last_shape && last_shape != &s) { shape_switches++; s.switches++; }
    last_shape = &s;
    physical_weight_bytes += weight_bytes;
    s.weight_bytes += weight_bytes;
    xrt::bo& bb = bo_fused(s, tensors, count, physical_ns);
    aq.resize(K);
    float sx = quant_xq16(x, K, aq.data());
    memcpy(s.ba.map(), aq.data(), K * 2);
    sync_bo(s.ba, XCL_BO_SYNC_BO_TO_DEVICE);
    std::vector<float> y(physical_N);
    Job j{&s, &bb, y.data(), sx, physical_N, physical_N, nullptr};
    busy = true;
    launch(j);
    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e3;
    submit_ms += elapsed;
    s.submit_ms += elapsed;
    if (!wait(j)) return false;
    int offset = 0;
    for (int i = 0; i < count; i++) {
      memcpy(ys[i], y.data() + offset, tensors[i]->shape[0] * sizeof(float));
      offset += physical_ns[i];
    }
    return true;
  }

  bool gemv_fused_pair(const BaaTensor* const* tensors, int count,
                       const float* x0, const float* x1,
                       float* const* ys0, float* const* ys1, int min_physical_N = 0) {
    if (!ok || busy || count < 2) return false;
    auto t0 = std::chrono::steady_clock::now();
    int K = (int)tensors[0]->shape[1], physical_N = 0;
    size_t weight_bytes = 0;
    std::vector<int> physical_ns(count);
    for (int i = 0; i < count; i++) {
      int N = (int)tensors[i]->shape[0];
      if (tensors[i]->flags != 1 || tensors[i]->shape[1] != K) return false;
      physical_ns[i] = N <= 1024 ? 1024 : (N + 2047) & ~2047;
      physical_N += physical_ns[i];
      weight_bytes += (size_t)(K / 128) * (physical_ns[i] / 64) * 4352;
    }
    min_physical_N = (min_physical_N + 2047) & ~2047;
    if (physical_N < min_physical_N) {
      int padding = min_physical_N - physical_N;
      physical_ns.push_back(padding);
      physical_N += padding;
      weight_bytes += (size_t)(K / 128) * (padding / 64) * 4352;
    }
    if (K != 2048 || (physical_N != 10240 && physical_N != 12288)) return false;
    Shape& s = shape_for(K, physical_N, 2);
    if (last_shape && last_shape != &s) { shape_switches++; s.switches++; }
    last_shape = &s;
    physical_weight_bytes += weight_bytes;
    s.weight_bytes += weight_bytes;
    xrt::bo& bb = bo_fused(s, tensors, count, physical_ns);
    aq.resize(K); aq2.resize(K);
    float sx0 = quant_xq16(x0, K, aq.data());
    float sx1 = quant_xq16(x1, K, aq2.data());
    int16_t* a = static_cast<int16_t*>(s.ba.map());
    for (int tile = 0; tile < K / 128; tile++) {
      memcpy(a + (size_t)(tile * 2) * 128, aq.data() + tile * 128, 128 * 2);
      memcpy(a + (size_t)(tile * 2 + 1) * 128, aq2.data() + tile * 128, 128 * 2);
    }
    sync_bo(s.ba, XCL_BO_SYNC_BO_TO_DEVICE);
    std::vector<float> y0(physical_N), y1(physical_N);
    Job j{&s, &bb, y0.data(), sx0, physical_N, physical_N, nullptr};
    j.y2 = y1.data(); j.sx2 = sx1; j.M = 2;
    busy = true;
    launch(j);
    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e3;
    submit_ms += elapsed;
    s.submit_ms += elapsed;
    if (!wait(j)) return false;
    int offset = 0;
    for (int i = 0; i < count; i++) {
      size_t bytes = tensors[i]->shape[0] * sizeof(float);
      memcpy(ys0[i], y0.data() + offset, bytes);
      memcpy(ys1[i], y1.data() + offset, bytes);
      offset += physical_ns[i];
    }
    return true;
  }

  bool wait(Job& j) {
    auto t0 = std::chrono::steady_clock::now();
    Shape& s = *j.shape;
    for (int att = 0; att < 4; att++) {
      try { j.run->wait2(); }
      catch (const std::exception& e) {
        fprintf(stderr, "[npu] run 超时/挂起：%s — 不重试（看门狗语义）\n", e.what());
        runs++; s.runs++;
        ok = false; busy = false;
        double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e3;
        wait_ms += elapsed;
        s.wait_ms += elapsed;
        return false;
      }
      runs++; s.runs++;
      sync_bo(s.bc, XCL_BO_SYNC_BO_FROM_DEVICE);
      const uint16_t* c = (const uint16_t*)s.bc.map();
      bool allzero = true;
      for (int i = 0; i < j.M * j.N; i++) if (c[i] != 0) { allzero = false; break; }
      if (!allzero) {
        if (j.M == 1) {
          for (int i = 0; i < j.N; i++) j.y[i] = bf16_to_f32(c[i]) * j.sx;
        } else {
          float* ys[2] = {j.y, j.y2};
          float scales[2] = {j.sx, j.sx2};
          for (int tr = 0; tr < j.physical_N / 2048; tr++)
            for (int col = 0; col < 8; col++)
              for (int row = 0; row < 4; row++)
                for (int m = 0; m < 2; m++)
                  for (int i = 0; i < 64; i++) {
                    int src = (((tr * 8 + col) * 4 + row) * 2 + m) * 64 + i;
                    int dst = tr * 2048 + row * 512 + col * 64 + i;
                    ys[m][dst] = bf16_to_f32(c[src]) * scales[m];
                  }
        }
        busy = false;
        double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e3;
        wait_ms += elapsed;
        s.wait_ms += elapsed;
        return true;
      }
      flakes++;
      if (att != 3) launch(j);
    }
    fprintf(stderr, "[npu] flake ×4（记数继续）\n");
    busy = false;
    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e3;
    wait_ms += elapsed;
    s.wait_ms += elapsed;
    return false;
  }

  // 预平铺物理 N 补齐到 2048；目录和 y 仍使用逻辑 N。
  bool gemv(const BaaTensor& t, const float* x, float* y) {
    Job j;
    submit(t, x, y, j);
    return wait(j);
  }
  ~NpuBackend() {
    if (runs) fprintf(stderr, "[npu] runs=%ld flakes=%ld (%.2f%%)\n", runs, flakes, 100.0 * flakes / runs);
  }
};
