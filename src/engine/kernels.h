// kernels.h — Baatch 引擎 CPU kernels（AVX-512, Zen5）
// gemv_xbf3 = M3'-1 vnni2 实测版（vpermb+vpmultishiftqb+vpdpbusd，2 大核 39 GB/s）原样搬入。
#pragma once
#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <atomic>
#include <functional>
#include <thread>
#include <vector>
#include <algorithm>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#define XBF_SB 256
#define XBF_ROW_BYTES(sb_per_row) ((sb_per_row) * 98)

static inline float bf16_to_f32(uint16_t b) {
  uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f;
}
static inline uint16_t f32_to_bf16(float f) {
  uint32_t u; memcpy(&u, &f, 4);
  return (uint16_t)((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}

// ---------------- 极简线程池（普通 GEMV 仍只用 LP0/LP2） ----------------
struct Pool {
  std::vector<std::thread> threads;
  std::atomic<uint64_t> go{0};
  std::atomic<int> active{2}, finished{0};
  std::function<void(int)> fn;
  std::atomic<bool> stop{false};
  int workers;
  Pool() {
    const char* env = getenv("CPU_WIDE_THREADS");
    workers = env ? atoi(env) : 2;
    workers = std::max(1, std::min(8, workers));
    static const DWORD_PTR affinity[] = {1ull << 0, 1ull << 2, 1ull << 4, 1ull << 6,
                                         1ull << 8, 1ull << 10, 1ull << 12, 1ull << 14};
    for (int id = 1; id < std::max(workers, 2); id++) threads.emplace_back([this, id] {
      SetThreadAffinityMask(GetCurrentThread(), affinity[id]);
      uint64_t seen = 0;
      while (!stop.load(std::memory_order_relaxed)) {
        uint64_t g = go.load(std::memory_order_acquire);
        if (g != seen) {
          seen = g;
          if (id < active.load(std::memory_order_relaxed)) {
            fn(id);
            finished.fetch_add(1, std::memory_order_release);
          }
        } else _mm_pause();
      }
    });
  }
  ~Pool() {
    stop.store(true, std::memory_order_relaxed);
    for (auto& thread : threads) thread.join();
  }
  void dispatch(std::function<void(int)> f, int count) {
    fn = std::move(f);
    active.store(count, std::memory_order_relaxed);
    finished.store(0, std::memory_order_relaxed);
    go.fetch_add(1, std::memory_order_release);
    fn(0);
    while (finished.load(std::memory_order_acquire) != count - 1) _mm_pause();
  }
  void run(std::function<void(int)> f) { dispatch(std::move(f), 2); }
  void run_all(std::function<void(int)> f) { dispatch(std::move(f), workers); }
};

static inline void gemv_bf16_rows(const uint16_t* W, int r0, int r1, int C,
                                  const float* x, float* y) {
  for (int r = r0; r < r1; r++) {
    const uint16_t* wr = W + (size_t)r * C;
    __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
    for (int c = 0; c + 32 <= C; c += 32) {
      __m512i lo = _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)(wr + c))), 16);
      __m512i hi = _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)(wr + c + 16))), 16);
      acc0 = _mm512_fmadd_ps(_mm512_castsi512_ps(lo), _mm512_loadu_ps(x + c), acc0);
      acc1 = _mm512_fmadd_ps(_mm512_castsi512_ps(hi), _mm512_loadu_ps(x + c + 16), acc1);
    }
    y[r] = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
  }
}
static inline void gemv_bf16(Pool& pool, const uint16_t* W, int R, int C,
                             const float* x, float* y) {
  int half = (R / 2 + 15) & ~15;
  pool.run([&](int id) {
    if (id == 0) gemv_bf16_rows(W, 0, half, C, x, y);
    else gemv_bf16_rows(W, half, R, C, x, y);
  });
}

static const __m512i VN2_IDX = [] {
  alignas(64) uint8_t idx[64];
  for (int j = 0; j < 8; j++)
    for (int t = 0; t < 8; t++)
      idx[8*j + t] = (t < 3) ? (uint8_t)(3*j + t) : 0;
  return _mm512_load_si512(idx);
}();
static const __m512i VN2_CTRL = [] {
  alignas(64) uint8_t c[64];
  for (int j = 0; j < 8; j++)
    for (int t = 0; t < 8; t++) c[8*j + t] = 3 * t;
  return _mm512_load_si512(c);
}();
static const __m512i VN2_SEVEN = [] { return _mm512_set1_epi8(7); }();

static void gemv_xbf3_rows(const uint8_t* w, int row0, int nrows, int n_in,
                           const int8_t* xq, const float* sumxq_sb, float sx, float* y) {
  const int sb_per_row = n_in / XBF_SB;
  const size_t row_bytes = (size_t)sb_per_row * 98;
  for (int r = 0; r < nrows; r++) {
    const uint8_t* rowp = w + (size_t)(row0 + r) * row_bytes;
    float yr = 0.f;
    for (int sb = 0; sb < sb_per_row; sb++) {
      const uint8_t* wsb = rowp + sb * 98;
      float scale = bf16_to_f32(wsb[0] | (wsb[1] << 8));
      __m512i acc0 = _mm512_setzero_si512(), acc1 = _mm512_setzero_si512();
      __m512i acc2 = _mm512_setzero_si512(), acc3 = _mm512_setzero_si512();
      for (int wnd = 0; wnd < 4; wnd++) {
        __m512i z = _mm512_maskz_loadu_epi8(0x0000FFFFFFFFFFFFULL, wsb + 2 + 24 * wnd);
        __m512i g = _mm512_permutexvar_epi8(VN2_IDX, z);
        __m512i f = _mm512_and_si512(_mm512_multishift_epi64_epi8(VN2_CTRL, g), VN2_SEVEN);
        __m512i xv = _mm512_loadu_si512(xq + sb * 256 + 64 * wnd);
        __m512i& a = (wnd == 0) ? acc0 : (wnd == 1) ? acc1 : (wnd == 2) ? acc2 : acc3;
        a = _mm512_dpbusd_epi32(a, f, xv);
      }
      int isum = _mm512_reduce_add_epi32(_mm512_add_epi32(
          _mm512_add_epi32(acc0, acc1), _mm512_add_epi32(acc2, acc3)));
      yr += (float)(isum - 4 * (int)sumxq_sb[sb]) * sx * scale;
    }
    y[row0 + r] = yr;
  }
}
static inline void gemv_xbf3(Pool& pool, const uint8_t* W, int R, int C,
                             const int8_t* xq, const float* sumxq, float sx, float* y) {
  int half = (R / 2 + 15) & ~15;
  pool.run([&](int id) {
    if (id == 0) gemv_xbf3_rows(W, 0, half, C, xq, sumxq, sx, y);
    else gemv_xbf3_rows(W, half, R - half, C, xq, sumxq, sx, y);
  });
}

static inline float quant_xq(const float* x, int n, int8_t* xq, float* sumxq) {
  const __m512i abs_mask = _mm512_set1_epi32(0x7fffffff);
  __m512 vmax = _mm512_setzero_ps();
  for (int i = 0; i < n; i += 16) {
    __m512 v = _mm512_loadu_ps(x + i);
    vmax = _mm512_max_ps(vmax, _mm512_castsi512_ps(_mm512_and_si512(abs_mask, _mm512_castps_si512(v))));
  }
  float amax = _mm512_reduce_max_ps(vmax);
  float sx = std::max(amax / 127.f, 1e-12f);
  __m512 vsx = _mm512_set1_ps(sx);
  for (int sb = 0; sb < n / XBF_SB; sb++) {
    __m512i vsum = _mm512_setzero_si512();
    for (int i = 0; i < 256; i += 16) {
      __m512 v = _mm512_div_ps(_mm512_loadu_ps(x + sb * 256 + i), vsx);
      __m512i q = _mm512_cvtps_epi32(v);
      vsum = _mm512_add_epi32(vsum, q);
      _mm_storeu_si128((__m128i*)(xq + sb * 256 + i), _mm512_cvtsepi32_epi8(q));
    }
    sumxq[sb] = (float)_mm512_reduce_add_epi32(vsum);
  }
  return sx;
}

static inline float quant_xq16(const float* x, int n, int16_t* xq) {
  const __m512i abs_mask = _mm512_set1_epi32(0x7fffffff);
  __m512 vmax = _mm512_setzero_ps();
  for (int i = 0; i < n; i += 16) {
    __m512 v = _mm512_loadu_ps(x + i);
    vmax = _mm512_max_ps(vmax, _mm512_castsi512_ps(_mm512_and_si512(abs_mask, _mm512_castps_si512(v))));
  }
  float sx = std::max(_mm512_reduce_max_ps(vmax) / 127.f, 1e-12f);
  __m512 vsx = _mm512_set1_ps(sx);
  for (int i = 0; i < n; i += 16) {
    __m512 v = _mm512_div_ps(_mm512_loadu_ps(x + i), vsx);
    __m512i q = _mm512_cvtps_epi32(v);
    _mm256_storeu_si256((__m256i*)(xq + i), _mm512_cvtsepi32_epi16(q));
  }
  return sx;
}

static inline void gemv_f32_rows(const float* W, int r0, int r1, int C,
                                 const float* x, float* y) {
  for (int r = r0; r < r1; r++) {
    const float* wr = W + (size_t)r * C;
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
    for (int c = 0; c + 32 <= C; c += 32) {
      a0 = _mm512_fmadd_ps(_mm512_loadu_ps(wr + c), _mm512_loadu_ps(x + c), a0);
      a1 = _mm512_fmadd_ps(_mm512_loadu_ps(wr + c + 16), _mm512_loadu_ps(x + c + 16), a1);
    }
    y[r] = _mm512_reduce_add_ps(_mm512_add_ps(a0, a1));
  }
}
static inline void gemv_f32(Pool& pool, const float* W, int R, int C,
                            const float* x, float* y) {
  int half = (R / 2 + 15) & ~15;
  pool.run([&](int id) {
    if (id == 0) gemv_f32_rows(W, 0, half, C, x, y);
    else gemv_f32_rows(W, half, R, C, x, y);
  });
}

static inline void rmsnorm(const float* x, const uint16_t* w_bf, int n, float* y,
                           float eps = 1e-6f, bool plus1 = false) {
  float ss = 0;
  for (int i = 0; i < n; i++) ss += x[i] * x[i];
  float inv = 1.f / sqrtf(ss / n + eps);
  if (plus1)
    for (int i = 0; i < n; i++) y[i] = x[i] * inv * (1.f + bf16_to_f32(w_bf[i]));
  else
    for (int i = 0; i < n; i++) y[i] = x[i] * inv * bf16_to_f32(w_bf[i]);
}
static inline void rmsnorm_f32(const float* x, const float* w, int n, float* y,
                               float eps = 1e-6f, bool plus1 = false) {
  float ss = 0;
  for (int i = 0; i < n; i++) ss += x[i] * x[i];
  float inv = 1.f / sqrtf(ss / n + eps);
  if (plus1)
    for (int i = 0; i < n; i++) y[i] = x[i] * inv * (1.f + w[i]);
  else
    for (int i = 0; i < n; i++) y[i] = x[i] * inv * w[i];
}
static inline float sigmoidf(float x) { return 1.f / (1.f + expf(-x)); }
static inline float softplusf(float x) { return x > 20.f ? x : log1pf(expf(x)); }
static inline float siluf(float x) { return x / (1.f + expf(-x)); }

template <int HD>
static inline float attn_qk_prefill(const float* q, const float* k, int len, int stride,
                                    float* scores) {
  float mx = -1e30f;
  int p = 0;
  for (; p + 4 <= len; p += 4) {
    const float* k0 = k + (size_t)p * stride;
    const float* k1 = k0 + stride;
    const float* k2 = k1 + stride;
    const float* k3 = k2 + stride;
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (int i = 0; i < HD; i++) {
      s0 += q[i] * k0[i];
      s1 += q[i] * k1[i];
      s2 += q[i] * k2[i];
      s3 += q[i] * k3[i];
    }
    scores[p] = s0 / sqrtf((float)HD);
    scores[p + 1] = s1 / sqrtf((float)HD);
    scores[p + 2] = s2 / sqrtf((float)HD);
    scores[p + 3] = s3 / sqrtf((float)HD);
    mx = std::max(mx, scores[p]);
    mx = std::max(mx, scores[p + 1]);
    mx = std::max(mx, scores[p + 2]);
    mx = std::max(mx, scores[p + 3]);
  }
  for (; p < len; p++) {
    const float* kp = k + (size_t)p * stride;
    float s = 0;
    for (int i = 0; i < HD; i++) s += q[i] * kp[i];
    scores[p] = s / sqrtf((float)HD);
    mx = std::max(mx, scores[p]);
  }
  return mx;
}

template <int HD>
static inline float gdn_recur_row_prefill(float* sr, const float* k, const float* q,
                                          float v, float b, __m512 decay) {
  __m512 acc = _mm512_setzero_ps();
  for (int mm = 0; mm < HD; mm += 16) {
    __m512 sv = _mm512_mul_ps(_mm512_loadu_ps(sr + mm), decay);
    _mm512_storeu_ps(sr + mm, sv);
    acc = _mm512_fmadd_ps(sv, _mm512_loadu_ps(k + mm), acc);
  }
  const __m512 d = _mm512_set1_ps((v - _mm512_reduce_add_ps(acc)) * b);
  acc = _mm512_setzero_ps();
  for (int mm = 0; mm < HD; mm += 16) {
    __m512 updated = _mm512_fmadd_ps(_mm512_loadu_ps(k + mm), d, _mm512_loadu_ps(sr + mm));
    _mm512_storeu_ps(sr + mm, updated);
    acc = _mm512_fmadd_ps(updated, _mm512_loadu_ps(q + mm), acc);
  }
  return _mm512_reduce_add_ps(acc);
}

template <int HD>
static inline void attn_pv_prefill(const float* v, int len, int stride, const float* scores,
                                   float sum, float* out) {
  for (int p = 0; p < len; p++) {
    const float* vp = v + (size_t)p * stride;
    const __m512 w = _mm512_set1_ps(scores[p] / sum);
    for (int i = 0; i < HD; i += 16) {
      __m512 acc = _mm512_loadu_ps(out + i);
      __m512 delta = _mm512_mul_ps(w, _mm512_loadu_ps(vp + i));
      _mm512_storeu_ps(out + i, _mm512_add_ps(acc, delta));
    }
  }
}
