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

// ---------------- 极简 2 线程池（绑大核 LP0/LP2） ----------------
struct Pool {
  std::thread th;
  std::atomic<uint64_t> go{0}, done{0};
  std::function<void(int)> fn;  // fn(worker_id)
  bool stop = false;
  Pool() {
    th = std::thread([this] {
      SetThreadAffinityMask(GetCurrentThread(), 1ull << 2);  // LP2（主线程在 LP0）
      uint64_t seen = 0;
      while (!stop) {
        uint64_t g = go.load(std::memory_order_acquire);
        if (g != seen) { seen = g; fn(1); done.store(seen, std::memory_order_release); }
        else _mm_pause();
      }
    });
  }
  ~Pool() { stop = true; th.join(); }
  // 主线程跑 id=0，worker 跑 id=1
  void run(std::function<void(int)> f) {
    fn = std::move(f);
    go.fetch_add(1, std::memory_order_release);
    fn(0);
    while (done.load(std::memory_order_acquire) != go.load()) _mm_pause();
  }
};

// ---------------- bf16 GEMV：y[r] = Σ_c W[r,c]·x[c]（W 为 bf16 行主序 [R,C]） ----------------
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

// ---------------- XBF3（vnni2）GEMV ----------------
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

// w: [rows][C/256][98B]；xq: int8 激活（自然序）；sumxq: 每 sb 的 Σxq；sx: 激活 scale
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

// 激活量化：xq = round(x/sx)，sx = max|x|/127；sumxq 每 256-sb
static inline float quant_xq(const float* x, int n, int8_t* xq, float* sumxq) {
  float amax = 0;
  for (int i = 0; i < n; i++) amax = std::max(amax, fabsf(x[i]));
  float sx = std::max(amax / 127.f, 1e-12f);
  __m512 vsx = _mm512_set1_ps(sx);
  for (int sb = 0; sb < n / XBF_SB; sb++) {
    for (int i = 0; i < 256; i += 16) {
      __m512 v = _mm512_div_ps(_mm512_loadu_ps(x + sb * 256 + i), vsx);
      __m128i q8 = _mm512_cvtsepi32_epi8(_mm512_cvtps_epi32(v));  // cvtps 已含最近舍入
      _mm_storeu_si128((__m128i*)(xq + sb * 256 + i), q8);
    }
    int sum = 0;
    for (int i = 0; i < 256; i++) sum += xq[sb * 256 + i];
    sumxq[sb] = (float)sum;
  }
  return sx;
}

// ---------------- f32 GEMV（路由器/小权重不压） ----------------
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

// ---------------- 小 ops（fp32） ----------------
// qwen3_5_moe 两种 RMSNorm：plus1=true → ×(1+w)（input/post/qk/final norm，零中心权重）；
// plus1=false → ×w（GDN 输出门控 norm，ones 初始化权重）
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
