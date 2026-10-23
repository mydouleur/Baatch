// kernels_xbf.h — XBF4/XBF2 vnni GEMV（M3'-a 阶段4；vnni2 同族 vpsrlw+vpand+vpdpbusd）
// XBF4 sb = [2B bf16 scale][128B nibble]（值 2i lo、2i+1 hi）
// XBF2 sb = [2B bf16 scale][64B]（值 4i+t 在 bits [2t,2t+2)）
// 激活拆分在量化时一次完成（见 quant_xq_split）：XBF4 偶/奇两路，XBF2 模4 四路。
// 修正：y = (Σ field·xq − OFF·Σxq)·sx·s_sb，OFF = 8（XBF4）/ 2（XBF2）
#pragma once
#include "kernels.h"

static const __m512i XBF4_LO = _mm512_set1_epi8(0x0F);
static const __m512i XBF2_M = _mm512_set1_epi8(0x03);

// ---------------- XBF4 ----------------
// xe/xo：每 sb 128B（偶/奇值各 128），连续排布
static void gemv_xbf4_rows(const uint8_t* w, int row0, int nrows, int n_in,
                           const int8_t* xe, const int8_t* xo,
                           const float* sumxq_sb, float sx, float* y) {
  const int sb_per_row = n_in / XBF_SB;
  const size_t row_bytes = (size_t)sb_per_row * 130;
  for (int r = 0; r < nrows; r++) {
    const uint8_t* rowp = w + (size_t)(row0 + r) * row_bytes;
    float yr = 0.f;
    for (int sb = 0; sb < sb_per_row; sb++) {
      const uint8_t* wsb = rowp + sb * 130;
      float scale = bf16_to_f32(wsb[0] | (wsb[1] << 8));
      __m512i acc0 = _mm512_setzero_si512(), acc1 = _mm512_setzero_si512();
      __m512i acc2 = _mm512_setzero_si512(), acc3 = _mm512_setzero_si512();
      for (int h = 0; h < 2; h++) {   // 128B = 2×64B（每 64B = 128 值）
        __m512i z = _mm512_loadu_si512(wsb + 2 + 64 * h);
        __m512i lo = _mm512_and_si512(z, XBF4_LO);
        __m512i hi = _mm512_and_si512(_mm512_srli_epi16(z, 4), XBF4_LO);
        __m512i xv_e = _mm512_loadu_si512(xe + sb * 128 + 64 * h);
        __m512i xv_o = _mm512_loadu_si512(xo + sb * 128 + 64 * h);
        __m512i& a0 = h ? acc2 : acc0;
        __m512i& a1 = h ? acc3 : acc1;
        a0 = _mm512_dpbusd_epi32(a0, lo, xv_e);
        a1 = _mm512_dpbusd_epi32(a1, hi, xv_o);
      }
      int isum = _mm512_reduce_add_epi32(_mm512_add_epi32(
          _mm512_add_epi32(acc0, acc1), _mm512_add_epi32(acc2, acc3)));
      yr += (float)(isum - 8 * (int)sumxq_sb[sb]) * sx * scale;
    }
    y[row0 + r] = yr;
  }
}

// ---------------- XBF2 ----------------
// xq4[t]：每 sb 64B（值 ≡ t mod 4），连续排布
static void gemv_xbf2_rows(const uint8_t* w, int row0, int nrows, int n_in,
                           const int8_t* const* xq4,
                           const float* sumxq_sb, float sx, float* y) {
  const int sb_per_row = n_in / XBF_SB;
  const size_t row_bytes = (size_t)sb_per_row * 66;
  for (int r = 0; r < nrows; r++) {
    const uint8_t* rowp = w + (size_t)(row0 + r) * row_bytes;
    float yr = 0.f;
    for (int sb = 0; sb < sb_per_row; sb++) {
      const uint8_t* wsb = rowp + sb * 66;
      float scale = bf16_to_f32(wsb[0] | (wsb[1] << 8));
      __m512i z = _mm512_loadu_si512(wsb + 2);   // 64B = 256 值
      int isum = 0;
      for (int t = 0; t < 4; t++) {
        __m512i f = _mm512_and_si512(_mm512_srli_epi16(z, 2 * t), XBF2_M);
        __m512i xv = _mm512_loadu_si512(xq4[t] + sb * 64);
        __m512i a = _mm512_dpbusd_epi32(_mm512_setzero_si512(), f, xv);
        isum += _mm512_reduce_add_epi32(a);
      }
      yr += (float)(isum - 2 * (int)sumxq_sb[sb]) * sx * scale;
    }
    y[row0 + r] = yr;
  }
}

// XBF4 单行反解（embedding 查表）：out[i] = (field_i − 8)·s_sb
static inline void dequant_xbf4_row(const uint8_t* wrow, int C, float* out) {
  for (int sb = 0; sb < C / 256; sb++) {
    const uint8_t* wsb = wrow + sb * 130;
    float sc = bf16_to_f32(wsb[0] | (wsb[1] << 8));
    for (int i = 0; i < 128; i++) {
      uint8_t byte = wsb[2 + i];
      out[sb * 256 + 2 * i] = ((int)(byte & 0xF) - 8) * sc;
      out[sb * 256 + 2 * i + 1] = ((int)(byte >> 4) - 8) * sc;
    }
  }
}

// 激活拆分：XBF4 → xe/xo（各 n/2）；XBF2 → xq4[4]（各 n/4）
static inline void xq_split4(const int8_t* xq, int n, int8_t* xe, int8_t* xo) {
  for (int i = 0; i < n / 2; i++) { xe[i] = xq[2 * i]; xo[i] = xq[2 * i + 1]; }
}
static inline void xq_split2(const int8_t* xq, int n, int8_t* xq4) {
  for (int i = 0; i < n / 4; i++)
    for (int t = 0; t < 4; t++) xq4[t * (n / 4) + i] = xq[4 * i + t];
}
