// test_kernels.cpp — XBF4/XBF2 vnni kernel 自包含数值单测（任务 M3'-a 阶段4）
// 随机权重 → 打包（C++ 内实现 pack4/pack2 + 标量参考反量化）→ kernel vs fp64 参考
#include "kernels_xbf.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>

// ---- 标量参考：field = q + OFF，y[r] = (Σ field·xq − OFF·Σxq)·sx·s
template <int OFF>
static void ref_gemv(const uint8_t* packed, const uint16_t* scales_bf16,
                     int R, int C, const int8_t* xq, float sx, const float* sumxq,
                     float* y, int vals_per_byte) {
  int sb_per_row = C / 256;
  for (int r = 0; r < R; r++) {
    double acc = 0;
    for (int sb = 0; sb < sb_per_row; sb++) {
      int isum = 0;
      for (int i = 0; i < 256; i++) {
        int byte = packed[(r * sb_per_row + sb) * (256 / vals_per_byte) + i / vals_per_byte];
        int field = (byte >> ((i % vals_per_byte) * (8 / vals_per_byte))) & ((1 << (8 / vals_per_byte)) - 1);
        isum += field * xq[sb * 256 + i];
      }
      float s = bf16_to_f32(scales_bf16[r * sb_per_row + sb]);
      acc += (double)(isum - OFF * (int)sumxq[sb]) * sx * s;
    }
    y[r] = (float)acc;
  }
}

template <int BITS>
static int run_case(int R, int C, std::mt19937& rng) {
  constexpr int OFF = BITS == 4 ? 8 : 2;
  constexpr int VPB = 8 / BITS;             // 2 or 4
  int sb_per_row = C / 256;
  int wrow = sb_per_row * (2 + 256 / VPB);  // 2B scale + 数据
  std::vector<uint8_t> w((size_t)R * wrow);
  std::vector<float> x(C);
  std::vector<int8_t> xq(C);
  std::vector<float> sumxq(C / 256);
  std::vector<float> y(R, -1e30f), yref(R);
  // 随机 field
  std::uniform_int_distribution<int> fd(0, (1 << BITS) - 1);
  std::uniform_real_distribution<float> xd(-1.f, 1.f);
  for (auto& v : x) v = xd(rng);
  float amax = 0; for (auto v : x) amax = std::max(amax, fabsf(v));
  float sx = amax / 127.f;
  for (int i = 0; i < C; i++) xq[i] = (int8_t)lroundf(x[i] / sx);
  for (int sb = 0; sb < C / 256; sb++) { int s = 0; for (int i = 0; i < 256; i++) s += xq[sb*256+i]; sumxq[sb] = s; }
  for (int r = 0; r < R; r++) {
    for (int sb = 0; sb < sb_per_row; sb++) {
      uint8_t* base = w.data() + r * wrow + sb * (2 + 256 / VPB);
      uint16_t sc = 0x3F80;  // scale 1.0
      base[0] = sc & 0xFF; base[1] = sc >> 8;
      for (int i = 0; i < 256; i += VPB) {
        int byte = 0;
        for (int t = 0; t < VPB; t++) byte |= fd(rng) << (t * BITS);
        base[2 + i / VPB] = (uint8_t)byte;
      }
    }
  }
  if constexpr (BITS == 4) {
    std::vector<int8_t> xe(C / 2), xo(C / 2);
    xq_split4(xq.data(), C, xe.data(), xo.data());
    gemv_xbf4_rows(w.data(), 0, R, C, xe.data(), xo.data(), sumxq.data(), sx, y.data());
  } else {
    std::vector<int8_t> xq4buf(C);
    xq_split2(xq.data(), C, xq4buf.data());
    const int8_t* xq4[4] = {xq4buf.data(), xq4buf.data() + C/4, xq4buf.data() + C/2, xq4buf.data() + 3*C/4};
    gemv_xbf2_rows(w.data(), 0, R, C, xq4, sumxq.data(), sx, y.data());
  }
  // 参考（从打包数据反解）
  std::vector<uint8_t> dat(R * sb_per_row * 256 / VPB);
  std::vector<uint16_t> sc(R * sb_per_row);
  for (int r = 0; r < R; r++)
    for (int sb = 0; sb < sb_per_row; sb++) {
      const uint8_t* base = w.data() + r * wrow + sb * (2 + 256 / VPB);
      sc[r * sb_per_row + sb] = base[0] | (base[1] << 8);
      memcpy(dat.data() + (r * sb_per_row + sb) * (256 / VPB), base + 2, 256 / VPB);
    }
  ref_gemv<OFF>(dat.data(), sc.data(), R, C, xq.data(), sx, sumxq.data(), yref.data(), VPB);
  if constexpr (BITS == 4) {
    std::vector<uint8_t> w128((size_t)R * (C / 128) * 66);
    std::vector<int> sums128(C / 128);
    std::vector<int8_t> xe(C / 2), xo(C / 2);
    std::vector<float> y128(R);
    for (int b = 0; b < C / 128; b++) {
      for (int i = 0; i < 128; i++) sums128[b] += xq[b * 128 + i];
      for (int r = 0; r < R; r++) {
        const uint8_t* src = w.data() + (size_t)r * wrow + (b / 2) * 130;
        uint8_t* dst = w128.data() + ((size_t)r * (C / 128) + b) * 66;
        memcpy(dst, src, 2);
        memcpy(dst + 2, src + 2 + (b % 2) * 64, 64);
      }
    }
    xq_split4(xq.data(), C, xe.data(), xo.data());
    gemv_xbf4_128_rows(w128.data(), R, C, xe.data(), xo.data(), sums128.data(), sx, y128.data());
    for (int r = 0; r < R; r++) {
      float rel = fabsf(y128[r] - yref[r]) / (fabsf(yref[r]) + 1e-3f);
      if (rel > 2e-2f && fabsf(y128[r] - yref[r]) > 1e-2f) {
        printf("XBF4-128 row=%d FAIL rel=%.3e\n", r, rel);
        return 1;
      }
    }
    std::vector<float> y_npu(R);
    gemv_xbf4_128_rows_npu(w128.data(), R, C, xe.data(), xo.data(), sums128.data(), sx, y_npu.data());
    for (int r = 0; r < R; r++) {
      float ref_npu = 0.f;
      for (int b = 0; b < C / 128; b++) {
        int isum = -8 * sums128[b];
        for (int i = 0; i < 128; i++) {
          int byte = dat[(r * sb_per_row + b / 2) * 128 + (b % 2) * 64 + i / 2];
          int field = (byte >> ((i % 2) * 4)) & 15;
          isum += field * xq[b * 128 + i];
        }
        float acc_bf = bf16_to_f32(f32_to_bf16((float)isum));
        const uint8_t* wb = w128.data() + ((size_t)r * (C / 128) + b) * 66;
        float scale = bf16_to_f32(wb[0] | (wb[1] << 8));
        ref_npu = bf16_to_f32(f32_to_bf16(ref_npu + acc_bf * scale));
      }
      ref_npu *= sx;
      if (y_npu[r] != ref_npu) {
        printf("XBF4-128-NPU row=%d FAIL got=%.9g ref=%.9g\n", r, y_npu[r], ref_npu);
        return 1;
      }
    }
  }
  int bad = 0; float max_rel = 0;
  for (int r = 0; r < R; r++) {
    float d = fabsf(y[r] - yref[r]);
    float rel = d / (fabsf(yref[r]) + 1e-3f);
    if (rel > max_rel) max_rel = rel;
    if (rel > 2e-2f && d > 1e-2f) bad++;
  }
  printf("XBF%d R=%d C=%d: %s (bad=%d max_rel=%.2e)\n", BITS, R, C, bad ? "FAIL" : "PASS", bad, max_rel);
  return bad;
}

int main() {
  std::mt19937 rng(7);
  int bad = 0;
  bad += run_case<4>(2048, 512, rng);    // down_proj 形状
  bad += run_case<4>(1024, 2048, rng);   // gate_up 形状
  bad += run_case<4>(248320 / 64, 2048, rng);  // lm_head 缩样
  bad += run_case<2>(2048, 512, rng);
  bad += run_case<2>(1024, 2048, rng);
  printf(bad ? "OVERALL FAIL\n" : "OVERALL PASS\n");
  return bad;
}
