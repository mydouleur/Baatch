// xbf3_vsel.cc — XBF3-BP 二进制平面 GEMV v7（M2'-5）：band+ne+select+mac，无移位无转换。
//
// 数学：q = b0 + 2·b1 − 4·b2（3-bit 补码 = 平面 2 权重 −4，零符号修正）。
//   y[row] = Σ_sb scale_sb · Σ_p w_p · Σ_{i: bit p of value i set} x[i]   （w = 1, 2, −4）
//
// 实现：每 (plane p, bit k)：
//   t = bit_and(Pp, 1<<k)        // v32int8：plane 32B = 32 值
//   m = t != 0                   // 掩码
//   acc = mac(acc, select(0, xp[p][k][sb], m), ones)   // MAC 单元累加（aie2p 无浮点加）
// xp 由 host 预排+预加权：xp[p][k][sb][lane] = x[sb*256 + 8*lane + k] × {1,2,-4}[p]。

#include <stdint.h>
#include <aie_api/aie.hpp>

#ifdef CHUNK_ROWS_C
#define CHUNK_ROWS CHUNK_ROWS_C
#else
#define CHUNK_ROWS 128
#endif

// 3 平面 × 8 相位 × 2 sb × 32 lanes = 1536 floats = 6KB
aie::vector<float, 32> xp7[3 * 8 * 2];  // 非 static：与 load TU 共享

extern "C"
void xbf3_gemv_chunk7(const uint8_t* __restrict w, float* __restrict y) {
  const aie::vector<uint8_t, 32> z8 = aie::zeros<uint8_t, 32>();
  const aie::vector<float, 32> onef = aie::broadcast<float, 32>(1.0f);
  for (int r = 0; r < CHUNK_ROWS; r++) {
    const uint8_t* rowp = w + r * (2 * 98);
    aie::accum<accfloat, 32> row_acc = aie::zeros<accfloat, 32>();
    for (int sb = 0; sb < 2; sb++, rowp += 98) {
      uint32_t sbits = (uint32_t)rowp[0] | ((uint32_t)rowp[1] << 8);
      uint32_t fbits = sbits << 16;
      float scale;
      __builtin_memcpy(&scale, &fbits, sizeof(float));
      aie::vector<float, 32> scale_v = aie::broadcast<float, 32>(scale);

      aie::vector<uint8_t, 32> p0 = *aie::cbegin_unaligned_vector<32>((const uint8_t*)(rowp + 2));
      aie::vector<uint8_t, 32> p1 = *aie::cbegin_unaligned_vector<32>((const uint8_t*)(rowp + 34));
      aie::vector<uint8_t, 32> p2 = *aie::cbegin_unaligned_vector<32>((const uint8_t*)(rowp + 66));

      aie::accum<accfloat, 32> sb_acc = aie::zeros<accfloat, 32>();
#pragma unroll
      for (int k = 0; k < 8; k++) {
        aie::vector<uint8_t, 32> mk = aie::broadcast<uint8_t, 32>((uint8_t)(1 << k));
        const aie::vector<float, 32> zerof = aie::zeros<float, 32>();
#pragma unroll
        for (int p = 0; p < 3; p++) {
          aie::vector<uint8_t, 32> t = aie::bit_and(p == 0 ? p0 : (p == 1 ? p1 : p2), mk);
          aie::mask<32> m = aie::gt(t, z8);
          aie::vector<float, 32> sel = aie::select(zerof, xp7[p * 16 + k * 2 + sb], m);
          sb_acc = aie::mac(sb_acc, sel, onef);
        }
      }
      row_acc = aie::mac(row_acc, scale_v, sb_acc.to_vector<float>());
    }
    y[r] = aie::reduce_add(row_acc.to_vector<float>());
  }
}

// 管线测试用（不计算）
extern "C"
void xbf3_discard_chunk7(const uint8_t* __restrict w, float* __restrict y) {
  y[0] = (float)w[0];
}
