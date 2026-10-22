// xbf3_bps.cc — XBF3-BP 向量化 GEMV v6（M2'-5）：减 stall 版。
//
// 相对 v4：
//  - int16→int8 的 DM 往返改为 band 拆 lo/hi 字节直接 to_float（hi 通道 xp 预除 256）
//  - scale 在向量域折入，每行一次 reduce（消除每 sb 的标量软浮点——AIE2P 无标量 FP）
//  - 两个 sb 的 k 循环独立累加链（acc_lo/acc_hi per sb），行尾合并
//
// 提取（与 v4 同，已仿真）：t_p=band(Pp,0x0101<<k); acc=t0<<(8-k)+t1<<(9-k)+t2<<(10-k)
// （mul/mac 加权上移，权重折进乘子）；srs>>8 + band 0x0707 → lo/hi 字节 = 无符号场 u；
// q=(u^4)-4（按字节无借位）。lo 字节=value[8*(2j)+k]，hi=value[8*(2j+1)+k]。

#include <stdint.h>
#include <aie_api/aie.hpp>

#define XBF3_ROW_SBS 2
#ifdef CHUNK_ROWS_C
#define CHUNK_ROWS CHUNK_ROWS_C
#else
#define CHUNK_ROWS 128
#endif

// 防强度归约的常量表（非 const）
static int16_t g_bitmask_tab[8] = {0x0101,0x0202,0x0404,0x0808,0x1010,0x2020,0x4040,(int16_t)0x8080};
static int16_t g_shift_tab[8]  = {256,128,64,32,16,8,4,2};          // 1<<(8-k)
static int16_t g_shift_tab1[8] = {512,256,128,64,32,16,8,4};         // 1<<(9-k)
static int16_t g_shift_tab2[8] = {1024,512,256,128,64,32,16,8};      // 1<<(10-k)

// 激活相位平面（fp32）：xp_lo[(k*2+sb)*16+j] = x[sb*256+8*(2j)+k]
//                       xp_hi[(k*2+sb)*16+j] = x[sb*256+8*(2j+1)+k] / 256
// （非 static：与 xbf3_bps_load.cc 共享）
aie::vector<float, 16> xp_lo[16];
aie::vector<float, 16> xp_hi[16];

extern "C"
void xbf3_gemv_chunk6(const uint8_t* __restrict w, float* __restrict y) {
  const aie::vector<uint16_t, 16> m07  = aie::broadcast<uint16_t, 16>(0x0707);
  const aie::vector<int16_t, 16>  m04  = aie::broadcast<int16_t, 16>(0x0404);
  const aie::vector<int16_t, 16>  mlo  = aie::broadcast<int16_t, 16>(0x00FF);
  const aie::vector<int16_t, 16>  mhi  = aie::broadcast<int16_t, 16>((int16_t)0xFF00);
  const aie::vector<int16_t, 16>  m80  = aie::broadcast<int16_t, 16>(0x0080);
  aie::vector<uint16_t, 16> mk_v[8], sh0_v[8], sh1_v[8], sh2_v[8];
  for (int k = 0; k < 8; k++) {
    mk_v[k]  = aie::broadcast<uint16_t, 16>((uint16_t)g_bitmask_tab[k]);
    sh0_v[k] = aie::broadcast<uint16_t, 16>((uint16_t)g_shift_tab[k]);
    sh1_v[k] = aie::broadcast<uint16_t, 16>((uint16_t)g_shift_tab1[k]);
    sh2_v[k] = aie::broadcast<uint16_t, 16>((uint16_t)g_shift_tab2[k]);
  }
  for (int r = 0; r < CHUNK_ROWS; r++) {
    const uint8_t* rowp = w + r * (2 * 98);
    aie::accum<accfloat, 16> row_acc = aie::zeros<accfloat, 16>();
    for (int sb = 0; sb < 2; sb++, rowp += 98) {
      uint32_t sbits = (uint32_t)rowp[0] | ((uint32_t)rowp[1] << 8);
      uint32_t fbits = sbits << 16;
      float scale;
      __builtin_memcpy(&scale, &fbits, sizeof(float));
      aie::vector<float, 16> scale_v = aie::broadcast<float, 16>(scale);

      aie::vector<uint16_t, 16> p0 = *aie::cbegin_unaligned_vector<16>((const uint16_t*)(rowp + 2));
      aie::vector<uint16_t, 16> p1 = *aie::cbegin_unaligned_vector<16>((const uint16_t*)(rowp + 34));
      aie::vector<uint16_t, 16> p2 = *aie::cbegin_unaligned_vector<16>((const uint16_t*)(rowp + 66));

      aie::accum<accfloat, 16> sb_acc = aie::zeros<accfloat, 16>();
#pragma unroll
      for (int k = 0; k < 8; k++) {
        aie::vector<uint16_t, 16> t0 = aie::bit_and(p0, mk_v[k]);
        aie::vector<uint16_t, 16> t1 = aie::bit_and(p1, mk_v[k]);
        aie::vector<uint16_t, 16> t2 = aie::bit_and(p2, mk_v[k]);
        aie::accum<acc32, 16> acc = aie::mul(t0, sh0_v[k]);
        acc = aie::mac(acc, t1, sh1_v[k]);
        acc = aie::mac(acc, t2, sh2_v[k]);
        aie::vector<uint16_t, 16> e =
            aie::bit_and(acc.to_vector_sign<uint16_t>(false, 8), m07);
        aie::vector<int16_t, 16> es = aie::bit_xor(e.cast_to<int16_t>(), m04) - m04;
        // lo 字节需符号扩展：(b ^ 0x80) - 0x80；hi 字节在 int16 符号位上天然正确
        aie::vector<int16_t, 16> lo_s =
            aie::bit_xor(aie::bit_and(es, mlo), m80) - m80;
        aie::vector<float, 16> flo = aie::to_float<float>(lo_s);
        aie::vector<float, 16> fhi = aie::to_float<float>(aie::bit_and(es, mhi));
        sb_acc = aie::mac(sb_acc, flo, xp_lo[k * 2 + sb]);
        sb_acc = aie::mac(sb_acc, fhi, xp_hi[k * 2 + sb]);
      }
      row_acc = aie::mac(row_acc, scale_v, sb_acc.to_vector<float>());
    }
    y[r] = aie::reduce_add(row_acc.to_vector<float>());
  }
}

// 管线测试用（不计算）
extern "C"
void xbf3_discard_chunk6(const uint8_t* __restrict w, float* __restrict y) {
  y[0] = (float)w[0];
}
