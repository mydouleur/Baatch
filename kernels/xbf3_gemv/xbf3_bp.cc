// xbf3_bp.cc — XBF3 位平面变体（XBF3-BP）向量化 GEMV kernel（AIE2P / aie_api）。
//
// 布局（对应 tools/quant/xbf3.py pack3_bp）：行主序，每 superblock(256 权重) =
//   [2B bf16 scale][plane0 32B][plane1 32B][plane2 32B]
// plane p byte b bit k = value[8b+k] 的第 p 位（value = q & 7）。
//
// 提取（AIE2P 无逐车道移位指令，走 MAC 单元：mul + 带移位的 srs）：
//   lane j (16b) = 字节对 (2j, 2j+1)；acc = Pp * (1<<(8-k)) 把 bit k 顶到
//   lo 字节 bit8 / hi 字节 bit16；srs >>8 后 band 0x0101 得每字节 0/1。
//   E_k = P0_k + 2*P1_k + 4*P2_k；符号 (e^4)-4（按字节无借位）。
//   E 的 int8 视图按字节序 = 权重序 {8b+k}，配 host 预排的相位平面 xp（fp32）。
//
// 激活（host 预排为 fp32 相位平面）：xp[k][sb][i] = x[sb*256 + 8i + k]。

#include <stdint.h>
#include <aie_api/aie.hpp>

#define XBF3_IN 512
#define XBF3_SB 256
#define XBF3_ROW_SBS (XBF3_IN / XBF3_SB)
#define XBF3_SB_BYTES 98
#ifdef CHUNK_ROWS_C
#define CHUNK_ROWS CHUNK_ROWS_C
#else
#define CHUNK_ROWS 16
#endif

// 防强度归约的常量表（非 const，编译器无法折成 shl / 优化掉）
static int16_t g_shift_tab[8] = {256, 128, 64, 32, 16, 8, 4, 2};      // 1<<(8-k)
static int16_t g_shift_tab1[8] = {512, 256, 128, 64, 32, 16, 8, 4};    // 1<<(9-k)
static int16_t g_shift_tab2[8] = {1024, 512, 256, 128, 64, 32, 16, 8}; // 1<<(10-k)
static int16_t g_bitmask_tab[8] = {0x0101,0x0202,0x0404,0x0808,0x1010,0x2020,0x4040,(int16_t)0x8080};
// int16→int8 视图的 DM 暂存
static int16_t g_scratch16[16] __attribute__((aligned(32)));

// 激活相位平面（fp32），load_act 拷入：8 phase × 2 sb × 32 lanes
// （非 static：与 xbf3_bp_load.cc 共享，避免 .o 重复符号）
aie::vector<float, 32> xp_v[8 * XBF3_ROW_SBS];


extern "C"
void xbf3_gemv_chunk4(const uint8_t* __restrict w, float* __restrict y) {
  const aie::vector<int16_t, 16> m07 = aie::broadcast<int16_t, 16>(0x0707);
  const aie::vector<int8_t, 32>  m4  = aie::broadcast<int8_t, 32>(4);
  for (int r = 0; r < CHUNK_ROWS; r++) {
    const uint8_t* rowp = w + r * (XBF3_ROW_SBS * XBF3_SB_BYTES);
    float acc_row = 0.0f;
    for (int sb = 0; sb < XBF3_ROW_SBS; sb++, rowp += XBF3_SB_BYTES) {
      uint32_t sbits = (uint32_t)rowp[0] | ((uint32_t)rowp[1] << 8);
      uint32_t fbits = sbits << 16;
      float scale;
      __builtin_memcpy(&scale, &fbits, sizeof(float));

      // 平面起点非 32B 对齐（scale 2B 前缀 + 196B 行距）→ 用非对齐迭代器
      aie::vector<int16_t, 16> p0 = *aie::cbegin_unaligned_vector<16>((const int16_t*)(rowp + 2));
      aie::vector<int16_t, 16> p1 = *aie::cbegin_unaligned_vector<16>((const int16_t*)(rowp + 34));
      aie::vector<int16_t, 16> p2 = *aie::cbegin_unaligned_vector<16>((const int16_t*)(rowp + 66));

      aie::accum<accfloat, 32> vacc = aie::zeros<accfloat, 32>();
#pragma unroll
      for (int k = 0; k < 8; k++) {
        // 先按位隔离 bit k（0x0101<<k 掩码），再按平面加权 mul/mac 上移到
        // 字节内 bits[0..2]（权重 1/2/4 折进乘子），srs>>8 落位，band 清场。
        const aie::vector<int16_t, 16> mk =
            aie::broadcast<int16_t, 16>(g_bitmask_tab[k]);
        aie::vector<int16_t, 16> t0 = aie::bit_and(p0, mk);
        aie::vector<int16_t, 16> t1 = aie::bit_and(p1, mk);
        aie::vector<int16_t, 16> t2 = aie::bit_and(p2, mk);
        aie::accum<acc32, 16> acc = aie::mul(t0, aie::broadcast<int16_t, 16>(g_shift_tab[k]));
        acc = aie::mac(acc, t1, aie::broadcast<int16_t, 16>(g_shift_tab1[k]));
        acc = aie::mac(acc, t2, aie::broadcast<int16_t, 16>(g_shift_tab2[k]));
        aie::vector<int16_t, 16> e = aie::bit_and(acc.to_vector<int16_t>(8), m07);
        // int16 车道 → int8 视图只能走 DM 往返（cast_to 窄化会生成非法 shl）
        aie::store_v(g_scratch16, e);
        aie::vector<int8_t, 32> e8 = aie::load_v<32>((const int8_t*)g_scratch16);
        e8 = aie::bit_xor(e8, m4) - m4;
        aie::vector<float, 32> qb = aie::to_float<float>(e8);
        vacc = aie::mac(vacc, qb, xp_v[k * XBF3_ROW_SBS + sb]);
      }
      float sacc = aie::reduce_add(vacc.to_vector<float>());
      acc_row += scale * sacc;
    }
    y[r] = acc_row;
  }
}

// 管线测试用（不计算）
extern "C"
void xbf3_discard_chunk4(const uint8_t* __restrict w, float* __restrict y) {
  y[0] = (float)w[0];
}

// 调试 v2：y = [p0(16f) | p1(16f) | p2(16f) | e8(k=0)(32f) | e8(k=1)(32f) | scale(1f)]
extern "C"
void xbf3_debug4(const uint8_t* __restrict w, float* __restrict y) {
  const aie::vector<int16_t, 16> m07 = aie::broadcast<int16_t, 16>(0x0707);
  const aie::vector<int8_t, 32>  m4  = aie::broadcast<int8_t, 32>(4);
  const uint8_t* rowp = w;
  aie::vector<int16_t, 16> p0 = *aie::cbegin_unaligned_vector<16>((const int16_t*)(rowp + 2));
  aie::vector<int16_t, 16> p1 = *aie::cbegin_unaligned_vector<16>((const int16_t*)(rowp + 34));
  aie::vector<int16_t, 16> p2 = *aie::cbegin_unaligned_vector<16>((const int16_t*)(rowp + 66));
  aie::store_v(y + 0,  aie::to_float<float>(p0));
  aie::store_v(y + 16, aie::to_float<float>(p1));
  aie::store_v(y + 32, aie::to_float<float>(p2));
#pragma unroll
  for (int k = 0; k < 2; k++) {
    const aie::vector<int16_t, 16> mk = aie::broadcast<int16_t, 16>(g_bitmask_tab[k]);
    aie::vector<int16_t, 16> t0 = aie::bit_and(p0, mk);
    aie::vector<int16_t, 16> t1 = aie::bit_and(p1, mk);
    aie::vector<int16_t, 16> t2 = aie::bit_and(p2, mk);
    aie::accum<acc32, 16> acc = aie::mul(t0, aie::broadcast<int16_t, 16>(g_shift_tab[k]));
    acc = aie::mac(acc, t1, aie::broadcast<int16_t, 16>(g_shift_tab1[k]));
    acc = aie::mac(acc, t2, aie::broadcast<int16_t, 16>(g_shift_tab2[k]));
    aie::vector<int16_t, 16> e = aie::bit_and(acc.to_vector<int16_t>(8), m07);
    aie::store_v(g_scratch16, e);
    aie::vector<int8_t, 32> e8 = aie::load_v<32>((const int8_t*)g_scratch16);
    e8 = aie::bit_xor(e8, m4) - m4;
    aie::vector<float, 32> qf = aie::to_float<float>(e8);
    aie::store_v(y + 48 + k * 32, qf);
  }
  uint32_t sbits = (uint32_t)rowp[0] | ((uint32_t)rowp[1] << 8);
  uint32_t fbits = sbits << 16;
  __builtin_memcpy(&y[112], &fbits, 4);
}
