// dequant_dbg.cc — W4 反量化链分阶段 dump（Peano bug 定位，任务"修 0a"）
// 对 128×64 大瓦片逐 8×8 小瓦片跑 TileFuse 原版反量化链，三阶段分别落盘：
//   st1 = unpack 后 uint8（未减 zp）   [128*64 B]
//   st2 = sub zp 后 int8               [128*64 B]
//   st3 = to_float<bf16> × scale 后    [128*64*2 B]
#define NOCPP
#include <stdint.h>
#include <aie_api/aie.hpp>

extern "C" void dequant_dbg(const int8_t* __restrict b,
                            uint8_t* __restrict outA,   // [0]=st1 u8, [8192]=st2 i8
                            bfloat16* __restrict st3) {
  uint8_t* st1 = outA;
  int8_t* st2 = (int8_t*)(outA + 8192);
  const unsigned colA = 16, colB = 8, size_B = 64;
  const int8_t* pB_quantized = b;
  const int8_t* pBs_b = b + 64 * 128 / 2;
  const int8_t* pBzp_b = b + 64 * 128 / 2 + 64 * 2;
  for (unsigned i = 0; i < colA; ++i) {
    for (unsigned j = 0; j < colB; ++j) {
      const bfloat16* pBs_bf16 = reinterpret_cast<const bfloat16*>(pBs_b);
      const aie::vector<bfloat16, 8> Bs_row = aie::load_v<8>(pBs_bf16 + j * 8);
      const aie::vector<bfloat16, size_B> Bs_replicated = Bs_row.template grow_replicate<size_B>();
      const aie::vector<int8_t, 16> Bzp_row = aie::load_v<16>(pBzp_b + j * 16);
      const aie::vector<int8_t, size_B> Bzp = Bzp_row.template grow_replicate<size_B>();

      const aie::vector<int8_t, 32> B_quantized =
          aie::load_v<32>(pB_quantized + (i * colB + j) * size_B / 2);
      const aie::vector<uint4, size_B> B_uint4 = B_quantized.template cast_to<uint4>();
      const aie::vector<uint8, size_B> B_uint8 = B_uint4.template unpack();
      const aie::vector<int8_t, size_B> B_int8 = B_uint8.template cast_to<int8>();
      aie::store_v(st1 + (i * colB + j) * size_B, B_uint8);
      const aie::vector<int8_t, size_B> B_int8_zeroed = aie::sub(B_int8, Bzp);
      aie::store_v(st2 + (i * colB + j) * size_B, B_int8_zeroed);
      const aie::vector<bfloat16, size_B> B_dequantized = aie::to_float<bfloat16>(B_int8_zeroed);
      const auto B_scaled = aie::mul(B_dequantized, Bs_replicated);
      aie::store_v(st3 + (i * colB + j) * size_B, B_scaled.template to_vector<bfloat16>());
    }
  }
}
