#include <stdint.h>
#include <aie_api/aie.hpp>
static int16_t g_bitmask_tab[8] = {0x0101,0x0202,0x0404,0x0808,0x1010,0x2020,0x4040,(int16_t)0x8080};
static int16_t g_shift_tab[8]  = {256,128,64,32,16,8,4,2};
static int16_t g_shift_tab1[8] = {512,256,128,64,32,16,8,4};
static int16_t g_shift_tab2[8] = {1024,512,256,128,64,32,16,8};
static aie::vector<float, 16> xp_lo[16];
static aie::vector<float, 16> xp_hi[16];
extern "C" void t20(const uint8_t* w, float* y) {
  const aie::vector<uint16_t, 16> m07 = aie::broadcast<uint16_t, 16>(0x0707);
  const aie::vector<int16_t, 16>  m04 = aie::broadcast<int16_t, 16>(0x0404);
  const aie::vector<int16_t, 16>  mlo = aie::broadcast<int16_t, 16>(0x00FF);
  const aie::vector<int16_t, 16>  mhi = aie::broadcast<int16_t, 16>((int16_t)0xFF00);
  aie::vector<uint16_t, 16> p0 = aie::load_v<16>((const uint16_t*)w);
  aie::vector<uint16_t, 16> p1 = aie::load_v<16>((const uint16_t*)(w+32));
  aie::vector<uint16_t, 16> p2 = aie::load_v<16>((const uint16_t*)(w+64));
  aie::accum<accfloat, 16> sb_lo = aie::zeros<accfloat, 16>();
  aie::accum<accfloat, 16> sb_hi = aie::zeros<accfloat, 16>();
#pragma unroll
  for (int k = 0; k < 8; k++) {
    aie::vector<uint16_t, 16> mk = aie::broadcast<uint16_t, 16>((uint16_t)g_bitmask_tab[k]);
    aie::vector<uint16_t, 16> t0 = aie::bit_and(p0, mk);
    aie::vector<uint16_t, 16> t1 = aie::bit_and(p1, mk);
    aie::vector<uint16_t, 16> t2 = aie::bit_and(p2, mk);
    aie::accum<acc32, 16> acc = aie::mul(t0, aie::broadcast<uint16_t, 16>((uint16_t)g_shift_tab[k]));
    acc = aie::mac(acc, t1, aie::broadcast<uint16_t, 16>((uint16_t)g_shift_tab1[k]));
    acc = aie::mac(acc, t2, aie::broadcast<uint16_t, 16>((uint16_t)g_shift_tab2[k]));
    aie::vector<uint16_t, 16> e = aie::bit_and(acc.to_vector_sign<uint16_t>(false, 8), m07);
    aie::vector<int16_t, 16> es = aie::bit_xor(e.cast_to<int16_t>(), m04) - m04;
    aie::vector<float, 16> flo = aie::to_float<float>(aie::bit_and(es, mlo));
    aie::vector<float, 16> fhi = aie::to_float<float>(aie::bit_and(es, mhi));
    sb_lo = aie::mac(sb_lo, flo, xp_lo[k]);
    sb_hi = aie::mac(sb_hi, fhi, xp_hi[k]);
  }
  aie::vector<float,16> rv = sb_lo.to_vector<float>();
  y[0] = aie::reduce_add(rv);
  aie::vector<float,16> rv2 = sb_hi.to_vector<float>();
  y[1] = aie::reduce_add(rv2);
}
