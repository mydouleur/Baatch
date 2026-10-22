// v6: 激活相位平面加载（xp_lo/xp_hi 定义在 xbf3_bps.cc）
#include <stdint.h>
#include <aie_api/aie.hpp>
extern aie::vector<float, 16> xp_lo[16];
extern aie::vector<float, 16> xp_hi[16];

extern "C"
void xbf3_load_act6(const uint8_t* __restrict w, float* __restrict y) {
  const float* src = (const float*)w;
  float* dlo = (float*)xp_lo;
  float* dhi = (float*)xp_hi;
  for (int i = 0; i < 256; i++) { dlo[i] = src[i]; dhi[i] = src[256 + i]; }
  (void)y;
}
