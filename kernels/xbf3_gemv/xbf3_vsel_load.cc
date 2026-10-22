// v7: 激活加载（xp7 定义在 xbf3_vsel.cc）
#include <stdint.h>
#include <aie_api/aie.hpp>
extern aie::vector<float, 32> xp7[3 * 8 * 2];

extern "C"
void xbf3_load_act7(const uint8_t* __restrict w, float* __restrict y) {
  const float* src = (const float*)w;
  float* dst = (float*)xp7;
  for (int i = 0; i < 3 * 8 * 2 * 32; i++) dst[i] = src[i];
  (void)y;
}
