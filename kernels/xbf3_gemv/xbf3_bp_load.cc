// v4: 激活相位平面加载（xp_v 定义在 xbf3_bp.cc）
#include <stdint.h>
#include <aie_api/aie.hpp>
#define XBF3_ROW_SBS 2
extern aie::vector<float, 32> xp_v[8 * XBF3_ROW_SBS];

extern "C"
void xbf3_load_act4(const uint8_t* __restrict w, float* __restrict y) {
  const float* src = (const float*)w;
  float* dst = (float*)xp_v;
  for (int i = 0; i < 8 * XBF3_ROW_SBS * 32; i++) dst[i] = src[i];
  (void)y;
}
