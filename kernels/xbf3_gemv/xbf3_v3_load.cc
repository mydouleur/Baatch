// v3: 激活加载（xbuf_v3 定义在 xbf3_v3_gemv.cc）
#include <stdint.h>
extern float xbuf_v3[512];

extern "C"
void xbf3_load_act(const uint8_t* __restrict w, float* __restrict y) {
  const float* src = (const float*)w;
  for (int i = 0; i < 512; i++) xbuf_v3[i] = src[i];
  (void)y;
}
