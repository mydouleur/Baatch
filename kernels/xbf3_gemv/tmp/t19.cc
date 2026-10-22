#include <stdint.h>
#include <aie_api/aie.hpp>
static aie::vector<float,16> g_xp[16];
extern "C" void t19(const int16_t* p, float* o) {
  aie::vector<int16_t,16> v = aie::load_v<16>(p);
  aie::vector<float,16> f = aie::to_float<float>(v);
  aie::accum<accfloat,16> acc = aie::zeros<accfloat,16>();
  for (int k = 0; k < 8; k++)
    acc = aie::mac(acc, f, g_xp[k]);
  aie::vector<float,16> rv = acc.to_vector<float>();
  aie::store_v(o, rv);
}
