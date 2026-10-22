#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" void t16(const float* a, const float* b, float* o) {
  aie::vector<float,16> va = aie::load_v<16>(a);
  aie::vector<float,16> vb = aie::load_v<16>(b);
  aie::accum<accfloat,16> acc = aie::zeros<accfloat,16>();
  acc = aie::mac(acc, va, vb);
  aie::store_v(o, acc.to_vector<float>());
}
