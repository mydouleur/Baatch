#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" void t23(const float* p, float* o) {
  aie::vector<float,16> a = aie::load_v<16>(p);
  aie::vector<float,16> b = aie::load_v<16>(p+16);
  aie::vector<float,16> c = a + b;
  aie::store_v(o, c);
}
