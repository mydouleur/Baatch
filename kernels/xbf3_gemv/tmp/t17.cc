#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" void t17(const int16_t* a, float* o) {
  aie::vector<int16_t,16> v = aie::load_v<16>(a);
  aie::vector<int16_t,16> m = aie::broadcast<int16_t,16>(0x00FF);
  aie::vector<float,16> f = aie::to_float<float>(aie::bit_and(v, m));
  aie::store_v(o, f);
}
