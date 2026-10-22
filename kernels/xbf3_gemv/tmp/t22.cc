#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" void t22(const float* p, float* y) {
  for (int r = 0; r < 4; r++) {
    aie::vector<float,16> v = aie::load_v<16>(p);
    y[r] = aie::reduce_add(v);
  }
}
