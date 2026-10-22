#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" void t18(const uint16_t* p, float* o) {
  aie::vector<uint16_t,16> v = aie::load_v<16>(p);
  aie::vector<uint16_t,16> mk = aie::broadcast<uint16_t,16>(0x0101);
  aie::vector<uint16_t,16> sh = aie::broadcast<uint16_t,16>(256);
  aie::vector<uint16_t,16> t = aie::bit_and(v, mk);
  aie::accum<acc32,16> acc = aie::mul(t, sh);
  aie::vector<uint16_t,16> e = acc.to_vector_sign<uint16_t>(false, 8);
  aie::vector<int16_t,16> es = aie::bit_xor(e.cast_to<int16_t>(), aie::broadcast<int16_t,16>(0x0404));
  es = es - aie::broadcast<int16_t,16>(0x0404);
  aie::vector<float,16> f = aie::to_float<float>(es);
  aie::store_v(o, f);
}
