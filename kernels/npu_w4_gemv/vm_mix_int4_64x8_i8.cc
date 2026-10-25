// W4A16-int GEMV kernel — drop-in replacement for vm_mix_int4_64x8.cc.
// Same extern "C" symbols / buffer ABI as the bf16 version so the existing
// aie.mlir design links unchanged; only the arithmetic path differs:
//   dequant: int4 unpack -> int8, subtract zp (int domain, no scale), -> int16
//   MAC:     aie::accumulate<64> int16 coeff x int16 data -> acc32
//            (avoids the bf16 variadic-accumulate lowering that miscompiles
//             under Peano/xmc for |v| in {5,7}; int16 is AMD's own GEMM path)
//   epilog:  acc32 -> fp32, one fp32 mul by group scale, add into c, -> bf16
//
// ABI notes:
//   a_in : declared bfloat16[128] (256B); host packs int16 quantized activations
//          (values in [-128,127]) into the full 256B slot; kernel reinterprets.
//   b_in : 4096B packed w4 + 64x bf16 scales + 128B zp (layout unchanged).
//   c_out: bfloat16[64], accumulated in memory across k-tile calls.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#define NOCPP

#include <stdint.h>
#include <stdlib.h>
#include <type_traits>

#define REL_WRITE 0
#define REL_READ 1

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>

#include "zero.cc"

// half-k dequant buffer: 64 rows x 64 cols int16 (8KB, same as bf16 original)
alignas(aie::vector_decl_align) static int16_t b_buf_i16[64 * 64];

static void vecmat_w4a16int(const int16_t *__restrict a_i16, const int8_t *__restrict b,
                            bfloat16 *__restrict c) {
  constexpr unsigned K = 128;   // DIM_K
  constexpr unsigned N = 64;    // DIM_N
  // round-to-nearest-even for the accfloat->bf16 store path (default floor is
  // biased and accumulates systematic error across k-tile calls)
  aie::set_rounding(aie::rounding_mode::conv_even);
  const int8_t *__restrict pW = b;
  const bfloat16 *__restrict pS = reinterpret_cast<const bfloat16 *>(b + N * K / 2);
  const int8_t *__restrict pZ = b + N * K / 2 + N * 2;

  aie::accum<acc32, 64> acc = aie::zeros<acc32, 64>();

  for (unsigned kh = 0; kh < 2; kh++) {
    // ---- dequant half k (64 rows): unpack w4 -> int8, sub zp, -> int16 ----
    // block (i,j): 8k x 8n, 32B packed; lane = k1*8 + n1 (matches bf16 version)
    for (unsigned i = 0; i < 8; ++i) {
      for (unsigned j = 0; j < 8; ++j) {
        const aie::vector<int8, 16> Bzp_row = aie::load_v<16>(pZ + j * 16);
        const aie::vector<int8, 64> Bzp = Bzp_row.template grow_replicate<64>();

        const aie::vector<int8, 32> B_packed =
            aie::load_v<32>(pW + ((kh * 8 + i) * 8 + j) * 32);
        const aie::vector<uint4, 64> B_uint4 = B_packed.template cast_to<uint4>();
        const aie::vector<uint8, 64> B_uint8 = B_uint4.template unpack();
        const aie::vector<int8, 64> B_int8 = B_uint8.template cast_to<int8>();
        const aie::vector<int8, 64> B_z = aie::sub(B_int8, Bzp);
        aie::store_v(b_buf_i16 + (i * 8 + j) * 64, B_z.template unpack_sign<int16>(true));
      }
    }

    // ---- MAC half: acc[n] += sum_k a[k] * B[k][n], int16 x int16 -> acc32 ----
    const int16_t *__restrict a_ptr = a_i16 + kh * 64;
    const int16_t *__restrict b_ptr = b_buf_i16;
    AIE_LOOP_MIN_ITERATION_COUNT(8)
    for (unsigned rr = 0; rr < 8; ++rr) {
      // scalar-domain coeff extraction: pack 2 activations per u32 load to
      // dodge Peano's lane-4..7 vector scalar-extraction miscompile
      uint32_t a01 = *reinterpret_cast<const uint32_t *>(a_ptr + 0);
      uint32_t a23 = *reinterpret_cast<const uint32_t *>(a_ptr + 2);
      uint32_t a45 = *reinterpret_cast<const uint32_t *>(a_ptr + 4);
      uint32_t a67 = *reinterpret_cast<const uint32_t *>(a_ptr + 6);
      const int16_t a_s[8] = {(int16_t)(a01 & 0xFFFF), (int16_t)(a01 >> 16),
                              (int16_t)(a23 & 0xFFFF), (int16_t)(a23 >> 16),
                              (int16_t)(a45 & 0xFFFF), (int16_t)(a45 >> 16),
                              (int16_t)(a67 & 0xFFFF), (int16_t)(a67 >> 16)};
      // interleaved load->mac: only one live 512-bit b_vec (Peano spills of
      // multiple live 512b vectors are suspected to clobber lanes)
#define MAC_T(t) { \
      const aie::vector<int16, 64> bv = aie::concat( \
          aie::load_v<8>(b_ptr + 0 * 64 + t * 8), aie::load_v<8>(b_ptr + 1 * 64 + t * 8), \
          aie::load_v<8>(b_ptr + 2 * 64 + t * 8), aie::load_v<8>(b_ptr + 3 * 64 + t * 8), \
          aie::load_v<8>(b_ptr + 4 * 64 + t * 8), aie::load_v<8>(b_ptr + 5 * 64 + t * 8), \
          aie::load_v<8>(b_ptr + 6 * 64 + t * 8), aie::load_v<8>(b_ptr + 7 * 64 + t * 8)); \
      acc = aie::mac(acc, a_s[t], bv); }
      MAC_T(0) MAC_T(1) MAC_T(2) MAC_T(3) MAC_T(4) MAC_T(5) MAC_T(6) MAC_T(7)
#undef MAC_T
      b_ptr += 512;
      a_ptr += 8;
    }
  }

  // ---- epilog: c += bf16(acc) * scale (original code's verified primitives) ----
  const aie::vector<int32, 64> acci = acc.template to_vector<int32>();
  const aie::vector<bfloat16, 64> accb = aie::to_float<bfloat16>(acci);
  const aie::vector<bfloat16, 64> sv = aie::load_v<64>(pS);
  aie::accum<accfloat, 64> prod = aie::mul(accb, sv);
  aie::accum<accfloat, 64> cacc;
  cacc.from_vector(aie::load_v<64>(c));
  aie::store_v(c, aie::add(cacc, prod).template to_vector<bfloat16>());
}

static void vecmat_w4a16int_m2(const int16_t *__restrict a0, const int16_t *__restrict a1,
                               const int8_t *__restrict b,
                               bfloat16 *__restrict c0, bfloat16 *__restrict c1) {
  constexpr unsigned K = 128;
  constexpr unsigned N = 64;
  aie::set_rounding(aie::rounding_mode::conv_even);
  const int8_t *__restrict pW = b;
  const bfloat16 *__restrict pS = reinterpret_cast<const bfloat16 *>(b + N * K / 2);
  const int8_t *__restrict pZ = b + N * K / 2 + N * 2;
  aie::accum<acc32, 64> acc0 = aie::zeros<acc32, 64>();
  aie::accum<acc32, 64> acc1 = aie::zeros<acc32, 64>();
  for (unsigned kh = 0; kh < 2; kh++) {
    for (unsigned i = 0; i < 8; ++i) {
      for (unsigned j = 0; j < 8; ++j) {
        const aie::vector<int8, 16> Bzp_row = aie::load_v<16>(pZ + j * 16);
        const aie::vector<int8, 64> Bzp = Bzp_row.template grow_replicate<64>();
        const aie::vector<int8, 32> B_packed =
            aie::load_v<32>(pW + ((kh * 8 + i) * 8 + j) * 32);
        const aie::vector<uint4, 64> B_uint4 = B_packed.template cast_to<uint4>();
        const aie::vector<uint8, 64> B_uint8 = B_uint4.template unpack();
        const aie::vector<int8, 64> B_int8 = B_uint8.template cast_to<int8>();
        const aie::vector<int8, 64> B_z = aie::sub(B_int8, Bzp);
        aie::store_v(b_buf_i16 + (i * 8 + j) * 64, B_z.template unpack_sign<int16>(true));
      }
    }
    const int16_t *__restrict a_ptr0 = a0 + kh * 64;
    const int16_t *__restrict a_ptr1 = a1 + kh * 64;
    const int16_t *__restrict b_ptr = b_buf_i16;
    AIE_LOOP_MIN_ITERATION_COUNT(8)
    for (unsigned rr = 0; rr < 8; ++rr) {
      int16_t a_s[2][8];
      for (int m = 0; m < 2; m++) {
        const int16_t* ap = m ? a_ptr1 : a_ptr0;
        for (int t = 0; t < 8; t += 2) {
          uint32_t av = *reinterpret_cast<const uint32_t *>(ap + t);
          a_s[m][t] = (int16_t)(av & 0xFFFF);
          a_s[m][t + 1] = (int16_t)(av >> 16);
        }
      }
#define MAC_M2(t) { \
      const aie::vector<int16, 64> bv = aie::concat( \
          aie::load_v<8>(b_ptr + 0 * 64 + t * 8), aie::load_v<8>(b_ptr + 1 * 64 + t * 8), \
          aie::load_v<8>(b_ptr + 2 * 64 + t * 8), aie::load_v<8>(b_ptr + 3 * 64 + t * 8), \
          aie::load_v<8>(b_ptr + 4 * 64 + t * 8), aie::load_v<8>(b_ptr + 5 * 64 + t * 8), \
          aie::load_v<8>(b_ptr + 6 * 64 + t * 8), aie::load_v<8>(b_ptr + 7 * 64 + t * 8)); \
      acc0 = aie::mac(acc0, a_s[0][t], bv); \
      acc1 = aie::mac(acc1, a_s[1][t], bv); }
      MAC_M2(0) MAC_M2(1) MAC_M2(2) MAC_M2(3)
      MAC_M2(4) MAC_M2(5) MAC_M2(6) MAC_M2(7)
#undef MAC_M2
      b_ptr += 512;
      a_ptr0 += 8;
      a_ptr1 += 8;
    }
  }
  const aie::vector<bfloat16, 64> sv = aie::load_v<64>(pS);
  const aie::vector<int32, 64> acci0 = acc0.template to_vector<int32>();
  const aie::vector<bfloat16, 64> accb0 = aie::to_float<bfloat16>(acci0);
  aie::accum<accfloat, 64> prod0 = aie::mul(accb0, sv);
  aie::accum<accfloat, 64> cacc0;
  cacc0.from_vector(aie::load_v<64>(c0));
  aie::store_v(c0, aie::add(cacc0, prod0).template to_vector<bfloat16>());
  const aie::vector<int32, 64> acci1 = acc1.template to_vector<int32>();
  const aie::vector<bfloat16, 64> accb1 = aie::to_float<bfloat16>(acci1);
  aie::accum<accfloat, 64> prod1 = aie::mul(accb1, sv);
  aie::accum<accfloat, 64> cacc1;
  cacc1.from_vector(aie::load_v<64>(c1));
  aie::store_v(c1, aie::add(cacc1, prod1).template to_vector<bfloat16>());
}

__attribute__((noinline))
static void vecmat_w4a16int_m2_isolated(const int16_t *a0, const int16_t *a1,
                                        const int8_t *b, bfloat16 *c0, bfloat16 *c1) {
  vecmat_w4a16int_m2(a0, a1, b, c0, c1);
}

#if defined(DIM_M) && DIM_M == 8
alignas(aie::vector_decl_align) static int32_t m8_acc_buf[8 * 64];

__attribute__((noinline))
static void vecmat_w4a16int_m8_partial(const int16_t *a0, const int16_t *a1,
                                       const int8_t *b, bfloat16 *c0, bfloat16 *c1,
                                       int m, unsigned kh) {
  constexpr unsigned N = 64;
  aie::set_rounding(aie::rounding_mode::conv_even);
  aie::accum<acc32, 64> acc0 = aie::zeros<acc32, 64>();
  aie::accum<acc32, 64> acc1 = aie::zeros<acc32, 64>();
  if (kh) {
    acc0.from_vector(aie::load_v<64>(m8_acc_buf + m * 64));
    acc1.from_vector(aie::load_v<64>(m8_acc_buf + (m + 1) * 64));
  }
  const int16_t *a_ptr0 = a0 + kh * 64;
  const int16_t *a_ptr1 = a1 + kh * 64;
  const int16_t *b_ptr = b_buf_i16;
  AIE_LOOP_MIN_ITERATION_COUNT(8)
  for (unsigned rr = 0; rr < 8; ++rr) {
    int16_t a_s[2][8];
    for (int row = 0; row < 2; row++) {
      const int16_t *ap = row ? a_ptr1 : a_ptr0;
      for (int t = 0; t < 8; t += 2) {
        uint32_t av = *reinterpret_cast<const uint32_t *>(ap + t);
        a_s[row][t] = (int16_t)(av & 0xFFFF);
        a_s[row][t + 1] = (int16_t)(av >> 16);
      }
    }
#define MAC_M8_PARTIAL(t) { \
    const aie::vector<int16, 64> bv = aie::concat( \
        aie::load_v<8>(b_ptr + 0 * 64 + t * 8), aie::load_v<8>(b_ptr + 1 * 64 + t * 8), \
        aie::load_v<8>(b_ptr + 2 * 64 + t * 8), aie::load_v<8>(b_ptr + 3 * 64 + t * 8), \
        aie::load_v<8>(b_ptr + 4 * 64 + t * 8), aie::load_v<8>(b_ptr + 5 * 64 + t * 8), \
        aie::load_v<8>(b_ptr + 6 * 64 + t * 8), aie::load_v<8>(b_ptr + 7 * 64 + t * 8)); \
    acc0 = aie::mac(acc0, a_s[0][t], bv); \
    acc1 = aie::mac(acc1, a_s[1][t], bv); }
    MAC_M8_PARTIAL(0) MAC_M8_PARTIAL(1) MAC_M8_PARTIAL(2) MAC_M8_PARTIAL(3)
    MAC_M8_PARTIAL(4) MAC_M8_PARTIAL(5) MAC_M8_PARTIAL(6) MAC_M8_PARTIAL(7)
#undef MAC_M8_PARTIAL
    b_ptr += 512;
    a_ptr0 += 8;
    a_ptr1 += 8;
  }
  const aie::vector<int32, 64> acci0 = acc0.template to_vector<int32>();
  const aie::vector<int32, 64> acci1 = acc1.template to_vector<int32>();
  if (!kh) {
    aie::store_v(m8_acc_buf + m * 64, acci0);
    aie::store_v(m8_acc_buf + (m + 1) * 64, acci1);
    return;
  }
  const bfloat16 *pS = reinterpret_cast<const bfloat16 *>(b + N * 128 / 2);
  const aie::vector<bfloat16, 64> sv = aie::load_v<64>(pS);
  const aie::vector<bfloat16, 64> accb0 = aie::to_float<bfloat16>(acci0);
  aie::accum<accfloat, 64> prod0 = aie::mul(accb0, sv);
  aie::accum<accfloat, 64> cacc0;
  cacc0.from_vector(aie::load_v<64>(c0));
  aie::store_v(c0, aie::add(cacc0, prod0).template to_vector<bfloat16>());
  const aie::vector<bfloat16, 64> accb1 = aie::to_float<bfloat16>(acci1);
  aie::accum<accfloat, 64> prod1 = aie::mul(accb1, sv);
  aie::accum<accfloat, 64> cacc1;
  cacc1.from_vector(aie::load_v<64>(c1));
  aie::store_v(c1, aie::add(cacc1, prod1).template to_vector<bfloat16>());
}

static void vecmat_w4a16int_m8(const int16_t *a, const int8_t *b, bfloat16 *c) {
  constexpr unsigned N = 64;
  const int8_t *pW = b;
  const int8_t *pZ = b + N * 128 / 2 + N * 2;
  for (unsigned kh = 0; kh < 2; kh++) {
    for (unsigned i = 0; i < 8; ++i) {
      for (unsigned j = 0; j < 8; ++j) {
        const aie::vector<int8, 16> Bzp_row = aie::load_v<16>(pZ + j * 16);
        const aie::vector<int8, 64> Bzp = Bzp_row.template grow_replicate<64>();
        const aie::vector<int8, 32> B_packed =
            aie::load_v<32>(pW + ((kh * 8 + i) * 8 + j) * 32);
        const aie::vector<uint4, 64> B_uint4 = B_packed.template cast_to<uint4>();
        const aie::vector<uint8, 64> B_uint8 = B_uint4.template unpack();
        const aie::vector<int8, 64> B_int8 = B_uint8.template cast_to<int8>();
        const aie::vector<int8, 64> B_z = aie::sub(B_int8, Bzp);
        aie::store_v(b_buf_i16 + (i * 8 + j) * 64, B_z.template unpack_sign<int16>(true));
      }
    }
    for (int m = 0; m < 8; m += 2)
      vecmat_w4a16int_m8_partial(a + m * 128, a + (m + 1) * 128, b,
                                 c + m * 64, c + (m + 1) * 64, m, kh);
  }
}
#endif

extern "C" {

void vecmat_vectorized_bf16_bf16(bfloat16 *a_in, int8 *b_in, bfloat16 *c_out) {
  vecmat_w4a16int(reinterpret_cast<const int16_t *>(a_in), b_in, c_out);
}

void zero_vectorized_bf16(bfloat16 *c_out) { zero_vectorized<bfloat16, 64, 1>(c_out); }

#ifdef DIM_M
void vecmat_vectorized_bf16_bf16_m(bfloat16 *a_in, int8 *b_in, bfloat16 *c_out) {
  const int16_t* q = reinterpret_cast<const int16_t *>(a_in);
#if DIM_M == 2
  vecmat_w4a16int_m2(q, q + 128, b_in, c_out, c_out + 64);
#elif DIM_M == 8
  vecmat_w4a16int_m8(q, b_in, c_out);
#elif DIM_M == 4
  for (int m = 0; m < DIM_M; m += 2)
    vecmat_w4a16int_m2_isolated(q + m * 128, q + (m + 1) * 128, b_in,
                                c_out + m * 64, c_out + (m + 1) * 64);
#else
  for (int m = 0; m < DIM_M; m++) vecmat_w4a16int(q + m * 128, b_in, c_out + m * 64);
#endif
}

void zero_vectorized_bf16_m(bfloat16 *c_out) {
  for (int m = 0; m < DIM_M; m++) zero_vectorized<bfloat16, 64, 1>(c_out + m * 64);
}
#endif

} // extern "C"
