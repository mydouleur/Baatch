//===- vm_mix_int4_64x8.cc --------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
// Copyright (C) 2025, University of Illinois Urbana-Champaign.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#define REL_WRITE 0
#define REL_READ 1

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>

#include "zero.cc"

template <typename T_in_A, typename T_in_B, typename T_out, int N, int K>
void vecmat_scalar(T_in_A *a, T_in_B *b, T_out *c) {
  
  // GEVM: a is vector (K,), b is matrix (K, N), c is vector (N,)
  // M=1 (always), N is output dimension, K is reduction dimension
  // Note: For mixed precision, b should already be dequantized to T_in_A type
  for (int col = 0; col < N; col++) {
    T_out runningSum = 0;
    for (int i = 0; i < K; i++) {
      runningSum += a[i] * b[i * N + col];
    }
    c[col] += runningSum;
  }
  
}

alignas(aie::vector_decl_align) static bfloat16 b_buf[64 * 64];  // Peano 移植：半 k 缓冲（DM 预算）

template <typename T_in_A, typename T_in_B, typename T_out, typename T_acc, unsigned n, unsigned k,
          unsigned r, unsigned s>
void vecmat_vectorized(T_in_A *__restrict a, T_in_B *__restrict b,
                       T_out *__restrict c) {
  static_assert(n % r == 0 && k % 2 == 0);
  static_assert(s == 8); // s is fixed to 8 for vectorization
  static_assert(k % s == 0);
  static_assert(std::is_same<T_in_A, bfloat16>::value ||
                std::is_same<T_in_A, int16_t>::value);

  // Pre-process B matrix: dequantize int8 values to bf16 and store in b_buf
  const T_in_B *__restrict pB_quantized = b;
  const T_in_B *__restrict pBs_b = b + n * k / 2;
  const T_in_B *__restrict pBzp_b = b + n * k / 2 + n * 2; // times 2 because bf16 scale factor is 2 bytes

  const unsigned colA = k / 8;      // k / 8 (number of row tiles)
  const unsigned colB = n / 8;      // n / 8 (number of column tiles)
  const unsigned size_B = 8 * 8;    // 64 (elements per small tile)
  const unsigned half_r = r / 2;    // 8
  const unsigned four_r = r * 4;    // 64
  const unsigned k_half = k / 2;    // 每个 k 半趟的行数
  const unsigned colA_half = colA / 2;

  // Peano 移植说明：原版 b_buf 持有全部 k×n 反量化值（16KB），Peano 链接布局下
  // 核内 DM 溢出 64B。改为两个 k 半趟：dequant 半程 → MAC 半程 → 再半程；
  // c 经内存往返累加（调用前已 zero），语义不变。
  for (unsigned kh = 0; kh < 2; kh++) {
    for (unsigned i = 0; i < colA_half; ++i) {
      for (unsigned j = 0; j < colB; ++j) {
        // Load scale factors
        const bfloat16 *pBs_bf16 = reinterpret_cast<const bfloat16*>(pBs_b);
        const aie::vector<bfloat16, 8> Bs_row = aie::load_v<8>(pBs_bf16 + j * 8);
        const aie::vector<T_in_A, size_B> Bs_replicated = Bs_row.template grow_replicate<size_B>();

        // Load zero point values
        const aie::vector<T_in_B, 16> Bzp_row = aie::load_v<16>(pBzp_b + j * 16);
        const aie::vector<int8, size_B> Bzp = Bzp_row.template grow_replicate<size_B>();

        const aie::vector<T_in_B, 32> B_quantized =
            aie::load_v<32>(pB_quantized + ((kh * colA_half + i) * colB + j) * size_B / 2);
        const aie::vector<uint4, size_B> B_uint4 = B_quantized.template cast_to<uint4>();
        const aie::vector<uint8, size_B> B_uint8 = B_uint4.template unpack();
        const aie::vector<int8, size_B> B_int8 = B_uint8.template cast_to<int8>();
        const aie::vector<int8, size_B> B_int8_zeroed = aie::sub(B_int8, Bzp);
        const aie::vector<bfloat16, size_B> B_dequantized = aie::to_float<bfloat16>(B_int8_zeroed);
        const auto B_scaled = aie::mul(B_dequantized, Bs_replicated);

        aie::store_v(b_buf + (i * colB + j) * size_B, B_scaled.template to_vector<T_in_A>());
      }
    }

    T_out *__restrict c_ptr = c;
    AIE_LOOP_MIN_ITERATION_COUNT(n / four_r)
    for (int col = 0; col < n; col += four_r) {
      aie::accum<T_acc, four_r> c_acc;
      c_acc.from_vector(aie::load_v<four_r>(c_ptr));

      T_in_A *__restrict a_ptr = a + kh * k_half;
      T_in_A *__restrict b_ptr = b_buf + col * s;

      // TileFuse 原版 accumulate 形（1.3.4 wheel 的 aie_api 在 aie2p 上对 ±5/±7
      // 有错——疑似旧版 unpack/accumulate lowering；CI 升级 1.4.4 后复测）
      for (int row = 0; row < k_half; row += s) {
#define BVEC(t) aie::concat(aie::load_v<half_r>(b_ptr + 0 * s * half_r + t * half_r), \
                            aie::load_v<half_r>(b_ptr + 1 * s * half_r + t * half_r), \
                            aie::load_v<half_r>(b_ptr + 2 * s * half_r + t * half_r), \
                            aie::load_v<half_r>(b_ptr + 3 * s * half_r + t * half_r), \
                            aie::load_v<half_r>(b_ptr + 4 * s * half_r + t * half_r), \
                            aie::load_v<half_r>(b_ptr + 5 * s * half_r + t * half_r), \
                            aie::load_v<half_r>(b_ptr + 6 * s * half_r + t * half_r), \
                            aie::load_v<half_r>(b_ptr + 7 * s * half_r + t * half_r))
        const aie::vector<T_in_A, four_r> b_vec_0 = BVEC(0);
        const aie::vector<T_in_A, four_r> b_vec_1 = BVEC(1);
        const aie::vector<T_in_A, four_r> b_vec_2 = BVEC(2);
        const aie::vector<T_in_A, four_r> b_vec_3 = BVEC(3);
        const aie::vector<T_in_A, four_r> b_vec_4 = BVEC(4);
        const aie::vector<T_in_A, four_r> b_vec_5 = BVEC(5);
        const aie::vector<T_in_A, four_r> b_vec_6 = BVEC(6);
        const aie::vector<T_in_A, four_r> b_vec_7 = BVEC(7);
#undef BVEC

        // 逐 lane 标量×向量 mac（aie::accumulate 变长形在 Peano 下对 |v|∈{5,7} 出错——
        // dequant_dbg 分阶段 dump 证明 unpack/sub/cvt/reload 全对，唯 accumulate 可疑；
        // 显式 8 行：三元参考链会被 Peano 编成跳表 j p2（上机挂起），必须展开）
        c_acc = aie::mac(c_acc, a_ptr[row + 0], b_vec_0);
        c_acc = aie::mac(c_acc, a_ptr[row + 1], b_vec_1);
        c_acc = aie::mac(c_acc, a_ptr[row + 2], b_vec_2);
        c_acc = aie::mac(c_acc, a_ptr[row + 3], b_vec_3);
        c_acc = aie::mac(c_acc, a_ptr[row + 4], b_vec_4);
        c_acc = aie::mac(c_acc, a_ptr[row + 5], b_vec_5);
        c_acc = aie::mac(c_acc, a_ptr[row + 6], b_vec_6);
        c_acc = aie::mac(c_acc, a_ptr[row + 7], b_vec_7);

        b_ptr += n * s;
        a_ptr += s;
      }

      aie::store_v(c_ptr, c_acc.template to_vector<T_out>());
      c_ptr += four_r;
    }
  }
}


extern "C" {

// If you want to compile microkernels with different inner tile sizes,
// define DIM_M and DIM_K at compile time using -DDIM_M 16 etc.
// For GEVM: DIM_M represents the output dimension tile size (n)
//           DIM_K represents the reduction dimension tile size (k)
// These dimensions must be divisible by the r, s dimensions used in
// the kernels.

#ifndef DIM_N
#define DIM_N 64  // Output dimension tile size (n)
#endif

#ifndef DIM_K
#define DIM_K 128  // Reduction dimension tile size (k)
#endif

#define combos(X)                                                              \
  X(bfloat16, bf16, int8, i8, bfloat16, bf16, accfloat, 16, 8)                    \
  //X(int16, i16, int16, i16, int32, i32, acc32, 8, 8)

#define vecmat_scalar_c_func(ctype_in_A, mlir_type_in_A, ctype_in_B,          \
                             mlir_type_in_B, ctype_out, mlir_type_out,         \
                             ctype_acc, r, s)                                 \
  void vecmat_scalar_##mlir_type_in_A##_##mlir_type_out(                      \
      ctype_in_A *a_in, ctype_in_B *b_in, ctype_out *c_out) {                 \
    vecmat_scalar<ctype_in_A, ctype_in_B, ctype_out, DIM_N, DIM_K>(a_in, b_in, c_out);    \
  }

#define vecmat_vectorized_c_func(ctype_in_A, mlir_type_in_A, ctype_in_B,      \
                                 mlir_type_in_B, ctype_out, mlir_type_out,     \
                                 ctype_acc, r, s)                             \
  void vecmat_vectorized_##mlir_type_in_A##_##mlir_type_out(                  \
      ctype_in_A *a_in, ctype_in_B *b_in, ctype_out *c_out) {                 \
    vecmat_vectorized<ctype_in_A, ctype_in_B, ctype_out, ctype_acc, DIM_N,    \
                      DIM_K, r, s>(a_in, b_in, c_out);                    \
  }

#define zero_vectorized_c_func(ctype_in_A, mlir_type_in_A, ctype_in_B,        \
                               mlir_type_in_B, ctype_out, mlir_type_out,       \
                               ctype_acc, r, s)                               \
  void zero_vectorized_##mlir_type_out(ctype_out *c_out) {                    \
    zero_vectorized<ctype_out, DIM_N, 1>(c_out);                              \
  }

#define zero_scalar_c_func(ctype_in_A, mlir_type_in_A, ctype_in_B,            \
                           mlir_type_in_B, ctype_out, mlir_type_out,           \
                           ctype_acc, r, s)                                   \
  void zero_scalar_##mlir_type_out(ctype_out *c_out) {                        \
    zero_scalar<ctype_out, DIM_N, 1>(c_out);                                  \
  }

combos(vecmat_scalar_c_func) combos(vecmat_vectorized_c_func)
    combos(zero_vectorized_c_func) combos(zero_scalar_c_func)

} // extern "C"