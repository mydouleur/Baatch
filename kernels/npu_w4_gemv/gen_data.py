#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""NPU W4A16 GEMV 测试数据（任务 0a，TileFuse 布局移植）。

GEMV: C[N] = A[K] @ B[K,N]，B 为 int4（+bf16 scale + int8 zp，group_size=128 沿 K）。

输出（--out dir）：
  a_bf16.bin  : A[K]，bf16 bits（uint16）
  b_packed.bin: B 预平铺打包流——大瓦片 128(K)x64(N) = 4096B 权重 + 128B scale + 128B zp，
                顺序 col_mod 主序（lt % 8, lt // 8）（与 vector_matrix.py 的 DMA 步长一致）
  c_ref.bin   : 参考 C[N] fp32（fp64 累加；A 用 bf16 舍入值，与 NPU 消费一致）
"""
import argparse, os
import numpy as np

GS = 128          # 量化 group 沿 K
LT_K, LT_N = 128, 64   # 大瓦片
NCOL = 8

def f32_to_bf16_u16(x):
    u = np.asarray(x, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)

def bf16_u16_to_f32(u):
    return (u.astype(np.uint32) << 16).view(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_w4")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--size", type=int, default=4096, help="K=N（须与设计编译尺寸一致）")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    K = N = a.size

    rng = np.random.default_rng(a.seed)
    A = (rng.random(K) * 0.19).astype(np.float32)
    A_bf = f32_to_bf16_u16(A)                      # NPU 实际消费的 bf16
    A_f = bf16_u16_to_f32(A_bf).astype(np.float64)

    W = rng.integers(0, 16, size=(K, N), dtype=np.uint8)          # [K,N] 4-bit
    ng = K // GS
    S = (rng.random((ng, N)) * 0.05).astype(np.float32)           # scale
    S_bf = f32_to_bf16_u16(S)
    S_f = bf16_u16_to_f32(S_bf)
    Z = rng.integers(0, 16, size=(ng, N), dtype=np.uint8)         # zero point

    # ---- 参考：C = A @ dequant(B)，fp64 ----
    Wdeq = ((W.astype(np.int16) - np.repeat(Z, GS, axis=0).astype(np.int16))
            * np.repeat(S_f, GS, axis=0)).astype(np.float64)      # [K,N]
    C = (A_f[:, None] * Wdeq).sum(axis=0)
    C.astype(np.float32).tofile(os.path.join(a.out, "c_ref.bin"))

    # ---- 打包 B（移植 TileFuse gemv_w4a16/test.cpp）----
    # 大瓦片 = 128(K)x64(N) = 4096B 权重 + 128B scale + 128B zp = 4352B
    # 顺序：col_mod(=lt_col%8) → lt_col → ltr（与 vector_matrix.py DMA 步长一致）
    nlt_col = N // LT_N
    nlt = (K // LT_K) * nlt_col
    LT_BYTES = LT_K * LT_N // 2 + LT_N * 2 + LT_N * 2   # 4352
    B = np.zeros((nlt, LT_BYTES), dtype=np.uint8)
    idx = 0
    for cm in range(NCOL):
        for lt_col in range(cm, nlt_col, NCOL):
            for ltr in range(K // LT_K):
                blk = W[ltr*LT_K:(ltr+1)*LT_K, lt_col*LT_N:(lt_col+1)*LT_N]
                blk = blk.reshape(16, 8, 8, 8).transpose(0, 2, 1, 3)  # [str,stc,i,j]
                pairs = blk.reshape(16, 8, 8, 4, 2)
                B[idx, :4096] = (pairs[..., 0] | (pairs[..., 1] << 4)).reshape(-1)
                # scales
                s16 = S_bf[ltr, lt_col*LT_N:(lt_col+1)*LT_N]            # 64 × uint16
                B[idx, 4096:4096+128] = s16.view(np.uint8)
                # zps：byte[group*16 + repeat*8 + i] = zp[group*8+i]
                z = Z[ltr, lt_col*LT_N:(lt_col+1)*LT_N].reshape(8, 8)   # [group, i]
                zr = np.concatenate([z, z], axis=1)                     # [group, repeat*8+i]
                B[idx, 4224:4352] = zr.reshape(-1)
                idx += 1

    A_bf.tofile(os.path.join(a.out, "a_bf16.bin"))
    B.reshape(-1).tofile(os.path.join(a.out, "b_packed.bin"))
    print(f"K={K} N={N} -> {a.out}/  B={B.size}B ({B.size/1048576:.1f}MB) tiles={idx}")

if __name__ == "__main__":
    main()
