#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""NPU W4A16-int GEMV 测试数据（rai_explore int8/int16 MAC 路径）。

与 gen_data.py 的差异：A 量化为 int16 车道里的整数值（[-128,127]），
按 int16 小端直接写入 a_bf16.bin 每 tile 256B 槽（128 x int16 恰好填满）。
B 打包格式不变。C 参考 = 整数精确累加后逐 tile 经 bf16 输出链（fp32 模拟）。

模式：w1 / w1z8 / const:w,z / akt:t / apulse:k / arows:v0,v1 / a1 /
wn / wk / sn / sk / zn / zk（A 改整数；M>1 A 按 tile,m,k 打包）
"""
import argparse, os
import numpy as np

GS = 128
LT_K, LT_N = 128, 64
NCOL = 8

def f32_to_bf16_u16(x):
    u = np.asarray(x, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)

def bf16_u16_to_f32(u):
    return (u.astype(np.uint32) << 16).view(np.float32)

def f32_to_bf16_floor_f32(x):
    """bf16 截断（向 -inf）舍入——与硬件 accfloat->bf16 srs 路径一致（实测确认）。"""
    x32 = np.asarray(x, dtype=np.float32)
    u = x32.view(np.uint32)
    b = u >> 16
    rem = u & 0xFFFF
    b = np.where(rem != 0, np.where(x32 < 0, b + 1, b), b)
    return (b.astype(np.uint32) << 16).view(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_i8")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--size", type=int, default=4096)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--m", type=int, default=1)
    ap.add_argument("--pattern", default=None)
    ap.add_argument("--independent-rows", action="store_true")
    ap.add_argument("--activation", help="reuse packed A from another generated dataset")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    K = a.size
    N = a.n if a.n is not None else a.size
    M = a.m

    rng = np.random.default_rng(a.seed)
    A = rng.integers(-128, 128, size=K).astype(np.int16)

    W = rng.integers(0, 16, size=(K, N), dtype=np.uint8)
    ng = K // GS
    S = (rng.random((ng, N)) * 0.05).astype(np.float32)
    S_bf = f32_to_bf16_u16(S)
    S_f = bf16_u16_to_f32(S_bf).astype(np.float64)
    Z = rng.integers(0, 16, size=(ng, N), dtype=np.uint8)

    if a.pattern and a.pattern == "wn":
        # W[k][n] = n%16：验证 n 映射/解包车道顺序
        A = np.ones(K, dtype=np.int16)
        W = (np.arange(N) % 16).astype(np.uint8)[None, :].repeat(K, axis=0)
        S = np.ones((ng, N), dtype=np.float32); S_bf = f32_to_bf16_u16(S); S_f = S
        Z = np.zeros((ng, N), dtype=np.uint8)
    elif a.pattern and a.pattern == "wk":
        # W[k][n] = k%16：验证 k 行序/解包
        A = np.ones(K, dtype=np.int16)
        W = (np.arange(K) % 16).astype(np.uint8)[:, None].repeat(N, axis=1)
        S = np.ones((ng, N), dtype=np.float32); S_bf = f32_to_bf16_u16(S); S_f = S
        Z = np.zeros((ng, N), dtype=np.uint8)
    elif a.pattern and a.pattern == "zn":
        # Z[n] = n%16, W=8：验证 zp 车道映射
        A = np.ones(K, dtype=np.int16)
        W = np.full((K, N), 8, dtype=np.uint8)
        S = np.ones((ng, N), dtype=np.float32); S_bf = f32_to_bf16_u16(S); S_f = S
        Z = (np.arange(N) % 16).astype(np.uint8)[None, :].repeat(ng, axis=0)
    elif a.pattern and a.pattern == "zk":
        A = np.ones(K, dtype=np.int16)
        W = np.full((K, N), 8, dtype=np.uint8)
        S = np.ones((ng, N), dtype=np.float32); S_bf = f32_to_bf16_u16(S); S_f = S
        Z = (np.arange(ng) % 16).astype(np.uint8)[:, None].repeat(N, axis=1)
    elif a.pattern and a.pattern == "sk":
        A = np.ones(K, dtype=np.int16)
        W = np.full((K, N), 5, dtype=np.uint8)
        S = (1 + np.arange(ng) % 3).astype(np.float32)[:, None].repeat(N, axis=1)
        S_bf = f32_to_bf16_u16(S); S_f = bf16_u16_to_f32(S_bf)
        Z = np.ones((ng, N), dtype=np.uint8)
    elif a.pattern and a.pattern == "sa":
        # A[k] = k%5-2（带符号）：验证激活符号/车道
        A = ((np.arange(K) % 5) - 2).astype(np.int16)
        W = np.full((K, N), 3, dtype=np.uint8)
        S = np.ones((ng, N), dtype=np.float32); S_bf = f32_to_bf16_u16(S); S_f = S
        Z = np.ones((ng, N), dtype=np.uint8)
    elif a.pattern and a.pattern == "sn":
        # S[g][n] = 1+n%3：验证 scale 车道映射
        A = np.ones(K, dtype=np.int16)
        W = np.full((K, N), 5, dtype=np.uint8)
        S = (1 + np.arange(N) % 3).astype(np.float32)[None, :].repeat(ng, axis=0)
        S_bf = f32_to_bf16_u16(S); S_f = bf16_u16_to_f32(S_bf)
        Z = np.ones((ng, N), dtype=np.uint8)
    elif a.pattern and a.pattern.startswith("ap1"):
        # 脉冲 + W=1,Z=0,S=1：acc=1，直接测输出链增益
        k0 = int(a.pattern.split(":")[1])
        A = np.zeros(K, dtype=np.int16); A[k0] = 1
        W = np.ones((K, N), dtype=np.uint8)
        S = np.ones((ng, N), dtype=np.float32); S_bf = f32_to_bf16_u16(S); S_f = S
        Z = np.full((ng, N), 0, dtype=np.uint8)
    elif a.pattern and a.pattern.startswith("apulse"):
        k0 = int(a.pattern.split(":")[1])
        A = np.zeros(K, dtype=np.int16); A[k0] = 1
        W = np.ones((K, N), dtype=np.uint8)
        S = np.ones((ng, N), dtype=np.float32); S_bf = f32_to_bf16_u16(S); S_f = S
        Z = np.full((ng, N), 8, dtype=np.uint8)
    elif a.pattern and a.pattern.startswith("akt"):
        t = int(a.pattern.split(":")[1])
        A = np.zeros(K, dtype=np.int16); A[128*t:128*(t+1)] = 1
        W = np.ones((K, N), dtype=np.uint8)
        S = np.ones((ng, N), dtype=np.float32); S_bf = f32_to_bf16_u16(S); S_f = S
        Z = np.full((ng, N), 8, dtype=np.uint8)
    elif a.pattern and a.pattern.startswith("arows"):
        A = np.ones(K, dtype=np.int16)
        W = np.ones((K, N), dtype=np.uint8)
        S = np.ones((ng, N), dtype=np.float32); S_bf = f32_to_bf16_u16(S); S_f = S
        Z = np.full((ng, N), 8, dtype=np.uint8)
    elif a.pattern and a.pattern.startswith("const"):
        wv, zv = (int(x) for x in a.pattern.split(":")[1].split(","))
        A = np.ones(K, dtype=np.int16)
        W = np.full((K, N), wv, dtype=np.uint8)
        S = np.ones((ng, N), dtype=np.float32); S_bf = f32_to_bf16_u16(S); S_f = S
        Z = np.full((ng, N), zv, dtype=np.uint8)
    elif a.pattern == "a1":
        A = np.ones(K, dtype=np.int16)
    elif a.pattern == "w1":
        A = np.ones(K, dtype=np.int16)
        W = np.ones((K, N), dtype=np.uint8)
        S = np.ones((ng, N), dtype=np.float32); S_bf = f32_to_bf16_u16(S); S_f = S
        Z = np.zeros((ng, N), dtype=np.uint8)
    elif a.pattern == "w1z8":
        A = np.ones(K, dtype=np.int16)
        W = np.ones((K, N), dtype=np.uint8)
        S = np.ones((ng, N), dtype=np.float32); S_bf = f32_to_bf16_u16(S); S_f = S
        Z = np.full((ng, N), 8, dtype=np.uint8)

    # ---- 参考：逐 tile 整数精确 acc，再按 kernel 输出链（bf16 量化 acc -> x S -> 累加）----
    # kernel: prod = bf16(acc) * S (accfloat), c = bf16(c + prod)
    logical_N = N
    physical_N = (N + 2047) // 2048 * 2048
    if physical_N != N:
        W = np.pad(W, ((0, 0), (0, physical_N - N)), constant_values=8)
        S = np.pad(S, ((0, 0), (0, physical_N - N)), constant_values=0)
        S_bf = f32_to_bf16_u16(S); S_f = bf16_u16_to_f32(S_bf)
        Z = np.pad(Z, ((0, 0), (0, physical_N - N)), constant_values=8)
        N = physical_N

    if a.independent_rows:
        if a.pattern:
            ap.error("--independent-rows cannot be combined with --pattern")
        A = rng.integers(-128, 128, size=(M, K)).astype(np.int16)
    else:
        A = np.repeat(A[None, :], M, axis=0)
    if a.pattern and a.pattern.startswith("arows"):
        values = [int(x) for x in a.pattern.split(":")[1].split(",")]
        if len(values) != M:
            ap.error(f"arows requires {M} comma-separated values")
        for m, value in enumerate(values):
            A[m, :] = value
    if a.activation:
        packed = np.fromfile(a.activation, dtype=np.int16)
        if packed.size != M * K:
            ap.error(f"--activation has {packed.size} values, expected {M * K}")
        A = packed.reshape(K // 128, M, 128).transpose(1, 0, 2).reshape(M, K).copy()
    C = np.zeros((M, N), dtype=np.float64)
    for m in range(M):
        for t in range(K // GS):
            acc = (A[m, t*GS:(t+1)*GS].astype(np.int64)[:, None] *
                   (W[t*GS:(t+1)*GS].astype(np.int64) - Z[t].astype(np.int64)[None, :])).sum(axis=0)
            acc_bf = bf16_u16_to_f32(f32_to_bf16_u16(acc.astype(np.float32)))
            prod = acc_bf.astype(np.float64) * S_f[t]
            C[m] = bf16_u16_to_f32(f32_to_bf16_u16((C[m] + prod).astype(np.float32))).astype(np.float64)
    if M > 1:
        native = np.empty(M * N, dtype=np.float64)
        for tr in range(N // 2048):
            for col in range(8):
                for row in range(4):
                    dst = (((tr * 8 + col) * 4 + row) * M) * 64
                    src = tr * 2048 + row * 512 + col * 64
                    native[dst:dst + M * 64] = C[:, src:src + 64].reshape(-1)
        native.astype(np.float32).tofile(os.path.join(a.out, "c_ref.bin"))
    else:
        C.astype(np.float32).tofile(os.path.join(a.out, "c_ref.bin"))

    # ---- A 打包：每 128-tile 256B = 128 x int16 ----
    if M > 1:
        A = A.reshape(M, K // 128, 128).transpose(1, 0, 2)
    A.tofile(os.path.join(a.out, "a_bf16.bin"))  # int16 小端，文件名兼容 host

    # ---- B 打包（与 gen_data.py 完全一致）----
    nlt_col = N // LT_N
    nlt = (K // LT_K) * nlt_col
    LT_BYTES = LT_K * LT_N // 2 + LT_N * 2 + LT_N * 2
    B = np.zeros((nlt, LT_BYTES), dtype=np.uint8)
    idx = 0
    for cm in range(NCOL):
        for lt_col in range(cm, nlt_col, NCOL):
            for ltr in range(K // LT_K):
                blk = W[ltr*LT_K:(ltr+1)*LT_K, lt_col*LT_N:(lt_col+1)*LT_N]
                blk = blk.reshape(16, 8, 8, 8).transpose(0, 2, 1, 3)
                pairs = blk.reshape(16, 8, 8, 4, 2)
                B[idx, :4096] = (pairs[..., 0] | (pairs[..., 1] << 4)).reshape(-1)
                s16 = S_bf[ltr, lt_col*LT_N:(lt_col+1)*LT_N]
                B[idx, 4096:4096+128] = s16.view(np.uint8)
                z = Z[ltr, lt_col*LT_N:(lt_col+1)*LT_N].reshape(8, 8)
                zr = np.concatenate([z, z], axis=1)
                B[idx, 4224:4352] = zr.reshape(-1)
                idx += 1

    B.reshape(-1).tofile(os.path.join(a.out, "b_packed.bin"))
    print(f"K={K} N={N} -> {a.out}/  B={B.size}B ({B.size/1048576:.1f}MB) tiles={idx}")

if __name__ == "__main__":
    main()
