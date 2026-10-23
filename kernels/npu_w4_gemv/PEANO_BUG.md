# Peano (llvm-aie nightly) aie2p 数值 bug issue 草稿

> 先不发，待审。目标仓库：Xilinx/llvm-aie。

## Title

[aie2p] aie::accumulate variadic (8-vector) bf16 GEMV produces value-dependent wrong lanes under Peano

## Environment

- llvm-aie nightly 22.0.0.2026102001+4f5df13d (also reproduces with the aie_api shipped in mlir_aie 1.3.4 and 1.4.4 wheels — i.e. not an aie_api header issue)
- Host: Ryzen AI 9 H 365 (XDNA2 / npu2), mlir_aie 1.3.4 driver flow, Windows XRT
- Compile: `clang++ --target=aie2p-none-unknown-elf -O2 -std=c++20`

## Symptom

TileFuse's W4A16 GEMV microkernel (`vm_mix_int4_64x8.cc`, aie_kernels/aie2) computes wrong
results under Peano for specific bf16 values. The same kernel passes with xchesscc
(per TileFuse paper artifact). Dequant chain (vldb.unpack/vsub.8/vups.4x/magic-add/
vconv/vst.conv) proven correct by staged dumps — only the `aie::accumulate<64>(acc, a_vec, 0, b0..b7)`
call is suspect.

Failing value class: |v| ∈ {5, 7} (bf16 mantissa with the 1/4 bit set). Examples (A ≡ 1.0,
b ≡ constant v, K=4096 reduction, expected = v×4096):

| v | got | expected |
|---|---|---|
| +5 | 19712 | 20480 |
| +7 | 26880 | 28672 |
| −5 | −21248 | −20480 |
| −7 | −30464 | −28672 |
| ±{0,1,2,3,4,6,8}, 12 | exact | — |

The error localizes to k ≡ 13 (mod 16): that lane's contribution is replaced by a
value-dependent transform X(v): X(5)=2, X(7)=0, X(−5)=−8, X(−7)=−14.

## Minimal repro (to attach)

[kernels/npu_w4_gemv/dequant_dbg.cc + dequant_dbg.py in our repo — staged dump proving
dequant path clean; the vm kernel's MAC loop is the failing construct]

## Workaround

Replacing the variadic `aie::accumulate` with 8 × `aie::mac(acc, scalar_a, b_vec)`
(scalar element × vector) — numerically correct.

## Notes

- Related: llvm-aie#995 (bfp conversion chain crashes)
- mlir-aie#3690 (xchesscc/Peano divergence context)
