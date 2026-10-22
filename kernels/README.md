# kernels/ — XDNA2 AIE kernel（M2' 主战场）

## 路线（M0 探针实证）

Windows 上没有 mlir-aie 编译链；可行路径是 **ctrlcode txn 合成**：
- `probes/bandwidth/gen_elf.py` 已能合成与 AMD stock 逐字节等价的控制流 ELF
- host 侧经 XRT elf-flow 加载运行（`probes/bandwidth/host_df_bw.cpp`）
- AIE 核心 C 代码用 Peano 编译（`llvm_aie` wheel 自带 clang.exe，Windows 原生）

## 开发顺序（docs/M2_ARCH.md §6）

1. `xbf3_gemv` —— XBF3 专家 GEMV 流式 kernel（decode 生命线，目标 ≥55 GB/s 权重流）
2. `gdn_decode` —— 线性注意力状态更新融合 kernel（纯元素级递推，无 matmul）
3. `xbf3_gemm_small` —— MTP verify 步小 batch GEMM
4. `conv1d_k4`、`norm/rope/glue` 小 kernel

每个 kernel 一个子目录：`.c`（核心代码）+ ctrlcode 生成脚本 + host 测试。
