# kernels/ — XDNA2 AIE kernel（M2' 主战场）

## 路线（2026-10-22 定案）

**CI 编译 + 本机实测**：

- 编译：GitHub Actions（ubuntu + mlir-aie，Apache-2.0 工具链）→ xclbin/elf artifact
- 实测：本机 Windows + XRT host（M0 探针已验证 elf-flow 加载运行可行）
- kernel 100% 从自有源码编译，零 AMD 专有 blob——license 最干净路线

**已放弃的路线**（2026-10-22，证据见 m0/poc_archive/，工作区留存）：Windows 本地
ctrlcode txn 逆向合成——能跑通 shim DMA 但计算 tile 写入路径导致过一次系统级宕机
（内核态驱动被实验控制码挂死），时间成本与稳定性不可接受。

## 开发顺序（docs/M2_ARCH.md §6）

1. `xbf3_gemv` —— XBF3 专家 GEMV 流式 kernel（decode 生命线，目标 ≥55 GB/s 权重流）
2. `gdn_decode` —— 线性注意力状态更新融合 kernel（纯元素级递推，无 matmul）
3. `xbf3_gemm_small` —— MTP verify 步小 batch GEMM
4. `conv1d_k4`、`norm/rope/glue` 小 kernel

每个 kernel 一个子目录：核心 C 源码 + IRON/MLIR 设计脚本 + host 测试。
本机迭代闭环：`git push` → CI 编译 → `gh run download` 取 artifact → XRT 上机。

## 历史资产（可复用）

M0 探针（probes/bandwidth/）：host XRT 骨架（host_df_bw.cpp）、txn 格式解码器
（txn_decode.py，格式源自 Apache-2.0 的 XRT 公开源码，无法律问题）、
实测带宽数据（RESULTS.md）。
