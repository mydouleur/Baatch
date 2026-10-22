# npu_w4_gemv — NPU W4A16 GEMV 带宽实测（任务 0a，decode 配平拼图）

W4（4-bit，车道对齐）权重流式 GEMV，目标：真实计算的权重流带宽 **≥55 GB/s**
（达成则 "2 大核 CPU（39 GB/s 已实测）+ NPU" 配平成立，裸 decode ~56 tok/s）。

## 结构（TileFuse gemv_w4a16 全优化版移植：microkernel + dataflow）

- 设计 `vector_matrix.py`（来自 TileFuse fork，Apache-2.0）：8 列 × 4 行 = **32 核全开**；
  shim→memtile→core 两级 ObjectFifo；memtile 把一个 4 瓦片 bundle 分发到 4 个计算行；
  A（bf16 激活向量）经 memtile 广播；runtime_sequence 用 `npu_dma_memcpy_nd` +
  **显式 bd_id**（每列 2 个 BD 段 → 尝试占满 shim 双 MM2S 通道，绕开 M2' 实测的
  IRON rt.fill 单通道上限 37.6 GB/s）。
- 微内核 `vm_mix_int4_64x8.cc`（TileFuse aie_kernels/aie2，Peano 移植：chess/event
  提示符已剥离）：每核每次调用 64 输出 × 128 约减——int4 unpack（车道对齐，掩码即可）
  → 减 zp → bf16 转换 → 乘 scale → b_buf；再 `aie::accumulate<64>` VMAC
  （8 激活 × 64 输出/指令 = 512 MAC）。
- 布局（与 TileFuse test.cpp 一致，gen_data.py 移植）：大瓦片 128(K)×64(N) =
  4096B 权重 + 128B bf16 scale（每输出列 1 个）+ 128B int8 zp（重复 2 次），
  大瓦片按 col_mod 主序交错预平铺；小瓦片 8×8，字节 = lo|hi<<4。
- 形状：K=N=4096（B=8.5MB），每轮全量流式过片。

## 构建/运行

```
# CI（ubuntu + mlir_aie latest + llvm-aie nightly）产出 build/insts.bin + design.pdi
python gen_data.py --out data_w4          # 测试向量 + c_ref.bin（fp64 参考）
# 本机：pack_xclbin（同 xbf3_gemv 流程）后
cl ... test_host.cpp ... -Fe:test_host.exe
test_host.exe build/w4_gemv.xclbin build/insts.bin data_w4 40
```

## 实测记录

（待上机填写：min/avg/max GB/s + verify 结果）
