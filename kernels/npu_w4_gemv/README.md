# npu_w4_gemv — NPU W4A16 GEMV 带宽实测（任务 0a，decode 配平拼图）

W4（4-bit，车道对齐）权重流式 GEMV，验收线：真实计算的权重流带宽 **≥55 GB/s**。

## 结论（2026-10-23，任务 0a 收官）

**未达标，数值未闭合。** 结构/管线/数据映射全部验证正确，但 Peano（llvm-aie nightly
2026-10-20）对 TileFuse W4 反量化链的编译存在**值相关数值 bug**（详见下方取证），
随机数据 verify FAIL。带宽侧实测（真实计算路径、数值有 bug 的内核）：

| 形状 | B 大小 | per-run min/avg/max | 权重流 GB/s（min/avg/max） |
|---|---|---|---|
| K=N=4096 | 8.5MB | 0.419/0.419/0.419 ms | 21.3/21.3/21.3 |
| K=N=8192 | 34.0MB（>L3 24MB，真 DRAM 流） | 1.207/1.210/1.216 ms | 29.3/29.5/29.6 |

即使数值修好，当前 Peano 管线形态也只有 ~30 GB/s 量级（每列 ~3.7 GB/s），
距 55 还差结构与计算两重优化。**配平结论：NPU W4 侧不成立，decode 配平暂时只有
CPU 2 大核（39 GB/s，M3'-1 vnni2 已实测）一条腿。**

## 结构（TileFuse gemv_w4a16 全优化版移植：microkernel + dataflow）

- 设计 `vector_matrix.py`：8 列 × 4 行 = 32 核；shim→memtile→core 两级 ObjectFifo；
  memtile 把 4 瓦片 bundle 分发到 4 计算行；A（bf16 激活）memtile 广播；
  runtime_sequence 用 `npu_dma_memcpy_nd` + 显式 bd_id 双 BD 段。
- 微内核 `vm_mix_int4_64x8.cc`（TileFuse aie_kernels/aie2，Apache-2.0）：每核每次调用
  64 输出 × 128 约减：int4 unpack → 减 zp → bf16 转换 → 乘 scale → b_buf →
  `aie::accumulate<64>` VMAC。**Peano 移植改动**：(1) chess/event 提示符剥离；
  (2) b_buf 半 k 化（16KB→8KB 两个 k 半趟）——原版在 Peano 链接布局下核内 DM
  溢出 64-256B（ fifo depth 全降 1 仍差 64B，根解靠 b_buf 减半，fifo depth 回 2）。
- 布局（gen_data.py 移植自 TileFuse test.cpp，round-trip 校验全过）：大瓦片
  128(K)×64(N) = 4096B 权重 + 128B bf16 scale + 128B int8 zp（×2 重复），
  col_mod 主序交错预平铺；小瓦片 8×8，字节 = lo|hi<<4。

## 数值 bug 取证（值相关，Peano codegen 嫌疑）

恒值扫描（A=1, scale=1，4096 输出全同值；期望 = (w−z)×4096）：

| (w,z) | v=w−z | 实测 | 判定 |
|---|---|---|---|
| (1,0) | +1 | 4096 ✓ | PASS |
| (2,1),(1,4),(3,0),(2,8),(0,8),(1,7),(1,9),(8,8),(9,8)… | 多值 | 全 ✓ | PASS |
| (1,8)/(2,9)/(3,10)/(0,7)/(7,14) | **−7** | −30464（exp −28672） | FAIL |
| (5,0)/(13,8) | +5 | 19712（exp 20480） | FAIL |
| (8,1) | +7 | 26880（exp 28672） | FAIL |

- **触发集 = |v| ∈ {5, 7}**（bf16 尾数含 1/4 位的值），其余值（含 ±8、±6、0..4）全对。
- w=7,z=0 即触发 → 与 zp 减法无关；失败跟在反量化后**数值**走，与字节模式无关。
- 列奇偶分离测试（(wlo,whi)=(5,0)/(0,5)）：输出按奇偶干净分裂 → n 映射/分发无误。
- k 梳齿测试（A 仅 k≡r (mod 16) 为 1）：k%16==13 的位置贡献异常（v=7 时 X=0、
  v=5 时 X=2、v=−5 时 X=−8、v=−7 时 X=−14），其余位置也有细微交互（非线性）。
- A 路径干净（z=0 梳齿全对）；B 大瓦片/小瓦片打包 round-trip 全对（numpy 反解比对）。
- X(v) 无简单闭式；v=12（0b1100）全对而 13/14/15 错——疑似 unpack 或 int8→bf16
  转换在特定车道/值上的 lowering 错误。

反汇编（本机 llvm-objdump，见下）：反量化链为
`vldb.unpack → vsub.8(−zp) → vups.4x（int8→fp32 累加器）→ vadd/vsub.f（magic 常数）
→ vconv.bf16.fp32 → vmul.f(×scale) → vst.conv.bf16.fp32`；MAC 循环大量用
`vshift` 链重组 b 行 + `vmac.f`。疑点最集中在 unpack/转换序列（值相关只在
转换类指令里可能出现）。

### 已排除

- aie_api 版本：1.3.4 与 1.4.4 头文件编译结果**同样错**（同一 Peano nightly）。
- 我的 kh 半趟改动：值无关结构性改动，且错误对称分布于两个半趟。
- TileFuse 布局/host 打包：round-trip 全对；恒值图样系统隔离到位。
- DMA/分发/输出映射：恒值与奇偶图样证明正确。
- `aie::accumulate` 变长形：换成逐行 broadcast+mac 后输出全零（broadcast 标量用法
  在 Peano 下另有问题），不能作为对照，已回退。

### 根因判断

Peano（llvm-aie 22.0.0.2026102001）对反量化链中某条指令（`vldb.unpack` /
`vups.4x` / `vst.conv` 序列）的 aie2p lowering 存在值相关错误。TileFuse 官方流程
用 xchesscc（Vitis），未走 Peano，故其论文数字不受影响。**要闭合需：写一个
独立最小复现（单核单瓦片）报给 llvm-aie，或改用 xchesscc 编 kernel。**

## 工具链资产（本轮新增）

- **本机 Windows 全链**：mlir_aie 1.4.4 win wheel + llvm-aie nightly win wheel
  （`/tmp/aievenv`）：kernel 本地 Peano 编译 + llvm-objdump 反汇编可用
  （本轮取证即靠它），免 20 分钟 CI 往返做 kernel 级调试。
- CI（kernels.yml）：npu_w4_gemv 步骤 = 1.3.4 aiecc 驱动 + 1.4.4 aie_api 头文件
  混合（1.4.4 wheel 无 python aiecc.py，bin/aiecc 是原生工具 CLI 不兼容；
  link_with 位置已做版本自适应：1.3.4 在 external_func 上，main 时代在 core 上）。
- mlir_aie wheel 实际版本坑：`latest-wheels-4` 提供的是 **1.3.4**；新版在
  发布页 v1.4.4+。1.4.4 起 taplib 无 TensorAccessSequence。

## 构建/运行

```
python gen_data.py --out data_w4 [--size 4096|8192] [--pattern w1|w1z8|const:w,z|akt:t|apulse:k]
# CI 产 build_*/{insts.bin,design.pdi} + work_*/main_*.json；本机：
bash ../xbf3_gemv/pack_xclbin.sh work_4096 build_4096 w4_4096.xclbin   # （在 art 目录内）
cl ... test_host.cpp ... -Fe:test_host.exe   # XRT 头文件/lib 见 m0/probe
test_host.exe build_4096/w4_4096.xclbin build_4096/insts.bin data_w4 20
```

Host ABI：arg0=opcode 3，arg1=instr BO，arg2=instr 字数，arg3=A，arg4=B，arg5=C（bf16）。
