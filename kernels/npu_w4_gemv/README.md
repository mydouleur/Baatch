# npu_w4_gemv — NPU W4A16 GEMV 带宽实测（任务 0a，decode 配平拼图）

> 本文主体是 0a 历史记录。后续 W4A8 int16 VMAC 已数值闭合并进入引擎；当前分工以 `docs/DESIGN.md` §10 为准。M>1 prefill 仍未闭合，勿据早期结论判断当前状态。

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

### 已排除（dequant_dbg 分阶段 dump 硬件实证，w=5 图样）

- **unpack / sub zp / to_float+scale / b_buf 重组 reload：全部正确**——
  dequant_dbg 四级 dump（st1 unpack、st2 sub、st3 cvt+scale、st4 MAC 的 concat
  重组路径）与 numpy 参考逐 lane 一致。
- aie_api 版本：1.3.4 与 1.4.4 头文件编译结果同样错（同一 Peano nightly）。
- 我的 kh 半趟改动、TileFuse 布局、host 打包、DMA/分发/输出映射：全清白。

### 根因判断（收窄到 accumulate lowering）

罪域唯一剩下 `aie::accumulate<64>(acc, a_vec, 0, b0..b7)` 变长形的 Peano lowering
（反汇编见其编为共享 y1/y2 操作数的 4 条 vmac.f + vextbcst 车道提取链，值相关错误
只能来自这条链的融合乘加路径——±5/±7 恰为 bf16 尾数 1/4 位值，疑 VMAC 内部
bf16→bfp16 转换或部分积对齐的舍入模式错误）。

绕写尝试（均失败，勿重复）：
- 逐 lane `aie::mac(acc, 标量, b_vec)`：ternary 选 b_vec 被编成跳表 `j p2` → 上机挂起；
  显式 8 行展开后输出变为**与权重无关的常数 3584**（b 数据路径死——另一条
  Peano lowering 坑），未再深挖。
- 1.4.4 aie_api 头文件：同错。

### 收敛路径（留后续）

1. 给 llvm-aie 报 issue（草稿 PEANO_BUG.md，待发）——附 dequant_dbg 分阶段 dump 复现。
2. 或 xchesscc 编 kernel（mlir-aie#3690：xchesscc 编的 core 在 IRON raw-xrt 下有
   挂起前科，需先验证该 issue 状态）。
3. 或完全绕开 bf16 accumulate：b_buf 用 fp32 + fp32 VMAC / 或 int8 VMAC 路径重写。

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

## M4/M8 standalone prefill 原型（2026-10-26）

`test_prefill_batch.cpp` 在同一可执行文件内顺序运行 M1×M 与 batched M，二者使用相同的随机 A/B 和 CPU reference。A 按 `[k_tile,m,k]` 打包，M>1 的 C 按 `[transfer,col,row,m,n]` 验证。每次迭代逐值检查精确差异和 `rel>2e-2 && abs>5e-3` 容差；计时包含 run 创建、提交、等待和每次输出同步，不包含 CPU reference 比对。`gen_data_i8.py --independent-rows` 生成独立随机行。

```bash
bash build_prefill_host.sh
K=2048 N=12288 M=4 bash build_m2.sh
python gen_data_i8.py --out <data> --size 2048 --n 12288 --m 4 --independent-rows
# 每次上机都必须使用外部看门狗；路径按实际资产替换。
timeout --kill-after=5s 40s <work>/test_prefill_batch.exe \
  <m4.xclbin> <m4.insts> <m1.xclbin> <m1.insts> <data> 2048 12288 4 3 both
```

M4/M8 分支按行对复用 M2 微内核，每次权重 DMA 服务 M 行；`noinline` 隔离是必需的，内联两个行对会产生约 1KB 向量栈 spill 并挂起。M>=4 的 A 传输使用一个二维 ND BD 描述全部 k-tile；旧版 16 个独立 BD 会随机缺一个 k-tile。M1/M2 的 A 路径与 FIFO 深度保持原样。四累加器版本在 N2048/N12288 均 ERT timeout；A FIFO 深度 10/12/16 超过实际 BD 分配能力，5 个 A BD 轮转也未闭合，均未采用。

K2048/N12288 随机独立行与常数独立行各 20 轮均在 M4/M8 逐值 PASS。随后相同随机数据、同 host 二进制的 1000 轮性能对照每轮也全部逐值 PASS，暖态取 995 轮：

| 对照 | batch 平均 ms | 每 token ms | 物理权重字节 | 有效 GB/s | batch 相对 M1×M |
|---|---:|---:|---:|---:|---:|
| M1×4 | 1.764343 | 0.441086 | 53,477,376 | 30.310 | — |
| M4 | 0.713996 | 0.178499 | 13,369,344 | 18.725 | -59.532%，2.471× |
| M1×8 | 3.510150 | 0.438769 | 106,954,752 | 30.470 | — |
| M8 | 1.324756 | 0.165595 | 13,369,344 | 10.092 | -62.259%，2.650× |

M8 相对 M4 每 token 仅再降 7.229%，batch 时间为 1.855×；权重复用可扩展，但行对计算成本接近线性增长。这些是 standalone 投影结果，不是端到端 prefill tok/s。早期诊断共 3 次 timeout 后登记复位，每次复位后的 M1 均 PASS；ND 修复后的 M4/M8 与性能对照均 0 flake，默认引擎未接入这些资产。

### M8 跨对象压力与共享解包（2026-10-26）

`test_prefill_stress.cpp` 使用两组独立权重、两组独立随机行及四种交叉 reference，按 w0a0/w1a0/w0a1/w1a1 循环。阶段 0 复用全部 BO，阶段 1 每次重建权重 BO，阶段 2 每次重建 A/B BO，阶段 3 每次重建整个 Runner；可用末尾参数 `0..3` 单选阶段。`gen_data_i8.py --activation <a_bf16.bin>` 复用另一组打包 A 来生成交叉 reference，默认生成行为不变。每个调用仍检查 M×N 全部值。

最终 M8 在持久 context 的阶段 0/1/2 各 1000 次调用全部逐值精确 PASS，合计 3000 次、0 flake；随机/常数独立行各 20 轮及三轮 1000 轮性能对照也全部精确 PASS。旧 M8 同样通过持久 context 的交叉压力，因此此前引擎的成对 token 块损坏未在 standalone 可控条件下复现，不能据此断定引擎根因。阶段 3 是独立遗留风险：旧 M8 在 256 轮出现 1 轮零块，新 M8 在两次 64 轮各出现 1 轮零块；不是成对块形态，无 ERT timeout，本批未复位。高频 context 重建尚未闭合，持久 context 结果不能外推为该路径 0 flake。

M8 每个 64-K 半块现在只解包一次，四个行对依次复用 `b_buf_i16`；每次仍只有两个活跃 acc32，128-K 的部分和使用独立 2KB 状态保存，归约和 epilog 顺序不变。代码仅在 `DIM_M==8` 编译，M4 仍走原行对路径；M1/M2 生成 MLIR 与修改前逐字节一致，稳定 M1 默认未改。

相同 host、数据与 M1 控件的三组旧→新 1000 轮对照（暖态 995 轮）：

| 对照 | 旧 M8 batch ms | 新 M8 batch ms | 时间变化 |
|---|---:|---:|---:|
| 1 | 1.620230 | 1.152395 | -28.875% |
| 2 | 1.667515 | 1.061933 | -36.316% |
| 3 | 1.294035 | 0.878897 | -32.081% |
| 算术均值 | 1.527260 | 1.031075 | -32.489%，1.481× |

均值对应旧/新每 token 0.190908/0.128884ms，有效权重带宽 8.754/12.966GB/s；M8 每次物理权重 13,369,344B。三轮 M1×8 控件均值约 3.494496ms、每 token 0.436812ms、物理权重 106,954,752B、30.607GB/s。最终独立重建再测 M8 为 0.881932ms，1000 轮全值 PASS。表中对照均为串行单 context 测量；全程无 ERT timeout/reset。仅支持下一步持久 context 的受限 standalone/引擎实验，不是全 Shape 或生产验收。

### 受限引擎接入（2026-10-26）

引擎 `NPU_M8=1` 仅使用 K2048/N12288 资产，完整 8-token 块走持久 M8 Shape/run/BO，尾 token 走 M1；A 按 `[tile,m,k]` 打包，C 按 `[transfer,col,row,m,n]` 重排。`NPU_M8_CHECK=1` 在完整引擎内对每层每块直接计算 CPU NPU-semantics reference，而非复用 standalone 结论。65-token 320 块、256-token 1280 块均全值精确 PASS；M8 与 M1 的 65/256 下一 token 及两轮 65+64 OUT_IDS 一致，0 flake。相对当前 HEAD 默认 CPU projection 的两组总 prefill A/B：65-token 快 7.09%/5.05%，256-token 快 9.42%/11.74%。NPU 与默认 CPU 的输出 IDs 不保证一致，故默认不启用。旧 M2 成对块损坏的独立根因未据此宣称修复；高频整 Runner 重建仍有零块风险，禁止用重建规避。
