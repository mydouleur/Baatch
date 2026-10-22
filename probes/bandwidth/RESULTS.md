# M0 带宽探针结果：XDNA2 NPU 路径 DRAM 流式带宽实测

机器：Ryzen AI 9 H 365（Strix Point，XDNA2，8 列），32GB LPDDR5x-8000
驱动/运行时：NPU driver 32.0.203.329，XRT 2.19.0（`C:\Windows\System32\xrt_coreutil.dll`），NPU FW 1.0.21.43
日期：2026-10-22。所有数字均为本机真实运行输出（日志见 `run_*.log`，逐迭代数据见 `iters*.csv`）。

## 头条数字

**探针形态**：AMD 官方 df-bw（data-fabric bandwidth）设计——4 列 ShimTile DMA 回环（loopback）：
每次迭代从 DRAM 读 1 GB 进 NPU、同时写回 1 GB 到 DRAM（读+写并发，数据逐比特校验 PASS）。
报告的 GB/s = 2 GB / 迭代耗时 = **读写合计**（单向约为一半）。

| 功耗模式 | 迭代数 | mean | median | p5 | p95 | max | warm min | 日志 |
|---|---|---|---|---|---|---|---|---|
| default | 200 | 54.09 | — | — | — | 58.14 | 51.25 | run_pmode_default.log |
| performance（机器初始态） | 200 | 60.17 | — | — | — | 64.13 | 56.36 | run_pmode_performance.log |
| turbo | 200 | 61.52 | — | — | — | 64.57 | 57.75 | run_pmode_turbo.log |
| **turbo（sustained 长跑）** | **1000** | **59.65** | **59.97** | **57.26** | **61.80** | **63.66** | 53.83 | run_turbo_1000.log + iters_turbo_1000.csv |

- performance 与 turbo 无显著差异（~60–62 GB/s）；default 低约 10%（~54 GB/s）。
  切换命令（无需管理员）：`xrt-smi configure --pmode turbo|performance|default`（测量后已恢复为初始的 performance）。
- 早期一批标为 performance 的 53.8 GB/s 数据（run_4col_500.log 等）受当时后台 git/网络任务干扰，作废；以干净重测为准。
- 1000 次 turbo 长跑无热衰减：前 100 次均值 59.28 vs 后 100 次 59.78 GB/s。
- 每次运行的第 1 次迭代为冷启动（buffer 页调入），仅 ~25–35 GB/s；统计时应剔除。
- 单次迭代 2 GB 流量、~33 ms，延迟来自 DMA 本身而非 host 开销。

## 关键实验结论

1. **列数不是瓶颈**。8 列无法通过现成 xclbin 直接测（见坑 5），改用两个并发 4 列 hw_context（驱动支持空间共享）各跑 df-bw：
   干净环境下（performance）两进程各 ~30.3 GB/s，**合计 60.6 GB/s，与单 4 列上下文（60.8）完全相同**。
   → 4 列已把 ShimTile→NoC→DRAM 共享路径压满；瓶颈在共享内存路径，不在列/shim 数量。
2. **长 burst 至关重要**（验证了设计文档假设）：BD 块从 32 MB 减到 16 MB 后，等效带宽掉到 ~20–40 GB/s（近腰斩）；
   32 MB×4 迭代（每 BD 实际传 128 MB）是该设计的最优配置；BD 长度字段不支持 64 MB（超时，超字段位宽）。
3. **功耗模式**：performance ≈ turbo（~60–61），default 低约 8%（~55）。
4. **读写未分离**：loopback 设计里 shim MM2S（读）数据经流开关内部回环到 S2MM（写），读必须配对写。
   纯读上限（LLM decode 场景真正关心的数）本探针测不了——需要 MemTile/计算核做数据汇的设计，
   而 Windows 上没有可用的 AIE 编译链路（见坑 1-3）。若读写通道配额独立（AMD 内部建模值读 32 / 写 32 GB/s），
   decode 纯读场景可能只有 ~30–33 GB/s；若总线可被读独占，则可达 ~60。**这是 M1' 之前必须回答的问题**。
5. 对照：destevez X-engine 在 Krackan（4 列 XDNA2）实测 54.6/65.9/77.6 min/avg/max。本机 4 列
   mean 59.7–60.9 / max 63.7–63.9 与其 avg 同量级；其 77.6 max 未在本机复现（可能为其固件/内存/计数口径差异）。

## 方法（如何复现）

```
cd d:\xdna_llm\m0\probe
# 编译 host（MSVC；vsenv.sh 是本机手工配的工具链环境，vcvars64.bat 坏）
bash -c "source vsenv.sh && cl -nologo -EHsc -O2 -std:c++17 -Zc:__cplusplus \
  -ID:\xdna_llm\m0\probe\xrt_src\src\runtime_src\core\include host_df_bw.cpp xrt_coreutil.lib -Fe:host_df_bw.exe"
# 运行（iters, buf MB, kernel 名）
host_df_bw.exe validate_elf_17f0_10.xclbin df_bw_17f0_10.elf 1000 1024 DPU
```

流程（XRT "elf flow"，全部用户态、无管理员）：
1. `xrt::device` + `register_xclbin(validate_elf_17f0_10.xclbin)`（4 列分区 PDI）+ `xrt::hw_context`
2. `xrt::module{xrt::elf{df_bw_17f0_10.elf}}`：`.ctrltext` = aie-rt 事务序列（shim BD 编程），
   `.rela.dyn` + `.dynsym`("3"/"5"=kernel 参数号） 供 XRT elf_patcher 把 BO 地址打进 BD
   （type-5 reloc = shim_dma_48：BD addr = bo_addr + addend + 0x80000000）
3. `xrt::ext::kernel{hwctx, mod, "DPU"}`，`xrt::ext::bo` 1 GB ×2（host_only、免 sync），
   run 参数：arg0=3（txn buffer opcode）、arg3=ifm、arg5=ofm、其余 0
4. 每迭代 `run.start()/wait2()` 计时；末次全量比对 ifm/ofm。

## 产物清单（d:\xdna_llm\m0\probe\）

| 文件 | 说明 |
|---|---|
| `host_df_bw.cpp` / `host_df_bw.exe` | 探针 host 程序（XRT C++ API） |
| `gen_elf.py` | ctrlcode ELF 合成器：4 列输出与 AMD stock 逐字节一致（除 patch 占位址），支持 --cols/--span |
| `df_bw_4col_gen.elf` | 合成 4 列版（已验证等价，54.5 GB/s warm avg） |
| `df_bw_8col.elf` | 合成 8 列版（在 preemption_4x8 xclbin 上超时，见坑 5） |
| `txn_decode.py` | aie-rt txn buffer / ctrlcode ELF 解码器 |
| `vsenv.sh`、`xrt_coreutil.def`、`xrt_coreutil.lib` | MSVC 手工环境 + xrt_coreutil.dll 导入库（pefile 导出生成 def，lib.exe 打包） |
| `validate_elf_17f0_10.xclbin`、`df_bw_17f0_10.elf` | 从 `m0\npu_mcdm_stack_prod` 复制的运行资产 |
| `run_*.log`、`iters*.csv` | 全部原始运行日志与逐迭代数据 |
| `xrt_src/` | XRT 源码稀疏克隆（gh-proxy 镜像），含 elf_patcher/runner/测试参考 |

## 过程中发现的关键事实（坑）

1. **Windows 上没有 mlir-aie 编译链**。Ryzen AI 1.8 自带的 `aiecompiler.exe` 是 cardano **ADF graph C++** 编译器
   （`--part=xc10AIE2P_ML-die-0x-e-S-es1` 支持 AIE2P），喂 `.mlir` 直接报错 77-6203「Unsupported file extension」。
   任务书预设的「写 MLIR → aiecompiler → xclbin」路线在 Windows 不存在。
2. **Windows 上无法打包自定义 xclbin**：WAIC 的 xclbin 打包 Makefile 需要 Vitis（bootgen/v++/xclbinutil）和 AMD 内网
   platform 仓库；`transform_static` 是 Linux ELF。Windows 正式流程 = 预编译 overlay xclbin + 运行时 PM bin 补丁 + 每算子
   txn 控制码。因此本探针复用官方 validate elf-flow 资产，自定义空间在 **ctrlcode（txn 序列）** 层。
3. **txn 二进制格式**（从 `dmacompiler/host_runtime/txn.hpp` + stock ctrlcode 逆向验证）：
   16B 头 `{Major,Minor,DevGen=4(AIE2P),NumRows=6,NumCols,NumMemTileRows=1, NumOps, TxnSize}`；
   op = WRITE(0, 24B) / BLOCKWRITE(1, 16B+data，shim BD 编程走这个) / C_TCT(128) / C_DDR_PATCH(129)。
   BD 块 = 8 字：`[len, addr_lo, addr_hi, 0, 0x80000000, len, 0, len]`，实测每 BD 传输量 = 4 × len。
4. **XRT host 侧**：`xrt::ext::bo` 是 host_only 一致性内存，**不要调 sync()**（报 "No host side buffer in destination
   buffer"）；kernel 名随 xclbin 不同（validate_elf 里是 `DPU`，validate 非 elf 版是 `DPU_PDI_0` 且参数补丁路径会失败）；
   MSVC 需 `-Zc:__cplusplus` 否则 XRT 头走 boost 分支；需补生成的 `xrt/detail/version-slim.h`（已放桩）。
5. **8 列 xclbin 死路**：全栈唯一 8 列分区 xclbin 是 `preemption_4x8_17f0.xclbin`；其 `DPU` kernel 元数据走 legacy
   bank 路径（set_arg 即失败），`DPU_ELF` kernel 需要 preemption save/restore 协议，裸跑超时（ERT_CMD_STATE_TIMEOUT，
   进程退出后设备自动恢复）。mlx 路线的 `Nx4` overlay partition_num_cols=1 也不可用。结论：单列宽探针 + 双上下文
   并发已足够回答"8 列是否更高"（否）。
6. **网络**：gitcode 直连不稳；`gh-proxy.com` 镜像可 clone（建议 blobless + sparse，全树 blob 拉取会卡死）；
   curl 拉 raw 文件 SSL 报错率高，优先 git 协议。
7. 附带基线（xrt-smi validate，performance 模式）：GEMM 51.3 TOPS、latency 83.6 µs、throughput 55612 ops/s。

## 对引擎的含义（M1'/M2' 输入）

- 带宽预算：**combined r+w sustained ≈ 60–61 GB/s（performance/turbo）/ 55 GB/s（default）**；decode 若读写比悬殊需重新标定。
- 数据流设计必须坚持 ≥32 MB 连续块（16 MB 块带宽近腰斩）。
- 列扩展无收益，4 列分区即可吃满；列数应省下来给计算/其他上下文共享。
- 待办（建议 M1' 前完成）：纯读上限探针需要 MemTile sink 设计 → 需要能产 ctrlcode 的编译链
  （WSL2 装上游 mlir-aie/aiecc，或继续逆向 memtile txn 编程手工合成）。

## 补记（2026-10-22 凌晨）

纯读探针做过一轮 2 小时尝试（MemTile sink ctrlcode 手工合成方向），未落地可运行产物，问题保持开放。
后续路线：WSL2 装上游 mlir-aie 用 aiecc 产自定义设计，或继续 memtile txn 逆向。在结论出来前，
decode 带宽预算按 **30–60 GB/s 区间**保守规划（方案 A 裸 decode 预期 22–45 tok/s，×MTP 后 40–80）。
