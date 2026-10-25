# Baatch 项目交接文档（智能体用）

> 最后更新：2026-10-26。接手先读本文与 `Baatch/docs/DESIGN.md` §10。历史报告只作证据，不作当前分工依据。

## 项目一句话

为 Tiel Coder 35B-A3B（40 层 = 30 GDN + 10 全注意力，256 专家 top-8，含 MTP）在 AMD XDNA2 NPU（Ryzen AI 9 H 365）上做低功耗推理引擎。

## 当前设计定案（2026-10-25 实测修订）

- **XBF2/XBF4 routed experts 默认走 CPU**；XBF4 的 NPU 路径仅留实验对照。
- **共享专家、routed experts、GDN 状态递推、embedding、router/top-k、采样、fallback 默认走 CPU**。
- **attention/GDN 大投影 decode 默认走两核 AVX-512 CPU，并逐块复现 NPU BF16 数值语义；`NPU_DECODE_PROJ=1` 保留旧 NPU 对照**。prefill 大投影与 out_proj 默认两核 CPU；lm_head 在 decode 默认 CPU、prefill 末尾走 NPU。`NPU_PROJ=1`、`NPU_OUT_PROJ=1` 保留 prefill NPU 对照。
- GDN qkv+z、attention q+k+v 使用 fused projections；不得仅按“专家/非专家”机械分流。

## 目录地图

| 路径 | 内容 |
|---|---|
| `Baatch/` | 引擎仓库 |
| `Baatch/docs/DESIGN.md` | 总纲与决策记录，§10 为 v1 定案 |
| `Baatch/docs/BAA_FORMAT.md` | `.baa` 容器规范 |
| `Baatch/src/engine/` | C++/MSVC 引擎 |
| `Baatch/tools/quant/baa_writer_v1.py` | XBF2/XBF4 打包器 |
| `Baatch/kernels/npu_w4_gemv/` | W4 GEMV/GEMM 与测试 |
| `models/` | BF16 权重、XBF24 容器、imatrix，均不入 git |
| `m0/probe/rai_explore/` | W4A8 kernel、本地构建链与实测报告 |
| `ref/llama.cpp` | 数学参考实现，注意 qwen35moe ≠ qwen3next |

## 当前实现状态

- **M3 功能闭环**：NPU 密集 decode + CPU routed experts；三次独立 32-token 输出一致，累计 24849 次调用、0 flake。
- **2026-10-25 分流实测**：GDN 递推仅约 9ms/token，暂缓双核；XBF4 gate_up+down 都回 CPU 后约 0.85–0.93s/token，shared down 与 CPU routed experts 重叠后约 0.74–0.76s/token。K2048/N2048 的无效 ping-pong 已接入默认资产。
- **Fused projections 已接默认路径**：GDN qkv+z 使用 K2048/N12288，attention q+k+v 也补零复用 N12288（k/v 各 1024），decode 形状切换从约 21 次/token 降至约 2 次；`NPU_WIDE_ATTN=0` 可回退 N10240。共享 gate/up/down 与 out_proj 默认 CPU；64-token prefill 两轮 CPU out 3871/3958ms、NPU out 4393/4653ms，NPU 调用 5121→2561，`NPU_OUT_PROJ=1` 可对照。
- **两核匿名驻留与 CPU lm_head 已设默认**：BAA 默认一次性读入单份匿名内存（不保留额外专家副本），`BAA_MMAP=1` 回退；host 在 LP1，routed/Pool 在 LP0/LP2，`MOE_HOST_SIBLING=0` 回退。预平铺 XBF4 使用 128 通道、66B 行缓存；XBF2 四路点积合并为一次归约。decode lm_head 默认两核 CPU，`CPU_LM_HEAD=0` 回退 NPU。GDN 32 头已拆到两核并行，默认短 decode 约 50–70ms/token，65-token prefill 5.64s，输出一致且 NPU 0 flake。常驻并发 XBF4 约 38+40=78GB/s，XBF2 约 20+41=61GB/s；四核/八核、额外专家复制和四行 GEMV 未接入。prefill 期间后台预热 decode out/lm_head 行缓存，router 已两核化，NPU 每 Shape 复用 `xrt::run`；真实 65+64 测试：decode 中位 57.90ms、均值 58.71ms、p95 61.9ms、最大 95.5ms，TTFT 5.88s；首轮 576ms 尖峰已消除。匿名模型启动读写约 12s。
- **prefill MoE 按专家多 token batch 已接默认**：`moe_prefill.h` 按 router 分组执行 gather→XBF4/XBF2 多 token GEMM→scatter，同一权重读取服务多个 token；dense 投影仍用稳定 M=1。XBF4 GEMM 已合并两个 128-half 后只做一次整数归约，T=1/2/8/65 的 layer0/layer10 direct 对拍全部 PASS（rel_l2≤1.1e-8）；layer0 65-token MoE 21.71→19.05ms，变化 prompt 全模型 prefill 4936→4718ms。浮点舍入路径更接近 direct，近并列路由仍可能改变 IDs。
- **prefill 分项与 attention 两核化**：`PROF=1` 现覆盖 prefill proj/state/out/MoE。256-token 基线为 proj 6.47s、MoE 5.32s、state 1.55s、out 1.34s；attention 16 头拆两核后每层 state 从约 100–106ms 降至 54–56ms，总计少 0.461s。65+16 对照 prefill 4071→4018ms，IDs 一致、0 flake；decode 数学不变。
- **2026-10-25 批次验收**：默认 CPU prefill projection/out 路径在变化 prompt 65+64 上 prefill 3449ms（18.84 tok/s），decode 平均 65.49ms、中位 62.6ms、p95 66.8ms，2561 次 NPU 调用 0 flake；256-token prefill 11011ms（23.25 tok/s），MoE 5.356s。四 token GEMM、router batch、共享专家 batch、GDN 双行、异步小投影和下一专家预取均因全模型回退或未复现收益而未接入。
- **2026-10-26 decode 单 token 优化**：历史 58.71ms 与 65.49ms 使用不同 prompt/路由，不是同条件代码回退；同输入旧、新二进制也未复现该回退。decode 独立分项测得旧 NPU 投影约 31.27ms/token，其中硬件 wait 为主，BO sync/输出转换不是瓶颈。默认改为两核 CPU 投影，并逐 128 块复现 NPU 整数点积、BF16 最近偶数转换与累加；普通 FP32 CPU 路径会分叉，未采用。最终同二进制 65+64 A/B：NPU 对照平均 65.37ms、中位 62.0ms、p95 74.10ms，默认平均 53.82ms、中位 52.9ms、p95 57.18ms（18.58 tok/s），64 个 OUT_IDS 完全一致；对照 2561 次 NPU 调用 0 flake，默认仅 prefill lm_head 1 次调用 0 flake；默认独立重复为平均 50.26ms、中位 50.2ms、p95 51.29ms。attention 16 头 decode 两核化且保持每头标量归约，RoPE 三角函数复用；`PROF=1 PROF_DECODE=1` 输出 decode 独立逐层、MoE 与 NPU submit/wait 分项。25ms/token 尚未达到。
- **2026-10-26 MoE decode 调度优化**：routed 改为两核各连续处理 4 个专家，再按原 top-k 顺序归并；scheduled worker 先算 shared，主线程单核 router 后执行 routed，使 router 与 shared 重叠。direct/scheduled 的 layer0/layer10 哈希一致，固定 `ids=0..64` 的 65+64 OUT_IDS 一致。HEAD 两轮平均 53.306/55.483ms、MoE 17.454/18.225ms；候选三轮平均 53.011/53.256/53.489ms、MoE 16.414/16.490/16.127ms，标准 kernel tests PASS、decode NPU 0 flake。25ms/token 未达到。XBF2/XBF4 双行 GEMV 与量化临时缓冲复用无收益或回退，未接入。
- **2026-10-26 decode 单行 XBF4 内核优化**：128 通道块改为两块独立点积/整数归约，FP32 与 NPU-equivalent BF16 仍按原块序累加；BF16 最近偶数转换使用 Zen5 硬件指令。GDN qkv+z 与 attention q+k+v 各复用一次激活量化，不使用 `thread_local`。固定 65+64 的 HEAD 两轮平均 51.863/51.650ms，最终三轮 50.800/49.967/49.375ms；同窗口紧邻 A/B 为 51.650→49.375ms，proj 17.289→15.152ms。最终三轮中位 50.7/49.1/48.95ms、p95 51.7/51.3/50.4ms、19.685/20.013/20.253 tok/s；OUT_IDS 与 HEAD 一致，layer0/layer10 direct/scheduled 哈希一致，标准 tests PASS，decode NPU 0 flake。四块展开无继续收益，未接入；25ms/token 未达到。
- **2026-10-26 decode out_proj 两核修正**：`mv_out` 原漏传线程池，实际单核执行；现复用两核 Pool，66B 行布局、单行 XBF4 与 FP32 块序不变。每 token 实际逻辑权重流量为 out_proj 173.015MB、lm_head 262.226MB。固定 65+64 的同配置 HEAD 平均 48.533ms、out 6.952ms（24.89GB/s）；最终两轮平均 45.194/46.125ms、中位 45.15/46.0ms、p95 46.2/47.65ms、22.127/21.680 tok/s，out 4.569/4.616ms（37.87/37.48GB/s），lm_head 6.327/6.302ms（41.45/41.61GB/s），OUT_IDS 与 HEAD 一致、tests PASS、0 flake。独立 scale＋连续 64B payload 布局及 lm_head packed 实验无收益，已撤回；25ms/token 未达到。
- **uniform 是过渡实验**：约 171–220ms/token，物理流量 4.73GB/token；dense 为 1.185GB/token，uniform 不是最终低功耗格式。
- **XBF4 routed experts NPU 数值路径已验收**：按需 host 转 tile + 512MB 缓存；40 层 80 个 GEMV 全部 PASS，最大 rel_l2=0.00546。两次独立 8-token 输出一致，11420 次调用、0 flake；prefill/decode 一致。量化噪声可改变近并列路由及后续 greedy token，不要求与 CPU 逐 token 相同，质量门留待后续。
- **MoE 单层对照**：`bench_moe` 的 current/planned 输出一致，planned 暖态无显著收益。E4×2 batch 数值通过但 layer0 暖态约 12.0ms，慢于 current 约 9.1ms。列并行 p4 每列持完整专家，数值通过、约 10.8ms，优于串行 batch 但仍慢于 current；均不接默认引擎。E6、长序列 p8、独立 BD＋去除专家间等待均挂起；不得重复这些变体。
- **2026-10-26 M=1 routed 复核**：`bench_moe --mode npu` 保持 direct 的 router/top-k、CPU shared 与原顺序归并，仅顺序执行 XBF4 routed GEMV。gate_up/down 都复用稳定 K2048/N2048（down 从 K512 补零），每 XBF4 层 top-8 物理流量 35.652MB，CPU 行主序为 12.780MB。layer0 暖态 NPU 两轮 3.341/3.620ms，CPU direct 0.336ms；NPU 慢 9.9–10.8×，两轮 1280 次暖态调用 0 flake，CPU direct 对拍 rel_l2≤0.000158。无净收益，未接完整 decode，XBF2 保持 CPU。
- **M=1 out_proj 优化已接默认资产**：修复 N>=2048 时最后一次不存在的 ping-pong 传输；K4096/N2048 暖态 avg 0.332→0.221ms（约 -33%）。65-token prefill 49.63→46.21s（约 -6.9%），OUT_IDS 一致、38193 次调用 0 flake；K512/N2048 无收益未接入。
- **M2 FIFO 与共享解包已闭合 K2048 主 fused Shape**：A FIFO 深度在 M>1 时从 2 增至 8，修复跨组覆盖；M2 微内核两行共享一次 W4 解包和 B 加载，M1 不变。N12288 随机/独立行 PASS，1000 次均值 0.678ms vs M1×2 1.012ms；N10240 0.619ms vs 0.760ms；N2048/N4096/N8192 为 0.288/0.360/0.534ms，均 PASS。K4096/N2048 M2 仍确定性 ERT timeout，复位后 M1 PASS；按 K 加深 FIFO 超过每通道 24 BD 上限。通用 M4 数值 FAIL，未接入。M2 成对引擎实验虽一度快约 11%，但跨权重/激活时序下 token 42/43、50/51、54/55 出现成对块损坏，最终 hidden rel_l2=0.528；新建 run 不能修复，引擎接入已移除，默认与源码均为 M1。fused `_m` 的 wn 车道错误仍未修复；双 A FIFO/单流交织/A 等待均禁止重复。
- **2026-10-26 M4/M8 standalone 已闭合**：新增同二进制 M1×M 全成本 host 与独立随机行 reference，计时含 run/submit/wait/每次输出同步。M4/M8 按行对复用 M2，`noinline` 避免约 1KB 向量栈 spill；M>=4 改用一个二维 ND A BD 描述全部 k-tile，修复旧独立 BD 随机缺一个 k-tile，M1/M2 不变。K2048/N12288 随机与常数独立行各 20 轮 PASS；1000 轮随机性能每轮逐值 PASS：M4 0.713996ms vs M1×4 1.764343ms（-59.532%），每 token 0.178499ms；M8 1.324756ms vs M1×8 3.510150ms（-62.259%），每 token 0.165595ms。物理权重 13,369,344B vs 53,477,376/106,954,752B；M8 相对 M4 每 token 仅再降 7.229%，复用可扩展但行对计算近线性。早期诊断 3 次复位后 M1 均 PASS，ND 修复后 M4/M8 与对照 0 flake。仅 standalone，未接引擎，不得外推端到端 prefill。
- **2026-10-26 M8 跨对象压力与共享解包**：新增四组合权重×激活 stress，持久 context 下 BO 复用、每次换权重 BO、每次换 A/B BO 各 1000 轮全值精确 PASS，旧引擎成对块损坏未复现。高频整 Runner 重建是独立遗留问题：旧 M8 1/256、新 M8 2/128 出现零块，不得宣称该路径闭合。M8 每半 K 解包降为一次，四个行对复用并保存 2KB 部分和；三组 1000 轮旧→新均值 1.527260→1.031075ms（-32.489%），每 token 0.128884ms、12.966GB/s，全部数值 PASS。M1/M2 MLIR 逐字不变，M4 保持原路径；本批 0 ERT timeout/reset。仅 standalone，引擎仍默认 M1。
- **2026-10-26 M8 受限引擎实验**：`NPU_M8=1` 仅启用 K2048/N12288 fused prefill，8-token 块使用持久 Shape/run/BO，尾 token 走 M1；其他 Shape 不进入 M8，默认 CPU projection 与旧 `NPU_PROJ=1` M1 不变。`NPU_M8_CHECK=1` 对每层每块直接运行同量化、同 BF16 块序 CPU reference；65-token 320 块、256-token 1280 块均全值精确 PASS，0 flake。M8 与旧 M1 的 65/256 下一 token 和两轮 65+64 OUT_IDS 完全一致；默认 CPU 65+64 也与 HEAD 一致。两组总 prefill A/B 均有净收益：65-token 3529.8→3279.5ms、3655.7→3470.9ms；256-token 12279.1→11122.3ms、12395.9→10941.2ms。最终 65/256 候选 proj/state/out/MoE/final/lm_head 分别 1041.120/225.751/434.339/1574.939/0.003/117.159ms 与 1866.940/1233.019/1553.073/6008.643/0.004/101.271ms。NPU 与默认 CPU 数值语义不同，65/256 下一 token 及 65+64 OUT_IDS 不保证与 CPU 相同，故只保留实验开关、不设默认；旧 M2 根因未据此宣称修复，高频 Runner 重建仍禁止。
- **2026-10-26 prefill MoE 8-token 实验**：`MOE_PREFILL8=1` 每块先完成 8 个 token 的 router/top-k，再按 expert 聚合，输出写回 top-k slot 后严格按 slot 0→7 归并；XBF4 GEMM 的一次权重解包复用于块内所有 token，XBF2 保持共享解包。layer0/layer10 的自然/固定/随机路由及 65-token 尾块均与逐 token direct 逐位同哈希，GEMM/GEMV 单测 PASS。固定 `ids=0..255`：CPU 三轮平均 9779.4ms（26.18 tok/s）、MoE 3359.1ms，HEAD 单次 11870.3ms、MoE 5922.4ms；M8 两轮平均 8334.3ms（30.72 tok/s）、MoE 3239.7ms，HEAD 单次 10836.6ms、MoE 5950.3ms，NPU 每轮 1281 次调用 0 flake。65-token 与变化 prompt 亦有净收益，decode 平均性能不变。候选与 direct 语义一致，但不同于旧整段 expert 序归并，完整 OUT_IDS 可分叉，故默认路径不变，仅保留实验开关。
- **量化质量版未完成**：当前打包仍是朴素 absmax；XBF2 的 imatrix 尚未接入，PPL/任务质量验收尚未开始。
- 多 kernel xclbin 合并未提速；full ELF 与 XRT 2.19 不兼容；FastFlowLM 的 `xrt::module` 路径也未消除切换。

## 下一步顺序

1. 继续按 `PROF=1 PROF_DECODE=1` 的 decode 单 token 分项优化 MoE、out_proj、GDN 与 lm_head，目标 25ms/token；不得重复已挂起的 weight-ahead 双静态入口、insts 热换或多 context。
2. 建立 decode/prefill 性能、功耗与长上下文内存基线；dense M>1 仅在全部主形状闭合后接入。
3. 实现 MTP 并实测接受率、净加速。
4. 最后做 imatrix 质量版、探针、PPL、代码任务与路由一致性。

## 实测锚点

- NPU 纯读上限约 51GB/s；W4 GEMV 43.6GB/s；uniform kernel 约 38GB/s。
- xclbin 切换约 2.6ms/次；同形状调用约 0.43ms/次；汇合 p50 70µs。
- CPU vnni LUT GEMV：1/2/4 大核约 21/39/49GB/s。

## 红线与规矩

- `mmul<4,4,8>` 变体禁止上机；NPU 实验一律带独立看门狗，超时后停止，不在异常 context 上重试。
- 大文件不进 git；子代理串行；本地留痕不删。
- NPU 挂起后可调用用户已登记的 `XDNA-NPU-Reset`（`schtasks /run /tn "XDNA-NPU-Reset"`），随后先跑 M=1 安全探针；不得自行请求新的提权或重启整机。
- 引擎 Apache-2.0；出货工具链用 llvm-aie 上游公开版，AMD RAI 编译器仅开发期。
- 中文 Windows；Windows Python 路径用 `D:/`；conda env `ryzen-ai-1.8.0`。

## 用户偏好

低功耗是立项目标；质量优先于体积；全量 turbo（eco 后议）；不赶发布；先保证正确分工和数值闭环，再做性能；回答简洁，动手术前先说方案。
