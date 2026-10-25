# Baatch 项目交接文档（智能体用）

> 最后更新：2026-10-24。接手先读本文与 `Baatch/docs/DESIGN.md` §10。历史报告只作证据，不作当前分工依据。

## 项目一句话

为 Tiel Coder 35B-A3B（40 层 = 30 GDN + 10 全注意力，256 专家 top-8，含 MTP）在 AMD XDNA2 NPU（Ryzen AI 9 H 365）上做低功耗推理引擎。

## 当前设计定案（2026-10-25 实测修订）

- **XBF2/XBF4 routed experts 默认走 CPU**；XBF4 的 NPU 路径仅留实验对照。
- **共享专家、routed experts、GDN 状态递推、embedding、router/top-k、采样、fallback 默认走 CPU**。
- **attention/GDN 大投影、MTP 头 → NPU**；lm_head 在 decode 默认走两核 AVX-512 CPU，在 prefill 末尾走 NPU；out_proj 在 decode/prefill 均默认走 AVX-512 CPU，`NPU_OUT_PROJ=1` 保留 prefill NPU 对照，其他部件按实测总成本分流。
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
- **uniform 是过渡实验**：约 171–220ms/token，物理流量 4.73GB/token；dense 为 1.185GB/token，uniform 不是最终低功耗格式。
- **XBF4 routed experts NPU 数值路径已验收**：按需 host 转 tile + 512MB 缓存；40 层 80 个 GEMV 全部 PASS，最大 rel_l2=0.00546。两次独立 8-token 输出一致，11420 次调用、0 flake；prefill/decode 一致。量化噪声可改变近并列路由及后续 greedy token，不要求与 CPU 逐 token 相同，质量门留待后续。
- **MoE 单层对照**：`bench_moe` 的 current/planned 输出一致，planned 暖态无显著收益。E4×2 batch 数值通过但 layer0 暖态约 12.0ms，慢于 current 约 9.1ms。列并行 p4 每列持完整专家，数值通过、约 10.8ms，优于串行 batch 但仍慢于 current；均不接默认引擎。E6、长序列 p8、独立 BD＋去除专家间等待均挂起；不得重复这些变体。
- **M=1 out_proj 优化已接默认资产**：修复 N>=2048 时最后一次不存在的 ping-pong 传输；K4096/N2048 暖态 avg 0.332→0.221ms（约 -33%）。65-token prefill 49.63→46.21s（约 -6.9%），OUT_IDS 一致、38193 次调用 0 flake；K512/N2048 无收益未接入。
- **M2 FIFO 与共享解包已闭合 K2048 主 fused Shape**：A FIFO 深度在 M>1 时从 2 增至 8，修复跨组覆盖；M2 微内核两行共享一次 W4 解包和 B 加载，M1 不变。N12288 随机/独立行 PASS，1000 次均值 0.678ms vs M1×2 1.012ms；N10240 0.619ms vs 0.760ms；N2048/N4096/N8192 为 0.288/0.360/0.534ms，均 PASS。K4096/N2048 M2 仍确定性 ERT timeout，复位后 M1 PASS；按 K 加深 FIFO 超过每通道 24 BD 上限。通用 M4 数值 FAIL，未接入。`NPU_M2=1` 成对路径默认关闭：配 CPU out 的 64-token prefill M2 3447ms、M1 3877ms，约快 11.1%；65-token 因尾 M1/BO 开销为 3943 vs 3863ms，仍无收益，默认保持 M1。fused `_m` 的 wn 车道错误仍未修复；双 A FIFO/单流交织/A 等待均禁止重复。
- **量化质量版未完成**：当前打包仍是朴素 absmax；XBF2 的 imatrix 尚未接入，PPL/任务质量验收尚未开始。
- 多 kernel xclbin 合并未提速；full ELF 与 XRT 2.19 不兼容；FastFlowLM 的 `xrt::module` 路径也未消除切换。

## 下一步顺序

1. 按 `docs/DESIGN.md` §10.5 的真实流量与并发带宽设计常驻 context、weight-ahead/activation-later；不得再用 `max(NPU,CPU)` 直接承诺 18ms。
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
