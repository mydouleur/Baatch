# Baatch 项目交接文档（智能体用）

> 最后更新：2026-10-24。接手先读本文与 `Baatch/docs/DESIGN.md` §10。历史报告只作证据，不作当前分工依据。

## 项目一句话

为 Tiel Coder 35B-A3B（40 层 = 30 GDN + 10 全注意力，256 专家 top-8，含 MTP）在 AMD XDNA2 NPU（Ryzen AI 9 H 365）上做低功耗推理引擎。

## 当前设计定案（用户 2026-10-24 复核）

- **XBF4 routed experts → NPU**；不得因属于 routed expert 就固定放 CPU。
- **XBF2 routed experts + imatrix → CPU**，使用 vnni/LUT 快路径。
- **共享专家、attention/GDN 投影、MTP 头、lm_head → NPU**。
- **embedding 查表、router/top-k、采样、fallback → CPU**。其他部件只有在 CPU 的功耗与速度综合性价比明显更好时才切到 CPU。
- 分发必须同时考虑 scheme、带宽、同步成本和功耗，不按“专家/非专家”机械分流。

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
- **M=1 prefill 已闭环**：GDN/attention 64-token 分块，65-token 全模型 prefill/decode 输出一致。
- **uniform 是过渡实验**：约 171–220ms/token，物理流量 4.73GB/token；dense 为 1.185GB/token，uniform 不是最终低功耗格式。
- **XBF4 routed experts NPU 数值路径已验收**：按需 host 转 tile + 512MB 缓存；40 层 80 个 GEMV 全部 PASS，最大 rel_l2=0.00546。两次独立 8-token 输出一致，11420 次调用、0 flake；prefill/decode 一致。量化噪声可改变近并列路由及后续 greedy token，不要求与 CPU 逐 token 相同，质量门留待后续。
- **MoE 单层对照**：`bench_moe` 的 current/planned 输出一致，planned 暖态无显著收益。E4×2 batch 数值通过但 layer0 暖态约 12.0ms，慢于 current 约 9.1ms。列并行 p4 每列持完整专家，数值通过、约 10.8ms，优于串行 batch 但仍慢于 current；均不接默认引擎。E6、长序列 p8、独立 BD＋去除专家间等待均挂起；不得重复这些变体。
- **M=1 out_proj 优化已接默认资产**：修复 N>=2048 时最后一次不存在的 ping-pong 传输；K4096/N2048 暖态 avg 0.332→0.221ms（约 -33%）。65-token prefill 49.63→46.21s（约 -6.9%），OUT_IDS 一致、38193 次调用 0 flake；K512/N2048 无收益未接入。
- **M>1 prefill 仅单形状闭合**：K2048/N2048 M2 的 akt/独立行/随机及 wn/wk/sn/sk/zn/zk 全 PASS，M2 0.265ms vs M1×2 0.332ms（约 -20%），资产为 `k2048n2048m2fix`。多块 N4096/N8192 随机激活 FAIL，K4096 M2 ERT timeout，未接入引擎； fused `_m` 的 wn 仍有车道错误，逐行调用 M1 kernel 才正确。双 A FIFO/单流交织/A 等待均死锁，禁止直接重复。
- **量化质量版未完成**：当前打包仍是朴素 absmax；XBF2 的 imatrix 尚未接入，PPL/任务质量验收尚未开始。
- 多 kernel xclbin 合并未提速；full ELF 与 XRT 2.19 不兼容；FastFlowLM 的 `xrt::module` 路径也未消除切换。

## 下一步顺序

1. 修正并验收 XBF4 routed experts 的 NPU 路径，保持 XBF2 在 CPU。
2. 优化 NPU 多形状切换和无效 padding，保留 CPU 仅做明确适合的部件。
3. 闭合 prefill M>1，建立 decode/prefill 性能与功耗基线。
4. 实现 MTP 并实测接受率、净加速。
5. 最后做 imatrix 质量版、探针、PPL、代码任务与路由一致性。

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
