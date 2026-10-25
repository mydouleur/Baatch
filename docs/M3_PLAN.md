# M3 后续执行计划

## 当前定案

- XBF2/XBF4 routed experts 默认走 CPU；XBF4 的 NPU 路径仅留实验对照。
- 共享专家与 routed experts 默认走 CPU；attention/GDN 大投影 decode 与 MTP 头走 NPU；lm_head 在 decode 默认走两核 CPU，在 prefill 末尾走 NPU。
- GDN 状态递推、embedding、router/top-k、采样和 fallback → CPU；prefill 大投影与 out_proj 也默认两核 CPU，`NPU_PROJ=1`/`NPU_OUT_PROJ=1` 保留对照。65-token CPU projection 3321ms、NPU 3824ms。
- GDN qkv+z、attention q+k+v 默认共同复用 N12288；`NPU_WIDE_ATTN=0` 回退，`CPU_LM_HEAD=0` 回退 NPU lm_head。最终变化 prompt 65+64：prefill 3449ms，decode 平均 65.49ms、中位 62.6ms；256-token prefill 11011ms，NPU 0 flake。

## 已完成

- M3'-a：全模型 CPU decode、XBF24 容器、专家粒度 scheme 分发。
- M3'-b：NPU W4A8 密集路径、异步 submit/wait、超时熔断、hw_context 生命周期管理。
- M3 功能验收：三次独立 32-token 输出一致、0 flake。
- M=1 dense prefill：64-token 分块；MoE 按专家多 token GEMM 已接入，T=1/2/8/65 对拍通过，65-token prefill 5.92s（10.98 tok/s），输出一致。
- CPU 激活量化、整数求和、偶奇/模4拆分已向量化。prefill `PROF=1` 覆盖 proj/state/out/MoE；attention 16 头两核化使 256-token state 总计减少 0.461s。
- XBF4 routed experts NPU：40 层 80 个 GEMV 全部 PASS，最大 rel_l2=0.00546；独立生成及 prefill/decode 一致，0 flake，XBF2 保持 CPU。
- MoE 单层 current/planned 输出一致，planned 暖态无显著收益。E4×2 batch 数值通过但更慢（layer0 约 12.0ms vs current 约 9.1ms）。列并行 p4 每列持完整专家，数值通过、约 10.8ms，优于串行 batch 但仍慢于 current；均不进入全模型。
- 两核单份匿名模型驻留、SMT host、66B 行缓存、XBF2 合并归约与 CPU lm_head 已接默认；默认 decode 约 53–70ms/token（14–19 tok/s），NPU 0 flake。`BAA_MMAP=1`、`MOE_HOST_SIBLING=0`、`MOE_SCHEDULED=0`、`CPU_LM_HEAD=0` 可回退。常驻并发 XBF4 约 78GB/s、XBF2 约 61GB/s，后者仍未达 80GB/s 目标。prefill 后台预热 decode 行缓存，GDN 32 头两核并行后，router 两核化、NPU run 复用后，真实 65+64 的 decode 中位 57.90ms、平均 58.71ms、p95 61.9ms，TTFT 5.88s。

## 当前执行顺序

1. **XBF4 routed experts NPU 验收：已完成**
   路由近并列时量化噪声会改变后续 greedy token；不把与 CPU 逐 token 相同作为验收条件，质量门留待后续。
2. **NPU 执行图与无效带宽优化**
   N12288 复用已消除 decode 主投影切换，GDN 更新/输出融合少读 63MB/token。当前约 1.78GB/token，CPU4+NPU 实测聚合约 77GB/s，18ms 需要约 99GB/s 且受逐层依赖限制。下一步验证常驻 context、weight-ahead/activation-later；不得回退 insts 热换或多 context。
3. **prefill M>1 与性能功耗基线**
   K2048 主 fused Shape 的 M2 standalone 已闭合且有 kernel 收益；K4096/N2048 仍确定性 ERT timeout。成对引擎路径出现块损坏，hidden rel_l2=0.528，接入已移除，默认保持 M1。继续按逐层成本优化 MoE/GEMM 与 decode，并建立功耗对照。
4. **MTP**
   实现 draft/verify/accept 与状态回滚，实测接受率和净加速。
5. **量化质量**
   XBF2 imatrix、探针、PPL、代码任务和路由一致性。

## 不重复投入

- insts 热换、简单多 kernel xclbin 合并：无收益或有稳定性风险。
- full ELF：XRT 2.19 context 创建不兼容。
- FastFlowLM `xrt::module`：可运行但未消除切换。
- uniform 大 padding：只作过渡，不作为最终低功耗格式。
- E4×2 MoEBatch 数值可用但更慢，不接默认路径；E6、长序列 p8、独立 BD＋去除专家间等待会挂起，勿重复。
- M2 双独立 A FIFO、单 A 流交织、A DMA 等待均发生死锁；不要直接重复。fused `_m` 的 wn 车道错误未修复前不得替换稳定版逐行 M1 计算。
