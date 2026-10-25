# M3 后续执行计划

## 当前定案

- XBF2/XBF4 routed experts 默认走 CPU；XBF4 的 NPU 路径仅留实验对照。
- 共享专家、attention/GDN 大投影、MTP 头、lm_head → NPU。
- GDN 状态递推、embedding、router/top-k、采样和 fallback → CPU；其他部件按实测总成本分流。
- 默认 CPU routed experts 与 NPU shared down 重叠，暖态约 0.74–0.76s/token；65-token prefill 41.83s、0 flake。

## 已完成

- M3'-a：全模型 CPU decode、XBF24 容器、专家粒度 scheme 分发。
- M3'-b：NPU W4A8 密集路径、异步 submit/wait、超时熔断、hw_context 生命周期管理。
- M3 功能验收：三次独立 32-token 输出一致、0 flake。
- M=1 prefill：64-token 分块，65-token 全模型 prefill/decode 输出一致。
- CPU 激活量化、整数求和、偶奇/模4拆分已向量化。
- XBF4 routed experts NPU：40 层 80 个 GEMV 全部 PASS，最大 rel_l2=0.00546；独立生成及 prefill/decode 一致，0 flake，XBF2 保持 CPU。
- MoE 单层 current/planned 输出一致，planned 暖态无显著收益。E4×2 batch 数值通过但更慢（layer0 约 12.0ms vs current 约 9.1ms）。列并行 p4 每列持完整专家，数值通过、约 10.8ms，优于串行 batch 但仍慢于 current；均不进入全模型。

## 当前执行顺序

1. **XBF4 routed experts NPU 验收：已完成**
   路由近并列时量化噪声会改变后续 greedy token；不把与 CPU 逐 token 相同作为验收条件，质量门留待后续。
2. **NPU 切换与无效带宽优化**
   K4096/N2048 已去掉无效 ping-pong，暖态 avg 0.332→0.221ms，65-token prefill 约快 6.9%，已接默认资产。其余形状继续按收益逐项验收；uniform 仅作过渡。
3. **prefill M>1 与性能功耗基线**
   仅 K2048/N2048 M2 数值闭合且比 M1×2 快约 20%；多块 N4096/N8192 和 K4096 未闭合，全部主投影接入前不得替换 M1。随后建立 decode/prefill 性能与功耗对照。
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
