# M3' 引擎编写计划（0a/0b/0c 全过后启动）

目标：能跑的端到端 decode——加载 .baa、喂 prompt、产出连贯文本。CPU 为主线（0b 已验证），NPU 4-bit 侧视 0a 结果接入。

## M3'-a：引擎骨架 + CPU 端到端（主线）

**模块划分**（对应 docs/M2_ARCH.md §3）：

1. `src/loader/` —— .baa 读取（mmap + 目录二分），先做"safetensors 直读 + 内存中量化"的调试路径，正式 .baa 由 tools/quant 的写入器产出
2. `src/compute/` —— CPU kernel 集：
   - vnni2 专家 GEMV（已有，kernels/cpu_xbf3/，迁移为库）
   - GDN decode 融合 kernel（元素级递推，数学照抄 ref/llama.cpp delta-net-base.cpp:286-368；SSM state fp32 每层 2MiB）
   - 全注意力 decode（10 层，QK^T/softmax/AV + attn_output_gate + rope_multi）
   - conv1d k=4、RMSNorm、SiLU、router（softmax+topk-8+归一化）
3. `src/sched/` —— decode 循环、KV（10 层）+ SSM state（30 层）管理、采样器（temp/top-p/top-k）
4. `src/tokenize/` —— BPE（用 models/Ornith 的 tokenizer.json；可选 huggingface tokenizers 静态库或自研最小 BPE）
5. 数值验收：单层输出 vs numpy 参考（tools/quant 已有读取器）；端到端：固定 prompt + temp=0 的输出与 llama.cpp GGUF 版对照（Q4 精度差异内的 token 一致率）

**验收**：temp=0 下对 5 个固定 prompt 产出连贯代码/文本；decode 速度实测记录。

## M3'-b：量化模型全量打包

- tools/quant 写 .baa 写入器（XBF3 专家 + XBF4 attn/lm_head）
- 全模型离线量化（CPU 4–6h，通宵跑）——先用快速量化（absmax）通流程，AWQ 校准版后续替换
- 引擎加载 .baa 全模型跑通

## M3'-c：NPU 接入（视 0a 结果）

- 0a ≥55 GB/s：attn 投影 + lm_head 上 NPU，与 CPU 专家流汇合（0c 已验证汇合开销可忽略）
- 0a 不达标：NPU 侧降级为 prefill-only，decode 纯 CPU 2–4 核

## 明确不做（本里程碑）

prefill 大 GEMM NPU 化（M4'）、MTP（M4'）、serve API 层（M4'）、前缀缓存（M4'）
