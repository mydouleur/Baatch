# Baatch

面向 AMD XDNA2 NPU 的低功耗 LLM 推理引擎。首发模型为 Tiel Coder 35B-A3B（Qwen3.5-MoE 混合线性注意力架构）。

## 当前设计

- **共享专家与 XBF2/XBF4 routed experts 默认走 CPU；attention/GDN 大投影 decode 与 MTP 头走 NPU，prefill 大投影和 out_proj 默认两核 CPU；lm_head decode 走 CPU、prefill 末尾走 NPU。`NPU_PROJ=1`/`NPU_OUT_PROJ=1` 保留对照**。
- **XBF2 + imatrix routed experts → CPU**。
- **embedding 查表、router/top-k、采样和 fallback → CPU**；其他部件只有在 CPU 综合性价比明显更好时才切到 CPU。
- `.baa` 容器保存逻辑形状与布局 flags，量化格式与后端可独立演进。

## 当前状态

- M3 混合 decode 已完成，三次独立 32-token 输出一致、0 flake。
- M=1 prefill 已完成，65-token 全模型 prefill/decode 输出一致。
- XBF4 routed experts 的 NPU 路径已验收但总成本更高，现默认 CPU；M>1 prefill、MTP、imatrix 和 PPL 尚未完成。
- uniform 仅是减少切换的过渡实验，不是最终格式。

## 文档

- `docs/DESIGN.md` —— 总纲与 v1 定案（§10）
- `docs/M2_ARCH.md` —— 当前引擎架构
- `docs/BAA_FORMAT.md` —— `.baa` 容器规范
- `src/README.md` —— 实现状态与实验记录
- `notes/qwen3next_impl.md` —— llama.cpp 数学参考笔记

## 开发环境

大文件（权重、XRT 头文件、参考实现）不入库。构建使用 Windows/MSVC、XRT、本地 mlir-aie/Peano 工具链；出货工具链使用上游公开版本。

## 许可证

Apache-2.0。Tiel/Ornith 权重为 MIT。
