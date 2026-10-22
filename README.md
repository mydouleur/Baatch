# Baatch

面向 AMD XDNA2 NPU 的低功耗 LLM 推理引擎。首发目标：Tiel Coder 35B-A3B（Qwen3.5-MoE 混合线性注意力架构）在 Ryzen AI 300 系上的高能效推理。

Low-power LLM inference engine for AMD XDNA2 NPUs. First target: Tiel-Coder-35B-A3B on Ryzen AI 300.

## 核心设计

- **`.baa` 容器格式**：厂商中立模型容器，内部以方案 tag 标识量化方案（XBF3 = XDNA2 3-bit，3.0625 bpw；未来 IBF3 = Intel 等）
- **零反量化数据流**：压缩权重直接流入 NPU，片上 unpack 进原生 MAC 车道，DRAM 带宽全用于权重流
- **混合执行**：NPU 热专家 + CPU 冷专家 turbo 混部（decode），prefill 走 NPU 大 GEMM
- **MTP 投机解码**：模型自带已训练 MTP 头，验证步把小 GEMM 喂给 NPU 甜区

## 实测基线（M0，Ryzen AI 9 H 365）

- NPU 路径 DRAM 带宽：读写合并 sustained **~60–61 GB/s**（turbo/performance 模式），4 列即压满共享路径
- xrt-smi validate：GEMM 51.3 TOPS
- XBF3 量化 SNR（CPU 参考实现）：旋转后 14.05 dB，达 3-bit 高斯源理论极限带

详见 `probes/bandwidth/RESULTS.md` 与 `reports/xbf3_v0_report.md`。

## 文档

- `docs/DESIGN.md` —— 项目总纲（目标、格式、量化、功耗模型、路线图）
- `docs/M2_ARCH.md` —— 引擎架构设计方案
- `notes/qwen3next_impl.md` —— llama.cpp qwen35moe 参照实现调研

## 搭建开发环境

大文件（权重、XRT 头文件、参考实现）一律不入库，用脚本拉取：

```bash
scripts/fetch_models.sh        # 从 ModelScope 拉 BF16 权重（~67GB）
scripts/fetch_xrt_headers.sh   # 固定 commit 稀疏克隆 XRT 头文件
scripts/fetch_llamacpp_ref.sh  # 浅克隆 llama.cpp 参照实现
scripts/setup_toolchain.ps1    # 检查 Ryzen AI SW / 驱动 / conda 环境
```

## 许可证

Apache-2.0。模型权重（Tiel/Ornith）为 MIT，随模型分发。

## 状态

M0 ✅（工具链 + 带宽基线）/ M1' ✅（量化管线 v0）/ M2' 🚧（XBF3 GEMV kernel 开发中）
