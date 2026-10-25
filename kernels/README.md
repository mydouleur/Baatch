# kernels/ — XDNA2 AIE kernel

## 当前路线

- 本机 Windows 全链构建：MLIR 生成 → Peano kernel 编译 → aiecc/xclbinutil → XRT，常见形状约 15 秒，不再依赖 CI 往返。
- 出货工具链使用上游公开 llvm-aie；AMD RAI 编译器仅用于开发。
- XBF4 routed experts 与共享/密集权重走 NPU；XBF2 routed experts 走 CPU，不在 NPU 强行实现 2-bit LUT。

## 当前资产

- `npu_w4_gemv/`：W4A8 int16 VMAC GEMV，M=1 数值已闭合；M>1 prefill 仍在调试。
- `cpu_xbf3/`：CPU 低位宽 LUT/vnni 参考。
- `xbf3_gemv/`：早期 XBF3 实验，非当前 v1 主路线。
- 本地形状库位于 `models/npu/`（不入 git）。

## 约束

- `mmul<4,4,8>` 变体禁止上机；新 kernel 必须有独立 watchdog。
- 不使用 insts 热换或多 context 并发提交来规避切换问题。
- 历史 README/REPORT 保留实验证据；当前分工以 `docs/DESIGN.md` §10 为准。
