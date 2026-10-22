# xbf3_gemv — XBF3 打包权重 GEMV kernel（M2'-1）

第一个真 kernel：XBF3 压缩态权重从 DRAM 流入，片上 unpack 3-bit + MAC，fp32 累加输出。

## v1 规格（正确性优先）

- 形状：OUT=2048 × IN=512（专家 down_proj 形状），单列单核
- 权重：XBF3（每 superblock 256 权重 = 2B BF16 scale + 96B packed 3-bit 补码，行主序），
  总计 401,408 B，经 shim→core ObjectFifo 以 16 行（3,136 B）双缓冲块流式输入
- 激活：x[512] fp32，一次性载入驻留 tile DM
- 输出：y[2048] fp32，16 行/块产出
- 核心代码 `xbf3_gemv.cc`：标量 C（无 intrinsics 依赖），数值与 numpy 参考（tools/quant/xbf3.py）一致

## 构建（CI）

```
python3 xbf3_gemv.py --xclbin-path build/xbf3_gemv.xclbin --insts-path build/insts.bin
```
（ubuntu + mlir_aie wheel；见 .github/workflows/kernels.yml）

## 本机实测（Windows + XRT）

```
python gen_data.py --out data          # 生成测试向量（参考反量化对照）
# 编译 host（MSVC，环境见 probes/bandwidth/vsenv.sh）：
cl -nologo -EHsc -O2 -std:c++17 -Zc:__cplusplus -I<m0 probe>/xrt_src/src/runtime_src/core/include \
   test_host.cpp <m0 probe>/xrt_coreutil.lib -Fe:test_host.exe
test_host.exe build/xbf3_gemv.xclbin build/insts.bin data 20
```

Host ABI（MLIR_AIE 惯例）：arg0=opcode 3，arg1=instr BO，arg2=instr 字数，
arg3=wts / arg4=act / arg5=out（按 runtime sequence 操作数顺序 [wts, act, out]）。

## 实测记录

（待 CI 首轮产物上机后填写：数值校验 + 带宽）
