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

### 2026-10-22 v1 上机（Ryzen AI 9 H 365，XDNA2，driver 32.0.203.329 / XRT 2.19）

- **数值校验：PASS**。随机数据（正态权重 ×0.05 + 正态激活），2048 输出全对，
  max_rel = 1.11e-04（fp32 累加顺序噪声水平）。50 次迭代稳定。
- 图样测试：全 1（y=512）PASS；全 8 电平周期（y=-256）PASS；scale 区分图样抓获过
  一个真 bug（sb 循环未推进 rowp，sb1 重复读 sb0，已修复）。
- 时延：~119 ms/次（128 个 3136B 小块串行 DMA + 标量计算主导）——此数字不代表带宽能力，
  带宽版（memtile + 大块 + 向量化）是 M2'-2 的事。
- 管线：CI（ubuntu + mlir_aie 1.3.4 + llvm-aie nightly）产出 insts.bin / design.pdi /
  metadata json；本机用 XRT 2.21.75 SDK 的 `xclbinutil.exe`（XRT GitHub release
  `xrt_windows_sdk.zip`，免安装免管理员）组装 xclbin——**AIE_PARTITION 的
  `file_name` 需改成本地相对路径**（xclbinutil 组装时把 PDI 内容嵌进 section）。
- 关键坑：mlir_aie wheel 不含 Peano（需另装 llvm-aie nightly wheel）；aiecc 的
  xclbin 生成依赖 xclbinutil，缺失时静默成功（exit 0 无产物）；1.3.4 与 main 的
  IRON API 差异大（Runtime()/rt.sequence 习语 vs 新 fn_args 风格）。

## v2/v3 带宽版（M2'-2）

设计演进（全部数值兼容 v1 的 XBF3 布局）：

| 版本 | 结构 | 实测权重流（warm avg） |
|---|---|---|
| v1 scalar | shim→core 直送，16 行/块，单专家 | 119 ms/次（小块串行主导） |
| v2 discard | shim→memtile(256行)→core，E=64 拼接 24.5MB | 8.0 GB/s |
| v2 discard 大块 | 同上但 128/1024 行块 | 9.6 GB/s |
| v3 discard 4 列 | 4 列并行（shim→memtile→core），行切分，E=64 | **min 23.4 / avg 29.1 / max 30.3 GB/s**（50 次，warm） |
| v3 scalar 4 列 | 同上 + 真实计算 | verify **PASS**（max_rel 1.11e-04）；有效流 0.11 GB/s（计算墙） |

### 瓶颈分析（v2 → v3 的依据）

- 单列 discard 8–10 GB/s，把 L2/L1 块从 256/16 行加到 1024/128 行几乎不变（9.6）→
  **不是 DMA 任务粒度问题，是单列单 shim 通道上限**（对照 M0：df-bw 4 列 8 流道 = 60 GB/s，
  每 shim 读流道 ≈ 7.5–10 GB/s，吻合）。
- v3：4 列 ×（shim→memtile→core），行切分（每列 512 行/专家），激活并入每列流首块
  （规避每 shim 2 MM2S + 1 S2MM 的通道预算——分离 act fifo 时 AIEPlaceTiles 失败）。
- 已知限制：act 块独占一个 L2 对象（slab 对齐要求），核内跳过填充块。

### v4（XBF3-BP 位平面 + 向量化）结论（2026-10-22，M2'-3 收尾）

- **v4 scalar（真实计算）verify PASS**（64 专家 × 4 列全随机，max_rel=1.02e-04）：
  min/avg/max = 0.57/0.57/0.58 GB/s 有效权重流（45ms/25MB），比 v3 标量版快 **42×**
  （v3 的 0.014 GB/s 是 AIE2P 无标量浮点单元、软浮点库调用所致）。
- **v4 discard 管线**：min/avg/max = **31.1/32.2/33.0 GB/s**（4 列单 MM2S 通道）。
- 图样+隔离测试全套 PASS：全 1 / 8 电平周期 / 随机 q×(x=1) / (q=1)×随机 x /
  x=δ 脉冲配对 / q 随机×x=i+1 配对。
- **未达 55 GB/s**。瓶颈归因：
  - 计算墙：~2500 cyc/行（提取序列 8 相位 × ~15 op + 每 sb reduce_add/scale），
    extract:MAC ≈ 10:1——3-bit 奇数位宽对 16/32 位车道根本不对齐，移位指令在
    AIE2P 向量单元不存在（仅 MAC 累加器 lsrs + 整寄存器 funnel shift），
    提取只能走 mul+srs+band 组合。
  - 管线墙：单 MM2S 通道/列 → 32 GB/s；df-bw 的 60 GB/s 用的是每列 2 通道。
- 明确的下一步（留 M2'-4+）：(a) 每列双 MM2S 通道（管线 → 60 GB/s）；
  (b) 每列多核（4 tile 行）摊薄计算；(c) 提取改用 parallel_lookup 硬件 LUT 或
  int4 中间格式+原生 unpack；(d) reduce/scale 开销向量化。
- 关键坑（本里程碑新增）：aie_api `load_v` 按 32B 对齐**静默截断**（须用
  `cbegin_unaligned_vector`）；`cast_to<int8>` 窄化生成非法 G_SHL（改走 DM 往返）；
  `x+x` 被 instcombine 折成向量 shl（改乘子权重表进内存防折）；acc→vector 的
  srs 默认算术移位+饱和（bit7 掩码穿透，须 `to_vector_sign<uint16_t>(false,8)`）；
  mlir_aie wheel 的 aie_api 需 `-D__AIE_API_AIE_ADF_HPP__` 跳过 Vitis adf.h；
  CI 手工管线里 ExternalFunction 的 compile_flags 不生效（-D 要加在 clang++ 行）；
  gen_data 的 act slab 在 bp 模式下必须写 xp 相位平面而非原始 x（抓获于图样隔离）。

### v3 结论（2026-10-22 收尾）

- **数值正确性保持**：v3 scalar 四列 × 64 专家（131072 输出）verify PASS，
  max_rel=1.11e-04 与 v1 完全相同（图样 bug 史：chunk_rows 不一致时"每块只对前 16 行"）。
- **未达 55 GB/s**。当前管线上限 ~30 GB/s（4 列 × 单 MM2S 通道）；达 55+ 需要：
  (a) 每列双 MM2S 通道或 8 列（df-bw 实测 8 流道 = 60 GB/s）；
  (b) **向量化 unpack+MAC**（标量 fp32 是 0.11 GB/s 的计算墙，距管线 ~270×，
      需 aie2p intrinsics bf16/int8 向量车道，~64–512 MAC/cycle）。
- 编译链坑（v3 调试记录）：`NPU2Col1` 是 `-d npu2` 的默认设备（多列设计必须显式
  `NPU2()`）；手工 aiecc 管线里 ExternalFunction 的 compile_flags 不生效
  （-D 须加在 clang++ 命令行）；多 kernel .o 符号重复（同一 .cc 复制多份 .o →
  ld.lld duplicate symbol，须拆源文件）；artifact 里 build_*/work_* 目录定位。
