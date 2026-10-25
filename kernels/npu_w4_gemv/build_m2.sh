#!/usr/bin/env bash
set -euo pipefail
K=${K:-2048}
N=${N:-2048}
M=${M:-2}
RE=/d/xdna_llm/m0/probe/rai_explore
SRC=/d/xdna_llm/Baatch/kernels/npu_w4_gemv
OUT=/d/xdna_llm/Baatch/models/npu/k${K}n${N}m${M}fix
mkdir -p "$OUT/work" "$OUT/build"
cd "$SRC"
/tmp/aievenv/Scripts/python.exe vector_matrix.py --dev npu2 -K "$K" -N "$N" -M "$M" > "$OUT/work/aie144.mlir"
RAI="/c/ProgramData/miniforge3/envs/ryzen-ai-1.8.0/Lib/site-packages/win64.o/tools/peano"
NG="/tmp/aievenv/Lib/site-packages/llvm-aie"
"$RAI/bin/clang.exe" --target=aie2p-none-unknown-elf -O2 -std=c++20 -D__AIE_API_AIE_ADF_HPP__ \
  -Wno-macro-redefined -Wno-unknown-attributes -I. \
  -isystem "/tmp/w144/x/mlir_aie/include" -isystem "$NG/include/c++/v1" \
  -isystem "$NG/include/aie2p-none-unknown-elf/c++/v1" -isystem "$NG/include/aie2p-none-unknown-elf" \
  -isystem "$RE/clang22inc" -DDIM_N=64 -DDIM_K=128 -DDIM_M="$M" \
  -c vm_mix_int4_64x8_i8.cc -o "$OUT/work/vm_i8_m.o"
/tmp/aievenv/Lib/site-packages/mlir_aie/bin/llvm-objcopy.exe \
  --remove-section .tctmemtab --remove-section .tctmemstrtab --remove-section .tctmemtabl \
  "$OUT/work/vm_i8_m.o" "$OUT/work/vm_64x128.o"
cd "$OUT"
/tmp/aievenv/Scripts/aiecc.exe --peano="$RE/peano21" --tmpdir=work --dump-intermediates \
  --npu-insts-name=build/insts.bin --pdi-name=build/design.pdi --xclbin-name=build/w4i.xclbin \
  --xclbinutil-path="C:/Users/CHarLes/AppData/Local/Temp/xrtsdk/xrt_sdk/xrt/xclbinutil.exe" \
  --get-npu-insts --get-pdi --get-xclbin work/aie144.mlir
cp build/w4i.xclbin "../k${K}n${N}m${M}fix.xclbin"
cp build/insts.bin "../k${K}n${N}m${M}fix.insts.bin"
echo "M${M} fix build done"
