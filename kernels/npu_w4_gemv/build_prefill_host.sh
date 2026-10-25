#!/usr/bin/env bash
set -euo pipefail
SRC=/d/xdna_llm/Baatch/kernels/npu_w4_gemv
OUT=${OUT:-/d/xdna_llm/Work/prefill_batch}
mkdir -p "$OUT"
OUT_W=$(cygpath -w "$OUT")
source /d/xdna_llm/m0/probe/vsenv.sh
cd "$SRC"
cl -nologo -EHsc -O2 -std:c++17 -Zc:__cplusplus \
  -ID:/xdna_llm/m0/probe/xrt_src/src/runtime_src/core/include \
  test_prefill_batch.cpp D:/xdna_llm/m0/probe/xrt_coreutil.lib \
  -Fo:"$OUT_W\\test_prefill_batch.obj" -Fe:"$OUT_W\\test_prefill_batch.exe"
cl -nologo -EHsc -O2 -std:c++17 -Zc:__cplusplus \
  -ID:/xdna_llm/m0/probe/xrt_src/src/runtime_src/core/include \
  test_prefill_stress.cpp D:/xdna_llm/m0/probe/xrt_coreutil.lib \
  -Fo:"$OUT_W\\test_prefill_stress.obj" -Fe:"$OUT_W\\test_prefill_stress.exe"
