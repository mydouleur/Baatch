#!/usr/bin/env bash
# pack_xclbin.sh <work_dir> <build_dir> <out.xclbin>
# 本机组装 xclbin：aiecc 缺 xclbinutil 时的本地步骤（xrt_windows_sdk.zip 的 xclbinutil.exe）
set -e
WORK=$1; BUILD=$2; OUT=$3
XCLBINUTIL=${XCLBINUTIL:-/tmp/xrtsdk/xrt_sdk/xrt/xclbinutil.exe}
python - "$WORK" "$BUILD" <<'EOF'
import json, sys, shutil, os
work, build = sys.argv[1], sys.argv[2]
src = os.path.join(work, "main_aie_partition.json")
j = json.load(open(src))
j["aie_partition"]["PDIs"][0]["file_name"] = "design.pdi"
json.dump(j, open(os.path.join(build, "main_aie_partition.json"), "w"), indent=1)
shutil.copy(os.path.join(work, "main_kernels.json"), build)
shutil.copy(os.path.join(work, "main_mem_topology.json"), build)
EOF
cd "$BUILD" && "$XCLBINUTIL" \
  --add-replace-section MEM_TOPOLOGY:JSON:main_mem_topology.json \
  --add-kernel main_kernels.json \
  --add-replace-section AIE_PARTITION:JSON:main_aie_partition.json \
  --force --output "$OUT" 2>&1 | tail -2
echo "packed: $BUILD/$OUT"
