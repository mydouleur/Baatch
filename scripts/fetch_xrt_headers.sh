#!/bin/bash
# 稀疏克隆 XRT 源码（仅头文件目录），固定 commit
# 用法: scripts/fetch_xrt_headers.sh [目标目录，默认 ../ref/xrt（仓库外）]
set -euo pipefail
DST="${1:-$(dirname "$0")/../../ref/xrt}"
mkdir -p "$(dirname "$DST")"
if [ -d "$DST/.git" ]; then echo "已存在: $DST"; exit 0; fi

# GitHub 慢时走 gh-proxy 镜像
for REMOTE in "https://github.com/Xilinx/XRT.git" "https://gh-proxy.com/https://github.com/Xilinx/XRT.git"; do
  git clone --filter=blob:none --sparse "$REMOTE" "$DST" && break
done
cd "$DST"
git sparse-checkout set src/runtime_src/core/include
echo "XRT 头文件就绪: $DST/src/runtime_src/core/include"
