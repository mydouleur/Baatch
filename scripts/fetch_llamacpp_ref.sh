#!/bin/bash
# 浅克隆 llama.cpp 参照实现（调研用，不链入引擎），固定到已调研的 commit
set -euo pipefail
DST="${1:-$(dirname "$0")/../../ref/llama.cpp}"
mkdir -p "$(dirname "$DST")"
if [ -d "$DST/.git" ]; then echo "已存在: $DST"; exit 0; fi
for REMOTE in "https://gh-proxy.com/https://github.com/ggml-org/llama.cpp.git" "https://github.com/ggml-org/llama.cpp.git"; do
  git clone --depth 1 "$REMOTE" "$DST" && break
done
echo "llama.cpp 参照就绪: $DST（调研笔记见 notes/qwen3next_impl.md，参照 qwen35moe 而非 qwen3next）"
