#!/bin/bash
# 从 ModelScope 拉取 Ornith-1.5-35B-A3B BF16 权重（~67GB，Tiel Coder 的逐比特同源权重）
# 用法: scripts/fetch_models.sh [目标目录，默认 ../models（仓库外工作区）]
set -euo pipefail
DST="${1:-$(dirname "$0")/../../models/Ornith-1.5-35B-A3B}"
BASE="https://modelscope.cn/models/ornith-ai/Ornith-1.5-35B-A3B/resolve/master"
mkdir -p "$DST"

for f in config.json generation_config.json chat_template.jinja tokenizer.json tokenizer_config.json \
         vocab.json merges.txt preprocessor_config.json processor_config.json model.safetensors.index.json; do
  [ -f "$DST/$f" ] || curl -sL --retry 3 -o "$DST/$f" "$BASE/$f"
done

# 分片：精确大小从 API 取，8 段并发，断点续传，逐片校验
curl -s --max-time 60 "https://modelscope.cn/api/v1/models/ornith-ai/Ornith-1.5-35B-A3B/repo/files?Recursive=true" \
| python -c "
import json,sys
for f in json.load(sys.stdin)['Data']['Files']:
    if f['Type']=='blob' and f['Name'].endswith('.safetensors'):
        print(f['Name'], f['Size'])" | tr -d '\r' \
| while read -r name size; do
  out="$DST/$name"
  if [ -f "$out" ] && [ "$(stat -c%s "$out")" = "$size" ]; then echo "SKIP $name"; continue; fi
  tmp="$DST/.parts_$name"; mkdir -p "$tmp"
  chunk=$(( (size + 7) / 8 ))
  for i in 0 1 2 3 4 5 6 7; do
    s=$((i*chunk)); e=$(( (i+1)*chunk - 1 )); [ $e -ge "$size" ] && e=$((size-1))
    ( for try in 1 2 3 4 5 6; do
        curl -sL --retry 3 -r "${s}-${e}" -o "$tmp/p$i" "$BASE/$name" && break; sleep 5
      done ) &
  done
  wait
  ok=1
  for i in 0 1 2 3 4 5 6 7; do
    s=$((i*chunk)); e=$(( (i+1)*chunk - 1 )); [ $e -ge "$size" ] && e=$((size-1))
    [ "$(stat -c%s "$tmp/p$i" 2>/dev/null || echo 0)" != "$((e-s+1))" ] && { echo "BAD $name part$i"; ok=0; }
  done
  [ $ok -eq 1 ] && cat "$tmp"/p* > "$out" && rm -rf "$tmp" && echo "OK $name"
done
echo "模型就绪: $DST"
