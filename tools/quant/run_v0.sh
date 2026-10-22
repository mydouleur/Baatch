#!/usr/bin/env bash
# M1' XBF3 v0 one-shot reproduction.
# Creates the venv (if missing), installs numpy, runs the self-test and the
# analysis, leaving results in tools/quant/results_v0.json.
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/Scripts/python.exe
if [ ! -x "$PY" ]; then
  python -m venv .venv
  "$PY" -m pip install -q numpy
fi

"$PY" safetensors_reader.py   # reader smoke test
"$PY" xbf3.py                 # quantizer self-test (pack/unpack, rotation)
"$PY" analyze.py "$@"         # real measurements -> results_v0.json
