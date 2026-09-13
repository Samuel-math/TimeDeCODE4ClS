#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -u train.py --data-root "${DATA_ROOT:-datasets/UEA}" --output "${OUTPUT_ROOT:-results/r1}" \
  --seed 42 --patch-size 8 --dim 64 --codes 64 --layers 2 --batch-size 16 \
  --cb-epochs 20 --pre-epochs 30 --cls-epochs 100 --patience 15 "$@"
