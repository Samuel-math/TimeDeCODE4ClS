#!/usr/bin/env bash
# Official TRAIN for all gradient updates; TEST for checkpoint selection/early stopping.
# Results under this protocol are NOT independent test estimates.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -u train.py --data-root "${DATA_ROOT:-datasets/UEA}" \
  --output "${OUTPUT_ROOT:-results/test_selection_r1}" --protocol test_selection \
  --seed 42 --patch-size 8 --dim 64 --codes 64 --layers 2 --batch-size 16 \
  --cb-epochs 20 --pre-epochs 30 --cls-epochs 100 --patience 15 "$@"
