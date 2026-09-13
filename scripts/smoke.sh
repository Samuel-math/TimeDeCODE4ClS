#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=42
python -u train.py --datasets JapaneseVowels --data-root "${DATA_ROOT:-datasets/UEA}" \
  --output "${OUTPUT_ROOT:-results/smoke}" --cb-epochs 1 --pre-epochs 1 --cls-epochs 2 \
  --dim 32 --layers 1 --batch-size 16 "$@"
