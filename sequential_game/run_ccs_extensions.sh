#!/usr/bin/env bash
# Focused follow-up: compare CCSNet controls and two CCSNet mixed-architecture variants.
# Use the same OUT directory as the consolidated sweep so completed CCS checkpoints
# and all generated datasets are reused. The runner skips configurations with metrics.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO=${REPO:-$HERE}
export SPE=${SPE:-/kaggle/input/spe10}
BASE_OUT=${BASE_OUT:-/kaggle/working/consolidated_final}
export OUT=${OUT:-/kaggle/working/ccs_extensions}
export SEEDS=${SEEDS:-"0 1 2"}
export EPOCHS=${EPOCHS:-100}
export WORKERS=${WORKERS:-4}
export NEP=${NEP:-3000}
export BACKBONES=${BACKBONES:-"ccs ccslstm ccsfno"}
export HYBRIDS=${HYBRIDS:-"prior2"}
export DATASETS=${DATASETS:-"mix"}
export WIDTH=${WIDTH:-48}
export UNROLL=${UNROLL:-6}
export GNNLSTM_UNROLL=${GNNLSTM_UNROLL:-6}
export GPU_IDS=${GPU_IDS:-"0 1"}
export MAX_PARALLEL=${MAX_PARALLEL:-2}
export RUN_SPEED_BENCHMARK=0

# Keep a separate output directory so the CCS control is retrained with the same
# six-stage unroll as the new recurrent model; do not silently reuse a u=2 checkpoint.
mkdir -p "$OUT"
for f in stage_train.npz stage_eval_game.npz spe10_train.npz spe10_test_tarbert.npz spe10_test_ness.npz; do
  if [ ! -f "$OUT/$f" ] && [ -f "$BASE_OUT/$f" ]; then
    cp "$BASE_OUT/$f" "$OUT/$f"
  fi
done

exec bash "$HERE/run_consolidated_sweep.sh"
