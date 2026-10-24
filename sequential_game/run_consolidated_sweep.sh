#!/usr/bin/env bash
# One consolidated architecture x hybrid x training-data experiment.
# Baselines reproduce the 6-cell unet matrix; additional backbones compare published
# designs and the coupled U-Net/GNN and FNO/GNN candidates on the same data and splits.
#
#   bash run_consolidated_sweep.sh
#   BACKBONES="unet split runet ugnn fgnn" SEEDS="0" bash run_consolidated_sweep.sh
#   HYBRIDS="prior2" DATASETS="mix" bash run_consolidated_sweep.sh
# Kaggle T4x2 default: two independent training jobs, one pinned to each GPU.
set -euo pipefail
REPO=${REPO:-/kaggle/working/repo/sequential_game}
SPE=${SPE:-/kaggle/input/spe10}
OUT=${OUT:-/kaggle/working/consolidated}
SEEDS=${SEEDS:-"0"}
EPOCHS=${EPOCHS:-100}
WORKERS=${WORKERS:-4}
NEP=${NEP:-3000}
BACKBONES=${BACKBONES:-"unet ccs gnn split runet gnnlstm fno ufno ugnn fgnn"}
HYBRIDS=${HYBRIDS:-"none prior prior2"}
DATASETS=${DATASETS:-"game mix"}
WIDTH=${WIDTH:-48}
UNROLL=${UNROLL:-1}
GNNLSTM_UNROLL=${GNNLSTM_UNROLL:-6}
SPEED_CASES=${SPEED_CASES:-3}
GPU_IDS=${GPU_IDS:-"0 1"}
read -r -a GPUS <<< "$GPU_IDS"
if [ ${#GPUS[@]} -eq 0 ]; then GPUS=(cpu); fi
MAX_PARALLEL=${MAX_PARALLEL:-${#GPUS[@]}}
mkdir -p "$OUT"; cd "$REPO"

gen() { local f=$1; shift; [ -f "$f" ] && echo "[reuse] $f" || python make_stage_dataset.py --out "$f" "$@"; }
gen "$OUT/stage_train.npz"        --n_episodes "$NEP" --workers "$WORKERS"
gen "$OUT/stage_eval_game.npz"    --eval_geos --workers "$WORKERS"
gen "$OUT/spe10_train.npz"        --spe10_dir "$SPE" --spe10_region both --spe10_band train --spe10_coarsen 4 --n_episodes "$NEP" --workers "$WORKERS"
gen "$OUT/spe10_test_tarbert.npz" --spe10_dir "$SPE" --spe10_region tarbert --spe10_band test --spe10_coarsen 4 --n_episodes 200 --workers "$WORKERS"
gen "$OUT/spe10_test_ness.npz"    --spe10_dir "$SPE" --spe10_region ness --spe10_band test --spe10_coarsen 4 --n_episodes 200 --workers "$WORKERS"

declare -A EVAL=([game]=$OUT/stage_eval_game.npz [tarbert]=$OUT/spe10_test_tarbert.npz [ness]=$OUT/spe10_test_ness.npz)

for ev in game tarbert ness; do
  python prior_diagnostic.py --data "${EVAL[$ev]}" --n_sub 8 --max_rows 1200 > "$OUT/prior_diag_$ev.md"
done

run_one() {
  local seed=$1 data=$2 bb=$3 h=$4 gpu=$5 tag model_unroll
  local -a train
  tag=${bb}_${h}_${data}_s${seed}
  model_unroll=$UNROLL
  [[ "$bb" = gnnlstm || "$bb" = ccslstm ]] && model_unroll=$GNNLSTM_UNROLL
  if [ "$data" = game ]; then train=("$OUT/stage_train.npz"); else train=("$OUT/stage_train.npz" "$OUT/spe10_train.npz"); fi
  if [ ! -f "$OUT/${tag}_metrics.json" ]; then
    echo "[start] $tag on $gpu"
    if [ "$gpu" = cpu ]; then
      python train_stage_surrogate.py --data "${train[@]}" --backbone "$bb" --hybrid "$h" \
        --unroll "$model_unroll" --width "$WIDTH" --seed "$seed" --epochs "$EPOCHS" --device cpu --out "$OUT/$tag.pt" \
        > "$OUT/train_$tag.log" 2>&1
    else
      CUDA_VISIBLE_DEVICES="$gpu" python train_stage_surrogate.py --data "${train[@]}" --backbone "$bb" --hybrid "$h" \
        --unroll "$model_unroll" --width "$WIDTH" --seed "$seed" --epochs "$EPOCHS" --device cuda --out "$OUT/$tag.pt" \
        > "$OUT/train_$tag.log" 2>&1
    fi
    grep -E "^(params|ep +(1|[0-9]*0) |best| test)" "$OUT/train_$tag.log" || true
  else
    echo "[reuse] $tag"
  fi
  if [ ! -f "$OUT/${tag}_metrics.json" ]; then
    echo "[FAILED] $tag -- last log lines:"; tail -8 "$OUT/train_$tag.log"; return 1
  fi
  for ev in game tarbert ness; do
    python validate_stage_surrogate.py --ckpt "$OUT/$tag.pt" --eval_data "${EVAL[$ev]}" \
      --eval_name "$ev" --out "$OUT/val_${tag}_$ev" > /dev/null || echo "[validate FAILED] $tag $ev"
  done
  echo "[done] $tag"
}

PIDS=()
JOB=0
FAILED=0
for seed in $SEEDS; do
  for data in $DATASETS; do
    for bb in $BACKBONES; do
      for h in $HYBRIDS; do
        gpu=${GPUS[$((JOB % ${#GPUS[@]}))]}
        run_one "$seed" "$data" "$bb" "$h" "$gpu" &
        PIDS+=("$!")
        ((JOB+=1))
        if [ ${#PIDS[@]} -ge "$MAX_PARALLEL" ]; then
          for pid in "${PIDS[@]}"; do wait "$pid" || FAILED=1; done
          PIDS=()
        fi
      done
    done
  done
done
for pid in "${PIDS[@]}"; do wait "$pid" || FAILED=1; done

python summarize_consolidated_sweep.py --dir "$OUT"
if [ "${RUN_SPEED_BENCHMARK:-1}" = "1" ] && compgen -G "$OUT/*.pt" > /dev/null; then
  # Benchmark all checkpoints so physics hybrid costs are represented as well.
  python benchmark_stage_speed.py --data "$OUT/stage_eval_game.npz" --ckpt_dir "$OUT" \
    --cases "$SPEED_CASES" --out "$OUT/cpu_speed"
fi
exit "$FAILED"
