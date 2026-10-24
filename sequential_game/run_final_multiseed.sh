#!/usr/bin/env bash
# Final confirmatory surrogate run. Resume-safe across Kaggle notebook versions.
# Core matrix: U-Net, CCSNet-style dual decoder, CCS-GNN exchange;
# all hybrids x data regimes x seeds. Reference finalists: prior2 + mixed data.
set -euo pipefail

REPO=${REPO:-/kaggle/working/repo/sequential_game}
SPE=${SPE:-/kaggle/input/spe10model2}
OUT=${OUT:-/kaggle/working/consolidated_final}
SEEDS=${SEEDS:-"0 1 2"}
EPOCHS=${EPOCHS:-100}
WORKERS=${WORKERS:-4}
NEP=${NEP:-3000}
WIDTH=${WIDTH:-48}
UNROLL=${UNROLL:-2}
HI_W=${HI_W:-3.0}
PMAX_W=${PMAX_W:-5.0}
SAT_W=${SAT_W:-0.5}
PENALTY_W=${PENALTY_W:-0.1}
SELECTION_PENALTY_W=${SELECTION_PENALTY_W:-0.1}
GPU_IDS=${GPU_IDS:-"0 1"}
MAX_PARALLEL=${MAX_PARALLEL:-2}
SPEED_CASES=${SPEED_CASES:-3}

read -r -a GPUS <<< "$GPU_IDS"
if [ ${#GPUS[@]} -eq 0 ]; then GPUS=(cpu); fi
mkdir -p "$OUT"
cd "$REPO"

gen() {
  local f=$1; shift
  if [ -f "$f" ]; then
    echo "[reuse data] $f"
  else
    python make_stage_dataset.py --out "$f" "$@"
  fi
}

gen "$OUT/stage_train.npz"        --n_episodes "$NEP" --workers "$WORKERS"
gen "$OUT/stage_eval_game.npz"    --eval_geos --workers "$WORKERS"
gen "$OUT/spe10_train.npz"        --spe10_dir "$SPE" --spe10_region both --spe10_band train --spe10_coarsen 4 --n_episodes "$NEP" --workers "$WORKERS"
gen "$OUT/spe10_test_tarbert.npz" --spe10_dir "$SPE" --spe10_region tarbert --spe10_band test --spe10_coarsen 4 --n_episodes 200 --workers "$WORKERS"
gen "$OUT/spe10_test_ness.npz"    --spe10_dir "$SPE" --spe10_region ness --spe10_band test --spe10_coarsen 4 --n_episodes 200 --workers "$WORKERS"

declare -A EVAL=( [game]="$OUT/stage_eval_game.npz" [tarbert]="$OUT/spe10_test_tarbert.npz" [ness]="$OUT/spe10_test_ness.npz" )
for ev in game tarbert ness; do
  if [ ! -f "$OUT/prior_diag_$ev.md" ]; then
    python prior_diagnostic.py --data "${EVAL[$ev]}" --n_sub 8 --max_rows 1200 > "$OUT/prior_diag_$ev.md"
  fi
done

run_one() {
  local seed=$1 bb=$2 hybrid=$3 data=$4 gpu=$5 tag
  local -a train
  tag=${bb}_${hybrid}_${data}_s${seed}
  if [ "$data" = game ]; then
    train=("$OUT/stage_train.npz")
  else
    train=("$OUT/stage_train.npz" "$OUT/spe10_train.npz")
  fi

  if [ ! -f "$OUT/${tag}_metrics.json" ] || [ ! -f "$OUT/$tag.pt" ]; then
    echo "[start] $tag on $gpu"
    if [ "$gpu" = cpu ]; then
      python train_stage_surrogate.py --data "${train[@]}" --backbone "$bb" --hybrid "$hybrid" \
        --unroll "$UNROLL" --width "$WIDTH" --seed "$seed" --epochs "$EPOCHS" --device cpu \
        --hi_w "$HI_W" --pmax_w "$PMAX_W" --sat_w "$SAT_W" --penalty_w "$PENALTY_W" \
        --selection_penalty_w "$SELECTION_PENALTY_W" --out "$OUT/$tag.pt" > "$OUT/train_$tag.log" 2>&1
    else
      CUDA_VISIBLE_DEVICES="$gpu" python train_stage_surrogate.py --data "${train[@]}" --backbone "$bb" --hybrid "$hybrid" \
        --unroll "$UNROLL" --width "$WIDTH" --seed "$seed" --epochs "$EPOCHS" --device cuda \
        --hi_w "$HI_W" --pmax_w "$PMAX_W" --sat_w "$SAT_W" --penalty_w "$PENALTY_W" \
        --selection_penalty_w "$SELECTION_PENALTY_W" --out "$OUT/$tag.pt" > "$OUT/train_$tag.log" 2>&1
    fi
    grep -E "^(params|ep +(1|[0-9]*0) |best| test)" "$OUT/train_$tag.log" || true
  else
    echo "[reuse model] $tag"
  fi

  if [ ! -f "$OUT/${tag}_metrics.json" ] || [ ! -f "$OUT/$tag.pt" ]; then
    echo "[FAILED] $tag -- last log lines:"; tail -8 "$OUT/train_$tag.log"; return 1
  fi
  for ev in game tarbert ness; do
    if [ -f "$OUT/val_${tag}_${ev}.json" ]; then
      echo "[reuse validation] $tag $ev"
    else
      python validate_stage_surrogate.py --ckpt "$OUT/$tag.pt" --eval_data "${EVAL[$ev]}" \
        --eval_name "$ev" --out "$OUT/val_${tag}_${ev}" > /dev/null || echo "[validate FAILED] $tag $ev"
    fi
  done
  echo "[done] $tag"
}

# Full factorial final baseline: 3 model families x 3 hybrid modes x 2 data regimes.
CORE_BACKBONES=(unet ccs ccsgnn)
HYBRIDS=(none prior prior2)
DATASETS=(game mix)
# Strong results from the completed single-seed study receive seed replication in
# the best-supported physics/data setting, avoiding another full 10-backbone sweep.
FINALISTS=(gnn split ugnn fgnn)
JOBS=()
for seed in $SEEDS; do
  for data in "${DATASETS[@]}"; do
    for bb in "${CORE_BACKBONES[@]}"; do
      for hybrid in "${HYBRIDS[@]}"; do
        JOBS+=("$seed $bb $hybrid $data")
      done
    done
  done
  for bb in "${FINALISTS[@]}"; do
    JOBS+=("$seed $bb prior2 mix")
  done
done

echo "[plan] ${#JOBS[@]} model fits; seeds=[$SEEDS], epochs=$EPOCHS, unroll=$UNROLL; output=$OUT"
PIDS=()
JOB=0
FAILED=0
for spec in "${JOBS[@]}"; do
  read -r seed bb hybrid data <<< "$spec"
  gpu=${GPUS[$((JOB % ${#GPUS[@]}))]}
  run_one "$seed" "$bb" "$hybrid" "$data" "$gpu" &
  PIDS+=("$!")
  ((JOB+=1))
  if [ ${#PIDS[@]} -ge "$MAX_PARALLEL" ]; then
    for pid in "${PIDS[@]}"; do wait "$pid" || FAILED=1; done
    PIDS=()
  fi
done
for pid in "${PIDS[@]}"; do wait "$pid" || FAILED=1; done

python summarize_consolidated_sweep.py --dir "$OUT" --out "$OUT/final_multiseed_report.md"
if compgen -G "$OUT/*.pt" > /dev/null; then
  # One representative seed per configuration keeps this final comparison practical.
  python benchmark_stage_speed.py --data "$OUT/stage_eval_game.npz" --ckpt_dir "$OUT" \
    --cases "$SPEED_CASES" --seed 0 --out "$OUT/cpu_speed_seed0"
fi
exit "$FAILED"
