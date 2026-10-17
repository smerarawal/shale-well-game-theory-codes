#!/usr/bin/env bash
# run_hybrid_2x2.sh -- hybrid physics+learned stage surrogate vs the pure learned one, with and without
# channelised (SPE10) training data. Kaggle defaults; override with env vars.
#
#   SEEDS="0" bash run_hybrid_2x2.sh            # first pass: 6 runs
#   SEEDS="1" bash run_hybrid_2x2.sh            # add a second seed later (data/ckpts are reused)
#
# data=game : 3000 episodes on the game's random Gaussian-type fields (same as before)
# data=mix  : those + 3000 episodes on SPE10 patches from the TRAIN band (first half of the long axis)
# eval sets : game truths (24), SPE10 Tarbert and Upper Ness patches from the disjoint TEST band
set -euo pipefail
REPO=${REPO:-/kaggle/working/repo/sequential_game}
SPE=${SPE:-/kaggle/input/spe10}
OUT=${OUT:-/kaggle/working/hyb}
SEEDS=${SEEDS:-"0"}
EPOCHS=${EPOCHS:-100}
WORKERS=${WORKERS:-4}
NEP=${NEP:-3000}
mkdir -p "$OUT"; cd "$REPO"

gen() { local f=$1; shift; [ -f "$f" ] && echo "[reuse] $f" || python make_stage_dataset.py --out "$f" "$@"; }
gen $OUT/stage_train.npz        --n_episodes $NEP --workers $WORKERS
gen $OUT/stage_eval_game.npz    --eval_geos --workers $WORKERS
gen $OUT/spe10_train.npz        --spe10_dir $SPE --spe10_region both   --spe10_band train --spe10_coarsen 4 --n_episodes $NEP --workers $WORKERS
gen $OUT/spe10_test_tarbert.npz --spe10_dir $SPE --spe10_region tarbert --spe10_band test  --spe10_coarsen 4 --n_episodes 200 --workers $WORKERS
gen $OUT/spe10_test_ness.npz    --spe10_dir $SPE --spe10_region ness    --spe10_band test  --spe10_coarsen 4 --n_episodes 200 --workers $WORKERS

declare -A EVAL=([game]=$OUT/stage_eval_game.npz [tarbert]=$OUT/spe10_test_tarbert.npz [ness]=$OUT/spe10_test_ness.npz)

# physics prior alone (teacher-forced, one stage): how accurate is the exact-physics part on each geology?
for ev in game tarbert ness; do
  echo "=== prior diagnostic: $ev ==="; python prior_diagnostic.py --data ${EVAL[$ev]} --n_sub 8 --max_rows 1200 | tee $OUT/prior_diag_$ev.md
done

for s in $SEEDS; do
  for data in game mix; do
    if [ $data = game ]; then DATA="$OUT/stage_train.npz"; else DATA="$OUT/stage_train.npz $OUT/spe10_train.npz"; fi
    for h in none prior prior2; do
      tag=${h}_${data}_s$s
      [ -f $OUT/$tag.pt ] || python train_stage_surrogate.py --data $DATA --hybrid $h --seed $s --epochs $EPOCHS --out $OUT/$tag.pt 2>&1 | tee $OUT/train_$tag.log | grep -E "^(ep +(1|[0-9]*0) |best|params| test)" || true
      for ev in game tarbert ness; do
        python validate_stage_surrogate.py --ckpt $OUT/$tag.pt --eval_data ${EVAL[$ev]} --eval_name $ev --out $OUT/val_${tag}_$ev > /dev/null
      done
      echo "[done] $tag"
    done
  done
done
python summarize_hybrid_2x2.py --dir $OUT
