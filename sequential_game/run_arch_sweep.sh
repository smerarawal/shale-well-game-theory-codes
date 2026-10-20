#!/usr/bin/env bash
# run_arch_sweep.sh -- mix-only architecture sweep. For the unified architecture x
# hybrid x game/mix run, use run_consolidated_sweep.sh.
#
# Backbones (stage_surrogate.py): unet, ccs, gnn, split, ugnn (coupled U-Net/GNN), fgnn (coupled FNO/GNN),
# runet, fno, ufno. The coupled variants pass predicted pressure change into the saturation GNN and feed
# its saturation change back into a pressure correction.
# Training data: game fields + SPE10 train band ("mix"); eval: game truths, SPE10 Tarbert / Upper Ness test band.
# Standalone: regenerates the data if $OUT does not already hold it (reuses it otherwise).
#
#   bash run_arch_sweep.sh                                   # 10 backbones x {none, prior, prior2}, seed 0, 60 epochs
#   HYBRIDS="prior2" bash run_arch_sweep.sh                  # only the full hybrid (about 1/3 of the time)
#   BACKBONES="gnn split" UNROLL=3 bash run_arch_sweep.sh    # multi-stage (3 chained stages) training
#   WIDTH=96 BACKBONES="gnn" bash run_arch_sweep.sh          # size-matched GNN
#   SEEDS="0 1" bash run_arch_sweep.sh
#
# A run counts as finished only when <tag>_metrics.json exists (written at the end of training), so an
# interrupted run is retrained, never mistaken for a finished one. One failing backbone does not stop the sweep.
set -uo pipefail
REPO=${REPO:-/kaggle/working/repo/sequential_game}
SPE=${SPE:-/kaggle/input/spe10}
OUT=${OUT:-/kaggle/working/hyb}
SEEDS=${SEEDS:-"0"}
EPOCHS=${EPOCHS:-60}
WORKERS=${WORKERS:-4}
NEP=${NEP:-3000}
BACKBONES=${BACKBONES:-"unet ccs gnn split ugnn fgnn runet gnnlstm fno ufno"}
HYBRIDS=${HYBRIDS:-"none prior prior2"}
UNROLL=${UNROLL:-1}
WIDTH=${WIDTH:-48}      # e.g. WIDTH=96 BACKBONES=gnn for a size-matched GNN (0.23M params at 48 vs 1.2-1.9M for the others)
mkdir -p "$OUT"; cd "$REPO"

gen() { local f=$1; shift; [ -f "$f" ] && echo "[reuse] $f" || python make_stage_dataset.py --out "$f" "$@"; }
gen $OUT/stage_train.npz        --n_episodes $NEP --workers $WORKERS
gen $OUT/stage_eval_game.npz    --eval_geos --workers $WORKERS
gen $OUT/spe10_train.npz        --spe10_dir $SPE --spe10_region both   --spe10_band train --spe10_coarsen 4 --n_episodes $NEP --workers $WORKERS
gen $OUT/spe10_test_tarbert.npz --spe10_dir $SPE --spe10_region tarbert --spe10_band test  --spe10_coarsen 4 --n_episodes 200 --workers $WORKERS
gen $OUT/spe10_test_ness.npz    --spe10_dir $SPE --spe10_region ness    --spe10_band test  --spe10_coarsen 4 --n_episodes 200 --workers $WORKERS
declare -A EVAL=([game]=$OUT/stage_eval_game.npz [tarbert]=$OUT/spe10_test_tarbert.npz [ness]=$OUT/spe10_test_ness.npz)

usuf=""; [ "$UNROLL" -gt 1 ] && usuf="-u$UNROLL"; [ "$WIDTH" -ne 48 ] && usuf="$usuf-w$WIDTH"
for s in $SEEDS; do
  for bb in $BACKBONES; do
    for h in $HYBRIDS; do
      tag=${h}-${bb}${usuf}_mix_s$s
      if [ ! -f $OUT/${tag}_metrics.json ]; then
        python train_stage_surrogate.py --data $OUT/stage_train.npz $OUT/spe10_train.npz --hybrid $h --backbone $bb \
          --unroll $UNROLL --width $WIDTH --seed $s --epochs $EPOCHS --out $OUT/$tag.pt > $OUT/train_$tag.log 2>&1
        grep -E "^(params|ep +(1|[0-9]*0) |best| test)" $OUT/train_$tag.log || true
      else
        echo "[reuse] $tag"
      fi
      if [ ! -f $OUT/${tag}_metrics.json ]; then
        echo "[FAILED] $tag -- last log lines:"; tail -8 $OUT/train_$tag.log; continue
      fi
      for ev in game tarbert ness; do
        python validate_stage_surrogate.py --ckpt $OUT/$tag.pt --eval_data ${EVAL[$ev]} --eval_name $ev --out $OUT/val_${tag}_$ev > /dev/null \
          || echo "[validate FAILED] $tag $ev"
      done
      echo "[done] $tag"
    done
  done
done
python summarize_arch_sweep.py --dir $OUT
