#!/bin/bash
# Step 4: queue of GPTQ builds on 2 GPUs (layers 0-24 on G0, 25-48 on G1). Usage: buildq.sh "NAME:KEEP:TIERS ..."
#   KEEP = a keep-set tag (KEEP_DIR/keep_<tag>.pt) or a number K (KEEP_DIR/keep_reapsum_<K>.pt); TIERS with '_' for ','.
# env: CAP (comma list of capture dirs), KEEP_DIR ($FN_WORK/keep), BUILD_DIR ($FN_WORK/builds), G0 (0), G1 (1), BATCH (32)
source "$(dirname "$0")/../env.sh"
CAP=${CAP:-$FN_WORK/cap/tr_a}; KEEP_DIR=${KEEP_DIR:-$FN_WORK/keep}; BUILD_DIR=${BUILD_DIR:-$FN_WORK/builds}
for spec in $1; do
  IFS=: read n K T <<< "$spec"
  T=${T//_/,}
  case $K in *[!0-9]*) KF=$KEEP_DIR/keep_$K.pt;; *) KF=$KEEP_DIR/keep_reapsum_$K.pt;; esac
  for r in 0-24:${G0:-0} 25-48:${G1:-1}; do
    python3 "$TRAINING_DIR/prune_gptq/gptq_fn.py" build --cap $CAP --keep $KF --tiers $T \
      --out $BUILD_DIR/$n --layers ${r%:*} --gpu ${r#*:} --batch ${BATCH:-32} > $LOGDIR/build_${n}_${r#*:}.log 2>&1 &
  done
  wait
  n_ok=$(ls $BUILD_DIR/$n/lowbit-L*.pt | wc -l)
  echo "$n: $n_ok lowbit files"
  [ $n_ok = 49 ] && { python3 "$TRAINING_DIR/prune_gptq/mkreadme.py" $BUILD_DIR/$n > /dev/null; touch $BUILD_DIR/$n/DONE; }
done
