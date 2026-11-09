#!/bin/bash
# Degeneration test + holdout40 (one direct-mode attempt per test input, 240k-token cap) for each checkpoint in turn,
# served with the evaluation numerics at TP4 under a 26 GiB per-GPU budget. Repeat with SFX=_r2 ... for more runs.
# Usage: holdout.sh TAG:CKPT [TAG:CKPT ...]    env: GPUS (0,1,2,3), PORT (8155), LG (26), SFX ("")
# Output: $D3_WORK/hold/TAG$SFX.jsonl (+ _text.jsonl), $LOGDIR/hold_TAG$SFX.*.log; prints score_hold.py's line
source "$(dirname "$0")/../env.sh"
GPUS=${GPUS:-0,1,2,3}; PORT=${PORT:-8155}; LG=${LG:-26}; SFX=${SFX:-}; H=$D3_WORK/hold
mkdir -p $H
for spec in "$@"; do
  TAG=${spec%%:*}; CK=${spec#*:}; L=$LOGDIR/hold_$TAG$SFX.srv.log
  (setsid env L4_GIB=$LG EMULATE_GPU_GIB= MTP=3 bash "$TRAINING_DIR/serve/serve_eval.sh" $GPUS $PORT $CK > $L 2>&1 &)
  sleep 10; n=0; until grep -a -q -E 'Application startup complete|Traceback|core.py:1366' $L || [ $n -gt 200 ]; do sleep 15; n=$((n+1)); done
  grep -a -q 'Application startup complete' $L || { echo "$TAG: server failed"; bash "$TRAINING_DIR/serve/kill_server.sh" $PORT $GPUS; continue; }
  timeout 3600 python3 "$TRAINING_DIR/eval/degen_test.py" $PORT 16 12000 2>/dev/null | grep -a degenerate | sed "s/^/$TAG /"
  timeout 14400 python3 "$TRAINING_DIR/harness/direct.py" --ports $PORT --tasks @$TRAINING_DIR/eval/holdout40.txt --attempts 1 \
    --max_tokens 240000 --max_model_len 262144 --concurrency 48 --out $H/$TAG$SFX.jsonl --save_text $H/${TAG}${SFX}_text.jsonl \
    > $LOGDIR/hold_$TAG$SFX.log 2>&1
  python3 "$TRAINING_DIR/eval/score_hold.py" $H/$TAG$SFX.jsonl | sed "s/^/$TAG /"
  bash "$TRAINING_DIR/serve/kill_server.sh" $PORT $GPUS; sleep 20
done
echo PAIRDONE
