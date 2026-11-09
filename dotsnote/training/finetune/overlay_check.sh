#!/bin/bash
# Served check of a fine-tune overlay: serve CKPT (with ARC_DELTA_FILE=DELTA if given) with the validation numerics
# minus MTP (so prompt logprobs are exact), dump overlay_check.py logprobs to $D3_WORK/overlay/TAG.json, stop the server.
# Usage: overlay_check.sh TAG CKPT [DELTA]    env: GPUS (0,1), PORT (8497)
source "$(dirname "$0")/../env.sh"
TAG=$1; CK=$2; DELTA=${3:-}; GPUS=${GPUS:-0,1}; PORT=${PORT:-8497}; L=$LOGDIR/overlay_$TAG.srv.log
mkdir -p $D3_WORK/overlay
(setsid env L4_GIB=66 KLOC=2048 MTP=0 EMULATE_GPU_GIB= ${DELTA:+ARC_DELTA_FILE=$DELTA} bash "$TRAINING_DIR/serve/serve_eval.sh" \
  $GPUS $PORT $CK --seed 6161 > $L 2>&1 &)
sleep 10; n=0; until grep -a -q -E 'Application startup complete|Traceback|core.py:1366' $L || [ $n -gt 200 ]; do sleep 15; n=$((n+1)); done
grep -a -q 'Application startup complete' $L || { echo "$TAG: server failed"; bash "$TRAINING_DIR/serve/kill_server.sh" $PORT $GPUS; exit 1; }
python3 "$TRAINING_DIR/finetune/overlay_check.py" dump $PORT $D3_WORK/overlay/$TAG.json
grep -a 'ARC_DELTA_FILE' $L | head -2
bash "$TRAINING_DIR/serve/kill_server.sh" $PORT $GPUS
echo "OVLDONE $TAG"
