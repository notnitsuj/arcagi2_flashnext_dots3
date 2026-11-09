#!/bin/bash
# Validation run of one checkpoint on the 200-task training-split validation shard (finetune/split.json "val", disjoint
# from the fine-tune tasks): serve with the evaluation numerics (TP2, KLOC 2048, MTP 3, 66 GiB budget, fixed seed),
# 4 direct-mode attempts per test input at a 64k cap, then the degeneration test (16 x 12k tokens). Run
# finetune/split.py first. Export ARC_DELTA_FILE to evaluate an overlay instead of a full checkpoint.
# Usage: gate.sh TAG CKPT    env: GPUS (2,3), PORT (8496), ATT (4), SEED (6161), TEMP (0.7)
# Output: $D3_WORK/gate/TAG.jsonl (+ _text.jsonl, .lines = server-log range for MTP acceptance), $LOGDIR/gate_TAG.*.log
source "$(dirname "$0")/../env.sh"
TAG=$1; CK=$2
GPUS=${GPUS:-2,3}; PORT=${PORT:-8496}; L=$LOGDIR/gate_$TAG.srv.log; G=$D3_WORK/gate
mkdir -p $G
(setsid env L4_GIB=66 KLOC=2048 MTP=3 EMULATE_GPU_GIB= bash "$TRAINING_DIR/serve/serve_eval.sh" $GPUS $PORT \
  $CK --seed ${SEED:-6161} > $L 2>&1 &)
sleep 10; n=0; until grep -a -q -E 'Application startup complete|Traceback|core.py:1366' $L || [ $n -gt 200 ]; do sleep 15; n=$((n+1)); done
grep -a -q 'Application startup complete' $L || { echo "$TAG: server failed"; bash "$TRAINING_DIR/serve/kill_server.sh" $PORT $GPUS; exit 1; }
a=$(wc -l < $L)
timeout 43200 python3 "$TRAINING_DIR/harness/direct.py" --ports $PORT --split $(realpath -m "$D3_WORK/ft/val") \
  --attempts ${ATT:-4} --max_tokens 65536 --max_model_len 98304 --concurrency 64 --temp ${TEMP:-0.7} \
  --out $G/$TAG.jsonl --save_text $G/${TAG}_text.jsonl > $LOGDIR/gate_$TAG.gen.log 2>&1
echo "gen rc=$? rows=$(grep -c '"task"' $G/$TAG.jsonl)"
b=$(wc -l < $L)
echo "$a $b" > $G/$TAG.lines
timeout 7200 python3 "$TRAINING_DIR/eval/degen_test.py" $PORT 16 12000 > $LOGDIR/gate_$TAG.degen.log 2>&1
tail -1 $LOGDIR/gate_$TAG.degen.log
bash "$TRAINING_DIR/serve/kill_server.sh" $PORT $GPUS
echo "GATEDONE $TAG"
