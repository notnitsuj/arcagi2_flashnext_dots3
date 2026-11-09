#!/bin/bash
# Step 2: eager FP8 capture server (plugin/fnplug.py with FN_CAP_DIR; MTP 1 so the MTP layer's routing is recorded)
# plus a teacher-forced replay of traces. env: TRACES (space-separated traces.jsonl.gz), HO (held-out task file), TAG,
# SPLIT (position split, 0 = off), CAP ($FN_WORK/cap), CAP_LEN (82000), REPLAY_LEN (81000), GPUS (0,1), PORT (8000).
# SERVE_ONLY=1 only starts the server (for replay_tc.py) and keeps it in the foreground.
source "$(dirname "$0")/../env.sh"
M=$FN_BASE_MODEL
CAP=${CAP:-$FN_WORK/cap}; mkdir -p "$CAP" "$LOGDIR"; rm -f "$CAP/on"
PORT=${PORT:-8000}
export PYTHONPATH=$TRAINING_DIR/plugin${PYTHONPATH:+:$PYTHONPATH} FN_CAP_DIR=$CAP FN_CAP_P=${FN_CAP_P:-0.08} \
  VLLM_USE_DEEP_GEMM=${VLLM_USE_DEEP_GEMM:-0}
trap 'kill $(jobs -p) 2>/dev/null' EXIT
CUDA_VISIBLE_DEVICES=${GPUS:-0,1} vllm serve $M --served-model-name m --host 127.0.0.1 --port $PORT \
  --tensor-parallel-size 2 --enable-expert-parallel --language-model-only --enforce-eager \
  --gpu-memory-utilization 0.85 --max-num-seqs 16 --max-model-len ${CAP_LEN:-82000} --no-enable-prefix-caching \
  --max-num-batched-tokens 16384 --attention-config.indexer_kv_dtype fp8 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' > $LOGDIR/serve_cap.log 2>&1 &
for i in $(seq 1 240); do curl -sf localhost:$PORT/health >/dev/null && break; sleep 10; done
curl -sf localhost:$PORT/health >/dev/null || { echo "server failed"; tail -80 $LOGDIR/serve_cap.log; exit 1; }
grep "\[fnplug\]" $LOGDIR/serve_cap.log | head -5; grep -c "capture hook" $LOGDIR/serve_cap.log
[ "${SERVE_ONLY:-0}" = 1 ] && { echo SERVER_READY; wait; exit 0; }
python3 "$TRAINING_DIR/prune_gptq/replay.py" --traces $TRACES --endpoint 127.0.0.1:$PORT --model-dir $M --cap $CAP \
  --holdout-file $HO --max-len ${REPLAY_LEN:-81000} --conc 16 --tag "${TAG:-}" --done-file $CAP/replayed.txt \
  --split-pos ${SPLIT:-0} 2>&1 | tee $LOGDIR/replay${TAG:-}.log
sleep 20; grep "\[fnplug\]" $LOGDIR/serve_cap.log | tail -5
touch $CAP/DONE; echo CAPTURE_DONE
