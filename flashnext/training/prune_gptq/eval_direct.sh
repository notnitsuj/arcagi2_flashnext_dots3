#!/bin/bash
# Direct-mode quality arms: one serve_build.sh server per GPU pair (2i, 2i+1, port 8000+i), then harness/probe/run.py
# (direct answers, 2 attempts, 96k cap) into $FN_WORK/evals/<name>; compare with evstat.py / evci.py.
# Usage: ARMS="name1:BUILD_DIR name2:fp8" [TASKS=file] [MODES=direct] [ATT=2] bash training/prune_gptq/eval_direct.sh
source "$(dirname "$0")/../env.sh"
M=$FN_BASE_MODEL; EV=$ARC_DATA/evaluation
export VLLM_USE_DEEP_GEMM=${VLLM_USE_DEEP_GEMM:-0}
trap 'kill $(jobs -p) 2>/dev/null' EXIT
i=0; declare -a NAMES PORTS RUNS
for nb in $ARMS; do
  n=${nb%%:*}; b=${nb#*:}; g=$((2 * i)),$((2 * i + 1)); p=$((8000 + i))
  bash "$TRAINING_DIR/serve/serve_build.sh" $b $g $p > $LOGDIR/serve_$n.log 2>&1 &
  NAMES[$i]=$n; PORTS[$i]=$p; i=$((i + 1))
done
for j in $(seq 0 $((i - 1))); do
  p=${PORTS[$j]}
  for t in $(seq 1 240); do curl -sf localhost:$p/health >/dev/null && break; sleep 10; done
  curl -sf localhost:$p/health >/dev/null || { echo "server ${NAMES[$j]} failed"; tail -30 $LOGDIR/serve_${NAMES[$j]}.log; }
done
for j in $(seq 0 $((i - 1))); do
  n=${NAMES[$j]}; p=${PORTS[$j]}
  python3 "$TRAINING_DIR/harness/probe/run.py" --endpoints 127.0.0.1:$p --model-dir $M --split $EV \
    --tasks @${TASKS:-$TRAINING_DIR/eval/holdout40.txt} --modes ${MODES:-direct} --attempts ${ATT:-2} --max-tokens 98304 \
    --max-model-len 131072 --concurrency 128 --seed 0 --out $FN_WORK/evals/$n > $LOGDIR/run_$n.log 2>&1 &
  RUNS+=($!)
done
wait "${RUNS[@]}"
