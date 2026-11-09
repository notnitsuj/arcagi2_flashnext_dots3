#!/bin/bash
# Step 1: the FP8 model writes its own traces (direct + REPL, one attempt each) on ARC-AGI-2 training tasks.
# Usage: [ENDPOINTS=host:port,...] [OUT=$FN_WORK/traces] [NTASK=320] [MAXTOK=65536] bash training/prune_gptq/traces.sh
# Without ENDPOINTS it starts one TP2 server per GPU pair in GPU_PAIRS ("0,1 2,3") with context SERVE_LEN (81920).
source "$(dirname "$0")/../env.sh"
M=$FN_BASE_MODEL
OUT=${OUT:-$FN_WORK/traces}; mkdir -p "$OUT" "$LOGDIR"
TR=$ARC_DATA/training; EV=$ARC_DATA/evaluation
[ -s "$OUT/tasks.txt" ] || python3 - <<PY
import os, random
tr = sorted(f[:-5] for f in os.listdir("$TR") if f.endswith(".json"))
ev = set(f[:-5] for f in os.listdir("$EV") if f.endswith(".json"))
assert not (set(tr) & ev), "train/eval overlap"
sel = random.Random(2027).sample(tr, int("${NTASK:-320}"))
open("$OUT/tasks.txt", "w").write("\n".join(sel) + "\n")
print(len(sel), "training tasks; eval overlap 0")
PY
start() {  # gpus port seed
  CUDA_VISIBLE_DEVICES=$1 vllm serve $M --served-model-name m --host 127.0.0.1 --port $2 \
    --tensor-parallel-size 2 --enable-expert-parallel --language-model-only --trust-remote-code \
    --gpu-memory-utilization 0.90 --max-num-seqs 64 --max-model-len ${SERVE_LEN:-81920} --enable-prefix-caching \
    --attention-config.indexer_kv_dtype fp8 --seed $3 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' > $LOGDIR/serve_$2.log 2>&1 &
}
if [ -z "${ENDPOINTS:-}" ]; then
  trap 'kill $(jobs -p) 2>/dev/null' EXIT
  i=0; EPS=()
  for g in ${GPU_PAIRS:-0,1 2,3}; do start $g $((8000 + i)) $((11 + i)); EPS+=(127.0.0.1:$((8000 + i))); i=$((i + 1)); done
  for e in "${EPS[@]}"; do
    for t in $(seq 1 180); do curl -sf $e/health >/dev/null && break; sleep 10; done
    curl -sf $e/health >/dev/null || { echo "server $e failed"; tail -50 $LOGDIR/serve_${e##*:}.log; exit 1; }
  done
  ENDPOINTS=$(IFS=,; echo "${EPS[*]}"); MAXLEN=${SERVE_LEN:-81920}
fi
echo "endpoints $ENDPOINTS $(date)"
python3 "$TRAINING_DIR/harness/probe/run.py" --endpoints $ENDPOINTS --model-dir $M --split $TR \
  --tasks @$OUT/tasks.txt --modes direct,repl --attempts 1 --max-tokens ${MAXTOK:-65536} --max-model-len ${MAXLEN:-131072} \
  --concurrency ${CONC:-24} --seed 5 --out $OUT 2>&1 | tee $LOGDIR/traces_$(basename $OUT).log
echo "TRACES_DONE $(date)"; touch $OUT/DONE
