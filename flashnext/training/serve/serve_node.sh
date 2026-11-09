#!/bin/bash
# NGPU/TP servers of one build on this machine (the serve_build.sh recipe: TP + expert parallelism, MTP 3, BF16 experts
# and trained router rows from BUILD/ovr), ports PORT0.., seeds 4000+i. Used for RL rollouts and evaluation pools.
# Usage: BUILD=dir [NGPU=4] [TP=2] [GPU0=0] [PORT0=8000] bash training/serve/serve_node.sh
source "$(dirname "$0")/../env.sh"
B=${BUILD:?set BUILD to a build directory}; N=${NGPU:-4}; TP=${TP:-2}; G0=${GPU0:-0}; P0=${PORT0:-8000}
M=$FN_BASE_MODEL
while [ ! -f "$B/DONE" ]; do sleep 30; done   # export_build.py and buildq.sh write DONE last
export PYTHONPATH=$TRAINING_DIR/plugin${PYTHONPATH:+:$PYTHONPATH} FN_KEEP=$B/keep.pt FN_OVR=$B/ovr
echo "serve_node: build=$B N=$N TP=$TP"
for s in $(seq 0 $((N / TP - 1))); do
  G=$(seq -s, $((G0 + s * TP)) $((G0 + s * TP + TP - 1)))
  CUDA_VISIBLE_DEVICES=$G VLLM_NO_USAGE_STATS=1 vllm serve $M --served-model-name m --host 0.0.0.0 --port $((P0 + s)) \
    --tensor-parallel-size $TP --enable-expert-parallel --language-model-only \
    --gpu-memory-utilization 0.90 --max-num-seqs 64 --max-model-len 131072 \
    --enable-prefix-caching --attention-config.indexer_kv_dtype fp8 --seed $((4000 + s)) \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' > "$LOGDIR/serve_$(basename $B)_$((P0 + s)).log" 2>&1 &
  sleep 5
done
wait
