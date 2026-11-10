#!/bin/bash
# Distillation teacher: NGPU/TP servers of the unpruned FP8 model without MTP, for top-20 prompt log-prob dumps.
# Usage: [NGPU=8] [TP=2] [GPU0=0] [PORT0=8000] bash training/kd/serve_teacher.sh    (env SEQS (32), MBT (8192))
source "$(dirname "$0")/../env.sh"
N=${NGPU:-8}; TP=${TP:-2}; G0=${GPU0:-0}; P0=${PORT0:-8000}
M=$FN_BASE_MODEL
echo "serve_teacher: N=$N TP=$TP"
for s in $(seq 0 $((N / TP - 1))); do
  G=$(seq -s, $((G0 + s * TP)) $((G0 + s * TP + TP - 1)))
  CUDA_VISIBLE_DEVICES=$G VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 vllm serve $M --served-model-name m --host 0.0.0.0 \
    --port $((P0 + s)) --tensor-parallel-size $TP --enable-expert-parallel --language-model-only \
    --gpu-memory-utilization 0.90 --max-num-seqs ${SEQS:-32} --max-model-len 131072 \
    --max-num-batched-tokens ${MBT:-8192} --max-logprobs 20 \
    --attention-config.indexer_kv_dtype fp8 --seed $((3000 + s)) > "$LOGDIR/teacher_$((P0 + s)).log" 2>&1 &
  sleep 5
done
wait
