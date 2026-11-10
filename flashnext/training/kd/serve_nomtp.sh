#!/bin/bash
# One server of a build without MTP (plugin/fnplug.py: pruning, BF16 experts and trained router rows from
# BUILD/ovr), for prompt log-prob dumps of the served build (served-KL check). Usage: serve_nomtp.sh BUILD GPUS PORT
source "$(dirname "$0")/../env.sh"
B=$1; G=$2; P=$3
export PYTHONPATH=$TRAINING_DIR/plugin${PYTHONPATH:+:$PYTHONPATH} FN_KEEP=$B/keep.pt FN_OVR=$B/ovr
NG=$(echo $G | tr ',' '\n' | wc -l)
CUDA_VISIBLE_DEVICES=$G exec vllm serve $FN_BASE_MODEL --served-model-name m --host 0.0.0.0 --port $P \
  --tensor-parallel-size $NG --enable-expert-parallel --language-model-only \
  --gpu-memory-utilization 0.90 --max-num-seqs 16 --max-model-len 131072 --max-num-batched-tokens 8192 \
  --max-logprobs 20 --attention-config.indexer_kv_dtype fp8
