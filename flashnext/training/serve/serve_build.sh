#!/bin/bash
# Serve a low-bit build (or the unpruned FP8 model) for evaluation and rollouts with stock vLLM 0.30 + plugin/fnplug.py:
# pruning by load-time renaming, BF16 dequantized experts from BUILD/ovr, trained router rows when ovr/ has them.
# Usage: serve_build.sh BUILD GPUS PORT [extra vllm args]    (BUILD = a build dir with keep.pt, or "fp8")
# env: MTP (3; 0 = off), UTIL (0.90), LEN (131072), SEQS (64)
source "$(dirname "$0")/../env.sh"
B=$1; G=$2; P=$3; shift 3
M=$FN_BASE_MODEL
NG=$(echo $G | tr ',' '\n' | wc -l)
X=()
if [ "$B" != fp8 ]; then
  K=$(python3 -c "import torch;print(int(torch.load('$B/keep.pt')['keep'][0].sum()))")
  export PYTHONPATH=$TRAINING_DIR/plugin${PYTHONPATH:+:$PYTHONPATH} FN_KEEP=$B/keep.pt
  [ -d $B/ovr ] && export FN_OVR=$B/ovr
  echo "serve_build: $B K=$K ovr=${FN_OVR:-none}"
fi
SPEC=()
[ "${MTP:-3}" -gt 0 ] && SPEC=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP:-3}}")
CUDA_VISIBLE_DEVICES=$G exec vllm serve $M --served-model-name m --host 0.0.0.0 --port $P \
  --tensor-parallel-size $NG --enable-expert-parallel --language-model-only \
  --gpu-memory-utilization ${UTIL:-0.90} --max-num-seqs ${SEQS:-64} --max-model-len ${LEN:-131072} \
  --enable-prefix-caching --attention-config.indexer_kv_dtype fp8 "${X[@]}" "${SPEC[@]}" "$@"
