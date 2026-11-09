#!/bin/bash
# Plain vLLM server for the FP8 base model or any checkpoint.
# Usage: serve.sh GPUS PORT MODEL [extra vllm args]
# env: UTIL (gpu-memory-utilization, 0.92), VENV (optional virtualenv to activate), CUDA_HOME (else /usr/local/cuda*).
# NCCL peer-to-peer and FlashInfer / symmetric-memory all-reduce are off: they hung on the 8xH100 host we used
# (harmless elsewhere).
gpu=$1; port=$2; model=$3; shift 3
[ -n "${VENV:-}" ] && . "$VENV/bin/activate"
if [ -z "${CUDA_HOME:-}" ]; then
  for c in /usr/local/cuda $(ls -d /usr/local/cuda-* 2>/dev/null | sort -V -r); do
    [ -x $c/bin/nvcc ] && { export CUDA_HOME=$c; break; }
  done
fi
NCCL_P2P_DISABLE=1 VLLM_ALLREDUCE_USE_FLASHINFER=0 VLLM_ALLREDUCE_USE_SYMM_MEM=0 VLLM_FLASHINFER_ALLREDUCE_BACKEND=mnnvl \
  CUDA_VISIBLE_DEVICES=$gpu exec vllm serve $model --port $port --served-model-name m \
  --max-model-len 131072 --gpu-memory-utilization ${UTIL:-0.92} --max-num-seqs 64 \
  --enable-prefix-caching "$@"
