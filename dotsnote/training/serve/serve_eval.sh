#!/bin/bash
# Evaluation server: lb144r (or a checkpoint in its layout) on H100/H200 GPUs with the numerics of the 4xL4 submission
# (portable sparse attention, sparse-aware host KV offload, W8A8 dense GEMMs, packed sliding-window attention, low-bit
# MoE kernels, host all-reduce, SM89 code paths) under a per-GPU memory budget. Tensor parallel over all GPUS.
# Usage: serve_eval.sh GPUS PORT CKPT [extra vllm args]
# env: L4_GIB (per-GPU budget in GiB, 20.9), MTP (draft tokens, 3; 0 = off), KLOC (768), CHUNK (1024), MAXLEN (262144),
#      SEQS (64), CACHE_ROWS (131072), INDEX_SHARE (FFSFSFSFSFFSF), EMULATE_GPU_GIB (21.4; set it empty to turn the
#      emulation off), EXTRA_ENV (more VAR=VAL), ARC_W8A8_CONFIG_DIR (default: the tiles in serve/w8a8_tiles.json,
#      unpacked to $D3_WORK/w8a8_cfg), ARC_DELTA_FILE (serve a fine-tune overlay on top of CKPT)
source "$(dirname "$0")/../env.sh"
gpus=$1; port=$2; ckpt=$3; shift 3
n=$(echo $gpus | tr ',' '\n' | wc -l)
first_gpu=$(echo $gpus | cut -d, -f1)
tot_mib=$(nvidia-smi -i $first_gpu --query-gpu=memory.total --format=csv,noheader,nounits)
UTIL=${UTIL:-$(python3 -c "print(round(${L4_GIB:-20.9} * 1024 / $tot_mib, 4))")}
W8=${ARC_W8A8_CONFIG_DIR:-$D3_WORK/w8a8_cfg}
[ -d "$W8" ] || python3 -c "import json, os, sys; os.makedirs(sys.argv[2], exist_ok=True); [json.dump(c, open(os.path.join(sys.argv[2], f), 'w')) for f, c in json.load(open(sys.argv[1])).items()]" \
  "$TRAINING_DIR/serve/w8a8_tiles.json" "$W8"
SPEC=(); [ "${MTP:-3}" != "0" ] && SPEC=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP:-3}}")
echo "serve_eval: UTIL=$UTIL MTP=${MTP:-3} CHUNK=${CHUNK:-1024} EXTRA_ENV=${EXTRA_ENV:-}"
exec env ARC_FORCE_FP8_MARLIN=1 ARC_EMULATE_SM89=1 ARC_EMULATE_GPU_GIB=${EMULATE_GPU_GIB-21.4} \
  ARC_DOTS3_PORTABLE_DSA=1 ARC_DOTS3_OFFLOAD=1 ARC_OFFLOAD_CACHE_ROWS=${CACHE_ROWS:-131072} ARC_EMBED_HOST=1 \
  ARC_INDEXER_PREFILL_MULT=4 VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128 VLLM_ENGINE_READY_TIMEOUT_S=5400 \
  ARC_FP8_W8A8=1 ARC_W8A8_CONFIG_DIR=$W8 ARC_SWA_FUSED=2 \
  ARC_OFFLOAD_SPLIT_BH=32 ARC_OFFLOAD_SPLIT_WARPS=8 ARC_OFFLOAD_RESOLVE=1 ARC_LOWBIT_PAIRS=1 ARC_HOST_AR=1 \
  NCCL_NVLS_ENABLE=0 ARC_SWA_FP8=1 ARC_SHARD_INDEXER=1 ARC_SHARD_BS=80 ARC_OFFLOAD_WAYS=8 \
  ARC_INDEX_SHARE=${INDEX_SHARE-FFSFSFSFSFFSF} ARC_SHARD_KLOC=${KLOC:-768} ${EXTRA_ENV:-} UTIL=$UTIL \
  bash "$TRAINING_DIR/serve/serve.sh" $gpus $port $ckpt --tensor-parallel-size $n \
  --language-model-only --trust-remote-code --disable-custom-all-reduce --kv-cache-dtype fp8 \
  --max-num-batched-tokens ${CHUNK:-1024} --max-num-seqs ${SEQS:-64} --max-model-len ${MAXLEN:-262144} \
  --block-size 32 --prefix-cache-retention-interval 1280 \
  --kernel-config '{"enable_flashinfer_autotune": false}' --attention-config '{"flash_attn_version": 2}' \
  --compilation-config '{"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[8,16,32,64,96,128,144,160,176,192,256],"pass_config":{"fuse_allreduce_rms":false}}' \
  "${SPEC[@]}" "$@"
