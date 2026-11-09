# Source me before draft_head/dump.py and vacc.py: vLLM 0.32.0 plus the serving plugin overlay of
# kaggle/code_dataset/fnext/, configured as deployed (low-bit experts, FP8 dense projections, FP8 KV, MTP).
# Needs VLLM032_SP (site-packages of a vLLM 0.32.0 environment) and DRAFT_TARGET (the low-bit build to run).
# CUDA_HOME must point to a CUDA 13 toolkit (Triton / NVRTC compile the low-bit kernels at start-up).
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
REPO_DIR=$(cd "$TRAINING_DIR/.." && pwd)
SP=${VLLM032_SP:?set VLLM032_SP to the site-packages directory of a vLLM 0.32.0 environment}
OVL=${OVERLAY:-$FN_WORK/overlay}
[ -d "$OVL/fnext_plugins" ] || bash "$REPO_DIR/kaggle/code_dataset/fnext/apply.sh" "$OVL" "$SP"
export PYTHONPATH=$OVL${PYTHONPATH:+:$PYTHONPATH} LD_LIBRARY_PATH=$SP/nvidia/cu13/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export VLLM_ENABLE_V1_MULTIPROCESSING=0 FN_MOE_LOWBIT=${DRAFT_TARGET:?set DRAFT_TARGET to a low-bit build directory} \
  FN_MOE_PAIRS=1 FN_MOE_MARLIN4=1 FN_DENSE_FP8=1 VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128 OMP_NUM_THREADS=8
