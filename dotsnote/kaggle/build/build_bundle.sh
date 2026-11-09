#!/bin/bash
# Assemble the runtime dataset folder from this repository:
#   1. compile the five arcstats kernels (engine/build_kernels.sh; CU13_HOME = a pip CUDA 13 toolkit, default the one
#      inside the runtime environment at /tmp/arc_py312);
#   2. pack cuda_toolkit.tarbin: CUDA 13.0 headers, nvcc and the runtime from pip wheels (the math libraries' shared
#      objects are left out: the launcher links against the environment's own CUDA libraries);
#   3. pack_runtime_dataset.py: every text file byte-identical to dataset version 5, plus the binaries;
#   4. build_nb_bundle.py: kernels without the local toolkit path, the toolkit tar with owner 0:0, a leftover scan.
# usage: build/build_bundle.sh OUT_DIR [ENV_TAR [TRITON_CACHE [DELTA]]]
#   ENV_TAR       arc_py312.tarbin (build_env.sh + build_nb_env.py, or the published one)
#   TRITON_CACHE  triton_cache_lv5.tarbin (the published one, or a cache saved by a run with "triton_cache_save": true)
#   DELTA         p1s_delta.safetensors (the fine-tuned delta; see ../training)
# Needs uv, gcc and internet (pip wheels). The scratch folder is created next to OUT_DIR, so that the large
# binaries are hard-linked rather than copied when they are on the same filesystem.
set -euo pipefail
H=$(cd "$(dirname "$0")" && pwd)
OUT=$(realpath -m "$1"); ENV_TAR=${2:-}; TRITON=${3:-}; DELTA=${4:-}
mkdir -p "$(dirname "$OUT")"
W=$(mktemp -d "$(dirname "$OUT")/.bundle_XXXXXX")
trap 'rm -rf "$W"' EXIT
mkdir -p "$W/kernels" "$W/cuda"
CU=${CU13_HOME:-/tmp/arc_py312/lib/python3.12/site-packages/nvidia/cu13}
CU13_HOME=$CU bash "$H/../../engine/build_kernels.sh" "$W/kernels"
uv pip install --target "$W/cuda" --link-mode=copy --no-deps "nvidia-cuda-nvcc==13.0.88" "nvidia-cuda-crt==13.0.88" \
  "nvidia-cuda-cccl==13.0.85" "nvidia-cuda-runtime==13.0.96" "nvidia-nvvm==13.0.88" "nvidia-cuda-nvrtc==13.4.92" \
  "nvidia-nvjitlink==13.4.92" "nvidia-curand==10.4.0.35" "nvidia-cublas==13.1.1.3" "nvidia-cusparse==12.6.3.3" \
  "nvidia-cusolver==12.0.4.66"
ln -sfn libcudart.so.13 "$W/cuda/nvidia/cu13/lib/libcudart.so"
ln -sfn lib "$W/cuda/nvidia/cu13/lib64"
tar -C "$W/cuda" --exclude='nvidia/cu13/lib/libcublas*' --exclude='nvidia/cu13/lib/libcusparse*' \
  --exclude='nvidia/cu13/lib/libcusolver*' --exclude='nvidia/cu13/lib/libnvblas*' -cf "$W/cuda_toolkit.tarbin" nvidia
ARGS=(--kernels "$W/kernels" --cuda-tar "$W/cuda_toolkit.tarbin")
[ -n "$ENV_TAR" ] && ARGS+=(--env-tar "$ENV_TAR")
[ -n "$TRITON" ] && ARGS+=(--triton-cache "$TRITON")
[ -n "$DELTA" ] && ARGS+=(--delta "$DELTA")
python3 "$H/pack_runtime_dataset.py" "$W/raw" "${ARGS[@]}"
# the kernels record the toolkit path given to nvlink: replace its prefix (and any home path) with a neutral one
python3 "$H/build_nb_bundle.py" "$W/raw" "$OUT" --build-root "$(dirname "$(dirname "$CU")")" --build-root "$HOME"
