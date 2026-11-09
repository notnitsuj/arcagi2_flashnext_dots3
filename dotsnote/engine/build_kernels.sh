#!/bin/bash
# Build the five compiled modules of arcstats (offload_attn, offload_attn2, swa_attn, topk_select: nvcc; host_arc: gcc).
#   CU13_HOME  the pip CUDA 13 toolkit's nvidia/cu13 directory (see build_offload_attn.sh)
#   $1         output directory (default: arcstats_pkg/arcstats next to this script)
set -euo pipefail
H=$(cd "$(dirname "$0")" && pwd)
for s in offload_attn offload_attn2 swa_attn topk_select host_arc; do bash $H/build_$s.sh "$@"; done
