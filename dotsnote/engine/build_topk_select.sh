#!/bin/bash
# Build arcstats/topk_select.so (exact radix top-k select + merge for the sharded indexer, csrc/topk_select.cu;
# sm_89 L4 + sm_90 H100) with a pip CUDA 13 toolkit; static cudart (same recipe as build_offload_attn.sh).
#   CU13_HOME  the toolkit's nvidia/cu13 directory (default: the one inside the runtime environment, nvcc 13.4)
#   $1         output directory (default: arcstats_pkg/arcstats next to this script)
set -euo pipefail
CU=${CU13_HOME:-/tmp/arc_py312/lib/python3.12/site-packages/nvidia/cu13}
A=$(cd "$(dirname "$0")" && pwd)/arcstats_pkg/arcstats
OUT=${1:-$A}
PATH=$CU/bin:$PATH nvcc -O3 -std=c++17 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 \
  -shared -Xcompiler -fPIC -cudart static -Xptxas -v -I$CU/include -L$CU/lib -o $OUT/topk_select.so $A/csrc/topk_select.cu \
  2> >(grep -E "error|warning|[1-9][0-9]* bytes spill" >&2 || true)
echo built $OUT/topk_select.so
