#!/bin/bash
# Build arcstats/swa_attn.so (single-pass all-FP8 SWA decode attention, csrc/swa_attn.cu; sm_89 L4 + sm_90 H100)
# with a pip CUDA 13 toolkit; static cudart (same recipe as build_offload_attn.sh).
#   CU13_HOME  the toolkit's nvidia/cu13 directory (default: the one inside the runtime environment, nvcc 13.4)
#   $1         output directory (default: arcstats_pkg/arcstats next to this script)
set -euo pipefail
CU=${CU13_HOME:-/tmp/arc_py312/lib/python3.12/site-packages/nvidia/cu13}
A=$(cd "$(dirname "$0")" && pwd)/arcstats_pkg/arcstats
OUT=${1:-$A}
PATH=$CU/bin:$PATH nvcc -O3 -std=c++17 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 \
  -shared -Xcompiler -fPIC -cudart static -Xptxas -v -I$CU/include -L$CU/lib -o $OUT/swa_attn.so $A/csrc/swa_attn.cu \
  2> >(grep -E "error|warning|registers|spill|smem" >&2)
echo built $OUT/swa_attn.so
