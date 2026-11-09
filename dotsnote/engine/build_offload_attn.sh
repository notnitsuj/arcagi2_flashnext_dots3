#!/bin/bash
# Build arcstats/offload_attn.so (sm_89 L4 + sm_90 H100) with a pip CUDA 13 toolkit; static cudart.
# Built to a temp file and renamed into place: running servers keep the old (mapped) inode instead of faulting on a
# library rewritten under them.
#   CU13_HOME  the toolkit's nvidia/cu13 directory (default: the one inside the runtime environment, nvcc 13.4)
#   $1         output directory (default: arcstats_pkg/arcstats next to this script)
set -euo pipefail
CU=${CU13_HOME:-/tmp/arc_py312/lib/python3.12/site-packages/nvidia/cu13}
A=$(cd "$(dirname "$0")" && pwd)/arcstats_pkg/arcstats
OUT=${1:-$A}
TMP=$OUT/.offload_attn.so.tmp.$$
PATH=$CU/bin:$PATH nvcc -O3 -std=c++17 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 \
  -shared -Xcompiler -fPIC -cudart static -I$CU/include -L$CU/lib -o $TMP $A/csrc/offload_attn.cu
mv -f $TMP $OUT/offload_attn.so
echo built $OUT/offload_attn.so
