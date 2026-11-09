#!/bin/bash
# Build the CPU reducer of the CPU-reduced host all-reduce (ARC_HOST_AR_CPU): csrc/host_arc.c -> arcstats/host_arc.so.
# Baseline x86-64-v2 code; the AVX2 and AVX-512 paths are compiled with target attributes and picked at run time.
#   $1  output directory (default: arcstats_pkg/arcstats next to this script)
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd)/arcstats_pkg/arcstats
OUT=${1:-$D}
gcc -O3 -march=x86-64-v2 -mtune=generic -ffp-contract=off -fno-math-errno -fPIC -shared -Wall \
    -o $OUT/host_arc.so $D/csrc/host_arc.c -lpthread -lm
echo "built $OUT/host_arc.so"
