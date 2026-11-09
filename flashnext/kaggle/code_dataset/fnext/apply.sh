#!/bin/bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
OUT=${1:?usage: apply.sh OUT_DIR [SITE_PACKAGES]}
SP=${2:-$ROOT/.venv-fn032/lib/python3.12/site-packages}
PY=${FN_PY:-python3}
[ -d "$SP/vllm" ] || { echo "no vllm in $SP"; exit 1; }
rm -rf "$OUT"; mkdir -p "$OUT"
cp -al "$SP/vllm" "$OUT/vllm" 2>/dev/null || { rm -rf "$OUT/vllm"; cp -a "$SP/vllm" "$OUT/vllm"; }
SUBS=${FN_APPLY_ONLY:-base kv mtp ngram moe session}
mkdir -p "$OUT/fnext_plugins"
REG="def register():\n    import importlib\n"
for s in $SUBS; do
  d=$HERE/$s; [ -d "$d" ] || continue
  for p in $(ls "$d"/*.patch 2>/dev/null | sort); do
    echo "[$s] $(basename $p)"; $PY "$HERE/apply_patch.py" "$OUT" "$p"
  done
  if ls "$d"/*.py >/dev/null 2>&1; then
    mkdir -p "$OUT/fnext_plugins/$s"; cp "$d"/*.py "$OUT/fnext_plugins/$s/"
    [ -f "$OUT/fnext_plugins/$s/__init__.py" ] || touch "$OUT/fnext_plugins/$s/__init__.py"
    if [ -f "$d/plugin.py" ]; then
      REG="$REG    importlib.import_module('fnext_plugins.$s.plugin').register()\n"
      echo "[$s] plugin.py registered"
    fi
  fi
done
printf "$REG" > "$OUT/fnext_plugins/__init__.py"
mkdir -p "$OUT/fnext_plugins-0.dist-info"
printf "Metadata-Version: 2.1\nName: fnext-plugins\nVersion: 0\n" > "$OUT/fnext_plugins-0.dist-info/METADATA"
printf "[vllm.general_plugins]\nfnext = fnext_plugins:register\n" > "$OUT/fnext_plugins-0.dist-info/entry_points.txt"
find "$OUT/vllm" -name "*.pyc" -newer "$OUT/fnext_plugins/__init__.py" -delete 2>/dev/null || true
echo "overlay ready: PYTHONPATH=$OUT (subdirs: $SUBS)"
