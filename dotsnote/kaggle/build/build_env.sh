#!/bin/bash
# Rebuild the runtime's offline Python environment as an archive of /tmp/arc_py312 (the path the launcher unpacks it to):
# a standalone CPython 3.12.15, the 200 pinned packages of requirements-arc-py312.lock.txt (vLLM 0.30.0, PyTorch 2.13.0
# for CUDA 13.0, Triton 3.7.1, FlashInfer 0.6.18.post1, ...), the vLLM patch and the arcstats plugin.
#   usage: build/build_env.sh OUT_TAR      (needs uv and internet; overwrites /tmp/arc_py312)
# Then re-pack it for the dataset: python3 build/build_nb_env.py OUT_TAR <dataset folder>/arc_py312.tarbin
# The launcher installs the dataset's own copies of the arcstats modules and of the patched vLLM files over this
# environment at every start, so the plugin and the patch here only have to be importable.
set -euo pipefail
H=$(cd "$(dirname "$0")" && pwd)
ENGINE=$(cd "$H/../../engine" && pwd)
OUT=$(realpath -m "$1")
uv python install 3.12.15
PYROOT=$(dirname "$(dirname "$(uv python find 3.12.15)")")
rm -rf /tmp/arc_py312
cp -a "$PYROOT" /tmp/arc_py312
rm -f /tmp/arc_py312/lib/python3.12/EXTERNALLY-MANAGED
PY=/tmp/arc_py312/bin/python3.12
uv pip install --python $PY --link-mode=copy -r "$H/requirements-arc-py312.lock.txt" \
  --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
SP=$($PY -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
(cd "$SP" && patch -p1 --forward < "$ENGINE/vllm_patch/vllm-0.30.0.patch")
uv pip install --python $PY --link-mode=copy "$ENGINE/arcstats_pkg"
printf 20261003 > /tmp/arc_py312/BUILD
$PY -c "import vllm, arcstats; print('vllm', vllm.__version__, '+ arcstats plugin')"
tar -C /tmp -cf "$OUT" arc_py312
echo "wrote $OUT"
