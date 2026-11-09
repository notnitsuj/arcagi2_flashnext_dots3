#!/bin/bash
# Full servable checkpoint from train.py's trained.pt: copies the lb144r files (never links them), then writes
# lowbit/lowbit-L*.pt with the trained fp16 scales (codes and zeros unchanged) and rewrites the shards that hold the
# trained norms and router rows. Expects the lowbit/ layout of the prune/GPTQ recipe.
# Usage: build_ckpt.sh RUN_DIR [GPU]   -> RUN_DIR/ckpt      env: CKPT (the lb144r checkpoint, $D3_LB144R)
set -euo pipefail
source "$(dirname "$0")/../env.sh"
RUN=$1; GPU=${2:-0}
SRC=${CKPT:-$D3_LB144R}; OUT=$RUN/ckpt
mkdir -p $OUT/lowbit
for f in $SRC/*; do
  b=$(basename $f); [ -f "$f" ] || continue
  [ -e "$OUT/$b" ] || cp --dereference "$f" "$OUT/$b"
done
CUDA_VISIBLE_DEVICES=$GPU python3 - "$RUN" "$SRC" "$OUT" <<'EOF'
import sys, torch
run, src, out = sys.argv[1:4]
sd = torch.load(f"{run}/trained.pt", map_location="cpu")
layers = sorted({int(k.split(".")[1]) for k in sd if k.endswith(".tgu") or k.endswith(".td")})
n = 0
for l in range(1, 46):
    d = torch.load(f"{src}/lowbit/lowbit-L{l:02d}.pt", map_location="cpu", weights_only=False)
    tg, td = sd.get(f"layers.{l}.mlp.tgu"), sd.get(f"layers.{l}.mlp.td")
    for i, e in enumerate(d["experts"]):
        if tg is not None:
            d[e]["gu_s"] = (d[e]["gu_s"].float() * tg[i].float().exp()).half()
        if td is not None:
            d[e]["d_s"] = (d[e]["d_s"].float() * td[i].float().exp()).half()
    torch.save(d, f"{out}/lowbit/lowbit-L{l:02d}.pt"); n += 1
import re, json
from safetensors.torch import load_file, save_file
names = {"ln1": "input_layernorm.weight", "ln2": "post_attention_layernorm.weight",
         "attn.q_ln": "self_attn.q_a_layernorm.weight", "attn.kv_ln": "self_attn.kv_a_layernorm.weight",
         "attn.kr_ln": "self_attn.k_rope_only_layernorm.weight", "mlp.gate": "mlp.gate.weight"}
upd = {}
for k, v in sd.items():
    if k.endswith(".tgu") or k.endswith(".td"):
        continue
    if k == "norm":
        upd["model.norm.weight"] = v
        continue
    m = re.match(r"layers\.(\d+)\.(.+)$", k)
    assert m and m.group(2) in names, k
    upd[f"model.layers.{m.group(1)}.{names[m.group(2)]}"] = v
if upd:
    wm = json.load(open(f"{src}/model.safetensors.index.json"))["weight_map"]
    for f in sorted({wm[k] for k in upd}):
        t = load_file(f"{src}/{f}")
        for k in upd:
            if wm[k] == f:
                assert t[k].shape == upd[k].shape, k
                t[k] = upd[k].to(t[k].dtype).contiguous()
        save_file(t, f"{out}/{f}", metadata={"format": "pt"})
print(f"wrote {n} lowbit layers (trained scales for {len(layers)} layers); {len(upd)} bf16 tensors updated")
EOF
du -sh $OUT
