import argparse, glob, json, os, re, shutil
from collections import defaultdict
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
SRC = Path(os.environ.get('D3_BASE_MODEL', 'models/dots3-note-prev-fp8'))
ap = argparse.ArgumentParser()
ap.add_argument('--stats', required=True, help='torch file: {prefix: {count, wsum, tokens}} or a dir of such files')
ap.add_argument('--keep', type=int, required=True)
ap.add_argument('--metric', default='wsum', choices=['wsum', 'count'])
ap.add_argument('--out', required=True)
ap.add_argument('--plan', default=None, help='json {layer: [kept expert ids]} (overrides --keep; per-layer counts ok)')
ap.add_argument('--no-experts', action='store_true', help='omit routed-expert tensors (for arc_lowbit checkpoints)')
a = ap.parse_args()
files = sorted(glob.glob(os.path.join(a.stats, '*.pt'))) if os.path.isdir(a.stats) else [a.stats]
agg = defaultdict(lambda: 0)
for f in files:
    for p, d in torch.load(f).items():
        agg[p] = agg[p] + d[a.metric].double()
keep = {}
for p, v in agg.items():
    layer = int(re.search('layers\\.(\\d+)\\.', p).group(1))
    keep[layer] = sorted(torch.topk(v, a.keep).indices.tolist())
if a.plan:
    keep = {int(k): sorted(v) for k, v in json.load(open(a.plan)).items()}
cfg = json.load(open(SRC / 'config.json'))
n_moe = sum((1 for i in range(cfg['num_hidden_layers']) if i >= cfg['first_k_dense_replace']))
assert len(keep) == n_moe, f'stats cover {len(keep)} MoE layers, expected {n_moe}'
cov = {l: float(agg_v[keep[l]].sum() / agg_v.sum()) for l, agg_v in ((int(re.search('layers\\.(\\d+)\\.', p).group(1)), v) for p, v in agg.items())}
print(f'kept routing mass: min {min(cov.values()):.3f} mean {sum(cov.values()) / len(cov):.3f}')
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
remap = {l: {e: i for i, e in enumerate(ks)} for l, ks in keep.items()}
pat = re.compile('model\\.layers\\.(\\d+)\\.mlp\\.experts\\.(\\d+)\\.(.*)')
weight_map, shard_i, buf, buf_bytes = ({}, 0, {}, 0)

def flush():
    global shard_i, buf, buf_bytes
    if not buf:
        return
    name = f'model-{shard_i:05d}.safetensors'
    save_file(buf, str(out / name), metadata={'format': 'pt'})
    for k in buf:
        weight_map[k] = name
    shard_i += 1
    buf = {}
    buf_bytes = 0
for f in sorted(SRC.glob('model-*.safetensors')):
    with safe_open(str(f), 'pt') as sf:
        for k in sf.keys():
            if 'vision' in k or 'audio' in k:
                continue
            m = pat.match(k)
            if m:
                l, e = (int(m.group(1)), int(m.group(2)))
                if l in remap:
                    if e not in remap[l] or a.no_experts:
                        continue
                    k2 = f'model.layers.{l}.mlp.experts.{remap[l][e]}.{m.group(3)}'
                    t = sf.get_tensor(k)
                else:
                    k2, t = (k, sf.get_tensor(k))
            else:
                t = sf.get_tensor(k)
                k2 = k
                lm = re.match('model\\.layers\\.(\\d+)\\.mlp\\.gate\\.(weight|e_score_correction_bias)$', k)
                if lm and int(lm.group(1)) in keep:
                    t = t[keep[int(lm.group(1))]].contiguous()
            buf[k2] = t
            buf_bytes += t.numel() * t.element_size()
            if buf_bytes > 4000000000.0:
                flush()
flush()
json.dump({'metadata': {}, 'weight_map': weight_map}, open(out / 'model.safetensors.index.json', 'w'))
ks = {l: len(v) for l, v in keep.items()}
cfg['n_routed_experts'] = max(ks.values())
if len(set(ks.values())) > 1:
    cfg['arc_experts_per_layer'] = {str(l): k for l, k in ks.items()}
if a.no_experts:
    cfg['arc_lowbit'] = 'lowbit'
json.dump(cfg, open(out / 'config.json', 'w'), indent=2)
for fn in os.listdir(SRC):
    if not fn.startswith('model') and fn != 'config.json' and (SRC / fn).is_file():
        shutil.copy(SRC / fn, out / fn)
json.dump({'keep': keep, 'coverage': cov}, open(out / 'pruning.json', 'w'))
print('done', out)
