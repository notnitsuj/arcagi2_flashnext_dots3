import argparse, glob, json, os, re, time
from pathlib import Path
import torch
import torch.nn.functional as F
from safetensors import safe_open
SRC = Path(os.environ.get('D3_BASE_MODEL', 'models/dots3-note-prev-fp8'))
ap = argparse.ArgumentParser()
ap.add_argument('--out', default=os.path.join(os.environ.get('D3_WORK', 'work'), 'reap_scores.pt'))
ap.add_argument('--gpu', type=int, default=0)
ap.add_argument('--layers', default='1-45')
ap.add_argument('--calib', default=os.path.join(os.environ.get('D3_WORK', 'work'), 'calib_acts'))
ap.add_argument('--max-tok', type=int, default=0, help='subsample tokens per layer (0 = all)')
ap.add_argument('--masks', default=None, help='comma list of k: write uniform top-k REAP masks and exit')
ap.add_argument('--mask-dir', default=os.path.join(os.environ.get('D3_WORK', 'work'), 'masks'))
ap.add_argument('--metric', default='reap', help='reap | reapsum (= reap*count) | ean | wsum | count')
a = ap.parse_args()
if a.masks:
    S = torch.load(a.out)
    for k in [int(x) for x in a.masks.split(',')]:
        sc = {l: S[l]['reap'] * S[l]['count'] if a.metric == 'reapsum' else S[l][a.metric] for l in S}
        plan = {str(l): torch.topk(sc[l], k).indices.sort().values.tolist() for l in sorted(S)}
        Path(a.mask_dir).mkdir(parents=True, exist_ok=True)
        p = Path(a.mask_dir) / f'{a.metric}{k}.json'
        json.dump(plan, open(p, 'w'))
        print('wrote', p)
    raise SystemExit
idx = json.load(open(SRC / 'model.safetensors.index.json'))['weight_map']
dev = torch.device(f'cuda:{a.gpu}')
torch.cuda.set_device(dev)
handles = {}

def get(k):
    f = idx[k]
    if f not in handles:
        handles[f] = safe_open(str(SRC / f), 'pt')
    return handles[f].get_tensor(k)

def deq(k):
    w = get(k).to(dev, torch.float32)
    if w.dtype == torch.float32 and k[:-len('weight')] + 'weight_scale_inv' not in idx:
        return w
    s = get(k[:-len('weight')] + 'weight_scale_inv').to(dev, torch.float32)
    bo, bi = ((w.shape[0] + s.shape[0] - 1) // s.shape[0], (w.shape[1] + s.shape[1] - 1) // s.shape[1])
    return w * s.repeat_interleave(bo, 0)[:w.shape[0]].repeat_interleave(bi, 1)[:, :w.shape[1]]
res = torch.load(a.out) if os.path.exists(a.out) else {}
lo, hi = (int(x) for x in a.layers.split('-'))
for l in range(lo, hi + 1):
    if l in res:
        continue
    t0 = time.time()
    xs, ids0 = ([], [])
    for f in sorted(glob.glob(f'{a.calib}/L{l:02d}_*.pt')):
        d = torch.load(f)
        xs.append(d['x'])
        ids0.append(d['ids'])
    X = torch.cat(xs)
    I0 = torch.cat(ids0).long()
    if a.max_tok and X.shape[0] > a.max_tok:
        sel = torch.randperm(X.shape[0], generator=torch.Generator().manual_seed(0))[:a.max_tok]
        X, I0 = (X[sel], I0[sel])
    X = X.to(dev)
    I0 = I0.to(dev)
    Wg = get(f'model.layers.{l}.mlp.gate.weight').to(dev, torch.bfloat16)
    bias = get(f'model.layers.{l}.mlp.gate.e_score_correction_bias').to(dev, torch.float32)
    E = Wg.shape[0]
    T = X.shape[0]
    ids = torch.empty(T, 8, dtype=torch.long, device=dev)
    w = torch.empty(T, 8, device=dev)
    for i in range(0, T, 32768):
        sc = torch.sigmoid((X[i:i + 32768] @ Wg.T).float())
        tid = torch.topk(sc + bias, 8, dim=-1).indices
        g = sc.gather(1, tid)
        ids[i:i + 32768] = tid
        w[i:i + 32768] = g / g.sum(-1, keepdim=True)
    agree = (ids.sort(1).values == I0.sort(1).values).all(1).float().mean().item()
    flat_e, flat_t = (ids.reshape(-1), torch.arange(T, device=dev).repeat_interleave(8))
    flat_w = w.reshape(-1)
    order = torch.argsort(flat_e)
    flat_e, flat_t, flat_w = (flat_e[order], flat_t[order], flat_w[order])
    cnt = torch.bincount(flat_e, minlength=E)
    starts = torch.cumsum(cnt, 0) - cnt
    reap = torch.zeros(E, dtype=torch.float64)
    ean = torch.zeros(E, dtype=torch.float64)
    wsum = torch.zeros(E, dtype=torch.float64)
    for e in range(E):
        n = int(cnt[e])
        if n == 0:
            continue
        s = int(starts[e])
        tt = flat_t[s:s + n]
        ww = flat_w[s:s + n]
        p = f'model.layers.{l}.mlp.experts.{e}.'
        Wgate = deq(p + 'gate_proj.weight').to(torch.bfloat16)
        Wup = deq(p + 'up_proj.weight').to(torch.bfloat16)
        Wdn = deq(p + 'down_proj.weight').to(torch.bfloat16)
        x = X[tt]
        y = F.silu(x @ Wgate.T) * (x @ Wup.T) @ Wdn.T
        nrm = y.float().norm(dim=-1)
        reap[e] = float((ww * nrm).sum()) / n
        ean[e] = float(nrm.sum())
        wsum[e] = float(ww.sum())
    res[l] = {'reap': reap, 'ean': ean, 'count': cnt.double().cpu(), 'wsum': wsum, 'tokens': T, 'id_agree': agree}
    torch.save(res, a.out)
    print(f'L{l:02d}: {T} tok, id agree {agree:.4f}, {time.time() - t0:.0f}s', flush=True)
    del X, I0
    torch.cuda.empty_cache()
print('done')
