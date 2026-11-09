import argparse, glob, json, os, re, shutil, time
from pathlib import Path
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
SRC = Path(os.environ.get('D3_BASE_MODEL', 'models/dots3-note-prev-fp8'))
ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('--tiers', default='256:3')
ap.add_argument('--layers', default='1-45')
ap.add_argument('--group', type=int, default=128)
ap.add_argument('--gpu', type=int, default=0)
ap.add_argument('--calib', default=os.path.join(os.environ.get('D3_WORK', 'work'), 'calib_acts'))
ap.add_argument('--stats', default=os.path.join(os.environ.get('D3_WORK', 'work'), 'stats_reapsum.pt'))
ap.add_argument('--batch', type=int, default=32, help='experts per GPTQ batch')
ap.add_argument('--damp', type=float, default=0.01)
ap.add_argument('--rtn', action='store_true', help='skip error feedback (RTN with the same grids) for ablation')
ap.add_argument('--assemble', action='store_true')
ap.add_argument('--no-codes', action='store_true', help='skip codes-L*.pt (eval-only runs; codes are ~6 GB/layer)')
ap.add_argument('--pack', default=None, help='write deployable packed experts to PACK/lowbit-L{l}.pt (implies --no-codes)')
ap.add_argument('--no-fp8', action='store_true', help='skip the FP8 fake-quant shards (deployment-only runs)')
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
G = a.group
idx = json.load(open(SRC / 'model.safetensors.index.json'))['weight_map']
pat = re.compile('model\\.layers\\.(\\d+)\\.mlp\\.experts\\.(\\d+)\\.(gate|up|down)_proj\\.weight(_scale_inv)?$')
if a.assemble:
    wm = {}
    for f in sorted(SRC.glob('model-*.safetensors')):
        d = {}
        with safe_open(str(f), 'pt') as sf:
            for k in sf.keys():
                m = pat.match(k)
                if 'vision' in k or 'audio' in k or (m and (out / f'experts-L{int(m.group(1)):02d}.safetensors').exists()):
                    continue
                d[k] = sf.get_tensor(k)
        if d:
            save_file(d, str(out / f'rest-{f.name}'), metadata={'format': 'pt'})
            wm.update({k: f'rest-{f.name}' for k in d})
    for f in sorted(out.glob('experts-L*.safetensors')):
        with safe_open(str(f), 'pt') as sf:
            wm.update({k: f.name for k in sf.keys()})
    json.dump({'metadata': {}, 'weight_map': wm}, open(out / 'model.safetensors.index.json', 'w'))
    for fn in os.listdir(SRC):
        if not fn.startswith('model') and (SRC / fn).is_file():
            shutil.copy(SRC / fn, out / fn)
    print('assembled', out, len(wm), 'tensors')
    raise SystemExit
dev = torch.device(f'cuda:{a.gpu}')
torch.cuda.set_device(dev)
torch.set_num_threads(8)
tiers = [(int(x.split(':')[0]), int(x.split(':')[1])) for x in a.tiers.split(',')]
st = torch.load(a.stats)
rank = {}
for p, v in st.items():
    l = int(re.search('layers\\.(\\d+)\\.', p).group(1))
    o = torch.argsort(v['wsum'].double(), descending=True)
    r = torch.empty_like(o)
    r[o] = torch.arange(len(o))
    rank[l] = r

def bits_of(l, e):
    r = int(rank[l][e])
    for end, b in tiers:
        if r < end:
            return b
    return 0
handles = {}

def get(k):
    f = idx[k]
    if f not in handles:
        handles[f] = safe_open(str(SRC / f), 'pt')
    return handles[f].get_tensor(k)

def deq(k):
    w = get(k).to(dev, torch.float32)
    s = get(k[:-len('weight')] + 'weight_scale_inv').to(dev, torch.float32)
    bo, bi = ((w.shape[0] + s.shape[0] - 1) // s.shape[0], (w.shape[1] + s.shape[1] - 1) // s.shape[1])
    return w * s.repeat_interleave(bo, 0)[:w.shape[0]].repeat_interleave(bi, 1)[:, :w.shape[1]]

def grids(W, bits):
    E, R, C = W.shape
    Wg = W.view(E, R, C // G, G)
    maxq = 2 ** bits - 1
    best = None
    for clip in torch.linspace(1.0, 0.55, 10).tolist():
        lo, hi = (Wg.amin(-1) * clip, Wg.amax(-1) * clip)
        s = ((hi - lo) / maxq).clamp_min(1e-10)
        z = torch.round(-lo / s)
        q = torch.clamp(torch.round(Wg / s[..., None]) + z[..., None], 0, maxq)
        err = ((q - z[..., None]) * s[..., None] - Wg).pow(2).sum(-1)
        if best is None:
            best = [s, z, err]
        else:
            m = err < best[2]
            best = [torch.where(m, s, best[0]), torch.where(m, z, best[1]), torch.where(m, err, best[2])]
    return (best[0], best[1])

def gptq(W, H, bits, n_tok):
    E, R, C = W.shape
    maxq = 2 ** bits - 1
    s, z = grids(W, bits)
    if a.rtn:
        q = torch.clamp(torch.round(W.view(E, R, C // G, G) / s[..., None]) + z[..., None], 0, maxq)
        dq = ((q - z[..., None]) * s[..., None]).view(E, R, C)
        return (dq, q.view(E, R, C).to(torch.uint8), s, z)
    H = H.clone()
    diag = torch.diagonal(H, dim1=1, dim2=2)
    dead = diag == 0
    diag[dead] = 1.0
    W = W.masked_fill(dead[:, None, :], 0.0)
    lam = diag.mean(1) * (a.damp + 4.0 / n_tok.clamp_min(1).to(H.dtype))
    H += lam[:, None, None] * torch.eye(C, device=H.device)
    perm = torch.argsort(torch.diagonal(H, dim1=1, dim2=2), dim=1, descending=True)
    W = torch.gather(W, 2, perm[:, None, :].expand(E, R, C))
    H = torch.gather(torch.gather(H, 1, perm[:, :, None].expand(E, C, C)), 2, perm[:, None, :].expand(E, C, C))
    gid = perm // G
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    Q = torch.zeros_like(W)
    codes = torch.zeros(E, R, C, dtype=torch.uint8, device=W.device)
    ar = torch.arange(E, device=W.device)
    B = 128
    for i1 in range(0, C, B):
        i2 = min(i1 + B, C)
        W1 = W[:, :, i1:i2].clone()
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[:, i1:i2, i1:i2]
        for i in range(i2 - i1):
            w = W1[:, :, i]
            d = Hinv1[:, i, i]
            g = gid[:, i1 + i]
            sc = s[ar, :, g]
            zz = z[ar, :, g]
            q = torch.clamp(torch.round(w / sc) + zz, 0, maxq)
            dqv = (q - zz) * sc
            Q[:, :, i1 + i] = dqv
            codes[:, :, i1 + i] = q.to(torch.uint8)
            err = (w - dqv) / d[:, None]
            W1[:, :, i:] -= err[:, :, None] * Hinv1[:, i, i:][:, None, :]
            Err1[:, :, i] = err
        W[:, :, i2:] -= torch.bmm(Err1, Hinv[:, i1:i2, i2:])
    inv = torch.argsort(perm, dim=1)
    Q = torch.gather(Q, 2, inv[:, None, :].expand(E, R, C))
    codes = torch.gather(codes, 2, inv[:, None, :].expand(E, R, C))
    return (Q, codes, s, z)

def to_fp8_block(w):
    o, i = w.shape
    bo, bi = ((o + 127) // 128, (i + 127) // 128)
    wp = torch.zeros(bo * 128, bi * 128, device=w.device)
    wp[:o, :i] = w
    blk = wp.view(bo, 128, bi, 128).abs().amax((1, 3)).clamp_min(1e-12) / 448.0
    q = (wp.view(bo, 128, bi, 128) / blk[:, None, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return (q.view(bo * 128, bi * 128)[:o, :i].contiguous().cpu(), blk.float().cpu())
lo_l, hi_l = (int(x) for x in a.layers.split('-'))
for l in range(lo_l, hi_l + 1):
    if (out / f'experts-L{l:02d}.safetensors').exists() or (a.pack and (Path(a.pack) / f'lowbit-L{l:02d}.pt').exists()):
        continue
    t0 = time.time()
    xs, ids = ([], [])
    for f in sorted(glob.glob(f'{a.calib}/L{l:02d}_*.pt')):
        d = torch.load(f)
        xs.append(d['x'])
        ids.append(d['ids'])
    X = torch.cat(xs).to(dev)
    I = torch.cat(ids).to(dev).long()
    n_exp = len(rank[l])
    by_bits = {}
    for e in range(n_exp):
        b = bits_of(l, e)
        if b:
            by_bits.setdefault(b, []).append(e)
    tensors, codes_out, stats = ({}, {}, [])
    for b, experts in sorted(by_bits.items()):
        for j in range(0, len(experts), a.batch):
            es = experts[j:j + a.batch]
            Wg = torch.stack([deq(f'model.layers.{l}.mlp.experts.{e}.gate_proj.weight') for e in es])
            Wu = torch.stack([deq(f'model.layers.{l}.mlp.experts.{e}.up_proj.weight') for e in es])
            Wd = torch.stack([deq(f'model.layers.{l}.mlp.experts.{e}.down_proj.weight') for e in es])
            rows = [(I == e).any(1).nonzero().squeeze(1) for e in es]
            n_tok = torch.tensor([len(r) for r in rows], device=dev)
            H1 = torch.stack([X[r].float().T @ X[r].float() if len(r) else torch.zeros(X.shape[1], X.shape[1], device=dev) for r in rows])
            Wgu = torch.cat([Wg, Wu], 1)
            Qgu, Cgu, Sgu, Zgu = gptq(Wgu, H1, b, n_tok)
            del H1
            Qg, Qu = (Qgu[:, :Wg.shape[1]], Qgu[:, Wg.shape[1]:])
            H2 = []
            for k, r in enumerate(rows):
                if len(r):
                    xr = X[r].float()
                    h = F.silu(xr @ Qg[k].T) * (xr @ Qu[k].T)
                    H2.append(h.T @ h)
                else:
                    H2.append(torch.zeros(Wd.shape[2], Wd.shape[2], device=dev))
            Qd, Cd, Sd, Zd = gptq(Wd, torch.stack(H2), b, n_tok)
            for k, e in enumerate(es):
                for nm, W0, Wq in (('gate', Wg[k], Qg[k]), ('up', Wu[k], Qu[k]), ('down', Wd[k], Qd[k])):
                    base = f'model.layers.{l}.mlp.experts.{e}.{nm}_proj.'
                    tensors[base + 'weight'], tensors[base + 'weight_scale_inv'] = to_fp8_block(Wq)
                    stats.append((b, float((Wq - W0).pow(2).mean() / W0.pow(2).mean())))
                keep_dev = bool(a.pack)
                codes_out[e] = {'bits': b, 'gu': Cgu[k] if keep_dev else Cgu[k].cpu(), 'gu_s': Sgu[k].half().cpu(), 'gu_z': Zgu[k].to(torch.uint8).cpu(), 'd': Cd[k] if keep_dev else Cd[k].cpu(), 'd_s': Sd[k].half().cpu(), 'd_z': Zd[k].to(torch.uint8).cpu(), 'n_tok': int(n_tok[k])}
            del Wg, Wu, Wd, Wgu, Qgu, Qd, H2
    for e in range(n_exp):
        if not bits_of(l, e):
            for nm in ('gate', 'up', 'down'):
                k = f'model.layers.{l}.mlp.experts.{e}.{nm}_proj.weight'
                tensors[k] = get(k)
                tensors[k[:-len('weight')] + 'weight_scale_inv'] = get(k[:-len('weight')] + 'weight_scale_inv')
    if a.pack:
        import sys as _sys
        _sys.path.insert(0, os.path.join(os.environ.get('D3_ENGINE', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'engine')), 'arcstats_pkg'))
        from arcstats.lowbit_moe import _pack_planar
        kept = sorted(codes_out)
        pk = {'experts': kept}
        for e in kept:
            c = codes_out[e]
            pk[e] = {'bits': c['bits'], 'gu': {k_: v_.cpu() for k_, v_ in _pack_planar(c['gu'], c['bits']).items()}, 'gu_s': c['gu_s'], 'gu_z': c['gu_z'], 'd': {k_: v_.cpu() for k_, v_ in _pack_planar(c['d'], c['bits']).items()}, 'd_s': c['d_s'], 'd_z': c['d_z']}
        Path(a.pack).mkdir(parents=True, exist_ok=True)
        torch.save(pk, Path(a.pack) / f'lowbit-L{l:02d}.pt')
    if not a.no_fp8:
        save_file(tensors, str(out / f'experts-L{l:02d}.safetensors'), metadata={'format': 'pt'})
    if not a.no_codes and (not a.pack):
        torch.save(codes_out, out / f'codes-L{l:02d}.pt')
    msg = ', '.join((f'{b}b relMSE {sum((x for bb, x in stats if bb == b)) / max(1, sum((1 for bb, _ in stats if bb == b))):.4f}' for b in sorted({b for b, _ in stats})))
    print(f'L{l:02d}: {len(stats) // 3} experts, {X.shape[0]} rows, {time.time() - t0:.0f}s, {msg}', flush=True)
    del X, I
    torch.cuda.empty_cache()
