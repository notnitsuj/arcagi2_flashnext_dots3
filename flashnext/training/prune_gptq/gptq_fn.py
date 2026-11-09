import argparse, glob, json, os, re, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
ap = argparse.ArgumentParser()
ap.add_argument('mode', choices=['keep', 'build', 'eval'])
ap.add_argument('--cap', required=True)
ap.add_argument('--src', default=os.environ.get('FN_BASE_MODEL', 'models/Qwen3.8-Flash-Next-FP8'))
ap.add_argument('--out')
ap.add_argument('--keep')
ap.add_argument('--ks', default='384,256,192,128')
ap.add_argument('--crit', default='reapsum', choices=['reapsum', 'wsum', 'freq'])
ap.add_argument('--tiers', default='all:3')
ap.add_argument('--layers', default='0-48')
ap.add_argument('--gpu', type=int, default=0)
ap.add_argument('--weighted', type=int, default=1)
ap.add_argument('--damp', type=float, default=0.01)
ap.add_argument('--batch', type=int, default=32)
ap.add_argument('--rtn', action='store_true')
ap.add_argument('--no-ovr', action='store_true')
ap.add_argument('--max-rows', type=int, default=0)
ap.add_argument('--cand', default='')
a = ap.parse_args()
G, TOPK, NE, HID, INTER, MTP = (128, 10, 512, 2560, 640, 48)
SRC = Path(a.src)
idx = json.load(open(SRC / 'model.safetensors.index.json'))['weight_map']
dev = torch.device(f'cuda:{a.gpu}')
torch.cuda.set_device(dev)
torch.set_num_threads(8)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
handles = {}

def pre(l):
    return 'mtp.layers.0' if l == MTP else f'model.language_model.layers.{l}'

def get(k):
    f = idx[k]
    if f not in handles:
        handles[f] = safe_open(str(SRC / f), 'pt')
    return handles[f].get_tensor(k)

def deq(k):
    w = get(k).to(dev)
    if w.dtype != torch.float8_e4m3fn:
        return w.float()
    w = w.float()
    s = get(k[:-len('weight')] + 'weight_scale_inv').to(dev, torch.float32)
    bo, bi = ((w.shape[0] + s.shape[0] - 1) // s.shape[0], (w.shape[1] + s.shape[1] - 1) // s.shape[1])
    return w * s.repeat_interleave(bo, 0)[:w.shape[0]].repeat_interleave(bi, 1)[:, :w.shape[1]]

def expert_w(l, es):
    p = pre(l)
    return [torch.stack([deq(f'{p}.mlp.experts.{e}.{nm}_proj.weight') for e in es]) for nm in ('gate', 'up', 'down')]

def load_stats(cap):
    out = None
    for d in cap.split(','):
        s = torch.load(f'{d}/stats.pt')
        if out is None:
            out = s
        else:
            for l in s['layers']:
                out['counts'][l] = out['counts'][l] + s['counts'][l]
                out['wsum'][l] = out['wsum'][l] + s['wsum'][l]
                out['tokens'][l] = out['tokens'][l] + s['tokens'][l]
    return out

def load_cap(l, cap):
    xs, ls = ([], [])
    for f in sorted(sum((glob.glob(f'{d}/L{l:02d}_*.pt') for d in cap.split(',')), [])):
        d = torch.load(f)
        xs.append(d['x'])
        ls.append(d['logits'])
    X, LG = (torch.cat(xs), torch.cat(ls))
    if a.max_rows and X.shape[0] > a.max_rows:
        sel = torch.randperm(X.shape[0], generator=torch.Generator().manual_seed(0))[:a.max_rows]
        X, LG = (X[sel], LG[sel])
    return (X.to(dev), LG.to(dev).float())

def route(LG, keep_ids=None):
    if keep_ids is not None:
        LG = LG[:, keep_ids]
    w, ids = torch.topk(torch.softmax(LG, -1), TOPK, -1)
    return (ids, w / w.sum(-1, keepdim=True))

def expert_fwd(x, wg, wu, wd):
    return F.silu(x @ wg.T) * (x @ wu.T) @ wd.T

def layers():
    lo, hi = (int(x) for x in a.layers.split('-'))
    return range(lo, hi + 1)

def mode_keep():
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    st = load_stats(a.cap)
    scores = {}
    for l in layers():
        t0 = time.time()
        freq, wsum = (st['counts'][l].double(), st['wsum'][l].double())
        if a.crit == 'reapsum':
            X, LG = load_cap(l, a.cap)
            ids, w = route(LG)
            rs = torch.zeros(NE, dtype=torch.float64, device=dev)
            for j in range(0, NE, a.batch):
                es = list(range(j, j + a.batch))
                Wg, Wu, Wd = expert_w(l, es)
                for k, e in enumerate(es):
                    tok, slot = (ids == e).nonzero(as_tuple=True)
                    if len(tok) == 0:
                        continue
                    y = expert_fwd(X[tok].float(), Wg[k], Wu[k], Wd[k])
                    rs[e] = (w[tok, slot].double() * y.norm(dim=-1).double()).sum()
            sc = rs.cpu() * (st['tokens'][l] / X.shape[0])
            del X, LG
        else:
            sc = wsum if a.crit == 'wsum' else freq
        scores[l] = sc
        print(f'L{l}: scored in {time.time() - t0:.0f}s; top share of wsum at 256: {float(wsum[torch.argsort(sc, descending=True)[:256]].sum() / wsum.sum()):.3f}', flush=True)
    torch.save(scores, out / f'scores_{a.crit}.pt')
    for K in [int(k) for k in a.ks.split(',')]:
        keep = torch.zeros(MTP + 1, NE, dtype=torch.bool)
        cov = []
        for l in layers():
            keep[l, torch.argsort(scores[l], descending=True)[:K]] = True
            ws = st['wsum'][l].double()
            cov.append(float(ws[keep[l]].sum() / ws.sum()))
        torch.save({'keep': keep, 'crit': a.crit, 'K': K}, out / f'keep_{a.crit}_{K}.pt')
        print(f'K={K}: routed-weight coverage mean {sum(cov) / len(cov):.4f} min {min(cov):.4f}', flush=True)

def grids(W, bits):
    E, R, C = W.shape
    Wg = W.view(E, R, C // G, G)
    maxq = 2 ** bits - 1
    best = None
    for clip in torch.linspace(1.0, 0.55, 10).tolist():
        lo, hi = (Wg.amin(-1) * clip, Wg.amax(-1) * clip)
        s = ((hi - lo) / maxq).clamp_min(1e-10)
        z = torch.round(-lo / s).clamp(0, maxq)
        q = torch.clamp(torch.round(Wg / s[..., None]) + z[..., None], 0, maxq)
        err = ((q - z[..., None]) * s[..., None] - Wg).pow(2).sum(-1)
        if best is None:
            best = [s, z, err]
        else:
            m = err < best[2]
            best = [torch.where(m, s, best[0]), torch.where(m, z, best[1]), torch.where(m, err, best[2])]
    return (best[0].half().float().clamp_min(6e-08), best[1])

def gptq(W, H, bits, n_tok):
    E, R, C = W.shape
    maxq = 2 ** bits - 1
    s, z = grids(W, bits)
    if a.rtn or H is None:
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
    for tr in range(12):
        try:
            L = torch.linalg.cholesky(H)
            Hinv = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)
            break
        except RuntimeError:
            H += (diag.mean(1) * 0.01 * 2 ** tr)[:, None, None] * torch.eye(C, device=H.device)
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

def tiers_for(K):
    out = []
    for t in a.tiers.split(','):
        end, b = t.split(':')
        out.append((K if end == 'all' else int(end), int(b)))
    return out

def mode_build():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from lowbit_pack import _pack_planar
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'ovr').mkdir(exist_ok=True)
    keep = torch.load(a.keep)['keep'].bool()
    torch.save({'keep': keep, 'src': a.keep, 'tiers': a.tiers, 'rtn': a.rtn, 'weighted': a.weighted}, out / 'keep.pt')
    st = load_stats(a.cap) if a.cap != 'none' else None
    kj = {}
    for l in layers():
        if (out / f'lowbit-L{l:02d}.pt').exists():
            continue
        t0 = time.time()
        kept = keep[l].nonzero().squeeze(1).tolist()
        K = len(kept)
        kj[l] = kept
        wsum = st['wsum'][l].double()[kept] if st else -torch.arange(K, dtype=torch.float64)
        order = torch.argsort(wsum, descending=True)
        rank = torch.empty_like(order)
        rank[order] = torch.arange(K)
        tiers = tiers_for(K)
        bits = []
        for j in range(K):
            b = next((bb for end, bb in tiers if int(rank[j]) < end))
            bits.append(b)
        X = LG = None
        if not a.rtn:
            X, LG = load_cap(l, a.cap)
            ids, w = route(LG, torch.tensor(kept, device=dev))
            del LG
        pk = {'experts': kept}
        ovr, stats = ({}, [])
        gwt = get(f'{pre(l)}.mlp.gate.weight')
        torch.save({'weight': gwt[kept].clone(), 'keep': kept}, out / f'router-L{l:02d}.pt')
        for b in sorted(set(bits)):
            js = [j for j in range(K) if bits[j] == b]
            for i0 in range(0, len(js), a.batch):
                jb = js[i0:i0 + a.batch]
                es = [kept[j] for j in jb]
                Wg, Wu, Wd = expert_w(l, es)
                Wgu = torch.cat([Wg, Wu], 1)
                H1 = H2 = None
                n_tok = torch.zeros(len(jb), device=dev)
                rows = []
                if X is not None:
                    H1 = torch.zeros(len(jb), HID, HID, device=dev)
                    for k, j in enumerate(jb):
                        tok, slot = (ids == j).nonzero(as_tuple=True)
                        rows.append((tok, w[tok, slot]))
                        n_tok[k] = len(tok)
                        if len(tok):
                            xr = X[tok].float()
                            if a.weighted:
                                xr = xr * w[tok, slot][:, None]
                            H1[k] = xr.T @ xr
                Qgu, Cgu, Sgu, Zgu = gptq(Wgu, H1, b, n_tok)
                del H1
                Qg, Qu = (Qgu[:, :INTER], Qgu[:, INTER:])
                if X is not None:
                    H2 = torch.zeros(len(jb), INTER, INTER, device=dev)
                    for k, (tok, ww) in enumerate(rows):
                        if len(tok):
                            xr = X[tok].float()
                            h = F.silu(xr @ Qg[k].T) * (xr @ Qu[k].T)
                            if a.weighted:
                                h = h * ww[:, None]
                            H2[k] = h.T @ h
                Qd, Cd, Sd, Zd = gptq(Wd, H2, b, n_tok)
                del H2
                for k, j in enumerate(jb):
                    e = kept[j]
                    for nm, W0, Wq in (('gate', Wg[k], Qg[k]), ('up', Wu[k], Qu[k]), ('down', Wd[k], Qd[k])):
                        ovr[f'{pre(l)}.mlp.experts.{j}.{nm}_proj.weight'] = Wq.to(torch.bfloat16).cpu()
                        stats.append((b, nm, float((Wq - W0).pow(2).sum() / W0.pow(2).sum()), int(n_tok[k])))
                    pk[e] = {'bits': b, 'gu': {kk: v.cpu() for kk, v in _pack_planar(Cgu[k], b).items()}, 'gu_s': Sgu[k].half().cpu(), 'gu_z': Zgu[k].to(torch.uint8).cpu(), 'd': {kk: v.cpu() for kk, v in _pack_planar(Cd[k], b).items()}, 'd_s': Sd[k].half().cpu(), 'd_z': Zd[k].to(torch.uint8).cpu()}
                del Wg, Wu, Wd, Wgu, Qgu, Qd
        if not a.no_ovr:
            save_file(ovr, str(out / 'ovr' / f'ovr-L{l:02d}.safetensors'), metadata={'format': 'pt'})
        torch.save(pk, out / f'lowbit-L{l:02d}.pt')
        summ = {}
        for b, nm, e, n in stats:
            s_ = summ.setdefault(f'{b}b', {'n': 0, 'wmse': 0.0, 'rows': []})
            s_['n'] += 1
            s_['wmse'] += e
            s_['rows'].append(n)
        for v in summ.values():
            v['wmse'] /= v['n']
            r = sorted(v['rows'])
            v['rows'] = {'min': r[0], 'med': r[len(r) // 2]}
        q = [sum((bits[j] for j in range(K) if j * 4 // K == r)) for r in range(4)]
        json.dump({'layer': l, 'K': K, 'bits': {str(bb): bits.count(bb) for bb in set(bits)}, 'summary': summ, 'ep4_quarter_bitsum': q, 'ep4_imbalance': max(q) / (sum(q) / 4) - 1}, open(out / f'gptq-L{l:02d}.json', 'w'))
        del X
        torch.cuda.empty_cache()
        print(f"L{l}: K={K} bits {dict(((bb, bits.count(bb)) for bb in set(bits)))} { {k: round(v['wmse'], 4) for k, v in summ.items()}} {time.time() - t0:.0f}s", flush=True)
    json.dump({str(l): keep[l].nonzero().squeeze(1).tolist() for l in range(keep.shape[0])}, open(out / 'keep.json', 'w'))

def mode_eval():
    keep = torch.load(a.keep)['keep'].bool()
    cands = dict((c.split('=', 1) for c in a.cand.split(',') if c))
    fo = open(a.out, 'a')
    for l in layers():
        X, LG = load_cap(l, a.cap)
        X = X.float()
        ids0, w0 = route(LG)
        kept = keep[l].nonzero().squeeze(1)
        ids1, w1 = route(LG, kept.to(dev))
        ids1o = kept.to(dev)[ids1]
        y0 = torch.zeros(X.shape[0], HID, device=dev)
        yp = {'prune_fp8': torch.zeros_like(y0)} | {c: torch.zeros_like(y0) for c in cands}
        hs = {c: safe_open(f'{d}/ovr/ovr-L{l:02d}.safetensors', 'pt') for c, d in cands.items()}
        kl = kept.tolist()
        loc = {e: j for j, e in enumerate(kl)}
        for j0 in range(0, NE, a.batch):
            es = list(range(j0, j0 + a.batch))
            Wg, Wu, Wd = expert_w(l, es)
            for k, e in enumerate(es):
                tok, slot = (ids0 == e).nonzero(as_tuple=True)
                if len(tok):
                    y0.index_add_(0, tok, w0[tok, slot][:, None] * expert_fwd(X[tok], Wg[k], Wu[k], Wd[k]))
                if e in loc:
                    tok, slot = (ids1o == e).nonzero(as_tuple=True)
                    if len(tok):
                        xr, ww = (X[tok], w1[tok, slot][:, None])
                        yp['prune_fp8'].index_add_(0, tok, ww * expert_fwd(xr, Wg[k], Wu[k], Wd[k]))
                        for c, h in hs.items():
                            p = f'{pre(l)}.mlp.experts.{loc[e]}.'
                            q = [h.get_tensor(p + f'{nm}_proj.weight').to(dev).float() for nm in ('gate', 'up', 'down')]
                            yp[c].index_add_(0, tok, ww * expert_fwd(xr, *q))
        den = float(y0.pow(2).sum())
        rec = {'layer': l, 'rows': X.shape[0]} | {c: float((v - y0).pow(2).sum()) / den for c, v in yp.items()}
        fo.write(json.dumps(rec) + '\n')
        fo.flush()
        print(rec, flush=True)
        del X, LG, y0, yp
        torch.cuda.empty_cache()
{'keep': mode_keep, 'build': mode_build, 'eval': mode_eval}[a.mode]()
