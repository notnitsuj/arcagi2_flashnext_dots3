import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mtp_torch import MTP
MODEL = os.environ.get('FN_BASE_MODEL', 'models/Qwen3.8-Flash-Next-FP8')
BUILD = os.environ.get('FN_DRAFT_BUILD', os.path.join(os.environ.get('FN_WORK', 'work'), 'builds', 'rl1'))
HOT = os.environ.get('FN_HOT_VOCAB', os.path.join(os.environ.get('FN_WORK', 'work'), 'hot32k_tr.pt'))

def target_probs(tid, tlp, top_p=0.95):
    p = torch.softmax(tlp.float(), -1)
    asc = p.flip(-1)
    cum = asc.cumsum(-1)
    drop = (cum <= 1 - top_p).flip(-1)
    drop[..., 0] = False
    p = p.masked_fill(drop, 0)
    return p / p.sum(-1, keepdim=True)

def draft_probs_at(logits, ids, hot_mask=None, topk=20):
    if hot_mask is not None:
        logits = logits.masked_fill(~hot_mask, float('-inf'))
    v, ix = logits.topk(topk, -1)
    lse = torch.logsumexp(v, -1, keepdim=True)
    kth = v[:, -1:]
    l_at = logits.gather(1, ids)
    q = torch.where(l_at >= kth, (l_at - lse).exp(), torch.zeros_like(l_at))
    return (q, logits.argmax(-1))
TC_OPEN, TC_CLOSE = (248058, 248059)
_TOKSTR = {}
GRID_RE = __import__('re').compile('^[\\d\\s,\\[\\]]*$')
TYPES = ('prose', 'grid', 'code')

def token_types(ids):
    if 'tok' not in _TOKSTR:
        from transformers import AutoTokenizer
        _TOKSTR['tok'] = AutoTokenizer.from_pretrained(MODEL)
    tok = _TOKSTR['tok']
    ids = ids.tolist()
    st = []
    for i in ids:
        if i not in _TOKSTR:
            _TOKSTR[i] = bool(GRID_RE.match(tok.decode([i])))
        st.append(_TOKSTR[i])
    st = np.array(st + [False] * 8)
    win = np.ones(len(ids), bool)
    for j in range(8):
        win &= st[j:j + len(ids)]
    a = np.array(ids)
    depth = np.cumsum((a == TC_OPEN).astype(np.int32) - (a == TC_CLOSE).astype(np.int32))
    ty = np.where(win, 1, 0)
    ty[depth > 0] = 2
    return torch.from_numpy(ty)

def load_rec(d, dev):
    d = Path(d)
    ids = torch.from_numpy(np.load(d / 'ids.npy').astype(np.int64))
    gen = torch.from_numpy(np.load(d / 'gen.npy').astype(np.bool_))
    hid = torch.from_numpy(np.load(d / 'hid.npy', mmap_mode='r').view(np.int16).copy()).view(torch.bfloat16)
    tid = torch.from_numpy(np.load(d / 'tid.npy').astype(np.int64))
    tlp = torch.from_numpy(np.load(d / 'tlp.npy'))
    return (ids, gen, hid, tid, tlp)

def pad128(x, T):
    Tp = -(-T // 2048) * 2048 if T > 2048 else -(-T // 128) * 128
    if x.shape[0] == Tp:
        return x
    return torch.cat([x, x.new_zeros((Tp - x.shape[0],) + tuple(x.shape[1:]))])

def run_steps(m, ids, hid, k, dev):
    T = ids.shape[0]
    idsd = ids.to(dev)
    tok = lambda s: pad128(torch.cat([idsd[1 + s:], idsd.new_zeros(1 + s)]), T)
    h = pad128(hid.to(dev), T)
    sample, multi, st = m.step0(h, tok(0))
    outs = [sample[:T]]
    extra = []
    for s in range(1, k):
        sample, multi = m.stepn(multi, tok(s), s, st, extra)
        outs.append(sample[:T])
    return outs

@torch.no_grad()
def eval_rec(m, d, k, dev, hot_mask, chunk=4096):
    ids, gen, hid, tid, tlp = load_rec(d, dev)
    T = ids.shape[0]
    outs = run_steps(m, ids, hid, k, dev)
    genp = torch.cat([gen, torch.zeros(k + 3, dtype=torch.bool)])
    base = genp[1:T + 1] & genp[2:T + 2]
    base[T - 2:] = False
    bt = base.nonzero()[:, 0]
    A = torch.zeros(len(bt), k)
    top1 = torch.zeros(len(bt), k)
    for s in range(k):
        r = bt + 1 + s
        ok = (r < T) & genp[(bt + 2 + s).clamp(max=T + k)]
        rr = r.clamp(max=T - 1)
        for i in range(0, len(bt), chunk):
            sl = slice(i, i + chunk)
            p = target_probs(tid[rr[sl]], tlp[rr[sl]]).to(dev)
            lg = m.logits(outs[s][bt[sl].to(dev)])
            q, am = draft_probs_at(lg, tid[rr[sl]].to(dev), hot_mask)
            a = torch.minimum(p, q).sum(-1).cpu()
            A[sl, s] = torch.where(ok[sl], a, torch.zeros_like(a))
            nxt = ids[(bt[sl] + 2 + s).clamp(max=T - 1)]
            top1[sl, s] = (am.cpu() == nxt).float() * ok[sl]
    ty = token_types(ids)[(bt + 2).clamp(max=T - 1)]
    return (A, top1, ty)

def summarize(A):
    cp = torch.cumprod(A, 1)
    return {'n': int(A.shape[0]), 'alpha': [round(float(x), 4) for x in A.mean(0)], 'uncond': [round(float(x), 4) for x in cp.mean(0)], 'tok_step': round(1 + float(cp.sum(1).mean()), 4), 'tok_step3': round(1 + float(cp[:, :3].sum(1).mean()), 4)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', required=True)
    ap.add_argument('--keys', default='')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--delta', default='', help='trained MTP state (train.py ckpt .pt)')
    ap.add_argument('--k', type=int, default=3)
    ap.add_argument('--no-hot', action='store_true')
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    dev = torch.device('cuda')
    m = MTP(MODEL, BUILD, dev)
    if a.delta:
        sd = torch.load(a.delta, map_location=dev)
        sd = sd.get('params', sd)
        missing = m.load_trained(sd)
        print('loaded delta', a.delta, 'missing', missing, flush=True)
    hot = None
    if not a.no_hot:
        hot = torch.zeros(m.lm_head.shape[0], dtype=torch.bool, device=dev)
        ids = torch.load(HOT)
        hot[torch.as_tensor(ids['ids'] if isinstance(ids, dict) else ids).to(dev)] = True
    ds = sorted((p.parent for p in Path(a.dump).glob('*/DONE')))
    if a.keys:
        want = set(open(a.keys[1:]).read().split()) if a.keys.startswith('@') else set(a.keys.split(','))
        ds = [d for d in ds if d.name in want]
    if a.limit:
        ds = ds[:a.limit]
    As, TOP1, TY = ([], [], [])
    t0 = time.time()
    for i, d in enumerate(ds):
        A, tp1, ty = eval_rec(m, d, a.k, dev, hot)
        As.append(A)
        TOP1.append(tp1)
        TY.append(ty)
        s = summarize(A)
        print(f'[{i + 1}/{len(ds)}] {d.name} {json.dumps(s)} {time.time() - t0:.0f}s', flush=True)
    A = torch.cat(As)
    res = summarize(A)
    res['top1'] = [round(float(x), 4) for x in torch.cat(TOP1).mean(0)]
    res['per_seq_tok_step'] = round(float(np.mean([summarize(x)['tok_step'] for x in As])), 4)
    res['n_seq'] = len(ds)
    ty = torch.cat(TY)
    res['by_type'] = {nm: dict(summarize(A[ty == i]), share=round(float((ty == i).float().mean()), 4)) for i, nm in enumerate(TYPES)}
    print('TOTAL', json.dumps(res), flush=True)
    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1)
if __name__ == '__main__':
    main()
