import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
TRACES = Path(os.environ.get('D3_FID_TRACES', os.path.join(os.environ.get('D3_WORK', 'work'), 'fid', 'fp8_text.jsonl')))
TASKS = ['269e22fb', '3a25b0d8', '446ef5d2', '21897d95', '5dbc8537', 'a32d8b75']
K = 20
BANDS = [(0, 4096), (4096, 16384), (16384, 1 << 30)]

def load_traces(n, length):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.environ.get('D3_BASE_MODEL', 'models/dots3-note-prev-fp8'))
    recs = {}
    for l in open(TRACES):
        r = json.loads(l)
        if r['task'] in TASKS[:n] and r['task'] not in recs:
            recs[r['task']] = r
    out = []
    for t in TASKS[:n]:
        r = recs[t]
        p = tok(r['prompt'], add_special_tokens=False).input_ids
        ids = tok(r['prompt'] + r['completion'], add_special_tokens=False).input_ids[:length]
        out.append((t, ids, len(p)))
    return out

def dump(port, out, n, length):
    import httpx
    res = {}
    for t, ids, plen in load_traces(n, length):
        t0 = time.time()
        r = httpx.post(f'http://127.0.0.1:{port}/v1/completions', json={'model': 'm', 'prompt': ids, 'max_tokens': 1, 'prompt_logprobs': K, 'temperature': 0}, timeout=7200).json()
        pl = r['choices'][0]['prompt_logprobs']
        L = len(ids)
        top_ids = np.full((L, K), -1, np.int32)
        top_lp = np.full((L, K), -1000000000.0, np.float32)
        act = np.zeros(L, np.float32)
        for i, d in enumerate(pl):
            if d is None:
                continue
            items = sorted(((int(k), v['logprob'], v.get('rank', 99)) for k, v in d.items()), key=lambda x: x[2])
            tops = [x for x in items if x[2] <= K][:K]
            top_ids[i, :len(tops)] = [x[0] for x in tops]
            top_lp[i, :len(tops)] = [x[1] for x in tops]
            act[i] = d[str(ids[i])]['logprob'] if str(ids[i]) in d else tops[-1][1]
        res[f'{t}_ids'] = np.asarray(ids, np.int32)
        res[f'{t}_top_ids'], res[f'{t}_top_lp'], res[f'{t}_act'] = (top_ids, top_lp, act)
        res[f'{t}_plen'] = np.asarray(plen)
        print(f'{t}: {L} tok in {time.time() - t0:.0f}s, completion NLL {-act[plen:].mean():.4f}', flush=True)
    np.savez_compressed(out, **res)

def compare(ref_path, cand_path):
    R, C = (np.load(ref_path), np.load(cand_path))
    tasks = sorted({k.rsplit('_', 2)[0] for k in R.files if k.endswith('_top_ids')} & {k.rsplit('_', 2)[0] for k in C.files if k.endswith('_top_ids')})
    acc = {b: [0.0, 0.0, 0.0, 0.0, 0] for b in BANDS}
    for t in tasks:
        plen = int(R[f'{t}_plen'])
        n = min(len(R[f'{t}_act']), len(C[f'{t}_act']))
        ri, rl = (R[f'{t}_top_ids'][:n], R[f'{t}_top_lp'][:n])
        ci, cl = (C[f'{t}_top_ids'][:n], C[f'{t}_top_lp'][:n])
        sl = slice(plen, n)
        rI, rL, cI, cL = (ri[sl], rl[sl], ci[sl], cl[sl])
        eq = (rI[:, :, None] == cI[:, None, :]) & (rI[:, :, None] >= 0)
        floor = np.where(cI >= 0, cL, np.inf).min(1)
        lc = np.where(eq.any(2), np.where(eq, cL[:, None, :], 0).sum(2), floor[:, None])
        pr = np.where(rI >= 0, np.exp(rL), 0.0)
        kl = (pr * (np.where(rI >= 0, rL, 0) - lc)).sum(1)
        agree = (rI[:, 0] == cI[:, 0]).astype(np.float64)
        nc, nr = (-C[f'{t}_act'][sl], -R[f'{t}_act'][sl])
        pos = np.arange(plen, n)
        for b in BANDS:
            m = (pos >= b[0]) & (pos < b[1])
            a_ = acc[b]
            a_[0] += kl[m].sum()
            a_[1] += agree[m].sum()
            a_[2] += nc[m].sum()
            a_[3] += nr[m].sum()
            a_[4] += int(m.sum())
    tot = [sum((acc[b][j] for b in BANDS)) for j in range(5)]

    def fmt(a):
        return f'KL {a[0] / a[4]:.4f}  top1 {a[1] / a[4]:.4f}  dNLL {(a[2] - a[3]) / a[4]:+.4f}' if a[4] else '-'
    print(f'{Path(cand_path).stem:28s} ALL {fmt(tot)} (n={tot[4]})  | ' + ' | '.join((f'{b[0] // 1024}k+: {fmt(acc[b])}' for b in BANDS)))
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['dump', 'cmp'])
    ap.add_argument('args', nargs='+')
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--len', type=int, default=32768)
    a = ap.parse_args()
    if a.cmd == 'dump':
        dump(int(a.args[0]), a.args[1], a.n, a.len)
    else:
        for c in a.args[1:]:
            compare(a.args[0], c)
