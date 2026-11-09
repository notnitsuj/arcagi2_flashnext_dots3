import argparse, glob, json, math, os, random, sys, urllib.request
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'student'))
import student_fn as S

def server_lp(ep, ids):
    body = {'model': 'm', 'prompt': [int(x) for x in ids], 'max_tokens': 1, 'temperature': 0.0, 'prompt_logprobs': 20}
    r = json.load(urllib.request.urlopen(urllib.request.Request(f'http://{ep}/v1/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'}), timeout=3600))
    return r['choices'][0]['prompt_logprobs']

def student_lp(model, ids, pos):
    with torch.no_grad():
        h = model.hidden(torch.as_tensor(ids, dtype=torch.long), ckpt=False)
        out = []
        p = torch.as_tensor(pos, device=model.last).long()
        y = torch.as_tensor(ids, device=model.last)[p]
        for a in range(0, len(pos), 1024):
            ls = torch.log_softmax((h[p[a:a + 1024] - 1] @ model.lm_head.t()).float(), -1)
            out.append(ls.gather(1, y[a:a + 1024, None])[:, 0].cpu())
        return torch.cat(out).numpy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--recs', required=True)
    ap.add_argument('--n', type=int, default=32)
    ap.add_argument('--endpoint', required=True)
    ap.add_argument('--build', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--resume', default=None)
    ap.add_argument('--after', default=None)
    ap.add_argument('--gpus', default='0')
    ap.add_argument('--max-len', type=int, default=30000)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    idx = [json.loads(l) for l in open(f'{a.recs}/index.jsonl')]
    rep = {'records': len(idx), 'truncated_records': sum((r.get('cut', 0) > 0 for r in idx)), 'cut_tokens': sum((r.get('cut', 0) for r in idx)), 'kept_tokens': sum((r['ngen'] for r in idx))}
    rng = random.Random(a.seed)
    cand = [r for r in idx if r['T'] <= a.max_len]
    by = {}
    for r in cand:
        by.setdefault(r['cls'], []).append(r)
    pick = []
    for c, rs in sorted(by.items()):
        rng.shuffle(rs)
        pick += rs[:max(1, a.n * len(rs) // len(cand))]
    pick = pick[:a.n]
    model = S.Student(a.model, a.build, [int(g) for g in a.gpus.split(',')], trainable=())

    def load(p):
        st = torch.load(p, map_location='cpu')
        for n, q in model.named_parameters():
            if n in st:
                q.data.copy_(st[n].to(q.dtype))
    if a.resume:
        load(a.resume)
    acc = {k: [] for k in ('srv_raw', 'stu', 'srv_samp', 'in_top20', 'in_nucleus', 'nucleus_logmass')}
    per = []
    for r in pick:
        z = np.load(f"{a.recs}/{r['key']}.npz")
        ids, pos = (z['ids'], z['pos'])
        plp = server_lp(a.endpoint, ids)
        stu = student_lp(model, ids, pos)
        sr, ss, it, inn, lm = ([], [], [], [], [])
        for j, t in enumerate(pos):
            d = plp[t] or {}
            tok = str(int(ids[t]))
            top = sorted(((v['logprob'], k) for k, v in d.items()), reverse=True)[:20]
            lps = np.array([x for x, _ in top])
            pr = np.exp(lps - lps.max())
            pr /= pr.sum()
            keep = np.cumsum(pr) - pr <= 0.95
            kp = pr * keep
            kp /= kp.sum()
            names = [k for _, k in top]
            sr.append(d[tok]['logprob'] if tok in d else float('nan'))
            it.append(tok in names[:20])
            inn.append(tok in [n for n, k in zip(names, keep) if k])
            ss.append(math.log(kp[names.index(tok)]) if tok in names and kp[names.index(tok)] > 0 else float('nan'))
            lm.append(float(np.log(np.exp(lps)[keep].sum())))
        acc['srv_raw'] += sr
        acc['stu'] += list(stu)
        acc['srv_samp'] += ss
        acc['in_top20'] += it
        acc['in_nucleus'] += inn
        acc['nucleus_logmass'] += lm
        per.append({'key': r['key'], 'cls': r['cls'], 'T': r['T'], 'n': len(pos)})
    sr, st_, ss = (np.array(acc[k], float) for k in ('srv_raw', 'stu', 'srv_samp'))
    m = ~np.isnan(sr)
    rep.update({'audit_records': len(pick), 'audit_tokens': int(len(sr)), 'server_vs_trainer_abs_dlp': float(np.mean(np.abs(sr[m] - st_[m]))), 'server_minus_trainer_dlp': float(np.mean(sr[m] - st_[m])), 'frac_tokens_abs_dlp_gt_0.1': float(np.mean(np.abs(sr[m] - st_[m]) > 0.1)), 'sampling_minus_raw_lp': float(np.nanmean(ss - sr)), 'frac_sampled_in_top20': float(np.mean(acc['in_top20'])), 'frac_sampled_in_nucleus': float(np.mean(acc['in_nucleus'])), 'mean_nucleus_logmass': float(np.mean(acc['nucleus_logmass']))})
    if a.after:
        load(a.after)
        aft = []
        for r in pick:
            z = np.load(f"{a.recs}/{r['key']}.npz")
            aft += list(student_lp(model, z['ids'], z['pos']))
        aft = np.array(aft)
        rep.update({'staleness_abs_dlp': float(np.mean(np.abs(aft - st_))), 'staleness_mean_dlp': float(np.mean(aft - st_)), 'frac_tokens_ratio_outside_0.8_1.2': float(np.mean(np.abs(np.exp(aft - st_) - 1) > 0.2))})
    rep['records_used'] = per
    json.dump(rep, open(a.out, 'w'), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != 'records_used'}, indent=1))
if __name__ == '__main__':
    main()
