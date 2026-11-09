import argparse, gzip, json
from collections import Counter
ap = argparse.ArgumentParser()
ap.add_argument('--traces', nargs='+', required=True)
ap.add_argument('--model-dir', required=True)
ap.add_argument('--out', default=None)
ap.add_argument('--size', type=int, default=32768)
ap.add_argument('--holdout-frac', type=float, default=0.25, help='last share of records used to measure coverage')
a = ap.parse_args()
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.model_dir)
recs = []
for t in a.traces:
    for l in gzip.open(t, 'rt'):
        r = json.loads(l)
        recs.append(r['token_ids'] if r.get('token_ids') else r['completion'])
n_ho = int(len(recs) * a.holdout_frac)
fit, ho = (recs[:len(recs) - n_ho], recs[len(recs) - n_ho:])
cf, ch = (Counter(), Counter())
for r in fit:
    cf.update(r if isinstance(r, list) else tok(r, add_special_tokens=False).input_ids)
for r in ho:
    ch.update(r if isinstance(r, list) else tok(r, add_special_tokens=False).input_ids)
order = [t for t, _ in cf.most_common()]
tot_f, tot_h = (sum(cf.values()), sum(ch.values()))
print(f'records fit {len(fit)} ({tot_f} tok), held-out {len(ho)} ({tot_h} tok); distinct ids fit {len(cf)}')
for n in (1024, 2048, 4096, 8192, 16384, 32768, 65536):
    s = set(order[:n])
    cov_f = sum((cf[t] for t in s)) / tot_f
    cov_h = sum((v for t, v in ch.items() if t in s)) / max(1, tot_h)
    print(f'top {n:6d}: coverage fit {cov_f:.5f}  held-out {cov_h:.5f}')
if a.out:
    import torch
    ids = sorted(order[:a.size])
    extra = set(tok.all_special_ids) | {tok.convert_tokens_to_ids(x) for x in tok.get_added_vocab()}
    ids = set(ids) | {i for i in extra if i is not None and i >= 0}
    fill = 0
    for i in range(len(tok)):
        if len(ids) >= a.size:
            break
        if i not in ids:
            ids.add(i)
            fill += 1
    print(f'seen ids {len(order)}, filled {fill} unseen low ids')
    ids = sorted(ids)
    torch.save({'ids': torch.tensor(ids, dtype=torch.long)}, a.out)
    print('wrote', a.out, len(ids))
