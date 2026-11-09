import argparse, gzip, json, os, sys, zlib
from pathlib import Path
import numpy as np
EVAL_DIRS = [os.path.join(os.environ.get('ARC_DATA', 'data/ARC-AGI-2/data'), 'evaluation')]

def base_of(task):
    return task.split('~')[0].split('@')[0]

def split_of(base, frac):
    return 'ho' if zlib.crc32(base.encode()) % 1000 < frac * 1000 else 'tr'

def gen_mask(tok, prompt, text, spans, temp=1.0):
    full = prompt + text
    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    ids = np.array(enc['input_ids'], np.int32)
    offs = np.array(enc['offset_mapping'], np.int64)
    P = len(prompt)
    gm = np.zeros(len(full) + 1, np.int8)
    for s, e, t in spans:
        if t is not None and abs(float(t) - temp) > 1e-06:
            continue
        gm[P + s:P + e] = 1
    cs = np.concatenate([[0], np.cumsum(gm)])
    en = np.minimum(np.maximum(offs[:, 1], offs[:, 0] + 1), len(full))
    inside = cs[en] - cs[offs[:, 0]] == en - offs[:, 0]
    inside &= offs[:, 1] > offs[:, 0]
    inside[0] = False
    return (ids, inside.astype(np.uint8))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('rolls', nargs='+')
    ap.add_argument('--max-len', type=int, default=65536)
    ap.add_argument('--ho-frac', type=float, default=0.1)
    ap.add_argument('--tok', default=os.environ.get('FN_BASE_MODEL', 'models/Qwen3.8-Flash-Next-FP8'))
    a = ap.parse_args()
    ev = set()
    for d in EVAL_DIRS:
        if Path(d).is_dir():
            ev |= {p.stem for p in Path(d).glob('*.json')}
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tok)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    idx = open(out / 'index.jsonl', 'w')
    st = {'written': 0, 'tr': 0, 'ho': 0, 'gen_tr': 0, 'gen_ho': 0}
    for d in a.rolls:
        tag = Path(d).name
        recs = {}
        for l in open(Path(d) / 'samples.jsonl'):
            r = json.loads(l)
            if r.get('arm') != 'repl' or r.get('status') == 'error' or 'rl' not in r:
                continue
            assert base_of(r['task']) not in ev, f"evaluation task {r['task']}"
            recs[r['task'], r['j']] = r
        try:
            for l in gzip.open(Path(d) / 'traces.jsonl.gz', 'rt'):
                t = json.loads(l)
                r = recs.get((t['task'], t['j']))
                if r is None:
                    continue
                ids, gen = gen_mask(tok, r['rl']['prompt'], t['text'], r['rl']['spans'])
                T = min(len(ids), a.max_len)
                if gen[:T].sum() < 16:
                    continue
                sp = split_of(base_of(r['task']), a.ho_frac)
                key = f"{tag}_{r['task']}_{r['j']}".replace('/', '_')
                np.savez(out / f'{key}.npz', ids=ids[:T], gen=gen[:T])
                idx.write(json.dumps({'key': key, 'task': r['task'], 'split': sp, 'T': int(T), 'ngen': int(gen[:T].sum())}) + '\n')
                st['written'] += 1
                st[sp] += 1
                st['gen_' + sp] += int(gen[:T].sum())
        except (EOFError, json.JSONDecodeError):
            pass
    print(json.dumps(st), file=sys.stderr)
    json.dump(st, open(out / 'stats.json', 'w'))
if __name__ == '__main__':
    main()
