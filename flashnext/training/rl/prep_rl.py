import argparse, glob, gzip, json, os, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
TOK = os.environ.get('FN_BASE_MODEL', 'models/Qwen3.8-Flash-Next-FP8')
EVAL_DIR = os.path.join(os.environ.get('ARC_DATA', 'data/ARC-AGI-2/data'), 'evaluation')

def read_gz_lines(path):
    import zlib
    raw = open(path, 'rb').read()

    def header(i):
        if raw[i + 3] == 0:
            return True
        if raw[i + 3] != 8:
            return False
        e = raw.find(b'\x00', i + 10, i + 266)
        return e > i + 10 and all((32 <= c < 127 for c in raw[i + 10:e]))
    starts = [i for i in range(len(raw) - 10) if raw[i] == 31 and raw[i + 1] == 139 and (raw[i + 2] == 8) and header(i)] + [len(raw)]
    buf = []
    k = 0
    while k < len(starts) - 1:
        pos = starts[k]
        j = k + 1
        d = zlib.decompressobj(16 + zlib.MAX_WBITS)
        try:
            out = d.decompress(raw[pos:starts[j]])
            ok = d.eof
        except zlib.error:
            out, ok = (b'', False)
        if out:
            buf.append(out if ok else out.rsplit(b'\n', 1)[0] + b'\n')
        k = j
    for l in b''.join(buf).decode('utf-8', 'replace').splitlines():
        if l.strip():
            yield l

def reward(r, R):
    vr, vw, ur, uw = R
    ver = [p for p in r.get('progs', []) if p.get('verified')]
    if ver:
        return (vr if all(ver[0].get('tests_correct') or [False]) else vw, 'v' + ('r' if all(ver[0].get('tests_correct') or [False]) else 'w'))
    ok = r.get('answer_correct') or []
    good = bool(ok) and all(ok) and (len(ok) == r.get('n_test', len(ok)))
    return (ur if good else uw, 'u' + ('r' if good else 'w'))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('rolls', nargs='+')
    ap.add_argument('--max-len', type=int, default=65536)
    ap.add_argument('--rewards', default='1,-1,0.5,-0.2')
    ap.add_argument('--temp', type=float, default=1.0)
    ap.add_argument('--min-group', type=int, default=4)
    ap.add_argument('--tag', default='')
    ap.add_argument('--len-penalty', type=float, default=0.0, help='subtract LP * min(1, used / --len-cap) from every reward (session-length pressure)')
    ap.add_argument('--len-cap', type=float, default=98304)
    ap.add_argument('--budget-aware', type=float, default=0.0, help='scale POSITIVE rewards by max(floor, 1 - BA * tokens/len-cap): a right answer is worth less the more of the budget it spent (negatives unchanged)')
    ap.add_argument('--ba-floor', type=float, default=0.2)
    ap.add_argument('--concise', type=int, default=0, help='concise-success training: per group only the K shortest RIGHT samples (vr/ur) get advantage +1, everything else is dropped (no negatives, no length penalty)')
    ap.add_argument('--concise-neg', type=float, default=0.0, help='with --concise: verified-wrong samples (vw) in the same group get advantage -X (precision)')
    ap.add_argument('--require-mixed', action='store_true', help='keep only groups whose UNSHAPED rewards differ (correctness signal), so length/budget terms only re-weight within groups that carry a correctness signal')
    ap.add_argument('--exclude', default='', help='task-id files to drop (e.g. held-out val tasks)')
    ap.add_argument('--only', default='', help='task-id files: keep only these tasks (e.g. rare-success selection)')
    a = ap.parse_args()
    R = [float(x) for x in a.rewards.split(',')]
    ev = {p.stem for p in Path(EVAL_DIR).glob('*.json')} if Path(EVAL_DIR).is_dir() else set()
    drop = set()
    for f in filter(None, a.exclude.split(',')):
        drop |= set(open(f).read().split())
    keep = set()
    for f in filter(None, a.only.split(',')):
        keep |= set(open(f).read().split())
    recs, texts = ({}, {})
    for d in a.rolls:
        tag = Path(d).name
        for l in open(Path(d) / 'samples.jsonl'):
            r = json.loads(l)
            if r.get('arm') != 'repl' or r.get('status') == 'error' or 'rl' not in r:
                continue
            base = r['task'].split('~')[0].split('@')[0]
            assert base not in ev, f'evaluation task {base} in RL data'
            if base in drop or (keep and r['task'] not in keep):
                continue
            recs[tag, r['task'], r['j']] = r
        for l in read_gz_lines(Path(d) / 'traces.jsonl.gz'):
            try:
                t = json.loads(l)
            except json.JSONDecodeError:
                continue
            k = (tag, t['task'], t['j'])
            if k in recs:
                texts[k] = t['text']
    groups = defaultdict(list)
    for k, r in recs.items():
        if k in texts:
            groups[k[1]].append(k)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    idx = open(out / 'index.jsonl', 'w')
    st = defaultdict(int)
    for g, ks in sorted(groups.items()):
        if len(ks) < a.min_group:
            st['small_group'] += 1
            continue
        rw = [reward(recs[k], R) for k in ks]
        if a.require_mixed and len({x[0] for x in rw}) < 2:
            st['flat_correctness'] += 1
            continue
        if a.budget_aware:
            rw = [(r_ * max(a.ba_floor, 1 - a.budget_aware * min(1.0, (recs[k].get('used') or 0) / a.len_cap)) if r_ > 0 else r_, c_) for (r_, c_), k in zip(rw, ks)]
        if a.len_penalty:
            rw = [(r_ - a.len_penalty * min(1.0, (recs[k].get('used') or 0) / a.len_cap), c_) for (r_, c_), k in zip(rw, ks)]
        rs = np.array([x[0] for x in rw], np.float64)
        n = len(rs)
        adv = rs - (rs.sum() - rs) / (n - 1)
        if a.concise:
            right = [i for i, x in enumerate(rw) if x[1] in ('vr', 'ur')]
            right.sort(key=lambda i: recs[ks[i]].get('used') or 0)
            adv = np.zeros(n)
            adv[right[:a.concise]] = 1.0
            if a.concise_neg and right:
                for i, x in enumerate(rw):
                    if x[1] == 'vw':
                        adv[i] = -a.concise_neg
        for k, x in zip(ks, rw):
            st[x[1]] += 1
        if np.allclose(adv, 0):
            st['flat_group'] += 1
            continue
        st['groups'] += 1
        for k, A, (Rv, cls) in zip(ks, adv, rw):
            if abs(A) < 1e-06:
                continue
            r, text = (recs[k], texts[k])
            prompt = r['rl']['prompt']
            full = prompt + text
            enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
            ids = np.array(enc['input_ids'], np.int32)
            offs = np.array(enc['offset_mapping'], np.int64)
            P = len(prompt)
            gm = np.zeros(len(full) + 1, np.int8)
            for s, e, temp in r['rl']['spans']:
                if temp is not None and abs(float(temp) - a.temp) > 1e-06:
                    continue
                gm[P + s:P + e] = 1
            cs = np.concatenate([[0], np.cumsum(gm)])
            st_, en_ = (offs[:, 0], np.maximum(offs[:, 1], offs[:, 0] + 1))
            inside = cs[np.minimum(en_, len(full))] - cs[st_] == np.minimum(en_, len(full)) - st_
            inside &= offs[:, 1] > offs[:, 0]
            inside[0] = False
            T = min(len(ids), a.max_len)
            pos = np.nonzero(inside[:T])[0].astype(np.int32)
            if len(pos) < 16:
                st['empty'] += 1
                continue
            key = f'{a.tag}{k[0]}_{k[1]}_{k[2]}'
            np.savez(out / f'{key}.npz', ids=ids[:T], pos=pos, adv=np.float32(A))
            ntrunc = int(inside.sum() - len(pos))
            idx.write(json.dumps({'key': key, 'task': k[1], 'j': k[2], 'R': Rv, 'cls': cls, 'adv': round(float(A), 4), 'T': int(len(ids)), 'ngen': int(len(pos)), 'cut': ntrunc, 'used': r.get('used')}) + '\n')
            st['written'] += 1
            st['gen_tok'] += len(pos)
            st['cut_tok'] += ntrunc
    print(json.dumps(st), file=sys.stderr)
    json.dump(st, open(out / 'stats.json', 'w'))
if __name__ == '__main__':
    main()
