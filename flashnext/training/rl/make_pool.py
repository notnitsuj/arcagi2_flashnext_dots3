import argparse, glob, hashlib, json, os, random
from pathlib import Path

def d8(g):
    out = []
    x = [list(r) for r in g]
    for _ in range(4):
        out.append(x)
        out.append([r[::-1] for r in x])
        x = [list(r) for r in zip(*x[::-1])]
    return out

def relabel(g):
    m, out = ({}, [])
    for r in g:
        rr = []
        for v in r:
            if v not in m:
                m[v] = len(m)
            rr.append(m[v])
        out.append(rr)
    return out

def canon(g):
    return hashlib.md5(min((json.dumps(relabel(t)) for t in d8(g))).encode()).hexdigest()

def grids(task):
    for k in ('train', 'test'):
        for p in task.get(k, []):
            for kk in ('input', 'output'):
                if kk in p and p[kk]:
                    yield p[kk]

def ncell(g):
    return sum((len(r) for r in g))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--arc1', required=True, help='ARC-AGI-1 data dir with training/ and evaluation/')
    ap.add_argument('--rearc', required=True, help='RE-ARC dir with tasks/<id>.json (list of {input, output})')
    ap.add_argument('--arc2-train', required=True)
    ap.add_argument('--arc2-eval', required=True)
    ap.add_argument('--eval-json', nargs='*', default=[])
    ap.add_argument('--rearc-per-task', type=int, default=2)
    ap.add_argument('--min-cells', type=int, default=9)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--rearc-tag', default='ra', help='id prefix of the RE-ARC instances (use a new one per seed)')
    ap.add_argument('--no-arc1', action='store_true')
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ev_ids, ev_h = (set(), set())
    for f in glob.glob(f'{a.arc2_eval}/*.json'):
        ev_ids.add(Path(f).stem)
        for g in grids(json.load(open(f))):
            ev_h.add(canon(g))
    for f in a.eval_json:
        for tid, t in json.load(open(f)).items():
            ev_ids.add(tid)
            for g in grids(t):
                ev_h.add(canon(g))
    tr2 = {Path(f).stem for f in glob.glob(f'{a.arc2_train}/*.json')}
    cands = {}
    for sub in () if a.no_arc1 else ('training', 'evaluation'):
        for f in sorted(glob.glob(f'{a.arc1}/{sub}/*.json')):
            tid = Path(f).stem
            if tid in tr2:
                continue
            cands[f'a1_{tid}'] = json.load(open(f))
    for f in sorted(glob.glob(f'{a.rearc}/tasks/*.json')):
        tid = Path(f).stem
        ex = json.load(open(f))
        for k in range(a.rearc_per_task):
            n = rng.choice([3, 4])
            pick = rng.sample(range(len(ex)), n + 1)
            cands[f'{a.rearc_tag}_{tid}_{k}'] = {'train': [ex[i] for i in pick[:n]], 'test': [ex[pick[n]]]}
    rep = {'eval_ids': len(ev_ids), 'eval_grid_canon': len(ev_h), 'candidates': len(cands), 'dropped_id': 0, 'dropped_grid_match': 0, 'dropped_examples': []}
    kept = []
    for cid, t in cands.items():
        base = cid.split('_')[1]
        if base in ev_ids:
            rep['dropped_id'] += 1
            continue
        hit = [ncell(g) for g in grids(t) if ncell(g) >= a.min_cells and canon(g) in ev_h]
        if hit:
            rep['dropped_grid_match'] += 1
            if len(rep['dropped_examples']) < 20:
                rep['dropped_examples'].append([cid, hit[:3]])
            continue
        json.dump(t, open(out / f'{cid}.json', 'w'))
        kept.append(cid)
    rep['kept'] = len(kept)
    rep['kept_a1'] = sum((c.startswith('a1_') for c in kept))
    rep['kept_ra'] = sum((c.startswith(a.rearc_tag + '_') for c in kept))
    (out / '_tasks.txt').write_text('\n'.join(kept) + '\n')
    json.dump(rep, open(out / '_report.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != 'dropped_examples'}))
if __name__ == '__main__':
    main()
