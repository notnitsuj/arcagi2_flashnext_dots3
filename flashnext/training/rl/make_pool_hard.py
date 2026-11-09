import argparse, glob, json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_pool as M

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--rearc', required=True)
    ap.add_argument('--arc2-eval', required=True)
    ap.add_argument('--eval-json', nargs='*', default=[])
    ap.add_argument('--inform', default=None)
    ap.add_argument('--hard-frac', type=float, default=0.1)
    ap.add_argument('--per-gen', type=int, default=2)
    ap.add_argument('--per-gen-inform', type=int, default=6)
    ap.add_argument('--min-cells', type=int, default=9)
    ap.add_argument('--seed', type=int, default=2)
    ap.add_argument('--tag', default='rh')
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ev_ids, ev_h = (set(), set())
    for f in glob.glob(f'{a.arc2_eval}/*.json'):
        ev_ids.add(Path(f).stem)
        for g in M.grids(json.load(open(f))):
            ev_h.add(M.canon(g))
    for f in a.eval_json:
        for tid, t in json.load(open(f)).items():
            ev_ids.add(tid)
            for g in M.grids(t):
                ev_h.add(M.canon(g))
    inform = set()
    if a.inform:
        for x in open(a.inform).read().split():
            p = x.split('_')
            inform.add(p[1] if len(p) >= 2 else x)
    meta = json.load(open(f'{a.rearc}/metadata.json'))
    rep = {'generators': 0, 'inform_generators': 0, 'candidates': 0, 'dropped_id': 0, 'dropped_grid_match': 0}
    kept = []
    for f in sorted(glob.glob(f'{a.rearc}/tasks/*.json')):
        gen = Path(f).stem
        rep['generators'] += 1
        if gen in ev_ids:
            rep['dropped_id'] += 1
            continue
        ex = json.load(open(f))
        d = meta[gen]['pso_difficulties']
        assert len(d) == len(ex), (gen, len(d), len(ex))
        order = sorted(range(len(ex)), key=lambda i: -d[i])
        hard = order[:max(8, int(len(ex) * a.hard_frac))]
        n = a.per_gen_inform if gen in inform else a.per_gen
        rep['inform_generators'] += gen in inform
        for k in range(n):
            m = rng.choice([3, 4])
            pick = rng.sample(hard, m + 1)
            t = {'train': [ex[i] for i in pick[:m]], 'test': [ex[pick[m]]]}
            rep['candidates'] += 1
            if any((M.ncell(g) >= a.min_cells and M.canon(g) in ev_h for g in M.grids(t))):
                rep['dropped_grid_match'] += 1
                continue
            cid = f'{a.tag}_{gen}_{k}'
            json.dump(t, open(out / f'{cid}.json', 'w'))
            kept.append(cid)
    rep['kept'] = len(kept)
    (out / '_tasks.txt').write_text('\n'.join(kept) + '\n')
    json.dump(rep, open(out / '_report.json', 'w'), indent=1)
    print(json.dumps(rep))
if __name__ == '__main__':
    main()
