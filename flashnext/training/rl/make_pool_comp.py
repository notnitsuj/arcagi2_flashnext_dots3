import argparse, glob, json, random, signal, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_pool as M

class TO(Exception):
    pass

def _alarm(*_):
    raise TO()

def to_list(g):
    return [list(r) for r in g]

def valid(g):
    return isinstance(g, (tuple, list)) and 0 < len(g) <= 30 and all((isinstance(r, (tuple, list)) for r in g)) and (0 < len(g[0]) <= 30) and all((len(r) == len(g[0]) for r in g)) and all((isinstance(v, int) and 0 <= v <= 9 for r in g for v in r))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--rearc-src', required=True)
    ap.add_argument('--arc2-eval', required=True)
    ap.add_argument('--eval-json', nargs='*', default=[])
    ap.add_argument('--n-tasks', type=int, default=600)
    ap.add_argument('--max-tries', type=int, default=20000)
    ap.add_argument('--diff', default='0.5,1.0')
    ap.add_argument('--min-cells', type=int, default=9)
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--tag', default='rc')
    a = ap.parse_args()
    import types
    mpl = types.ModuleType('matplotlib')
    plt = types.ModuleType('matplotlib.pyplot')
    col = types.ModuleType('matplotlib.colors')
    col.ListedColormap = col.Normalize = object
    mpl.pyplot, mpl.colors = (plt, col)
    sys.modules.update({'matplotlib': mpl, 'matplotlib.pyplot': plt, 'matplotlib.colors': col})
    sys.path.insert(0, a.rearc_src)
    import generators as G
    import verifiers as V
    lo, hi = (float(x) for x in a.diff.split(','))
    rng = random.Random(a.seed)
    random.seed(a.seed)
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
    gens = {k[len('generate_'):]: getattr(G, k) for k in dir(G) if k.startswith('generate_')}
    vers = {k[len('verify_'):]: getattr(V, k) for k in dir(V) if k.startswith('verify_')}
    ids = sorted(set(gens) & set(vers) - ev_ids)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGALRM, _alarm)

    def call(f, *x):
        signal.alarm(1)
        try:
            return f(*x)
        finally:
            signal.alarm(0)
    rep = {'pairs_tried': 0, 'pairs_ok': 0, 'dropped_grid_match': 0, 'kept': 0, 'ids': len(ids)}
    kept, seen = ([], set())
    while len(kept) < a.n_tasks and rep['pairs_tried'] < a.max_tries:
        A, B = rng.sample(ids, 2)
        if (A, B) in seen:
            continue
        seen.add((A, B))
        rep['pairs_tried'] += 1
        pairs, ok = ([], True)
        n = rng.choice([4, 5])
        for _ in range(n):
            try:
                ex = call(gens[A], lo, hi)
                I = ex['input']
                oa = call(vers[A], I)
                ob = call(vers[B], oa)
            except BaseException:
                ok = False
                break
            if not (valid(I) and valid(ob)):
                ok = False
                break
            pairs.append((to_list(I), to_list(oa), to_list(ob)))
        if not ok:
            continue
        if all((p[2] == p[1] for p in pairs)) or all((p[2] == p[0] for p in pairs)):
            continue
        if len({json.dumps(p[2]) for p in pairs}) < 2:
            continue
        rep['pairs_ok'] += 1
        t = {'train': [{'input': p[0], 'output': p[2]} for p in pairs[:-1]], 'test': [{'input': pairs[-1][0], 'output': pairs[-1][2]}]}
        if any((M.ncell(g) >= a.min_cells and M.canon(g) in ev_h for g in M.grids(t))):
            rep['dropped_grid_match'] += 1
            continue
        cid = f'{a.tag}_{A}_{B}'
        json.dump(t, open(out / f'{cid}.json', 'w'))
        kept.append(cid)
    rep['kept'] = len(kept)
    (out / '_tasks.txt').write_text('\n'.join(kept) + '\n')
    json.dump(rep, open(out / '_report.json', 'w'), indent=1)
    print(json.dumps(rep))
if __name__ == '__main__':
    main()
