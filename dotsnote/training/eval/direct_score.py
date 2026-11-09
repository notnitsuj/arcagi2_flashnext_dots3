import argparse, json, os, random
from collections import defaultdict
from pathlib import Path
import numpy as np
DATA = Path(os.environ.get('ARC_DATA', 'data/ARC-AGI-2/data'))
ap = argparse.ArgumentParser()
ap.add_argument('logs', nargs='+')
ap.add_argument('--caps', default='16000,32000,64000,96000,128000,none')
ap.add_argument('--attempts', type=int, default=2)
ap.add_argument('--split', default='evaluation')
a = ap.parse_args()
ids = sorted((p.stem for p in (DATA / a.split).glob('*.json')))
ntest = {p.stem: len(json.load(open(p))['test']) for s in ('evaluation', 'training') for p in (DATA / s).glob('*.json')}
hold = set(random.Random(2026).sample(ids, 40))
for path in a.logs:
    R = [json.loads(l) for l in open(path)]
    by = defaultdict(list)
    for r in R:
        by[r['task'], r['test']].append(r)
    tasks = sorted({r['task'] for r in R})
    ct = np.array([r['ct'] for r in R])
    print(f"== {path}: {len(R)} attempts over {len(tasks)} tasks | ct median {np.median(ct):.0f} p90 {np.percentile(ct, 90):.0f} max {ct.max()} | closed think {np.mean([r['closed_think'] for r in R]):.2f} parsed {np.mean([r['parsed'] for r in R]):.2f} | per-attempt acc {np.mean([bool(r['correct']) for r in R]):.3f}")
    for cap in a.caps.split(','):
        cap = None if cap == 'none' else int(cap)
        tot = hsum = 0.0
        toks = 0
        complete = 0
        for t in tasks:
            s = 0
            for ti in range(ntest[t]):
                atts = sorted(by[t, ti], key=lambda r: r['attempt'])[:a.attempts]
                toks += sum((min(r['ct'], cap or r['ct']) for r in atts))
                s += any((r['correct'] and (cap is None or r['ct'] <= cap) for r in atts))
            tot += s / ntest[t]
            hsum += s / ntest[t] if t in hold else 0
            complete += all((len(by[t, ti]) >= a.attempts for ti in range(ntest[t])))
        nh = len([t for t in tasks if t in hold])
        print(f'  cap/attempt={str(cap):>7}: {tot:6.2f}/{len(tasks)} = {100 * tot / len(tasks):5.1f}%   holdout {hsum:5.2f}/{nh}   out-tok/task {toks / len(tasks):8.0f}   ({complete} tasks complete)')
