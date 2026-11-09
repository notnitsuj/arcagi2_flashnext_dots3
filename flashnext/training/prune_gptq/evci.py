import glob, json, os, random
from collections import defaultdict

def load(p):
    d = defaultdict(dict)
    for l in open(p):
        if l.strip():
            r = json.loads(l)
            d[r['task'], r['test']][r['attempt']] = r
    return d
EV = os.path.join(os.environ.get('FN_WORK', 'work'), 'evals')
REFS = {'': load(EV + '/fp8ref/results.jsonl')}
if os.path.exists(EV + '/e120_fp8/results.jsonl'):
    REFS['e120_'] = load(EV + '/e120_fp8/results.jsonl')
for p in sorted(glob.glob(EV + '/*/results.jsonl')):
    name = os.path.basename(os.path.dirname(p))
    if name in ('fp8ref', 'e120_fp8', 'r_fp8'):
        continue
    if name.startswith('r_') and 'r_' not in REFS and os.path.exists(EV + '/r_fp8/results.jsonl'):
        REFS['r_'] = load(EV + '/r_fp8/results.jsonl')
    ref = REFS['e120_'] if name.startswith('e120_') else REFS['r_'] if name.startswith('r_') else REFS['']
    c = load(p)
    tasks = defaultdict(list)
    for k in c:
        prs = [p for p in ((0, 1), (2, 3)) if all((x in c[k] and x in ref.get(k, {}) for x in p))]
        if not prs:
            continue
        att = [x for p in prs for x in p]
        a2 = sum((float(c[k][i]['sub_correct'] or c[k][j]['sub_correct']) for i, j in prs)) / len(prs)
        b2 = sum((float(ref[k][i]['sub_correct'] or ref[k][j]['sub_correct']) for i, j in prs)) / len(prs)
        a1 = sum((c[k][x]['sub_correct'] for x in att)) / len(att)
        b1 = sum((ref[k][x]['sub_correct'] for x in att)) / len(att)
        tasks[k[0]].append((a2, b2, a1, b1))
    T = list(tasks)
    if not T:
        continue

    def stat(ts):
        s = [0.0] * 4
        for t in ts:
            v = tasks[t]
            for i in range(4):
                s[i] += sum((x[i] for x in v)) / len(v)
        return [x / len(ts) * 100 for x in s]
    a2, b2, a1, b1 = stat(T)
    rng = random.Random(0)
    d2, d1 = ([], [])
    for _ in range(4000):
        bs = [rng.choice(T) for _ in T]
        x = stat(bs)
        d2.append(x[0] - x[1])
        d1.append(x[2] - x[3])
    d2.sort()
    d1.sort()
    q = lambda v: (v[int(0.05 * len(v))], v[int(0.95 * len(v))])
    print(f'{name:10s} tasks={len(T):2d} pass@2 {a2:5.1f} vs {b2:5.1f} d={a2 - b2:+5.1f} 90%CI [{q(d2)[0]:+5.1f},{q(d2)[1]:+5.1f}] | pass@1 {a1:5.1f} vs {b1:5.1f} d={a1 - b1:+5.1f} [{q(d1)[0]:+5.1f},{q(d1)[1]:+5.1f}]')
