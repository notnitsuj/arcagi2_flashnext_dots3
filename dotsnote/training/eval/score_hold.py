import json, sys, random, collections
for f in sys.argv[1:]:
    if f.endswith('_text.jsonl'):
        continue
    rows = [json.loads(l) for l in open(f) if l.strip()]
    rows = [r for r in rows if 'task' in r]
    if not rows:
        continue
    by = collections.defaultdict(lambda: collections.defaultdict(bool))
    for r in rows:
        by[r['task']][r.get('test', 0)] |= bool(r.get('correct'))
    ts = {t: sum(v.values()) / len(v) for t, v in by.items()}
    n = len(ts)
    s = sum(ts.values())
    vals = list(ts.values())
    rnd = random.Random(0)
    bs = sorted((sum((rnd.choice(vals) for _ in range(n))) / n for _ in range(4000)))
    ct = [r.get('ct', 0) for r in rows]
    parsed = sum((bool(r.get('parsed')) for r in rows)) / len(rows)
    print(f'{f:45s} tasks={n:3d} score={100 * s / n:5.1f}% CI90=[{100 * bs[200]:4.1f},{100 * bs[3800]:4.1f}] parsed={parsed:.2f} mean_ct={sum(ct) / len(ct) / 1000.0:.0f}k rows={len(rows)}')
