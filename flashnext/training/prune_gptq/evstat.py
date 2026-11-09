import glob, json, os
from collections import defaultdict
EV = os.path.join(os.environ.get('FN_WORK', 'work'), 'evals')
REF = {(r['task'], r['test'], r['attempt']): r for r in map(json.loads, open(EV + '/fp8ref/results.jsonl'))}
for d in sorted(glob.glob(EV + '/*/results.jsonl')):
    rows = [json.loads(l) for l in open(d) if l.strip()]
    if not rows:
        continue
    by = defaultdict(dict)
    for r in rows:
        by[r['task'], r['test']][r['attempt']] = r['sub_correct']
    full = {k: v for k, v in by.items() if len(v) >= 2}
    task = defaultdict(list)
    for (t, ti), v in full.items():
        task[t].append(float(any(v.values())))
    p2 = sum((sum(v) / len(v) for v in task.values())) / max(len(task), 1)
    p1 = sum((r['sub_correct'] for r in rows)) / len(rows)
    ct = sum((r['ct'] for r in rows)) / len(rows)
    fo = sum((r['forced'] for r in rows)) / len(rows)
    com = [r for r in rows if (r['task'], r['test'], r['attempt']) in REF]
    rp1 = sum((REF[r['task'], r['test'], r['attempt']]['sub_correct'] for r in com)) / max(len(com), 1)
    cp1 = sum((r['sub_correct'] for r in com)) / max(len(com), 1)
    rct = sum((REF[r['task'], r['test'], r['attempt']]['ct'] for r in com)) / max(len(com), 1)
    print(f'{os.path.basename(os.path.dirname(d)):10s} n={len(rows):3d} tasks2={len(task):2d} pass@2={100 * p2:5.1f} pass@1={100 * p1:5.1f} ct={ct / 1000.0:5.1f}k forced={fo:.2f} | paired-vs-fp8 pass@1 {100 * cp1:5.1f} vs {100 * rp1:5.1f} (ref ct {rct / 1000.0:4.1f}k)')
