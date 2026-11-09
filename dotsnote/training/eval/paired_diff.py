import json, sys, random, collections

def arm(spec):
    per = collections.defaultdict(list)
    for f in spec.split(','):
        by = collections.defaultdict(lambda: collections.defaultdict(bool))
        for l in open(f):
            r = json.loads(l)
            if 'task' in r:
                by[r['task']][r.get('test', 0)] |= bool(r.get('correct'))
        for t, v in by.items():
            per[t].append(sum(v.values()) / len(v))
    return {t: sum(v) / len(v) for t, v in per.items()}
A, B = (arm(sys.argv[1]), arm(sys.argv[2]))
ts = sorted(set(A) & set(B))
d = [B[t] - A[t] for t in ts]
n = len(ts)
rnd = random.Random(0)
bs = sorted((sum((rnd.choice(d) for _ in range(n))) / n for _ in range(10000)))
print(f'tasks {n}: A {100 * sum((A[t] for t in ts)) / n:.1f}%  B {100 * sum((B[t] for t in ts)) / n:.1f}%  B-A {100 * sum(d) / n:+.1f} pts, 90% CI [{100 * bs[500]:+.1f}, {100 * bs[9500]:+.1f}]  tasks B better {sum((x > 0 for x in d))}, worse {sum((x < 0 for x in d))}')
