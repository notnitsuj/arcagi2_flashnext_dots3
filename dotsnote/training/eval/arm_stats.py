import itertools, json, random, statistics as st, sys

def load(path):
    d = {}
    ct = []
    for l in open(path):
        r = json.loads(l)
        if 'task' in r:
            d.setdefault(r['task'], {})[r.get('test', 0)] = bool(r.get('correct'))
            ct.append(r.get('ct', 0))
    return ({t: sum(v.values()) / len(v) for t, v in d.items()}, ct)
arms = {}
for spec in sys.argv[1:]:
    name, files = spec.split('=')
    arms[name] = [load(f) for f in files.split(',')]
names = list(arms)
tasks = sorted(set.intersection(*[set(r[0]) for a in arms.values() for r in a]))
rnd = random.Random(0)

def score(run):
    return sum((run[0][t] for t in tasks)) / len(tasks)
for n in names:
    sc = [score(r) for r in arms[n]]
    ct = [c for r in arms[n] for c in r[1]]
    print(f"{n:8s} runs={len(sc):2d} mean={100 * st.mean(sc):5.1f}%  runs: {' '.join((f'{100 * s:.1f}' for s in sc))}  mean_ct={st.mean(ct) / 1000.0:.0f}k cap={sum((c >= 239000 for c in ct)) / len(ct):.3f}")
A = arms[names[0]]
for n in names[1:]:
    B = arms[n]
    mean = lambda runs, t: sum((r[0][t] for r in runs)) / len(runs)
    d = [mean(B, t) - mean(A, t) for t in tasks]
    obs = sum(d) / len(tasks)
    bs2, bs1 = ([], [])
    for _ in range(4000):
        ra = [rnd.choice(A) for _ in A]
        rb = [rnd.choice(B) for _ in B]
        ts = [rnd.choice(tasks) for _ in tasks]
        bs2.append(sum((mean(rb, t) - mean(ra, t) for t in ts)) / len(ts))
        bs1.append(sum((rnd.choice(d) for _ in tasks)) / len(tasks))
    bs2.sort()
    bs1.sort()
    sa, sb = ([score(r) for r in A], [score(r) for r in B])
    allr = sa + sb
    k = len(sb)
    obs_r = st.mean(sb) - st.mean(sa)
    combos = list(itertools.combinations(range(len(allr)), k)) if len(allr) <= 22 else None
    if combos is not None and len(combos) <= 200000:
        ds = [st.mean((allr[i] for i in c)) - st.mean((allr[i] for i in range(len(allr)) if i not in c)) for c in combos]
    else:
        ds = []
        for _ in range(20000):
            p = allr[:]
            rnd.shuffle(p)
            ds.append(st.mean(p[:k]) - st.mean(p[k:]))
    pval = sum((x <= obs_r + 1e-12 for x in ds)) / len(ds)
    print(f'{n} - {names[0]}: {100 * obs:+.1f} pts  90% CI two-level [{100 * bs2[200]:+.1f}, {100 * bs2[3800]:+.1f}]  task-only [{100 * bs1[200]:+.1f}, {100 * bs1[3800]:+.1f}]  run-perm p(B<=A)={pval:.3f}  tasks better {sum((x > 0 for x in d))} worse {sum((x < 0 for x in d))}')
