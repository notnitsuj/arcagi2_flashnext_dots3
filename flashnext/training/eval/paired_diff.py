import argparse, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from policy_sim import Pool, simulate, POLICIES, DEF

def per_task(pool, tasks, pol, B, draws):
    rng = random.Random(1234)
    acc = {t: 0.0 for t in tasks}
    tok = 0.0
    for _ in range(draws):
        res, tpi, _ = simulate(pool, tasks, pol, B, rng)
        tok += tpi
        for t in tasks:
            acc[t] += res[t][0]
    return ({t: v / draws for t, v in acc.items()}, tok / draws)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', nargs='+', required=True)
    ap.add_argument('--ref', nargs='+', required=True)
    ap.add_argument('--budgets', default='50000,80000,118000,140000')
    ap.add_argument('--policy', default='repl_nostop')
    ap.add_argument('--cap', type=int, default=81920)
    ap.add_argument('--draws', type=int, default=100)
    ap.add_argument('--exclude', default='', help='comma list of task ids to leave out (e.g. overlap-audit hits)')
    a = ap.parse_args()
    pb, pr = (Pool(a.build), Pool(a.ref))
    tasks = sorted((set(pb.tasks) & set(pr.tasks)) - set(filter(None, a.exclude.split(','))))
    pol = DEF | POLICIES[a.policy] | {'cap_r': a.cap}
    print(f'{len(tasks)} paired tasks; policy {a.policy}, cap {a.cap}; build REPL samples {sum((len(pb.repl[t]) for t in tasks))}, ref {sum((len(pr.repl[t]) for t in tasks))}')
    for B in [int(x) for x in a.budgets.split(',')]:
        sb, tb = per_task(pb, tasks, pol, B, a.draws)
        sr, tr = per_task(pr, tasks, pol, B, a.draws)
        d = [sb[t] - sr[t] for t in tasks]
        rng = random.Random(7)
        boots = sorted((100 * sum((rng.choice(d) for _ in d)) / len(d) for _ in range(2000)))
        mb, mr = (100 * sum(sb.values()) / len(tasks), 100 * sum(sr.values()) / len(tasks))
        print(f'B {B / 1000.0:4.0f}k  build {mb:5.1f}  ref {mr:5.1f}  diff {mb - mr:+5.1f} [{boots[50]:+5.1f},{boots[1949]:+5.1f}]  tok/in {tb / 1000.0:.0f}k/{tr / 1000.0:.0f}k')
if __name__ == '__main__':
    main()
