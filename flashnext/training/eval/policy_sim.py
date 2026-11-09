import argparse, csv, json, os, random
from collections import defaultdict
from pathlib import Path

def key(g):
    return None if g is None else json.dumps(g, separators=(',', ':'))

class Pool:

    def __init__(self, dirs):
        self.direct = defaultdict(list)
        self.repl = defaultdict(list)
        self.n_test, self.gt = ({}, {})
        for d in dirs:
            for l in open(Path(d) / 'samples.jsonl'):
                r = json.loads(l)
                if r['status'] == 'error':
                    continue
                self.n_test[r['task']] = r['n_test']
                if r['arm'] == 'direct':
                    self.direct[r['task'], r['test']].append(r)
                else:
                    self.repl[r['task']].append(r)
        self.tasks = sorted(self.n_test)
        self.marks = sorted({c['at'] for v in self.direct.values() for r in v for c in r['ckpts']})

    def direct_out(self, r, cap, force_tok):
        if r['used'] <= cap and r['status'] != 'cap':
            kind = 'direct' if r['status'] == 'answer' else 'forced'
            return (r['used'], (r['answer'], kind, r.get('answer_correct', False)))
        cks = [c for c in r['ckpts'] if c['at'] <= cap]
        if not cks:
            return (cap, (None, 'none', False))
        c = cks[-1]
        return (c['at'] + c['tokens'], (c['grid'], 'forced', c['correct']))

    def repl_out(self, r, cap, n_test):
        progs = [p for p in r.get('progs', []) if p['used'] <= cap]
        ver = [p for p in progs if p['verified']]
        out = {}
        if ver:
            p = ver[-1]
            for k in range(n_test):
                g = p['tests'][k] if k < len(p['tests']) else None
                if g is not None:
                    out[k] = [(g, 'verified', 1.0, p['tests_correct'][k])]
            tok = min(r['used'], max((p['used'] for p in ver))) if r['status'] == 'verified' else min(r['used'], cap)
            return (tok, out)
        if r['used'] <= cap:
            for k, g in enumerate(r.get('answer') or []):
                if k < n_test:
                    out.setdefault(k, []).append((g, 'forced' if r.get('forced') else 'direct', 1.0, r['answer_correct'][k] if k < len(r['answer_correct']) else False))
            tok = r['used']
        else:
            tok = cap
        nv = [p for p in progs if p.get('tests') and p['n_ok'] > 0]
        if nv:
            b = max(nv, key=lambda p: (p['n_ok'], p['k']))
            for k in range(n_test):
                g = b['tests'][k] if k < len(b['tests']) else None
                if g is not None:
                    out.setdefault(k, []).append((g, 'partial', b['n_ok'] / len(b['train_ok']), b['tests_correct'][k]))
        return (tok, out)

def parse_pattern(p):
    pre, _, rep = p.partition('*')
    return (list(pre), list(rep))

def simulate(pool, tasks, pol, budget_per_input, rng):
    pre, rep = parse_pattern(pol['pattern'])
    n_inputs = sum((pool.n_test[t] for t in tasks))
    B = budget_per_input * n_inputs
    order = {}
    for t in tasks:
        order['R', t] = rng.sample(range(len(pool.repl.get(t, []))), len(pool.repl.get(t, [])))
        for k in range(pool.n_test[t]):
            v = pool.direct.get((t, k), [])
            order['D', t, k] = rng.sample(range(len(v)), len(v))
    st = {t: {'units': 0, 'ptr': defaultdict(int), 'cands': [defaultdict(lambda: [0.0, 0, 0, None, set()]) for _ in range(pool.n_test[t])], 'done': False, 'tok': 0} for t in tasks}
    used = 0
    w = {'verified': pol['wv'], 'direct': pol['wd'], 'forced': pol['wf'], 'partial': pol['wp']}

    def settled(t, k):
        c = st[t]['cands'][k]
        if not c:
            return False
        top = max(c.values(), key=lambda x: x[0])
        nv, nd = (top[1], top[2])
        rule, s = (pol['settle_rule'], pol['settle'])
        if rule == 'none':
            return False
        if rule == 'cross':
            return nv >= 1 and nd >= 1 and (nv + nd >= s)
        if rule == 'ver2':
            return nv >= s
        return nv + nd >= s

    def add(t, k, g, kind, wt, ok):
        if g is None:
            return
        c = st[t]['cands'][k][key(g)]
        c[0] += w[kind] * wt
        c[1] += kind == 'verified'
        c[2] += kind == 'direct'
        c[3] = ok
    exhausted_unsettled = 0
    rounds = 0
    while True:
        active = [t for t in tasks if not st[t]['done']]
        if not active:
            break
        active.sort(key=lambda t: (st[t]['units'], t))
        progressed = False
        for t in active:
            s = st[t]
            u = s['units']
            arm = pre[u] if u < len(pre) else rep[(u - len(pre)) % len(rep)] if rep else None
            if arm is None or s['units'] >= pol['kmax']:
                s['done'] = True
                continue
            unset = [k for k in range(pool.n_test[t]) if not settled(t, k)]
            if not unset:
                s['done'] = True
                continue
            if arm == 'R':
                i = s['ptr']['R']
                if i >= len(order['R', t]):
                    if 'D' in pre + rep and any((s['ptr']['D', k] < len(order['D', t, k]) for k in unset)):
                        arm = 'D'
                    else:
                        s['done'] = True
                        exhausted_unsettled += 1
                        continue
            if arm == 'D':
                avail = [k for k in unset if s['ptr']['D', k] < len(order['D', t, k])]
                if not avail:
                    if s['ptr']['R'] < len(order['R', t]) and 'R' in pre + rep:
                        arm = 'R'
                    else:
                        s['done'] = True
                        exhausted_unsettled += 1
                        continue
            need = pol['cap_r'] if arm == 'R' else pol['cap_d'] * len(avail)
            if used + need > B:
                continue
            if arm == 'R':
                r = pool.repl[t][order['R', t][s['ptr']['R']]]
                s['ptr']['R'] += 1
                tok, out = pool.repl_out(r, pol['cap_r'], pool.n_test[t])
                for k, lst in out.items():
                    for g, kind, wt, ok in lst:
                        add(t, k, g, kind, wt, ok)
            else:
                tok = 0
                for k in avail:
                    r = pool.direct[t, k][order['D', t, k][s['ptr']['D', k]]]
                    s['ptr']['D', k] += 1
                    tk, (g, kind, ok) = pool.direct_out(r, pol['cap_d'], 0)
                    tok += tk
                    if kind != 'none':
                        add(t, k, g, kind, 1.0, ok)
            used += tok
            s['tok'] += tok
            s['units'] += 1
            progressed = True
        rounds += 1
        if not progressed:
            break
    res = {}
    for t in tasks:
        p2 = p1 = 0
        for k in range(pool.n_test[t]):
            c = st[t]['cands'][k]
            top = sorted(c.values(), key=lambda x: -x[0])[:2]
            p2 += any((x[3] for x in top))
            p1 += bool(top and top[0][3])
        res[t] = (p2 / pool.n_test[t], p1 / pool.n_test[t], st[t]['tok'])
    return (res, used / n_inputs, exhausted_unsettled)
POLICIES = {'direct': dict(pattern='*D', settle_rule='any2', settle=2), 'direct_nostop': dict(pattern='*D', settle_rule='none', settle=99), 'repl_ver2': dict(pattern='*R', settle_rule='ver2', settle=2), 'repl_ver1': dict(pattern='*R', settle_rule='ver2', settle=1), 'repl_nostop': dict(pattern='*R', settle_rule='none', settle=99), 'repl_ver3': dict(pattern='*R', settle_rule='ver2', settle=3), 'repl_ver4': dict(pattern='*R', settle_rule='ver2', settle=4), 'repl_any3': dict(pattern='*R', settle_rule='any2', settle=3), 'hyb_D*R_cross': dict(pattern='D*R', settle_rule='cross', settle=2), 'hyb_D*R_any2': dict(pattern='D*R', settle_rule='any2', settle=2), 'hyb_R*D_cross': dict(pattern='R*D', settle_rule='cross', settle=2), 'hyb_DR_cross': dict(pattern='*DR', settle_rule='cross', settle=2), 'hyb_RD_any2': dict(pattern='*RD', settle_rule='any2', settle=2), 'hyb_DR_any2': dict(pattern='*DR', settle_rule='any2', settle=2), 'hyb_RRD_any2': dict(pattern='*RRD', settle_rule='any2', settle=2), 'hyb_D*R_any3': dict(pattern='D*R', settle_rule='any2', settle=3)}
DEF = dict(wv=1.0, wd=1.0, wf=0.3, wp=0.25, kmax=99, cap_r=98304, cap_d=98304)

def atlas_classes(p=None):
    if not p or not os.path.isfile(p):
        return {}
    return {(r['task'], int(r['test'])): r['klass'] for r in csv.DictReader(open(p))}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pools', nargs='+')
    ap.add_argument('--budgets', default='59000,100000,118000,140000,165000,185000,200000,235000,353000')
    ap.add_argument('--policies', default='all')
    ap.add_argument('--caps', default='', help='sweep caps: cap_r values (comma) applied to REPL arms')
    ap.add_argument('--capd', default='', help='sweep direct caps (must be pool checkpoint marks)')
    ap.add_argument('--draws', type=int, default=200)
    ap.add_argument('--tasks', default=None)
    ap.add_argument('--json', default=None)
    ap.add_argument('--by-class', action='store_true')
    ap.add_argument('--atlas', default=None, help='CSV with task, test, klass columns (used by --by-class)')
    a = ap.parse_args()
    pool = Pool(a.pools)
    tasks = pool.tasks
    if a.tasks:
        sel = set(open(a.tasks[1:]).read().split() if a.tasks.startswith('@') else a.tasks.split(','))
        tasks = [t for t in tasks if t in sel]
    print(f'pool: {len(tasks)} tasks, {sum((pool.n_test[t] for t in tasks))} inputs, {sum((len(v) for v in pool.direct.values()))} direct + {sum((len(v) for v in pool.repl.values()))} REPL samples; direct marks {pool.marks}')
    names = list(POLICIES) if a.policies == 'all' else a.policies.split(',')
    caps_r = [int(x) for x in a.caps.split(',') if x] or [DEF['cap_r']]
    caps_d = [int(x) for x in a.capd.split(',') if x] or [DEF['cap_d']]
    klass = atlas_classes(a.atlas) if a.by_class else {}
    rows = []
    for B in [int(x) for x in a.budgets.split(',')]:
        for nm in names:
            for cr in caps_r:
                for cd in caps_d:
                    pol = DEF | POLICIES[nm] | {'cap_r': cr, 'cap_d': cd}
                    if 'R' not in pol['pattern'] and cr != caps_r[0]:
                        continue
                    if 'D' not in pol['pattern'] and cd != caps_d[0]:
                        continue
                    rng = random.Random(1234)
                    p2s, p1s, toks, exh = ([], [], [], [])
                    per_task = defaultdict(list)
                    for _ in range(a.draws):
                        res, tpi, ex = simulate(pool, tasks, pol, B, rng)
                        bt = [rng.choice(tasks) for _ in tasks]
                        p2s.append(100 * sum((res[t][0] for t in tasks)) / len(tasks))
                        p1s.append(100 * sum((res[t][1] for t in tasks)) / len(tasks))
                        toks.append(tpi)
                        exh.append(ex)
                        per_task['boot'].append(100 * sum((res[t][0] for t in bt)) / len(bt))
                        for t in tasks:
                            per_task[t].append(res[t][0])
                    bo = sorted(per_task['boot'])
                    row = {'B': B, 'policy': nm, 'cap_r': cr, 'cap_d': cd, 'pass2': sum(p2s) / len(p2s), 'pass1': sum(p1s) / len(p1s), 'ci_lo': bo[int(0.025 * len(bo))], 'ci_hi': bo[int(0.975 * len(bo)) - 1], 'tok_per_input': sum(toks) / len(toks), 'exhausted': sum(exh) / len(exh)}
                    if klass:
                        byc = defaultdict(lambda: [0.0, 0])
                        for t in tasks:
                            m = sum(per_task[t]) / len(per_task[t])
                            for k in range(pool.n_test[t]):
                                c = klass.get((t, k), '?')
                                byc[c][0] += m
                                byc[c][1] += 1
                        row['by_class'] = {c: round(100 * v[0] / v[1], 1) for c, v in byc.items()}
                    rows.append(row)
                    print(f"B {B / 1000.0:4.0f}k  {nm:16s} capR {cr // 1024:3d}k capD {cd // 1024:3d}k  pass@2 {row['pass2']:5.1f} [{row['ci_lo']:4.1f},{row['ci_hi']:4.1f}]  pass@1 {row['pass1']:5.1f}  tok/in {row['tok_per_input'] / 1000.0:5.0f}k  exh {row['exhausted']:.1f}" + (f"  {row['by_class']}" if klass else ''), flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1))
if __name__ == '__main__':
    main()
