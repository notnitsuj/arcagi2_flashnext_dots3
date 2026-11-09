import json, os, re, sys, random, collections
from pathlib import Path
G = Path(os.environ.get('D3_WORK', 'work')) / 'gate'
LOGS = Path(os.environ.get('LOGDIR', os.path.join(os.environ.get('D3_WORK', 'work'), 'logs')))

def load(tag):
    rows = [json.loads(l) for l in open(G / f'{tag}.jsonl')]
    by = collections.defaultdict(list)
    for r in rows:
        by[r['task']].append(r)
    return (rows, by)

def mtp(tag):
    a, b = map(int, open(G / f'{tag}.lines').read().split())
    acc = dr = 0
    for i, l in enumerate(open(LOGS / f'gate_{tag}.srv.log', errors='replace')):
        if a <= i < b and 'SpecDecoding metrics' in l:
            m = re.search('Accepted: (\\d+) tokens, Drafted: (\\d+) tokens', l)
            if m:
                acc += int(m.group(1))
                dr += int(m.group(2))
    p1 = LOGS / f'gate_{tag}.srv.part1.log'
    if p1.exists():
        on = False
        for l in open(p1, errors='replace'):
            on = on or 'startup complete' in l
            m = re.search('Accepted: (\\d+) tokens, Drafted: (\\d+) tokens', l) if on else None
            if m:
                acc += int(m.group(1))
                dr += int(m.group(2))
    return dict(accept_len=round(1 + 3 * acc / max(dr, 1), 3), draft_accept=round(acc / max(dr, 1), 4))

def degen(tag):
    p = LOGS / f'gate_{tag}.degen.log'
    ls = [l.strip() for l in open(p) if l.startswith('degenerate')] if p.exists() else []
    return ls[-1] if ls else None

def stats(by, tasks):
    rs = [r for t in tasks for r in by.get(t, [])]
    c = [r for r in rs if r['correct']]
    return dict(n=len(rs), acc=sum((bool(r['correct']) for r in rs)) / max(len(rs), 1), tok_per_correct=sum((r['ct'] for r in c)) / max(len(c), 1), n_correct=len(c), mean_tok=sum((r['ct'] for r in rs)) / max(len(rs), 1), cap_rate=sum((r['finish'] == 'length' for r in rs)) / max(len(rs), 1), wrong_finished=sum((not r['correct'] and r['finish'] == 'stop' for r in rs)) / max(len(rs), 1))

def matched(bb, bs, tasks):
    nb = ns = 0.0
    for t in tasks:
        for ti in {r['test'] for r in bb.get(t, [])}:
            cb = [r['ct'] for r in bb[t] if r['test'] == ti and r['correct']]
            cs = [r['ct'] for r in bs.get(t, []) if r['test'] == ti and r['correct']]
            if cb and cs:
                nb += sum(cb) / len(cb)
                ns += sum(cs) / len(cs)
    return ns / nb - 1 if nb else float('nan')

def main():
    bt, st = (sys.argv[1], sys.argv[2])
    _, bb = load(bt)
    _, bs = load(st)
    keys = {(r['task'], r['test'], r['attempt']) for v in bb.values() for r in v} & {(r['task'], r['test'], r['attempt']) for v in bs.values() for r in v}
    for by in (bb, bs):
        for t in list(by):
            by[t] = [r for r in by[t] if (r['task'], r['test'], r['attempt']) in keys]
    tasks = sorted({k[0] for k in keys})
    split = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'finetune', 'split.json')))['val']
    out = {'n_tasks': len(tasks), 'n_attempts': len(keys)}
    for name, by in ((bt, bb), (st, bs)):
        out[name] = stats(by, tasks)
        out[name]['by_stratum'] = {s: stats(by, [t for t in tasks if split.get(t) == s]) for s in ('easy', 'hard', 'unk')}
        try:
            out[name]['mtp'] = mtp(name)
        except Exception as e:
            out[name]['mtp'] = repr(e)
        out[name]['degen'] = degen(name)
    rng = random.Random(0)

    def diffs(ts):
        a, b = (stats(bb, ts), stats(bs, ts))
        return (b['acc'] - a['acc'], b['tok_per_correct'] / a['tok_per_correct'] - 1, matched(bb, bs, ts), b['mean_tok'] / a['mean_tok'] - 1)
    point = diffs(tasks)
    boots = [diffs([rng.choice(tasks) for _ in tasks]) for _ in range(10000)]
    names = ['d_acc', 'd_tok_per_correct', 'd_tok_per_correct_matched', 'd_mean_tok']
    out['diff'] = {}
    for i, n in enumerate(names):
        v = sorted((x[i] for x in boots))
        out['diff'][n] = dict(point=round(point[i], 4), ci90=[round(v[500], 4), round(v[9499], 4)])
    json.dump(out, open(G / f'cmp_{bt}_{st}.json', 'w'), indent=1)
    print(json.dumps(out, indent=1))
if __name__ == '__main__':
    main()
