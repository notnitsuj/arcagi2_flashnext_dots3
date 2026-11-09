import argparse, json
from collections import Counter, defaultdict

def cls(r):
    ver = [p for p in r.get('progs', []) if p.get('verified')]
    if ver:
        return 'vr' if all(ver[0].get('tests_correct') or [False]) else 'vw'
    ok = r.get('answer_correct') or []
    return 'ur' if ok and all(ok) and (len(ok) == r.get('n_test', len(ok))) else 'uw'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dirs', nargs='+')
    ap.add_argument('--tasks')
    ap.add_argument('--informative')
    a = ap.parse_args()
    sel = set(open(a.tasks).read().split()) if a.tasks else None
    per_task = defaultdict(list)
    for d in a.dirs:
        c, used, n = (Counter(), 0, 0)
        for l in open(f'{d}/samples.jsonl'):
            r = json.loads(l)
            if r.get('arm') != 'repl' or r.get('status') == 'error':
                continue
            if sel and r['task'].split('~')[0].split('@')[0] not in sel:
                continue
            k = cls(r)
            c[k] += 1
            used += r.get('used', 0)
            n += 1
            per_task[r['task']].append(k)
        v = c['vr'] + c['vw']
        print(f"{d}: n={n} verified={v / max(n, 1):.3f} precision={c['vr'] / max(v, 1):.3f} solve={(c['vr'] + c['ur']) / max(n, 1):.3f} {dict(c)} tok={used / max(n, 1) / 1000.0:.1f}k")
    inf = sorted((t for t, ks in per_task.items() if any((k != 'vr' for k in ks))))
    mixed = [t for t in inf if any((k in ('vr', 'ur') for k in per_task[t]))]
    print(f'tasks={len(per_task)} not-all-vr={len(inf)} mixed(some right)={len(mixed)}')
    if a.informative:
        open(a.informative, 'w').write('\n'.join(inf) + '\n')
if __name__ == '__main__':
    main()
