import argparse, json
from collections import defaultdict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results')
    ap.add_argument('--caps', default='32768,65536,98304')
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.results) if l.strip()]
    cell = {(r['mode'], r['task'], r['test'], r['attempt']): r for r in rows}
    modes = sorted({r['mode'] for r in rows})
    inputs = sorted({(r['task'], r['test']) for r in rows})
    out = {}
    for m in modes:
        rs = [r for r in rows if r['mode'] == m]
        n = len(rs)
        p1 = sum((r['sub_correct'] for r in rs)) / n
        by_task = defaultdict(list)
        n_in = 0
        for t, ti in inputs:
            c = [cell.get((m, t, ti, at)) for at in (0, 1)]
            if not all(c):
                continue
            n_in += 1
            by_task[t].append(any((x['sub_correct'] for x in c)))
        p2 = sum((sum(v) / len(v) for v in by_task.values())) / max(1, len(by_task))
        p2_in = sum((sum(v) for v in by_task.values())) / max(1, n_in)
        d = {'attempts': n, 'inputs': n_in, 'tasks': len(by_task), 'pass1': p1, 'pass2_task': p2, 'pass2_input': p2_in, 'mean_ct': sum((r['ct'] for r in rs)) / n, 'forced': sum((r['forced'] for r in rs)) / n, 'wall_s': sum((r['t1'] - r['t0'] for r in rs)) / n, 'tok_per_correct_attempt': sum((r['ct'] for r in rs)) / max(1, sum((r['sub_correct'] for r in rs)))}
        if m == 'repl':
            ver = [r for r in rs if r['any_verified']]
            d['n_tool'] = sum((r['n_tool'] for r in rs)) / n
            d['attempt_cov'] = len(ver) / n
            d['ver_prec'] = sum((r['sub_correct'] for r in ver)) / max(1, len(ver))
            d['n_ver'] = len(ver)
            nov = [r for r in rs if not r['any_verified']]
            d['grid_acc_unverified'] = sum((r['grid_correct'] for r in nov)) / max(1, len(nov))
            d['grid_acc_all'] = sum((r['grid_correct'] for r in rs)) / n
            d['prog_vs_grid_disagree'] = sum((1 for r in ver if r['pred'] is not None and (not any((p['verified'] and p['agree_grid'] for p in r['progs'][-1:]))))) / max(1, len(ver))
            cov_in = [any((cell[m, t, ti, at]['any_verified'] for at in (0, 1))) for t, ti in inputs if all(((m, t, ti, at) in cell for at in (0, 1)))]
            d['input_cov'] = sum(cov_in) / max(1, len(cov_in))
            pv = []
            for t, ti in inputs:
                c = [cell.get((m, t, ti, at)) for at in (0, 1)]
                if all(c) and any((x['any_verified'] for x in c)):
                    pv.append(any((x['any_verified'] and x['sub_correct'] for x in c)))
            d['input_ver_prec_any'] = sum(pv) / max(1, len(pv))
            caps = {}
            for cap in [int(x) for x in a.caps.split(',')]:

                def sub_ok(r):
                    vp = [p for p in r['progs'] if p['verified'] and p['used'] <= cap]
                    if vp:
                        return vp[-1]['test_correct']
                    return r['grid_correct'] and r['ct'] <= cap and (not r['forced'])
                bt = defaultdict(list)
                for t, ti in inputs:
                    c = [cell.get((m, t, ti, at)) for at in (0, 1)]
                    if all(c):
                        bt[t].append(any((sub_ok(x) for x in c)))
                caps[cap] = sum((sum(v) / len(v) for v in bt.values())) / max(1, len(bt))
            d['pass2_task_at_cap_lower_bound'] = caps
        out[m] = d
    if {'direct', 'repl'} <= set(modes):
        bt = defaultdict(list)
        for t, ti in inputs:
            x, y = (cell.get(('direct', t, ti, 0)), cell.get(('repl', t, ti, 0)))
            if x and y:
                bt[t].append(x['sub_correct'] or y['sub_correct'])
        out['direct0+repl0'] = {'pass2_task': sum((sum(v) / len(v) for v in bt.values())) / max(1, len(bt))}
    print(json.dumps(out, indent=1))
if __name__ == '__main__':
    main()
