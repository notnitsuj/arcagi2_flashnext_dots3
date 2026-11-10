import sys
from pathlib import Path
import numpy as np

def main():
    td = Path(sys.argv[1])
    bds = [Path(p) for p in sys.argv[2:]]
    keys = sorted(set.intersection(*[{f.stem for f in d.glob('*.npz')} for d in [td] + bds]))
    print(len(keys), 'common sequences')
    for bd in bds:
        acc = {}
        for k in keys:
            t, s = (np.load(td / f'{k}.npz'), np.load(bd / f'{k}.npz'))
            T = min(len(t['ids']), len(s['ids']))
            plen = int(t['plen'])
            m = np.zeros(T, bool)
            for a, b in t['gen']:
                m[a:min(b, T)] = True
            m[:plen] = False
            pos = np.nonzero(m)[0] - plen
            tid, tlp = (t['top_ids'][pos], t['top_lp'][pos].astype(np.float64))
            sid, slp = (s['top_ids'][pos], s['top_lp'][pos].astype(np.float64))
            floor = slp.min(1, keepdims=True)
            match = tid[:, :, None] == sid[:, None, :]
            sl = np.where(match.any(2), (match * slp[:, None, :]).sum(2), floor)
            pt, ps = (np.exp(tlp), np.exp(sl))
            valid = tid >= 0
            tail_t = np.clip(1 - (pt * valid).sum(1), 1e-08, None)
            tail_s = np.clip(1 - (ps * valid).sum(1), 1e-08, None)
            kl = (pt * (tlp - sl) * valid).sum(1) + tail_t * (np.log(tail_t) - np.log(tail_s))
            agree = tid[:, 0] == sid[:, 0]
            mode = 'repl' if '_repl_' in k else 'direct'
            a = acc.setdefault(mode, [0.0, 0.0, 0])
            a[0] += kl.sum()
            a[1] += agree.sum()
            a[2] += len(kl)
        print(bd.name, {m: {'kl': round(v[0] / v[2], 5), 'top1': round(v[1] / v[2], 4), 'n': v[2]} for m, v in sorted(acc.items())})
if __name__ == '__main__':
    main()
