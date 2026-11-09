import argparse, os
import torch
ap = argparse.ArgumentParser()
ap.add_argument('--cap', required=True)
ap.add_argument('--scores', required=True)
ap.add_argument('--ks', default='256,320,384')
ap.add_argument('--php', type=float, default=0.99)
ap.add_argument('--out', required=True)
ap.add_argument('--tag', default='hd')
a = ap.parse_args()
L = range(49)
st = lambda p: torch.load(f'{a.cap}/{p}/stats.pt')
P, H, C, R = (st('hd_pre'), st('hd_post'), st('tc_call'), st('tc_resp'))
T0 = st('tc_pre')
hd = {l: (H['wsum'][l] - P['wsum'][l]).clamp_min(0) for l in L}
call = {l: (C['wsum'][l] - T0['wsum'][l]).clamp_min(0) for l in L}
resp = {l: (R['wsum'][l] - C['wsum'][l]).clamp_min(0) for l in L}
gen = st('tr_a')['wsum']
sc = torch.load(a.scores)
print('header tokens/layer:', H['tokens'][0] - P['tokens'][0])

def cov(keep, w):
    c = [float(w[l][keep[l]].sum() / w[l].sum()) for l in L]
    return (sum(c) / len(c), min(c))
for n in ('mix0.5_256', 'mix0.5_320', 'mix0.5_384'):
    if not os.path.exists(f'{a.out}/keep_{n}.pt'):
        continue
    k = torch.load(f'{a.out}/keep_{n}.pt')['keep']
    print(f'{n:14s} header cov mean %.4f min %.4f | call cov %.4f' % (*cov(k, hd), cov(k, call)[0]))
for K in [int(x) for x in a.ks.split(',')]:
    keep = torch.zeros(49, 512, dtype=torch.bool)
    npro = []
    for l in L:
        o = torch.argsort(hd[l], descending=True)
        c = torch.cumsum(hd[l][o], 0) / hd[l].sum()
        pro = o[:min(int((c < a.php).sum()) + 1, K)]
        keep[l, pro] = True
        npro.append(len(pro))
        s = sc[l] / sc[l].sum() + 0.5 * call[l] / call[l].sum() + 0.25 * resp[l] / resp[l].sum()
        rest = [int(e) for e in torch.argsort(s, descending=True) if not keep[l, e]]
        keep[l, rest[:K - len(pro)]] = True
    torch.save({'keep': keep, 'crit': f'header-protect {a.php} + mix0.5', 'K': K}, f'{a.out}/keep_{a.tag}_{K}.pt')
    print(f'{a.tag}_{K}: protected/layer mean {sum(npro) / 49:.0f} max {max(npro)} | header cov %.4f min %.4f | call cov %.4f | general cov %.4f min %.4f' % (*cov(keep, hd), cov(keep, call)[0], *cov(keep, gen)))
