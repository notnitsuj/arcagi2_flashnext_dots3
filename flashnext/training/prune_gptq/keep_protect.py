import argparse
import torch
ap = argparse.ArgumentParser()
ap.add_argument('--cap', required=True)
ap.add_argument('--scores', required=True)
ap.add_argument('--ks', default='256,320')
ap.add_argument('--pcall', type=float, default=0.98)
ap.add_argument('--presp', type=float, default=0.95)
ap.add_argument('--out', required=True)
a = ap.parse_args()
S = {p: torch.load(f'{a.cap}/{p}/stats.pt') for p in ('tc_pre', 'tc_call', 'tc_resp')}
L = range(49)
call = {l: (S['tc_call']['wsum'][l] - S['tc_pre']['wsum'][l]).clamp_min(0) for l in L}
resp = {l: (S['tc_resp']['wsum'][l] - S['tc_call']['wsum'][l]).clamp_min(0) for l in L}
print('tokens/layer: call', S['tc_call']['tokens'][0] - S['tc_pre']['tokens'][0], 'resp', S['tc_resp']['tokens'][0] - S['tc_call']['tokens'][0])
sc = torch.load(a.scores)

def cov(keep, w):
    c = [float(w[l][keep[l]].sum() / w[l].sum()) for l in L]
    return (sum(c) / len(c), min(c), c.index(min(c)))
for K in (448, 384, 320, 256, 224, 192):
    try:
        k = torch.load(f'{a.out}/keep_reapsum_{K}.pt')['keep']
    except FileNotFoundError:
        continue
    print(f'reapsum K={K}: tool-call cov mean %.4f min %.4f (L%d) | tool-resp cov mean %.4f min %.4f (L%d)' % (*cov(k, call), *cov(k, resp)))

def need(w, p):
    o = torch.argsort(w, descending=True)
    c = torch.cumsum(w[o], 0) / w.sum()
    return o[:int((c < p).sum()) + 1]
for K in [int(x) for x in a.ks.split(',')]:
    keep = torch.zeros(49, 512, dtype=torch.bool)
    npro = []
    for l in L:
        pro = torch.unique(torch.cat([need(call[l], a.pcall), need(resp[l], a.presp)]))
        if len(pro) > K:
            pro = torch.argsort(call[l] + resp[l], descending=True)[:K]
        keep[l, pro] = True
        npro.append(len(pro))
        rest = [int(e) for e in torch.argsort(sc[l], descending=True) if not keep[l, e]]
        keep[l, rest[:K - len(pro)]] = True
        assert int(keep[l].sum()) == K
    torch.save({'keep': keep, 'crit': f'reapsum+toolprotect(call {a.pcall}, resp {a.presp})', 'K': K}, f'{a.out}/keep_tool_{K}.pt')
    print(f'tool K={K}: protected/layer mean {sum(npro) / 49:.0f} max {max(npro)} | call cov %.4f min %.4f | resp cov %.4f min %.4f' % (cov(keep, call)[0], cov(keep, call)[1], cov(keep, resp)[0], cov(keep, resp)[1]))
    base = {l: sc[l] for l in L}
    tot = torch.load(f'{a.cap}/tr_a/stats.pt')['wsum']
    print(f'   general routed-weight coverage (tr_a): %.4f' % cov(keep, tot)[0])
import os
gen = torch.load(f'{a.cap}/tr_a/stats.pt')['wsum']
for K in [int(x) for x in a.ks.split(',')]:
    for alpha, beta in ((0.5, 0.25), (1.0, 0.5), (2.0, 1.0)):
        keep = torch.zeros(49, 512, dtype=torch.bool)
        for l in L:
            s = sc[l] / sc[l].sum() + alpha * call[l] / call[l].sum() + beta * resp[l] / resp[l].sum()
            keep[l, torch.argsort(s, descending=True)[:K]] = True
        tag = f'mix{alpha:g}_{K}'
        torch.save({'keep': keep, 'crit': f'reapsum+{alpha}*toolcall+{beta}*toolresp', 'K': K}, f'{a.out}/keep_{tag}.pt')
        print(f'{tag}: call cov %.4f min %.4f | resp cov %.4f | general cov %.4f min %.4f' % (cov(keep, call)[0], cov(keep, call)[1], cov(keep, resp)[0], cov(keep, gen)[0], cov(keep, gen)[1]))
