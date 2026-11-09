import argparse, json, math, queue, random, sys, threading, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
sys.path.insert(0, str(Path(__file__).resolve().parent))
from accept import BUILD, HOT, MODEL, eval_rec, load_rec, pad128, summarize, target_probs, token_types
from mtp_torch import MTP

def loader(ds, epochs, seed):
    q = queue.Queue(maxsize=3)

    def run():
        rng = random.Random(seed)
        for ep in range(epochs):
            order = list(ds)
            rng.shuffle(order)
            for d in order:
                q.put((ep, d, load_rec(d, None)))
        q.put(None)
    threading.Thread(target=run, daemon=True).start()
    while True:
        x = q.get()
        if x is None:
            return
        yield x

def kl_chunk(m, sample, W, pidx, p, tv):
    lg = (sample @ W.t()).float()
    lq = lg.log_softmax(-1)
    lqp = lq.gather(1, pidx.clamp(min=0))
    valid = pidx >= 0
    pl = torch.where(valid, p, torch.zeros_like(p))
    loss = (pl * (torch.log(pl.clamp_min(1e-30)) - lqp)).sum(-1)
    if tv:
        v, _ = lg.topk(20, -1)
        kth = v[:, -1:]
        lse = torch.logsumexp(v, -1, keepdim=True)
        lg_at = lg.gather(1, pidx.clamp(min=0))
        q20 = torch.where((lg_at >= kth) & valid, (lg_at - lse).exp(), torch.zeros_like(lg_at))
        loss = loss + tv * (1 - torch.minimum(pl, q20).sum(-1))
    return loss.sum()

def kl_alpha_chunk(m, sample, W, pidx, p):
    lg = (sample @ W.t()).float()
    valid = pidx >= 0
    pl = torch.where(valid, p, torch.zeros_like(p))
    lqp = lg.log_softmax(-1).gather(1, pidx.clamp(min=0))
    kl = (pl * (torch.log(pl.clamp_min(1e-30)) - lqp)).sum(-1)
    v, _ = lg.topk(20, -1)
    lse = torch.logsumexp(v, -1, keepdim=True)
    lg_at = lg.gather(1, pidx.clamp(min=0))
    q20 = torch.where((lg_at >= v[:, -1:]) & valid, (lg_at - lse).exp(), torch.zeros_like(lg_at))
    return (kl, torch.minimum(pl, q20).sum(-1))

def parse_bins(spec):
    if not spec:
        return None
    ub, w = zip(*[(int(x.split(':')[0]), float(x.split(':')[1])) for x in spec.split(',')])
    return (torch.tensor(ub), torch.tensor(w))

def pos_weights(T, bins, ty, prose_w, dev):
    w = torch.ones(T, device=dev)
    if bins is not None:
        ub, bw = bins
        i = torch.bucketize(torch.arange(T), ub, right=True).clamp(max=len(bw) - 1)
        w = w * bw[i].to(dev)
    if prose_w != 1.0:
        w = w * torch.where(ty.to(dev) == 0, prose_w, 1.0)
    return w

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', required=True, nargs='+')
    ap.add_argument('--out', required=True)
    ap.add_argument('--ho-dump', default='')
    ap.add_argument('--ho-n', type=int, default=24)
    ap.add_argument('--groups', default='dense,router,scales,norms')
    ap.add_argument('--steps', type=int, default=3, help='draft steps trained (1 = step 0 only)')
    ap.add_argument('--sw', default='1,0.6,0.4,0.3')
    ap.add_argument('--lr', type=float, default=2e-05)
    ap.add_argument('--lr-scales', type=float, default=0.0005)
    ap.add_argument('--lr-router', type=float, default=2e-05)
    ap.add_argument('--wd', type=float, default=0.0)
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--accum', type=int, default=4)
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--eval-every', type=int, default=25)
    ap.add_argument('--max-updates', type=int, default=0)
    ap.add_argument('--tv', type=float, default=0.0)
    ap.add_argument('--chunk', type=int, default=8192)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-train-len', type=int, default=40960, help='crop records to their first N positions')
    ap.add_argument('--init', default='', help='start from a train.py checkpoint (e.g. an earlier run best.pt)')
    ap.add_argument('--obj', default='kl', choices=['kl', 'elen'], help='elen: maximise expected accepted length sum_s prod_{j<=s} alpha_j (+ --kl-w x KL anchor)')
    ap.add_argument('--kl-w', type=float, default=0.3)
    ap.add_argument('--select', default='tok_step3', choices=['tok_step3', 'tok_step'], help='early-stop metric')
    ap.add_argument('--prose-w', type=float, default=1.0, help='position weight for prose (not grid/code) positions')
    ap.add_argument('--ctx-bins', default='', help="context-bin position weights 'UB:W,...' (deployed traffic / trace)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(a), open(out / 'args.json', 'w'), indent=1)
    torch.manual_seed(a.seed)
    dev = torch.device('cuda')
    m = MTP(MODEL, BUILD, dev)
    groups = tuple((g for g in a.groups.split(',') if g))
    if a.init:
        sd0 = torch.load(a.init, map_location='cpu')
        assert not m.load_trained(sd0.get('params', sd0)), 'init: unknown params'
        print('init from', a.init, flush=True)
    m.set_trainable(groups)
    bins = parse_bins(a.ctx_bins)
    tr = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
    print('trainable:', [(n, tuple(p.shape)) for n, p in tr], flush=True)
    print('n trainable', sum((p.numel() for _, p in tr)), flush=True)
    master = {n: p.detach().float().clone().requires_grad_(False) for n, p in tr}
    pg = []
    for n, _ in tr:
        lr = a.lr_scales if n in ('moe.tgu', 'moe.td') else a.lr_router if n == 'moe.gate' else a.lr
        pg.append({'params': [master[n]], 'lr': lr, 'base_lr': lr, 'name': n})
    opt = torch.optim.AdamW(pg, betas=(0.9, 0.99), eps=1e-08, weight_decay=a.wd)
    hot = torch.as_tensor((lambda x: x['ids'] if isinstance(x, dict) else x)(torch.load(HOT))).long().to(dev)
    hot_mask = torch.zeros(m.lm_head.shape[0], dtype=torch.bool, device=dev)
    hot_mask[hot] = True
    pos_in_hot = torch.full((m.lm_head.shape[0],), -1, dtype=torch.long, device=dev)
    pos_in_hot[hot] = torch.arange(len(hot), device=dev)
    W = m.lm_head[hot]
    sw = [float(x) for x in a.sw.split(',')][:a.steps]
    ds = []
    for dd in a.dump:
        ds += sorted((p.parent for p in Path(dd).glob('*/DONE')))
    ho = sorted((p.parent for p in Path(a.ho_dump).glob('*/DONE')))[:a.ho_n] if a.ho_dump else []
    n_upd = a.max_updates or math.ceil(len(ds) * a.epochs / a.accum)
    print(f'train records {len(ds)}, ho {len(ho)}, updates {n_upd}', flush=True)
    log = open(out / 'log.jsonl', 'a')

    def evaluate():
        m.eval()
        As = [eval_rec(m, d, max(3, a.steps), dev, hot_mask)[0] for d in ho]
        r = summarize(torch.cat(As))
        r['per_seq_tok_step'] = round(float(np.mean([summarize(x)['tok_step'] for x in As])), 4)
        m.train()
        return r
    best = None
    if ho:
        r0 = evaluate()
        print('eval base', json.dumps(r0), flush=True)
        log.write(json.dumps({'upd': 0, 'eval': r0}) + '\n')
        log.flush()
        best = (r0[a.select], 0)
    upd, seen, t0 = (0, 0, time.time())
    acc_loss, acc_n = ([0.0] * a.steps, [0] * a.steps)
    elen_log = []
    for ep, d, (ids, gen, hid, tid, tlp) in loader(ds, a.epochs, a.seed):
        if ids.shape[0] > a.max_train_len:
            n_ = a.max_train_len
            ids, gen, hid, tid, tlp = (ids[:n_], gen[:n_], hid[:n_], tid[:n_], tlp[:n_])
        T = ids.shape[0]
        idsd = ids.to(dev)
        genp = torch.cat([gen, torch.zeros(a.steps + 3, dtype=torch.bool)]).to(dev)
        tok = lambda s: pad128(torch.cat([idsd[1 + s:], idsd.new_zeros(1 + s)]), T)
        h = pad128(hid.to(dev, non_blocking=True), T)
        tidd, tlpd = (tid.to(dev), tlp.to(dev))
        sample, multi, st = m.step0(h, tok(0))
        extra = []
        total = 0.0
        if a.obj == 'elen':
            ty = token_types(ids)[torch.arange(T).add(2).clamp(max=T - 1)]
            b0 = torch.arange(T - 2, device=dev)
            b0 = b0[genp[b0 + 1] & genp[b0 + 2]]
            w = pos_weights(T, bins, ty, a.prose_w, dev)[b0]
            W_ = w / w.sum().clamp_min(1e-06)
            cp = torch.ones(len(b0), device=dev)
            elen = 0.0
            for s in range(a.steps):
                if s > 0:
                    sample, multi = m.stepn(multi, tok(s), s, st, extra)
                ok = (b0 + 2 + s < T) & genp[(b0 + 2 + s).clamp(max=T + 1)]
                r = (b0 + 1 + s).clamp(max=T - 1)
                p = target_probs(tidd[r], tlpd[r])
                pidx = pos_in_hot[tidd[r]]
                kls, als = ([], [])
                for i in range(0, len(b0), a.chunk):
                    sl = slice(i, i + a.chunk)
                    k_, a_ = checkpoint(kl_alpha_chunk, m, sample[b0[sl]], W, pidx[sl], p[sl], use_reentrant=False)
                    kls.append(k_)
                    als.append(a_)
                kl, al = (torch.cat(kls), torch.cat(als))
                al = torch.where(ok, al, torch.zeros_like(al))
                cp = cp * al
                elen = elen + (W_ * cp).sum()
                L_kl = (W_ * torch.where(ok, kl, torch.zeros_like(kl))).sum()
                total = total + a.kl_w * sw[s] * L_kl
                acc_loss[s] += float(L_kl)
                acc_n[s] += 1
            total = total - elen
            acc_loss[0] += 0.0
            elen_log.append(float(elen))
        for s in range(a.steps if a.obj == 'kl' else 0):
            if s > 0:
                sample, multi = m.stepn(multi, tok(s), s, st, extra)
            bt = torch.arange(T - 2 - s, device=dev)
            ok = genp[bt + 1] & genp[bt + 2 + s]
            bt = bt[ok]
            r = bt + 1 + s
            p = target_probs(tidd[r], tlpd[r])
            pidx = pos_in_hot[tidd[r]]
            L = 0.0
            for i in range(0, len(bt), a.chunk):
                sl = slice(i, i + a.chunk)
                L = L + checkpoint(kl_chunk, m, sample[bt[sl]], W, pidx[sl], p[sl], a.tv, use_reentrant=False)
            n = max(1, len(bt))
            acc_loss[s] += float(L) / n
            acc_n[s] += 1
            total = total + sw[s] * L / n
        (total / a.accum).backward()
        del sample, multi, st, extra, h
        seen += 1
        if seen % a.accum:
            continue
        lrf = min(1.0, (upd + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, upd / max(1, n_upd))))
        for g in opt.param_groups:
            g['lr'] = g['base_lr'] * lrf
        gn = 0.0
        for n, p in tr:
            gr = p.grad.float() if p.grad is not None else torch.zeros_like(master[n])
            master[n].grad = gr
            gn += float(gr.pow(2).sum())
            p.grad = None
        gn = gn ** 0.5
        torch.nn.utils.clip_grad_norm_([master[n] for n, _ in tr], 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            for n, p in tr:
                p.copy_(master[n].to(p.dtype))
        upd += 1
        rec = {'upd': upd, 'ep': ep, 'loss': [round(x / max(1, c), 5) for x, c in zip(acc_loss, acc_n)], 'gnorm': round(gn, 4), 'lr': lrf, 't': round(time.time() - t0), 'mem': round(torch.cuda.max_memory_allocated() / 2 ** 30, 1)}
        if elen_log:
            rec['elen'] = round(float(np.mean(elen_log)), 4)
            elen_log.clear()
        acc_loss, acc_n = ([0.0] * a.steps, [0] * a.steps)
        if ho and (upd % a.eval_every == 0 or upd == n_upd):
            r = evaluate()
            rec['eval'] = r
            if r[a.select] > best[0]:
                best = (r[a.select], upd)
                torch.save({'params': {n: master[n].detach().cpu() for n, _ in tr}, 'upd': upd, 'eval': r}, out / 'best.pt')
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + '\n')
        log.flush()
        if upd >= n_upd:
            break
    torch.save({'params': {n: master[n].detach().cpu() for n, _ in tr}, 'upd': upd}, out / 'last.pt')
    print('DONE best', best, flush=True)
if __name__ == '__main__':
    main()
