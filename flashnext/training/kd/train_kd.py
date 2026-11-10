import argparse, glob, json, math, os, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'student'))
import student_fn as S
M = os.environ.get('FN_BASE_MODEL', 'models/Qwen3.8-Flash-Next-FP8')
TC_OPEN, TC_CLOSE = (248058, 248059)

def kl21(h, lm_head, tid, tlp, w):
    ls = torch.log_softmax((h @ lm_head.t()).float(), -1)
    valid = tid >= 0
    lt = torch.where(valid, tlp.float(), torch.full_like(tlp, -10000.0, dtype=torch.float32))
    pt = lt.exp()
    lsk = ls.gather(1, tid.clamp_min(0).long())
    ps = torch.where(valid, lsk.exp(), torch.zeros_like(lsk))
    tail_t = (1 - pt.sum(1)).clamp_min(1e-08)
    tail_s = (1 - ps.sum(1)).clamp_min(1e-08)
    kl = (pt * (lt - lsk)).masked_fill(~valid, 0).sum(1) + tail_t * (tail_t.log() - tail_s.log())
    agree = (ls.argmax(1) == tid[:, 0]).float()
    return ((kl * w).sum(), kl.sum().detach(), agree.sum().detach())

def load(fn, L):
    z = np.load(fn)
    ids = z['ids'][:L]
    T, plen = (len(ids), int(z['plen']))
    rows, = [np.zeros(T, bool)]
    for a, b in z['gen']:
        rows[a:min(b, T)] = True
    rows[:plen] = False
    rows[0] = False
    pos = np.nonzero(rows)[0]
    return (ids, plen, pos, z['top_ids'][pos - plen], z['top_lp'][pos - plen])

def tool_weights(ids, pos, tw):
    w = np.ones(len(pos), np.float32)
    if tw == 1.0:
        return w
    inside = np.zeros(len(ids), bool)
    st = None
    for i, t in enumerate(ids):
        if t == TC_OPEN:
            st = i
        elif t == TC_CLOSE and st is not None:
            inside[st:i + 1] = True
            st = None
    w[inside[pos]] = tw
    return w

def seq_loss(model, fn, L, chunk, grad, tw, mw):
    ids, plen, pos, tid, tlp = load(fn, L)
    if len(pos) == 0:
        return None
    dev = model.last
    h = model.hidden(torch.as_tensor(ids, dtype=torch.long), ckpt=grad)
    w = torch.as_tensor(tool_weights(ids, pos, tw) * mw, device=dev)
    pos_t = torch.as_tensor(pos, device=dev)
    tid = torch.as_tensor(tid, device=dev)
    tlp = torch.as_tensor(tlp, device=dev)
    tot, kls, ags = (0.0, 0.0, 0.0)
    for a in range(0, len(pos), chunk):
        b = min(len(pos), a + chunk)
        args = (h[pos_t[a:b] - 1], model.lm_head, tid[a:b], tlp[a:b], w[a:b])
        if grad:
            l, k, g = checkpoint(kl21, *args, use_reentrant=False)
        else:
            l, k, g = kl21(*args)
        tot, kls, ags = (tot + l, kls + k, ags + g)
    return (tot, kls, ags, len(pos), float(w.sum()))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(os.environ.get('FN_WORK', 'work'), 'kd', 'teacher'))
    ap.add_argument('--recs', default=os.path.join(os.environ.get('FN_WORK', 'work'), 'student', 'recs.jsonl'))
    ap.add_argument('--build', default=os.path.join(os.environ.get('FN_WORK', 'work'), 'builds', 'k320q34'))
    ap.add_argument('--model', default=M, help='checkpoint dir for non-expert weights (make_dense.py copy is fine)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--gpus', default='0')
    ap.add_argument('--len', type=int, default=32768)
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--accum', type=int, default=1, help='sequences per replica per step')
    ap.add_argument('--trainable', default='scales,router')
    ap.add_argument('--lr-scales', type=float, default=0.0005)
    ap.add_argument('--lr-router', type=float, default=2e-05)
    ap.add_argument('--lr-norms', type=float, default=2e-05)
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--tool-weight', type=float, default=2.0)
    ap.add_argument('--direct-weight', type=float, default=0.5)
    ap.add_argument('--modes', default='repl,direct')
    ap.add_argument('--val-n', type=int, default=16, help='val sequences per mode')
    ap.add_argument('--val-len', type=int, default=16384)
    ap.add_argument('--eval-every', type=int, default=50)
    ap.add_argument('--chunk', type=int, default=1024)
    ap.add_argument('--layers', type=int, default=None)
    ap.add_argument('--resume', default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--eval-only', action='store_true')
    ap.add_argument('--src', default='lowbit', help='lowbit (the build) | fp8 (original experts: fidelity floor)')
    a = ap.parse_args()
    ws, rank = (int(os.environ.get('WORLD_SIZE', '1')), int(os.environ.get('RANK', '0')))
    if ws > 1:
        import datetime
        dist.init_process_group('nccl', device_id=torch.device('cuda:0'), timeout=datetime.timedelta(hours=3))
    out = Path(a.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        json.dump(vars(a), open(out / 'args.json', 'w'), indent=1)
    logf = open(out / 'log.jsonl', 'a') if rank == 0 else None

    def log(d):
        d['t'] = time.time()
        if rank == 0:
            print(json.dumps(d), flush=True)
            logf.write(json.dumps(d) + '\n')
            logf.flush()
    split = {}
    mode = {}
    for l in open(a.recs):
        r = json.loads(l)
        split[r['key']] = r['split']
        mode[r['key']] = r['mode']
    files = sorted(glob.glob(f'{a.data}/*.npz'))
    key = lambda f: Path(f).stem
    tr = [f for f in files if split.get(key(f)) == 'train' and mode[key(f)] in a.modes.split(',')]
    va = {m: [f for f in files if split.get(key(f)) == 'val' and mode[key(f)] == m][:a.val_n] for m in ('repl', 'direct')}
    log({'ev': 'data', 'train': len(tr), 'val': {m: len(v) for m, v in va.items()}, 'ws': ws})
    t0 = time.time()
    gpus = [int(g) for g in a.gpus.split(',')]
    tset = tuple(a.trainable.split(','))
    model = S.Student(a.model, a.build, gpus, trainable=tset if a.src == 'lowbit' else (), layers=a.layers, src=a.src)
    log({'ev': 'built', 's': round(time.time() - t0), 'mem_gb': [round(torch.cuda.memory_allocated(g) / 2 ** 30, 1) for g in gpus]})
    if ws > 1:
        dist.barrier()
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if a.resume:
        st = torch.load(a.resume, map_location='cpu')
        for n, p in params:
            if n in st:
                p.data.copy_(st[n].to(p.dtype))
    groups = {'scales': [], 'router': [], 'norms': []}
    for n, p in params:
        g = 'scales' if n.endswith(('.tgu', '.td')) else 'router' if n.endswith('mlp.gate') else 'norms'
        groups[g].append((n, p))
    masters = {n: p.detach().float().clone() for n, p in params}
    lrs = {'scales': a.lr_scales, 'router': a.lr_router, 'norms': a.lr_norms}
    opt = None if not params else torch.optim.Adam([{'params': [masters[n] for n, _ in groups[g]], 'lr': lrs[g], 'name': g} for g in groups if groups[g]], betas=(0.9, 0.99), eps=1e-10)
    for gr in opt.param_groups if opt else []:
        gr['lr0'] = gr['lr']
    log({'ev': 'params', **{g: sum((p.numel() for _, p in v)) for g, v in groups.items()}})

    def evaluate(step):
        model.eval()
        res = {}
        for m, fl in va.items():
            mine = fl[rank::ws]
            kl = torch.zeros(3, device=model.last, dtype=torch.float64)
            with torch.no_grad():
                for f in mine:
                    r = seq_loss(model, f, a.val_len, a.chunk, False, 1.0, 1.0)
                    if r is None:
                        continue
                    kl += torch.tensor([float(r[1]), float(r[2]), r[3]], device=model.last, dtype=torch.float64)
            if ws > 1:
                dist.all_reduce(kl)
            res[m] = {'kl': round(float(kl[0] / kl[2]), 5), 'top1': round(float(kl[1] / kl[2]), 4), 'n': int(kl[2])}
        log({'ev': 'val', 'step': step, **res})
        model.train()
        return res

    def save(step):
        if rank == 0:
            torch.save({n: p.detach().cpu() for n, p in params}, out / 'trained.pt')
            log({'ev': 'saved', 'step': step})
    evaluate(0)
    if a.eval_only:
        return
    rng = random.Random(a.seed)
    order = []
    step = 0
    tl = time.time()
    while step < a.steps:
        if len(order) < ws * a.accum:
            ep = tr[:]
            rng.shuffle(ep)
            order += ep
        batch = order[:ws * a.accum]
        order = order[ws * a.accum:]
        mine = batch[rank::ws]
        for p in masters.values():
            p.grad = None
        for _, p in params:
            p.grad = None
        stats = torch.zeros(4, device=model.last, dtype=torch.float64)
        for f in mine:
            mw = a.direct_weight if mode[key(f)] == 'direct' else 1.0
            r = seq_loss(model, f, a.len, a.chunk, True, a.tool_weight, mw)
            if r is None:
                continue
            r[0].backward()
            stats += torch.tensor([float(r[1]), float(r[2]), r[3], r[4]], device=model.last, dtype=torch.float64)
        bydev = {}
        for n, p in params:
            bydev.setdefault(p.device, []).append((n, p))
        if ws > 1:
            dist.all_reduce(stats)
        wsum = max(float(stats[3]), 1.0)
        for dev, lst in bydev.items():
            flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).float().flatten() for _, p in lst])
            if ws > 1:
                dist.all_reduce(flat)
            flat /= wsum
            o = 0
            for n, p in lst:
                k = p.numel()
                masters[n].grad = flat[o:o + k].view(p.shape).to(masters[n].device)
                o += k
        step += 1
        fr = min(1.0, step / max(1, a.warmup)) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / a.steps)))
        for gr in opt.param_groups:
            gr['lr'] = gr['lr0'] * max(fr, 0.05 if step < a.steps else fr)
        gn = {g: round(float(torch.sqrt(sum(((masters[n].grad.float() ** 2).sum() for n, _ in v)))), 6) for g, v in groups.items() if v}
        opt.step()
        with torch.no_grad():
            for n, p in params:
                p.data.copy_(masters[n].to(p.device, p.dtype))
        log({'ev': 'step', 'step': step, 'kl': round(float(stats[0] / max(stats[2], 1)), 5), 'top1': round(float(stats[1] / max(stats[2], 1)), 4), 'tok': int(stats[2]), 'gn': gn, 's': round(time.time() - tl, 1), 'mem_gb': round(torch.cuda.max_memory_allocated(model.last) / 2 ** 30, 1)})
        tl = time.time()
        if step % a.eval_every == 0 or step == a.steps:
            save(step)
            evaluate(step)
    if ws > 1:
        dist.destroy_process_group()
if __name__ == '__main__':
    main()
