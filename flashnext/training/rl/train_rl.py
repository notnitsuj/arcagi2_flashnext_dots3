import argparse, glob, json, math, os, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'student'))
import student_fn as S
M = os.environ.get('FN_BASE_MODEL', 'models/Qwen3.8-Flash-Next-FP8')

def lp_chunk(h, lm_head, y):
    ls = torch.log_softmax((h @ lm_head.t()).float(), -1)
    lp = ls.gather(1, y[:, None].long())[:, 0]
    ent = -(ls.exp() * ls).sum(1)
    return (lp, ent.detach())

def seq_pg(model, fn, chunk, grad, A_clip=None, old=None, eps=0.2, keep=None):
    z = np.load(fn)
    ids, pos, A = (z['ids'], z['pos'], float(z['adv']))
    if A_clip is not None:
        A = max(-A_clip, min(A_clip, A))
    dev = model.last
    h = model.hidden(torch.as_tensor(ids, dtype=torch.long), ckpt=grad)
    pos_t = torch.as_tensor(pos, device=dev).long()
    y = torch.as_tensor(ids, device=dev)[pos_t]
    old_t = torch.as_tensor(old, device=dev, dtype=torch.float32) if old is not None else None
    tot, slp, sent, nclip = (0.0, 0.0, 0.0, 0)
    for a in range(0, len(pos), chunk):
        b = min(len(pos), a + chunk)
        args = (h[pos_t[a:b] - 1], model.lm_head, y[a:b])
        lp, ent = checkpoint(lp_chunk, *args, use_reentrant=False) if grad else lp_chunk(*args)
        if keep is not None:
            keep.append(lp.detach().float().cpu().numpy())
        if old_t is None:
            tot = tot + A * lp.sum()
        else:
            r = torch.exp(lp - old_t[a:b])
            surr = torch.minimum(r * A, torch.clamp(r, 1 - eps, 1 + eps) * A)
            nclip += int(((r < 1 - eps) | (r > 1 + eps)).sum())
            tot = tot + surr.sum()
        slp += float(lp.detach().sum())
        sent += float(ent.sum())
    return (tot, A, slp, sent, len(pos), nclip)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='prep_rl.py dir (comma list allowed)')
    ap.add_argument('--build', default=os.path.join(os.environ.get('FN_WORK', 'work'), 'builds', 'k320q34'))
    ap.add_argument('--model', default=M)
    ap.add_argument('--out', required=True)
    ap.add_argument('--gpus', default='0')
    ap.add_argument('--steps', type=int, default=0, help='0 = one pass over the data (--epochs)')
    ap.add_argument('--epochs', type=float, default=1.0)
    ap.add_argument('--per-step', type=int, default=48, help='samples per optimizer step (all replicas)')
    ap.add_argument('--tok-norm', type=float, default=16384.0)
    ap.add_argument('--trainable', default='scales,router')
    ap.add_argument('--lr-scales', type=float, default=0.0003)
    ap.add_argument('--lr-router', type=float, default=1.5e-05)
    ap.add_argument('--lr-norms', type=float, default=1e-05)
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--kl-anchor', type=float, default=0.0)
    ap.add_argument('--clip-adv', type=float, default=2.0)
    ap.add_argument('--ppo-clip', type=float, default=0.0, help="> 0: PPO clipped surrogate vs the round's start policy (old log-probs computed once at start)")
    ap.add_argument('--chunk', type=int, default=1024)
    ap.add_argument('--resume', default=None, help='trained.pt of the previous round (= the rollout policy)')
    ap.add_argument('--anchor-from', default=None, help='trained.pt that defines theta0 for --kl-anchor (default: the build as loaded, i.e. before --resume)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-files', type=int, default=0)
    ap.add_argument('--layers', type=int, default=None, help='first N layers only (smoke tests)')
    ap.add_argument('--dry', action='store_true', help='forward/backward only, no optimizer step (memory/speed test)')
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
    files = []
    for d in a.data.split(','):
        files += sorted(glob.glob(f'{d}/*.npz'))
    rng = random.Random(a.seed)
    rng.shuffle(files)
    if a.max_files:
        files = files[:a.max_files]
    lens = {f: os.path.getsize(f) for f in files}
    nsteps = a.steps or max(1, int(len(files) * a.epochs) // a.per_step)
    log({'ev': 'data', 'files': len(files), 'steps': nsteps, 'ws': ws})
    t0 = time.time()
    gpus = [int(g) for g in a.gpus.split(',')]
    model = S.Student(a.model, a.build, gpus, trainable=tuple(a.trainable.split(',')), layers=a.layers)
    log({'ev': 'built', 's': round(time.time() - t0), 'mem_gb': [round(torch.cuda.memory_allocated(g) / 2 ** 30, 1) for g in gpus]})
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    dev0 = torch.device(f'cuda:{gpus[0]}')

    def all_reduce(t):
        if t.device == dev0:
            dist.all_reduce(t)
        else:
            t0 = t.to(dev0)
            dist.all_reduce(t0)
            t.copy_(t0.to(t.device))
    anchor = None
    if a.kl_anchor > 0:
        src = torch.load(a.anchor_from, map_location='cpu') if a.anchor_from else None
        anchor = {n: (src[n].float() if src is not None and n in src else p.detach().float().clone()).to(p.device) for n, p in params}
    if a.resume:
        st = torch.load(a.resume, map_location='cpu')
        k = 0
        for n, p in params:
            if n in st:
                p.data.copy_(st[n].to(p.dtype))
                k += 1
        log({'ev': 'resumed', 'from': a.resume, 'tensors': k})
    if ws > 1:
        dist.barrier()
    groups = {'scales': [], 'router': [], 'norms': []}
    for n, p in params:
        g = 'scales' if n.endswith(('.tgu', '.td')) else 'router' if n.endswith('mlp.gate') else 'norms'
        groups[g].append((n, p))
    masters = {n: p.detach().float().clone() for n, p in params}
    lrs = {'scales': a.lr_scales, 'router': a.lr_router, 'norms': a.lr_norms}
    opt = torch.optim.Adam([{'params': [masters[n] for n, _ in groups[g]], 'lr': lrs[g], 'name': g} for g in groups if groups[g]], betas=(0.9, 0.99), eps=1e-10)
    for gr in opt.param_groups:
        gr['lr0'] = gr['lr']
    log({'ev': 'params', **{g: sum((p.numel() for _, p in v)) for g, v in groups.items()}})

    def save(step):
        if rank == 0:
            torch.save({n: p.detach().cpu() for n, p in params}, out / 'trained.pt')
            if step % 10 == 0 or step == nsteps:
                torch.save({n: p.detach().cpu() for n, p in params}, out / f'trained_s{step}.pt')
            log({'ev': 'saved', 'step': step})
    lpdir = out / 'lpold'

    def oldp(f):
        return str(lpdir / (Path(f).stem + '.npy'))
    if a.ppo_clip > 0:
        if rank == 0:
            lpdir.mkdir(parents=True, exist_ok=True)
        if ws > 1:
            dist.barrier()
        model.eval()
        t0 = time.time()
        for f in files[rank::ws]:
            if not os.path.exists(oldp(f)):
                keep = []
                with torch.no_grad():
                    seq_pg(model, f, a.chunk, False, keep=keep)
                np.save(oldp(f), np.concatenate(keep).astype(np.float32))
        if ws > 1:
            dist.barrier()
        model.train()
        log({'ev': 'lpold', 'files': len(files), 's': round(time.time() - t0)})
    order = []
    step = 0
    tl = time.time()
    model.train()
    while step < nsteps:
        if len(order) < a.per_step:
            ep = files[:]
            rng.shuffle(ep)
            order += ep
        batch = order[:a.per_step]
        order = order[a.per_step:]
        batch.sort(key=lambda f: -lens[f])
        mine = batch[rank::ws]
        for p in masters.values():
            p.grad = None
        for _, p in params:
            p.grad = None
        stats = torch.zeros(9, device=model.last, dtype=torch.float64)
        for f in mine:
            old = np.load(oldp(f)) if a.ppo_clip > 0 else None
            obj, A, slp, sent, n, nclip = seq_pg(model, f, a.chunk, True, A_clip=a.clip_adv, old=old, eps=a.ppo_clip)
            (-obj / a.tok_norm).backward()
            s = [1, n, slp if A > 0 else 0, n if A > 0 else 0, slp if A < 0 else 0, n if A < 0 else 0, sent, abs(A), nclip]
            stats += torch.tensor(s, device=model.last, dtype=torch.float64)
        bydev = {}
        for n, p in params:
            bydev.setdefault(p.device, []).append((n, p))
        if ws > 1:
            all_reduce(stats)
        nseq = max(float(stats[0]), 1.0)
        for dev, lst in bydev.items():
            flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).float().flatten() for _, p in lst])
            if ws > 1:
                all_reduce(flat)
            flat /= nseq
            o = 0
            for n, p in lst:
                k = p.numel()
                g = flat[o:o + k].view(p.shape).to(masters[n].device)
                if anchor is not None:
                    g = g + a.kl_anchor * (masters[n] - anchor[n].to(masters[n].device))
                masters[n].grad = g
                o += k
        step += 1
        fr = min(1.0, step / max(1, a.warmup)) * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - 1) / nsteps)))
        for gr in opt.param_groups:
            gr['lr'] = gr['lr0'] * fr
        gn = {g: round(math.sqrt(sum((float((masters[n].grad.float() ** 2).sum()) for n, _ in v))), 6) for g, v in groups.items() if v}
        if not a.dry:
            opt.step()
            with torch.no_grad():
                for n, p in params:
                    p.data.copy_(masters[n].to(p.device, p.dtype))
        dn = {g: round(math.sqrt(sum((float(((masters[n] - anchor[n].to(masters[n].device)) ** 2).sum()) for n, _ in v))), 5) for g, v in groups.items() if v} if anchor else {}
        log({'ev': 'step', 'step': step, 'nseq': int(stats[0]), 'tok': int(stats[1]), 'lp_pos': round(float(stats[2] / max(stats[3], 1)), 4), 'lp_neg': round(float(stats[4] / max(stats[5], 1)), 4), 'ent': round(float(stats[6] / max(stats[1], 1)), 4), 'clip_frac': round(float(stats[8] / max(stats[1], 1)), 4), 'absA': round(float(stats[7] / nseq), 3), 'gn': gn, 'dist': dn, 'lr_s': round(opt.param_groups[0]['lr'], 7), 's': round(time.time() - tl, 1), 'mem_gb': [round(torch.cuda.max_memory_allocated(g) / 2 ** 30, 1) for g in gpus]})
        tl = time.time()
        if not a.dry and (step % 5 == 0 or step == nsteps):
            save(step)
    if ws > 1:
        dist.destroy_process_group()
if __name__ == '__main__':
    main()
