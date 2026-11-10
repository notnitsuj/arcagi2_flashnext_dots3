import argparse, glob, json, math, os, random, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
sys.path.insert(0, str(Path(__file__).resolve().parent))
import student as S
TEMP = 1.0

def kl21(h, lm_head, tid, tlp):
    ls = torch.log_softmax((h @ lm_head.t()).float() / TEMP, -1)
    valid = tid >= 0
    lt = torch.where(valid, tlp, torch.full_like(tlp, -1000000000.0))
    if TEMP != 1.0:
        lt = torch.log_softmax(lt / TEMP, -1)
    pt = lt.exp()
    lsk = ls.gather(1, tid.clamp_min(0))
    ps = torch.where(valid, lsk.exp(), torch.zeros_like(lsk))
    tail_t = (1 - pt.sum(1)).clamp_min(1e-08)
    tail_s = (1 - ps.sum(1)).clamp_min(1e-08)
    kl = (pt * (lt - lsk)).masked_fill(~valid, 0).sum(1) + tail_t * (tail_t.log() - tail_s.log())
    agree = (ls.argmax(1) == tid[:, 0]).float()
    return (kl.sum(), agree.sum())

def ce_chunk(h, lm_head, tgt):
    lg = (h @ lm_head.t()).float()
    ce = torch.nn.functional.cross_entropy(lg, tgt, reduction='sum')
    return (ce, (lg.argmax(1) == tgt).float().sum())

def seq_ce(model, ids, plen, chunk=1024, grad=True):
    dev = model.last
    T = len(ids)
    h = model.hidden(torch.as_tensor(ids, dtype=torch.long), ckpt=grad)
    tgt = torch.as_tensor(ids, device=dev).long()
    tot, agr = (0.0, 0.0)
    for a in range(plen, T, chunk):
        b = min(T, a + chunk)
        args = (h[a - 1:b - 1], model.lm_head, tgt[a:b])
        if grad:
            ce, ag = checkpoint(ce_chunk, *args, use_reentrant=False)
        else:
            ce, ag = ce_chunk(*args)
        tot = tot + ce
        agr = agr + ag
    return (tot, agr, max(0, T - plen))

def seq_loss(model, ids, plen, tid, tlp, chunk=1024, grad=True):
    dev = model.last
    T = len(ids)
    h = model.hidden(torch.as_tensor(ids, dtype=torch.long), ckpt=grad)
    tid = torch.as_tensor(tid, device=dev).long()
    tlp = torch.as_tensor(tlp, device=dev).float()
    tot, agr = (0.0, 0.0)
    for a in range(plen, T, chunk):
        b = min(T, a + chunk)
        args = (h[a - 1:b - 1], model.lm_head, tid[a:b], tlp[a:b])
        if grad:
            kl, ag = checkpoint(kl21, *args, use_reentrant=False)
        else:
            kl, ag = kl21(*args)
        tot = tot + kl
        agr = agr + ag
    return (tot, agr, max(0, T - plen))

def load_dir(d, tasks=None, exclude=None, maxlen=None):
    out = []
    for f in sorted(glob.glob(f'{d}/*.npz')):
        t = Path(f).name.split('_')[0]
        if tasks is not None and t not in tasks or (exclude is not None and t in exclude):
            continue
        z = np.load(f)
        if 'top_ids' not in z.files:
            if maxlen is not None and len(z['ids']) > maxlen:
                print('skip (too long)', f, flush=True)
                continue
            out.append(dict(name=Path(f).stem, ids=z['ids'], plen=int(z['plen']), w=float(z['w']) if 'w' in z.files else 1.0))
            continue
        L = len(z['ids']) if maxlen is None else min(maxlen, len(z['ids']))
        out.append(dict(name=Path(f).stem, ids=z['ids'][:L], plen=int(z['plen']), tid=z['top_ids'][:L], tlp=z['top_lp'][:L]))
    return out

def load_fid(ref, maxlen):
    R = np.load(ref)
    out = []
    for k in R.files:
        if k.endswith('_top_ids'):
            t = k[:-len('_top_ids')]
            L = min(maxlen, len(R[f'{t}_ids']))
            out.append(dict(name='fid_' + t, ids=R[f'{t}_ids'][:L], plen=int(R[f'{t}_plen']), tid=R[f'{t}_top_ids'][:L], tlp=R[f'{t}_top_lp'][:L]))
    return out
LOSS = 'kd'

def loss_of(model, r, grad=True):
    if LOSS == 'ce':
        return seq_ce(model, r['ids'], r['plen'], grad=grad)
    return seq_loss(model, r['ids'], r['plen'], r['tid'], r['tlp'], grad=grad)

@torch.no_grad()
def evaluate(model, data):
    kl = ag = n = 0.0
    for r in data:
        a, b, c = loss_of(model, r, grad=False)
        kl += float(a)
        ag += float(b)
        n += c
    return (kl / max(n, 1), ag / max(n, 1), int(n))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=os.environ.get('D3_LB144R', os.path.join(os.environ.get('D3_WORK', 'work'), 'lb144r')))
    ap.add_argument('--data', nargs='+', default=[os.path.join(os.environ.get('D3_WORK', 'work'), 'ft', 'sft')])
    ap.add_argument('--val-tasks', default='', help='file with task ids to hold out for the validation curve')
    ap.add_argument('--fid', default=None, help='kd: reference npz of eval/fidelity.py dump (FP8 preview) for a measurement-only KL on the fidelity traces')
    ap.add_argument('--fid-len', type=int, default=8192)
    ap.add_argument('--gpus', default='0,1,2,3,4')
    ap.add_argument('--len', type=int, default=8192)
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--accum', type=int, default=1)
    ap.add_argument('--lr', type=float, default=0.0005)
    ap.add_argument('--lr-norm', type=float, default=0.0001)
    ap.add_argument('--lr-router', type=float, default=2e-05)
    ap.add_argument('--trainable', default='scales')
    ap.add_argument('--eval-every', type=int, default=50)
    ap.add_argument('--max-val', type=int, default=24)
    ap.add_argument('--out', required=True)
    ap.add_argument('--resume', default=None)
    ap.add_argument('--export', action='store_true', help='write OUT/lowbit (serving layout) at the end')
    ap.add_argument('--eval-only', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--temp', type=float, default=1.0, help='kd: KL at this temperature (both sides); evals stay at T=1')
    ap.add_argument('--layers', type=int, default=None, help='debug: first N layers only')
    ap.add_argument('--loss', choices=['kd', 'ce'], default='ce', help='ce: masked cross-entropy on the completion tokens (SFT npz: ids, plen, w); kd: 21-bin KL to the teacher top-20 log-probabilities (distill/teacher_dump.py npz)')
    ap.add_argument('--epochs', type=float, default=None, help='steps = epochs * n_train / accum (overrides --steps)')
    ap.add_argument('--val-n', type=int, default=0, help='ce: hold out the sequences of this many tasks (seeded)')
    ap.add_argument('--fp32-master', type=int, default=None, help='fp32 masters for trainable bf16 tensors (default: 1 with --loss ce, 0 with --loss kd)')
    a = ap.parse_args()
    a.loss_name = 'ce' if a.loss == 'ce' else 'kl'
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / 'log.jsonl', 'a')
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    global TEMP, LOSS
    LOSS = a.loss
    val_tasks = set(open(a.val_tasks).read().split()) if a.val_tasks else set()
    if a.loss == 'ce' and a.val_n:
        allt = sorted({Path(f).name.split('_')[0] for d in a.data for f in glob.glob(f'{d}/*.npz')} - val_tasks)
        val_tasks |= set(random.Random(a.seed + 1).sample(allt, a.val_n))
        (out / 'val_tasks.txt').write_text('\n'.join(sorted(val_tasks)) + '\n')
    train = sum((load_dir(d, exclude=val_tasks, maxlen=a.len) for d in a.data), [])
    val = sum((load_dir(d, tasks=val_tasks, maxlen=a.len) for d in a.data), [])[:a.max_val]
    if a.epochs is not None:
        a.steps = max(1, round(a.epochs * len(train) / a.accum))
        print(f'steps = {a.steps} ({a.epochs} epochs, accum {a.accum})', flush=True)
    fid = load_fid(a.fid, a.fid_len) if a.fid else []
    print(f"train {len(train)} seqs / {sum((len(r['ids']) - r['plen'] for r in train))} completion tok, val {len(val)} seqs, fid {len(fid)}", flush=True)
    t0 = time.time()
    model = S.Student(a.ckpt, [int(g) for g in a.gpus.split(',')], trainable=tuple(a.trainable.split(',')), layers=a.layers)
    print(f'model loaded in {time.time() - t0:.0f}s; ' + ', '.join((f'cuda:{i} {torch.cuda.memory_allocated(i) / 2 ** 30:.1f}G' for i in range(torch.cuda.device_count()))), flush=True)
    if a.fp32_master if a.fp32_master is not None else a.loss == 'ce':
        for n_, p in model.named_parameters():
            if p.requires_grad and p.dtype != torch.float32:
                p.data = p.data.float()
    params = [p for p in model.parameters() if p.requires_grad]
    if a.resume:
        sd = torch.load(a.resume, map_location='cpu')
        for n_, p in model.named_parameters():
            if n_ in sd:
                p.data.copy_(sd[n_].to(p.device))

    def kind(n_):
        return 'scale' if n_.endswith('.tgu') or n_.endswith('.td') else 'router' if n_.endswith('mlp.gate') else 'norm'
    lrs = {'scale': a.lr, 'router': a.lr_router, 'norm': a.lr_norm}
    groups = [{'params': [p for n_, p in model.named_parameters() if p.requires_grad and kind(n_) == k], 'lr': lrs[k]} for k in ('scale', 'norm', 'router')]
    groups = [g for g in groups if g['params']]
    print('trainable params:', sum((p.numel() for p in params)), flush=True)
    opt = torch.optim.Adam(groups, betas=(0.9, 0.99), eps=1e-12) if params else None
    base_lrs = [g['lr'] for g in groups]

    def do_eval(step):
        global TEMP
        t = time.time()
        tr_temp, TEMP = (TEMP, 1.0)
        rec = {'step': step}
        if val:
            rec[f'val_{a.loss_name}'], rec['val_top1'], rec['val_n'] = evaluate(model, val)
        if fid:
            rec['fid_kl'], rec['fid_top1'], rec['fid_n'] = evaluate(model, fid)
        rec['eval_s'] = round(time.time() - t)
        print('EVAL', json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + '\n')
        log.flush()
        TEMP = tr_temp
    do_eval(0)
    TEMP = a.temp
    if a.eval_only:
        return
    order = []
    step = 0
    tl = tn = 0.0
    while step < a.steps:
        if not order:
            order = list(range(len(train)))
            random.shuffle(order)
        ts = time.time()
        frac = step / max(1, a.steps)
        warm = min(1.0, (step + 1) / 10)
        for g, lr0 in zip(groups, base_lrs):
            g['lr'] = lr0 * warm * 0.5 * (1 + math.cos(math.pi * frac))
        ntok = 0
        for _ in range(a.accum):
            if not order:
                order = list(range(len(train)))
                random.shuffle(order)
            r = train[order.pop()]
            if len(r['ids']) <= r['plen']:
                continue
            kl, ag, n = loss_of(model, r)
            (kl * r.get('w', 1.0) / 4096.0).backward()
            tl += float(kl.detach())
            tn += n
            ntok += len(r['ids'])
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        rec = {'step': step, f'train_{a.loss_name}': tl / max(tn, 1), 'tok': ntok, 's': round(time.time() - ts, 1), 'lr': groups[0]['lr'], 'peak_g': [round(torch.cuda.max_memory_allocated(d) / 2 ** 30, 1) for d in model.devs]}
        tl = tn = 0.0
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + '\n')
        log.flush()
        if step % a.eval_every == 0 or step == a.steps:
            do_eval(step)
            torch.save({n_: p.detach().cpu() for n_, p in model.named_parameters() if p.requires_grad}, out / 'trained.pt')
    if a.export:
        model.export_scales(out / 'lowbit', Path(a.ckpt) / model.cfg.get('arc_lowbit', 'lowbit'))
        print('exported', out / 'lowbit', flush=True)
if __name__ == '__main__':
    main()
