import os, sys, tempfile
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
FAIL = []

def check(name, cond, info=''):
    print(('PASS ' if cond else 'FAIL ') + name + (f'  {info}' if info else ''))
    if not cond:
        FAIL.append(name)
from accept import draft_probs_at, target_probs
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch
g = torch.Generator().manual_seed(0)
V = 1000
for scale in (0.5, 2.0, 6.0):
    lg = torch.randn(64, V, generator=g, dtype=torch.float64) * scale
    ref = apply_top_k_top_p_pytorch(lg.clone(), torch.full((64,), 20), torch.full((64,), 0.95)).softmax(-1)
    v, ix = lg.topk(20, -1)
    p = target_probs(ix, v - lg.logsumexp(-1, keepdim=True), 0.95).double()
    mine = torch.zeros_like(ref).scatter_(1, ix, p)
    check(f'target_probs == vLLM top-k20/top-p0.95 (scale {scale})', torch.allclose(mine, ref, atol=1e-05), f'max err {(mine - ref).abs().max():.2e}')
lg = torch.randn(8, V, generator=g) * 3
hot = torch.rand(V, generator=g) < 0.7
ids = torch.stack([torch.randperm(V, generator=g)[:20] for _ in range(8)])
q, am = draft_probs_at(lg, ids, hot)
full = lg.masked_fill(~hot, float('-inf'))
kth = full.topk(20, -1).values[:, -1:]
qf = full.masked_fill(full < kth, float('-inf')).softmax(-1)
check('draft q (hot + top20) at ids', torch.allclose(q, qf.gather(1, ids), atol=1e-06))
check('draft argmax inside hot vocab', bool(hot[am].all()))
p = torch.rand(8, 20, generator=g)
p /= p.sum(-1, keepdim=True)
pf = torch.zeros(8, V).scatter_(1, ids, p)
check('alpha = 1 - TV', torch.allclose(torch.minimum(p, q).sum(-1), 1 - 0.5 * (pf - qf).abs().sum(-1), atol=1e-05))
from mtp import fnmtp
from safetensors.torch import save_file
ovr = {'mtp.fc_hidden.weight': torch.ones(4, 4, dtype=torch.bfloat16)}
stream = [('mtp.fc_hidden.weight', torch.zeros(4, 4, dtype=torch.bfloat16)), ('model.language_model.layers.0.mlp.gate.weight', torch.zeros(2, 4, dtype=torch.bfloat16))]
seen = set()
out = list(fnmtp.ovr_stream(iter(stream), ovr, seen))
check('ovr replaces the named tensor only', torch.equal(out[0][1], ovr['mtp.fc_hidden.weight']) and out[1][1] is stream[1][1] and (seen == {'mtp.fc_hidden.weight'}))
try:
    list(fnmtp.ovr_stream(iter([('mtp.fc_hidden.weight', torch.zeros(4, 5, dtype=torch.bfloat16))]), ovr, set()))
    check('ovr shape mismatch refused', False)
except AssertionError:
    check('ovr shape mismatch refused', True)
with tempfile.TemporaryDirectory() as td:
    save_file({'mtp.layers.0.mlp.experts.3.down_proj.weight': torch.zeros(2, 2)}, f'{td}/bad.safetensors')
    try:
        fnmtp.load_ovr(f'{td}/bad.safetensors')
        check('ovr refuses expert tensors', False)
    except AssertionError:
        check('ovr refuses expert tensors', True)
    save_file(ovr, f'{td}/ok.safetensors')
    check('ovr loads mtp dense', set(fnmtp.load_ovr(f'{td}/ok.safetensors')) == set(ovr))
for k in [k for k in os.environ if k.startswith('FN_MTP_')]:
    del os.environ[k]
check('FN_MTP_OVR unset -> flags off', not fnmtp.flags_on())
import importlib.util
_sp = importlib.util.spec_from_file_location('mtpft_dump', os.path.join(HERE, 'dump.py'))
_dm = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(_dm)
assemble = _dm.assemble
T = 10
x = np.arange(T * 3).reshape(T, 3)
chunks = [(np.array([4, 5, 6, 7, -1]), x[[4, 5, 6, 7, 0]]), (np.array([0, 1, 2, 3]), x[:4]), (np.array([8, 9, 10, 11]), np.concatenate([x[8:], x[:2]]))]
(o,), seen = assemble(chunks, T)
check('assemble covers each position once', (seen == 1).all() and np.array_equal(o, x))
import train as TR
Vh = 300
Wl = torch.randn(Vh, 16, generator=g)
smp = torch.randn(8, 16, generator=g)
pidx = torch.stack([torch.randperm(Vh, generator=g)[:20] for _ in range(8)])
pidx[0, 5] = -1
pp = torch.rand(8, 20, generator=g)
pp /= pp.sum(-1, keepdim=True)
kl, al = TR.kl_alpha_chunk(None, smp, Wl, pidx, pp)
qq, _ = draft_probs_at(smp @ Wl.t(), pidx.clamp(min=0))
ref = torch.minimum(torch.where(pidx >= 0, pp, 0 * pp), torch.where(pidx >= 0, qq, 0 * qq)).sum(-1)
check('elen alpha == accept alpha', torch.allclose(al, ref, atol=1e-05) and bool((kl > -1e-06).all()))
bw = TR.pos_weights(10, TR.parse_bins('4:1,8:2,100:3'), torch.tensor([0, 1] * 5), 2.0, 'cpu')
check('ctx-bin x prose weights', bw.tolist() == [2, 1, 2, 1, 2 * 2, 2, 4, 2, 6, 3])
print('ALL PASS' if not FAIL else f'FAILED: {FAIL}')
sys.exit(1 if FAIL else 0)
