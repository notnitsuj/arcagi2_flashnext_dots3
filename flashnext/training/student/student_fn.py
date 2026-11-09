import json, math, os, re, struct
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from torch.utils.checkpoint import checkpoint
G = 128
PFX = 'model.language_model.'
torch._dynamo.config.cache_size_limit = 4096
torch._dynamo.config.recompile_limit = 4096
torch._dynamo.config.accumulated_cache_size_limit = 8192
torch._dynamo.config.accumulated_recompile_limit = 8192
_flex = torch.compile(flex_attention, dynamic=False)
PAD = int(os.environ.get('KD_PAD', '2048'))
GROUPED = os.environ.get('KD_GROUPED', '1') == '1'
try:
    if os.environ.get('KD_NO_FLA') == '1':
        raise ImportError
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as _fla_gdr
except Exception:
    _fla_gdr = None

class Ckpt:

    def __init__(self, path):
        self.path = Path(path)
        self.cfg = json.load(open(self.path / 'config.json'))
        self.tcfg = self.cfg.get('text_config', self.cfg)
        idx = json.load(open(self.path / 'model.safetensors.index.json'))['weight_map']
        self.map = idx
        self.h = {}

    def _f(self, k):
        f = self.map[k]
        if f not in self.h:
            self.h[f] = safe_open(str(self.path / f), 'pt')
        return self.h[f]

    def has(self, k):
        return k in self.map

    def raw(self, k):
        return self._f(k).get_tensor(k)

    def w(self, k, dev, dtype=torch.bfloat16):
        t = self.raw(k).to(dev)
        sk = k[:-len('weight')] + 'weight_scale_inv'
        if t.dtype == torch.float8_e4m3fn:
            s = self.raw(sk).to(dev, torch.float32)
            t = t.float() * s.repeat_interleave(128, 0)[:t.shape[0]].repeat_interleave(128, 1)[:, :t.shape[1]]
        return t.to(dtype)

    def memmap(self, k):
        fn = self.path / self.map[k]
        with open(fn, 'rb') as fh:
            n = struct.unpack('<Q', fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        m = hdr[k]
        a, b = m['data_offsets']
        return (np.memmap(fn, dtype=np.uint8, mode='r', offset=8 + n + a, shape=(b - a,)), m['shape'], m['dtype'])

def rms1p(x, w, eps, group=None):
    xf = x.float()
    if group is not None:
        xf = xf.view(*xf.shape[:-1], -1, group)
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    if group is not None:
        xf = xf.flatten(-2)
    return (xf * (1.0 + w.float())).to(x.dtype)

def rope_tab(pos, dim, theta, dev):
    inv = 1.0 / theta ** (torch.arange(0, dim, 2, device=dev, dtype=torch.float32) / dim)
    f = pos.float()[:, None] * inv[None]
    e = torch.cat([f, f], -1)
    return (e.cos().to(torch.bfloat16), e.sin().to(torch.bfloat16))

def rope_part(x, cs):
    c, s = (cs[0][:, None, :], cs[1][:, None, :])
    rd = c.shape[-1]
    xr, xp = (x[..., :rd], x[..., rd:])
    x1, x2 = (xr[..., :rd // 2], xr[..., rd // 2:])
    rot = torch.cat([-x2, x1], -1)
    return torch.cat([xr * c + rot * s, xp], -1)

def _qz(planes, bits, z, KG):
    if bits == 4:
        p4 = planes[0]
        R = p4.shape[0]
        w = p4.view(R, KG, 1, 32).to(torch.int32) & 65535
        j = torch.arange(4, device=p4.device, dtype=torch.int32).view(1, 1, 4, 1)
        q = w >> 4 * j & 15
    else:
        p2, p1 = planes
        R = p2.shape[0]
        w = p2.view(R, KG, 1, 16).to(torch.int32) & 65535
        j = torch.arange(8, device=p2.device, dtype=torch.int32).view(1, 1, 8, 1)
        q = w >> 2 * j & 3
        if p1 is not None:
            h = p1.view(R, KG, 1, 16).to(torch.int32) & 255
            q = q | (h >> j & 1) << 2
    return q.reshape(R, KG, G).float() - z.float()[..., None]

def deq(planes, bits, z, s, dtype=torch.bfloat16):
    qz = _qz(planes, bits, z, s.shape[1])
    return (qz * s.float()[..., None]).reshape(qz.shape[0], -1).to(dtype)

class LowbitLinear(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, s, z, bits, *planes):
        ctx.bits = bits
        ctx.save_for_backward(x, s, z, *[p for p in planes if p is not None])
        ctx.has_p1 = len(planes) > 1 and planes[1] is not None
        return x @ deq(planes, bits, z, s, x.dtype).t()

    @staticmethod
    def backward(ctx, dy):
        x, s, z, *pl = ctx.saved_tensors
        if ctx.bits == 4:
            planes = (pl[0],)
        else:
            planes = (pl[0], pl[1] if ctx.has_p1 else None)
        dx = dy @ deq(planes, ctx.bits, z, s, dy.dtype) if ctx.needs_input_grad[0] else None
        ds = None
        if ctx.needs_input_grad[1]:
            dW = (dy.t() @ x).float()
            ds = (dW.view(*s.shape, G) * _qz(planes, ctx.bits, z, s.shape[1])).sum(-1).to(s.dtype)
        n = 4 + (1 if ctx.bits == 4 else 2)
        return (dx, ds) + (None,) * (n - 2)

def _gmm(a, b, offs):
    return torch._grouped_mm(a, b, offs=offs)
CH = int(os.environ.get('KD_DEQ_CHUNK', '64'))

class Bank:

    def __init__(self, ex, dev):
        self.E = len(ex)
        self.by = {}
        for b in sorted({e['bits'] for e in ex}):
            idx = [i for i, e in enumerate(ex) if e['bits'] == b]
            d = {'idx': torch.tensor(idx, device=dev)}
            for nm in ('gu', 'd'):
                keys = ('p4',) if b == 4 else ('p2', 'p1') if b == 3 else ('p2',)
                d[nm] = tuple((torch.stack([ex[i][nm][k].to(dev) for i in idx]) for k in keys))
                d[nm + '_z'] = torch.stack([ex[i][nm + '_z'].to(dev) for i in idx])
            self.by[b] = d

    def _qz(self, b, nm, sl):
        d = self.by[b]
        pl = tuple((p[sl] for p in d[nm]))
        z = d[nm + '_z'][sl]
        n, R = (z.shape[0], z.shape[1])
        flat = [p.reshape(n * R, -1) for p in pl]
        if b != 4 and len(flat) == 1:
            flat.append(None)
        return _qz(tuple(flat), b, z.reshape(n * R, -1), z.shape[2]).view(n, R, z.shape[2], G)

    def weights(self, nm, s):
        E, R, KG = s.shape
        W = torch.empty(E, R, KG * G, dtype=torch.bfloat16, device=s.device)
        for b, d in self.by.items():
            idx = d['idx']
            for a in range(0, len(idx), CH):
                sl = slice(a, a + CH)
                ii = idx[sl]
                W[ii] = (self._qz(b, nm, sl) * s[ii].float()[..., None]).view(len(ii), R, KG * G).to(torch.bfloat16)
        return W

    def dscale(self, nm, dW):
        E, R, C = dW.shape
        KG = C // G
        ds = torch.empty(E, R, KG, dtype=torch.float32, device=dW.device)
        for b, d in self.by.items():
            idx = d['idx']
            for a in range(0, len(idx), CH):
                sl = slice(a, a + CH)
                ii = idx[sl]
                ds[ii] = (dW[ii].float().view(len(ii), R, KG, G) * self._qz(b, nm, sl)).sum(-1)
        return ds

class GroupedExperts(torch.autograd.Function):

    @staticmethod
    def forward(ctx, xs, s_gu, s_d, offs, bank):
        Wgu = bank.weights('gu', s_gu)
        h = _gmm(xs, Wgu.transpose(1, 2), offs)
        del Wgu
        g, u = h.chunk(2, -1)
        act = F.silu(g) * u
        Wd = bank.weights('d', s_d)
        y = _gmm(act, Wd.transpose(1, 2), offs)
        ctx.bank = bank
        ctx.save_for_backward(xs, h, s_gu, s_d, offs)
        return y

    @staticmethod
    def backward(ctx, dy):
        xs, h, s_gu, s_d, offs = ctx.saved_tensors
        bank = ctx.bank
        dy = dy.contiguous().to(torch.bfloat16)
        g, u = h.chunk(2, -1)
        sg = torch.sigmoid(g.float())
        act = (g.float() * sg * u.float()).to(torch.bfloat16)
        Wd = bank.weights('d', s_d)
        dact = _gmm(dy, Wd, offs)
        del Wd
        ds_d = None
        if ctx.needs_input_grad[2]:
            dWd = _gmm(dy.t().contiguous(), act, offs)
            ds_d = bank.dscale('d', dWd)
            del dWd
        dg = dact.float() * u.float() * (sg * (1 + g.float() * (1 - sg)))
        du = dact.float() * g.float() * sg
        dh = torch.cat([dg, du], -1).to(torch.bfloat16)
        Wgu = bank.weights('gu', s_gu)
        dx = _gmm(dh, Wgu, offs)
        del Wgu
        ds_gu = None
        if ctx.needs_input_grad[1]:
            dWgu = _gmm(dh.t().contiguous(), xs, offs)
            ds_gu = bank.dscale('gu', dWgu)
            del dWgu
        return (dx, ds_gu, ds_d, None, None)

def P(t):
    return nn.Parameter(t, requires_grad=False)

class GatedResidual(nn.Module):

    def __init__(self, ck, p, dev, cfg, combine=True):
        super().__init__()
        self.hc, self.H, self.eps = (cfg['hc_count'], cfg['hidden_size'], cfg['rms_norm_eps'])
        self.norm = P(ck.w(p + 'hc_norm.weight', dev))
        self.down = P(ck.w(p + 'input_mix_weight_down.weight', dev))
        self.up = P(ck.w(p + 'input_mix_weight_up.weight', dev))
        self.inj = P(ck.w(p + 'block_inject_weight.weight', dev)) if combine else None

    def forward(self, h):
        hn = rms1p(h, self.norm, self.eps, self.H)
        w = F.silu(hn @ self.down.t() / self.hc)
        w = torch.sigmoid(w @ self.up.t())
        mixed = (w.unflatten(-1, (self.hc, self.H)) * hn.unflatten(-1, (self.hc, self.H))).mean(-2)
        if self.inj is None:
            return mixed
        return (mixed, 2 * torch.sigmoid(hn @ self.inj.t() / self.hc))

class GDN(nn.Module):

    def __init__(self, ck, p, dev, cfg):
        super().__init__()
        self.nv, self.nk = (cfg['linear_num_value_heads'], cfg['linear_num_key_heads'])
        self.dk, self.dv = (cfg['linear_key_head_dim'], cfg['linear_value_head_dim'])
        self.eps = cfg['rms_norm_eps']
        g = lambda n: P(ck.w(p + n, dev))
        self.qkv, self.z, self.b, self.a = (g('in_proj_qkv.weight'), g('in_proj_z.weight'), g('in_proj_b.weight'), g('in_proj_a.weight'))
        self.conv = g('conv1d.weight')
        self.out = g('out_proj.weight')
        self.norm = g('norm.weight')
        self.A_log = P(ck.raw(p + 'A_log').to(dev, torch.float32))
        self.dt_bias = P(ck.raw(p + 'dt_bias').to(dev, torch.float32))

    def forward(self, x):
        T = x.shape[0]
        qkv = x @ self.qkv.t()
        C = qkv.shape[1]
        k = self.conv.shape[-1]
        qkv = F.conv1d(F.pad(qkv.t()[None], (k - 1, 0)), self.conv, groups=C)[0].t()
        qkv = F.silu(qkv)
        kd, vd = (self.nk * self.dk, self.nv * self.dv)
        q, kk, v = qkv.split([kd, kd, vd], -1)
        q, kk, v = (q.view(T, self.nk, self.dk), kk.view(T, self.nk, self.dk), v.view(T, self.nv, self.dv))
        z = (x @ self.z.t()).view(T, self.nv, self.dv)
        beta = torch.sigmoid(x @ self.b.t())
        gdec = -self.A_log.exp() * F.softplus((x @ self.a.t()).float() + self.dt_bias)
        r = self.nv // self.nk
        q, kk = (q.repeat_interleave(r, 1), kk.repeat_interleave(r, 1))
        if _fla_gdr is not None:
            o, _ = _fla_gdr(q[None], kk[None], v[None], g=gdec[None], beta=beta[None], scale=self.dk ** (-0.5), initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True)
            o = o[0]
        else:
            from transformers.models.qwen4_exp.modeling_qwen4_exp import torch_chunk_gated_delta_rule as tcg
            o, _ = tcg.__wrapped__(q[None], kk[None], v[None], g=gdec[None], beta=beta[None], use_qk_l2norm_in_kernel=True) if hasattr(tcg, '__wrapped__') else tcg(q[None], kk[None], v[None], g=gdec[None], beta=beta[None], use_qk_l2norm_in_kernel=True)
            o = o[0]
        of = o.float()
        of = of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + self.eps)
        o = self.norm * of.to(x.dtype)
        act = torch.sigmoid if _GATE == 'sigmoid' else F.silu
        o = (o.float() * act(z.float())).to(x.dtype)
        return o.reshape(T, -1) @ self.out.t()

class QSA(nn.Module):

    def __init__(self, ck, p, dev, cfg):
        super().__init__()
        self.H, self.KV, self.D = (cfg['num_attention_heads'], cfg['num_key_value_heads'], cfg['head_dim'])
        self.eps = cfg['rms_norm_eps']
        g = lambda n: P(ck.w(p + n, dev))
        self.q, self.k, self.v, self.o = (g('q_proj.weight'), g('k_proj.weight'), g('v_proj.weight'), g('o_proj.weight'))
        self.qn, self.kn = (g('q_norm.weight'), g('k_norm.weight'))
        self.iqk = g('indexer.index_qk_proj.weight')
        self.iqn, self.ikn = (g('indexer.q_layernorm.weight'), g('indexer.k_layernorm.weight'))
        self.iH, self.iD = (cfg['indexer_n_heads'], cfg['indexer_head_dim'])
        self.ratio = cfg['indexer_compress_ratio']
        self.nblk = cfg['indexer_budget'] // self.ratio
        self._sel = None

    @torch.no_grad()
    def select(self, x, ctx):
        T, r = (x.shape[0], self.ratio)
        nb = T // r
        if (T + 1) // r <= self.nblk:
            return None
        key = (ctx['gen'], id(self))
        if self._sel is not None and self._sel[0] == key:
            return self._sel[1]
        qk = x @ self.iqk.t()
        q, kr = qk.split([self.iH * self.iD, self.iD], -1)
        q = rms1p(q.view(T, self.iH, self.iD), self.iqn, self.eps)
        q = rope_part(q, ctx['rope'])
        kb = kr.view(nb, r, self.iD).float().mean(1).to(kr.dtype)
        kb = rms1p(kb, self.ikn, self.eps)
        bstart = torch.arange(nb, device=x.device) * r
        kb = rope_part(kb[:, None], (ctx['rope'][0][bstart], ctx['rope'][1][bstart]))[:, 0]
        SB = torch.zeros(T, nb, dtype=torch.bool, device=x.device)
        ar = torch.arange(T, device=x.device)
        ncomp = (ar + 1) // r
        first = int((ncomp > self.nblk).nonzero()[0, 0])
        SB[:first] = torch.arange(nb, device=x.device)[None, :] < ncomp[:first, None]
        C = 1024
        qf, kf = (q.float(), kb.float())
        bi = torch.arange(nb, device=x.device)
        for i0 in range(first, T, C):
            i1 = min(T, i0 + C)
            s = torch.relu(torch.einsum('qhd,bd->qhb', qf[i0:i1], kf)).sum(1) / math.sqrt(self.iD)
            s.masked_fill_(bi[None, :] >= ncomp[i0:i1, None], float('-inf'))
            idx = s.topk(self.nblk, -1).indices
            SB[i0:i1].scatter_(1, idx, True)
        self._sel = (key, SB)
        return SB

    def block_mask(self, SB, T, dev):
        r = self.ratio
        if SB is None:
            return None
        nt = T // 128
        SBt = SB.view(nt, 128, nt, 128 // r)
        ti = torch.arange(nt, device=dev)
        diag = ti[:, None] == ti[None, :]
        below = ti[:, None] > ti[None, :]
        allb = SBt.all(3).all(1) & below
        part = (SBt.any(3).any(1) & below | diag) & ~allb

        def pack(b):
            return (b.sum(1).to(torch.int32)[None, None], torch.argsort((~b).to(torch.int8), dim=1, stable=True).to(torch.int32)[None, None])

        def mm(b, h, q, k):
            return (k <= q) & (SB[q, k // r] | (k >= (q + 1) // r * r))
        pn, pi = pack(part)
        fn, fi = pack(allb)
        return BlockMask.from_kv_blocks(pn, pi, fn, fi, BLOCK_SIZE=128, mask_mod=mm, seq_lengths=(T, T))

    def forward(self, x, ctx):
        T = x.shape[0]
        SB = self.select(x.detach(), ctx)
        bm = self.block_mask(SB, T, x.device) if SB is not None else ctx['causal']
        qg = (x @ self.q.t()).view(T, self.H, 2 * self.D)
        q, gate = (qg[..., :self.D], qg[..., self.D:])
        q = rms1p(q, self.qn, self.eps)
        k = rms1p((x @ self.k.t()).view(T, self.KV, self.D), self.kn, self.eps)
        v = (x @ self.v.t()).view(T, self.KV, self.D)
        q, k = (rope_part(q, ctx['rope']), rope_part(k, ctx['rope']))
        o = _flex(q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None], block_mask=bm, scale=self.D ** (-0.5), enable_gqa=True)[0].transpose(0, 1)
        o = o.reshape(T, -1) * torch.sigmoid(gate.reshape(T, -1))
        return o @ self.o.t()

class PLE(nn.Module):

    def __init__(self, ck, p, dev, cfg):
        super().__init__()
        self.hc, self.H, self.eps = (cfg['hc_count'], cfg['hidden_size'], cfg['rms_norm_eps'])
        self.n = cfg['ngram_size']
        self.hpn = cfg['heads_per_ngram']
        e = p + 'ple_embedding.'
        self.mult = ck.raw(e + 'layer_multipliers').long()
        self.offs = ck.raw(e + 'ngram_heads_offsets').long()
        self.sizes = ck.raw(e + 'ngram_heads_vocab_sizes').long()
        self.scale = float(ck.raw(e + 'ngram_embedding.weight_scale').float()[0])
        self.shards = []
        i = 0
        shm = os.environ.get('KD_PLE_SHM', '/dev/shm/ple')
        while ck.has(f'{e}ngram_embedding.shard_{i}.weight'):
            mm, shape, dt = ck.memmap(f'{e}ngram_embedding.shard_{i}.weight')
            assert dt == 'F8_E4M3', dt
            f = Path(shm) / f'shard_{i}.u8'
            if (Path(shm) / 'DONE').exists():
                mm = np.memmap(f, dtype=np.uint8, mode='r', shape=(shape[0] * shape[1],))
            self.shards.append(mm.reshape(shape))
            i += 1
        self.rows = self.shards[0].shape[0]
        self.dim = self.shards[0].shape[1]
        self.eos = cfg['eos_token_id'][0] if isinstance(cfg['eos_token_id'], list) else cfg['eos_token_id']
        g = lambda n: P(ck.w(p + n, dev))
        self.kp, self.vp = (g('key_proj.weight'), g('value_proj.weight'))
        self.nk, self.nq, self.nc = (g('norm_key.weight'), g('norm_query.weight'), g('norm_conv.weight'))
        self.conv = g('conv1d.weight')
        self.dil = cfg['ngram_size']

    def ngram_ids(self, ids):
        ctxlen = self.n - 1
        hist = torch.cat([torch.full((ctxlen,), self.eos, dtype=torch.long), ids])
        L = hist.shape[0]
        pos = torch.arange(L)
        eosp = torch.where(hist == self.eos, pos, -1)
        prev_incl = torch.cummax(eosp, 0).values
        prev = torch.cat([torch.tensor([-1]), prev_incl[:-1]])
        pis = pos - (prev + 1)
        sh = []
        for s in range(self.n):
            src = pos - s
            v = hist[src.clamp_min(0)]
            ok = (pis >= s) & (src >= 0)
            sh.append(torch.where(ok, v, torch.full_like(v, self.eos)))
        blocks = []
        for ng in range(2, self.n + 1):
            a, b = ((ng - 2) * self.hpn, (ng - 1) * self.hpn)
            mixed = sh[0] * self.mult[0]
            for q in range(1, ng):
                mixed = torch.bitwise_xor(mixed, sh[q] * self.mult[q])
            blocks.append(torch.remainder(mixed[:, None], self.sizes[a:b][None]) + self.offs[a:b][None])
        return torch.cat(blocks, -1)[ctxlen:]

    def embed(self, ids):
        rows = self.ngram_ids(ids.cpu().long()).numpy()
        sh, rr = (rows // self.rows, rows % self.rows)
        out = np.empty(rows.shape + (self.dim,), np.uint8)
        for s in np.unique(sh):
            m = sh == s
            out[m] = self.shards[int(s)][rr[m]]
        t = torch.from_numpy(out).view(torch.float8_e4m3fn)
        return t

    def forward(self, h, emb8):
        T = h.shape[0]
        emb = (emb8.to(h.device).float() * self.scale).to(torch.bfloat16).reshape(T, -1)
        key = rms1p(emb @ self.kp.t(), self.nk, self.eps, self.H).unflatten(-1, (self.hc, self.H))
        val = emb @ self.vp.t()
        qn = rms1p(h, self.nq, self.eps, self.H).unflatten(-1, (self.hc, self.H))
        gate = (key * qn).sum(-1, keepdim=True) / math.sqrt(self.H)
        gate = gate.abs().clamp_min(1e-06).sqrt() * gate.sign()
        gv = torch.sigmoid(gate) * val[:, None, :]
        gvn = rms1p(gv.flatten(-2), self.nc, self.eps, self.H)
        gv = gv.flatten(-2)
        pad = (self.conv.shape[-1] - 1) * self.dil
        c = F.conv1d(F.pad(gvn.t()[None], (pad, 0)), self.conv, groups=gvn.shape[1], dilation=self.dil)[0].t()
        return gv + F.silu(c)

class MoE(nn.Module):

    def __init__(self, ck, p, dev, cfg, build, l, src='lowbit'):
        super().__init__()
        self.k, self.l, self.src = (cfg['num_experts_per_tok'], l, src)
        g = lambda n: P(ck.w(p + n, dev))
        self.sgu = P(torch.cat([ck.w(p + 'shared_expert.gate_proj.weight', dev), ck.w(p + 'shared_expert.up_proj.weight', dev)]))
        self.sd = g('shared_expert.down_proj.weight')
        self.sgate = g('shared_expert_gate.weight')
        if src == 'lowbit':
            r = torch.load(Path(build) / f'router-L{l:02d}.pt', map_location='cpu', weights_only=False)
            self.gate = P(r['weight'].to(dev, torch.bfloat16))
            d = torch.load(Path(build) / f'lowbit-L{l:02d}.pt', map_location='cpu', weights_only=False)
            self.kept = list(d['experts'])
            ex = [d[e] for e in self.kept]
            self.E = len(ex)
            self.bits = [e['bits'] for e in ex]
            self.pk = []
            for e in ex if not GROUPED or os.environ.get('KD_BOTH') == '1' else []:
                pl = {}
                for nm in ('gu', 'd'):
                    if e['bits'] == 4:
                        pl[nm] = (e[nm]['p4'].to(dev),)
                    else:
                        pl[nm] = (e[nm]['p2'].to(dev), e[nm]['p1'].to(dev) if e['bits'] == 3 else None)
                self.pk.append(pl)
            self.bank = Bank(ex, dev) if GROUPED else None
            self.zgu = [e['gu_z'].to(dev) for e in ex]
            self.zd = [e['d_z'].to(dev) for e in ex]
            self.register_buffer('sgu0', torch.stack([e['gu_s'] for e in ex]).to(dev, torch.float32))
            self.register_buffer('sd0', torch.stack([e['d_s'] for e in ex]).to(dev, torch.float32))
            self.tgu = nn.Parameter(torch.zeros_like(self.sgu0), requires_grad=False)
            self.td = nn.Parameter(torch.zeros_like(self.sd0), requires_grad=False)
        else:
            self.gate = g('gate.weight')
            self.E = self.gate.shape[0]
            self.kept = list(range(self.E))
            self.fp8 = []
            for e in range(self.E):
                q = f'{p}experts.{e}.'
                ws = [(ck.raw(q + n + '.weight').to(dev), ck.raw(q + n + '.weight_scale_inv').to(dev, torch.float32)) for n in ('gate_proj', 'up_proj', 'down_proj')]
                self.fp8.append(ws)

    def scales(self, i):
        return (self.sgu0[i] * self.tgu[i].exp(), self.sd0[i] * self.td[i].exp())

    def expert(self, xe, i):
        if self.src == 'lowbit':
            sgu, sd = self.scales(i)
            b = self.bits[i]
            h = LowbitLinear.apply(xe, sgu, self.zgu[i], b, *self.pk[i]['gu'])
            g_, u = h.chunk(2, -1)
            return LowbitLinear.apply(F.silu(g_) * u, sd, self.zd[i], b, *self.pk[i]['d'])

        def dq(ws):
            t, s = ws
            return (t.float() * s.repeat_interleave(128, 0)[:t.shape[0]].repeat_interleave(128, 1)[:, :t.shape[1]]).to(torch.bfloat16)
        gw, uw, dw = (dq(w) for w in self.fp8[i])
        return F.silu(xe @ gw.t()) * (xe @ uw.t()) @ dw.t()

    def forward(self, x):
        T = x.shape[0]
        logits = (x @ self.gate.t()).float()
        pr = torch.softmax(logits, -1)
        w, ids = torch.topk(pr, self.k, -1)
        w = (w / w.sum(-1, keepdim=True)).to(x.dtype)
        flat = ids.flatten()
        order = torch.argsort(flat, stable=True)
        cnt = torch.bincount(flat, minlength=self.E).tolist()
        tok = order // self.k
        wsort = w.flatten()[order]
        if self.src == 'lowbit' and GROUPED and (not getattr(self, 'force_loop', False)):
            c = torch.tensor(cnt, device=x.device)
            cp = (c + 7) // 8 * 8
            st, stp = (torch.cumsum(c, 0) - c, torch.cumsum(cp, 0) - cp)
            es = flat[order]
            dest = stp[es] + torch.arange(es.shape[0], device=x.device) - st[es]
            Np = int(cp.sum())
            xs = torch.zeros(Np, x.shape[1], dtype=x.dtype, device=x.device).index_copy(0, dest, x[tok])
            offs = torch.cumsum(cp, 0).to(torch.int32)
            sgu = self.sgu0 * self.tgu.exp()
            sd = self.sd0 * self.td.exp()
            y = GroupedExperts.apply(xs, sgu, sd, offs, self.bank)[dest]
            y = y.float() * wsort[:, None].float()
            out = torch.zeros(T, x.shape[1], dtype=torch.float32, device=x.device).index_add_(0, tok, y)
            g_, u = (x @ self.sgu.t()).chunk(2, -1)
            sh = torch.sigmoid(x @ self.sgate.t()) * (F.silu(g_) * u @ self.sd.t())
            return (out + sh.float()).to(x.dtype)
        nz = [n for n in cnt if n]
        xs = torch.split(x[tok], nz)
        ys = []
        j = 0
        for e, n in enumerate(cnt):
            if n == 0:
                continue
            ys.append(self.expert(xs[j], e))
            j += 1
        y = torch.cat(ys, 0).float() * wsort[:, None].float()
        out = torch.zeros(T, x.shape[1], dtype=torch.float32, device=x.device).index_add_(0, tok, y)
        g_, u = (x @ self.sgu.t()).chunk(2, -1)
        sh = F.silu(g_) * u @ self.sd.t()
        sh = torch.sigmoid(x @ self.sgate.t()) * sh
        return (out + sh.float()).to(x.dtype)

class Layer(nn.Module):

    def __init__(self, ck, l, dev, cfg, build, src):
        super().__init__()
        p = f'{PFX}layers.{l}.'
        self.l = l
        self.kind = cfg['layer_types'][l]
        self.ple = PLE(ck, p + 'ple.', dev, cfg) if ck.has(p + 'ple.key_proj.weight') else None
        self.ahc = GatedResidual(ck, p + 'attn_hyper_connection.', dev, cfg)
        self.mhc = GatedResidual(ck, p + 'mlp_hyper_connection.', dev, cfg)
        self.mix = GDN(ck, p + 'linear_attn.', dev, cfg) if self.kind == 'linear_attention' else QSA(ck, p + 'self_attn.', dev, cfg)
        self.mlp = MoE(ck, p + 'mlp.', dev, cfg, build, l, src)

    def forward(self, h, ctx):
        if self.ple is not None:
            h = h + self.ple(h, ctx['ple'])
        x, inj = self.ahc(h)
        y = self.mix(x) if self.kind == 'linear_attention' else self.mix(x, ctx)
        h = h + (y.unsqueeze(-2) * inj.unsqueeze(-1)).flatten(-2)
        x, inj = self.mhc(h)
        y = self.mlp(x)
        return h + (y.unsqueeze(-2) * inj.unsqueeze(-1)).flatten(-2)
_GEN = [0]

class Student(nn.Module):

    def __init__(self, model_dir, build, gpus, trainable=('scales',), layers=None, split=None, src='lowbit'):
        super().__init__()
        ck = Ckpt(model_dir)
        cfg = ck.tcfg
        self.cfg = cfg
        gt = cfg.get('output_gate_type')
        assert gt in (None, 'sigmoid', 'silu'), gt
        global _GATE
        _GATE = gt
        n = cfg['num_hidden_layers'] if layers is None else layers
        self.devs = [torch.device(f'cuda:{g}') for g in gpus]
        if split is None:
            per = -(-n // len(gpus))
            split = [min(i // per, len(gpus) - 1) for i in range(n)]
        self.split = split
        self.embed = P(ck.raw(PFX + 'embed_tokens.weight').to(self.devs[0], torch.bfloat16))
        self.layers = nn.ModuleList()
        for l in range(n):
            self.layers.append(Layer(ck, l, self.devs[split[l]], cfg, build, src))
            torch.cuda.empty_cache()
        self.last = self.devs[split[n - 1]]
        self.mixer = GatedResidual(ck, PFX + 'hyper_connection_mixer.', self.last, cfg, combine=False)
        self.lm_head = P(ck.w('lm_head.weight', self.last))
        self.theta = cfg['rope_parameters']['rope_theta'] if 'rope_parameters' in cfg else cfg['rope_theta']
        prf = cfg.get('rope_parameters', {}).get('partial_rotary_factor', cfg.get('partial_rotary_factor', 1.0))
        self.rd = int(cfg['head_dim'] * prf)
        self.set_trainable(trainable)

    def set_trainable(self, trainable):
        for nm, p in self.named_parameters():
            on = 'scales' in trainable and (nm.endswith('.tgu') or nm.endswith('.td')) or ('router' in trainable and nm.endswith('mlp.gate')) or ('norms' in trainable and re.search('(\\.norm|\\.qn|\\.kn|\\.nk|\\.nq|\\.nc)$', nm) is not None)
            p.requires_grad_(on)

    def _ctx(self, T, dev, ple_emb):
        pos = torch.arange(T, device=dev)
        return {'rope': rope_tab(pos, self.rd, self.theta, dev), 'gen': _GEN[0], 'ple': ple_emb, 'causal': self._causal(T, dev)}

    def _causal(self, T, dev):
        key = ('c', T, dev)
        c = getattr(self, '_cc', {})
        self._cc = c
        if key not in c:
            from torch.nn.attention.flex_attention import create_block_mask
            c[key] = create_block_mask(lambda b, h, q, k: q >= k, None, None, T, T, device=dev, _compile=True)
        return c[key]

    def hidden(self, ids, ckpt=True):
        T0 = ids.shape[0]
        _GEN[0] += 1
        Tp = -(-T0 // PAD) * PAD
        ids = F.pad(ids, (0, Tp - T0))
        ple = None
        for layer in self.layers:
            if layer.ple is not None:
                ple = layer.ple.embed(ids)
        h = F.embedding(ids.to(self.devs[0]), self.embed).repeat(1, self.cfg['hc_count'])
        ctxs = {}
        for layer in self.layers:
            dev = layer.ahc.norm.device
            h = h.to(dev)
            if dev not in ctxs:
                ctxs[dev] = self._ctx(Tp, dev, ple)
            if ckpt and torch.is_grad_enabled():
                h = checkpoint(layer, h, ctxs[dev], use_reentrant=False)
            else:
                h = layer(h, ctxs[dev])
        h = h[:T0].to(self.last)
        return self.mixer(h)

    def export(self, out_dir, src_dir):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for layer in self.layers:
            m, l = (layer.mlp, layer.l)
            d = torch.load(Path(src_dir) / f'lowbit-L{l:02d}.pt', map_location='cpu', weights_only=False)
            for i, e in enumerate(d['experts']):
                sgu, sd = m.scales(i)
                d[e]['gu_s'] = sgu.detach().half().cpu()
                d[e]['d_s'] = sd.detach().half().cpu()
            torch.save(d, out_dir / f'lowbit-L{l:02d}.pt')
            r = torch.load(Path(src_dir) / f'router-L{l:02d}.pt', map_location='cpu', weights_only=False)
            r['weight'] = m.gate.detach().to(torch.bfloat16).cpu()
            torch.save(r, out_dir / f'router-L{l:02d}.pt')
_GATE = 'sigmoid'
