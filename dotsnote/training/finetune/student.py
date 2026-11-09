import json, math, os, re
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention
from torch.utils.checkpoint import checkpoint
G = 128
torch._dynamo.config.cache_size_limit = 4096
torch._dynamo.config.recompile_limit = 4096
torch._dynamo.config.accumulated_cache_size_limit = 8192
torch._dynamo.config.accumulated_recompile_limit = 8192
torch._dynamo.config.fail_on_recompile_limit_hit = True
_flex = torch.compile(flex_attention, dynamic=False)
PAD = 2048

class Ckpt:

    def __init__(self, path):
        self.path = Path(path)
        self.cfg = json.load(open(self.path / 'config.json'))
        self.h = {}
        self.map = {}
        for f in sorted(self.path.glob('model-*.safetensors')):
            self.h[f.name] = safe_open(str(f), 'pt')
            for k in self.h[f.name].keys():
                self.map[k] = f.name

    def raw(self, k):
        f = self.map[k]
        if f not in self.h:
            self.h[f] = safe_open(str(self.path / f), 'pt')
        return self.h[f].get_tensor(k)

    def has(self, k):
        return k in self.map

    def w(self, k, dev, dtype=torch.bfloat16):
        t = self.raw(k).to(dev)
        sk = k[:-len('weight')] + 'weight_scale_inv'
        if t.dtype == torch.float8_e4m3fn and sk in self.map:
            s = self.raw(sk).to(dev, torch.float32)
            bo = bi = 128
            t = t.float() * s.repeat_interleave(bo, 0)[:t.shape[0]].repeat_interleave(bi, 1)[:, :t.shape[1]]
        return t.to(dtype)

def codes_from_planar(p2, p1, KG):
    R = p2.shape[0]
    w = p2.view(R, KG, 1, 16).to(torch.int32) & 65535
    j = torch.arange(8, device=p2.device).view(1, 1, 8, 1)
    q = w >> 2 * j & 3
    if p1 is not None:
        h = p1.view(R, KG, 1, 16).to(torch.int32) & 255
        q = q | (h >> j & 1) << 2
    return q.reshape(R, KG * 128).to(torch.uint8)

def rms(x, w, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(x.dtype)

def rope_tab(pos, dim, theta, dev):
    inv = 1.0 / theta ** (torch.arange(0, dim, 2, device=dev, dtype=torch.float32) / dim)
    f = pos.float()[:, None] * inv[None]
    return (f.cos(), f.sin())

def rope_inter(x, cs):
    c, s = (cs[0][:, None, :], cs[1][:, None, :])
    xf = x.float()
    x1, x2 = (xf[..., 0::2], xf[..., 1::2])
    return torch.stack([x1 * c - x2 * s, x2 * c + x1 * s], -1).flatten(-2).to(x.dtype)

def rope_neox(x, cs):
    c, s = (cs[0][:, None, :], cs[1][:, None, :])
    xf = x.float()
    h = xf.shape[-1] // 2
    x1, x2 = (xf[..., :h], xf[..., h:])
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], -1).to(x.dtype)

class _QAct(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        sh = x.shape
        xg = x.float().reshape(-1, sh[-1] // G, G)
        sc = (xg.abs().amax(-1, keepdim=True) / 448.0).clamp_min(1e-12)
        return ((xg / sc).to(torch.float8_e4m3fn).float() * sc).reshape(sh).to(x.dtype)

    @staticmethod
    def backward(ctx, g):
        return g
ACT_FP8 = os.environ.get('DISTILL_ACT_FP8', '1') == '1'
NESTED_CKPT = os.environ.get('DISTILL_NESTED_CKPT', '0') == '1'
NESTED_MIN = int(os.environ.get('DISTILL_NESTED_MIN', '0'))
_nest = [False]
KV_FP8 = os.environ.get('DISTILL_KV_FP8', '1') == '1'

def lin(x, W):
    if ACT_FP8 and x.shape[-1] % G == 0:
        x = _QAct.apply(x)
    return x @ W.t()

def kvq(t):
    return (t.to(torch.float8_e4m3fn).to(t.dtype) - t).detach() + t if KV_FP8 else t

class LowbitLinear(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, s, codes, z):
        W = ((codes.float().view(*s.shape, G) - z.float()[..., None]) * s.float()[..., None]).view(codes.shape)
        ctx.save_for_backward(x, s, codes, z)
        return x @ W.to(x.dtype).t()

    @staticmethod
    def backward(ctx, dy):
        x, s, codes, z = ctx.saved_tensors
        qz = codes.float().view(*s.shape, G) - z.float()[..., None]
        W = (qz * s.float()[..., None]).view(codes.shape).to(dy.dtype)
        dx = dy @ W if ctx.needs_input_grad[0] else None
        ds = None
        if ctx.needs_input_grad[1]:
            dW = (dy.t() @ x).float().view(*s.shape, G)
            ds = (dW * qz).sum(-1).to(s.dtype)
        return (dx, ds, None, None)

def _qz(p2, p1, z, KG):
    R = p2.shape[0]
    w = p2.view(R, KG, 1, 16).to(torch.int32) & 65535
    j = torch.arange(8, device=p2.device, dtype=torch.int32).view(1, 1, 8, 1)
    q = w >> 2 * j & 3
    if p1 is not None:
        h = p1.view(R, KG, 1, 16).to(torch.int32) & 255
        q = q | (h >> j & 1) << 2
    return q.reshape(R, KG, G).float() - z.float()[..., None]

@torch.compile(dynamic=False)
def _deq_planar(p2, p1, z, s, dtype: torch.dtype):
    qz = _qz(p2, p1, z, s.shape[1])
    return (qz * s.float()[..., None]).reshape(p2.shape[0], -1).to(dtype)

@torch.compile(dynamic=False)
def _ds_planar(dW, p2, p1, z):
    qz = _qz(p2, p1, z, z.shape[1])
    return (dW.float().view(qz.shape) * qz).sum(-1)

class PackedLowbitLinear(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, s, p2, p1, z):
        ctx.save_for_backward(x, s, p2, p1, z)
        return x @ _deq_planar(p2, p1, z, s, x.dtype).t()

    @staticmethod
    def backward(ctx, dy):
        x, s, p2, p1, z = ctx.saved_tensors
        dx = dy @ _deq_planar(p2, p1, z, s, dy.dtype) if ctx.needs_input_grad[0] else None
        ds = None
        if ctx.needs_input_grad[1]:
            ds = _ds_planar(dy.t() @ x, p2, p1, z).to(s.dtype)
        return (dx, ds, None, None, None)
PACKED = os.environ.get('DISTILL_PACKED', '1') == '1'

def token_block_mask(M):
    T = M.shape[0]
    nb = T // 128
    Mb = M.view(nb, 128, nb, 128)
    anyb = Mb.any(3).any(1)
    allb = Mb.all(3).all(1)
    part = anyb & ~allb

    def pack(b):
        return (b.sum(1).to(torch.int32)[None, None], torch.argsort((~b).to(torch.int8), dim=1, stable=True).to(torch.int32)[None, None])
    pn, pi = pack(part)
    fn, fi = pack(allb)
    return BlockMask.from_kv_blocks(pn, pi, fn, fi, BLOCK_SIZE=128, mask_mod=lambda b, h, qi, ki: M[qi, ki], seq_lengths=(T, T))

class Attn(nn.Module):

    def __init__(self, ck, l, dev, cfg):
        super().__init__()
        self.swa = cfg['layer_types'][l] == 'sliding_attention'
        p = f'model.layers.{l}.self_attn.'
        g = lambda n: nn.Parameter(ck.w(p + n, dev), requires_grad=False)
        if self.swa:
            self.H, self.dn, self.dr, self.dv = (cfg['swa_num_attention_heads'], cfg['swa_qk_nope_head_dim'], cfg['swa_qk_rope_head_dim'], cfg['swa_v_head_dim'])
            self.qr, self.kvr, self.theta = (cfg['swa_q_lora_rank'], cfg['swa_kv_lora_rank'], cfg['swa_rope_theta'])
        else:
            self.H, self.dn, self.dr, self.dv = (cfg['num_attention_heads'], cfg['qk_nope_head_dim'], cfg['qk_rope_head_dim'], cfg['v_head_dim'])
            self.qr, self.kvr, self.theta = (cfg['q_lora_rank'], cfg['kv_lora_rank'], cfg['rope_theta'])
        self.eps = cfg['rms_norm_eps']
        self.qs = (cfg['hidden_size'] / self.qr) ** 0.5 if cfg['apply_mla_qkv_lora_rescale'] else 1.0
        self.kvs = (cfg['hidden_size'] / self.kvr) ** 0.5 if cfg['apply_mla_qkv_lora_rescale'] else 1.0
        self.q_a, self.kv_a, self.q_b, self.kv_b = (g('q_a_proj.weight'), g('kv_a_proj_with_mqa.weight'), g('q_b_proj.weight'), g('kv_b_proj.weight'))
        self.o, self.gp = (g('o_proj.weight'), g('g_proj.weight'))
        self.q_ln, self.kv_ln, self.kr_ln = (g('q_a_layernorm.weight'), g('kv_a_layernorm.weight'), g('k_rope_only_layernorm.weight'))
        self.window = cfg['sliding_window_size']
        self.topk = cfg['index_topk']
        if not self.swa:
            self.iH, self.iD = (cfg['index_n_heads'], cfg['index_head_dim'])
            self.wq_b, self.wk, self.wproj = (g('indexer.wq_b.weight'), g('indexer.wk.weight'), g('indexer.weights_proj.weight'))
            self.k_norm_w, self.k_norm_b = (g('indexer.k_norm.weight'), g('indexer.k_norm.bias'))
            self.full_theta = cfg['rope_theta']

    @torch.no_grad()
    def dsa_mask(self, x, qc, pos):
        T = x.shape[0]
        if T <= self.topk or os.environ.get('DISTILL_DENSE') == '1':
            return None
        q = (qc @ self.wq_b.t()).view(T, self.iH, self.iD)
        k = F.layer_norm((x @ self.wk.t()).float(), (self.iD,), self.k_norm_w.float(), self.k_norm_b.float(), 1e-06)
        cs = rope_tab(pos, 64, self.full_theta, x.device)
        q = torch.cat([rope_inter(q[..., :64], cs), q[..., 64:]], -1)
        k = torch.cat([rope_inter(k[:, None, :64], cs)[:, 0], k[:, 64:]], -1).to(q.dtype)
        w = (x @ self.wproj.t()).float() * self.iD ** (-0.5) * self.iH ** (-0.5)
        M = torch.zeros(T, T, dtype=torch.bool, device=x.device)
        C = 256 if T <= 16384 else 64
        ar = torch.arange(T, device=x.device)
        for i0 in range(self.topk, T, C):
            i1 = min(T, i0 + C)
            s = torch.relu(torch.einsum('qhd,kd->qhk', q[i0:i1], k[:i1]).float())
            lg = torch.einsum('qhk,qh->qk', s, w[i0:i1])
            lg.masked_fill_(ar[None, :i1] > ar[i0:i1, None], float('-inf'))
            idx = lg.topk(self.topk, -1).indices
            M[i0:i1].scatter_(1, idx, True)
        M[:self.topk] = ar[None, :] <= ar[:self.topk, None]
        return M

    def forward(self, x, pos, ctx):
        T = x.shape[0]
        qkv = lin(x, torch.cat([self.q_a, self.kv_a]))
        qc, kva = qkv.split([self.qr, self.kvr + self.dr], -1)
        qc = rms(qc, self.q_ln, self.eps) * self.qs
        kvc, kpe = kva.split([self.kvr, self.dr], -1)
        kvc = rms(kvc, self.kv_ln, self.eps) * self.kvs
        kpe = rms(kpe, self.kr_ln, self.eps)
        kvc = kvq(kvc)
        q = lin(qc, self.q_b).view(T, self.H, self.dn + self.dr)
        cs = ctx['rope'][self.theta]
        rp = rope_neox if os.environ.get('DISTILL_ROPE_NEOX') == '1' else rope_inter
        q = torch.cat([q[..., :self.dn], rp(q[..., self.dn:], cs)], -1)
        kpe = kvq(rp(kpe[:, None, :], cs))
        kv = lin(kvc, self.kv_b).view(T, self.H, self.dn + self.dv)
        k = torch.cat([kv[..., :self.dn], kpe.expand(T, self.H, self.dr)], -1)
        v = kv[..., self.dn:]
        if self.swa:
            bm = ctx['swa_mask']
        else:
            M = self.dsa_mask(x, qc.detach(), pos)
            if M is None:
                bm = ctx['causal_mask']
            else:
                bm = token_block_mask(M) if os.environ.get('DISTILL_OLD_BLOCKMASK') != '1' else create_block_mask(lambda b, h, qi, ki: M[qi, ki], None, None, T, T, device=x.device)
        o = _flex(q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None], block_mask=bm, scale=(self.dn + self.dr) ** (-0.5))[0].transpose(0, 1)
        gate = torch.sigmoid(lin(x, self.gp).float()).to(o.dtype)
        o = (o * gate[..., None]).reshape(T, self.H * self.dv)
        return lin(o, self.o)

class MLP(nn.Module):

    def __init__(self, ck, p, dev):
        super().__init__()
        self.gu = nn.Parameter(torch.cat([ck.w(p + 'gate_proj.weight', dev), ck.w(p + 'up_proj.weight', dev)]), requires_grad=False)
        self.d = nn.Parameter(ck.w(p + 'down_proj.weight', dev), requires_grad=False)

    def forward(self, x):
        g, u = lin(x, self.gu).chunk(2, -1)
        return lin(F.silu(g) * u, self.d)

class MoE(nn.Module):

    def __init__(self, ck, l, dev, cfg, lowbit_dir):
        super().__init__()
        p = f'model.layers.{l}.mlp.'
        self.gate = nn.Parameter(ck.w(p + 'gate.weight', dev), requires_grad=False)
        self.register_buffer('bias', ck.raw(p + 'gate.e_score_correction_bias').to(dev, torch.float32))
        self.shared = MLP(ck, p + 'shared_experts.', dev)
        self.k, self.I = (cfg['num_experts_per_tok'], cfg['moe_intermediate_size'])
        d = torch.load(Path(lowbit_dir) / f'lowbit-L{l:02d}.pt', map_location='cpu', weights_only=False)
        self.kept = d['experts']
        ex = [d[e] for e in self.kept]
        self.E = len(ex)
        self.bits = [e['bits'] for e in ex]
        if PACKED:
            self.pk = {nm: [(e[nm]['p2'].to(dev), e[nm]['p1'].to(dev) if e['bits'] == 3 else None) for e in ex] for nm in ('gu', 'd')}
        else:
            cg, cd = ([], [])
            for e in ex:
                for nm, lst in (('gu', cg), ('d', cd)):
                    pl = e[nm]
                    lst.append(codes_from_planar(pl['p2'].to(dev), pl['p1'].to(dev) if e['bits'] == 3 else None, e[nm + '_s'].shape[1]))
            self.register_buffer('cgu', torch.stack(cg))
            self.register_buffer('cd', torch.stack(cd))
        self.register_buffer('zgu', torch.stack([e['gu_z'] for e in ex]).to(dev))
        self.register_buffer('zd', torch.stack([e['d_z'] for e in ex]).to(dev))
        self.register_buffer('sgu0', torch.stack([e['gu_s'] for e in ex]).to(dev, torch.float32))
        self.register_buffer('sd0', torch.stack([e['d_s'] for e in ex]).to(dev, torch.float32))
        self.tgu = nn.Parameter(torch.zeros_like(self.sgu0), requires_grad=False)
        self.td = nn.Parameter(torch.zeros_like(self.sd0), requires_grad=False)

    def scales(self, e):
        return (self.sgu0[e] * self.tgu[e].exp(), self.sd0[e] * self.td[e].exp())

    def expert(self, xe, e):
        sgu, sd = self.scales(e)
        if PACKED:
            h = PackedLowbitLinear.apply(xe, sgu, *self.pk['gu'][e], self.zgu[e])
            g, u = h.chunk(2, -1)
            return PackedLowbitLinear.apply(F.silu(g) * u, sd, *self.pk['d'][e], self.zd[e])
        h = LowbitLinear.apply(xe, sgu, self.cgu[e], self.zgu[e])
        g, u = h.chunk(2, -1)
        return LowbitLinear.apply(F.silu(g) * u, sd, self.cd[e], self.zd[e])

    def forward(self, x):
        T = x.shape[0]
        logits = x.float() @ self.gate.float().t()
        sc = torch.sigmoid(logits)
        ids = torch.topk(sc + self.bias, self.k, -1).indices
        w = sc.gather(1, ids)
        w = w / w.sum(-1, keepdim=True)
        flat = ids.flatten()
        order = torch.argsort(flat, stable=True)
        cnt = torch.bincount(flat, minlength=self.E).tolist()
        tok = order // self.k
        wsort = w.flatten()[order]
        out = torch.zeros(T, x.shape[1], dtype=torch.float32, device=x.device)
        o = 0
        for e, n in enumerate(cnt):
            if n == 0:
                continue
            t = tok[o:o + n]
            xe = x[t]
            if _nest[0] and torch.is_grad_enabled():
                y = checkpoint(self.expert, xe, e, use_reentrant=False)
            else:
                y = self.expert(xe, e)
            out = out.index_add(0, t, y.float() * wsort[o:o + n, None])
            o += n
        return (out + self.shared(x).float()).to(x.dtype)

class Layer(nn.Module):

    def __init__(self, ck, l, dev, cfg, lowbit_dir):
        super().__init__()
        p = f'model.layers.{l}.'
        self.l, self.eps = (l, cfg['rms_norm_eps'])
        self.ln1 = nn.Parameter(ck.w(p + 'input_layernorm.weight', dev), requires_grad=False)
        self.ln2 = nn.Parameter(ck.w(p + 'post_attention_layernorm.weight', dev), requires_grad=False)
        self.attn = Attn(ck, l, dev, cfg)
        self.mlp = MoE(ck, l, dev, cfg, lowbit_dir) if l >= cfg['first_k_dense_replace'] else MLP(ck, p + 'mlp.', dev)

    def _a(self, h, pos, ctx):
        return self.attn(rms(h, self.ln1, self.eps), pos, ctx)

    def _m(self, h):
        return self.mlp(rms(h, self.ln2, self.eps))

    def forward(self, h, pos, ctx):
        if _nest[0] and torch.is_grad_enabled():
            h = h + checkpoint(self._a, h, pos, ctx, use_reentrant=False)
            return h + checkpoint(self._m, h, use_reentrant=False)
        h = h + self._a(h, pos, ctx)
        return h + self._m(h)

class Student(nn.Module):

    def __init__(self, ckpt, gpus, trainable=('scales',), layers=None, split=None):
        super().__init__()
        ck = Ckpt(ckpt)
        cfg = ck.cfg
        self.cfg = cfg
        n = cfg['num_hidden_layers'] if layers is None else layers
        lowbit_dir = Path(ckpt) / cfg.get('arc_lowbit', 'lowbit')
        self.devs = [torch.device(f'cuda:{g}') for g in gpus]
        if split is None:
            per = -(-n // len(gpus))
            split = [min(i // per, len(gpus) - 1) for i in range(n)]
        self.split = split
        self.embed = nn.Parameter(ck.raw('model.embed_tokens.weight').to(self.devs[0]), requires_grad=False)
        self.layers = nn.ModuleList()
        for l in range(n):
            self.layers.append(Layer(ck, l, self.devs[split[l]], cfg, lowbit_dir))
            torch.cuda.empty_cache()
        last = self.devs[split[n - 1]]
        self.norm = nn.Parameter(ck.w('model.norm.weight', last), requires_grad=False)
        self.lm_head = nn.Parameter(ck.raw('lm_head.weight').to(last), requires_grad=False)
        self.last = last
        self.set_trainable(trainable)

    def set_trainable(self, trainable):
        for nm, p in self.named_parameters():
            on = 'scales' in trainable and (nm.endswith('.tgu') or nm.endswith('.td')) or ('norms' in trainable and re.search('(ln1|ln2|q_ln|kv_ln|kr_ln)$|^norm$', nm) is not None) or ('router' in trainable and nm.endswith('mlp.gate'))
            p.requires_grad_(on)

    def _ctx(self, T, dev, cache):
        key = (T, dev)
        if key not in cache:
            pos = torch.arange(T, device=dev)
            w = self.cfg['sliding_window_size']
            cache[key] = {'pos': pos, 'rope': {th: rope_tab(pos, 64, th, dev) for th in (self.cfg['rope_theta'], self.cfg['swa_rope_theta'])}, 'swa_mask': create_block_mask(lambda b, h, q, k: (q >= k) & (q - k < w), None, None, T, T, device=dev, _compile=True), 'causal_mask': create_block_mask(lambda b, h, q, k: q >= k, None, None, T, T, device=dev, _compile=True)}
        return cache[key]

    def hidden(self, ids, ckpt=True):
        cache = getattr(self, '_mcache', {})
        self._mcache = cache
        T0 = ids.shape[0]
        _nest[0] = NESTED_CKPT and T0 >= NESTED_MIN
        Tp = -(-T0 // PAD) * PAD
        ids = F.pad(ids, (0, Tp - T0))
        h = F.embedding(ids.to(self.devs[0]), self.embed)
        for layer in self.layers:
            dev = layer.ln1.device
            h = h.to(dev)
            ctx = self._ctx(ids.shape[0], dev, cache)
            if ckpt and torch.is_grad_enabled():
                h = checkpoint(layer, h, ctx['pos'], ctx, use_reentrant=False)
            else:
                h = layer(h, ctx['pos'], ctx)
        h = h[:T0].to(self.last)
        return rms(h, self.norm, self.cfg['rms_norm_eps'])

    def logits(self, h):
        return (h @ self.lm_head.t()).float()

    def export_scales(self, out_dir, src_dir):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for layer in self.layers:
            if not isinstance(layer.mlp, MoE):
                continue
            l = layer.l
            d = torch.load(Path(src_dir) / f'lowbit-L{l:02d}.pt', map_location='cpu', weights_only=False)
            for i, e in enumerate(d['experts']):
                sgu, sd = layer.mlp.scales(i)
                d[e]['gu_s'] = sgu.detach().half().cpu()
                d[e]['d_s'] = sd.detach().half().cpu()
            torch.save(d, out_dir / f'lowbit-L{l:02d}.pt')
