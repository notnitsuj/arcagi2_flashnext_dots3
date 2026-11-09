import math, os, sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'student'))
import student_fn as S
MP = 'mtp.'
TRAINABLE = ('dense', 'router', 'scales', 'norms')

class MTP(nn.Module):

    def __init__(self, model_dir, build, dev, layer_idx=48):
        super().__init__()
        ck = S.Ckpt(model_dir)
        cfg = ck.tcfg
        self.cfg, self.dev = (cfg, dev)
        self.hc, self.H, self.eps = (cfg['hc_count'], cfg['hidden_size'], cfg['rms_norm_eps'])
        g = lambda n: S.P(ck.w(n, dev))
        self.embed = S.P(ck.raw(S.PFX + 'embed_tokens.weight').to(dev, torch.bfloat16))
        self.lm_head = S.P(ck.w('lm_head.weight', dev))
        self.n_emb = g(MP + 'pre_fc_norm_embedding.weight')
        self.n_hid = g(MP + 'pre_fc_norm_hidden.weight')
        self.fc_e = g(MP + 'fc_embedding.weight')
        self.fc_h = g(MP + 'fc_hidden.weight')
        p = MP + 'layers.0.'
        self.ahc = S.GatedResidual(ck, p + 'attn_hyper_connection.', dev, cfg)
        self.mhc = S.GatedResidual(ck, p + 'mlp_hyper_connection.', dev, cfg)
        self.attn = S.QSA(ck, p + 'self_attn.', dev, cfg)
        self.moe = S.MoE(ck, p + 'mlp.', dev, cfg, build, layer_idx, 'lowbit')
        self.mixer = S.GatedResidual(ck, MP + 'hyper_connection_mixer.', dev, cfg, combine=False)
        self.theta = cfg['rope_parameters']['rope_theta']
        self.rd = int(cfg['head_dim'] * cfg['rope_parameters']['partial_rotary_factor'])
        self._cm = {}
        self.kv_fp8 = os.environ.get('MTP_KV_FP8', '0') == '1'
        self.ckpt = True
        self.set_trainable(())

    def named_groups(self):
        p = MP + 'layers.0.'
        d = {'n_emb': MP + 'pre_fc_norm_embedding.weight', 'n_hid': MP + 'pre_fc_norm_hidden.weight', 'fc_e': MP + 'fc_embedding.weight', 'fc_h': MP + 'fc_hidden.weight'}
        for m, pre in (('ahc', p + 'attn_hyper_connection.'), ('mhc', p + 'mlp_hyper_connection.'), ('mixer', MP + 'hyper_connection_mixer.')):
            d[m + '.norm'] = pre + 'hc_norm.weight'
            d[m + '.down'] = pre + 'input_mix_weight_down.weight'
            d[m + '.up'] = pre + 'input_mix_weight_up.weight'
            if m != 'mixer':
                d[m + '.inj'] = pre + 'block_inject_weight.weight'
        a = p + 'self_attn.'
        d.update({'attn.q': a + 'q_proj.weight', 'attn.k': a + 'k_proj.weight', 'attn.v': a + 'v_proj.weight', 'attn.o': a + 'o_proj.weight', 'attn.qn': a + 'q_norm.weight', 'attn.kn': a + 'k_norm.weight', 'attn.iqk': a + 'indexer.index_qk_proj.weight', 'attn.iqn': a + 'indexer.q_layernorm.weight', 'attn.ikn': a + 'indexer.k_layernorm.weight'})
        m = p + 'mlp.'
        d.update({'moe.sgu': '<shared gate|up>', 'moe.sd': m + 'shared_expert.down_proj.weight', 'moe.sgate': m + 'shared_expert_gate.weight', 'moe.gate': '<router-L48>', 'moe.tgu': '<scales>', 'moe.td': '<scales>'})
        return d

    def set_trainable(self, groups):
        norms = {'n_emb', 'n_hid', 'ahc.norm', 'mhc.norm', 'mixer.norm', 'attn.qn', 'attn.kn', 'attn.iqn', 'attn.ikn'}
        for nm, prm in self.named_parameters():
            if nm in ('embed', 'lm_head'):
                on = False
            elif nm in ('moe.tgu', 'moe.td'):
                on = 'scales' in groups
            elif nm == 'moe.gate':
                on = 'router' in groups
            elif nm in norms:
                on = 'norms' in groups
            elif nm in ('attn.iqk',):
                on = 'indexer' in groups
            else:
                on = 'dense' in groups
            prm.requires_grad_(on)

    def ctx(self, T):
        if T not in self._cm:
            pos = torch.arange(T, device=self.dev)
            self._cm = {T: {'rope': S.rope_tab(pos, self.rd, self.theta, self.dev), 'gen': -1, 'causal': S.Student._causal(self, T, self.dev)}}
        return self._cm[T]

    def inp(self, hid, tok):
        T = hid.shape[0]
        e = S.rms1p(F.embedding(tok, self.embed), self.n_emb, self.eps) @ self.fc_e.t()
        h = S.rms1p(hid, self.n_hid, self.eps).view(T, self.hc, self.H) @ self.fc_h.t()
        return h + e[:, None, :]

    def block(self, h, attn_fn):
        h = h.flatten(-2)
        x, inj = self.ahc(h)
        y = attn_fn(x)
        h = h + (y.unsqueeze(-2) * inj.unsqueeze(-1)).flatten(-2)
        x, inj = self.mhc(h)
        y = self.moe(x)
        h = h + (y.unsqueeze(-2) * inj.unsqueeze(-1)).flatten(-2)
        return (h, self.mixer(h))

    def qkv(self, x, cs):
        a = self.attn
        T = x.shape[0]
        qg = (x @ a.q.t()).view(T, a.H, 2 * a.D)
        q, gate = (qg[..., :a.D], qg[..., a.D:])
        q = S.rms1p(q, a.qn, a.eps)
        k = S.rms1p((x @ a.k.t()).view(T, a.KV, a.D), a.kn, a.eps)
        v = (x @ a.v.t()).view(T, a.KV, a.D)
        q, k = (S.rope_part(q, cs), S.rope_part(k, cs))
        if self.kv_fp8:
            k = k + (k.to(torch.float8_e4m3fn).to(k.dtype) - k).detach()
            v = v + (v.to(torch.float8_e4m3fn).to(v.dtype) - v).detach()
        return (q, k, v, gate)

    def step0(self, hid, tok):
        T = hid.shape[0]
        S._GEN[0] += 1
        c = dict(self.ctx(T), gen=S._GEN[0])
        st = {}

        def attn_fn(x):
            a = self.attn
            SB = a.select(x.detach(), c)
            bm = a.block_mask(SB, T, x.device) if SB is not None else c['causal']
            q, k, v, gate = self.qkv(x, c['rope'])
            o = S._flex(q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None], block_mask=bm, scale=a.D ** (-0.5), enable_gqa=True)[0].transpose(0, 1)
            st.update(k=k, v=v, bm=bm)
            return o.reshape(T, -1) * torch.sigmoid(gate.reshape(T, -1)) @ a.o.t()
        multi, sample = self._run(hid, tok, attn_fn)
        return (sample, multi, st)

    def stepn(self, hid, tok, s, st, extra):
        T = hid.shape[0]
        c = self.ctx(T)
        a = self.attn
        pos = torch.arange(T, device=self.dev) + s
        cs = S.rope_tab(pos, self.rd, self.theta, self.dev)

        def attn_fn(x):
            q, k, v, gate = self.qkv(x, cs)
            o0, lse0 = S._flex(q.transpose(0, 1)[None], st['k'].transpose(0, 1)[None], st['v'].transpose(0, 1)[None], block_mask=st['bm'], scale=a.D ** (-0.5), enable_gqa=True, return_lse=True)
            o0, lse0 = (o0[0].transpose(0, 1).float(), lse0[0].transpose(0, 1).float())
            kv = extra[:s - 1] + [(k, v)]
            rep = a.H // a.KV
            ls, vs = ([lse0], [])
            for kj, vj in kv:
                kj = kj.repeat_interleave(rep, 1)
                ls.append((q.float() * kj.float()).sum(-1) * a.D ** (-0.5))
                vs.append(vj.repeat_interleave(rep, 1).float())
            L = torch.stack(ls, -1)
            w = torch.softmax(L, -1)
            o = o0 * w[..., :1]
            for j, vj in enumerate(vs):
                o = o + vj * w[..., j + 1:j + 2]
            if len(extra) == s - 1:
                extra.append((k, v))
            return o.to(x.dtype).reshape(T, -1) * torch.sigmoid(gate.reshape(T, -1)) @ a.o.t()
        multi, sample = self._run(hid, tok, attn_fn)
        return (sample, multi)

    def _run(self, hid, tok, attn_fn):
        f = lambda h_, t_: self.block(self.inp(h_, t_), attn_fn)
        if self.ckpt and torch.is_grad_enabled():
            return checkpoint(f, hid, tok, use_reentrant=False)
        return f(hid, tok)

    def logits(self, sample, vocab=None):
        W = self.lm_head if vocab is None else self.lm_head[vocab]
        return (sample @ W.t()).float()

    @torch.no_grad()
    def state_delta(self):
        names = self.named_groups()
        prm = dict(self.named_parameters())
        dense = {}
        for nm, key in names.items():
            if key.startswith('<'):
                continue
            dense[key] = prm[nm].detach().to(torch.bfloat16).cpu().contiguous()
        I = self.moe.sd.shape[1]
        p = MP + 'layers.0.mlp.shared_expert.'
        dense[p + 'gate_proj.weight'] = self.moe.sgu[:I].detach().to(torch.bfloat16).cpu().contiguous()
        dense[p + 'up_proj.weight'] = self.moe.sgu[I:].detach().to(torch.bfloat16).cpu().contiguous()
        return dense

    @torch.no_grad()
    def load_trained(self, sd):
        prm = dict(self.named_parameters())
        miss = []
        for k, v in sd.items():
            if k in prm:
                prm[k].copy_(v.to(prm[k].device, prm[k].dtype))
            else:
                miss.append(k)
        return miss
