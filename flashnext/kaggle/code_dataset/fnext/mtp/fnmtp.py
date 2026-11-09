import os
_ENV = lambda k, d=None: os.environ.get(k, d)

def _log(msg):
    print(f'[fnmtp] {msg}', flush=True)

def flags_on():
    return any((k.startswith('FN_MTP_') and v not in ('', '0') for k, v in os.environ.items()))

def fake_quant_rtn(w, bits=4, group=128):
    import torch
    shp = w.shape
    x = w.float().reshape(-1, shp[-1] // group, group)
    lo, hi = (x.amin(-1, keepdim=True), x.amax(-1, keepdim=True))
    qmax = 2 ** bits - 1
    scale = ((hi - lo) / qmax).clamp_min(1e-08)
    zero = torch.round(-lo / scale).clamp(0, qmax)
    q = torch.clamp(torch.round(x / scale) + zero, 0, qmax)
    return ((q - zero) * scale).reshape(shp).to(w.dtype)

def parse_fq(spec):
    kind, _, g = spec.partition(':')
    bits = {'int4': 4, 'int3': 3, 'int8': 8, 'int2': 2}[kind]
    group = int(g[1:]) if g.startswith('g') else 128
    return (bits, group)

def load_ids(path):
    import torch
    if path.endswith('.pt'):
        ids = torch.load(path)
        if isinstance(ids, dict):
            ids = ids['ids']
        return torch.as_tensor(ids, dtype=torch.long)
    return torch.tensor([int(x) for x in open(path).read().split()], dtype=torch.long)

def load_keep(path):
    import torch
    k = torch.load(path)
    if isinstance(k, dict):
        k = k['keep']
    k = torch.as_tensor(k).bool()
    if k.dim() == 2:
        k = k[48]
    return k

class DraftHead:

    def __init__(self, lm_head, quant='int8', ids=None, mode=None):
        import torch
        from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size, tensor_model_parallel_all_gather
        self.tp = get_tensor_model_parallel_world_size()
        self.rank = get_tensor_model_parallel_rank()
        self.quant = quant
        self.mode = mode or ('rep' if ids is not None else 'tp')
        si = lm_head.shard_indices
        w_loc = _dense_weight(lm_head)[:si.org_vocab_end_index - si.org_vocab_start_index]
        dev = w_loc.device
        if self.mode == 'rep' or ids is not None:
            n_loc = torch.tensor([w_loc.shape[0], si.org_vocab_start_index], device=dev)
            if self.tp > 1:
                ns = tensor_model_parallel_all_gather(n_loc[None], dim=0).tolist()
                vmax = max((n for n, _ in ns))
                pad = torch.zeros(vmax, w_loc.shape[1], dtype=w_loc.dtype, device=dev)
                pad[:w_loc.shape[0]] = w_loc
                allw = tensor_model_parallel_all_gather(pad[None], dim=0)
                full = torch.cat([allw[r, :n] for r, (n, _) in enumerate(ns)], 0)
                del allw, pad
            else:
                full = w_loc
            V = full.shape[0]
            ids = torch.arange(V, device=dev) if ids is None else ids.to(dev)
            ids = ids[(ids >= 0) & (ids < V)].unique()
            if self.mode == 'tp':
                chunk = (ids.numel() + self.tp - 1) // self.tp
                ids = ids[self.rank * chunk:(self.rank + 1) * chunk]
            w = full.index_select(0, ids)
            del full
        else:
            ids = torch.arange(si.org_vocab_start_index, si.org_vocab_end_index, device=dev)
            w = w_loc
        self.n = ids.numel()
        npad = -self.n % 128
        if npad:
            w = torch.cat([w, torch.zeros(npad, w.shape[1], dtype=w.dtype, device=dev)], 0)
            ids = torch.cat([ids, torch.full((npad,), -1, dtype=ids.dtype, device=dev)], 0)
        self.ids = ids.contiguous()
        self.valid_neg = torch.where(self.ids >= 0, 0.0, float('-inf')).to(device=dev, dtype=torch.float32)
        self.has_pad = bool(npad)
        wf = w.float()
        if quant == 'int8':
            s = wf.abs().amax(1).clamp_min(1e-08) / 127.0
            self.wq = torch.round(wf / s[:, None]).clamp(-127, 127).to(torch.int8)
            self.ws = s.to(torch.float32)[None, :].contiguous()
        elif quant == 'fp8':
            s = wf.abs().amax(1).clamp_min(1e-08) / 448.0
            self.wq = (wf / s[:, None]).to(torch.float8_e4m3fn)
            self.ws = s.to(torch.float32)[None, :].contiguous()
        elif quant == 'bf16':
            self.wq = w.to(torch.bfloat16).contiguous()
            self.ws = None
        else:
            raise ValueError(f'FN_MTP_HEAD={quant}')
        del wf, w
        self._fallback = False
        torch.cuda.empty_cache()
        _log(f'draft head {quant} mode={self.mode} rows={self.n} (+{npad} pad) {self.wq.numel() * self.wq.element_size() / 2 ** 20:.1f} MiB/rank, tp={self.tp}')

    def logits(self, h):
        import torch
        from vllm import _custom_ops as ops
        h2 = h.reshape(-1, h.shape[-1])
        if self.quant == 'bf16':
            out = torch.mm(h2.to(torch.bfloat16), self.wq.t())
        elif not self._fallback:
            try:
                if self.quant == 'int8':
                    hq, hs, _ = ops.scaled_int8_quant(h2.contiguous(), symmetric=True)
                else:
                    hq, hs = ops.scaled_fp8_quant(h2.contiguous(), use_per_token_if_dynamic=True)
                out = ops.cutlass_scaled_mm(hq, self.wq.t(), scale_a=hs, scale_b=self.ws, out_dtype=torch.bfloat16)
            except Exception as e:
                _log(f'cutlass path failed ({repr(e)[:160]}); using dequant matmul fallback')
                self._fallback = True
                return self.logits(h)
        else:
            out = torch.mm(h2.float(), (self.wq.float() * self.ws.t()).t())
        out = out.float()
        if self.has_pad:
            out = out + self.valid_neg
        return out

    def full_logits(self, h, vocab, dtype):
        import torch
        out = self.logits(h)
        full = torch.full((out.shape[0], vocab), float('-inf'), dtype=dtype, device=out.device)
        if self.has_pad:
            full[:, self.ids[:self.n]] = out[:, :self.n].to(dtype)
        else:
            full[:, self.ids] = out.to(dtype)
        return full

    def get_top_tokens(self, h):
        import torch
        from vllm.distributed import tensor_model_parallel_all_gather
        out = self.logits(h)
        val, idx = out.max(-1)
        tok = self.ids[idx]
        if self.mode == 'rep' or self.tp == 1:
            return tok.to(torch.int64)
        pair = torch.stack([val, tok.float()], -1)
        g = tensor_model_parallel_all_gather(pair, dim=-1).view(out.shape[0], self.tp, 2)
        best = g[:, :, 0].argmax(-1, keepdim=True)
        return g[:, :, 1].gather(-1, best).squeeze(-1).to(torch.int64)

def _dense_weight(lm_head):
    import torch
    w = lm_head.weight
    if w.dtype in (torch.bfloat16, torch.float16, torch.float32):
        return w.data
    if w.dtype == torch.float8_e4m3fn:
        s = getattr(lm_head, 'fn_fp8_scale', None)
        if s is not None:
            return (w.data.float() * s.data.float().reshape(-1, 1)[:w.shape[0]]).to(torch.bfloat16)
        for name in ('weight_scale_inv', 'weight_scale'):
            s = getattr(lm_head, name, None)
            if s is not None:
                s = s.data.float()
                if s.numel() == 1:
                    return (w.data.float() * s).to(torch.bfloat16)
                if s.dim() == 2 and s.shape[0] * 128 >= w.shape[0]:
                    s2 = s.repeat_interleave(128, 0)[:w.shape[0]].repeat_interleave(128, 1)[:, :w.shape[1]]
                    return (w.data.float() * s2).to(torch.bfloat16)
                return (w.data.float() * s.reshape(-1, 1)[:w.shape[0]]).to(torch.bfloat16)
    raise NotImplementedError(f'FN_MTP_HEAD: lm_head weight dtype {w.dtype} not supported')

def _mtp_cls():
    from vllm.models.qwen4_exp.nvidia import mtp as M
    return M.Qwen4ExpMTP

def _install_top_tokens():
    cls = _mtp_cls()

    def get_top_tokens(self, hidden_states, spec_step_idx=0):
        head = getattr(self, '_fn_draft_head', None)
        if head is not None:
            return head.get_top_tokens(hidden_states)
        return self.logits_processor.get_top_tokens(self.lm_head, hidden_states)
    cls.get_top_tokens = get_top_tokens

def _install_head_logits():
    cls = _mtp_cls()
    orig = cls.compute_logits

    def compute_logits(self, hidden_states, spec_step_idx=0):
        head = getattr(self, '_fn_draft_head', None)
        if head is None or head.mode != 'rep':
            return orig(self, hidden_states, spec_step_idx)
        dtype = getattr(self.logits_processor, 'head_dtype', None) or hidden_states.dtype
        return head.full_logits(hidden_states, self.config.vocab_size, dtype)
    cls.compute_logits = compute_logits

def _after_draft_load(model):
    if type(model).__name__ != 'Qwen4ExpMTP':
        return
    q = _ENV('FN_MTP_HEAD')
    if q:
        ids = load_ids(_ENV('FN_MTP_HEAD_VOCAB')) if _ENV('FN_MTP_HEAD_VOCAB') else None
        model._fn_draft_head = DraftHead(model.lm_head, quant=q, ids=ids, mode=_ENV('FN_MTP_HEAD_MODE'))
    if _ENV('FN_MTP_KEEP'):
        _install_keep(model, load_keep(_ENV('FN_MTP_KEEP')))

def _install_keep(model, keep):
    import torch
    runners = [m for m in model.modules() if hasattr(m, '_maybe_dispatch') and hasattr(m, 'moe_config')]
    assert runners, 'FN_MTP_KEEP: no MoE runner found in the MTP model'
    for r in runners:
        dev = next(model.parameters()).device
        mask = torch.where(keep.to(dev), 0.0, float('-inf')).to(torch.float32)
        orig = r._maybe_dispatch

        def md(hidden_states, router_logits, _orig=orig, _mask=mask):
            router_logits = (router_logits.float() + _mask).to(router_logits.dtype)
            return _orig(hidden_states, router_logits)
        r._maybe_dispatch = md
    _log(f'FN_MTP_KEEP: {int(keep.sum())}/{keep.numel()} MTP experts kept ({len(runners)} runner)')

def _wrap_load_model(cls, attr='model'):
    orig = cls.load_model

    def load_model(self, *a, **k):
        out = orig(self, *a, **k)
        _after_draft_load(getattr(self, attr))
        return out
    cls.load_model = load_model

def _install_exp_fq(spec):
    import re
    bits, group = parse_fq(spec)
    cls = _mtp_cls()
    orig = cls.load_weights
    pat = re.compile('mtp\\.layers\\.\\d+\\.mlp\\.experts\\.\\d+\\.(gate|up|down)_proj\\.weight$')

    def load_weights(self, weights):
        n = [0]

        def gen():
            import torch
            for name, w in weights:
                if pat.search(name) and w.dtype in (torch.bfloat16, torch.float16, torch.float32):
                    w = fake_quant_rtn(w.to('cuda') if torch.cuda.is_available() else w, bits, group).to(w.device)
                    n[0] += 1
                yield (name, w)
        out = orig(self, gen())
        _log(f'FN_MTP_EXP_FQ int{bits} g{group}: {n[0]} MTP expert tensors fake-quantized')
        assert n[0] > 0, 'FN_MTP_EXP_FQ: no BF16 MTP expert tensors seen (FP8 checkpoint? use fnplug FN_OVR)'
        return out
    cls.load_weights = load_weights

def ovr_stream(weights, ovr, seen):
    for name, w in weights:
        if name in ovr:
            o = ovr[name]
            assert tuple(o.shape) == tuple(w.shape) and o.dtype == w.dtype, (name, o.shape, w.shape, o.dtype, w.dtype)
            seen.add(name)
            w = o
        yield (name, w)

def load_ovr(path):
    import re
    from safetensors.torch import load_file
    ovr = load_file(path)
    bad = [k for k in ovr if not k.startswith('mtp.') or re.search('\\.experts\\.|\\.mlp\\.gate\\.weight$', k)]
    assert not bad, f'FN_MTP_OVR: refused tensors {bad[:5]}'
    return ovr

def _install_ovr(path):
    cls = _mtp_cls()
    orig = cls.load_weights
    ovr = load_ovr(path)

    def load_weights(self, weights):
        seen = set()
        out = orig(self, ovr_stream(weights, ovr, seen))
        miss = sorted(set(ovr) - seen)
        assert not miss, f'FN_MTP_OVR: tensors never seen in the checkpoint stream: {miss[:5]}'
        _log(f'FN_MTP_OVR: {len(seen)} MTP tensors replaced from {path}')
        return out
    cls.load_weights = load_weights

def _install_draft_topk(k):
    import torch
    cls = _mtp_cls()
    orig = cls.compute_logits

    def compute_logits(self, hidden_states, spec_step_idx=0):
        logits = orig(self, hidden_states, spec_step_idx)
        if logits is None:
            return logits
        kth = torch.topk(logits, k, dim=-1).values[..., -1:]
        return logits.masked_fill(logits < kth, float('-inf'))
    cls.compute_logits = compute_logits

def _install_draft_ban(ids):
    import torch
    cls = _mtp_cls()
    orig = cls.compute_logits
    ban = [int(x) for x in ids.split(',') if x.strip()]

    def compute_logits(self, hidden_states, spec_step_idx=0):
        logits = orig(self, hidden_states, spec_step_idx)
        if logits is not None:
            t = getattr(self, '_fn_ban_t', None)
            if t is None or t.device != logits.device:
                t = self._fn_ban_t = torch.tensor(ban, dtype=torch.long, device=logits.device)
            logits.index_fill_(-1, t, float('-inf'))
        return logits
    cls.compute_logits = compute_logits

def install():
    on = []
    if _ENV('FN_MTP_LOCAL_ARGMAX') or _ENV('FN_MTP_HEAD'):
        _install_top_tokens()
        on.append('get_top_tokens')
    if _ENV('FN_MTP_HEAD'):
        _install_head_logits()
        on.append('head_logits')
    if _ENV('FN_MTP_HEAD') or _ENV('FN_MTP_KEEP'):
        from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
        _wrap_load_model(DraftModelSpeculator)
        try:
            from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
            _wrap_load_model(SpecDecodeBaseProposer)
        except Exception:
            pass
        on.append(f'head={_ENV('FN_MTP_HEAD')}' if _ENV('FN_MTP_HEAD') else 'keep')
    if _ENV('FN_MTP_EXP_FQ'):
        _install_exp_fq(_ENV('FN_MTP_EXP_FQ'))
        on.append(f'exp_fq={_ENV('FN_MTP_EXP_FQ')}')
    if _ENV('FN_MTP_OVR'):
        _install_ovr(_ENV('FN_MTP_OVR'))
        on.append('ovr')
    if _ENV('FN_MTP_DRAFT_TOPK'):
        _install_draft_topk(int(_ENV('FN_MTP_DRAFT_TOPK')))
        on.append(f'draft_topk={_ENV('FN_MTP_DRAFT_TOPK')}')
    if _ENV('FN_MTP_DRAFT_BAN'):
        _install_draft_ban(_ENV('FN_MTP_DRAFT_BAN'))
        on.append(f'draft_ban={_ENV('FN_MTP_DRAFT_BAN')}')
    _log('installed: ' + (', '.join(on) or 'nothing'))
