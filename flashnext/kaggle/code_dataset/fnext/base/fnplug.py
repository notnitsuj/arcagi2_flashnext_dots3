import os
import re
EXP = re.compile('^(model\\.language_model\\.layers\\.(\\d+)|mtp\\.layers\\.(\\d+))\\.mlp\\.experts\\.(\\d+)\\.(.+)$')
GATE = re.compile('^(model\\.language_model\\.layers\\.(\\d+)|mtp\\.layers\\.(\\d+))\\.mlp\\.gate\\.weight$')
MTP_ROW = 48
OROWS = set()

def _row(m):
    return int(m.group(2)) if m.group(2) is not None else MTP_ROW + int(m.group(3))

def _log(msg):
    print(f'[fnplug] {msg}', flush=True)

def register():
    keep_p, ovr, cap = (os.environ.get('FN_KEEP'), os.environ.get('FN_OVR'), os.environ.get('FN_CAP_DIR'))
    if not (keep_p or ovr or cap):
        return
    import torch
    if cap:
        _install_capture()
    if keep_p or ovr:
        _install_loader(keep_p, ovr)
    if ovr:
        from vllm.model_executor.layers.quantization import fp8 as f8
        from vllm.model_executor.layers.fused_moe import RoutedExperts, UnquantizedFusedMoEMethod
        orig = f8.Fp8Config.get_quant_method
        orows = OROWS

        def gqm(self, layer, prefix):
            if isinstance(layer, RoutedExperts):
                mm = re.search('layers\\.(\\d+)\\.', prefix)
                r = int(mm.group(1)) if mm else -1
                if 'mtp' in prefix:
                    r = MTP_ROW + (r - MTP_ROW if r >= MTP_ROW else r)
                if r in orows:
                    _log(f'{prefix} (row {r}): unquantized BF16 experts from FN_OVR')
                    return UnquantizedFusedMoEMethod(layer.moe_config)
            return orig(self, layer, prefix)
        f8.Fp8Config.get_quant_method = gqm
        _log(f'FN_OVR: {len(orows)} MoE layers served unquantized (BF16), the rest stay FP8')

def _install_loader(keep_p, ovr):
    import torch
    from safetensors import safe_open
    from vllm.model_executor.model_loader import default_loader as dl
    keep = torch.load(keep_p)['keep'].bool() if keep_p else None
    newid = None
    if keep is not None:
        ks = keep.sum(1)
        assert (ks == ks[0]).all(), f'keep sets must have the same K in every row: {ks.tolist()}'
        newid = torch.full(keep.shape, -1, dtype=torch.long)
        for r in range(keep.shape[0]):
            idx = keep[r].nonzero().squeeze(1)
            newid[r, idx] = torch.arange(len(idx))
        _log(f'FN_KEEP {keep_p}: K={int(ks[0])}, rows={keep.shape[0]}')
    ofiles = {}
    if ovr:
        for f in sorted(os.listdir(ovr)):
            if f.endswith('.safetensors'):
                with safe_open(f'{ovr}/{f}', 'pt') as sf:
                    for k in sf.keys():
                        ofiles[k] = f
        orows = set()
        for k in ofiles:
            m = EXP.match(k)
            if m:
                orows.add(_row(m))
        _log(f'FN_OVR {ovr}: {len(ofiles)} tensors, rows {sorted(orows)[:3]}..{sorted(orows)[-3:]} ({len(orows)})')
    else:
        orows = set()
    OROWS.update(orows)
    gaw = dl.DefaultModelLoader.get_all_weights
    st = {'drop': 0, 'ren': 0, 'gate': 0, 'ovr': 0}

    def get_all_weights(self, *a, **k):
        emitted = set()
        hs = {}
        for name, t in gaw(self, *a, **k):
            m = EXP.match(name)
            if m:
                r, e = (_row(m), int(m.group(4)))
                if r in orows:
                    if r not in emitted:
                        emitted.add(r)
                        pre = m.group(1) + '.mlp.experts.'
                        for key, f in ofiles.items():
                            if key.startswith(pre):
                                if f not in hs:
                                    hs[f] = safe_open(f'{ovr}/{f}', 'pt')
                                st['ovr'] += 1
                                yield (key, hs[f].get_tensor(key))
                    st['drop'] += 1
                    continue
                if newid is not None and r < newid.shape[0]:
                    j = int(newid[r, e])
                    if j < 0:
                        st['drop'] += 1
                        continue
                    name = f'{m.group(1)}.mlp.experts.{j}.{m.group(5)}'
                    st['ren'] += 1
                yield (name, t)
                continue
            g = GATE.match(name)
            if g and keep is not None and (_row(g) < keep.shape[0]):
                t = t[keep[_row(g)].to(t.device)].contiguous()
                st['gate'] += 1
            yield (name, t)
        _log(f'loader: {st}')
    dl.DefaultModelLoader.get_all_weights = get_all_weights
CAP = {'rank': None, 'gen': None, 'sub': None, 'buf': {}, 'n': {}, 'chunk': {}, 'cnt': {}, 'wsum': {}, 'tok': {}, 'seen': 0}

def _cap_flush_all(sub):
    import torch
    d = os.path.join(os.environ['FN_CAP_DIR'], sub)
    os.makedirs(d, exist_ok=True)
    for li in list(CAP['buf']):
        _cap_flush(li, d)
    if CAP['cnt']:
        L = sorted(CAP['cnt'])
        torch.save({'layers': L, 'counts': {l: CAP['cnt'][l].cpu() for l in L}, 'wsum': {l: CAP['wsum'][l].cpu() for l in L}, 'tokens': {l: CAP['tok'][l] for l in L}}, f'{d}/stats.pt')
        _log(f'capture flushed to {d}: tokens/layer {CAP['tok'].get(0, 0)}')
    CAP.update(buf={}, n={}, chunk={}, cnt={}, wsum={}, tok={})

def _cap_flush(li, d):
    import torch
    xs = CAP['buf'].pop(li, [])
    if not xs:
        return
    os.makedirs(d, exist_ok=True)
    k = CAP['chunk'].get(li, 0)
    CAP['chunk'][li] = k + 1
    torch.save({'x': torch.cat([a for a, _ in xs]), 'logits': torch.cat([b for _, b in xs])}, f'{d}/L{li:02d}_{k:04d}.pt')
    CAP['n'][li] = 0

def _install_capture():
    import torch
    from vllm.models.qwen4_exp.nvidia import model as m
    orig = m.Qwen4ExpSparseMoeBlock.__init__

    def init(self, vllm_config, prefix='', *a, **k):
        orig(self, vllm_config, prefix, *a, **k)
        mm = re.search('layers\\.(\\d+)\\.', prefix)
        if not mm:
            return
        li = int(mm.group(1))
        if 'mtp' in prefix:
            li = MTP_ROW + (li - MTP_ROW if li >= MTP_ROW else li)
        _hook(self.gate, li, vllm_config.model_config.hf_text_config.num_experts_per_tok)
        _log(f'capture hook on {prefix} -> L{li}')
    m.Qwen4ExpSparseMoeBlock.__init__ = init

def _hook(gate, li, topk):
    import torch
    root = os.environ['FN_CAP_DIR']
    p = float(os.environ.get('FN_CAP_P', '0.08'))
    onf = os.path.join(root, 'on')

    def hook(mod, inp, kw, out):
        if CAP['rank'] is None:
            CAP['rank'] = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if CAP['rank'] != 0 or torch.cuda.is_current_stream_capturing():
            return
        if li == 0 or CAP['sub'] is None:
            try:
                sub = open(onf).read().strip() or None
            except OSError:
                sub = None
            if sub != CAP['sub']:
                if CAP['sub'] is not None:
                    _cap_flush_all(CAP['sub'])
                CAP['sub'] = sub
        if CAP['sub'] is None:
            return
        x = inp[0] if inp else next(iter(kw.values()))
        lg = out[0] if isinstance(out, tuple) else out
        if x.shape[0] == 0:
            return
        pr = torch.softmax(lg.float(), -1)
        w, ids = torch.topk(pr, topk, -1)
        w = w / w.sum(-1, keepdim=True)
        E = lg.shape[-1]
        if li not in CAP['cnt']:
            CAP['cnt'][li] = torch.zeros(E, dtype=torch.float64, device=lg.device)
            CAP['wsum'][li] = torch.zeros(E, dtype=torch.float64, device=lg.device)
            CAP['tok'][li] = 0
        CAP['cnt'][li].index_add_(0, ids.reshape(-1), torch.ones(ids.numel(), dtype=torch.float64, device=lg.device))
        CAP['wsum'][li].index_add_(0, ids.reshape(-1), w.reshape(-1).double())
        CAP['tok'][li] += x.shape[0]
        if CAP['gen'] is None:
            CAP['gen'] = torch.Generator(device=x.device)
            CAP['gen'].manual_seed(1234)
        sel = (torch.rand(x.shape[0], device=x.device, generator=CAP['gen']) < p).nonzero().squeeze(1)
        if sel.numel():
            CAP['buf'].setdefault(li, []).append((x[sel].to(torch.bfloat16).cpu(), lg[sel].to(torch.bfloat16).cpu()))
            CAP['n'][li] = CAP['n'].get(li, 0) + sel.numel()
            if CAP['n'][li] >= 32768:
                _cap_flush(li, os.path.join(root, CAP['sub']))
    gate.register_forward_hook(hook, with_kwargs=True)
