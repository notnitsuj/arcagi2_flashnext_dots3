import os
import re
LAYER = re.compile('layers\\.(\\d+)\\.mlp\\.experts$')
EXP = re.compile('^(model\\.language_model\\.layers\\.(\\d+)|mtp\\.layers\\.(\\d+))\\.mlp\\.experts\\.(\\d+)\\.(.+)$')
GATE = re.compile('^(model\\.language_model\\.layers\\.(\\d+)|mtp\\.layers\\.(\\d+))\\.mlp\\.gate\\.weight$')
MTP_ROW = 48
_DEBUG = os.environ.get('FN_MOE_DEBUG') == '1'
_ST = {'dir': None, 'layers': None, 'deltas': []}

def _log(msg):
    print(f'[fnmoe] {msg}', flush=True)

def _row(m):
    return int(m.group(2)) if m.group(2) is not None else MTP_ROW + int(m.group(3))
_PLACE = {'rows': None}
_RS = {'obj': None}

def _placement(l, E):
    if _PLACE['rows'] is None:
        import json
        p = os.environ.get('FN_EP_PLACE')
        _PLACE['rows'] = json.load(open(p))['rank'] if p else []
    rows = _PLACE['rows']
    r = rows[l] if l < len(rows) else None
    if r is not None:
        assert len(r) == E, f'FN_EP_PLACE row {l}: {len(r)} entries for {E} experts'
    return r

class _RouteStats:
    RING = 16384

    def __init__(self, d, E, ep_rank, dev):
        import queue, threading, time
        import torch
        self.d, self.E, self.rank = (d, E, ep_rank)
        self.cnt = torch.zeros(MTP_ROW + 1, E, dtype=torch.int64, device=dev)
        self.tok = torch.zeros(MTP_ROW + 1, dtype=torch.int64, device=dev)
        self.calls = torch.zeros(MTP_ROW + 1, dtype=torch.int64, device=dev)
        self.ring = torch.zeros(self.RING, dtype=torch.int32, device=dev)
        self.ctr = torch.zeros(1, dtype=torch.int64, device=dev)
        self.host = {k: torch.empty(getattr(self, k).shape, dtype=getattr(self, k).dtype, pin_memory=True) for k in self.KEYS}
        self.t0, self.steps, self.busy = (time.time(), 0, False)
        self.q = queue.Queue()
        os.makedirs(d, exist_ok=True)
        threading.Thread(target=self._writer, daemon=True).start()
        self._hook()
    KEYS = ('cnt', 'tok', 'calls', 'ring', 'ctr')
    EVERY = int(os.environ.get('FN_MOE_ROUTESTATS_EVERY', '1500'))

    def _hook(self):
        from vllm.v1.worker.gpu import model_runner as mr
        R = mr.GPUModelRunner
        if getattr(R, '_fn_routestats', False):
            return
        R._fn_routestats = True
        st0, rs = (R.sample_tokens, self)

        def sample_tokens(self_, *a, **kw):
            out = st0(self_, *a, **kw)
            rs.tick()
            return out
        R.sample_tokens = sample_tokens

    def tick(self):
        import torch
        self.steps += 1
        if self.steps % self.EVERY or self.busy or torch.cuda.is_current_stream_capturing():
            return
        for k in self.KEYS:
            self.host[k].copy_(getattr(self, k), non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        self.busy = True
        self.q.put(ev)

    def _writer(self):
        import time
        import torch
        while True:
            ev = self.q.get()
            try:
                ev.synchronize()
                out = {k: v.clone() for k, v in self.host.items()}
                out['t'], out['steps'] = (time.time() - self.t0, self.steps)
                p = f'{self.d}/rs_ep{self.rank}.pt'
                torch.save(out, p + '.tmp')
                os.replace(p + '.tmp', p)
            except Exception as e:
                _log(f'routestats write failed: {e!r}')
            finally:
                self.busy = False

    def add(self, l, topk_ids, tp_rank, tp_size):
        import torch
        from vllm.forward_context import get_forward_context, is_forward_context_available
        if not is_forward_context_available():
            return
        ip = get_forward_context().is_padding
        if ip is None:
            return
        n, N = (topk_ids.shape[0], ip.shape[0])
        if n == N:
            valid = ~ip
        else:
            if n > N or n * tp_size < N:
                if not getattr(self, '_warned', False):
                    self._warned = True
                    _log(f'routestats: L{l} topk rows {n} vs padding mask {N} (tp {tp_size}): not counted')
                return
            lo = min(tp_rank * n, N)
            hi = min(lo + n, N)
            valid = torch.zeros(n, dtype=torch.bool, device=topk_ids.device)
            if hi > lo:
                valid[:hi - lo] = ~ip[lo:hi]
        w = valid.to(torch.int64)
        ids = topk_ids.to(torch.int64).clamp(min=0, max=self.E - 1)
        self.cnt[l].scatter_add_(0, ids.reshape(-1), w[:, None].expand(ids.shape).reshape(-1))
        vs = w.sum()
        self.tok[l] += vs
        self.calls[l] += 1
        if l == 0:
            self.ring.index_copy_(0, self.ctr % self.RING, vs.to(torch.int32).view(1))
            self.ctr += 1

def _served(l):
    d = _ST['dir']
    if d is None or not os.path.exists(f'{d}/lowbit-L{l:02d}.pt'):
        return False
    lr = _ST['layers']
    return lr is None or lr[0] <= l <= lr[1]

def register():
    d = os.environ.get('FN_MOE_LOWBIT')
    if not d:
        return
    if os.environ.get('FN_KEEP') or os.environ.get('FN_OVR'):
        raise RuntimeError('FN_MOE_LOWBIT cannot be combined with fnplug FN_KEEP/FN_OVR')
    _ST['dir'] = d.rstrip('/')
    _ST['deltas'] = [x.rstrip('/') for x in os.environ.get('FN_MOE_DELTA', '').split(':') if x]
    if os.environ.get('FN_MOE_LAYERS'):
        lo, hi = (int(v) for v in os.environ['FN_MOE_LAYERS'].split('-'))
        _ST['layers'] = (lo, hi)
    _install_method()
    _install_loader()
    _install_padding_guard()
    _log(f'FN_MOE_LOWBIT={d} layers={_ST['layers'] or 'all with a file'}' + (f' deltas={_ST['deltas']}' if _ST['deltas'] else ''))

def apply_scale_deltas(pk, l, deltas=None):
    import torch
    for d in _ST['deltas'] if deltas is None else deltas:
        sp = f'{d}/scales-L{l:02d}.pt'
        if os.path.exists(sp):
            sc = torch.load(sp, map_location='cpu', weights_only=False)
            assert list(sc) == list(pk['experts']), f'{sp}: expert ids differ from the base build'
            for e, v in sc.items():
                pk[e]['gu_s'], pk[e]['d_s'] = (v['gu_s'], v['d_s'])
    return pk

def router_path(l, base=None, deltas=None):
    p = f'{(_ST['dir'] if base is None else base)}/router-L{l:02d}.pt'
    for d in _ST['deltas'] if deltas is None else deltas:
        if os.path.exists(f'{d}/router-L{l:02d}.pt'):
            p = f'{d}/router-L{l:02d}.pt'
    return p

def _install_method():
    from vllm.model_executor.layers.fused_moe import routed_experts as rx
    orig = rx.RoutedExperts._get_quant_method

    def _get_quant_method(self, prefix, quant_config, moe_config):
        m = LAYER.search(prefix)
        if m:
            l = MTP_ROW if 'mtp' in prefix else int(m.group(1))
            if _served(l):
                return _method_cls()(moe_config, l, _ST['dir'])
        return orig(self, prefix, quant_config, moe_config)
    rx.RoutedExperts._get_quant_method = _get_quant_method
_CLS = None
_LOWBIT_EXPERTS = None

def _method_cls():
    global _CLS
    if _CLS is not None:
        return _CLS
    import torch
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase
    try:
        from . import lowbit_kernels as lk
    except ImportError:
        import lowbit_kernels as lk

    class LowbitMoEMethod(FusedMoEMethodBase):

        def __init__(self, moe, layer_idx, d):
            super().__init__(moe)
            self.layer_idx, self.path = (layer_idx, f'{d}/lowbit-L{layer_idx:02d}.pt')
            self.lb = None
            self._dbg = 0

        def create_weights(self, layer, num_experts, hidden_size, intermediate_size_per_partition, params_dtype, **extra):
            pc = self.moe.moe_parallel_config
            if pc.tp_size > 1 and (not pc.use_ep):
                raise RuntimeError('FN_MOE_LOWBIT: g128 experts need whole experts per rank; run with --enable-expert-parallel (TEP / DP+EP)')
            self.num_local = num_experts

        def get_fused_moe_quant_config(self, layer):
            return FusedMoEQuantConfig.make()

        def process_weights_after_loading(self, layer):
            if self.lb is not None:
                return
            dev = torch.device('cuda', torch.cuda.current_device())
            pk = torch.load(self.path, map_location='cpu', weights_only=False, mmap=os.environ.get('FN_MOE_MMAP') == '1')
            if _ST['deltas']:
                apply_scale_deltas(pk, self.layer_idx)
            E = layer.global_num_experts
            emap = layer.expert_map
            pl = _placement(self.layer_idx, E) if emap is not None else None
            if pl is not None:
                r = self.moe.moe_parallel_config.ep_rank
                mine = [g for g in range(E) if pl[g] == r]
                assert len(mine) == layer.local_num_experts, f'FN_EP_PLACE L{self.layer_idx}: {len(mine)} experts on rank {r}, need {layer.local_num_experts}'
                new = torch.full((E,), -1, dtype=emap.dtype)
                new[mine] = torch.arange(len(mine), dtype=emap.dtype)
                emap.copy_(new.to(emap.device))
            if os.environ.get('FN_MOE_ROUTESTATS') and _RS['obj'] is None:
                _RS['obj'] = _RouteStats(os.environ['FN_MOE_ROUTESTATS'], E, self.moe.moe_parallel_config.ep_rank, dev)
            if emap is None:
                loc = list(range(E))
            else:
                em = emap.cpu().tolist()
                loc = sorted((g for g in range(E) if em[g] >= 0), key=lambda g: em[g])
                assert [em[g] for g in loc] == list(range(len(loc))), 'expert_map slots must be dense'
            if os.environ.get('FN_MOE_REF') == '1':
                self.lb = _RefExperts(pk, loc, E, dev)
            else:
                self.lb = lk.build_local(pk, loc, E, dev, block_m=int(os.environ.get('FN_MOE_BLOCK_M', '16')))
            self.moe_kernel = None
            mkf = os.environ.get('FN_MOE_MK', 'auto')
            if mkf == '1' or (mkf == 'auto' and self.moe.moe_parallel_config.dp_size > 1):
                self._make_kernel(layer, len(loc))
            bits = {}
            for g in loc:
                b = int(pk[pk['experts'][g]]['bits'])
                bits[b] = bits.get(b, 0) + 1
            del pk
            if self.layer_idx in (0, MTP_ROW):
                _log(f'L{self.layer_idx}: {len(loc)}/{E} experts on this rank, bits {bits}, {self.lb.nbytes() / 2 ** 20:.1f} MiB, pairs={getattr(self.lb.gu, 'pairs', False)}' + (f', placement FN_EP_PLACE first ids {loc[:4]}' if pl is not None else ''))

        def _make_kernel(self, layer, n_local):
            import vllm.model_executor.layers.fused_moe.modular_kernel as mk
            from vllm.model_executor.layers.fused_moe.all2all_utils import maybe_make_prepare_finalize
            qc = FusedMoEQuantConfig.make()
            self.moe_quant_config = qc
            pf = maybe_make_prepare_finalize(moe=self.moe, quant_config=qc, routing_tables=layer._expert_routing_tables(), allow_new_interface=True, use_monolithic=False)
            if pf is None:
                return
            ex = _LowbitExperts(self.moe, qc, self.lb)
            self.moe_kernel = mk.FusedMoEKernel(pf, ex)
            dev = torch.device('cuda', torch.cuda.current_device())
            H, I2 = (self.moe.hidden_dim, 2 * self.moe.intermediate_size_per_partition)
            self._w1 = torch.empty(n_local, I2, H, device='meta')
            self._w2 = torch.empty(n_local, H, I2 // 2, device='meta')
            if self.layer_idx in (0, MTP_ROW):
                _log(f'L{self.layer_idx}: modular kernel with {type(pf).__name__}')

        def apply(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
            rs = _RS['obj']
            if rs is not None and self.layer_idx != MTP_ROW:
                from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
                rs.add(self.layer_idx, topk_ids, get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size())
            if self.moe_kernel is not None:
                return self.moe_kernel.apply(hidden_states=x, w1=self._w1, w2=self._w2, topk_weights=topk_weights, topk_ids=topk_ids, activation=self.moe.activation, global_num_experts=layer.global_num_experts, expert_map=layer.expert_map, apply_router_weight_on_input=False, shared_experts=shared_experts, shared_experts_input=shared_experts_input)
            if _DEBUG and (not torch.cuda.is_current_stream_capturing()) and (self._dbg < 3):
                self._dbg += 1
                em = layer.expert_map
                _log(f'L{self.layer_idx} apply: x {tuple(x.shape)} {x.dtype} stride {x.stride()} | ids {tuple(topk_ids.shape)} {topk_ids.dtype} stride {topk_ids.stride()} min {int(topk_ids.min())} max {int(topk_ids.max())} | w {tuple(topk_weights.shape)} {topk_weights.dtype} | map {(None if em is None else (tuple(em.shape), em.dtype))}')
            n = topk_ids.shape[0]
            if x.shape[0] > n:
                y = self.lb(x[:n], topk_weights.contiguous(), topk_ids.contiguous(), layer.expert_map)
                return torch.cat([y, y.new_zeros(x.shape[0] - n, y.shape[1])])
            return self.lb(x, topk_weights.contiguous(), topk_ids.contiguous(), layer.expert_map)
    import vllm.model_executor.layers.fused_moe.modular_kernel as mk
    from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import TopKWeightAndReduceNoOP

    class _LowbitExperts(mk.FusedMoEExpertsModular):

        def __init__(self, moe, qc, lb):
            super().__init__(moe, qc)
            self.lb = lb

        @staticmethod
        def activation_format():
            return mk.FusedMoEActivationFormat.Standard

        @staticmethod
        def _supports_current_device():
            return True

        @staticmethod
        def _supports_no_act_and_mul():
            return False

        @staticmethod
        def _supports_quant_scheme(weight_key, activation_key):
            return True

        @staticmethod
        def _supports_activation(activation):
            return True

        @staticmethod
        def _supports_parallel_config(moe_parallel_config):
            return True

        def supports_chunking(self):
            return False

        def supports_expert_map(self):
            return True

        def workspace_shapes(self, M, N, K, topk, global_num_experts, local_num_experts, expert_tokens_meta, activation):
            return ((M, 0), (M, 0), (M, K))

        def finalize_weight_and_reduce_impl(self):
            return TopKWeightAndReduceNoOP()

        def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation, global_num_experts, expert_map, a1q_scale, a2_scale, workspace13, workspace2, expert_tokens_meta, apply_router_weight_on_input):
            output.copy_(self.lb(hidden_states, topk_weights.contiguous(), topk_ids.contiguous(), expert_map))
            return None

    class _RefExperts:

        def __init__(self, pk, loc, E, dev):
            W1, W2 = ([], [])
            for g in loc:
                ex = pk[pk['experts'][g]]
                ex = {k: {kk: vv.to(dev) for kk, vv in v.items()} if isinstance(v, dict) else v.to(dev) if torch.is_tensor(v) else v for k, v in ex.items()}
                W1.append(lk.dequant_expert(ex, 'gu').bfloat16())
                W2.append(lk.dequant_expert(ex, 'd').bfloat16())
            self.W1, self.W2, self.E = (torch.stack(W1), torch.stack(W2), E)
            self.gu = None

        def nbytes(self):
            return self.W1.numel() * 2 + self.W2.numel() * 2

        def __call__(self, x, w, ids, expert_map):
            from vllm.model_executor.layers.fused_moe import fused_experts
            return fused_experts(x, self.W1, self.W2, w, ids, global_num_experts=self.E, expert_map=expert_map)
    global _LOWBIT_EXPERTS
    _LOWBIT_EXPERTS = _LowbitExperts
    _CLS = LowbitMoEMethod
    return _CLS

def _install_padding_guard():
    import importlib
    for mod in ('vllm.model_executor.layers.fused_moe.router.fused_topk_router', 'vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router'):
        try:
            m = importlib.import_module(mod)
        except Exception:
            continue
        orig = getattr(m, '_get_padding_mask', None)
        if orig is None or getattr(orig, '_fn_guard', False):
            continue

        def guarded(num_tokens, _orig=orig):
            mask = _orig(num_tokens)
            if mask is not None and mask.shape[0] < num_tokens:
                return None
            return mask
        guarded._fn_guard = True
        m._get_padding_mask = guarded

def _install_loader():
    import torch
    from vllm.model_executor.model_loader import default_loader as dl
    gaw = dl.DefaultModelLoader.get_all_weights
    norms = {}
    nf = next((os.path.join(d, 'norms.safetensors') for d in reversed(_ST['deltas']) if os.path.exists(os.path.join(d, 'norms.safetensors'))), os.path.join(_ST['dir'], 'norms.safetensors'))
    if os.path.exists(nf):
        from safetensors.torch import load_file
        norms = load_file(nf)
        _log(f'norm overlay {nf}: {len(norms)} tensors')

    def get_all_weights(self, *a, **k):
        st = {'drop': 0, 'gate': 0, 'norm': 0}
        for name, t in gaw(self, *a, **k):
            if name in norms:
                assert norms[name].shape == t.shape, (name, norms[name].shape, t.shape)
                st['norm'] += 1
                yield (name, norms[name].to(t.dtype))
                continue
            m = EXP.match(name)
            if m and _served(_row(m)):
                st['drop'] += 1
                continue
            g = GATE.match(name)
            if g and _served(_row(g)):
                r = torch.load(router_path(_row(g)), map_location='cpu')
                t = r['weight'].to(t.dtype).contiguous()
                st['gate'] += 1
            yield (name, t)
        _log(f'loader: dropped {st['drop']} expert tensors, {st['gate']} routers sliced, {st['norm']} norms overlaid')
    dl.DefaultModelLoader.get_all_weights = get_all_weights
