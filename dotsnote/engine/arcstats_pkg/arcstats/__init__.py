"""vLLM plugin: accumulate per-layer MoE routing stats (count, gate-weight sum) when ARCSTATS_DIR is set.
Requires --enforce-eager (python code does not run inside CUDA graphs)."""
import os, atexit


def register():
    out = os.environ.get("ARCSTATS_DIR")
    if not out:
        return
    import torch
    from vllm.model_executor.models import deepseek_v2 as dv2

    orig_init, orig_fwd = dv2.DeepseekV2MoE.__init__, dv2.DeepseekV2MoE.forward
    registry = {}

    def init(self, *a, **kw):
        orig_init(self, *a, **kw)
        self._arc_prefix = kw.get("prefix", "")
        self._arc_k = a[0].num_experts_per_tok if a else kw["config"].num_experts_per_tok
        registry[self._arc_prefix] = self

    state = {"calls": 0}

    def dump():
        if not registry:
            return
        try:
            from vllm.distributed import get_tensor_model_parallel_rank
            if get_tensor_model_parallel_rank() != 0:
                return
        except Exception:
            pass
        d = {p: {"count": m._arc_cnt.cpu(), "wsum": m._arc_w.cpu(), "tokens": m._arc_tok}
             for p, m in registry.items() if hasattr(m, "_arc_cnt")}
        os.makedirs(out, exist_ok=True)
        torch.save(d, os.path.join(out, f"stats_{os.getpid()}.pt"))

    @torch.no_grad()
    def fwd(self, hidden_states, already_sequence_parallel=False):
        x = hidden_states.view(-1, hidden_states.shape[-1])
        logits = self.gate(x)[0].float()
        s = logits.sigmoid()
        b = self.gate.e_score_correction_bias
        sel = torch.topk(s + (b.float() if b is not None else 0), self._arc_k, dim=-1).indices
        w = s.gather(1, sel)
        w = w / w.sum(-1, keepdim=True)
        if not hasattr(self, "_arc_cnt"):
            self._arc_cnt = torch.zeros(s.shape[1], dtype=torch.float64, device=s.device)
            self._arc_w = torch.zeros_like(self._arc_cnt)
            self._arc_tok = 0
        self._arc_cnt.index_add_(0, sel.flatten(), torch.ones_like(w.flatten(), dtype=torch.float64))
        self._arc_w.index_add_(0, sel.flatten(), w.flatten().double())
        self._arc_tok += x.shape[0]
        state["calls"] += 1
        if state["calls"] % 450 == 0:
            dump()
        return orig_fwd(self, hidden_states, already_sequence_parallel)

    dv2.DeepseekV2MoE.__init__ = init
    print(f"[arcstats] routing stats enabled -> {out}", flush=True)
    dv2.DeepseekV2MoE.forward = fwd
    atexit.register(dump)


# ---------------------------------------------------------------- Dots3 dense-attention patch (ARC_DOTS3_DENSE=1)
# Runs the 13 DSA (sparse top-2048) layers as dense MLA on the portable Triton backend, skipping the
# indexer. This is what an L4 (SM89) deployment must do, since vLLM's sparse-MLA kernels need SM90+.
def _dense_patch():
    import torch
    from vllm.models.dots3_note.nvidia import model as dm

    class _FullView:
        def __init__(self, c):
            self._c = c
            self._o = {
                "swa_num_attention_heads": c.num_attention_heads,
                "swa_qk_nope_head_dim": c.qk_nope_head_dim,
                "swa_qk_rope_head_dim": c.qk_rope_head_dim,
                "swa_v_head_dim": c.v_head_dim,
                "swa_q_lora_rank": c.q_lora_rank,
                "swa_kv_lora_rank": c.kv_lora_rank,
                "swa_attention_gate_type": c.attention_gate_type,
                "swa_rope_theta": c.rope_parameters["rope_theta"],
                "sliding_window_size": None,
            }

        def __getattr__(self, k):
            o = object.__getattribute__(self, "_o")
            return o[k] if k in o else getattr(object.__getattribute__(self, "_c"), k)

    from dataclasses import replace
    from vllm.v1.attention.backends.mla.triton_mla import TritonMLABackend, TritonMLAImpl

    PHYS = config_phys = None  # set lazily from the hf config (swa_kv_lora_rank + swa_qk_rope_head_dim)

    from vllm.model_executor.layers.attention.mla_attention import QueryLenSupport
    from vllm.v1.attention.backend import AttentionCGSupport
    from vllm.v1.attention.backends.mla import triton_mla as _tm

    class SpecDenseBuilder(_tm.TritonMLAMetadataBuilder):
        """Lets causal multi-token (spec-verify) queries stay on the decode path."""
        query_len_support = QueryLenSupport.UNIFORM
        _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

        @classmethod
        def get_cudagraph_support(cls, vllm_config, kv_cache_spec):
            return cls._cudagraph_support

        def _reserve_attn_logits_workspace(self):
            if not _tm.is_workspace_manager_initialized():
                return
            B = self.vllm_config.scheduler_config.max_num_seqs * max(1, self.reorder_batch_threshold)
            max_splits = _tm._compute_num_kv_splits(self.model_config.max_model_len,
                                                    _tm.current_platform.num_compute_units())
            _tm.current_workspace_manager().get_simultaneous(
                ((B, self.num_heads * self.dcp_world_size, max_splits, self.mla_dims.kv_lora_rank + 1),
                 torch.float32),)

    import dataclasses as _dc

    def _causal_multi_mqa(impl, q, kv_cache, attn_metadata, layer):
        """Spec-verify (ql>1 causal tokens per request) via the stock single-token decode path:
        each draft token becomes its own decode row with causal length seq_len-(ql-1-i)."""
        nd, ndt = attn_metadata.num_decodes, attn_metadata.num_decode_tokens
        ql = ndt // max(1, nd)
        if ql <= 1:
            return TritonMLAImpl.forward_mqa(impl, q, kv_cache, attn_metadata, layer)
        dec = attn_metadata.decode
        bt, sl = dec.block_table, dec.seq_lens
        n_req = min(nd, bt.shape[0], sl.shape[0])
        off = torch.arange(ql - 1, -1, -1, device=sl.device, dtype=sl.dtype)
        bt2 = bt[:n_req].repeat_interleave(ql, dim=0)
        sl2 = (sl[:n_req].unsqueeze(1) - off.unsqueeze(0)).reshape(-1).clamp_min(0)
        md = _dc.replace(attn_metadata, num_decodes=n_req * ql,
                         decode=_dc.replace(dec, block_table=bt2, seq_lens=sl2))
        return TritonMLAImpl.forward_mqa(impl, q, kv_cache, md, layer)

    class PaddedDenseImpl(TritonMLAImpl):
        """Dense Triton MLA over cache rows padded to the SWA row width (uniform page size)."""

        def _v(self, c):
            return c[..., : self.kv_lora_rank + self.qk_rope_head_dim]

        def do_kv_cache_update(self, kv_c_normed, k_pe, kv_cache, slot_mapping, kv_cache_dtype, k_scale):
            return super().do_kv_cache_update(kv_c_normed, k_pe, self._v(kv_cache), slot_mapping,
                                              kv_cache_dtype, k_scale)

        def forward_mha(self, q, kv_c_normed, k_pe, kv_c_and_k_pe_cache, attn_metadata, k_scale, output,
                        output_scale=None):
            return super().forward_mha(q, kv_c_normed, k_pe, self._v(kv_c_and_k_pe_cache), attn_metadata,
                                       k_scale, output, output_scale)

        def forward_mqa(self, q, kv_c_and_k_pe_cache, attn_metadata, layer):
            return _causal_multi_mqa(self, q, self._v(kv_c_and_k_pe_cache), attn_metadata, layer)

    class PaddedDenseBackend(TritonMLABackend):
        @classmethod
        def supported_kv_cache_layouts(cls):
            return dm.Dots3NoteTritonMLABackend.supported_kv_cache_layouts()

        @staticmethod
        def get_name():
            return "DOTS3_PADDED_DENSE_TRITON_MLA"

        @staticmethod
        def get_impl_cls():
            return PaddedDenseImpl

        @staticmethod
        def get_builder_cls():
            return SpecDenseBuilder

    from vllm.model_executor.layers.attention.mla_attention import MLAAttention as _MLA

    unpad = os.environ.get("ARC_DOTS3_UNPAD") == "1"

    def make_padded_mla(phys):
        class PaddedDenseMLA(_MLA):
            def __init__(self, *a, **kw):
                kw["attn_backend"] = UnpaddedDenseBackend if unpad else PaddedDenseBackend
                kw.pop("prefill_backend_cls", None)
                super().__init__(*a, **kw)

            def get_kv_cache_spec(self, vllm_config):
                spec = super().get_kv_cache_spec(vllm_config)
                if not unpad:
                    return replace(spec, head_size=phys)
                # unpadded rows (576 x kv dtype); grow the block so one page ~= the SWA page
                # (block_size x 1088 x bf16) and pad the remainder, so pages unify without
                # storing 1088-wide rows for the dense layers.
                row = spec.page_size_bytes // spec.block_size
                target = spec.block_size * phys * 2
                bs = (target // row) // 16 * 16
                return replace(spec, block_size=bs, page_size_padded=target)
        return PaddedDenseMLA

    class UnpaddedDenseBackend(TritonMLABackend):
        @classmethod
        def supported_kv_cache_layouts(cls):
            return dm.Dots3NoteTritonMLABackend.supported_kv_cache_layouts()

        @staticmethod
        def get_name():
            return "DOTS3_UNPADDED_DENSE_TRITON_MLA"

        @staticmethod
        def get_impl_cls():
            return UnpaddedDenseImpl

        @staticmethod
        def get_builder_cls():
            return SpecDenseBuilder

    class UnpaddedDenseImpl(TritonMLAImpl):
        def forward_mqa(self, q, kv_c_and_k_pe_cache, attn_metadata, layer):
            return _causal_multi_mqa(self, q, kv_c_and_k_pe_cache, attn_metadata, layer)

    class DenseFull(dm.Dots3NoteSlidingAttention):
        def __init__(self, vllm_config, config, prefix, topk_indices_buffer=None):
            phys = config.swa_kv_lora_rank + config.swa_qk_rope_head_dim
            saved = dm.MLAAttention
            dm.MLAAttention = make_padded_mla(phys)
            try:
                super().__init__(vllm_config=vllm_config, config=_FullView(config), prefix=prefix,
                                 topk_indices_buffer=topk_indices_buffer)
            finally:
                dm.MLAAttention = saved

    dm.Dots3NoteFullAttention = DenseFull
    orig_adapt = dm.Dots3NoteModel._adapt_weights

    def adapt(self, weights):
        return orig_adapt(self, ((n, w) for n, w in weights if ".indexer." not in n))

    dm.Dots3NoteModel._adapt_weights = adapt


_orig_register = register


def _emulate_sm89():
    """ARC_EMULATE_SM89=1: make vLLM see compute capability 8.9 / "NVIDIA L4" on the dev GPUs so backend selection and
    kernel dispatch follow the Kaggle-L4 code paths (Triton still compiles real sm90 code, so everything runs)."""
    import torch
    from vllm.platforms.interface import DeviceCapability
    from vllm.platforms import cuda as pc
    cap = DeviceCapability(major=8, minor=9)
    seen = set()

    def _cap(c, device_id=0):
        if os.environ.get("ARC_EMULATE_SM89_LOG") == "1":
            import traceback
            fr = traceback.extract_stack(limit=4)[0]
            key = (fr.filename, fr.lineno)
            if key not in seen:
                seen.add(key)
                print(f"[sm89emu] capability query from {fr.filename.split('site-packages/')[-1]}:{fr.lineno} {fr.line}", flush=True)
        return cap
    for cls in (pc.NvmlCudaPlatform, pc.NonNvmlCudaPlatform):
        cls.get_device_capability = classmethod(_cap)
        cls.get_device_name = classmethod(lambda c, device_id=0: "NVIDIA L4")
    # (torch.cuda.get_device_capability is left alone: Triton derives its compile target from it)


def _share_mla_workspaces():
    """Every MLA metadata builder (one per KV-cache group: 47 for Dots3 with offload) allocates its own chunked-
    prefill workspace (64k tokens x latent width, ~136 MiB) outside vLLM's KV budget: ~6 GB, fatal on a 22 GB L4.
    Layers run one after another, so a single shared workspace per (shape, dtype, device) is enough."""
    import torch
    from vllm.model_executor.layers.attention import mla_attention as ma
    shared = {}
    orig = ma.MLACommonMetadataBuilder.__init__

    def init(self, *a, **kw):
        orig(self, *a, **kw)
        ws = getattr(self, "chunked_prefill_workspace", None)
        if isinstance(ws, torch.Tensor) and getattr(self, "dcp_world_size", 1) == 1:
            key = (tuple(ws.shape), ws.dtype, str(ws.device))
            if key not in shared:
                shared[key] = ws
            self.chunked_prefill_workspace = shared[key]
    ma.MLACommonMetadataBuilder.__init__ = init


def _emulate_gpu_memory(gib):
    """ARC_EMULATE_GPU_GIB=22.03: cap torch allocations per process at a real L4's capacity, so allocations outside
    vLLM's budget fail here as they would on Kaggle (vLLM's own budget still comes from --gpu-memory-utilization)."""
    import torch
    for d in range(torch.cuda.device_count()):
        tot = torch.cuda.get_device_properties(d).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, gib * 2**30 / tot), d)


def register():  # noqa: F811
    if os.environ.get("ARC_EMULATE_SM89") == "1":
        _emulate_sm89()
    if float(os.environ.get("ARC_EMULATE_GPU_GIB") or 0) > 0:
        from vllm.v1.worker.gpu import model_runner as _mr   # workers only: no CUDA init in the API server
        _orig_lm = _mr.GPUModelRunner.load_model

        def _lm(self, *a, **kw):
            _emulate_gpu_memory(float(os.environ["ARC_EMULATE_GPU_GIB"]))
            return _orig_lm(self, *a, **kw)
        _mr.GPUModelRunner.load_model = _lm
    _orig_register()
    if int(os.environ.get("ARC_FAKE_TP") or 0) > 1:   # local memory debugging: one process = rank 0 of a fake TP group
        from .fake_tp import install as _fake_tp_install
        _fake_tp_install()
    if os.environ.get("ARC_DOTS3_PORTABLE_DSA") == "1":
        _share_mla_workspaces()
    if os.environ.get("ARC_DOTS3_DENSE") == "1":
        _dense_patch()
    if os.environ.get("ARC_DOTS3_PORTABLE_DSA") == "1":
        from .dsa_sm89 import install
        install()
    if os.environ.get("ARC_EMBED_HOST") == "1":
        _embed_host_patch()
    if os.environ.get("ARC_ROPE_TRIM") == "1":
        _rope_trim_patch()
    _moe_patch()
    _lowbit_load_patch()
    if os.environ.get("ARC_DELTA_FILE"):   # in-format fine-tune overlay (norms/router in the weight stream)
        from .delta_overlay import install as _delta_install
        _delta_install()
    if os.environ.get("ARC_DUMP_ATTN"):
        _attn_dump_patch()
    if os.environ.get("ARC_MEMREPORT") == "1":
        _memreport_patch()
    if float(os.environ.get("ARC_MEMCENSUS_S") or 0) > 0:
        _memcensus_patch(float(os.environ["ARC_MEMCENSUS_S"]))
    if (os.environ.get("ARC_ALLOC_TRIM") == "1" or float(os.environ.get("ARC_MEMCENSUS_S") or 0) > 0
            or os.environ.get("ARC_MEMHIST")):
        _alloc_phase_patch()
    if os.environ.get("ARC_FORCE_FP8_MARLIN") == "1" or os.environ.get("ARC_FP8_W8A8") == "1":   # W8A8 is forced there too
        _force_fp8_marlin()
    if os.environ.get("ARC_LOGIT_BIAS"):
        _logit_bias_patch(os.environ["ARC_LOGIT_BIAS"])
    if os.environ.get("ARC_HOST_AR") == "1":
        _host_ar_patch()
        if os.environ.get("ARC_AR_LB") == "1":   # sub-int6 all-reduce payloads / error feedback
            from . import ar_lowbit
            ar_lowbit.install()
    if os.environ.get("ARC_SWA_FUSED") in ("1", "2"):
        from . import swa_fused
        swa_fused.install()
    if os.environ.get("ARC_DRAFT_VOCAB"):
        from . import draft_head
        draft_head.install()
    if os.environ.get("ARC_SESSION_EP") == "1":
        from . import session_ep
        session_ep.install()
    if os.environ.get("ARC_LMHEAD_FP8") == "1":
        from . import lmhead_fp8
        lmhead_fp8.install()
    if os.environ.get("ARC_MLA_FP8") == "1":
        from . import mla_fp8
        mla_fp8.install()
    if os.environ.get("ARC_MLA_QCAT") == "1":   # MLA decode query concat in place (bit-identical)
        from . import qcat
        qcat.install()
    if os.environ.get("ARC_GATE_MERGE") == "1":   # g_proj rows appended to fused_qkv_a_proj (bit-identical)
        from . import gmerge
        gmerge.install()
    if os.environ.get("ARC_SHARD_SAMPLE") == "1":   # verify sampling on the vocab shard (no logits all-gather)
        from . import shard_sample
        shard_sample.install()
    if os.environ.get("ARC_STEPLOG"):   # per-step draft/acceptance log from the scheduler (analysis only)
        from . import steplog
        steplog.install()
    if os.environ.get("ARC_MEMCAP") == "1":   # debug (lever 6): memory after each captured CUDA-graph size, rank 0
        _memcap_patch()
    if os.environ.get("ARC_PENALTY_LAZY") == "1" or os.environ.get("ARC_MLA_WS_TOKENS"):   # lever 6 reclaims
        from . import mem_reclaim
        mem_reclaim.install()
    if os.environ.get("ARC_KVB_REBUILD") == "1":   # free kv_b_proj FP8 weights, rebuild bit-exactly on use (lever 6)
        from . import kvb_rebuild
        kvb_rebuild.install()
    if os.environ.get("ARC_PARAM_COMPACT") == "1":   # lever 6: re-pack weights after load (outermost load wrapper)
        from . import mem_reclaim
        mem_reclaim.install_late()


def _host_ar_patch():
    """Route TP all-reduces (bf16, <= ARC_HOST_AR_MAX_MB) through arcstats.host_ar (shared pinned host memory) instead of
    NCCL's SHM ring: PCIe GPUs without P2P (L4) where NCCL picks Ring/LL."""
    import torch
    from vllm.distributed.device_communicators import cuda_communicator as cc
    from vllm.logger import init_logger
    log = init_logger("vllm.arcstats.host_ar")
    orig_init, orig_ar = cc.CudaCommunicator.__init__, cc.CudaCommunicator.all_reduce

    def init(self, *a, **kw):
        orig_init(self, *a, **kw)
        self._arc_har = None
        if self.world_size in (2, 4, 8) and self.unique_name.split(":")[0] == "tp":
            from .host_ar import HostAllReduce
            name = f"{os.getppid()}_{self.unique_name.replace(':', '_')}"
            self._arc_har = HostAllReduce(self.rank_in_group, self.world_size, name, self.cpu_group)
            from . import host_ar
            host_ar._INSTANCES.setdefault("tp", self._arc_har)
            log.info("host all-reduce enabled for %s (world %d, %.0f MB max)", self.unique_name, self.world_size,
                     self._arc_har.maxn * 2 / 2**20)

    def all_reduce(self, input_):
        h = self._arc_har
        if h is not None and input_.is_cuda and h.supports(input_):
            return h(input_)
        return orig_ar(self, input_)
    cc.CudaCommunicator.__init__ = init
    cc.CudaCommunicator.all_reduce = all_reduce


def _force_fp8_marlin():
    """Use the weight-only FP8 Marlin kernel for FP8 linears, as vLLM does on a real L4 (SM89) -- the H100 (and the
    SM89 platform emulation) would pick CUTLASS FP8 instead, hiding L4-only code paths.
    ARC_FP8_W8A8=1 forces vLLM's Triton W8A8 block-FP8 GEMM instead (activations quantized per 1x128 group, FP8
    tensor-core dot: the checkpoint's native dynamic-activation numerics), with tile configs from ARC_W8A8_CONFIG_DIR
    (files N=..,K=..,device_name=ANY,...json from dots3/tune_w8a8.py)."""
    from vllm.model_executor.kernels import linear as kl
    orig = kl.choose_scaled_mm_linear_kernel
    w8a8 = os.environ.get("ARC_FP8_W8A8") == "1"
    if w8a8:
        _w8a8_configs()

    def choose(config, possible_kernels, compute_capability=None, force_kernel=None, *, quantization):
        if force_kernel is None and quantization == "fp8_block_w8a8":   # block-FP8 only (lb* checkpoints)
            force_kernel = kl.TritonFp8BlockScaledMMKernel if w8a8 else kl.MarlinFP8ScaledMMLinearKernel
        return orig(config, possible_kernels, compute_capability, force_kernel, quantization=quantization)
    kl.choose_scaled_mm_linear_kernel = choose
    if w8a8 and os.environ.get("ARC_W8A8_WS") == "1":   # decode-sized GEMM per (N, K, M) table (run_l4 w8a8_ws_gate)
        from .w8a8_ws import install as _ws_install
        _ws_install()
    if w8a8 and os.environ.get("ARC_W8A8_V2") == "1":   # bit-identical block GEMM, per-(N, K, M) L4 tiles
        from .w8a8_v2 import install as _v2_install
        _v2_install()


def _w8a8_configs():
    """tile configs for the W8A8 block GEMM from ARC_W8A8_CONFIG_DIR (any device name); logs each new (N, K) once"""
    import glob, json
    from vllm.model_executor.layers.quantization.utils import fp8_utils as fu
    d = os.environ.get("ARC_W8A8_CONFIG_DIR", "")
    table = {}
    for f in glob.glob(os.path.join(d, "N=*,K=*,*.json")) if d else []:
        b = os.path.basename(f).split(",")
        table[(int(b[0][2:]), int(b[1][2:]))] = {int(k): v for k, v in json.load(open(f)).items()}
    orig = fu.get_w8a8_block_fp8_configs
    seen = set()

    def get(N, K, block_n, block_k):
        if (N, K) not in seen:
            seen.add((N, K))
            print(f"[w8a8] shape N={N} K={K} tuned={(N, K) in table}", flush=True)
        return table.get((N, K)) or orig(N, K, block_n, block_k)
    fu.get_w8a8_block_fp8_configs = get
    print(f"[w8a8] Triton W8A8 block-FP8 linears, {len(table)} tuned shapes from {d or '-'}", flush=True)


def _logit_bias_patch(path):
    """Server-wide additive logit bias {token_id: bias} from a JSON file (re-read when it changes), applied in
    LogitsProcessor so target and MTP draft see the same distribution -- vLLM rejects per-request logit_bias with
    speculative decoding. Used to penalize hesitation tokens (" Wait", " Alternatively", ...) in long reasoning."""
    import torch
    from vllm.model_executor.layers import logits_processor as lp
    orig = lp.LogitsProcessor.forward
    state = {"mtime": None, "dev": {}}

    def current(device):
        try:
            m = os.path.getmtime(path)
        except OSError:
            return None
        if m != state["mtime"]:
            import json
            d = {int(k): float(v) for k, v in json.load(open(path)).items()}
            state.update(mtime=m, ids=list(d), vals=list(d.values()), dev={})
        if not state["ids"]:
            return None
        if device not in state["dev"]:
            state["dev"][device] = (torch.tensor(state["ids"], device=device),
                                    torch.tensor(state["vals"], device=device))
        return state["dev"][device]

    def forward(self, *a, **kw):
        logits = orig(self, *a, **kw)
        if logits is not None:
            b = current(logits.device)
            if b is not None:
                logits[..., b[0]] += b[1].to(logits.dtype)
        return logits
    lp.LogitsProcessor.forward = forward


def _lowbit_load_patch():
    """After vLLM's weight load, materialize ArcLowbitExperts (packed low-bit routed experts) on the GPU."""
    from vllm.v1.worker.gpu import model_runner as mr
    orig = mr.GPUModelRunner.load_model

    def load_model(self, *a, **kw):
        r = orig(self, *a, **kw)
        import torch
        from .lowbit_moe import ArcLowbitExperts
        n = 0
        dev = torch.device("cuda", torch.cuda.current_device())
        for mod in self.model.modules():
            if isinstance(mod, ArcLowbitExperts) and mod.moe is None:
                n += mod.load(dev)
        if n and os.environ.get("ARC_LOWBIT_COMPACT") == "1":
            _compact_lowbit(self.model, ArcLowbitExperts)
        if n:
            torch.cuda.empty_cache()
            if hasattr(self, "model_memory_usage"):
                self.model_memory_usage += n
            print(f"[arc] loaded {n / 2**30:.2f} GiB of low-bit routed experts", flush=True)
        return r
    mr.GPUModelRunner.load_model = load_model


def _compact_lowbit(model, cls):
    """ARC_LOWBIT_COMPACT=1: the expert build (GPU slicing + pair conversion) interleaves its temporaries with the kept
    tensors, leaving ~1 GiB/GPU of holes between them that empty_cache cannot return. Round-trip the kept tensors
    through host RAM (one TP rank at a time, ~12.7 GiB host transient) so they are re-allocated back to back."""
    import fcntl, gc, time
    import torch
    t0 = time.time()
    objs = []
    for mod in model.modules():
        if isinstance(mod, cls) and mod.moe is not None:
            objs.append(mod.moe)
            objs += [v for v in vars(mod.moe).values() if hasattr(v, "__dict__") and not isinstance(v, (torch.Tensor, torch.nn.Module))]
    st = torch.cuda.memory_stats()
    before = st["reserved_bytes.all.current"] - st["allocated_bytes.all.current"]
    with open("/tmp/arc_lowbit_compact.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        host = []
        for o in objs:
            for k, v in list(vars(o).items()):
                if isinstance(v, torch.Tensor) and v.is_cuda:
                    host.append((o, k, v.cpu()))
                    setattr(o, k, None)
        gc.collect()
        torch.cuda.empty_cache()
        for o, k, h in sorted(host, key=lambda x: -x[2].nbytes):   # largest first
            setattr(o, k, h.to("cuda"))
        del host
        torch.cuda.synchronize()
        fcntl.flock(lk, fcntl.LOCK_UN)
    torch.cuda.empty_cache()
    st = torch.cuda.memory_stats()
    after = st["reserved_bytes.all.current"] - st["allocated_bytes.all.current"]
    print(f"[arc] low-bit experts compacted in {time.time() - t0:.1f} s: cached-free {before / 2**30:.3f} -> "
          f"{after / 2**30:.3f} GiB", flush=True)


def _embed_host_patch():
    """Vocab embeddings (main + MTP head, 2 x 1.45 GiB BF16) live in host memory; lookups gather rows over UVA."""
    from vllm.v1.worker.gpu import model_runner as mr
    orig = mr.GPUModelRunner.load_model

    def load_model(self, *a, **kw):
        r = orig(self, *a, **kw)
        from .sparse_offload import host_embeddings
        n = host_embeddings(self.model)
        sp = getattr(self, "speculator", None)
        if sp is not None and getattr(sp, "model", None) is not None:
            n += host_embeddings(sp.model)
        # vLLM budgets KV from the load-time weight footprint; credit the bytes that left the GPU
        if hasattr(self, "model_memory_usage"):
            self.model_memory_usage = max(0, self.model_memory_usage - n)
        print(f"[arc] moved {n / 2**30:.2f} GiB of embeddings to host", flush=True)
        return r
    mr.GPUModelRunner.load_model = load_model


def _rope_trim_patch():
    """ARC_ROPE_TRIM=1: rotary cos/sin caches are built for config.max_position_embeddings (524288 for Dots3) while
    positions never reach --max-model-len (262144): keep max_model_len + 1024 rows (shared caches stay shared)."""
    from vllm.v1.worker.gpu import model_runner as mr
    orig = mr.GPUModelRunner.load_model

    def load_model(self, *a, **kw):
        r = orig(self, *a, **kw)
        import torch
        keep = int(self.vllm_config.model_config.max_model_len) + 1024
        mods = [self.model]
        sp = getattr(self, "speculator", None)
        if sp is not None and getattr(sp, "model", None) is not None:
            mods.append(sp.model)
        new, freed = {}, 0
        for m in mods:
            for mod in m.modules():
                for name in ("cos_sin_cache", "cos_sin_cache_bf16"):
                    t = getattr(mod, name, None)
                    if not isinstance(t, torch.Tensor) or not t.is_cuda or t.dim() < 1 or t.shape[0] <= keep:
                        continue
                    k = t.data_ptr()
                    if k not in new:
                        new[k] = t[:keep].clone()
                        freed += t.untyped_storage().nbytes() - new[k].untyped_storage().nbytes()
                    setattr(mod, name, new[k])
        torch.cuda.empty_cache()
        print(f"[arc] rope caches trimmed to {keep} rows: {len(new)} cache(s), {freed / 2**20:.0f} MiB freed", flush=True)
        return r
    mr.GPUModelRunner.load_model = load_model


def _memcensus_patch(after_s):
    """Debug: ARC_MEMCENSUS_S seconds after load (i.e. under real decode load), rank 0 prints the device memory
    budget (driver used / torch reserved / allocated / peak) and the largest live CUDA tensors grouped by shape."""
    import collections, gc, threading, time
    import torch
    from vllm.v1.worker.gpu import model_runner as mr
    orig = mr.GPUModelRunner.load_model

    def census(dev):
        torch.cuda.set_device(dev)
        time.sleep(after_s)
        free, total = torch.cuda.mem_get_info(dev)
        st = torch.cuda.memory_stats(dev)
        g = 2 ** 30
        print(f"[memcensus] used {(total - free) / g:.2f} / {total / g:.2f} GiB; torch reserved "
              f"{st.get('reserved_bytes.all.current', 0) / g:.2f} (peak {st.get('reserved_bytes.all.peak', 0) / g:.2f}), "
              f"allocated {st.get('allocated_bytes.all.current', 0) / g:.2f} (peak {st.get('allocated_bytes.all.peak', 0) / g:.2f}); "
              f"non-torch {(total - free - st.get('reserved_bytes.all.current', 0)) / g:.2f}", flush=True)
        agg, seen = collections.Counter(), set()
        for o in gc.get_objects():
            try:
                if isinstance(o, torch.Tensor) and o.is_cuda:
                    s_ = o.untyped_storage()
                    if s_.data_ptr() in seen:
                        continue
                    seen.add(s_.data_ptr())
                    agg[(tuple(o.shape), str(o.dtype).replace("torch.", ""))] += s_.nbytes()
            except Exception:
                pass
        try:                                              # every allocator block, by size: weights, caches, pools
            segs = torch.cuda.memory_snapshot()
            act, inact, pool = collections.Counter(), 0, collections.Counter()
            for sg in segs:
                if sg.get("device") != dev:
                    continue
                pool[str(sg.get("segment_pool_id", "")) or "default"] += sg["total_size"]
                for b in sg["blocks"]:
                    if b["state"] == "active_allocated":
                        act[b["size"]] += 1
                    else:
                        inact += b["size"]
            print(f"[memcensus] segments {sum(pool.values()) / g:.2f} GiB, inactive (cached free) {inact / g:.2f} GiB; "
                  f"pools {[(k, round(v / g, 2)) for k, v in pool.most_common(6)]}", flush=True)
            for sz, n in sorted(act.items(), key=lambda kv: -kv[0] * kv[1])[:30]:
                print(f"[memcensus]   active {n:5d} x {sz / 2**20:9.2f} MiB = {n * sz / 2**20:9.1f} MiB", flush=True)
            _census_free_view(segs, dev)
            if os.environ.get("ARC_MEMHIST"):
                _census_segment_allocs(dev)
        except Exception as ex:
            print("[memcensus] snapshot failed", ex, flush=True)
        _memsnap_dump("census", dev)
        print(f"[memcensus] python-visible CUDA storages {sum(agg.values()) / g:.2f} GiB in {len(seen)}", flush=True)
        for k, v in agg.most_common(40):
            print(f"[memcensus]   {v / 2**20:9.1f} MiB  {k}", flush=True)
        if os.environ.get("ARC_MEMCENSUS_TRIM") == "1":   # probe: how much empty_cache returns, and how fast it comes back
            _mem_line("[memcensus] before trim", dev)
            torch.cuda.empty_cache()
            _mem_line("[memcensus] after trim", dev)
            dt = float(os.environ.get("ARC_MEMCENSUS_TRIM_S") or 120)
            for i in range(3):
                time.sleep(dt)
                _mem_line(f"[memcensus] trim +{(i + 1) * dt:.0f} s", dev)
                if os.environ.get("ARC_MEMHIST_KEEP") == "1":
                    _census_segment_allocs(dev)

    def load_model(self, *a, **kw):
        r = orig(self, *a, **kw)
        try:
            from vllm.distributed import get_tensor_model_parallel_rank
            if get_tensor_model_parallel_rank() == 0:
                threading.Thread(target=census, args=(torch.cuda.current_device(),), daemon=True).start()
        except Exception as ex:
            print("[memcensus] not started", ex, flush=True)
        return r
    mr.GPUModelRunner.load_model = load_model


def _memcap_patch():
    """ARC_MEMCAP=1 (debug, lever 6): rank 0 prints driver free / torch reserved before each CUDA-graph capture
    descriptor, so graph-exec (non-torch) and graph-pool memory can be attributed per capture size."""
    import torch
    from vllm.v1.worker.gpu import cudagraph_utils as cu
    orig = cu.CudaGraphManager.capture

    def capture(self, create_forward_fn, *a, **kw):
        def cff(desc, *x, **y):
            try:
                from vllm.distributed import get_tensor_model_parallel_rank
                if get_tensor_model_parallel_rank() == 0 and y.get("warmup", x[0] if x else True):
                    torch.cuda.synchronize()
                    _mem_line(f"[memcap] before {getattr(desc, 'num_tokens', desc)} {getattr(desc, 'cg_mode', '')}")
            except Exception:
                pass
            return create_forward_fn(desc, *x, **y)
        r = orig(self, cff, *a, **kw)
        try:
            from vllm.distributed import get_tensor_model_parallel_rank
            if get_tensor_model_parallel_rank() == 0:
                torch.cuda.synchronize()
                _mem_line(f"[memcap] end of capture ({len(self.graphs)} full graphs)")
        except Exception:
            pass
        return r
    cu.CudaGraphManager.capture = capture


def _memsnap_dump(tag, dev=None):
    """ARC_MEMSNAP=<dir> (debug, lever 6): pickle torch.cuda.memory._snapshot() (all segments/blocks, frames if
    ARC_MEMHIST recorded them) to <dir>/memsnap_<tag>.pkl for an offline per-site ledger."""
    d = os.environ.get("ARC_MEMSNAP")
    if not d:
        return
    try:
        import pickle
        import torch
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"memsnap_{tag}.pkl"), "wb") as f:
            pickle.dump(torch.cuda.memory._snapshot(), f)
        print(f"[memsnap] wrote {tag}", flush=True)
    except Exception as ex:
        print("[memsnap] failed", tag, repr(ex)[:200], flush=True)


def _mem_line(tag, dev=None):
    import torch
    dev = torch.cuda.current_device() if dev is None else dev
    free, total = torch.cuda.mem_get_info(dev)
    st = torch.cuda.memory_stats(dev)
    r, a = st.get("reserved_bytes.all.current", 0), st.get("allocated_bytes.all.current", 0)
    g = 2 ** 30
    print(f"{tag}: used {(total - free) / g:.3f} GiB, reserved {r / g:.3f} (peak {st.get('reserved_bytes.all.peak', 0) / g:.3f}), "
          f"allocated {a / g:.3f} (peak {st.get('allocated_bytes.all.peak', 0) / g:.3f}), cached-free {(r - a) / g:.3f}, "
          f"non-torch {(total - free - r) / g:.3f}", flush=True)


def _census_free_view(segs, dev):
    """Where the allocator's cached-free bytes sit: whole free segments (empty_cache returns them) vs holes inside
    segments that still hold live blocks (only expandable segments can return those), the free-block size histogram,
    and the top segments by free bytes with who allocated their live blocks (stack frames only if ARC_MEMHIST
    recorded the allocation)."""
    import collections
    g, m = 2 ** 30, 2 ** 20
    whole = holes = 0
    hist = collections.Counter()
    rows = []
    for sg in segs:
        if sg.get("device") != dev:
            continue
        blocks = sg["blocks"]
        fr = sum(b["size"] for b in blocks if b["state"] != "active_allocated")
        live = [b for b in blocks if b["state"] == "active_allocated"]
        for b in blocks:
            if b["state"] != "active_allocated":
                sz = b["size"] / m
                hist["<2M" if sz < 2 else "2-20M" if sz < 20 else "20-100M" if sz < 100 else "100-500M" if sz < 500 else ">=500M"] += b["size"]
        if not live:
            whole += fr
        else:
            holes += fr
        if fr:
            rows.append((fr, sg, live))
    print(f"[memcensus] cached-free: whole free segments {whole / g:.3f} GiB (empty_cache returns), holes in live segments "
          f"{holes / g:.3f} GiB; free blocks by size " + ", ".join(f"{k} {v / m:.0f} MiB" for k, v in sorted(hist.items())),
          flush=True)

    def who(b):
        fr = b.get("frames") or []
        for f in fr:
            fn = f.get("filename", "")
            if "arcstats" in fn or ("vllm" in fn and "torch/" not in fn):
                return f"{fn.split('site-packages/')[-1]}:{f.get('line')} {f.get('name')}"
        return "?" if not fr else f"{fr[0].get('filename', '').split('/')[-1]}:{fr[0].get('line')}"
    if any(b.get("frames") for sg in segs for b in sg["blocks"][:4]):   # live bytes by allocation site (ARC_MEMHIST)
        site = collections.Counter()
        cnt = collections.Counter()
        for sg in segs:
            if sg.get("device") == dev:
                for b in sg["blocks"]:
                    if b["state"] == "active_allocated":
                        site[who(b)] += b["size"]
                        cnt[who(b)] += 1
        for k, v in site.most_common(30):
            print(f"[memcensus]   live {v / m:8.1f} MiB in {cnt[k]:5d}  {k}", flush=True)
    for fr, sg, live in sorted(rows, key=lambda r: -r[0])[:12]:
        ls = sorted(live, key=lambda b: -b["size"])
        print(f"[memcensus]   seg {sg['total_size'] / m:8.1f} MiB pool {sg.get('segment_pool_id')} free {fr / m:7.1f} MiB "
              f"live {len(live)} ({sum(b['size'] for b in live) / m:.1f} MiB) e.g. "
              + "; ".join(f"{b['size'] / m:.1f} MiB {who(b)}" for b in ls[:3]), flush=True)


def _census_segment_allocs(dev):
    """With ARC_MEMHIST(+_KEEP): who made the allocator grow (cudaMalloc of new segments) during the recorded window,
    grouped by the first arcstats/vllm stack frame."""
    import collections
    import torch
    try:
        tr = torch.cuda.memory._snapshot().get("device_traces", [])
        tr = tr[dev] if dev < len(tr) else []
    except Exception as ex:
        print("[memcensus] no traces", ex, flush=True)
        return
    agg = collections.defaultdict(lambda: [0, 0, 0])
    for e in tr:
        if e.get("action") != "segment_alloc":
            continue
        k = "?"
        for f in e.get("frames") or []:
            fn = f.get("filename", "")
            if "arcstats" in fn or ("vllm" in fn and "torch/" not in fn):
                k = f"{fn.split('site-packages/')[-1].split('arcstats_pkg/')[-1]}:{f.get('line')} {f.get('name')}"
                break
        a = agg[k]
        a[0] += 1
        a[1] += e["size"]
        a[2] = max(a[2], e["size"])
    print(f"[memcensus] segment allocations in {len(tr)} trace events:", flush=True)
    for k, (n, tot, mx) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:15]:
        print(f"[memcensus]   {n:5d} x (max {mx / 2**20:7.1f} MiB) = {tot / 2**20:8.1f} MiB  {k}", flush=True)


def _alloc_phase_patch():
    """Memory at each startup phase (rank 0: after load, KV init, before/after graph capture) and
    ARC_ALLOC_TRIM=1: gc + empty_cache right after capture, so free blocks left by warmup/profiling do not stay cached
    into serving (with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True this also returns holes inside segments).
    ARC_MEMHIST=<n>: record allocator history (python stacks, <n> entries) from load to the end of capture, so the
    census can name who allocated the blocks that pin fragmented segments."""
    import gc
    import torch
    from vllm.v1.worker.gpu import model_runner as mr

    def rank0():
        try:
            from vllm.distributed import get_tensor_model_parallel_rank
            return get_tensor_model_parallel_rank() == 0
        except Exception:
            return True
    hist = int(os.environ.get("ARC_MEMHIST") or 0)
    o_load, o_kv, o_cap = mr.GPUModelRunner.load_model, mr.GPUModelRunner.initialize_kv_cache, mr.GPUModelRunner.capture_model

    def load_model(self, *a, **kw):
        if hist:
            torch.cuda.memory._record_memory_history(enabled="all", context="alloc", stacks="python", max_entries=hist)
        r = o_load(self, *a, **kw)
        if rank0():
            _mem_line("[memphase] after load_model")
        return r

    def initialize_kv_cache(self, *a, **kw):
        r = o_kv(self, *a, **kw)
        if rank0():
            _mem_line("[memphase] after initialize_kv_cache")
        return r

    def capture_model(self, *a, **kw):
        if rank0():
            _mem_line("[memphase] before capture_model (after warmup)")
        r = o_cap(self, *a, **kw)
        if kw.get("profile_only"):
            return r
        if rank0():
            _mem_line("[memphase] after capture_model")
        if os.environ.get("ARC_ALLOC_TRIM") == "1":
            gc.collect()
            torch.cuda.empty_cache()
            if rank0():
                _mem_line("[memphase] after trim")
                _memsnap_dump("after_trim")
        if hist and os.environ.get("ARC_MEMHIST_KEEP") != "1":
            torch.cuda.memory._record_memory_history(enabled=None)
        return r
    mr.GPUModelRunner.load_model = load_model
    mr.GPUModelRunner.initialize_kv_cache = initialize_kv_cache
    mr.GPUModelRunner.capture_model = capture_model


def _memreport_patch():
    """Debug: after load, print per-rank parameter/buffer bytes grouped by module path (rank 0 only)."""
    import collections, re
    import torch
    from vllm.v1.worker.gpu import model_runner as mr
    orig = mr.GPUModelRunner.load_model

    def load_model(self, *a, **kw):
        r = orig(self, *a, **kw)
        try:
            from vllm.distributed import get_tensor_model_parallel_rank
            if get_tensor_model_parallel_rank() != 0:
                return r
        except Exception:
            pass
        agg = collections.Counter()
        seen = set()
        mods = [("target", self.model)]
        sp = getattr(self, "speculator", None)
        if sp is not None and getattr(sp, "model", None) is not None:
            mods.append(("draft", sp.model))
        for tag, m in mods:
            for n, t in list(m.named_parameters()) + list(m.named_buffers()):
                if t.device.type != "cuda" or t.data_ptr() in seen:
                    continue
                seen.add(t.data_ptr())
                k = tag + ":" + re.sub(r"layers\.\d+\.", "layers.N.", n)
                k = re.sub(r"\.(w13|w2)_[a-z_]+$", r".\1_*", k)
                agg[k] += t.numel() * t.element_size()
        tot = sum(agg.values())
        print(f"[memreport] total {tot / 2**30:.2f} GiB, alloc {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)
        for k, v in agg.most_common(30):
            print(f"[memreport] {v / 2**20:9.1f} MiB  {k}", flush=True)
        import gc
        other = collections.Counter()
        for o in gc.get_objects():
            try:
                if isinstance(o, torch.Tensor) and o.is_cuda:
                    st = o.untyped_storage()
                    if st.data_ptr() in seen or o.data_ptr() in seen:
                        continue
                    seen.add(st.data_ptr())
                    other[(tuple(o.shape), str(o.dtype))] += st.nbytes()
            except Exception:
                pass
        print(f"[memreport] other python-visible CUDA tensors: {sum(other.values()) / 2**30:.2f} GiB", flush=True)
        for k, v in other.most_common(15):
            print(f"[memreport]   {v / 2**20:9.1f} MiB  {k}", flush=True)
        return r
    mr.GPUModelRunner.load_model = load_model

    import traceback
    from vllm.v1.core import kv_cache_utils as kcu
    if hasattr(kcu, "get_kv_cache_configs"):
        orig_cfgs = kcu.get_kv_cache_configs

        def get_kv_cache_configs(*a, **kw):
            cfgs = orig_cfgs(*a, **kw)
            c = cfgs[0] if isinstance(cfgs, (list, tuple)) else cfgs
            try:
                print(f"[memreport] KV: num_blocks={c.num_blocks} tensors={len(c.kv_cache_tensors)} "
                      f"total={sum(t.size for t in c.kv_cache_tensors) / 2**30:.2f} GiB", flush=True)
                for i, g in enumerate(c.kv_cache_groups):
                    sp = g.kv_cache_spec
                    inner = getattr(sp, "kv_cache_specs", None)
                    isp = next(iter(inner.values())) if inner else sp
                    print(f"[memreport] KV group {i}: {len(g.layer_names)} layers {type(isp).__name__} "
                          f"block={getattr(sp, 'block_size', '?')} page={getattr(sp, 'page_size_bytes', '?')} "
                          f"inner_page={getattr(isp, 'page_size_bytes', '?')} window={getattr(isp, 'sliding_window', None)} "
                          f"head={getattr(isp, 'head_size', '?')} dtype={getattr(isp, 'dtype', '?')} e.g. {g.layer_names[0]}",
                          flush=True)
                for t in c.kv_cache_tensors[:3]:
                    print(f"[memreport] KV tensor size={t.size / 2**20:.0f} MiB shared_by={len(t.shared_by)} e.g. {t.shared_by[:2]}", flush=True)
            except Exception as ex:
                print("[memreport] KV report failed", ex, flush=True)
            return cfgs
        kcu.get_kv_cache_configs = get_kv_cache_configs
        try:
            from vllm.v1.engine import core as _core
            _core.get_kv_cache_configs = get_kv_cache_configs
        except Exception:
            pass
    from vllm.v1.worker import workspace as ws
    orig_gs = ws.WorkspaceManager.get_simultaneous
    logged = set()

    def get_simultaneous(self, *specs, **kw):
        import math
        nbytes = sum(math.prod(sh) * torch.empty((), dtype=dt).element_size() for sh, dt in specs)
        if nbytes > (256 << 20):
            st = "".join(traceback.format_stack(limit=6)[:-1])
            if st not in logged:
                logged.add(st)
                print(f"[memreport] workspace request {nbytes / 2**20:.0f} MiB {specs}\n{st}", flush=True)
        return orig_gs(self, *specs, **kw)
    ws.WorkspaceManager.get_simultaneous = get_simultaneous


def _moe_patch():
    """(1) Non-uniform pruned checkpoints: config `arc_experts_per_layer` {layer: n} sets each MoE layer's routed-expert
    count (config.n_routed_experts = max over layers; checkpoint experts are numbered 0..n-1 per layer).
    (2) ARC_EXPERT_MASK=<json file {layer: [kept expert ids]}>: emulate pruning at runtime by adding -inf to the
    selection bias of dropped experts (noaux_tc top-8 among kept, weights renormalized = exactly the pruned model).
    The file is re-read when its mtime changes (eager mode only), so many pruning plans can be scored on one server."""
    import copy, json, re, time
    try:
        from vllm.models.dots3_note.nvidia import model as dm
    except Exception:
        return
    mask_path = os.environ.get("ARC_EXPERT_MASK")
    orig_init, orig_fwd = dm.Dots3NoteMoE.__init__, dm.Dots3NoteMoE.forward
    st = {"mtime": None, "checked": 0.0, "mods": {}}

    def init(self, config, *a, **kw):
        prefix = kw.get("prefix", "")
        m = re.search(r"layers\.(\d+)\.", prefix)
        per = getattr(config, "arc_experts_per_layer", None)
        if per and m and str(int(m.group(1))) in per:
            config = copy.copy(config)
            object.__setattr__(config, "n_routed_experts", int(per[str(int(m.group(1)))]))
        lowbit = getattr(config, "arc_lowbit", None)
        if lowbit:
            # routed experts come from <ckpt>/<arc_lowbit>/lowbit-L*.pt through arcstats.lowbit_moe (no FusedMoE params)
            from vllm.config import get_current_vllm_config
            from vllm.model_executor.models import deepseek_v2 as dv2
            from .lowbit_moe import ArcLowbitExperts
            ldir = os.path.join(get_current_vllm_config().model_config.model, lowbit)
            saved = dv2.FusedMoEFactory
            dv2.FusedMoEFactory = lambda **k: ArcLowbitExperts(lowbit_dir=ldir, **k)
            try:
                orig_init(self, config, *a, **kw)
            finally:
                dv2.FusedMoEFactory = saved
        else:
            orig_init(self, config, *a, **kw)
        if m:
            st["mods"][int(m.group(1))] = self

    def apply_mask():
        import torch
        st["checked"] = time.time()
        try:
            mt = os.stat(mask_path).st_mtime
        except OSError:
            return
        if mt == st["mtime"]:
            return
        st["mtime"] = mt
        keep = {int(k): v for k, v in json.load(open(mask_path)).items()}
        for l, mod in st["mods"].items():
            b = mod.gate.e_score_correction_bias
            if b is None:
                continue
            if not hasattr(mod, "_arc_bias0"):
                mod._arc_bias0 = b.data.clone()
            new = mod._arc_bias0.clone()
            if l in keep:
                pen = torch.full_like(new, -1e4)
                pen[torch.tensor(keep[l], device=new.device, dtype=torch.long)] = 0
                new += pen
            b.data.copy_(new)
        print(f"[arc] expert mask applied: {len(keep)} layers, mean kept "
              f"{sum(len(v) for v in keep.values()) / max(1, len(keep)):.1f}", flush=True)

    def fwd(self, *a, **kw):
        if mask_path and time.time() - st["checked"] > 1.0:
            apply_mask()
        return orig_fwd(self, *a, **kw)

    dm.Dots3NoteMoE.__init__ = init
    if mask_path:
        dm.Dots3NoteMoE.forward = fwd
    calib = os.environ.get("ARC_CALIB_DIR")
    if calib:
        # GPTQ calibration capture (eager, TP rank 0): per MoE layer, the MoE input rows (bf16) and their routed
        # top-k expert ids, one file per forward call.
        import torch
        os.makedirs(calib, exist_ok=True)
        base_fwd = dm.Dots3NoteMoE.forward
        cnt = {}

        def cfwd(self, hidden_states, *a, **kw):
            from vllm.distributed import get_tensor_model_parallel_rank
            l = next((k for k, v in st["mods"].items() if v is self), None)
            if l is not None and get_tensor_model_parallel_rank() == 0 and hidden_states.shape[0] > 64:
                with torch.no_grad():
                    logits = self.gate(hidden_states)
                    logits = logits[0] if isinstance(logits, tuple) else logits
                    sc = torch.sigmoid(logits.float())
                    b = self.gate.e_score_correction_bias
                    ids = torch.topk(sc + (b.float() if b is not None else 0), 8, dim=-1).indices
                n = cnt.get(l, 0); cnt[l] = n + 1
                torch.save({"x": hidden_states.detach().to(torch.bfloat16).cpu(), "ids": ids.to(torch.int16).cpu()},
                           os.path.join(calib, f"L{l:02d}_{n:05d}.pt"))
            return base_fwd(self, hidden_states, *a, **kw)
        dm.Dots3NoteMoE.forward = cfwd


def _attn_dump_patch():
    """Debug (ARC_DUMP_ATTN=dir): save (positions, attention output) of decoder layers 0-5 on TP rank 0, eager only."""
    import re, torch
    from vllm.models.dots3_note.nvidia import model as dm
    d = os.environ["ARC_DUMP_ATTN"]; os.makedirs(d, exist_ok=True)
    cnt = {}
    for cname in dir(dm):
        cls = getattr(dm, cname)
        if not (isinstance(cls, type) and "Attention" in cname and "forward" in cls.__dict__ and issubclass(cls, torch.nn.Module)):
            continue
        orig_init, orig_fwd = cls.__init__, cls.forward

        def init(self, *a, _oi=orig_init, **kw):
            _oi(self, *a, **kw)
            m = re.search(r"layers\.(\d+)\.self_attn$", kw.get("prefix", ""))
            self._arc_layer = int(m.group(1)) if m else None

        def fwd(self, positions, hidden_states, *a, _of=orig_fwd, **kw):
            out = _of(self, positions, hidden_states, *a, **kw)
            l = getattr(self, "_arc_layer", None)
            if l is not None and l <= 5 and positions.shape[0] > 64:
                from vllm.distributed import get_tensor_model_parallel_rank
                if get_tensor_model_parallel_rank() == 0:
                    n = cnt.get(l, 0); cnt[l] = n + 1
                    o = out[0] if isinstance(out, tuple) else out
                    torch.save({"pos": positions.cpu(), "out": o.float().cpu()}, f"{d}/L{l}_{n:03d}.pt")
            return out
        cls.__init__, cls.forward = init, fwd
    from vllm.models.dots3_note.nvidia import attention as da
    orig_rsw = da.Dots3NoteFlashAttnPrefillBackend.run_sliding_window
    st = {"n": 0}

    def rsw(self, q, k, v, cu_seq_lens_q, cu_seq_lens_k, max_seq_len_q, max_seq_len_k, sliding_window):
        o = orig_rsw(self, q, k, v, cu_seq_lens_q, cu_seq_lens_k, max_seq_len_q, max_seq_len_k, sliding_window)
        from vllm.distributed import get_tensor_model_parallel_rank
        if get_tensor_model_parallel_rank() == 0 and q.shape[0] > 64:
            n = st["n"]; st["n"] += 1
            if n % 33 == 0:   # first SWA layer (layer 2) of each forward
                torch.save({"q": q.float().cpu(), "k": k.float().cpu(), "v": v.float().cpu(), "o": o.float().cpu(),
                            "cq": cu_seq_lens_q.cpu(), "ck": cu_seq_lens_k.cpu(), "w": sliding_window,
                            "mq": max_seq_len_q, "mk": max_seq_len_k}, f"{d}/SWA_{n:05d}.pt")
        return o
    da.Dots3NoteFlashAttnPrefillBackend.run_sliding_window = rsw
