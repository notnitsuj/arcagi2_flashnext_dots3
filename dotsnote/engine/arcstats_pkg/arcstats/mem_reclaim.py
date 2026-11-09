"""Lever 6: exact GPU-memory reclaims for v22c serving, each behind its own default-off flag.

ARC_PENALTY_LAZY=1: vLLM's PenaltiesState allocates output_bin_counts [max_num_seqs, vocab] int32 (37 MiB at 64 x
  152064) at init and post_update() increments it for every sampled token, but it is only read when a request uses
  repetition / frequency / presence penalties (solve.py never does). It is allocated on the first such request
  instead (bincount() then builds that request's counts from its token ids, exactly as for any new penalty request);
  until then post_update() gets None and skips the counting. vLLM's sampler warmup (before graph capture) requests
  every penalty; those warmup requests run without penalties instead (their outputs are discarded). Outputs of real
  requests are unchanged.

ARC_MLA_WS_TOKENS=N: the MLA chunked-prefill workspace (shared by every MLA metadata builder) is sized for 65536 tokens
  x 1088 x bf16 = 136 MiB. In Dots3 only the SWA layers (and the SWA-type MTP layer) read it: their prefill gathers
  at most query_len + 512 rows per request (<= 1536 with 1024-token chunks), and requests whose windows do not fit
  together are split into separate varlen calls (per-sequence results are independent of the split; a single request
  that does not fit raises). N = 16384 keeps every realistic step in one call (splits are counted and logged) and frees
  ~102 MiB. The DSA layers use the offload path (their own staging buffers).

ARC_PARAM_COMPACT=1: at the end of model load, round-trip every CUDA nn.Parameter storage (dense weights, W_UK_T /
  W_UV, norms; not the low-bit experts, which ARC_LOWBIT_COMPACT already packs) through host memory and re-allocate
  them largest first, so the holes that load-time frees left between them (gmerge, weight post-processing: ~215 MiB
  that the 20 MiB-granular expandable segments cannot return) merge and are returned. Same bytes, same shapes and
  strides; every gc-visible tensor that shares a moved storage is re-pointed. Runs before KV init and graph capture,
  so no captured pointer refers to the old storages."""
import os

_SPLITS = {"steps": 0, "split": 0}
_WARM = {"done": False}


def _penalty_lazy():
    import numpy as np
    import torch
    from vllm.utils.math_utils import cdiv
    from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor
    from vllm.v1.worker.gpu.sample import penalties as pen

    def init(self, req_states):
        self.req_states = req_states
        max_num_reqs = req_states.max_num_reqs
        self.vocab_size = req_states.vocab_size
        self.device = req_states.device
        self.repetition_penalty = UvaBackedTensor(max_num_reqs, dtype=torch.float32)
        self.frequency_penalty = UvaBackedTensor(max_num_reqs, dtype=torch.float32)
        self.presence_penalty = UvaBackedTensor(max_num_reqs, dtype=torch.float32)
        self.use_penalty = np.zeros(max_num_reqs, dtype=bool)
        self.repetition_penalty.np.fill(1.0)
        self.repetition_penalty.copy_to_uva()
        self.prompt_bin_mask = torch.zeros(max_num_reqs, cdiv(self.vocab_size, 32), dtype=torch.int32,
                                           device=self.device)
        self.output_bin_counts = None          # allocated on the first request that uses a penalty
        self._new_penalties_reqs = []
        self._arc_max_reqs = max_num_reqs

    orig_add = pen.PenaltiesState.add_request

    def add_request(self, req_idx, sampling_params):
        if self.output_bin_counts is None and pen.use_penalty(sampling_params) and not _WARM["done"]:
            # vLLM's sampler warmup (SamplingParams.for_sampler_warmup) turns every penalty on; its outputs are
            # discarded, so warmup requests run without penalties (the penalty kernels are never used afterwards)
            self.repetition_penalty.np[req_idx] = 1.0
            self.frequency_penalty.np[req_idx] = 0.0
            self.presence_penalty.np[req_idx] = 0.0
            self.use_penalty[req_idx] = False
            return
        if self.output_bin_counts is None and pen.use_penalty(sampling_params):
            self.output_bin_counts = torch.zeros(self._arc_max_reqs, self.vocab_size, dtype=torch.int32,
                                                 device=self.device)
            print(f"[mem_reclaim] penalties in use: output_bin_counts allocated "
                  f"({self.output_bin_counts.numel() * 4 / 2**20:.0f} MiB)", flush=True)
        return orig_add(self, req_idx, sampling_params)
    pen.PenaltiesState.__init__ = init
    pen.PenaltiesState.add_request = add_request
    from vllm.v1.worker.gpu import model_runner as mr
    o_cap = mr.GPUModelRunner.capture_model

    def capture_model(self, *a, **kw):   # warmup (incl. the sampler warmup) runs before graph capture
        r = o_cap(self, *a, **kw)
        if not kw.get("profile_only"):
            _WARM["done"] = True
        return r
    mr.GPUModelRunner.capture_model = capture_model


def _mla_ws(tokens):
    from vllm.model_executor.layers.attention import mla_attention as ma
    orig = ma.MLACommonMetadataBuilder.determine_chunked_prefill_workspace_size

    def det(vllm_config):
        n = orig(vllm_config)
        return min(n, tokens)
    ma.MLACommonMetadataBuilder.determine_chunked_prefill_workspace_size = staticmethod(det)
    try:   # count SWA prefill steps whose windows needed more than one varlen call (would not with 65536)
        from vllm.models.dots3_note.nvidia import attention as da
        ob = da._build_sliding_window_metadata

        def build(**kw):
            md = ob(**kw)
            _SPLITS["steps"] += 1
            if len(md.chunks) > 1:
                _SPLITS["split"] += 1
                if _SPLITS["split"] <= 5 or _SPLITS["split"] % 1000 == 0:
                    print(f"[mem_reclaim] SWA prefill split into {len(md.chunks)} calls "
                          f"({_SPLITS['split']} of {_SPLITS['steps']} prefill builds)", flush=True)
            return md
        da._build_sliding_window_metadata = build
    except Exception as ex:
        print("[mem_reclaim] split counter not installed:", repr(ex)[:200], flush=True)


def param_compact(models):
    import gc
    import time
    import torch
    t0 = time.time()
    st = torch.cuda.memory_stats()
    r0, a0 = st["reserved_bytes.all.current"], st["allocated_bytes.all.current"]
    groups = {}
    for m in models:
        for p in m.parameters():
            if p.is_cuda and p.numel() > 0:
                groups.setdefault(p.untyped_storage().data_ptr(), {})[id(p)] = p
    for o in gc.get_objects():   # other tensors sharing a moved storage (views kept outside the module tree)
        try:
            if isinstance(o, torch.Tensor) and o.is_cuda and o.numel() > 0:
                k = o.untyped_storage().data_ptr()
                if k in groups:
                    groups[k].setdefault(id(o), o)
        except Exception:
            pass
    plan = []
    for k, ts in groups.items():
        ts = list(ts.values())
        stg = ts[0].untyped_storage()
        host = torch.empty(stg.nbytes(), dtype=torch.uint8, pin_memory=True)
        host.copy_(torch.empty(0, dtype=torch.uint8, device=ts[0].device).set_(stg))
        meta = [(t, t.dtype, t.size(), t.stride(), t.storage_offset()) for t in ts]
        plan.append((host, ts[0].device, meta))
    torch.cuda.synchronize()
    for host, dev, meta in plan:
        for t, dt, sz, sd, off in meta:
            t.data = torch.empty(0, dtype=dt, device=dev)
    gc.collect()
    torch.cuda.empty_cache()
    for host, dev, meta in sorted(plan, key=lambda x: -x[0].numel()):
        buf = torch.empty(host.numel(), dtype=torch.uint8, device=dev)
        buf.copy_(host)
        for t, dt, sz, sd, off in meta:
            t.data = torch.empty(0, dtype=dt, device=dev).set_(buf.untyped_storage(), off, sz, sd)
    del plan
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    st = torch.cuda.memory_stats()
    r1, a1 = st["reserved_bytes.all.current"], st["allocated_bytes.all.current"]
    print(f"[mem_reclaim] param compaction: {len(groups)} storages in {time.time() - t0:.1f} s, reserved "
          f"{r0 / 2**20:.0f} -> {r1 / 2**20:.0f} MiB, allocated {a0 / 2**20:.0f} -> {a1 / 2**20:.0f} MiB", flush=True)


def _param_compact_patch():
    from vllm.v1.worker.gpu import model_runner as mr
    orig = mr.GPUModelRunner.load_model

    def load_model(self, *a, **kw):
        r = orig(self, *a, **kw)
        models = [self.model]
        sp = getattr(self, "speculator", None)
        if sp is not None and getattr(sp, "model", None) is not None:
            models.append(sp.model)
        param_compact(models)
        return r
    mr.GPUModelRunner.load_model = load_model


def install():
    if os.environ.get("ARC_PENALTY_LAZY") == "1":
        _penalty_lazy()
    n = int(os.environ.get("ARC_MLA_WS_TOKENS", "0") or 0)
    if n > 0:
        _mla_ws(n)


def install_late():
    """installed after every other load_model wrapper (outermost): runs last at load"""
    if os.environ.get("ARC_PARAM_COMPACT") == "1":
        _param_compact_patch()
