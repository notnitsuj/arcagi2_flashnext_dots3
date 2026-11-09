"""ARC_KVB_REBUILD=1 (default off, lever 6): keep the block-FP8 kv_b_proj weights off the GPU
and rebuild each one, bit-exactly, right before it is used.

Every MLA layer holds kv_b_proj twice: vLLM's absorbed bf16 W_UK_T / W_UV (decode, every step) and the block-FP8 weight
itself (prefill only: the SWA layers' MHA prefill and the DSA `_project_kv`). The bf16 copies are round_bf16(code *
scale) of the FP8 codes (vLLM dequantizes kv_b_proj through its own GEMM against an identity), and the e4m3 codes are
recovered exactly from them: code = e4m3(bf16 / scale) (bf16 keeps 8 mantissa bits, e4m3 has 3).

1. Construction: kv_b_proj parameters are created on the CPU (ColumnParallelLinear prefix '*.kv_b_proj'). vLLM's own
   CPU-offload path (`device_loading_context`) moves them to the GPU for weight post-processing (W_UK_T / W_UV are
   computed there, on the GPU) and back to pinned host memory afterwards. Freeing GPU-resident weights instead would
   leave ~5 MiB holes between live weights that the caching allocator does not return (measured: run ex2-B1).
2. After load: each weight is rebuilt from W_UK_T / W_UV into a per-device scratch buffer (largest layer, ~5 MiB at
   TP4) and compared byte for byte with the host copy. Identical layers are converted: their weight becomes an empty
   FP8 tensor, the scale moves to the GPU, and forward first rebuilds the weight into the scratch buffer and points
   `weight` at it (same bytes, same GEMM kernel: bit-identical output). A layer that does not match keeps a GPU copy
   of its weight (correct, no saving).
Frees ~222 MiB/GPU on Dots3 at TP4 (47 layers incl. MTP). One Triton kernel (~5 MiB written) per kv_b_proj call:
prefill steps only (decode reads W_UK_T / W_UV).
ARC_KVB_REBUILD_CHECK=N: also compare the rebuilt weight with the kept host copy on the first N calls (debug)."""
import os

import torch
import triton
import triton.language as tl

_SCRATCH = {}
_STATS = {"calls": 0, "layers": 0, "freed": 0, "kept": 0, "cpu_built": 0}
_CHECK = int(os.environ.get("ARC_KVB_REBUILD_CHECK", "0") or 0)


@triton.jit
def _rebuild_kernel(uk_ptr, uv_ptr, s_ptr, out_ptr, L, P, V,
                    suk_n, suk_p, suk_l, suv_n, suv_l, suv_v, ss_r, ss_c, so_r,
                    BR: tl.constexpr, BL: tl.constexpr):
    """out[r, l] = e4m3(W[r, l] / s[r // 128, l // 128]) with W[n (P + V) + d, l] = W_UK_T[n, d, l] (d < P) or
    W_UV[n, l, d - P]. A tile's BR rows lie in one head segment (BR divides P and V)."""
    pr = tl.program_id(0)
    pl = tl.program_id(1)
    r0 = pr * BR
    n = r0 // (P + V)
    d0 = r0 - n * (P + V)
    rr = r0 + tl.arange(0, BR)
    ol = pl * BL + tl.arange(0, BL)
    lm = ol < L
    d = d0 + tl.arange(0, BR)
    if d0 < P:
        w = tl.load(uk_ptr + n * suk_n + d[:, None] * suk_p + ol[None, :] * suk_l, mask=lm[None, :], other=0.0)
    else:
        w = tl.load(uv_ptr + n * suv_n + ol[None, :] * suv_l + (d - P)[:, None] * suv_v, mask=lm[None, :], other=0.0)
    s = tl.load(s_ptr + (rr // 128)[:, None] * ss_r + (ol // 128)[None, :] * ss_c, mask=lm[None, :], other=1.0)
    c = tl.math.div_rn(w.to(tl.float32), s)
    tl.store(out_ptr + rr[:, None] * so_r + ol[None, :], c.to(tl.float8e4nv), mask=lm[None, :])


def _rebuild(uk, uv, si, out):
    """uk [N, P, L] bf16, uv [N, L, V] bf16 (any strides), si [ceil(rows/128), ceil(L/128)] fp32 -> out [rows, L] e4m3"""
    N, P, L = uk.shape
    V = uv.shape[2]
    BR = 16
    assert P % BR == 0 and V % BR == 0 and out.shape == (N * (P + V), L) and out.stride(1) == 1
    grid = (N * (P + V) // BR, triton.cdiv(L, 128))
    _rebuild_kernel[grid](uk, uv, si, out, L, P, V, uk.stride(0), uk.stride(1), uk.stride(2),
                          uv.stride(0), uv.stride(1), uv.stride(2), si.stride(0), si.stride(1), out.stride(0),
                          BR=BR, BL=128, num_warps=4)
    return out


def _pairs(models):
    """(kv_b_proj module, owner with W_UK_T / W_UV) for every MLA layer; one owner per kv_b_proj"""
    seen = {}
    for m in models:
        for mod in m.modules():
            kvb = getattr(mod, "kv_b_proj", None)
            uk, uv = getattr(mod, "W_UK_T", None), getattr(mod, "W_UV", None)
            if kvb is None or uk is None or uv is None or id(kvb) in seen:
                continue
            seen[id(kvb)] = (kvb, mod)
    return list(seen.values())


def _to_gpu(kvb, dev):
    """fallback: this layer keeps its weight on the GPU"""
    kvb.weight.data = kvb.weight.data.to(dev)
    kvb.weight_scale_inv.data = kvb.weight_scale_inv.data.to(dev)


def _convert(models):
    pairs = _pairs(models)
    good = []
    for kvb, own in pairs:
        w, si = getattr(kvb, "weight", None), getattr(kvb, "weight_scale_inv", None)
        uk, uv = own.W_UK_T, own.W_UV
        dev = uk.device
        ok = (w is not None and si is not None and w.dtype == torch.float8_e4m3fn and w.dim() == 2
              and uk.is_cuda and uk.dtype == torch.bfloat16 and uv.dtype == torch.bfloat16
              and uk.dim() == 3 and uv.dim() == 3
              and tuple(w.shape) == (uk.shape[0] * (uk.shape[1] + uv.shape[2]), uk.shape[2]) and w.is_contiguous()
              and si.dtype == torch.float32 and uk.shape[1] % 16 == 0 and uv.shape[2] % 16 == 0
              and getattr(uk, "_arc_fp8", None) is None)
        if ok:
            good.append((kvb, own))
        elif w is not None and si is not None and not w.is_cuda:
            _to_gpu(kvb, dev)
            _STATS["kept"] += 1
        else:
            _STATS["kept"] += 1
    nmax = {}
    for kvb, own in good:
        d = own.W_UK_T.device
        nmax[d] = max(nmax.get(d, 0), kvb.weight.numel())
    for d, n in nmax.items():
        _SCRATCH[d] = torch.empty(n, dtype=torch.float8_e4m3fn, device=d)
    freed = 0
    for kvb, own in good:
        w = kvb.weight.data
        dev = own.W_UK_T.device
        if w.is_cuda:
            _STATS["gpu_built"] = _STATS.get("gpu_built", 0) + 1
        else:
            _STATS["cpu_built"] += 1
        si = kvb.weight_scale_inv.data.to(dev)
        buf = _SCRATCH[dev][:w.numel()].view(w.shape)
        _rebuild(own.W_UK_T, own.W_UV, si, buf)
        same = torch.equal(buf.view(torch.uint8).cpu(), w.view(torch.uint8).cpu())
        if not same:
            if not w.is_cuda:
                _to_gpu(kvb, dev)
            _STATS["kept"] += 1
            continue
        kvb.weight_scale_inv.data = si
        kvb._arc_kvb = (own, tuple(w.shape))
        _patch_forward(kvb)
        if _CHECK > 0:
            kvb._arc_kvb_ref = w.to(dev)          # debug: keep a GPU reference
        if w.is_cuda:
            freed += w.numel()
        kvb.weight.data = torch.empty((0, w.shape[1]), dtype=w.dtype, device=dev)
        _STATS["layers"] += 1
    _STATS["freed"] = freed
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return pairs


def _patch_forward(kvb):
    orig = kvb.forward

    def forward(x, *a, **kw):
        own, shape = kvb._arc_kvb
        n = shape[0] * shape[1]
        buf = _SCRATCH[own.W_UK_T.device][:n].view(shape)
        _rebuild(own.W_UK_T, own.W_UV, kvb.weight_scale_inv, buf)
        kvb.weight.data = buf
        _STATS["calls"] += 1
        if _CHECK > 0 and _STATS["calls"] <= _CHECK:
            if not torch.equal(buf.view(torch.uint8), kvb._arc_kvb_ref.view(torch.uint8)):
                print(f"[kvb_rebuild] CHECK FAILED at call {_STATS['calls']}", flush=True)
            elif _STATS["calls"] == _CHECK:
                print(f"[kvb_rebuild] check: first {_CHECK} rebuilds identical", flush=True)
        return orig(x, *a, **kw)
    kvb.forward = forward


def _cpu_construct_patch():
    """kv_b_proj parameters are created on the CPU (vLLM's device_loading_context moves them to the GPU for weight
    post-processing and back to pinned host memory)"""
    from vllm.model_executor.layers import linear as lin
    orig = lin.ColumnParallelLinear.__init__

    def init(self, *a, **kw):
        if str(kw.get("prefix", "")).endswith("kv_b_proj"):
            with torch.device("cpu"):
                return orig(self, *a, **kw)
        return orig(self, *a, **kw)
    lin.ColumnParallelLinear.__init__ = init


def install():
    from vllm.v1.worker.gpu import model_runner as mr
    _cpu_construct_patch()
    orig = mr.GPUModelRunner.load_model

    def load_model(self, *a, **kw):
        r = orig(self, *a, **kw)
        models = [self.model]
        sp = getattr(self, "speculator", None)
        if sp is not None and getattr(sp, "model", None) is not None:
            models.append(sp.model)
        try:
            free0 = torch.cuda.mem_get_info()[0]
            pairs = _convert(models)
            print(f"[kvb_rebuild] {len(pairs)} kv_b_proj, {_STATS['layers']} rebuilt on use "
                  f"({_STATS['cpu_built']} host-resident), {_STATS['kept']} kept on GPU, freed on GPU "
                  f"{_STATS['freed'] / 2**20:.1f} MiB, scratch {sum(b.numel() for b in _SCRATCH.values()) / 2**20:.1f} MiB, "
                  f"driver free {free0 / 2**20:.0f} -> {torch.cuda.mem_get_info()[0] / 2**20:.0f} MiB", flush=True)
        except Exception as ex:   # converted layers passed the byte check; others keep (or regain) a GPU weight
            print("[kvb_rebuild] conversion error:", repr(ex)[:300], flush=True)
            for kvb, own in _pairs(models):
                if getattr(kvb, "_arc_kvb", None) is None and not kvb.weight.is_cuda:
                    _to_gpu(kvb, own.W_UK_T.device)
        return r
    mr.GPUModelRunner.load_model = load_model
