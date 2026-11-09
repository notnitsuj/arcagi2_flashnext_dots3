"""Offload decode attention v2 (csrc/offload_attn2.cu -> offload_attn2.so, built by dots3/build_offload_attn2.sh).
Drop-in for offload_cuda.split_partials(split=1, out=...) on the decode path: 8 warps per token CTA, Q fragments in
registers, row metadata staged in shared memory, optional FP8 P.V (P rounded to e4m3). the design notes.
  ARC_OFFLOAD_ATTN2 = 0 (default, off) | 1 (on: P.V in FP16, v1 numerics) | 2 (on: P.V in FP8)
  ARC_OFFLOAD_ATTN2_ST = 2 | 3 (pipeline stages of 32-row tiles, default 3)
  ARC_OFFLOAD_ATTN2_KERN = 2 (8 warps, CTA-wide phases) | 3 (head-split CTAs, 2 per L4 SM) | 4 (default: warp-specialized
    ping-pong: 4 loader warps + two head groups taking turns on the tensor cores) | 5 (ping-pong, the groups gather
    their own half tiles; 8 warps). ST 3 | 4 (default 4 for 4/5)"""
import ctypes
import os

import torch

MODE = int(os.environ.get("ARC_OFFLOAD_ATTN2", "0") or 0)
KERN = int(os.environ.get("ARC_OFFLOAD_ATTN2_KERN", "4") or 4)
ST = int(os.environ.get("ARC_OFFLOAD_ATTN2_ST", "4" if KERN >= 4 else "3"))
_L = None
_ERR = None


def lib():
    global _L, _ERR
    if _L is None and _ERR is None:
        p = os.environ.get("ARC_OFFLOAD_ATTN2_SO") or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                    "offload_attn2.so")
        try:
            L = ctypes.CDLL(p)
            f = L.oa2_v2_launch
            f.restype = ctypes.c_int
            f.argtypes = ([ctypes.c_void_p] * 6 + [ctypes.c_longlong] * 5 + [ctypes.c_int, ctypes.c_float,
                          ctypes.c_void_p, ctypes.c_longlong, ctypes.c_longlong, ctypes.c_float, ctypes.c_void_p,
                          ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p])
            m = L.oa2_mb_launch
            m.restype = ctypes.c_int
            m.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            a = L.oa2_abl_launch
            a.restype = ctypes.c_int
            a.argtypes = ([ctypes.c_void_p] * 4 + [ctypes.c_longlong] * 5 + [ctypes.c_int, ctypes.c_float,
                          ctypes.c_void_p, ctypes.c_longlong, ctypes.c_longlong, ctypes.c_float, ctypes.c_int,
                          ctypes.c_int, ctypes.c_void_p])
            p = L.oa2_prof
            p.restype = ctypes.c_int
            p.argtypes = [ctypes.c_void_p, ctypes.c_int]
            _L = L
        except (OSError, AttributeError) as ex:
            _ERR = repr(ex)
    return _L


def available():
    return lib() is not None


_SEEN = [False, False]


def ok(T, H, topk, q, idx, res):
    """the v2 kernel's preconditions (else callers keep the v1 path); logs the first use / first refusal"""
    r = _ok(T, H, topk, q, idx, res)
    if not _SEEN[int(r)]:
        _SEEN[int(r)] = True
        print(f"[offload] ATTN2 {'ON' if r else 'refused (v1 kept)'}: kern={KERN} mode={MODE} st={ST} T={T} H={H} "
              f"topk={topk} lib={available()} {_ERR or ''}", flush=True)
    return r


def _ok(T, H, topk, q, idx, res):
    return (MODE > 0 and available() and H == 32 and res is not None and topk <= 2048 and topk % 32 == 0
            and q.stride(2) == 1 and q.dtype == torch.bfloat16 and q.data_ptr() % 8 == 0 and q.stride(1) % 4 == 0
            and q.stride(0) % 4 == 0 and idx.dtype == torch.int32 and idx.stride(1) == 1 and idx.stride(0) % 4 == 0
            and idx.data_ptr() % 16 == 0 and res.dtype == torch.int32 and res.data_ptr() % 16 == 0)


def attend(q, host, cache, spill, idx, res, sm_scale, R, out, kv_scale, lflag=None, sflag=None, hfmt=0, pvf8=None,
           st=None, kern=None):
    """q [T, 32, 576] bf16, idx [T, TOPK] int32 host rows, res [T*TOPK] int32 resolved locations (as
    offload_cuda.split_partials), sm_scale includes log2(e) and kv_scale; out [T, 32, 512] bf16 normalized x kv_scale"""
    T, H, D = q.shape
    f8 = (MODE == 2) if pvf8 is None else bool(pvf8)
    k = KERN if kern is None else int(kern)
    cfg = (st or (ST if kern is None else (4 if k >= 4 else 3))) | (16 if f8 else 0) | {2: 0, 3: 32, 4: 64, 5: 128}[k]
    rc = lib().oa2_v2_launch(
        q.data_ptr(), host.data_ptr(), cache.data_ptr(), spill.data_ptr(), idx.data_ptr(), res.data_ptr(),
        T, R, q.stride(0), q.stride(1), idx.stride(0), idx.shape[1], float(sm_scale), out.data_ptr(),
        out.stride(0), out.stride(1), float(kv_scale), lflag.data_ptr() if lflag is not None else None,
        sflag.data_ptr() if sflag is not None else None, int(hfmt), cfg,
        torch.cuda.current_stream(q.device).cuda_stream)
    if rc != 0:
        raise RuntimeError(f"oa2_v2_launch failed: {rc}")
    return out


# ---- ARC_OFFLOAD_ATTN2_CMP=N[,until_s] (diagnostics, default off): every N-th v2 call on device 0 (until until_s
#      seconds after ARC_T0) also runs the shipping v1 kernel (offload_cuda.split_partials, split 1) on the same inputs
#      and logs the difference of the two outputs on real serving data: max / mean abs, rel = ||v2 - v1|| / ||v1||.
_CMP = os.environ.get("ARC_OFFLOAD_ATTN2_CMP", "")
_CMP_N, _CMP_UNTIL = (int(_CMP.split(",")[0]), float(_CMP.split(",")[1]) if "," in _CMP else 1e18) if _CMP else (0, 0)
_CMP_T0 = float(os.environ.get("ARC_T0", "0") or 0)
_cmp_calls = [0]
_cmp_accs = {}


def compare(q, host, cache, spill, idx, res, sm_scale, R, out, kv_scale, lflag=None, sflag=None, hfmt=0, pvf8=None):
    """after attend(): optionally rerun v1 on the same inputs and accumulate the output difference"""
    if not _CMP_N or out.device.index not in (0, None):
        return
    _cmp_calls[0] += 1
    if _cmp_calls[0] % _CMP_N:
        return
    import time
    if time.time() - _CMP_T0 > _CMP_UNTIL:
        return
    from . import offload_cuda
    ref = torch.empty_like(out)
    # rows the v2 call claimed / waited on are written by now (same stream): no flags for the rerun
    r = res[:idx.numel()].clone()
    r.sub_(torch.where(r >= (1 << 30), 1 << 30, 0).to(r.dtype))
    offload_cuda.split_partials(q, host, cache, spill, idx, r, None, 1, sm_scale, R, stages=2, out=ref,
                                kv_scale=kv_scale, hfmt=hfmt)
    d = (out.float() - ref.float())
    f8 = (MODE == 2) if pvf8 is None else bool(pvf8)
    a = _cmp_accs.setdefault(f8, {"n": 0, "max": 0.0, "sum_abs": 0.0, "sum_sq_d": 0.0, "sum_sq": 0.0, "cnt": 0})
    a["n"] += 1
    a["max"] = max(a["max"], float(d.abs().max()))
    a["sum_abs"] += float(d.abs().sum())
    a["cnt"] += d.numel()
    a["sum_sq_d"] += float((d * d).sum())
    a["sum_sq"] += float((ref.float() ** 2).sum())
    if a["n"] % 20 == 1:
        print(f"[attn2 cmp] calls={a['n']} T={q.shape[0]} kern={KERN} pv{'8' if f8 else '16'} max_abs={a['max']:.3e} "
              f"mean_abs={a['sum_abs'] / max(a['cnt'], 1):.3e} rel={(a['sum_sq_d'] / max(a['sum_sq'], 1e-30)) ** 0.5:.3e} "
              f"(this call max {float(d.abs().max()):.3e})", flush=True)


# ---- ARC_OFFLOAD_ATTN2_TIME=N (diagnostics, default off): CUDA events around every decode attention call on device 0
#      (v1 and v2 paths alike; no syncs: an event pair is read ~64 calls later, when it has long completed). Every N
#      timed calls it prints per arm (v1 / pv16 / pv8): calls, mean ms per call, mean T, and ms per 256 tokens.
_TIME = int(os.environ.get("ARC_OFFLOAD_ATTN2_TIME", "0") or 0)
_tq = []
_tacc = {}


def time_begin(dev):
    if not _TIME or dev.index not in (0, None):
        return None
    e = torch.cuda.Event(enable_timing=True)
    e.record()
    return e


def time_end(e0, arm, T):
    if e0 is None:
        return
    e1 = torch.cuda.Event(enable_timing=True)
    e1.record()
    _tq.append((arm, T, e0, e1))
    if len(_tq) < 128:
        return
    for arm_, T_, a, b in _tq[:64]:
        b.synchronize()   # completed long ago
        s = _tacc.setdefault(arm_, [0, 0.0, 0])
        s[0] += 1
        s[1] += a.elapsed_time(b)
        s[2] += T_
    del _tq[:64]
    tot = sum(v[0] for v in _tacc.values())
    if tot % _TIME < 64:
        print("[attn2 time] " + "  ".join(f"{k}: n={v[0]} {v[1] / v[0]:.3f} ms/call T={v[2] / v[0]:.0f} "
                                          f"{256 * v[1] / max(v[2], 1):.3f} ms/256tok" for k, v in sorted(_tacc.items())),
              flush=True)
