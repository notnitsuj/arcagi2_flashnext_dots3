"""CUDA split-K sparse MLA decode attention over FP8 offload rows (csrc/offload_attn.cu, prebuilt offload_attn.so
next to this file; built by dots3/build_offload_attn.sh). Same partial layout as sparse_offload._sparse_mla_offload_split_kernel
(m, l [T*SPLIT*32], acc [T*SPLIT*32, 512] fp32) so sparse_offload._combine_kernel merges it. 32 heads only."""
import ctypes
import os

import torch

_LIB = None
_PK = None
_ERR = None
_OV = None     # (offload_split_launch2, offload_fill_copy_launch) of builds with the overlapped fill
_HF = None     # (offload_split_launch_fmt, offload_fill_copy_launch_fmt, offload_hf_rows_launch): compact host rows
_GT = None     # offload_fill_copy_launch_gated (ARC_OFFLOAD_AR_GATE, link_gate.py) of builds that have it


def lib():
    global _LIB, _ERR
    if _LIB is None and _ERR is None:
        p = os.environ.get("ARC_OFFLOAD_CUDA_SO") or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                   "offload_attn.so")
        try:
            L = ctypes.CDLL(p)
            f = L.offload_split_launch
            f.restype = ctypes.c_int
            f.argtypes = ([ctypes.c_void_p] * 9 + [ctypes.c_longlong] * 5 + [ctypes.c_int, ctypes.c_int, ctypes.c_float,
                          ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong, ctypes.c_longlong,
                          ctypes.c_float])
            _LIB = f
            global _PK
            try:   # request-packed decode entry points (absent from .so builds that predate them)
                fu, fa = L.offload_pk_union_launch, L.offload_pk_attn_launch
                fu.restype = fa.restype = ctypes.c_int
                fu.argtypes = [ctypes.c_void_p, ctypes.c_longlong, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                               ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong, ctypes.c_int, ctypes.c_int,
                               ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
                fa.argtypes = ([ctypes.c_void_p] * 12 + [ctypes.c_int] * 4 + [ctypes.c_longlong] * 3 +
                               [ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_void_p,
                                ctypes.c_void_p, ctypes.c_longlong, ctypes.c_longlong, ctypes.c_float,
                                ctypes.c_void_p])
                _PK = (fu, fa)
            except AttributeError:
                _PK = None
            global _OV
            try:   # overlapped fill (ARC_OFFLOAD_OVERLAP): split kernel with wait flags + the flagged fill copy
                f2, fc = L.offload_split_launch2, L.offload_fill_copy_launch
                f2.restype = fc.restype = ctypes.c_int
                f2.argtypes = f.argtypes + [ctypes.c_void_p, ctypes.c_void_p]
                fc.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
                _OV = (f2, fc)
            except AttributeError:
                _OV = None
            global _HF
            try:   # compact host rows (ARC_OFFLOAD_ROW_FMT, offload_rowfmt.py)
                f3, fc3, fr = L.offload_split_launch_fmt, L.offload_fill_copy_launch_fmt, L.offload_hf_rows_launch
                f3.restype = fc3.restype = fr.restype = ctypes.c_int
                f3.argtypes = f.argtypes + [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
                fc3.argtypes = ([ctypes.c_void_p] * 8 + [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                ctypes.c_int])
                fr.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                               ctypes.c_void_p]
                _HF = (f3, fc3, fr)
            except AttributeError:
                _HF = None
            global _GT
            try:
                g = L.offload_fill_copy_launch_gated
                g.restype = ctypes.c_int
                g.argtypes = ([ctypes.c_void_p] * 8 + [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                              ctypes.c_int, ctypes.c_void_p])
                _GT = g
            except AttributeError:
                _GT = None
        except OSError as ex:
            _ERR = repr(ex)
    return _LIB


def attend_rows(q, rows, idx, sm_scale, kv_scale, out, split=1, stages=3):
    """prefill form: q [T, 32, 576] bf16, rows [N, 576] FP8 staged rows, idx [T, TOPK] int32 into rows (-1 = none)
    -> out [T, 32, 512] bf16 (sm_scale without log2(e)). split 1 writes out directly (no workspace)."""
    from . import sparse_offload as so
    T = q.shape[0]
    if split == 1:
        split_partials(q, rows, rows, rows, idx, None, None, 1, sm_scale * 1.4426950408889634 * kv_scale,
                       rows.shape[0], stages=stages, out=out, kv_scale=kv_scale)
        return out
    ws = so._split_ws(T * split * 32, q.device)
    split_partials(q, rows, rows, rows, idx, None, ws, split, sm_scale * 1.4426950408889634 * kv_scale, rows.shape[0],
                   stages=stages)
    so._combine_kernel[(T, 1)](ws[0], ws[1], ws[2], out, out.stride(0), out.stride(1), 32, kv_scale,
                               SPLIT=split, BLOCK_H=32, HP=32, DN=so.D_NOPE, num_warps=4)
    return out


def available():
    return lib() is not None


def overlap_available():
    lib()
    return _OV is not None


def hostfmt_available():
    lib()
    return _HF is not None


def hf_rows(host, rows, out, fmt):
    """test hook: compact host rows host[rows] (int64 ids) -> fp8 rows out [n, 576] via the CUDA dequant"""
    lib()
    rc = _HF[2](host.data_ptr(), rows.data_ptr(), rows.shape[0], out.data_ptr(), int(fmt),
                torch.cuda.current_stream(out.device).cuda_stream)
    if rc != 0:
        raise RuntimeError(f"offload_hf_rows_launch failed: {rc}")


def gate_available():
    lib()
    return _GT is not None


def fill_copy(ml, mcnt, cpd, host, cache, spill, lflag, sflag, ctas, stream=None, perm=None, ns=0, hfmt=0, gate=None):
    """overlapped fill copy (csrc fill_copy_kernel) on `stream`: miss list rows host -> cache line / spill row per cpd.
    With flags: only rows it claims (pending 1 -> 2), cleared to 0 after the write; lflag None: every row, no flags.
    hfmt: compact host row format code (offload_rowfmt.FMT; 0 = fp8 rows)"""
    lib()
    st = stream if stream is not None else torch.cuda.current_stream(cache.device)
    fn = _OV[1] if not hfmt else (lambda *a: _HF[1](*a, int(hfmt)))
    if gate is not None and _GT is not None:   # ARC_OFFLOAD_AR_GATE: pause while the all-reduce gate word is set
        fn = lambda *a: _GT(*a, int(hfmt), gate.data_ptr())
    rc = fn(ml.data_ptr(), mcnt.data_ptr(), cpd.data_ptr(), host.data_ptr(), cache.data_ptr(), spill.data_ptr(),
                lflag.data_ptr() if lflag is not None else None, sflag.data_ptr() if sflag is not None else None,
                int(ctas), st.cuda_stream,
                perm.data_ptr() if perm is not None and ns > 0 else None, int(ns) if perm is not None else 0)
    if rc != 0:
        raise RuntimeError(f"offload_fill_copy_launch failed: {rc}")


def split_partials(q, host, cache, spill, idx, res, ws, split, sm_scale, R, stages=3, out=None, kv_scale=1.0,
                   lflag=None, sflag=None, hfmt=0):
    """q [T, 32, 576] bf16 (last dim contiguous), idx [T, TOPK] int32, res [T*TOPK] int32 resolved locations (None:
    all rows from host[idx], R = its row count), ws = (m, l, acc) fp32 buffers of >= T*split*32 rows; sm_scale
    includes log2(e). stages: 2 | 3, +10 = 8-warp CTAs. out (split 1): bf16 [T, 32, 512] written normalized (x kv_scale), ws unused"""
    T, H, D = q.shape
    assert H == 32 and D == 576 and q.dtype == torch.bfloat16 and q.stride(2) == 1
    assert idx.dtype == torch.int32 and idx.stride(1) == 1 and (res is None or res.dtype == torch.int32)
    args = (q.data_ptr(), host.data_ptr(), cache.data_ptr(), spill.data_ptr(), idx.data_ptr(),
            res.data_ptr() if res is not None else None,
            *((None,) * 3 if ws is None else (ws[0].data_ptr(), ws[1].data_ptr(), ws[2].data_ptr())),
            T, R, q.stride(0), q.stride(1), idx.stride(0), idx.shape[1], split, float(sm_scale), stages,
            torch.cuda.current_stream(q.device).cuda_stream,
            out.data_ptr() if out is not None else None, out.stride(0) if out is not None else 0,
            out.stride(1) if out is not None else 0, float(kv_scale))
    if hfmt:   # compact host rows (host-resident entries / overlapped claims dequantize; needs res)
        assert res is not None
        rc = _HF[0](*args, lflag.data_ptr() if lflag is not None else None,
                    sflag.data_ptr() if sflag is not None else None, int(hfmt))
    elif lflag is not None or sflag is not None or _OV is not None:   # (also strips FRESH marks without flags)
        rc = _OV[0](*args, lflag.data_ptr() if lflag is not None else None,
                    sflag.data_ptr() if sflag is not None else None)
    else:
        rc = lib()(*args)
    if rc != 0:
        raise RuntimeError(f"offload_split_launch failed: {rc}")


# ---- request-packed decode (csrc pk_union_kernel / pk_attn_kernel)
PK_GMAX, PK_UNION_MAX = 6, 12 * 1024    # tokens per request; G x TOPK selections per request


def pk_available():
    lib()
    return _PK is not None


def pk_ok(gmax, topk, R):
    return pk_available() and 1 <= gmax <= PK_GMAX and gmax * topk <= PK_UNION_MAX and R < (1 << 26)


def pk_union(idx, ql, cu, gmax, nreq, R, u, mk, cnt, nmax):
    """idx [T, TOPK] int32 host rows (-1 / >= R invalid). Request r = tokens [cu[r], cu[r+1]) (cu None: r*ql ..
    min(r*ql + ql, T)), at most gmax tokens. -> u [nreq*nmax] int32 distinct valid rows per request (-1 padded),
    mk [nreq*nmax] uint8 bitmask of the request's tokens that selected the row, cnt [nreq] int32 #distinct"""
    T, topk = idx.shape
    assert idx.dtype == torch.int32 and idx.stride(1) == 1 and u.dtype == torch.int32 and mk.dtype == torch.uint8
    lib()
    rc = _PK[0](idx.data_ptr(), idx.stride(0), nreq, T, ql, cu.data_ptr() if cu is not None else None, gmax, R, topk,
                nmax, u.data_ptr(), mk.data_ptr(), cnt.data_ptr(), torch.cuda.current_stream(idx.device).cuda_stream)
    if rc != 0:
        raise RuntimeError(f"offload_pk_union_launch failed: {rc}")


def pk_attend(q, host, cache, spill, u, mk, res, cnt, cu, ql, gmax, nreq, nmax, ws, ks, sm_scale, R, cfg=2, out=None,
              kv_scale=1.0, sem=None):
    """q [T, 32, 576] bf16 (16 B aligned rows); u / mk / cnt from pk_union; res [nreq*nmax] resolved locations of u
    (None: rows from host[u]). sm_scale includes log2(e). cfg: 2 | 3 stages of 16-row tiles, +10 = 32-row tiles (more
    shared memory than sm_89 has). Output modes:
      out, ks = 1:            out [T, 32, 512] bf16 normalized (x kv_scale), ws unused
      out, ks > 1, ws, sem:   split-K partials in ws merged in-kernel by the last CTA of each (request, head group);
                              sem int32 [>= nreq * 32 / HG] zeroed once (the kernel leaves it zeroed)
      out None, ws:           partials only, ws = (m, l, acc) of >= T*ks*32 rows (_combine_kernel SPLIT=ks)"""
    T, H, D = q.shape
    assert H == 32 and D == 576 and q.dtype == torch.bfloat16 and q.stride(2) == 1
    assert q.data_ptr() % 16 == 0 and q.stride(0) % 8 == 0 and q.stride(1) % 8 == 0, "q rows must be 16 B aligned"
    lib()
    rc = _PK[1](q.data_ptr(), host.data_ptr(), cache.data_ptr(), spill.data_ptr(), u.data_ptr(), mk.data_ptr(),
                res.data_ptr() if res is not None else None, cnt.data_ptr(),
                cu.data_ptr() if cu is not None else None,
                *((None,) * 3 if ws is None else (ws[0].data_ptr(), ws[1].data_ptr(), ws[2].data_ptr())),
                nreq, T, ql, gmax, R, q.stride(0), q.stride(1), nmax, ks, float(sm_scale), cfg,
                torch.cuda.current_stream(q.device).cuda_stream,
                out.data_ptr() if out is not None else None, out.stride(0) if out is not None else 0,
                out.stride(1) if out is not None else 0, float(kv_scale), sem.data_ptr() if sem is not None else None)
    if rc != 0:
        raise RuntimeError(f"offload_pk_attn_launch failed: {rc}")
