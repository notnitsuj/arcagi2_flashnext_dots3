"""Exact top-k for the TP-sharded DSA indexer (ARC_FAST_TOPK, default off; csrc/topk_select.cu -> topk_select.so,
built by dots3/build_topk_select.sh).

  ARC_FAST_TOPK=1: local top-K_LOC per rank by one radix-select CTA per row that reads only the row's valid columns
                   and writes the all-gather payload [values | positions] directly (replaces torch.topk's multi-block
                   radix passes over the full max_model_len/P row + pos arithmetic + cat); the logits kernel then skips
                   its -inf tail stores.
  ARC_FAST_TOPK=2: also the post-all-gather merge (bitonic sort of the P*K_LOC candidates per row, written straight
                   into topk_indices_buffer; replaces strided copies + topk + gather + where + fill).
Selection is torch.topk's (same order-preserving key, -0 < +0, lowest index wins ties), so topk_indices_buffer holds
the identical index set. Level 1 keeps torch's merge (bit-identical in every bench_topk_l4 case); level 2 orders the
row by descending value then ascending gathered index, which differs from torch.topk only in the order of exactly
equal values (torch's own tie order is an artifact of its gather/sort kernels: +-0 merged, unstable small sorts).
All consumers (offload/sparse MLA attention, row-cache fill, shared S layers) treat a row as a set.
Falls back to the torch path when the library is missing (logged once)."""
import ctypes
import os

import torch

MODE = os.environ.get("ARC_FAST_TOPK", "0")
NT = int(os.environ.get("ARC_FAST_TOPK_NT", "1024"))      # local select CTA size (256 / 512 / 1024)
MNT = int(os.environ.get("ARC_FAST_TOPK_MNT", "1024"))    # merge CTA size (512 / 1024)
AGG = int(os.environ.get("ARC_FAST_TOPK_AGG", "0"))       # warp-aggregated histogram atomics
_L = None
_ERR = None


def lib():
    global _L, _ERR
    if _L is None and _ERR is None:
        p = os.environ.get("ARC_TOPK_SO") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "topk_select.so")
        try:
            L = ctypes.CDLL(p)
            fl, fm = L.topk_local_launch, L.topk_merge_launch
            fl.restype = fm.restype = ctypes.c_int
            fl.argtypes = [ctypes.c_void_p, ctypes.c_longlong, ctypes.c_void_p] + [ctypes.c_int] * 5 + \
                          [ctypes.c_void_p] + [ctypes.c_int] * 4 + [ctypes.c_void_p]
            fm.argtypes = [ctypes.c_void_p] + [ctypes.c_int] * 5 + [ctypes.c_void_p, ctypes.c_longlong, ctypes.c_int,
                                                                ctypes.c_void_p]
            _L = (fl, fm)
        except OSError as ex:
            _ERR = repr(ex)
            print(f"[fast_topk] {p} not loadable, torch.topk kept: {_ERR[:200]}", flush=True)
    return _L


def enabled(level=1):
    return MODE not in ("", "0") and int(MODE) >= level and lib() is not None


def local(lg, ctx, rank, P, kl, pk=None, nt=None, agg=None):
    """lg [rows, Lc] fp32/fp16 (col j = position j*P+rank; columns at or beyond the row's ctx are never read),
    ctx int32 [rows] -> pk [rows, 2*kl] fp32 = [top-kl values (fp32) | their positions (int32 bits)], selected
    columns in ascending column order; rows with < kl valid columns padded with -inf."""
    rows, Lc = lg.shape
    assert lg.stride(1) == 1 and ctx.dtype == torch.int32 and ctx.is_contiguous() and kl <= Lc
    if pk is None:
        pk = torch.empty(rows, 2 * kl, dtype=torch.float32, device=lg.device)
    dt = {torch.float32: 0, torch.float16: 1}[lg.dtype]
    vw = 4 if dt == 0 else 8
    vec = vw if (lg.data_ptr() % 16 == 0 and lg.stride(0) % vw == 0 and Lc % vw == 0) else 1
    rc = lib()[0](lg.data_ptr(), lg.stride(0), ctx.data_ptr(), rows, rank, P, Lc, kl, pk.data_ptr(), dt, vec,
                  nt or NT, AGG if agg is None else agg, torch.cuda.current_stream(lg.device).cuda_stream)
    if rc:
        raise RuntimeError(f"topk_local_launch: CUDA error {rc}")
    return pk


def merge(allp, n, P, kl, kk, out, nt=None):
    """allp [P*n, 2*kl] all-gathered payloads (rank-major) -> out [n, topk_tokens] int32: positions of the top kk
    (torch.topk sorted order), -1 where the value is +-inf and for columns >= kk."""
    assert allp.dtype == torch.float32 and allp.is_contiguous() and out.dtype == torch.int32 and out.stride(1) == 1
    rc = lib()[1](allp.data_ptr(), n, P, kl, kk, out.shape[1], out.data_ptr(), out.stride(0),
                  nt or MNT, torch.cuda.current_stream(allp.device).cuda_stream)
    if rc:
        raise RuntimeError(f"topk_merge_launch: CUDA error {rc}")
    return out
