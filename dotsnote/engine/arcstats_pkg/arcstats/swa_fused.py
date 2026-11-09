"""ARC_SWA_FUSED=1: Dots3 sliding-window (MLA-latent MQA) decode attention as one flash-decoding pass over the paged FP8
KV cache, instead of gather-to-bf16 + bmm + mask + softmax + bmm (~5.5% of an L4 decode step over 33 SWA layers).

Each decode token gets its own window as a list of physical cache rows (-1 = outside the window / unallocated page),
padded to 576 (>= window 513, a multiple of the split x block tiling). Dots3 SWA rows are 1088 wide (1024 latent + 64 rope), and sparse_offload's split-K kernel runs in its
resolved mode with every row a cache row. ARC_SWA_FUSED_CHECK=N compares against the original path for N calls.
"""
import os

import torch
import triton
import triton.language as tl

from . import sparse_offload as so

TOPK = 576
SPLIT = int(os.environ.get("ARC_SWA_FUSED_SPLIT", "3"))
CHECK = int(os.environ.get("ARC_SWA_FUSED_CHECK", "0"))
PACKED = os.environ.get("ARC_SWA_FUSED", "1") == "2"   # request-packed two-pass kernels (packed() below)
CUDA = int(os.environ.get("ARC_SWA_CUDA", "0"))        # with ARC_SWA_FUSED=2: single-pass all-FP8 CUDA kernel, cfg
SWITCH = os.environ.get("ARC_SWA_SWITCH")              # test only: file with orig | packed | cuda<cfg>, re-read every 256 calls
DUMP = os.environ.get("ARC_SWA_DUMP")                  # test only: dir for real inputs of every 397th call (q, windows)
DUMP_N = int(os.environ.get("ARC_SWA_DUMP_N", "6"))    # dumps per process


@triton.jit
def _swa_rows_kernel(block_table_ptr, seq_lens_ptr, res_ptr, bt_stride, rows_per_page, QL, WIN,
                     PAGE: tl.constexpr, TOPK: tl.constexpr, KP: tl.constexpr):
    """res[t, k] = cache row of position (q_pos - WIN + 1 + k) for decode token t = (req, j), or -1."""
    t = tl.program_id(0)
    req = t // QL
    j = t % QL
    qpos = tl.load(seq_lens_ptr + req).to(tl.int32) - QL + j
    k = tl.arange(0, KP)
    pos = qpos - WIN + 1 + k
    ok = (k < WIN) & (pos >= 0) & (qpos >= 0)
    page = tl.load(block_table_ptr + req * bt_stride + pos // PAGE, mask=ok, other=-1).to(tl.int64)
    ok = ok & (page >= 0)
    row = page * rows_per_page + pos % PAGE
    tl.store(res_ptr + t * TOPK + k, tl.where(ok, row, -1).to(tl.int32), mask=k < TOPK)


_SCALE = {}
_WS = {}
BN = int(os.environ.get("ARC_SWA_FUSED_BN", "32"))
WARPS = int(os.environ.get("ARC_SWA_FUSED_WARPS", "8"))
STAGES = int(os.environ.get("ARC_SWA_FUSED_STAGES", "1"))   # 2 stages need 102.5 KB smem > L4 101 KB


def _ws(n, dn, device):
    """split-K partials for DN-wide accumulators, one grow-only buffer per device shared by all SWA layers"""
    w = _WS.get(device)
    if w is None or w[0].numel() < n or w[2].numel() < n * dn:
        w = _WS[device] = (torch.empty(n, device=device), torch.empty(n, device=device),
                           torch.empty(n * dn, device=device))
    return w


def install():
    from vllm.models.dots3_note.nvidia import attention as da
    impl = da.Dots3NoteTritonMLAImpl
    orig = impl._forward_swa_mqa
    state = {"checked": 0}

    def fwd(self, q, kv_cache, decode, k_scale):
        kv_raw = kv_cache
        if not state.get("seen"):
            state["seen"] = 1
            print(f"[swa_fused] first call q={tuple(q.shape)} {q.dtype} kv={tuple(kv_cache.shape)} {kv_cache.dtype} "
                  f"strides={kv_cache.stride()} win={self.sliding_window} ql={decode.query_len}", flush=True)
        num_tokens, H, D = q.shape
        ql = decode.query_len
        page = kv_cache.shape[1]
        if kv_cache.dtype == torch.uint8:                  # vLLM keeps FP8 caches as raw bytes
            kv_cache = kv_cache.view(so.FP8)
        DN = self.kv_lora_rank
        DR = D - DN
        if (kv_cache.dtype != so.FP8 or kv_cache.shape[2] != D or kv_cache.stride(2) != 1 or kv_cache.stride(1) != D
                or kv_cache.stride(0) % 16 or num_tokens % ql or DN & (DN - 1) or DR & (DR - 1) or DR <= 0):
            return _path("orig (layout)", orig(self, q, kv_raw, decode, k_scale))
        # pages may carry more than PAGE rows of bytes (DSA index pages packed beside them, e.g. TP2): the packed and
        # CUDA kernels take the page stride as is; the per-token split-K path needs it in whole rows
        split_ok = kv_cache.stride(0) % D == 0 and self.sliding_window <= TOPK
        nreq = num_tokens // ql
        key = id(k_scale)
        if key not in _SCALE:
            _SCALE[key] = float(k_scale)                    # static after load: one sync per layer, ever
        kv_scale = _SCALE[key]
        pk, cu = PACKED, CUDA
        if SWITCH or DUMP:
            state["n"] = n = state.get("n", 0) + 1
            if SWITCH and n % 256 == 1:
                try:
                    state["mode"] = open(SWITCH).read().strip()
                except OSError:
                    pass
            mode = state.get("mode")
            if mode == "orig":
                return orig(self, q, kv_raw, decode, k_scale)
            if mode:
                pk, cu = True, int(mode[4:]) if mode.startswith("cuda") else 0
            if DUMP and n % 397 == 0 and state.get("dumped", 0) < DUMP_N:
                state["dumped"] = state.get("dumped", 0) + 1
                _dump(self, q, kv_cache, decode, kv_scale, f"{DUMP}/swa_{os.getpid()}_{n}.pt")
        if pk and cu and cuda_ok(q, DN):
            return _path(f"cuda{cu}", _check(self, q, kv_raw, decode, k_scale, cuda_attn(self, q, kv_cache, decode, kv_scale, cfg=cu)))
        if pk and ql * H <= 128 and DN % PK_DV == 0 and D % PK_BD == 0:
            return _path("packed", _check(self, q, kv_raw, decode, k_scale, packed(self, q, kv_cache, decode, kv_scale)))
        if not split_ok:
            return _path("orig (page stride)", orig(self, q, kv_raw, decode, k_scale))
        res = torch.empty((num_tokens, TOPK), dtype=torch.int32, device=q.device)
        bt = decode.block_table[:nreq]
        _swa_rows_kernel[(num_tokens,)](bt, decode.seq_lens[:nreq], res, bt.stride(0), kv_cache.stride(0) // D, ql,
                                        self.sliding_window, PAGE=page, TOPK=TOPK, KP=triton.next_power_of_2(TOPK),
                                        num_warps=4)
        BH = 16
        HB = triton.cdiv(H, BH)
        HP = HB * BH
        ws = _ws(num_tokens * SPLIT * HP, DN, q.device)
        out = torch.empty((num_tokens, H, DN), dtype=q.dtype, device=q.device)
        so._sparse_mla_offload_split_kernel[(num_tokens, HB, SPLIT)](
            q, kv_cache, kv_cache, res, res, ws[0], ws[1], ws[2], 1, 1 << 62,
            q.stride(0), q.stride(1), TOPK, H, self.scale * 1.4426950408889634 * kv_scale, res, kv_cache,
            TOPK=TOPK, SPLIT=SPLIT, BLOCK_H=BH, BLOCK_N=BN, HP=HP, DN=DN, DR=DR, W=1, ROWMAJOR=2,
            RES=1, num_warps=WARPS, num_stages=STAGES)
        so._combine_kernel[(num_tokens, HB)](ws[0], ws[1], ws[2], out, out.stride(0), out.stride(1), H, kv_scale,
                                             SPLIT=SPLIT, BLOCK_H=BH, HP=HP, DN=DN, num_warps=8)
        return _check(self, q, kv_raw, decode, k_scale, out)

    def _path(name, res):
        if name not in state:   # once per process and path: which kernel actually serves the SWA decode calls
            state[name] = 1
            print(f"[swa_fused] path {name}", flush=True)
        return res

    def _check(self, q, kv_raw, decode, k_scale, out):
        num_tokens, ql = q.shape[0], decode.query_len
        state["calls"] = state.get("calls", 0) + 1
        if CHECK and state["checked"] < CHECK and state["calls"] % 97 == 0:   # sampled: skip warmup-only calls
            state["checked"] += 1
            ref = orig(self, q, kv_raw, decode, k_scale)[0]
            err = float((out.float() - ref.float()).abs().max() / ref.float().abs().max().clamp_min(1e-6))
            if err > 2e-2 or state["checked"] % 10 == 1:
                print(f"[swa_fused] check {state['checked']} tokens={num_tokens} ql={ql} rel.max={err:.2e}", flush=True)
        return out, None

    impl._forward_swa_mqa = fwd
    print(f"[swa_fused] installed on {impl.__module__}.{impl.__name__}", flush=True)


# ---- ARC_SWA_FUSED=2: request-packed two-pass variant. All QL x H query rows of a request share one union window
# (WIN + QL - 1 positions), as in vLLM's gather+bmm path, but read straight from the FP8 pages: pass 1 writes masked
# fp32 scores [req, QL*H, NK] (QK over the full 1088-wide row in BD chunks), pass 2 does the softmax and P.V for one
# DV slice of the latent per program. KV is read twice in FP8 instead of gathered to bf16 and read again.
PK_BM_KEYS = int(os.environ.get("ARC_SWA_PK_BN", "64"))
PK_BD = int(os.environ.get("ARC_SWA_PK_BD", "64"))
PK_DV = int(os.environ.get("ARC_SWA_PK_DV", "128"))
PK_BN2 = int(os.environ.get("ARC_SWA_PK_BN2", "64"))
PK_W1 = int(os.environ.get("ARC_SWA_PK_W1", "4"))
PK_W2 = int(os.environ.get("ARC_SWA_PK_W2", "4"))
PK_S1 = int(os.environ.get("ARC_SWA_PK_S1", "3"))
PK_S2 = int(os.environ.get("ARC_SWA_PK_S2", "2"))


@triton.jit
def _swa_pk_scores_kernel(q_ptr, kv_ptr, bt_ptr, sl_ptr, s_ptr, stride_q_t, stride_q_h, bt_stride, row_stride,
                          H, WIN, NK, sm_scale,
                          QL: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr, D: tl.constexpr,
                          PAGE: tl.constexpr, NKP: tl.constexpr):
    r = tl.program_id(0)
    kb = tl.program_id(1)
    m = tl.arange(0, BM)
    j = m // H
    hh = m % H
    mm = j < QL
    sl = tl.load(sl_ptr + r).to(tl.int32)
    p0 = sl - QL - WIN + 1                                 # union window start (window of query j=0)
    k = kb * BN + tl.arange(0, BN)
    pos = p0 + k
    kv_ok = (k < NK) & (pos >= 0)
    page = tl.load(bt_ptr + r * bt_stride + pos // PAGE, mask=kv_ok, other=-1).to(tl.int64)
    kv_ok = kv_ok & (page >= 0)
    row = tl.where(kv_ok, page * row_stride + (pos % PAGE) * D, 0)
    qrow = q_ptr + (r * QL + j).to(tl.int64) * stride_q_t + hh * stride_q_h
    d = tl.arange(0, BD)
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for d0 in range(0, D, BD):
        qa = tl.load(qrow[:, None] + d0 + d[None, :], mask=mm[:, None], other=0.0)
        ka = tl.load(kv_ptr + row[None, :] + d0 + d[:, None], mask=kv_ok[None, :], other=0.0).to(tl.bfloat16)
        acc = tl.dot(qa, ka, acc)
    qpos = sl - QL + j
    ok = kv_ok[None, :] & (pos[None, :] <= qpos[:, None]) & (pos[None, :] > qpos[:, None] - WIN) & mm[:, None]
    tl.store(s_ptr + (r * BM + m[:, None]) * NKP + k[None, :], tl.where(ok, acc * sm_scale, -1.0e30))


@triton.jit
def _swa_pk_pv_kernel(kv_ptr, bt_ptr, sl_ptr, s_ptr, out_ptr, stride_o_t, stride_o_h, bt_stride, row_stride,
                      H, WIN, NK, kv_scale,
                      QL: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, DV: tl.constexpr, D: tl.constexpr,
                      PAGE: tl.constexpr, NKP: tl.constexpr):
    r = tl.program_id(0)
    dv0 = tl.program_id(1) * DV
    m = tl.arange(0, BM)
    j = m // H
    hh = m % H
    mm = j < QL
    sl = tl.load(sl_ptr + r).to(tl.int32)
    p0 = sl - QL - WIN + 1
    dv = tl.arange(0, DV)
    e_max = tl.zeros([BM], dtype=tl.float32) - 1.0e30
    e_sum = tl.zeros([BM], dtype=tl.float32)
    acc = tl.zeros([BM, DV], dtype=tl.float32)
    for k0 in range(0, NK, BN):
        k = k0 + tl.arange(0, BN)
        pos = p0 + k
        kv_ok = (k < NK) & (pos >= 0)
        page = tl.load(bt_ptr + r * bt_stride + pos // PAGE, mask=kv_ok, other=-1).to(tl.int64)
        kv_ok = kv_ok & (page >= 0)
        row = tl.where(kv_ok, page * row_stride + (pos % PAGE) * D, 0)
        s = tl.load(s_ptr + (r * BM + m[:, None]) * NKP + k[None, :])
        n_max = tl.maximum(tl.max(s, 1), e_max)
        re = tl.exp2(e_max - n_max)
        p = tl.where(s > -1.0e29, tl.exp2(s - n_max[:, None]), 0.0)
        v = tl.load(kv_ptr + row[:, None] + dv0 + dv[None, :], mask=kv_ok[:, None], other=0.0).to(tl.bfloat16)
        acc = acc * re[:, None] + tl.dot(p.to(tl.bfloat16), v)
        e_sum = e_sum * re + tl.sum(p, 1)
        e_max = n_max
    acc = acc / tl.maximum(e_sum, 1e-30)[:, None] * kv_scale
    t = (r * QL + j).to(tl.int64)
    tl.store(out_ptr + t[:, None] * stride_o_t + hh[:, None] * stride_o_h + dv0 + dv[None, :], acc.to(tl.bfloat16),
             mask=mm[:, None])


_SW = {}


def packed(self, q, kv_cache, decode, kv_scale):
    """kv_cache already an FP8 [blocks, PAGE, D] view; returns [T, H, DN] bf16"""
    num_tokens, H, D = q.shape
    ql = decode.query_len
    nreq = num_tokens // ql
    DN = self.kv_lora_rank
    win = self.sliding_window
    NK = win + ql - 1
    BM = max(16, triton.next_power_of_2(ql * H))
    NKP = triton.cdiv(NK, PK_BM_KEYS) * PK_BM_KEYS
    n = nreq * BM * NKP
    s = _SW.get(q.device)
    if s is None or s.numel() < n:
        s = _SW[q.device] = torch.empty(n, dtype=torch.float32, device=q.device)
    bt = decode.block_table[:nreq]
    sl = decode.seq_lens[:nreq]
    rs = kv_cache.stride(0)                                # page stride in elements (need not be PAGE * D)
    _swa_pk_scores_kernel[(nreq, NKP // PK_BM_KEYS)](
        q, kv_cache, bt, sl, s, q.stride(0), q.stride(1), bt.stride(0), rs, H, win, NK,
        self.scale * 1.4426950408889634 * kv_scale, QL=ql, BM=BM, BN=PK_BM_KEYS, BD=PK_BD, D=D,
        PAGE=kv_cache.shape[1], NKP=NKP, num_warps=PK_W1, num_stages=PK_S1)
    out = torch.empty((num_tokens, H, DN), dtype=q.dtype, device=q.device)
    _swa_pk_pv_kernel[(nreq, DN // PK_DV)](
        kv_cache, bt, sl, s, out, out.stride(0), out.stride(1), bt.stride(0), rs, H, win, NK, kv_scale,
        QL=ql, BM=BM, BN=PK_BN2 if NKP % PK_BN2 == 0 else PK_BM_KEYS, DV=PK_DV, D=D, PAGE=kv_cache.shape[1],
        NKP=NKP, num_warps=PK_W2, num_stages=PK_S2)
    return out


# ---- ARC_SWA_FUSED=2 + ARC_SWA_CUDA=cfg: single-pass all-FP8 CUDA kernel (csrc/swa_attn.cu, prebuilt swa_attn.so built
# by dots3/build_swa_attn.sh). One CTA per decode token (16 heads) over its own window, BN=32 rows per tile; Q.K and
# P.V both on e4m3 mma.sync with the cache bytes as the B operand (no FP8 -> FP16 conversion), the QL sibling CTAs of a
# request read the window from DRAM once (L2). cfg 1 = q and P as hi+lo e4m3 pairs (~bf16 accuracy), 2 = P hi only,
# 3 = q and P hi only. Falls back to the Triton packed kernels when the .so or the shapes do not fit.
_CU = {}


def _dump(self, q, kv_cache, decode, kv_scale, path):
    """real SWA decode inputs with the window pages compacted: kv [P, PAGE, D] uint8, bt remapped into it"""
    T = q.shape[0]
    ql = decode.query_len
    nreq = T // ql
    bt = decode.block_table[:nreq].cpu()
    sl = decode.seq_lens[:nreq].cpu()
    page = kv_cache.shape[1]
    win = self.sliding_window
    used = {}
    bt2 = torch.full_like(bt, -1)
    for r in range(nreq):
        lo = max(int(sl[r]) - ql - win + 1, 0)
        for pi in range(lo // page, max(int(sl[r]) - 1, 0) // page + 1):
            pg = int(bt[r, pi])
            if pg >= 0:
                bt2[r, pi] = used.setdefault(pg, len(used))
    pages = torch.tensor(list(used), dtype=torch.long, device=kv_cache.device)
    torch.save(dict(q=q.cpu(), kv=kv_cache.view(torch.uint8)[pages].cpu(), bt=bt2, sl=sl, ql=ql, win=win,
                    scale=float(self.scale), kv_scale=kv_scale, dn=self.kv_lora_rank), path)


def cuda_lib():
    if "f" not in _CU:
        import ctypes
        p = os.environ.get("ARC_SWA_CUDA_SO") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "swa_attn.so")
        try:
            f = ctypes.CDLL(p).swa_attn_launch
            f.restype = ctypes.c_int
            f.argtypes = ([ctypes.c_void_p] * 5 + [ctypes.c_longlong] * 5 + [ctypes.c_int] * 7 +
                          [ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_void_p])
            _CU["f"] = f
        except OSError as ex:
            _CU["f"] = None
            print(f"[swa_fused] CUDA kernel unavailable ({ex!r}); Triton packed path", flush=True)
    return _CU["f"]


def cuda_ok(q, DN):
    T, H, D = q.shape
    return (H % 16 == 0 and D == 1088 and DN == 1024 and q.dtype == torch.bfloat16 and q.stride(2) == 1
            and q.stride(0) % 8 == 0 and q.stride(1) % 8 == 0 and q.data_ptr() % 16 == 0 and cuda_lib() is not None)


def cuda_attn(self, q, kv_cache, decode, kv_scale, cfg=None):
    """kv_cache an FP8 [blocks, PAGE, D] view (rows contiguous); returns [T, 16, 1024] bf16"""
    T, H, D = q.shape
    ql = decode.query_len
    nreq = T // ql
    bt = decode.block_table[:nreq]
    sl = decode.seq_lens[:nreq]
    assert bt.dtype == torch.int32 and sl.dtype == torch.int32 and bt.stride(1) == 1
    out = torch.empty((T, H, self.kv_lora_rank), dtype=q.dtype, device=q.device)
    rc = cuda_lib()(q.data_ptr(), kv_cache.data_ptr(), bt.data_ptr(), sl.data_ptr(), out.data_ptr(),
                    q.stride(0), q.stride(1), out.stride(0), out.stride(1), kv_cache.stride(0) * kv_cache.element_size(),
                    bt.stride(0), T, H, D, ql, self.sliding_window, kv_cache.shape[1],
                    self.scale * 1.4426950408889634 * kv_scale, kv_scale, cfg or CUDA or 1,
                    torch.cuda.current_stream(q.device).cuda_stream)
    if rc != 0:
        raise RuntimeError(f"swa_attn_launch failed: {rc}")
    return out
