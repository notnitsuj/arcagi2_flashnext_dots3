"""Decode-sized block-FP8 W8A8 GEMM (ARC_W8A8_WS=1, default off): an alternative to vLLM's Triton
_w8a8_triton_block_scaled_mm for M <= ~256 tokens on the per-rank dense shapes.

  C[M, N] = sum_kb (Xq[:, kb] @ W[:, kb]^T) * xs[:, kb] * ws[n // 128, kb]      (block [128, 128], same math as vLLM)

Differences to the vLLM kernel (all per-shape tile parameters, picked by the L4 startup gate / tuner):
  * NSUB M sub-tiles per CTA: one weight tile [BN, 128] feeds NSUB dots of [BM, 128] (e.g. 3 x 64 rows = the whole
    M=192 decode batch), so each weight byte is read once per CTA instead of once per M tile (no reliance on L2 hits).
  * SPLIT_K: the K blocks are split over SPLIT_K CTAs per output tile; partial fp32 tiles go to a workspace and the
    last-arriving CTA of a tile (int32 arrival counter, self-resetting -> no memset launch, CUDA-graph safe) sums them
    in fixed split order (deterministic) and writes bf16. For skinny-N / large-K shapes (64x5120, 128x5120, ...).
  * FUSE_Q: the activation per-token-group (1x128) FP8 quantization runs in the GEMM prologue on the bf16 input
    (bit-identical to per_token_group_fp8_quant: amax / 448, x / s, clamp, RNE cast), removing the separate quant launch.
    Costs redundant quant work per N tile, so it only wins where the GEMM has few N tiles / is launch-bound.
Without SPLIT_K the per-element math (dot over a 128 block, * xs * ws, fp32 accumulate over K in order) is exactly the
vLLM kernel's, so outputs are bit-identical; with SPLIT_K only the fp32 summation order over K changes.

Config tables (JSON): {"N:K": {"M": {"BM":64,"NSUB":3,"BN":64,"SPLIT_K":1,"FUSE_Q":0,"num_warps":4,"num_stages":3}
                                    | null, ...}} -- nearest M key (like vLLM); null = use the vLLM kernel."""
import json
import os

import torch
import triton
import triton.language as tl

FP8_MAX = 448.0


@triton.jit
def _f32_to_e4m3(y):
    """round-to-nearest-even fp32 -> e4m3 in one step (Triton's own cast goes through f16 on sm89: double rounding)"""
    u = tl.inline_asm_elementwise(
        "{ .reg .b16 lo, hi;\n cvt.rn.satfinite.e4m3x2.f32 lo, $2, $1;\n cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3;\n"
        " mov.b32 $0, {lo, hi}; }", "=r,r,r,r,r", [y], dtype=tl.uint8, is_pure=True, pack=4)
    return u.to(tl.float8e4nv, bitcast=True)


@triton.jit
def _sub_dot(acc, X, XS, rows, offs_k, kb, M, K, stride_xm, stride_xsm, b, bs, FUSE_Q: tl.constexpr):
    rmask = rows < M
    xmask = rmask[:, None] & (offs_k[None, :] < K)
    if FUSE_Q:
        # == per_token_group_fp8_quant(group 128, eps 1e-10, no ue8m0): s = max(amax, eps) / 448, q = rn(clamp(x / s))
        x = tl.load(X + rows[:, None] * stride_xm + offs_k[None, :], mask=xmask, other=0.0).to(tl.float32)
        s = tl.math.div_rn(tl.maximum(tl.max(tl.abs(x), axis=1), 1e-10), 448.0)
        # exact division: x * (1 / s) flips ~20% of codes by one ulp (bf16 inputs hit fp8 rounding ties often)
        q = _f32_to_e4m3(tl.clamp(tl.math.div_rn(x, s[:, None]), -448.0, 448.0))
    else:
        q = tl.load(X + rows[:, None] * stride_xm + offs_k[None, :], mask=xmask, other=0.0)
        s = tl.load(XS + rows * stride_xsm + kb, mask=rmask, other=0.0)
    return acc + tl.dot(q, b) * s[:, None] * bs[None, :]


@triton.jit
def _store_or_part(acc, C, PART, rows, offs_n, M, N, stride_cm, ws_m, pid_k, SPLIT_K: tl.constexpr):
    msk = (rows[:, None] < M) & (offs_n[None, :] < N)
    if SPLIT_K == 1:
        tl.store(C + rows[:, None] * stride_cm + offs_n[None, :], acc.to(C.dtype.element_ty), mask=msk)
    else:
        tl.store(PART + (pid_k * ws_m + rows[:, None]) * N + offs_n[None, :], acc, mask=msk)


@triton.jit
def _reduce_store(C, PART, rows, offs_n, M, N, stride_cm, ws_m, BM: tl.constexpr, BN: tl.constexpr,
                  SPLIT_K: tl.constexpr):
    msk = (rows[:, None] < M) & (offs_n[None, :] < N)
    tot = tl.zeros((BM, BN), dtype=tl.float32)
    for s in tl.static_range(SPLIT_K):
        tot += tl.load(PART + (s * ws_m + rows[:, None]) * N + offs_n[None, :], mask=msk, other=0.0,
                       cache_modifier=".cg")
    tl.store(C + rows[:, None] * stride_cm + offs_n[None, :], tot.to(C.dtype.element_ty), mask=msk)


@triton.jit
def _w8a8_ws_kernel(X, XS, W, WSC, C, PART, CNT, M, N, K, ws_m,
                    stride_xm, stride_xsm, stride_wn, stride_wsn, stride_cm,
                    BM: tl.constexpr, NSUB: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                    SPLIT_K: tl.constexpr, FUSE_Q: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    nkb = tl.cdiv(K, BK)
    kb_lo = pid_k * nkb // SPLIT_K
    kb_hi = (pid_k + 1) * nkb // SPLIT_K
    offs_n = pid_n * BN + tl.arange(0, BN)
    nmask = offs_n < N
    offs_bk = tl.arange(0, BK)
    r0 = pid_m * (BM * NSUB) + tl.arange(0, BM)
    r1 = r0 + BM
    r2 = r0 + 2 * BM
    r3 = r0 + 3 * BM
    acc0 = tl.zeros((BM, BN), dtype=tl.float32)
    acc1 = tl.zeros((BM, BN), dtype=tl.float32)
    acc2 = tl.zeros((BM, BN), dtype=tl.float32)
    acc3 = tl.zeros((BM, BN), dtype=tl.float32)
    w_ptrs = W + offs_n[None, :] * stride_wn + offs_bk[:, None]
    ws_ptrs = WSC + (offs_n // 128) * stride_wsn
    for kb in range(kb_lo, kb_hi):
        offs_k = kb * BK + offs_bk
        b = tl.load(w_ptrs + kb * BK, mask=nmask[None, :] & (offs_k[:, None] < K), other=0.0)
        bs = tl.load(ws_ptrs + kb, mask=nmask, other=0.0)
        acc0 = _sub_dot(acc0, X, XS, r0, offs_k, kb, M, K, stride_xm, stride_xsm, b, bs, FUSE_Q)
        if NSUB > 1:
            acc1 = _sub_dot(acc1, X, XS, r1, offs_k, kb, M, K, stride_xm, stride_xsm, b, bs, FUSE_Q)
        if NSUB > 2:
            acc2 = _sub_dot(acc2, X, XS, r2, offs_k, kb, M, K, stride_xm, stride_xsm, b, bs, FUSE_Q)
        if NSUB > 3:
            acc3 = _sub_dot(acc3, X, XS, r3, offs_k, kb, M, K, stride_xm, stride_xsm, b, bs, FUSE_Q)
    _store_or_part(acc0, C, PART, r0, offs_n, M, N, stride_cm, ws_m, pid_k, SPLIT_K)
    if NSUB > 1:
        _store_or_part(acc1, C, PART, r1, offs_n, M, N, stride_cm, ws_m, pid_k, SPLIT_K)
    if NSUB > 2:
        _store_or_part(acc2, C, PART, r2, offs_n, M, N, stride_cm, ws_m, pid_k, SPLIT_K)
    if NSUB > 3:
        _store_or_part(acc3, C, PART, r3, offs_n, M, N, stride_cm, ws_m, pid_k, SPLIT_K)
    if SPLIT_K > 1:
        tl.debug_barrier()   # every thread's partial stores precede the (thread-0) release below
        tile = pid_m * tl.num_programs(0) + pid_n
        prev = tl.atomic_add(CNT + tile, 1, sem="acq_rel", scope="gpu")
        if prev == SPLIT_K - 1:   # last arrival for this output tile: sum the splits in fixed order
            _reduce_store(C, PART, r0, offs_n, M, N, stride_cm, ws_m, BM, BN, SPLIT_K)
            if NSUB > 1:
                _reduce_store(C, PART, r1, offs_n, M, N, stride_cm, ws_m, BM, BN, SPLIT_K)
            if NSUB > 2:
                _reduce_store(C, PART, r2, offs_n, M, N, stride_cm, ws_m, BM, BN, SPLIT_K)
            if NSUB > 3:
                _reduce_store(C, PART, r3, offs_n, M, N, stride_cm, ws_m, BM, BN, SPLIT_K)
            tl.atomic_xchg(CNT + tile, 0, sem="relaxed", scope="gpu")


_WS = {}     # (device, N, K) -> [part fp32, cnt int32]; per shape: two different shapes never share partials
_KEEP = []   # every workspace ever handed out stays alive: a CUDA graph captured with it keeps using its address
WS_CNT = 4096


def _workspace(dev, N, K, need):
    key = (str(dev), N, K)
    w = _WS.get(key)
    if w is None or w[0].numel() < need:
        part = torch.empty(need, dtype=torch.float32, device=dev)
        cnt = w[1] if w is not None else torch.zeros(WS_CNT, dtype=torch.int32, device=dev)   # self-resetting
        w = [part, cnt]
        _WS[key] = w
        _KEEP.append(w)
    return w


def ws_mm(x, xs, w, wsc, cfg, out_dtype=torch.bfloat16):
    """x: [M, K] fp8 (+ xs [M, K/128] fp32) or bf16 with cfg FUSE_Q=1 (xs ignored); w [N, K] fp8; wsc [N/128, K/128]"""
    M, K = x.shape
    N = w.shape[0]
    BM, NSUB, BN, SK = cfg["BM"], cfg.get("NSUB", 1), cfg["BN"], cfg.get("SPLIT_K", 1)
    fq = int(cfg.get("FUSE_Q", 0))
    assert x.stride(1) == 1 and w.stride(1) == 1 and (fq or xs.stride(1) == 1)
    c = torch.empty((M, N), dtype=out_dtype, device=x.device)
    gm = triton.cdiv(M, BM * NSUB)
    gn = triton.cdiv(N, BN)
    if SK > 1:
        ws_m = gm * BM * NSUB
        part, cnt = _workspace(x.device, N, K, SK * ws_m * N)
        assert gm * gn <= cnt.numel()
    else:
        ws_m, part, cnt = 0, c, c
    if xs is None:
        xs = x
    _w8a8_ws_kernel[(gn, SK, gm)](
        x, xs, w, wsc, c, part, cnt, M, N, K, ws_m,
        x.stride(0), xs.stride(0) if not fq else 0, w.stride(0), wsc.stride(0), c.stride(0),
        BM=BM, NSUB=NSUB, BN=BN, BK=128, SPLIT_K=SK, FUSE_Q=fq,
        num_warps=cfg.get("num_warps", 4), num_stages=cfg.get("num_stages", 3))
    return c


def load_table(path):
    """{(N, K): {M: cfg|None}} from a JSON file {"N:K": {"M": cfg|null}}"""
    if not path or not os.path.exists(path):
        return {}
    t = {}
    for nk, d in json.load(open(path)).items():
        n, k = (int(v) for v in nk.split(":"))
        t[(n, k)] = {int(m): c for m, c in d.items()}
    return t


def pick(table_nk, M):
    return table_nk[min(table_nk, key=lambda m: abs(m - M))]


def install():
    """route vLLM's Triton W8A8 block-FP8 linears through ws_mm for the (N, K, M) entries of ARC_W8A8_WS_CONFIG,
    restricted to ARC_W8A8_WS_SHAPES ("N:K,..." chosen by the startup gate; empty = all table shapes)"""
    from vllm.model_executor.kernels.linear.scaled_mm.triton import TritonFp8BlockScaledMMKernel as TK
    table = load_table(os.environ.get("ARC_W8A8_WS_CONFIG", ""))
    keep = os.environ.get("ARC_W8A8_WS_SHAPES", "")
    if keep:
        ks = {tuple(int(v) for v in e.split(":")[:2]) for e in keep.split(",") if e}
        table = {nk: v for nk, v in table.items() if nk in ks}
    force = json.loads(os.environ.get("ARC_W8A8_WS_FORCE") or "null")   # testing: this cfg for every shape at M<=256
    orig = TK.apply_weights
    seen = set()

    def apply_weights(self, layer, x, bias=None, **kw):
        params = self._get_layer_params(layer)
        w, wsc = params.weight, params.block_scale
        N, K = w.shape
        tnk = table.get((N, K)) if force is None else {256: force, 1024: None}
        if (tnk is None or params.input_scale is not None or list(self.weight_group_shape) != [128, 128]
                or w.dtype != torch.float8_e4m3fn or wsc.dtype != torch.float32):
            return orig(self, layer, x, bias, **kw)
        x2 = x.reshape(-1, K)
        M = x2.shape[0]
        cfg = pick(tnk, M) if M else None
        if cfg is None:
            return orig(self, layer, x, bias, **kw)
        if (N, K, M) not in seen and len(seen) < 512:
            seen.add((N, K, M))
            if M in (1, 64, 128, 160, 176, 192, 256):
                print(f"[w8a8-ws] N={N} K={K} M={M} cfg={cfg}", flush=True)
        od = self.config.out_dtype or x.dtype
        if cfg.get("FUSE_Q") and x.dtype in (torch.bfloat16, torch.float16):
            out = ws_mm(x2.contiguous(), None, w, wsc, cfg, od)
        else:
            q, s = self.quant_fp8(x2, None, None, use_triton=self.use_triton)
            out = ws_mm(q, s, w, wsc, dict(cfg, FUSE_Q=0), od)
        if bias is not None:
            out = out + bias
        return out.view(*x.shape[:-1], N)
    TK.apply_weights = apply_weights
    print(f"[w8a8-ws] decode W8A8 kernel for {len(table)} shapes: {sorted(table)}", flush=True)
    return table
