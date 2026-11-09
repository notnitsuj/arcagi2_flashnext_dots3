"""Fused small kernels around the low-bit MoE GEMMs (ARC_LOWBIT_FUSE=1, default off; decode batches only).

The current path runs ~16 small kernels per MoE layer besides the two GEMMs (~110 us per layer at M=256 on L4,
~5 ms per step): routing (sigmoid, +bias, top-k, gather, sum, div, mul and their casts), x -> fp16 plus the per-group
row sums xs (cast, float copy, reduce), silu_and_mul plus c2 -> fp16 plus xs. Here they are three kernels:
  route:     bf16 logits -> sigmoid (fp32) -> top-k of (score + bias) -> renormalized, scaled weights + ids
  prep_x:    bf16 x -> fp16 x and xs (fp32 sum of each 128-column group)
  silu_prep: bf16 c1 = [gate | up] -> c2 = bf16(bf16(silu(gate)) * up) as fp16 (vLLM's act_and_mul rounding), and xs
Not bit-identical to the torch ops: xs and the top-k weight sums use a different fp32 summation order, sigmoid/silu
use libdevice expf (ids identical except for exact ties). Validated by a long-context fidelity test (fidelity, KL); the
run_l4 gate (lowbit_fuse_gate) checks the layer output against the current path within bf16 noise.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _route_kernel(lg_ptr, bias_ptr, w_ptr, id_ptr, M, E, stride_l, scale,
                  TOPK: tl.constexpr, EP: tl.constexpr, RENORM: tl.constexpr, ROWS: tl.constexpr):
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    rm = r < M
    e = tl.arange(0, EP)
    em = e < E
    lg = tl.load(lg_ptr + r[:, None] * stride_l + e[None, :], mask=rm[:, None] & em[None, :], other=0.0).to(tl.float32)
    s = 1.0 / (1.0 + libdevice.exp(-lg))
    b = tl.load(bias_ptr + e, mask=em, other=0.0).to(tl.float32)
    sel = tl.where(em[None, :], s + b[None, :], float("-inf"))
    tot = tl.zeros((ROWS,), dtype=tl.float32)
    ws = tl.zeros((ROWS, TOPK), dtype=tl.float32)
    ids = tl.zeros((ROWS, TOPK), dtype=tl.int32)
    kk = tl.arange(0, TOPK)
    for k in tl.static_range(TOPK):
        mx = tl.max(sel, axis=1)
        # lowest index among the maxima (torch.topk on CUDA returns the lowest index first for ties in practice)
        am = tl.min(tl.where(sel == mx[:, None], e[None, :], EP), axis=1)
        hit = e[None, :] == am[:, None]
        sv = tl.sum(tl.where(hit, s, 0.0), axis=1)
        tot += sv
        ws = tl.where(kk[None, :] == k, sv[:, None], ws)
        ids = tl.where(kk[None, :] == k, am[:, None], ids)
        sel = tl.where(hit, float("-inf"), sel)
    if RENORM:
        ws = ws / tot[:, None]
    ws = ws * scale
    tl.store(w_ptr + r[:, None] * TOPK + kk[None, :], ws, mask=rm[:, None])
    tl.store(id_ptr + r[:, None] * TOPK + kk[None, :], ids, mask=rm[:, None])


def route(logits, bias, topk, renorm, scale):
    M, E = logits.shape
    w = torch.empty(M, topk, dtype=torch.float32, device=logits.device)
    ids = torch.empty(M, topk, dtype=torch.int32, device=logits.device)
    if bias is None:
        bias = torch.zeros(E, dtype=torch.float32, device=logits.device)
    rows = 4
    _route_kernel[(triton.cdiv(M, rows),)](logits, bias, w, ids, M, E, logits.stride(0), float(scale), TOPK=topk,
                                           EP=triton.next_power_of_2(E), RENORM=bool(renorm), ROWS=rows,
                                           num_warps=4)
    return w, ids


@triton.jit
def _prep_kernel(x_ptr, xh_ptr, xs_ptr, R, K, stride_x, ROWS: tl.constexpr, GB: tl.constexpr):
    """ROWS rows x GB groups of 128 columns per program."""
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    g = tl.program_id(1) * GB + tl.arange(0, GB)
    c = tl.arange(0, 128)
    KG = K // 128
    m = (r < R)[:, None, None] & (g < KG)[None, :, None]
    off = r[:, None, None] * stride_x + g[None, :, None] * 128 + c[None, None, :]
    xh = tl.load(x_ptr + off, mask=m, other=0.0).to(tl.float16)
    tl.store(xh_ptr + r[:, None, None] * K + g[None, :, None] * 128 + c[None, None, :], xh, mask=m)
    xs = tl.sum(xh.to(tl.float32), axis=2)
    tl.store(xs_ptr + r[:, None] * KG + g[None, :], xs, mask=(r < R)[:, None] & (g < KG)[None, :])


def prep_x(x, rows=4, gb=4):
    R, K = x.shape
    xh = torch.empty(R, K, dtype=torch.float16, device=x.device)
    xs = torch.empty(R, K // 128, dtype=torch.float32, device=x.device)
    _prep_kernel[(triton.cdiv(R, rows), triton.cdiv(K // 128, gb))](x, xh, xs, R, K, x.stride(0), ROWS=rows, GB=gb,
                                                                   num_warps=4)
    return xh, xs


@triton.jit
def _silu_prep_kernel(c1_ptr, c2_ptr, xs_ptr, R, I, ROWS: tl.constexpr, GB: tl.constexpr):
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    g = tl.program_id(1) * GB + tl.arange(0, GB)
    c = tl.arange(0, 128)
    IG = I // 128
    m = (r < R)[:, None, None] & (g < IG)[None, :, None]
    col = g[None, :, None] * 128 + c[None, None, :]
    base = c1_ptr + r[:, None, None] * (2 * I) + col
    gt = tl.load(base, mask=m, other=0.0).to(tl.float32)
    up = tl.load(base + I, mask=m, other=0.0).to(tl.float32)
    sl = libdevice.div_rn(gt, 1.0 + libdevice.exp(-gt)).to(tl.bfloat16).to(tl.float32)
    c2 = (sl * up).to(tl.bfloat16).to(tl.float16)
    tl.store(c2_ptr + r[:, None, None] * I + col, c2, mask=m)
    xs = tl.sum(c2.to(tl.float32), axis=2)
    tl.store(xs_ptr + r[:, None] * IG + g[None, :], xs, mask=(r < R)[:, None] & (g < IG)[None, :])


def silu_prep(c1, rows=8, gb=4):
    R, I2 = c1.shape
    I = I2 // 2
    c2 = torch.empty(R, I, dtype=torch.float16, device=c1.device)
    xs = torch.empty(R, I // 128, dtype=torch.float32, device=c1.device)
    _silu_prep_kernel[(triton.cdiv(R, rows), triton.cdiv(I // 128, gb))](c1, c2, xs, R, I, ROWS=rows, GB=gb,
                                                                        num_warps=4)
    return c2, xs


def moe_forward(moe, x, topk_w, topk_ids, combine_kernel=True):
    """LowbitMoE.__call__ with prep_x / silu_prep (same GEMMs and tiles); combine via the bit-exact lowbit2 kernel."""
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
    from .lowbit_moe import _tile
    M = x.shape[0]
    topk = topk_ids.shape[1]
    cg, cd = _tile("gate_up", M, moe.block_m), _tile("down", M, moe.block_m)
    al = {bm: moe_align_block_size(topk_ids, bm, moe.E) for bm in {cg[0], cd[0]}}
    xh, xs = prep_x(x)
    c1 = torch.empty(M * topk, moe.gu.N, dtype=torch.bfloat16, device=x.device)
    moe.gu(xh, c1, topk_w.reshape(-1), *al[cg[0]], topk, True, False, *cg, xs=xs)
    c2h, xs2 = silu_prep(c1)
    c3 = torch.empty(M * topk, moe.dn.N, dtype=torch.bfloat16, device=x.device)
    moe.dn(c2h, c3, topk_w.reshape(-1), *al[cd[0]], 1, False, True, *cd, xs=xs2)
    if combine_kernel and topk == 8:
        from .lowbit2 import combine
        return combine(c3, M, topk)
    return c3.view(M, topk, moe.dn.N).sum(1)
