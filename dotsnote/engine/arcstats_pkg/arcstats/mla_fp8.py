"""ARC_MLA_FP8=1: MLA's absorbed per-head matmuls (q_nope @ W_UK_T and attn_out @ W_UV, torch.bmm on bf16 copies of
kv_b_proj) read FP8 weights instead: half the weight bytes of a weight-streaming op that runs on every layer of every
decode step.

Lossless w.r.t. what vLLM computes: kv_b_proj is block-FP8 in the checkpoint (128 x 128 blocks, weight_scale_inv) and
vLLM's bf16 W_UK_T / W_UV are round_bf16(code * scale). code = W_bf16 / scale is recovered exactly (bf16 keeps 8
mantissa bits, e4m3 has 3) and checked bit-exact after the round trip; the kernel dequantizes to the same bf16 values
in registers (round_bf16(code * scale)) and multiplies in bf16 with fp32 accumulation, like the bf16 bmm.

Layers whose kv_b_proj has no usable block scale (e.g. Marlin-repacked) keep the bf16 path.
"""
import os

import torch
import triton
import triton.language as tl

_N_ON = [0, 0]
_WHY = {}


@triton.jit
def _bmm_fp8_kernel(a_ptr, w_ptr, s_ptr, c_ptr, B, J, K,
                    sa_n, sa_b, sa_k, sw_n, sw_j, sw_k, ss_n, ss_0, ss_1, sc_n, sc_b, sc_j,
                    SCALE_K: tl.constexpr, BB: tl.constexpr, BJ: tl.constexpr, BK: tl.constexpr):
    """C[n, b, j] = sum_k A[n, b, k] * bf16(W[n, j, k] * S) with S = s[n, j, k // 128] (SCALE_K) else s[n, k, j // 128].
    BJ, BK divide 128, so a tile's scales are one vector (over j or over k)."""
    n = tl.program_id(0)
    pj = tl.program_id(1)
    pb = tl.program_id(2)
    ob = pb * BB + tl.arange(0, BB)
    oj = pj * BJ + tl.arange(0, BJ)
    jm = oj < J
    acc = tl.zeros((BB, BJ), dtype=tl.float32)
    for k0 in range(0, K, BK):
        ok = k0 + tl.arange(0, BK)
        km = ok < K
        a = tl.load(a_ptr + n * sa_n + ob[:, None] * sa_b + ok[None, :] * sa_k, mask=(ob[:, None] < B) & km[None, :],
                    other=0.0)
        w = tl.load(w_ptr + n * sw_n + oj[:, None] * sw_j + ok[None, :] * sw_k, mask=jm[:, None] & km[None, :],
                    other=0.0).to(tl.float32)
        if SCALE_K:   # per output row j, this k block
            sv = tl.load(s_ptr + n * ss_n + oj * ss_0 + (k0 // 128) * ss_1, mask=jm, other=0.0)
            wb = (w * sv[:, None]).to(tl.bfloat16)          # = vLLM's bf16 weight
        else:         # per k (row p of kv_b), this j block
            sv = tl.load(s_ptr + n * ss_n + ok * ss_0 + ((pj * BJ) // 128) * ss_1, mask=km, other=0.0)
            wb = (w * sv[None, :]).to(tl.bfloat16)
        acc = tl.dot(a.to(tl.bfloat16), tl.trans(wb), acc)
    tl.store(c_ptr + n * sc_n + ob[:, None] * sc_b + oj[None, :] * sc_j, acc.to(c_ptr.dtype.element_ty),
             mask=(ob[:, None] < B) & jm[None, :])




def bmm_fp8(a, f, out, cfg=None):
    """a [N, B, K] (any strides), f = (codes [N, J, K] fp8, scales, scale_k), out [N, B, J] (any strides)"""
    w, s, scale_k = f
    N, B, K = a.shape
    J = w.shape[1]
    BB = 16 if B <= 16 else (32 if B <= 32 else 64)
    BJ, BK, nw, ns = cfg or ((64, 64, 4, 3) if B <= 64 else (128, 64, 4, 3))   # L4 + H100 bench: wide J tiles past B 64
    grid = (N, triton.cdiv(J, BJ), triton.cdiv(B, BB))
    _bmm_fp8_kernel[grid](a, w, s, out, B, J, K, a.stride(0), a.stride(1), a.stride(2),
                          w.stride(0), w.stride(1), w.stride(2), s.stride(0), s.stride(1), s.stride(2),
                          out.stride(0), out.stride(1), out.stride(2),
                          SCALE_K=scale_k, BB=BB, BJ=BJ, BK=BK, num_warps=nw, num_stages=ns)
    return out


def _codes(wbf, scale_full):
    """wbf bf16 weight, scale_full fp32 same shape -> fp8 codes, or None unless round_bf16(code * scale) == wbf"""
    c = (wbf.float() / scale_full).to(torch.float8_e4m3fn)
    back = (c.float() * scale_full).to(torch.bfloat16)
    return c if torch.equal(back.view(torch.int16), wbf.view(torch.int16)) else None


def _convert(layer):
    """build FP8 copies of W_UK_T [N, P, L] and W_UV [N, L, V] from kv_b_proj's block scales; free the bf16 ones"""
    kvb = getattr(layer, "kv_b_proj", None)
    si = getattr(kvb, "weight_scale_inv", None) if kvb is not None else None
    uk, uv = getattr(layer, "W_UK_T", None), getattr(layer, "W_UV", None)
    if si is None or uk is None or uv is None or uk.dtype != torch.bfloat16 or uv.dtype != torch.bfloat16:
        _WHY.setdefault("why", f"kv_b_proj params {[(k, tuple(v.shape), str(v.dtype)) for k, v in kvb.named_parameters()] if kvb is not None else None}, "
                        f"W_UK_T {getattr(uk, 'dtype', None)}")
        return False
    N, P, L = uk.shape
    V = uv.shape[2]
    rows = N * (P + V)
    if si.dim() != 2 or si.shape[0] != (rows + 127) // 128 or si.shape[1] != (L + 127) // 128:
        _WHY.setdefault("why", f"scale shape {tuple(si.shape)} vs rows {rows} L {L}")
        return False
    si = si.float()
    # scale of kv_b row r = n (P + V) + d, column l
    r_uk = (torch.arange(N, device=si.device)[:, None] * (P + V) + torch.arange(P, device=si.device)[None, :])  # [N, P]
    r_uv = (torch.arange(N, device=si.device)[:, None] * (P + V) + P + torch.arange(V, device=si.device)[None, :])
    s_uk = si[r_uk // 128]                     # [N, P, L/128]  (scale of row (n, p), column block)
    s_uv = si[r_uv // 128]                     # [N, V, L/128]
    lb = torch.arange(L, device=si.device) // 128
    c_uk = _codes(uk, s_uk[:, :, lb])          # [N, P, L]
    c_uv = _codes(uv.transpose(1, 2), s_uv[:, :, lb])     # [N, V, L]
    if c_uk is None or c_uv is None:
        _WHY.setdefault("why", f"round trip not exact (uk {c_uk is not None}, uv {c_uv is not None})")
        return False
    # W_UK_T bmm: C[n, b, l] = sum_p q[n, b, p] W[n, l, p] -> codes [N, L, P], scale s[n, p, l // 128] (SCALE_K=0)
    layer._arc_uk = (c_uk.transpose(1, 2).contiguous(), s_uk.contiguous(), False)
    # W_UV bmm: C[n, b, v] = sum_l x[n, b, l] W[n, v, l] -> codes [N, V, L], scale s[n, v, l // 128] (SCALE_K=1)
    layer._arc_uv = (c_uv.contiguous(), s_uv.contiguous(), True)
    shape_uk, shape_uv = uk.shape, uv.shape
    # keep shape-only stand-ins (forward reads W_UK_T.shape); the bf16 copies are freed
    layer.W_UK_T = torch.nn.Parameter(torch.empty(1, dtype=torch.bfloat16, device=uk.device).as_strided(shape_uk, (0, 0, 0)),
                                      requires_grad=False)
    layer.W_UV = torch.nn.Parameter(torch.empty(1, dtype=torch.bfloat16, device=uv.device).as_strided(shape_uv, (0, 0, 0)),
                                    requires_grad=False)
    layer.W_UK_T._arc_fp8 = layer._arc_uk
    layer.W_UV._arc_fp8 = layer._arc_uv
    return True


class _TorchProxy:
    """mla_attention's module-level `torch`: bmm against a converted weight goes to bmm_fp8"""

    def __getattr__(self, name):
        return getattr(torch, name)

    @staticmethod
    def bmm(a, b, *, out=None):
        f = getattr(b, "_arc_fp8", None)
        if f is None:
            return torch.bmm(a, b, out=out) if out is not None else torch.bmm(a, b)
        if out is None:
            out = torch.empty((a.shape[0], a.shape[1], f[0].shape[1]), dtype=a.dtype, device=a.device)
        return bmm_fp8(a, f, out)


def install():
    from vllm.model_executor.layers.attention import mla_attention as ma
    orig = ma.MLAAttention.process_weights_after_loading

    def pwal(self, act_dtype):
        orig(self, act_dtype)
        ok = False
        try:
            ok = _convert(self)
        except Exception as ex:   # keep the bf16 path
            print(f"[mla_fp8] conversion failed: {ex!r}", flush=True)
        _N_ON[0 if ok else 1] += 1
        if sum(_N_ON) in (1, 46):
            print(f"[mla_fp8] FP8 absorbed matmuls: {_N_ON[0]} layers on, {_N_ON[1]} kept bf16 {_WHY.get('why', '')}",
                  flush=True)
    ma.MLAAttention.process_weights_after_loading = pwal
    ma.torch = _TorchProxy()
