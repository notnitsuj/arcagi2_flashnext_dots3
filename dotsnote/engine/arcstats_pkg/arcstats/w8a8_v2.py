"""Block-FP8 W8A8 GEMM with per-(N, K, M) L4 tiles (ARC_W8A8_V2=1, default off).

  C[M, N] = sum_kb P_kb * xs[:, kb] * ws[n // 128, kb],   P_kb = Xq[:, kb] @ W[:, kb]^T   (block [128, 128])

Default (RESCALE=0): exactly vLLM's _w8a8_triton_block_scaled_mm math (fresh fp32 dot per 128-wide K block, then
* xs * ws, fp32 accumulation over K in order), bit-identical to it at every tile on sm89 (L4 test: 420/420
shape x M x tile cases, 0 differing outputs; local sm89-codegen server: identical greedy tokens and logprobs). What
differs from vLLM: unmasked loads (K % 128 == 0 asserted, rows/cols wrapped and the store masked), one weight scale
per K block per CTA (BN | 128), and a tile table keyed by the real M bucket (64 = MTP drafts, 256 = MTP-3 verify,
1024 = prefill chunks), picked on a real L4 under a serving-like duty cycle (bench_w8a8_v2.py ab) instead of
back-to-back graph replays, which power-throttle the L4 to ~1.0-1.4 GHz and favour the wrong tiles.

RESCALE=1 (experimental, NOT for serving): one accumulator kept in units of the current block's scale,
acc_kb = acc_{kb-1} * (S_{kb-1} / S_kb) + P_kb with the tensor core accumulating into acc. Mathematically identical
and ~40 registers/thread lighter, but FP8 tensor cores accumulate with reduced precision: feeding them a large running
accumulator flips 8-15% of bf16 outputs on L4 and 36% on H100 wgmma (rel. err ~5e-3), and was never faster on L4.

Tiles: ARC_W8A8_V2_CONFIG = JSON text or file {"N:K": {"M": {"BM":..,"BN":..,"GROUP_M":..,"num_warps":..,
"num_stages":..,"RESCALE":0}|null}} (nearest M key; null or a missing shape = vLLM kernel). ARC_W8A8_V2_ALL=1 also
routes shapes without an entry (M >= 32) through DEFAULT (testing). A zero weight-block scale (all-zero padding rows
of arcstats.gmerge) is treated as 1 in the RESCALE path. Installed by patching fp8_utils.w8a8_triton_block_scaled_mm
(dense linears via vLLM's custom op, and arcstats.lmhead_fp8's head)."""
import json
import os

import torch
import triton
import triton.language as tl


@triton.jit
def _v2_kernel(A, AS, B, BS, C, M, N, K,
               stride_am, stride_asm, stride_ask, stride_bn, stride_bsn, stride_bsk, stride_cm,
               BM: tl.constexpr, BN: tl.constexpr, GROUP_M: tl.constexpr, RESCALE: tl.constexpr):
    BK: tl.constexpr = 128
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_in_group = GROUP_M * num_pid_n
    gid = pid // num_in_group
    first_m = gid * GROUP_M
    gsize = min(num_pid_m - first_m, GROUP_M)
    pid_m = first_m + (pid % gsize)
    pid_n = (pid % num_in_group) // gsize

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    am = offs_m % M   # wrapped rows (vLLM's trick: unmasked loads; the store is masked)
    bn = offs_n % N
    a_ptrs = A + am[:, None] * stride_am + offs_k[None, :]
    b_ptrs = B + bn[None, :] * stride_bn + offs_k[:, None]
    as_ptrs = AS + am * stride_asm
    bs_ptr = BS + ((pid_n * BN) // 128) * stride_bsn

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    nk = K // BK
    if RESCALE:
        sa_prev = tl.full((BM,), 1.0, tl.float32)
        sb_prev = 1.0
        for kb in range(0, nk):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            sa = tl.load(as_ptrs + kb * stride_ask)
            sb = tl.load(bs_ptr + kb * stride_bsk)
            sb = tl.where(sb == 0.0, 1.0, sb)
            f = (sa_prev / sa) * (sb_prev / sb)
            acc = tl.dot(a, b, acc * f[:, None])
            sa_prev = sa
            sb_prev = sb
            a_ptrs += BK
            b_ptrs += BK
        acc = acc * (sa_prev * sb_prev)[:, None]
    else:   # vLLM's math (two accumulators), for A/B timing inside the same harness
        for kb in range(0, nk):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            sa = tl.load(as_ptrs + kb * stride_ask)
            sb = tl.load(bs_ptr + kb * stride_bsk)
            acc += tl.dot(a, b) * sa[:, None] * sb
            a_ptrs += BK
            b_ptrs += BK

    c = acc.to(C.dtype.element_ty)
    offs_cm = pid_m * BM + tl.arange(0, BM)
    c_ptrs = C + offs_cm[:, None] * stride_cm + offs_n[None, :]
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_n[None, :] < N))


DEFAULT = {"BM": 64, "BN": 64, "GROUP_M": 8, "num_warps": 4, "num_stages": 3, "RESCALE": 0}


def v2_mm(A, B, As, Bs, cfg, output_dtype=torch.bfloat16, rescale=False):
    """A [M, K] e4m3, B [N, K] e4m3, As [M, K/128] fp32, Bs [ceil(N/128), K/128] fp32 -> C [M, N]"""
    M, K = A.shape
    N = B.shape[0]
    assert K % 128 == 0 and 128 % cfg["BN"] == 0 and A.stride(1) == 1 and B.stride(1) == 1
    C = A.new_empty((M, N), dtype=output_dtype)
    grid = (triton.cdiv(M, cfg["BM"]) * triton.cdiv(N, cfg["BN"]),)
    _v2_kernel[grid](A, As, B, Bs, C, M, N, K, A.stride(0), As.stride(0), As.stride(1), B.stride(0),
                     Bs.stride(0), Bs.stride(1), C.stride(0), BM=cfg["BM"], BN=cfg["BN"],
                     GROUP_M=cfg.get("GROUP_M", 8), RESCALE=rescale, num_warps=cfg.get("num_warps", 4),
                     num_stages=cfg.get("num_stages", 3))
    return C


def load_table(spec):
    """spec = a JSON file path or the JSON text itself (inline in a mode file's env: no path on the pod/Kaggle)"""
    if not spec:
        return {}
    if spec.lstrip().startswith("{"):
        raw = json.loads(spec)
    elif os.path.exists(spec):
        raw = json.load(open(spec))
    else:
        return {}
    return {tuple(int(v) for v in nk.split(":")): {int(m): c for m, c in t.items()} for nk, t in raw.items()}


def pick(t, M):
    return t[min(t, key=lambda m: abs(m - M))] if t else None


def install():
    from vllm.model_executor.layers.quantization.utils import fp8_utils as fu
    table = load_table(os.environ.get("ARC_W8A8_V2_CONFIG", ""))
    every = os.environ.get("ARC_W8A8_V2_ALL") == "1"
    orig = fu.w8a8_triton_block_scaled_mm
    seen = set()

    def mm(A, B, As, Bs, block_size, output_dtype=torch.float16):
        N, K = B.shape
        M = A.numel() // A.shape[-1] if A.dim() else 0
        t = table.get((N, K))
        cfg = pick(t, M) if t else (DEFAULT if every and M >= 32 else None)
        if (cfg is None or M == 0 or list(block_size) != [128, 128] or K % 128 or 128 % cfg["BN"] or A.dtype != B.dtype
                or As.dtype != torch.float32 or A.stride(-1) != 1 or B.stride(-1) != 1):
            return orig(A, B, As, Bs, block_size, output_dtype)
        if (N, K, M) not in seen and len(seen) < 256:
            seen.add((N, K, M))
            if M in (64, 128, 256, 384, 1024):
                print(f"[w8a8-v2] N={N} K={K} M={M} cfg={cfg}", flush=True)
        out = v2_mm(A.reshape(-1, K), B, As.reshape(-1, As.shape[-1]), Bs, cfg, output_dtype,
                    rescale=bool(cfg.get("RESCALE", 0)))
        return out.view(*A.shape[:-1], N)
    fu.w8a8_triton_block_scaled_mm = mm
    print(f"[w8a8-v2] block-FP8 W8A8 GEMM (L4 tiles) for {len(table)} shapes{' (+ all others)' if every else ''}",
          flush=True)
