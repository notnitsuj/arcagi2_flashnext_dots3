import torch
import triton
import triton.language as tl

@triton.jit
def _bf16mm_kernel(x_ptr, w_ptr, y_ptr, p_ptr, M, N, K, KC, stride_xm, stride_wn, stride_ym, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SPLIT: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_m = tl.arange(0, BM)
    nmask = offs_n < N
    mmask = offs_m < M
    k0 = pid_s * KC
    acc = tl.zeros((BN, BM), dtype=tl.float32)
    ok = tl.arange(0, BK)
    wp = w_ptr + offs_n.to(tl.int64)[:, None] * stride_wn + (k0 + ok)[None, :]
    xp = x_ptr + offs_m[None, :] * stride_xm + (k0 + ok)[:, None]
    for kk in range(0, KC, BK):
        w = tl.load(wp + kk, mask=nmask[:, None], other=0.0, eviction_policy='evict_first')
        xt = tl.load(xp + kk, mask=mmask[None, :], other=0.0)
        acc = tl.dot(w, xt, acc)
    om = mmask[None, :] & nmask[:, None]
    if SPLIT == 1:
        tl.store(y_ptr + offs_m[None, :] * stride_ym + offs_n[:, None], acc.to(tl.bfloat16), mask=om)
    else:
        tl.store(p_ptr + (pid_s * M + offs_m[None, :]) * N + offs_n[:, None], acc, mask=om)

@triton.jit
def _reduce_kernel(p_ptr, y_ptr, MN, N, stride_ym, S: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < MN
    acc = tl.load(p_ptr + i, mask=m, other=0.0)
    for s in tl.static_range(1, S):
        acc += tl.load(p_ptr + s * MN + i, mask=m, other=0.0)
    r = i // N
    c = i % N
    tl.store(y_ptr + r * stride_ym + c, acc.to(tl.bfloat16), mask=m)

def mm(x, w, cfg, out=None):
    BN, BK, S, nw, ns = cfg
    M, K = x.shape
    N = w.shape[0]
    assert K % (S * BK) == 0 and w.stride(1) == 1 and (x.stride(1) == 1) and (M <= 64)
    BM = max(16, triton.next_power_of_2(M))
    y = torch.empty(M, N, dtype=torch.bfloat16, device=x.device) if out is None else out
    if M == 0:
        return y
    KC = K // S
    p = torch.empty(S, M, N, dtype=torch.float32, device=x.device) if S > 1 else y
    _bf16mm_kernel[triton.cdiv(N, BN), S](x, w, y, p, M, N, K, KC, x.stride(0), w.stride(0), y.stride(0), BM=BM, BN=BN, BK=BK, SPLIT=S, num_warps=nw, num_stages=ns)
    if S > 1:
        blk = 1024
        _reduce_kernel[triton.cdiv(M * N, blk),](p, y, M * N, N, y.stride(0), S=S, BLOCK=blk, num_warps=4)
    return y

def sweep_cfgs(M, K, N, quick=False):
    out = []
    for BN in (16, 32, 64, 128):
        for BK in (64, 128, 256):
            for S in (1, 2, 4, 5, 8, 10, 16, 20):
                if K % (S * BK):
                    continue
                ctas = triton.cdiv(N, BN) * S
                if ctas < 40 or ctas > 2400:
                    continue
                for nw in (4,) if quick else (2, 4, 8):
                    for ns in (3,) if quick else (2, 3, 4, 5):
                        if BN * BK * 2 * ns > 96 * 1024:
                            continue
                        out.append((BN, BK, S, nw, ns))
    return out
