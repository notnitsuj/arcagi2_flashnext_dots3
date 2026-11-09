"""ARC_AG_U16=1 (default off): smaller all-gather payload for the TP-sharded DSA indexer (shard_indexer.sharded_topk).

The payload per row is [kl fp32 values | kl int32 positions] (8 B per candidate). Every position is j*P + rank with
j < Lc = ceil(max_model_len / P) <= 65536, and rank is implied by the payload's slot in the rank-major all-gather
output, so only the local column j needs to cross the link: [kl fp32 values | kl uint16 j] (6 B per candidate, -25%
all-gather bytes: ~12.5 of the ~50 MB per rank per decode step on 4xL4). pack() runs before the all-gather, unpack()
after it rebuilds the original [P*n, 2*kl] payload bit for bit (values copied, positions = j*P + source rank), so the
merge and everything downstream are unchanged (bit-exact; dots3/test_ag_u16.py). Needs kl even and Lc <= 65536,
else the caller keeps the int32 payload.
"""
import os

import torch
import triton
import triton.language as tl

ENABLED = os.environ.get("ARC_AG_U16", "0") == "1"


@triton.jit
def _pack_kernel(pk_ptr, out_ptr, kl, P, rank, BLOCK: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = c < kl
    src = pk_ptr + r * (2 * kl)
    dst = out_ptr + r * (3 * kl // 2)                    # fp32 words per packed row
    tl.store(dst + c, tl.load(src + c, mask=m), mask=m)
    pos = tl.load(src.to(tl.pointer_type(tl.int32)) + kl + c, mask=m)
    j = (pos - rank) // P
    tl.store((dst + kl).to(tl.pointer_type(tl.int16)) + c, j.to(tl.int16), mask=m)   # low 16 bits (j <= 65535)


@triton.jit
def _unpack_kernel(in_ptr, out_ptr, n, kl, P, BLOCK: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)                    # row of the gathered payload, rank-major: src rank = r // n
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = c < kl
    src = in_ptr + r * (3 * kl // 2)
    dst = out_ptr + r * (2 * kl)
    tl.store(dst + c, tl.load(src + c, mask=m), mask=m)
    j = tl.load((src + kl).to(tl.pointer_type(tl.int16)) + c, mask=m).to(tl.int32) & 0xFFFF
    tl.store(dst.to(tl.pointer_type(tl.int32)) + kl + c, j * P + (r // n).to(tl.int32), mask=m)


def usable(kl, Lc):
    return ENABLED and kl % 2 == 0 and Lc <= 65536


def pack(pk, P, rank):
    """pk [n, 2*kl] fp32 ([values | int32 positions j*P+rank]) -> [n, 3*kl/2] fp32 words ([values | uint16 j])"""
    n, w = pk.shape
    kl = w // 2
    out = torch.empty(n, 3 * kl // 2, dtype=torch.float32, device=pk.device)
    BLOCK = 1024
    _pack_kernel[(n, triton.cdiv(kl, BLOCK))](pk, out, kl, P, rank, BLOCK=BLOCK, num_warps=4)
    return out


def unpack(allp, n, P):
    """all-gathered [P*n, 3*kl/2] (rank-major) -> [P*n, 2*kl] fp32 payload identical to gathering the int32 form"""
    rows, w = allp.shape
    kl = w * 2 // 3
    out = torch.empty(rows, 2 * kl, dtype=torch.float32, device=allp.device)
    BLOCK = 1024
    _unpack_kernel[(rows, triton.cdiv(kl, BLOCK))](allp, out, n, kl, P, BLOCK=BLOCK, num_warps=4)
    return out
