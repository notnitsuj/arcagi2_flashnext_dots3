import os
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
G = 128
SILU_LOCAL = os.environ.get('FN_MOE_SILU_LOCAL') == '1'

@triton.jit
def _silu_local_kernel(in_ptr, out_ptr, ids_ptr, map_ptr, R, N, E, ROWS: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    rv = r < R
    e = tl.load(ids_ptr + r, mask=rv, other=-1).to(tl.int64)
    rv = rv & (e >= 0) & (e < E)
    rv = rv & (tl.load(map_ptr + e, mask=rv, other=-1) >= 0)
    m = rv[:, None] & (c < N)[None, :]
    base = in_ptr + r.to(tl.int64)[:, None] * (2 * N) + c[None, :]
    x = tl.load(base, mask=m, other=0.0).to(tl.float32)
    y = tl.load(base + N, mask=m, other=0.0).to(tl.float32)
    s = tl.math.div_rn(x, 1.0 + libdevice.exp(-x))
    o = (s.to(out_ptr.dtype.element_ty).to(tl.float32) * y).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + r.to(tl.int64)[:, None] * N + c[None, :], o, mask=m)

def silu_local(activation, output, input, topk_ids=None, expert_map=None, **_):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation, apply_moe_activation
    R, N = output.shape
    if topk_ids is None or expert_map is None or activation != MoEActivation.SILU or (R == 0):
        return apply_moe_activation(activation, output, input, topk_ids=topk_ids, expert_map=expert_map)
    ids = topk_ids.reshape(-1)
    assert ids.numel() == R and input.is_contiguous() and output.is_contiguous()
    rows, block = (4, 256)
    _silu_local_kernel[triton.cdiv(R, rows), triton.cdiv(N, block)](input, output, ids, expert_map, R, N, expert_map.numel(), ROWS=rows, BLOCK=block, num_warps=4)
    return output

def codes_from_p4(p4, n_out, kg):
    w = p4.view(n_out, kg, 1, 32).to(torch.int32) & 65535
    j = torch.arange(4, device=p4.device).view(1, 1, 4, 1)
    return (w >> 4 * j & 15).reshape(n_out, kg * 128).to(torch.uint8)

def _gptq_pack(c):
    E, n, k = c.shape
    q = c.transpose(1, 2).contiguous().view(E, k // 8, 8, n).to(torch.int64)
    sh = (torch.arange(8, device=c.device) * 4).view(1, 1, 8, 1)
    w = (q << sh).sum(2)
    return torch.where(w >= 2 ** 31, w - 2 ** 32, w).to(torch.int32)

class MarlinPart:

    def __init__(self, exps, device):
        from vllm import _custom_ops as ops
        from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_moe_permute_scales, marlin_zero_points
        from vllm.scalar_type import scalar_types
        self.n = len(exps)
        cg, sg, zg, cd, sd, zd = ([], [], [], [], [], [])
        for ex in exps:
            assert int(ex['bits']) == 4
            n1, kg1 = ex['gu_s'].shape
            n2, kg2 = ex['d_s'].shape
            cg.append(codes_from_p4(ex['gu']['p4'].to(device), n1, kg1))
            cd.append(codes_from_p4(ex['d']['p4'].to(device), n2, kg2))
            sg.append(ex['gu_s'].to(device).half())
            sd.append(ex['d_s'].to(device).half())
            zg.append(ex['gu_z'].to(device).to(torch.int32))
            zd.append(ex['d_z'].to(device).to(torch.int32))
        self.I2, self.H = (n1, kg1 * G)
        self.I = n1 // 2
        w1 = _gptq_pack(torch.stack(cg))
        w2 = _gptq_pack(torch.stack(cd))
        self.w1 = ops.gptq_marlin_moe_repack(w1, self.H, self.I2, 4)
        self.w2 = ops.gptq_marlin_moe_repack(w2, self.I, self.H, 4)
        s1 = torch.stack(sg).transpose(1, 2).contiguous()
        s2 = torch.stack(sd).transpose(1, 2).contiguous()
        self.s1 = marlin_moe_permute_scales(s=s1, size_k=self.H, size_n=self.I2, group_size=G)
        self.s2 = marlin_moe_permute_scales(s=s2, size_k=self.I, size_n=self.H, group_size=G)
        self.z1 = torch.stack([marlin_zero_points(z.t().contiguous(), self.H // G, self.I2, 4) for z in zg])
        self.z2 = torch.stack([marlin_zero_points(z.t().contiguous(), self.I // G, self.H, 4) for z in zd])
        self.qt = scalar_types.uint4.id
        del cg, cd

    def nbytes(self):
        return sum((t.numel() * t.element_size() for t in (self.w1, self.w2, self.s1, self.s2, self.z1, self.z2)))

    def __call__(self, x, topk_w, topk_ids, expert_map, num_global):
        from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
        xh = (x if x.dtype == torch.float16 else x.to(torch.float16)).contiguous()
        y = fused_marlin_moe(xh, self.w1, self.w2, None, None, self.s1, self.s2, topk_w.contiguous(), topk_ids.contiguous(), self.qt, global_num_experts=num_global, expert_map=expert_map, w1_zeros=self.z1, w2_zeros=self.z2, **{'activation_func': silu_local} if SILU_LOCAL else {})
        return y
