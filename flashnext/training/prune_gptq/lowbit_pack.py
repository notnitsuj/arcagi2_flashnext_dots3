import torch

def _pack_planar(q, bits):
    R, K = q.shape
    q = q.to(torch.int32).view(R, K // 128, 128)
    out = {}
    if bits == 4:
        f = q.view(R, K // 128, 4, 32)
        w = sum(((f[:, :, j, :] & 15) << 4 * j for j in range(4)))
        out['p4'] = _to_i16(w).view(R, K // 4)
    else:
        f = q.view(R, K // 128, 8, 16)
        w = sum(((f[:, :, j, :] & 3) << 2 * j for j in range(8)))
        out['p2'] = _to_i16(w).view(R, K // 8)
        if bits == 3:
            h = sum(((f[:, :, j, :] >> 2 & 1) << j for j in range(8)))
            out['p1'] = h.to(torch.uint8).view(R, K // 8)
    return out

def _to_i16(w):
    w = w.to(torch.int32)
    return torch.where(w >= 32768, w - 65536, w).to(torch.int16)
