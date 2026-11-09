"""Achievable DRAM bandwidth on this GPU (the floor for weight-streaming kernels like the lowbit MoE): read-only
reduction, Triton streaming read (vector loads, a few bytes in flight per SM), device-to-device copy.
  python dots3/bench_dram.py [GB=1]
Prints 'dram <variant> <GB/s>'."""
import sys, time
import torch
import triton
import triton.language as tl

dev = torch.device("cuda")
GB = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
n = int(GB * 2**30) // 4
x = torch.ones(n, dtype=torch.int32, device=dev)
y = torch.empty_like(x)
out = torch.zeros(65536, dtype=torch.int32, device=dev)


@triton.jit
def _read(x_ptr, out_ptr, n, BLOCK: tl.constexpr, ITERS: tl.constexpr):
    pid = tl.program_id(0)
    acc = tl.zeros([BLOCK], dtype=tl.int32)
    base = pid.to(tl.int64) * BLOCK * ITERS
    for i in range(ITERS):
        o = base + i * BLOCK + tl.arange(0, BLOCK)
        acc ^= tl.load(x_ptr + o, mask=o < n, other=0, cache_modifier=".cg")
    tl.store(out_ptr + pid % 65536 + tl.arange(0, 1), tl.sum(acc, 0)[None])


def timeit(f, reps=10):
    f(); torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        f()
    torch.cuda.synchronize()
    return (time.time() - t0) / reps


print(f"dram {torch.cuda.get_device_name()} buffer {GB:.1f} GiB", flush=True)
print(f"dram torch_sum {n * 4 / timeit(lambda: x.sum()) / 1e9:.1f} GB/s", flush=True)
print(f"dram copy_d2d(read+write) {2 * n * 4 / timeit(lambda: y.copy_(x)) / 1e9:.1f} GB/s", flush=True)
for BLOCK, ITERS, W in ((1024, 8, 4), (2048, 8, 8), (4096, 4, 8), (1024, 32, 4), (4096, 16, 8)):
    g = (triton.cdiv(n, BLOCK * ITERS),)
    t = timeit(lambda: _read[g](x, out, n, BLOCK=BLOCK, ITERS=ITERS, num_warps=W))
    print(f"dram triton_read B{BLOCK} it{ITERS} w{W} {n * 4 / t / 1e9:.1f} GB/s", flush=True)
