"""Row-cache miss fill (host -> GPU over PCIe) on the target GPUs: is _sa_fill bandwidth- or latency-bound?
Host rows live in one shared /dev/shm buffer registered by every process (as the TP ranks share it in serving).
  python dots3/bench_fill_l4.py [NPROC=1,4] [M=4096,16384,32768]
Per (procs, M) prints 'fill <variant> procs <p> M <m> <us> us <GB/s> GB/s' (per GPU; the slowest process):
  prefill   LayerOffload._sa_prefill on M all-new rows (scan + fill + advance + resolve, the serving path, C=131072 W=8)
  g<B>w<w>  standalone gather of M rows (B rows per program, w warps; fp8 [B,512]+[B,64] loads like _sa_fill)
  g32i<w>   same with int32 loads (128+16 words per row)
  memcpy    contiguous M*576 B cudaMemcpy H2D from the registered buffer (DMA upper bound)"""
import os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, "arcstats_pkg"), HERE):
    if os.path.isdir(os.path.join(p, "arcstats")):
        sys.path.insert(0, p)

R = 2 * 1024 * 1024
D = 576


def child(gpu, nproc, Ms, start_at):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch, triton, triton.language as tl
    from arcstats import sparse_offload as so

    @triton.jit
    def gather8(host_ptr, rows_ptr, out_ptr, n, B: tl.constexpr):
        i = tl.program_id(0) * B + tl.arange(0, B)
        m = i < n
        h = tl.load(rows_ptr + i, mask=m, other=0).to(tl.int64)
        dn = tl.arange(0, 512)
        dr = 512 + tl.arange(0, 64)
        tl.store(out_ptr + i[:, None].to(tl.int64) * 576 + dn[None, :],
                 tl.load(host_ptr + h[:, None] * 576 + dn[None, :], mask=m[:, None]), mask=m[:, None])
        tl.store(out_ptr + i[:, None].to(tl.int64) * 576 + dr[None, :],
                 tl.load(host_ptr + h[:, None] * 576 + dr[None, :], mask=m[:, None]), mask=m[:, None])

    @triton.jit
    def gather32(host_ptr, rows_ptr, out_ptr, n, B: tl.constexpr):
        i = tl.program_id(0) * B + tl.arange(0, B)
        m = i < n
        h = tl.load(rows_ptr + i, mask=m, other=0).to(tl.int64)
        dn = tl.arange(0, 128)
        dr = 128 + tl.arange(0, 16)
        tl.store(out_ptr + i[:, None].to(tl.int64) * 144 + dn[None, :],
                 tl.load(host_ptr + h[:, None] * 144 + dn[None, :], mask=m[:, None]), mask=m[:, None])
        tl.store(out_ptr + i[:, None].to(tl.int64) * 144 + dr[None, :],
                 tl.load(host_ptr + h[:, None] * 144 + dr[None, :], mask=m[:, None]), mask=m[:, None])

    dev = torch.device("cuda")
    host = so._registered_host((R, D), so.FP8, key="fillbench")
    if gpu == 0 or nproc == 1:
        host.view(torch.uint8).view(-1)[:: 4096] = 1   # touch
    lo = so.LayerOffload.__new__(so.LayerOffload)
    lo._dump = None
    lo.R, lo.C, lo.W = R, 131072, 8
    lo.host = host
    lo.cache = torch.zeros((lo.C, D), dtype=so.FP8, device=dev)
    lo.tag = torch.full((lo.C,), -1, dtype=torch.int64, device=dev)
    lo.pin = torch.zeros(lo.C, dtype=torch.int32, device=dev)
    lo.rr = torch.zeros(lo.C // 8, dtype=torch.int32, device=dev)
    lo.cnt = torch.zeros(lo.C // 8, dtype=torch.int32, device=dev)
    lo.ht = torch.full(((1 << 20) + 256,), -1, dtype=torch.int64, device=dev)
    lo.mcnt = torch.zeros(1, dtype=torch.int32, device=dev)
    lo.ml = None
    g = torch.Generator(device="cpu").manual_seed(gpu)
    out = torch.empty((max(Ms), D), dtype=so.FP8, device=dev)
    h32 = host.view(torch.int32)
    o32 = out.view(torch.int32)
    variants = {}
    for M in Ms:
        sets = [torch.randperm(R, generator=g)[:M].to(dev) for _ in range(16)]
        k = [0]

        def nxt():
            k[0] += 1
            return sets[k[0] % len(sets)]
        variants[("prefill", M)] = lambda: lo._sa_prefill(nxt())
        for B, w in ((16, 4), (32, 4), (64, 4), (64, 8), (128, 8)):
            variants[(f"g{B}w{w}", M)] = (lambda B=B, w=w, M=M: gather8[(triton.cdiv(M, B),)](
                host, nxt(), out, M, B=B, num_warps=w))
        for w in (4, 8):
            variants[(f"g32i{w}", M)] = (lambda w=w, M=M: gather32[(triton.cdiv(M, 32),)](
                h32, nxt(), o32, M, B=32, num_warps=w))
        src = host.view(torch.uint8).view(-1)[: M * D]
        variants[("memcpy", M)] = lambda src=src, M=M: out.view(torch.uint8).view(-1)[: M * D].copy_(src, non_blocking=True)
    for f in variants.values():   # compile / warm
        f()
    torch.cuda.synchronize()
    while time.time() < start_at:
        time.sleep(0.01)
    res = []
    for (name, M), f in variants.items():
        while time.time() < start_at:   # all processes start each variant together
            time.sleep(0.001)
        reps = 30
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(reps):
            f()
        torch.cuda.synchronize()
        dt = (time.time() - t0) / reps
        res.append((name, M, dt))
        start_at += 2.0
    for name, M, dt in res:
        print(f"R {gpu} {name} {M} {dt * 1e6:.1f}", flush=True)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(int(sys.argv[2]), int(sys.argv[3]), [int(x) for x in sys.argv[4].split(",")], float(sys.argv[5]))
        return
    nprocs = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "1,4").split(",")]
    Ms = sys.argv[2] if len(sys.argv) > 2 else "4096,16384,32768"
    gpus = os.environ.get("BENCH_GPUS", "0,1,2,3").split(",")
    for n in nprocs:
        start_at = time.time() + float(os.environ.get("BENCH_FILL_WARM_S", "90"))
        ps = [subprocess.Popen([sys.executable, __file__, "--child", g, str(n), Ms, str(start_at)],
                               stdout=subprocess.PIPE, text=True) for g in gpus[:n]]
        rows = {}
        for p in ps:
            o, _ = p.communicate()
            for l in o.splitlines():
                if l.startswith("R "):
                    _, g, name, M, us = l.split()
                    rows.setdefault((name, int(M)), []).append(float(us))
        for (name, M), us in rows.items():
            t = max(us)
            print(f"fill {name} procs {n} M {M} {t:.1f} us {M * D / t / 1e3:.2f} GB/s", flush=True)


if __name__ == "__main__":
    main()
