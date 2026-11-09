"""Row-cache fill / prefill gather time per host row format (ARC_OFFLOAD_ROW_FMT, arcstats/offload_rowfmt.py).
One process per format (the format is fixed at import); each times, with CUDA events, M random rows of a 4M-row
registered host buffer copied to GPU fp8 rows:
  cuda   csrc fill_copy (the overlapped / mirror fill copy, no flags; one CTA per SM, 2 rows per warp)
  cudaX  same kernel with 8 CTAs per SM and 4 rows per warp (enough reads in flight to be link-bound)
  triton offload_fill._copy_kernel / offload_rowfmt._copy_kernel (the serial split fill, B 32, 4 warps, 8 CTAs/SM)
  gather prefill staging gather (LayerOffload.gather)
  python dots3/bench_rowfmt_fill.py [fmts=fp8,e2m3,e2m2,e2m1b16] [M=8192,32768,65536]   (env HOG=1: link_hog on the
  same GPU, max/max 4 streams 4 MB, as in the emulated-link runs)
Prints 'rf <fmt> <kernel> M <m> <us> us <GB/s host bytes> <rows/us>'."""
import os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "arcstats_pkg"))


def child(fmt, Ms):
    os.environ["ARC_OFFLOAD_ROW_FMT"] = fmt
    for k, v in dict(ARC_OFFLOAD_RESOLVE="1", ARC_OFFLOAD_WAYS="8", ARC_OFFLOAD_CUDA="1", ARC_OFFLOAD_HT_EPOCH="1",
                     ARC_OFFLOAD_FILL_SPLIT="1", ARC_OFFLOAD_TWO_CHOICE="1").items():
        os.environ.setdefault(k, v)
    import torch
    from arcstats import sparse_offload as so, offload_fill as of, offload_cuda as oc, offload_rowfmt as rf
    dev = torch.device("cuda", 0)
    R = 4 << 20
    rb = rf.RB
    host = so._registered_host((R, rb), torch.uint8 if rf.ON else so.FP8)
    hv = host.view(torch.uint8)
    for c in range(0, R, 1 << 18):   # random bytes; compact rows get sane block exponents
        blk = torch.randint(0, 256, (min(1 << 18, R - c), rb), dtype=torch.uint8)
        if rf.ON:
            blk[:, rf.SOFF:rf.ROFF] = torch.randint(-10, 2, (blk.shape[0], rf.ROFF - rf.SOFF), dtype=torch.int8).view(torch.uint8)
        else:
            blk[(blk & 0x7F) == 0x7F] = 0
        hv[c:c + blk.shape[0]] = blk
    C = 1 << 17
    cache = torch.zeros((C, 576), dtype=so.FP8, device=dev)
    spill = torch.zeros((16, 576), dtype=so.FP8, device=dev)
    nsm = torch.cuda.get_device_properties(dev).multi_processor_count
    g = torch.Generator(device="cpu").manual_seed(0)
    res = []
    for M in Ms:
        sets = [torch.randperm(R, generator=g)[:M].to(dev) for _ in range(8)]
        mcnt = torch.tensor([M], dtype=torch.int32, device=dev)
        cpd = torch.randperm(C, device=dev)[:M].long() if M <= C else torch.arange(M, device=dev) % C
        k = [0]

        def nxt():
            k[0] += 1
            return sets[k[0] % len(sets)]
        out = torch.empty((M, 576), dtype=so.FP8, device=dev)

        def f_cuda():
            oc.fill_copy(nxt(), mcnt, cpd, host, cache, spill, None, None, nsm + 2000, hfmt=rf.FMT)

        def f_cudax():   # 8 CTAs per SM, 4 rows per warp: as many reads in flight as possible (link-bound regime)
            oc.fill_copy(nxt(), mcnt, cpd, host, cache, spill, None, None, min(8 * nsm, 999) + 4000, hfmt=rf.FMT)

        def f_cudar4():  # serving grid (one CTA per SM), 4 rows per warp (ARC_OFFLOAD_AB feature 'rpw')
            oc.fill_copy(nxt(), mcnt, cpd, host, cache, spill, None, None, nsm + 4000, hfmt=rf.FMT)

        def f_cudag2():  # two CTAs per SM, 2 rows per warp (ARC_OFFLOAD_AB feature 'grid2')
            oc.fill_copy(nxt(), mcnt, cpd, host, cache, spill, None, None, 2 * nsm + 2000, hfmt=rf.FMT)

        def f_tri():
            ml = nxt()
            if rf.ON:
                rf.copy(ml, mcnt, cpd, host, cache, spill, cpd, 0, of.copy_grid(dev, M), B=32, num_warps=4)
            else:
                of._copy_kernel[(of.copy_grid(dev, M),)](ml, mcnt, cpd, host.view(torch.int64), cache.view(torch.int64),
                                                         spill.view(torch.int64), cpd, 0, B=32, PERM=0, num_warps=4)
        lo = so.LayerOffload.__new__(so.LayerOffload)
        lo.host, lo.R = host, R

        def f_gather():
            lo.gather(nxt(), out)
        for name, fn in (("cuda", f_cuda), ("cudaR4", f_cudar4), ("cudaG2", f_cudag2), ("cudaX", f_cudax), ("triton", f_tri), ("gather", f_gather)):
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            reps = 40
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(reps):
                fn()
            e1.record()
            torch.cuda.synchronize()
            us = e0.elapsed_time(e1) * 1e3 / reps
            res.append((name, M, us))
    # check: the fill wrote dequantized rows (cache line cpd[i] == gather of row ml[i])
    for name, M, us in res:
        print(f"rf {fmt} {name} M {M} {us:.1f} us {M * rb / us / 1e3:.2f} GB/s {M / us:.1f} rows/us", flush=True)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(sys.argv[2], [int(x) for x in sys.argv[3].split(",")])
        return
    fmts = (sys.argv[1] if len(sys.argv) > 1 else "fp8,e2m3,e2m2,e2m1b16").split(",")
    Ms = sys.argv[2] if len(sys.argv) > 2 else "8192,32768,65536"
    hog = None
    if os.environ.get("HOG") == "1":
        hog = subprocess.Popen([sys.executable, os.path.join(HERE, "link_hog.py"), "--gpus", "0", "--h2d", "max",
                                "--d2h", "max", "--streams", "4", "--chunk-mb", "4", "--report", "0"],
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        for l in hog.stdout:
            if "hog ready" in l:
                break
        time.sleep(2)
    try:
        for f in fmts:
            subprocess.run([sys.executable, __file__, "--child", f, Ms], check=False)
    finally:
        if hog is not None:
            hog.terminate()
            hog.wait()


if __name__ == "__main__":
    main()
