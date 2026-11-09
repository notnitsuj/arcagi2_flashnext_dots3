"""PCIe host->GPU row gathers for the offload row-cache fill, on ONE GPU, no server (synthetic random rows).
What limits _sa_fill (~40 ns per missed 576 B row on L4 = ~14 GB/s) and what an L4 can do instead:
  info     PCIe link, THP / shmem hugepage settings, host RAM
  dma      cudaMemcpyAsync of M*576 contiguous bytes (DMA engine upper bound) H2D / D2H
  batch    cudaMemcpyBatchAsync of M scattered 576 B rows (DMA engine on scattered rows; needs a host-side row list,
           so it is a reference, not a graph-capturable fill)
  gather   SM gathers of M random distinct rows from 4 KiB-page /dev/shm buffers (the serving layout: BUFS layer
           buffers x ROWS rows; each call reads one buffer like one layer): legacy fill loads (fp8 [B,512]+[B,64],
           one CTA per B rows) vs the persistent int64 gather of offload_fill (B rows/iter, warps, CTAs/SM sweep),
           sequential rows (no randomness), sorted rows, 1 buffer vs all buffers (GPU TLB reach)
  serve    LayerOffload._sa_prefill under a drifting synthetic stream (N entries/call, working set U rows, NEW new rows
           per call) at the serving footprint: default vs ARC_OFFLOAD_HT_EPOCH vs + ARC_OFFLOAD_FILL_SPLIT (B/warps/CTAs),
           CUDA-graph replay, us per layer call + misses/spills per call
  thp      the gather sweep's best configs on transparent-hugepage buffers (anonymous MAP_PRIVATE and shared memfd,
           MADV_HUGEPAGE) of the same footprint -- tells whether 2 MiB host pages lift the GPU TLB limit
Output lines start with 'pcie '. Usage:
  python dots3/bench_pcie_fill.py [--bufs 12] [--rows 7895040] [--misses 8192,32768] [--parts info,dma,batch,gather,serve,thp]
"""
import argparse, ctypes, gc, mmap, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, "arcstats_pkg"), HERE):
    if os.path.isdir(os.path.join(p, "arcstats")):
        sys.path.insert(0, p)
os.environ.setdefault("ARC_OFFLOAD_RESOLVE", "1")
os.environ.setdefault("ARC_OFFLOAD_WAYS", "8")
import torch
import triton
import triton.language as tl
from arcstats import sparse_offload as so, offload_fill as of

D = 576


def out(*a):
    print("pcie", *a, flush=True)


def rd(p):
    try:
        return open(p).read().strip()
    except OSError:
        return "?"


@triton.jit
def _legacy_gather(rows_ptr, n, host_ptr, out_ptr, B: tl.constexpr):
    """the _sa_fill_kernel load pattern: one CTA per B rows, fp8 [B,512] + [B,64]"""
    i = tl.program_id(0) * B + tl.arange(0, B)
    m = i < n
    h = tl.load(rows_ptr + i, mask=m, other=0).to(tl.int64)
    dn = tl.arange(0, 512)
    dr = 512 + tl.arange(0, 64)
    o = i.to(tl.int64)
    tl.store(out_ptr + o[:, None] * 576 + dn[None, :], tl.load(host_ptr + h[:, None] * 576 + dn[None, :], mask=m[:, None]),
             mask=m[:, None])
    tl.store(out_ptr + o[:, None] * 576 + dr[None, :], tl.load(host_ptr + h[:, None] * 576 + dr[None, :], mask=m[:, None]),
             mask=m[:, None])


class Buf:
    """registered host buffer: kind 'shm' (4 KiB pages, /dev/shm file like serving), 'anon_thp', 'memfd_thp'"""

    def __init__(self, nbytes, kind):
        self.kind, self.n, self.path = kind, nbytes, None
        A = 2 << 20
        if kind == "shm":
            self.path = f"/dev/shm/pciebench_{os.getpid()}_{id(self)}"
            fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
            os.ftruncate(fd, nbytes)
            self.mm = mmap.mmap(fd, nbytes, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
            os.close(fd)
            off = 0
        elif kind == "anon_thp":
            self.mm = mmap.mmap(-1, nbytes + A, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
            self.mm.madvise(mmap.MADV_HUGEPAGE)
            off = None
        elif kind == "memfd_thp":
            fd = os.memfd_create("pciebench")
            os.ftruncate(fd, nbytes + A)
            self.mm = mmap.mmap(fd, nbytes + A, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
            os.close(fd)
            self.mm.madvise(mmap.MADV_HUGEPAGE)
            off = None
        base = ctypes.addressof(ctypes.c_char.from_buffer(self.mm))
        if off is None:
            off = (-base) % A
        self.addr = base + off
        self.t = torch.frombuffer(self.mm, dtype=torch.uint8)[off:off + nbytes]
        self.t[::4096] = 1                     # fault every page in (THP: 2 MiB pages if the kernel grants them)
        err = torch.cuda.cudart().cudaHostRegister(self.addr, nbytes, 0)
        if int(err) != 0:
            raise RuntimeError(f"cudaHostRegister failed {err}")
        self.rows = self.t.view(-1, D)

    def huge_kib(self):
        """kB of this mapping backed by huge pages (smaps AnonHugePages / ShmemPmdMapped / FilePmdMapped)"""
        lo, hi = self.addr, self.addr + self.n
        tot, cur = 0, False
        for l in open("/proc/self/smaps"):
            if "-" in l.split(" ")[0] and len(l.split(" ")[0].split("-")) == 2 and ":" not in l.split(" ")[0]:
                a, b = (int(x, 16) for x in l.split(" ")[0].split("-"))
                cur = a < hi and b > lo
            elif cur and (l.startswith("AnonHugePages:") or l.startswith("ShmemPmdMapped:") or l.startswith("FilePmdMapped:")):
                tot += int(l.split()[1])
        return tot

    def free(self):
        torch.cuda.cudart().cudaHostUnregister(self.addr)
        self.rows = self.t = None
        gc.collect()
        try:
            self.mm.close()
        except BufferError:
            pass
        if self.path:
            os.unlink(self.path)


def huge_kib_total(bufs):
    """kB of huge-page-backed memory in VMAs overlapping any of the buffers (each VMA counted once)"""
    rng = [(b.addr, b.addr + b.n) for b in bufs]
    tot, cur = 0, False
    for l in open("/proc/self/smaps"):
        f = l.split(" ")[0]
        if "-" in f and ":" not in f:
            a, b = (int(x, 16) for x in f.split("-"))
            cur = any(a < hi and b > lo for lo, hi in rng)
        elif cur and l.split(":")[0] in ("AnonHugePages", "ShmemPmdMapped", "FilePmdMapped"):
            tot += int(l.split()[1])
    return tot


def timeit(fn, it, sync_each=False):
    fn(0)
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for k in range(it):
        fn(k + 1)
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1e3 / it


def rand_rows(R, M, it, dev, g, sort=False, seq=False):
    lst = []
    for _ in range(it + 1):
        if seq:
            s = int(torch.randint(0, R - M, (1,), generator=g))
            r = torch.arange(s, s + M, device=dev)
        else:
            r = torch.randint(0, R, (M,), generator=g, device="cpu").to(dev)
            if sort:
                r = r.sort().values
        lst.append(r.to(torch.int64))
    return lst


def gather_sweep(bufs, Ms, dev, label, configs, extra=True, it=20):
    g = torch.Generator().manual_seed(0)
    R = bufs[0].rows.shape[0]
    nb = len(bufs)
    res = {}
    for M in Ms:
        outb = torch.empty((M, D), dtype=torch.uint8, device=dev)
        nt = torch.full((1,), M, dtype=torch.int32, device=dev)
        rows = rand_rows(R, M, it, dev, g)
        h64 = [b.rows.view(torch.int64) for b in bufs]
        h8 = [b.rows for b in bufs]
        o64 = outb.view(torch.int64)
        # correctness of the int64 gather on one call
        of._gather_kernel[(64,)](rows[0], nt, h64[0], o64, B=32, num_warps=4)
        torch.cuda.synchronize()
        assert torch.equal(outb[:64].cpu(), h8[0][rows[0][:64].cpu()]), "gather mismatch"

        def rep(name, us):
            gbs = M * D / us / 1e3
            res[(name, M)] = us
            out(label, name, "M", M, f"{us:.1f} us", f"{gbs:.2f} GB/s", f"{us * 1e3 / M:.1f} ns/row")
        if extra:
            for B in (32, 64):
                rep(f"legacy_B{B}w4", timeit(lambda k: _legacy_gather[(triton.cdiv(M, B),)](
                    rows[k], M, h8[k % nb], outb, B=B, num_warps=4), it))
        for (B, w, c) in configs:
            grid = max(1, min(triton.cdiv(M, B), of._nsm(dev) * c))
            rep(f"pers_B{B}w{w}c{c}", timeit(lambda k: of._gather_kernel[(grid,)](rows[k], nt, h64[k % nb], o64, B=B,
                                                                                 num_warps=w), it))
        if extra:
            B, w, c = configs[0]
            grid = max(1, min(triton.cdiv(M, B), of._nsm(dev) * c))
            for nm, rl in (("sorted", rand_rows(R, M, it, dev, g, sort=True)), ("seq", rand_rows(R, M, it, dev, g, seq=True))):
                rep(f"pers_B{B}w{w}c{c}_{nm}", timeit(lambda k: of._gather_kernel[(grid,)](rl[k], nt, h64[k % nb], o64,
                                                                                         B=B, num_warps=w), it))
            rep(f"pers_B{B}w{w}c{c}_1buf", timeit(lambda k: of._gather_kernel[(grid,)](rows[k], nt, h64[0], o64, B=B,
                                                                                    num_warps=w), it))
        del outb
    return res


def part_dma(buf, dev):
    torch.cuda.init()
    try:
        rt = ctypes.CDLL("libcudart.so.13")
    except OSError:
        rt = ctypes.CDLL("libcudart.so.12")
    st = torch.cuda.current_stream().cuda_stream
    for M in (8192, 32768, 131072):
        n = M * D
        d = torch.empty(n, dtype=torch.uint8, device=dev)
        for kind, nm, (dst, src) in ((1, "H2D", (d.data_ptr(), buf.addr)), (2, "D2H", (buf.addr, d.data_ptr()))):
            us = timeit(lambda k: rt.cudaMemcpyAsync(ctypes.c_void_p(dst), ctypes.c_void_p(src), ctypes.c_size_t(n),
                                                     ctypes.c_int(kind), ctypes.c_void_p(st)), 20)
            out("dma", nm, "bytes", n, f"{us:.1f} us", f"{n / us / 1e3:.2f} GB/s")
    return rt


class _Loc(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _Attr(ctypes.Structure):
    _fields_ = [("srcAccessOrder", ctypes.c_int), ("srcLocHint", _Loc), ("dstLocHint", _Loc), ("flags", ctypes.c_uint)]


def part_batch(rt, buf, dev):
    import numpy as np
    if not hasattr(rt, "cudaMemcpyBatchAsync"):
        out("batch unavailable")
        return
    sobj = torch.cuda.Stream()                       # the batch API rejects the legacy default stream
    st = sobj.cuda_stream
    R = buf.rows.shape[0]
    torch.cuda.set_stream(sobj)
    for M in (8192, 32768):
        d = torch.empty((M, D), dtype=torch.uint8, device=dev)
        rows = np.random.default_rng(0).integers(0, R, M)
        dsts = (ctypes.c_void_p * M)(*[d.data_ptr() + i * D for i in range(M)])
        srcs = (ctypes.c_void_p * M)(*[buf.addr + int(r) * D for r in rows])
        sizes = (ctypes.c_size_t * M)(*([D] * M))
        attr = _Attr(1, _Loc(0, 0), _Loc(0, 0), 0)
        idx = (ctypes.c_size_t * 1)(0)
        rt.cudaMemcpyBatchAsync.restype = ctypes.c_int
        errs = []

        def f(k):
            errs.append(rt.cudaMemcpyBatchAsync(dsts, srcs, sizes, ctypes.c_size_t(M), ctypes.byref(attr), idx,
                                                ctypes.c_size_t(1), ctypes.c_void_p(st)))
        t0 = time.time()
        us = timeit(f, 5)
        cpu = (time.time() - t0) / 6 * 1e6
        if any(errs):
            rt.cudaGetLastError()
            out("batch error", errs[0])
            break
        ok = torch.equal(d[:16].cpu(), buf.rows[torch.as_tensor(rows[:16])])
        out("batch M", M, f"{us:.1f} us", f"{M * D / us / 1e3:.2f} GB/s", f"host call+sync {cpu:.0f} us", "ok" if ok else "MISMATCH")
    torch.cuda.synchronize()
    torch.cuda.set_stream(torch.cuda.default_stream())


def part_serve(los, dev, N, U, NEWs, configs, calls=48):
    for NEW in NEWs:
        g = torch.Generator(device=dev).manual_seed(NEW)
        R = los[0].R
        ws = torch.randperm(R, device=dev, generator=g)[:U]
        stream = []
        for t in range(calls):
            k = torch.randperm(U, device=dev, generator=g)[:NEW]
            ws = ws.clone()
            ws[k] = torch.randint(0, R, (NEW,), device=dev, generator=g)
            stream.append(ws[torch.randint(0, U, (N,), device=dev, generator=g)].int().contiguous())
        for name, ep, split, B, w, c, two, *srt in configs:
            of.EPOCH, of.SPLIT, of.FILL_B, of.FILL_WARPS, of.FILL_CTAS, of.TWO = ep, split, B, w, c, two
            of.SORT = srt[0] if srt else 0
            so._FILLX = ep or split or two
            for x in stream[:24]:          # steady state
                for lo in los:
                    lo._sa_prefill(x)
            torch.cuda.synchronize()
            mc = []
            for x in stream[24:28]:
                rb = los[0]._sa_prefill(x)
                mc.append((int(los[0].mcnt), int(rb["scnt"])))
                for lo in los[1:]:
                    lo._sa_prefill(x)
            static = torch.empty_like(stream[0])
            static.copy_(stream[27])
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for lo in los:
                    lo._sa_prefill(static)
            torch.cuda.current_stream().wait_stream(s)
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                for lo in los:
                    lo._sa_prefill(static)
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            tot = 0.0
            for x in stream[28:]:
                static.copy_(x)
                e0.record(); gr.replay(); e1.record()
                torch.cuda.synchronize()
                tot += e0.elapsed_time(e1)
            us = tot * 1e3 / ((calls - 28) * len(los))
            m = sum(a for a, _ in mc) / len(mc)
            sp = sum(b for _, b in mc) / len(mc)
            out("serve", name, "N", N, "U", U, "NEW", NEW, f"{us:.1f} us/layer-call", f"misses {m:.0f} spills {sp:.0f}",
                f"{us * 1e3 / max(m, 1):.1f} ns/miss")
            del gr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bufs", type=int, default=12)
    ap.add_argument("--rows", type=int, default=7895040)
    ap.add_argument("--misses", default="8192,32768")
    ap.add_argument("--parts", default="info,dma,batch,gather,serve,thp")
    ap.add_argument("--serve-n", type=int, default=393216)
    ap.add_argument("--serve-u", type=int, default=116000)
    ap.add_argument("--serve-new", default="8000,24000")
    a = ap.parse_args()
    parts = a.parts.split(",")
    Ms = [int(x) for x in a.misses.split(",")]
    dev = torch.device("cuda")
    torch.cuda.init()
    if "info" in parts:
        p = torch.cuda.get_device_properties(dev)
        q = subprocess.run(["nvidia-smi", "-i", os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0],
                            "--query-gpu=pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current",
                            "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
        out("info gpu", p.name, "sms", p.multi_processor_count, "pcie gen/max/width", q)
        out("info thp", rd("/sys/kernel/mm/transparent_hugepage/enabled"), "| shmem",
            rd("/sys/kernel/mm/transparent_hugepage/shmem_enabled"), "| defrag", rd("/sys/kernel/mm/transparent_hugepage/defrag"))
        mi = {l.split(":")[0]: l.split(":")[1].strip() for l in open("/proc/meminfo")}
        out("info mem", "MemTotal", mi.get("MemTotal"), "MemAvailable", mi.get("MemAvailable"), "Hugepagesize",
            mi.get("Hugepagesize"), "HugePages_Total", mi.get("HugePages_Total"))
        out("info shm", subprocess.run(["df", "-h", "/dev/shm"], capture_output=True, text=True).stdout.split("\n")[1])
    nbytes = a.rows * D
    cfg_sweep = [(B, w, c) for B in (16, 32, 64) for w in (2, 4, 8) for c in (4, 8, 16)
                 if not (B == 64 and w == 2)]
    best = [(32, 4, 8)]
    if any(x in parts for x in ("dma", "batch", "gather", "serve")):
        so.CACHE_ROWS = 131072
        t0 = time.time()
        los = [so.LayerOffload(a.rows, dev, key=f"pciebench{i}") for i in range(a.bufs)]   # serving layout (shared shm)
        for lo in los:
            lo.host.view(torch.uint8).view(-1)[::4096] = 1
        out("setup", a.bufs, "x", f"{nbytes / 2**30:.2f} GiB shm buffers", f"{time.time() - t0:.0f} s")

        class _B:
            pass
        bufs = []
        for lo in los:
            b = _B(); b.rows = lo.host.view(torch.uint8); b.addr = lo.host.data_ptr(); bufs.append(b)
        rt = None
        if "dma" in parts:
            rt = part_dma(bufs[0], dev)
        if "batch" in parts:
            try:
                part_batch(rt or ctypes.CDLL("libcudart.so.13"), bufs[0], dev)
            except Exception as ex:
                out("batch failed", repr(ex)[:200])
        if "gather" in parts:
            res = gather_sweep(bufs, Ms, dev, "gather4k", cfg_sweep)
            Mx = max(Ms)
            ranked = sorted((us, nm) for (nm, M), us in res.items() if M == Mx and nm.startswith("pers_B") and nm.count("_") == 1)
            best = [tuple(int(x) for x in nm[6:].replace("w", " ").replace("c", " ").split()) for _, nm in ranked[:3]]
            out("gather4k best", ranked[:3])
        if "serve" in parts:
            b0 = best[0]
            cfgs = [("default", False, False, 32, 4, 8, False), ("epoch", True, False, 32, 4, 8, False)]
            cfgs += [(f"epoch+split_B{b}w{w}c{c}", True, True, b, w, c, False) for (b, w, c) in dict.fromkeys(best + [(32, 4, 8)])]
            cfgs += [(f"epoch+split+sort64k_B{b0[0]}w{b0[1]}c{b0[2]}", True, True, *b0, False, 65536)]
            cfgs += [(f"epoch+two_B{b0[0]}w{b0[1]}c{b0[2]}", True, True, *b0, True)]   # policy change: fewer misses
            cfgs += [(f"epoch+two+sort64k_B{b0[0]}w{b0[1]}c{b0[2]}", True, True, *b0, True, 65536)]
            part_serve(los, dev, a.serve_n, a.serve_u, [int(x) for x in a.serve_new.split(",")], cfgs)
        del bufs
        for lo in los:
            lo.host = None
        los = None
        gc.collect()
        so._unlink_created()
        so._HostReg._created.clear()
    if "thp" in parts:
        for kind in ("anon_thp", "memfd_thp"):
            try:
                t0 = time.time()
                bufs = [Buf(nbytes, kind) for _ in range(a.bufs)]
                hk = huge_kib_total(bufs)
                out("thp", kind, a.bufs, "buffers", f"huge-page backed {hk / 2**20:.1f} of {a.bufs * nbytes / 2**30:.1f} GiB",
                    f"setup {time.time() - t0:.0f} s")
                gather_sweep(bufs, Ms, dev, kind, best + [(32, 4, 8)], extra=True)
            except Exception as ex:
                out("thp", kind, "failed", repr(ex)[:300])
            finally:
                for b in locals().get("bufs", []) or []:
                    try:
                        b.free()
                    except Exception:
                        pass
                bufs = []
                gc.collect()


if __name__ == "__main__":
    main()
