"""Random host-row gather into the GPU from a cudaHostRegister'ed buffer: 4 KB pages vs 2 MB pages
. The row-cache fill reads ~800k random 464 B rows per GPU per decode step from ~6 GB
host buffers per layer; if the GPU's translation of sysmem addresses (4 KB PTEs) limits the request rate, 2 MB pages
would raise it.

  python dots3/bench_hugepage_fill.py [GB=8] [M=65536,262144] [modes=shm,anon,thp,hugetlb]
modes: shm     MAP_SHARED file in /dev/shm (production layout: one buffer shared by the TP ranks), 4 KB pages
       memfd   memfd MAP_SHARED, madvise(MADV_HUGEPAGE) (shmem THP: only if /sys/kernel/mm/transparent_hugepage/
               shmem_enabled is 'advise' / 'always' / 'within_size')
       anon    MAP_PRIVATE|MAP_ANONYMOUS, 4 KB (MADV_NOHUGEPAGE)
       thp     MAP_PRIVATE|MAP_ANONYMOUS + MADV_HUGEPAGE (needs THP enabled 'madvise' / 'always')
       hugetlb MAP_HUGETLB (needs reserved hugetlbfs pages)
Prints the page backing (AnonHugePages / ShmemHugePages of the mapping from /proc/self/smaps) and
'hp <mode> M <m> <us> us <rows/us> <GB/s>' (median of 20 CUDA-event timed launches, 8 random row sets).
"""
import ctypes, mmap, os, sys, time

import torch
import triton
import triton.language as tl

GB = float(sys.argv[1]) if len(sys.argv) > 1 else 8
Ms = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "65536,262144").split(",")]
MODES = (sys.argv[3] if len(sys.argv) > 3 else "shm,memfd,anon,thp,hugetlb").split(",")
RB = 464                       # e2m3 host row bytes
W = RB // 4                    # int32 words per row
libc = ctypes.CDLL("libc.so.6", use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
PROT = 3; MAP_SHARED, MAP_PRIVATE, MAP_ANON, MAP_HUGETLB, MAP_POPULATE = 1, 2, 0x20, 0x40000, 0x8000
MADV_HUGEPAGE, MADV_NOHUGEPAGE = 14, 15
dev = torch.device("cuda", 0)
for f in ("enabled", "shmem_enabled", "defrag"):
    try:
        print(f"hp sysfs {f}: {open('/sys/kernel/mm/transparent_hugepage/' + f).read().strip()}", flush=True)
    except Exception as e:
        print(f"hp sysfs {f}: {e!r}", flush=True)
print("hp meminfo", " ".join(l.split(":")[0] + "=" + l.split()[1] for l in open("/proc/meminfo")
                             if "Huge" in l), flush=True)


@triton.jit
def _gather(host, rows, out, n, W: tl.constexpr, BW: tl.constexpr, RPP: tl.constexpr):
    pid = tl.program_id(0)
    w = tl.arange(0, BW)
    for i in tl.static_range(RPP):
        r = pid * RPP + i
        if r < n:
            row = tl.load(rows + r).to(tl.int64)
            v = tl.load(host + row * W + w, mask=w < W)
            tl.store(out + r.to(tl.int64) * W + w, v, mask=w < W)


def backing(addr, nbytes):
    """kB of the mapping backed by huge pages (from /proc/self/smaps)"""
    lines = open("/proc/self/smaps").read().splitlines()
    out, cur = {}, False
    for l in lines:
        p = l.split()
        if "-" in p[0] and len(p) >= 5 and all(c in "0123456789abcdef-" for c in p[0]):
            s, e = (int(x, 16) for x in p[0].split("-"))
            cur = s <= addr < e
        elif cur and p[0] in ("AnonHugePages:", "ShmemPmdMapped:", "Private_Hugetlb:", "Shared_Hugetlb:", "Rss:"):
            out[p[0][:-1]] = out.get(p[0][:-1], 0) + int(p[1])
    return out


def alloc(mode, nbytes):
    nbytes = (nbytes + (2 << 20) - 1) // (2 << 20) * (2 << 20)
    fd, path = -1, None
    if mode == "shm":
        path = f"/dev/shm/hpbench_{os.getpid()}"
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600); os.ftruncate(fd, nbytes); flags = MAP_SHARED
    elif mode == "memfd":
        fd = os.memfd_create("hpbench"); os.ftruncate(fd, nbytes); flags = MAP_SHARED
    elif mode == "hugetlb":
        flags = MAP_PRIVATE | MAP_ANON | MAP_HUGETLB
    else:
        flags = MAP_PRIVATE | MAP_ANON
    a = libc.mmap(None, nbytes, PROT, flags, fd, 0)
    if a in (None, ctypes.c_void_p(-1).value):
        raise OSError(ctypes.get_errno(), f"mmap {mode}")
    if mode in ("thp", "memfd"):
        print(f"hp {mode} madvise(HUGEPAGE) rc {libc.madvise(a, nbytes, MADV_HUGEPAGE)}", flush=True)
    if mode == "anon":
        libc.madvise(a, nbytes, MADV_NOHUGEPAGE)
    if fd >= 0:
        os.close(fd)
    if path:
        os.unlink(path)
    buf = (ctypes.c_uint8 * nbytes).from_address(a)
    t = torch.frombuffer(buf, dtype=torch.int32)
    t0 = time.time()
    for c in range(0, t.numel(), 1 << 26):   # touch (fault in) every page
        t[c:c + (1 << 26)].fill_(c >> 26)
    rc = torch.cuda.cudart().cudaHostRegister(a, nbytes, 0)
    print(f"hp {mode} {nbytes / 2**30:.1f} GiB touched in {time.time() - t0:.1f} s, register rc {rc}, "
          f"backing {backing(a, nbytes)}", flush=True)
    return a, nbytes, t


for mode in MODES:
    try:
        a, nb, t = alloc(mode, int(GB * 2**30))
    except Exception as e:
        print(f"hp {mode} unavailable: {e!r}", flush=True)
        continue
    R = nb // RB
    host = t   # CPU tensor; Triton uses its (UVA) pointer
    for M in Ms:
        g = torch.Generator(device=dev).manual_seed(M)
        sets = [torch.randint(0, R, (M,), device=dev, generator=g, dtype=torch.int64) for _ in range(8)]
        out = torch.empty(M, W, dtype=torch.int32, device=dev)
        RPP = 2
        f = lambda r: _gather[(triton.cdiv(M, RPP),)](host, r, out, M, W=W, BW=128, RPP=RPP, num_warps=1)
        f(sets[0]); torch.cuda.synchronize()
        ts = []
        for i in range(20):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record(); f(sets[i % 8]); e1.record(); torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1) * 1e3)
        ok = bool((out[:, 1] == ((sets[19 % 8] * W + 1) >> 26).int()).all())
        us = sorted(ts)[len(ts) // 2]
        print(f"hp {mode} M {M} {us:.0f} us {M / us:.1f} rows/us {M * RB / us / 1e3:.2f} GB/s check {ok}", flush=True)
    torch.cuda.cudart().cudaHostUnregister(a)
    del t, host
    libc.munmap = libc.munmap; libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]; libc.munmap(a, nb)
