import argparse, glob, os, re, sys, threading, time

def layer_key(path, order):
    m = re.search('L(\\d+)\\.(pt|safetensors)$', os.path.basename(path))
    return (int(m.group(1)) if m else -1, order, path)

def expert_files(dirs):
    out = []
    for k, d in enumerate(dirs):
        d = os.path.realpath(d)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            p = os.path.realpath(os.path.join(d, f))
            if os.path.isfile(p):
                out.append(layer_key(p, k))
    out.sort()
    seen, files = (set(), [])
    for _, _, p in out:
        if p not in seen:
            seen.add(p)
            files.append(p)
    return files

def wait_signal(log, pattern, count, timeout):
    t0 = time.time()
    rx = re.compile(pattern)
    while time.time() - t0 < timeout:
        try:
            with open(log, errors='replace') as f:
                if sum((1 for line in f if rx.search(line))) >= count:
                    return time.time() - t0
        except OSError:
            pass
        time.sleep(0.5)
    return None

def drop(paths):
    n = b = 0
    for p in paths:
        try:
            fd = os.open(p, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                n += 1
                b += os.fstat(fd).st_size
            finally:
                os.close(fd)
        except OSError:
            pass
    return (n, b)

def prefetch(files, threads, block):
    jobs = [(p, off) for p in files for off in range(0, os.path.getsize(p), block)]
    stat = {'bytes': 0}
    lock = threading.Lock()

    def worker(k):
        for p, off in jobs[k::threads]:
            try:
                with open(p, 'rb', buffering=0) as f:
                    f.seek(off)
                    n = len(f.read(block))
                with lock:
                    stat['bytes'] += n
            except OSError:
                pass
    ts = [threading.Thread(target=worker, args=(k,), daemon=True) for k in range(threads)]
    t0 = time.time()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return (stat['bytes'], time.time() - t0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dirs', nargs='+')
    ap.add_argument('--log', default='')
    ap.add_argument('--pattern', default='\\[fnmoe\\] loader: dropped')
    ap.add_argument('--count', type=int, default=4)
    ap.add_argument('--timeout', type=float, default=1800)
    ap.add_argument('--drop-glob', default='')
    ap.add_argument('--threads', type=int, default=16)
    ap.add_argument('--block-mb', type=int, default=16)
    a = ap.parse_args()
    files = expert_files(a.dirs)
    tot = sum((os.path.getsize(p) for p in files))
    print(f'[late_pf] {len(files)} expert files {tot / 2 ** 30:.1f} GiB, waiting for the loader signal', flush=True)
    if a.log:
        w = wait_signal(a.log, a.pattern, a.count, a.timeout)
        if w is None:
            print('[late_pf] signal not seen: no prefetch', flush=True)
            return
        print(f'[late_pf] signal after {w:.0f} s', flush=True)
    if a.drop_glob:
        n, b = drop(sorted(glob.glob(a.drop_glob)))
        print(f'[late_pf] dropped {n} n-gram shards {b / 2 ** 30:.1f} GiB from the page cache', flush=True)
    nb, dt = prefetch(files, a.threads, a.block_mb << 20)
    print(f'[late_pf] read {nb / 2 ** 30:.1f} GiB in {dt:.0f} s ({nb / 1000000.0 / max(dt, 1e-09):.0f} MB/s)', flush=True)
if __name__ == '__main__':
    sys.exit(main())
