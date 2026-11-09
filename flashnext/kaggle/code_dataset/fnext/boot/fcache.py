import os, sys, time

def files(paths):
    for p in paths:
        p = os.path.realpath(p)
        if os.path.isdir(p):
            for r, _, fs in os.walk(p):
                for f in fs:
                    yield os.path.realpath(os.path.join(r, f))
        elif os.path.isfile(p):
            yield p

def drop(paths):
    n = b = 0
    for f in files(paths):
        try:
            fd = os.open(f, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                n += 1
                b += os.fstat(fd).st_size
            finally:
                os.close(fd)
        except OSError:
            pass
    return (n, b)

def cg(name):
    for base in ('/sys/fs/cgroup', '/sys/fs/cgroup/memory'):
        p = os.path.join(base, name)
        if os.path.exists(p):
            return open(p).read()
    return ''

def stat():
    ms = dict((l.split() for l in cg('memory.stat').splitlines() if len(l.split()) == 2))
    gib = lambda k: int(ms.get(k, 0)) / 2 ** 30
    lim = cg('memory.max').strip() or '?'
    rb = sum((int(kv.split('=')[1]) for l in cg('io.stat').splitlines() for kv in l.split() if kv.startswith('rbytes=')))
    if not rb:
        vm = dict((l.split() for l in open('/proc/vmstat')))
        rb = int(vm.get('pgpgin', 0)) * 1024
    lim_g = f'{int(lim) / 2 ** 30:.0f}' if lim.isdigit() else lim
    return f'anon {gib('anon'):.1f} file {gib('file'):.1f} shmem {gib('shmem'):.1f} GiB (limit {lim_g}); storage read {rb / 2 ** 30:.1f} GiB'

def fixgrp(root):
    import json
    n = 0
    for r, _, fs in os.walk(root):
        for f in fs:
            if f.startswith('__grp__') and f.endswith('.json'):
                p = os.path.join(r, f)
                g = json.load(open(p))
                g['child_paths'] = {c: os.path.join(os.path.abspath(r), os.path.basename(v)) for c, v in g.get('child_paths', {}).items()}
                with open(p + '.tmp', 'w') as w:
                    json.dump(g, w)
                os.replace(p + '.tmp', p)
                n += 1
    return n
if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'drop':
        t = time.time()
        n, b = drop(sys.argv[2:])
        print(f'[fcache] dropped {n} files {b / 2 ** 30:.1f} GiB from the page cache in {time.time() - t:.1f} s', flush=True)
    elif cmd == 'fixgrp':
        print(f'[fcache] {fixgrp(sys.argv[2])} Triton group files re-pointed to {sys.argv[2]}', flush=True)
    elif cmd == 'stat':
        print(f'[fcache] {' '.join(sys.argv[2:])} {stat()}', flush=True)
