#!/usr/bin/env python3
import argparse, json, os, re, shutil, subprocess, sys, tarfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
TARGET = '/tmp/fn_py312'
JIT = ['flashinfer-jit-cache==0.7.0.post1+cu130', 'flashinfer-jit-cache-sm89==0.7.0.post1+cu130']

def sh(*a, **k):
    print('$', ' '.join(map(str, a))[:300], flush=True)
    return subprocess.run([str(x) for x in a], check=True, **k)

def main():
    ap = argparse.ArgumentParser(description='build the offline runtime archive fn_py312.tar in WORK')
    ap.add_argument('work')
    ap.add_argument('--lock', default=str(HERE / 'requirements-fn032.lock.txt'))
    ap.add_argument('--python-dir', default=str(Path.home() / '.local/share/uv/python/cpython-3.12.14-linux-x86_64-gnu'), help='a python-build-standalone CPython 3.12.14 (e.g. from `uv python install 3.12.14`)')
    a = ap.parse_args()
    LOCK, PYSRC = (Path(a.lock).resolve(), Path(a.python_dir))
    work = Path(a.work).resolve()
    env = work / 'fn_py312'
    shutil.rmtree(env, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    shutil.copytree(PYSRC, env, symlinks=True)
    py = env / 'bin/python3.12'
    for m in env.glob('lib/python3.12/EXTERNALLY-MANAGED'):
        m.unlink()
    uv = shutil.which('uv') or str(Path.home() / '.local/bin/uv')
    sh(uv, 'pip', 'install', '--python', py, '--link-mode=copy', '-r', LOCK, '--index-url', 'https://pypi.org/simple', '--extra-index-url', 'https://download.pytorch.org/whl/cu130', '--index-strategy', 'unsafe-best-match')
    sh(uv, 'pip', 'install', '--python', py, '--link-mode=copy', '--no-deps', '--index-url', 'https://flashinfer.ai/whl/cu130', *JIT)
    olds = sorted({str(PYSRC), str(PYSRC.resolve()), str(env), str(PYSRC.parent / 'cpython-3.12-linux-x86_64-gnu')}, key=len, reverse=True)
    n_fixed = 0
    for p in env.rglob('*'):
        if p.is_symlink() or not p.is_file() or p.suffix in ('.so', '.pyc', '.whl', '.a') or (p.stat().st_size > 8 << 20):
            continue
        b = p.read_bytes()
        if b'\x00' in b[:4096]:
            continue
        nb = b
        for o in olds:
            nb = nb.replace(o.encode(), TARGET.encode())
        if nb != b:
            p.write_bytes(nb)
            n_fixed += 1
    for p in env.rglob('*'):
        if p.is_symlink() and os.readlink(p).startswith(tuple(olds)):
            t = os.readlink(p)
            for o in olds:
                t = t.replace(o, TARGET)
            p.unlink()
            os.symlink(t, p)
    print(f'rewrote local paths in {n_fixed} text files', flush=True)
    rep = {}
    r = subprocess.run([str(py), '-c', 'import vllm, torch, flashinfer, triton, sys; print(sys.version.split()[0], vllm.__version__, torch.__version__, torch.version.cuda, flashinfer.__version__, triton.__version__)'], capture_output=True, text=True)
    rep['imports'] = (r.stdout or r.stderr).strip().splitlines()[-1]
    for c in env.rglob('__pycache__'):
        shutil.rmtree(c)
    subprocess.run([str(py), '-m', 'compileall', '-q', '-f', '-j', '16', '-s', str(env), '-p', TARGET, str(env / 'lib/python3.12')])
    local = sorted({*olds, str(work), str(Path.home())} - {'', '/'}, key=len, reverse=True)
    BAD = re.compile('|'.join((re.escape(x) for x in local)).encode())
    hits = []
    for p in env.rglob('*'):
        if p.is_file() and (not p.is_symlink()) and (p.stat().st_size < 64 << 20) and BAD.search(p.read_bytes()):
            hits.append(str(p.relative_to(env)))
    rep['local_path_hits'] = hits[:50]
    rep['n_local_path_hits'] = len(hits)
    mx = 'GLIBC_2.0'
    for so in env.rglob('*.so*'):
        if so.is_file() and (not so.is_symlink()):
            out = subprocess.run(['objdump', '-T', str(so)], capture_output=True, text=True).stdout
            for v in re.findall('GLIBC_2\\.(\\d+)', out):
                if int(v) > int(mx.split('.')[1]):
                    mx = f'GLIBC_2.{v}'
    rep['max_glibc'] = mx
    print(json.dumps(rep, indent=1), flush=True)
    out = work / 'fn_py312.tar'

    def norm(ti):
        ti.uid = ti.gid = 0
        ti.uname = ti.gname = ''
        return ti
    with tarfile.open(out, 'w', format=tarfile.GNU_FORMAT) as t:
        t.add(env, arcname='fn_py312', filter=norm)
    rep['tar_bytes'] = out.stat().st_size
    json.dump(rep, open(work / 'fn_py312_report.json', 'w'), indent=1)
    print(f"wrote {out} ({rep['tar_bytes'] / 2 ** 30:.2f} GiB)")
    if hits or int(mx.split('.')[1]) > 35 or '0.32.0' not in rep['imports']:
        sys.exit('CHECK FAILED: see report')
if __name__ == '__main__':
    main()
