#!/usr/bin/env python3
import argparse, base64, hashlib, json, os, re, shutil, subprocess, sys, tarfile, tempfile
from pathlib import Path
NEW_PREFIX = '/opt/python/cpython-3.12.15-linux-x86_64-gnu'
LIB = 'arc_py312/lib/python3.12'

def sh(*a, **k):
    return subprocess.run(a, check=True, **k)

def extract(tar, d):
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    sh('tar', '--numeric-owner', '-C', str(d), '-xf', str(tar))

def probe(env_root, home):
    code = '\nimport importlib.metadata as md, json, sys, sysconfig, os\nroot = os.path.realpath(sys.prefix)\nd = sorted((x.metadata["Name"], x.version) for x in md.distributions())\ncv = {k: str(v).replace(root, "<ROOT>") for k, v in sysconfig.get_config_vars().items()}\npaths = {k: v.replace(root, "<ROOT>") for k, v in sysconfig.get_paths().items()}\nimport importlib.util\nmods = {}\nfor m in ("vllm", "torch", "arcstats", "transformers", "triton", "flashinfer", "httpx"):\n    s = importlib.util.find_spec(m)\n    mods[m] = None if s is None else os.path.relpath(s.origin, root)\nprint(json.dumps({"dists": d, "config_vars": cv, "paths": paths, "mods": mods, "version": sys.version}))\n'
    r = subprocess.run([str(env_root / 'arc_py312/bin/python3.12'), '-I', '-B', '-c', code], capture_output=True, text=True, env={'PATH': '/usr/bin:/bin', 'HOME': str(home)})
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout)

def main():
    ap = argparse.ArgumentParser(description="Re-pack arc_py312.tar(bin) for the runtime dataset: tar owner 0:0 with numeric ids; the interpreter's build prefix in _sysconfigdata and the .pyc files that recorded it, and the plugin's direct_url.json, made neutral. Everything else keeps its bytes, member order, modes and mtimes. Checks that the environment reports the same distributions, imports and sysconfig as before.")
    ap.add_argument('src', help='the environment archive as built (build_env.sh): a tar of arc_py312/')
    ap.add_argument('out', help='the re-packed archive, e.g. <dataset folder>/arc_py312.tarbin')
    ap.add_argument('--work', help='scratch folder for the two extracted trees (about 2 x 8 GB; default: next to OUT)')
    ap.add_argument('--forbid', action='append', default=[], help='regex that must not remain in the archive (repeatable; default: your home directory)')
    a = ap.parse_args()
    src, out = (Path(a.src).resolve(), Path(a.out).resolve())
    out.parent.mkdir(parents=True, exist_ok=True)
    work = Path(a.work or tempfile.mkdtemp(prefix='.envbuild_', dir=str(out.parent)))
    forbid = a.forbid or [re.escape(os.path.expanduser('~'))]
    bad = re.compile('|'.join(forbid).encode(), re.I)
    old, new = (work / 'old', work / 'new')
    extract(src, old)
    extract(src, new)
    rep = {'changed': []}
    sc = new / LIB / '_sysconfigdata__linux_x86_64-linux-gnu.py'
    txt = sc.read_text()
    old_prefix = re.search('(/\\S*?/cpython-3\\.12\\.15-linux-x86_64-gnu)', txt).group(1)
    assert old_prefix != NEW_PREFIX
    st = sc.stat()
    sc.write_text(txt.replace(old_prefix, NEW_PREFIX))
    os.utime(sc, (st.st_atime, st.st_mtime))
    rep['changed'].append(str(sc.relative_to(new)))
    py = new / 'arc_py312/bin/python3.12'
    pycs = [p for p in (new / LIB).rglob('*.pyc') if old_prefix.encode() in p.read_bytes()]
    for p in pycs:
        s = p.parent.parent / (p.name.split('.')[0] + '.py')
        flags = int.from_bytes(p.read_bytes()[4:8], 'little')
        assert flags == 0 and s.exists(), (p, flags)
        dfile = '/tmp/' + str(s.relative_to(new))
        sh(str(py), '-I', '-B', '-c', 'import py_compile,sys; py_compile.compile(sys.argv[1], cfile=sys.argv[2], dfile=sys.argv[3], doraise=True, invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)', str(s), str(p), dfile, env={'PATH': '/usr/bin:/bin'})
        assert old_prefix.encode() not in p.read_bytes(), p
        rep['changed'].append(str(p.relative_to(new)))
    di = new / LIB / 'site-packages/arcstats-0.1.dist-info'
    du = di / 'direct_url.json'
    b = json.dumps({'url': 'file:///opt/build/arcstats_pkg', 'dir_info': {}}, separators=(',', ':')).encode()
    st = du.stat()
    du.write_bytes(b)
    os.utime(du, (st.st_atime, st.st_mtime))
    h = base64.urlsafe_b64encode(hashlib.sha256(b).digest()).rstrip(b'=').decode()
    rec = di / 'RECORD'
    st = rec.stat()
    lines = [f'arcstats-0.1.dist-info/direct_url.json,sha256={h},{len(b)}' if l.startswith('arcstats-0.1.dist-info/direct_url.json,') else l for l in rec.read_text().splitlines()]
    rec.write_text('\n'.join(lines) + '\n')
    os.utime(rec, (st.st_atime, st.st_mtime))
    rep['changed'] += [str(du.relative_to(new)), str(rec.relative_to(new))]
    left = [str(p.relative_to(new)) for p in new.rglob('*') if p.is_file() and (not p.is_symlink()) and bad.search(p.read_bytes())]
    rep['leftover_files'] = left
    r = subprocess.run(['diff', '-rq', '--no-dereference', str(old), str(new)], capture_output=True, text=True)
    difs = sorted(re.findall('Files \\S+?/old/(\\S+) and', r.stdout))
    rep['diff_only_listed'] = sorted(difs) == sorted(rep['changed']) and 'Only in' not in r.stdout
    members = [m.name for m in tarfile.open(src)]
    if out.exists():
        out.unlink()
    sh('tar', '--numeric-owner', '--owner=0', '--group=0', '--no-recursion', '-C', str(new), '-cf', str(out), '--files-from=-', input='\n'.join(members).encode())
    t_old = [(m.name, m.size, m.mode, m.type, m.linkname, m.mtime) for m in tarfile.open(src)]
    t_new = [(m.name, m.size, m.mode, m.type, m.linkname, m.mtime) for m in tarfile.open(out)]
    changed = set(rep['changed'])
    rep['members_identical_except_changed_sizes'] = len(t_old) == len(t_new) and all((x[0] == y[0] and x[2:5] == y[2:5] and (x[1] == y[1] or x[0] in changed) for x, y in zip(t_old, t_new)))
    rep['owners'] = sorted({(m.uid, m.gid, m.uname, m.gname) for m in tarfile.open(out)})
    with open(out, 'rb') as f:
        hits = set()
        for blk in iter(lambda: f.read(64 << 20), b''):
            hits |= {m.group(0).decode(errors='replace').lower() for m in bad.finditer(blk)}
    rep['tar_scan'] = sorted(hits)
    x, y = (probe(old, work), probe(new, work))
    rep['dists_identical'] = x['dists'] == y['dists']
    rep['n_dists'] = len(x['dists'])
    rep['imports_identical'] = x['mods'] == y['mods']
    rep['mods'] = y['mods']
    cv_d = {k: (x['config_vars'].get(k), y['config_vars'].get(k)) for k in set(x['config_vars']) | set(y['config_vars']) if x['config_vars'].get(k) != y['config_vars'].get(k)}
    rep['config_vars_differing'] = sorted(cv_d)
    rep['config_vars_only_prefix'] = all((str(u).replace(old_prefix, NEW_PREFIX) == str(v) for u, v in cv_d.values()))
    rep['paths_identical'] = x['paths'] == y['paths']
    ok = rep['diff_only_listed'] and rep['members_identical_except_changed_sizes'] and (not rep['tar_scan']) and (not left) and rep['dists_identical'] and rep['imports_identical'] and rep['config_vars_only_prefix'] and rep['paths_identical'] and (rep['owners'] == [(0, 0, '', '')])
    rep['ok'] = ok
    json.dump(rep, open(str(out) + '.report.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != 'config_vars_differing'}, indent=1)[:4000])
    print('config vars differing:', len(rep['config_vars_differing']))
    print('OK' if ok else 'FAIL')
    shutil.rmtree(work)
    sys.exit(0 if ok else 1)
if __name__ == '__main__':
    main()
