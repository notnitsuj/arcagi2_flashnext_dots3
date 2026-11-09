#!/usr/bin/env python3
import argparse, hashlib, json, os, shutil, sys, tarfile, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
SKIP_TOP = {'MANIFEST.md5', 'README.md', 'fnext', 'rc9'}
OVERLAYS = {'rc8': None, 'rc9': 'rc9'}
TAR_MTIMES = {'rc8': dict.fromkeys(['fnext/base', 'fnext/base/gdn_ba_overlap.py', 'fnext/base/host_dp_fused.py', 'fnext/base/load_ec.py', 'fnext/base/plugin.py', 'fnext/base/q8_dispatch.py', 'fnext/base/q8_rs.py', 'fnext/base/qsa_topk.py', 'fnext/boot', 'fnext/boot/late_prefetch.py', 'fnext/moe', 'fnext/moe/hyb_fuse.py', 'fnext/moe/plugin.py', 'fnext/moe/router_first.py'], 1793542467), 'rc9': {}}

def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(64 << 20), b''):
            h.update(b)
    return h.hexdigest()

def pack_overlay(src_dir, out_file, mtimes=None):

    def add(t, path, arc):
        ti = t.gettarinfo(str(path), arc)
        ti.uid = ti.gid = 0
        ti.uname = ti.gname = 'root'
        ti.mtime = (mtimes or {}).get(arc, 0)
        if ti.isdir():
            ti.mode = 509
            t.addfile(ti)
            for n in sorted(os.listdir(path)):
                if n.startswith('.') or n == '__pycache__' or n.endswith('.pyc'):
                    continue
                add(t, path / n, f'{arc}/{n}')
        elif ti.isfile():
            ti.mode = 436
            with open(path, 'rb') as f:
                t.addfile(ti, f)
        else:
            raise SystemExit(f'{path}: only regular files and directories can be packed')
    with tarfile.open(out_file, 'w', format=tarfile.GNU_FORMAT) as t:
        add(t, Path(src_dir) / 'fnext', 'fnext')

def main():
    ap = argparse.ArgumentParser(description='assemble the uploadable code dataset folder: the flat files of code_dataset/ plus fnext.tarbin packed from code_dataset/fnext/')
    ap.add_argument('out', help='output folder (flat; upload it with `kaggle datasets create|version -p OUT`)')
    ap.add_argument('--src', default=str(HERE.parent / 'code_dataset'))
    ap.add_argument('--release', choices=sorted(OVERLAYS), default='rc8', help='rc8: dataset version 11 (the code of RC8, notebook version 5); rc9: dataset version 13 (RC8 plus the files in code_dataset/rc9/, notebook version 7)')
    ap.add_argument('--fn-cache', help='compile cache to include as fn_cache.tarbin (the published one, or a merge)')
    ap.add_argument('--dataset-id', help='OWNER/SLUG: also write dataset-metadata.json for the Kaggle CLI')
    ap.add_argument('--title', default='Flash-Next ARC code')
    ap.add_argument('--strict', action='store_true', help='exit 1 if a file differs from MANIFEST.md5')
    a = ap.parse_args()
    src, out = (Path(a.src).resolve(), Path(a.out).resolve())
    manifest = src / 'MANIFEST.md5'
    if OVERLAYS[a.release]:
        ovl = src / OVERLAYS[a.release]
        assert ovl.is_dir(), f'{ovl}: overlay folder missing'
        tmp = Path(tempfile.mkdtemp(prefix='code_dataset_')) / 'code_dataset'
        shutil.copytree(src, tmp, ignore=shutil.ignore_patterns(OVERLAYS[a.release], '__pycache__', '*.pyc'))
        for f in sorted(ovl.rglob('*')):
            if f.is_file() and f.name != 'MANIFEST.md5':
                dst = tmp / f.relative_to(ovl)
                assert dst.exists(), f'{f}: the overlay may only replace existing files'
                shutil.copy2(f, dst)
        src, manifest = (tmp, ovl / 'MANIFEST.md5')
    assert (src / 'fnext').is_dir() and (src / 'kaggle_fn.sh').is_file(), f'{src} is not a code_dataset folder'
    assert not out.exists() or not any(out.iterdir()), f'{out} exists and is not empty'
    out.mkdir(parents=True, exist_ok=True)
    for p in sorted(src.iterdir()):
        if p.is_file() and p.name not in SKIP_TOP and (not p.name.startswith('.')):
            shutil.copy2(p, out / p.name)
    pack_overlay(src, out / 'fnext.tarbin', TAR_MTIMES[a.release])
    if a.fn_cache:
        with tarfile.open(a.fn_cache) as t:
            bad = [m.name for m in t.getmembers() if not (m.name in ('.', './triton', './modelinfos') or m.name.startswith(('./triton/', './modelinfos/')))]
        assert not bad, f'{a.fn_cache}: not a compile cache (unexpected members {bad[:5]})'
        shutil.copy2(a.fn_cache, out / 'fn_cache.tarbin')
    if a.dataset_id:
        json.dump({'id': a.dataset_id, 'title': a.title, 'licenses': [{'name': 'apache-2.0'}]}, open(out / 'dataset-metadata.json', 'w'), indent=1)
    want = {}
    if manifest.is_file():
        for line in open(manifest):
            if line.strip():
                h, n = line.split(None, 1)
                want[n.strip().lstrip('*')] = h
    bad = 0
    for n in sorted(set(want) | {p.name for p in out.iterdir() if p.name != 'dataset-metadata.json'}):
        p = out / n
        if not p.exists():
            print(f'  {n:22s} not included' + (' (optional: the launcher compiles kernels at startup instead)' if n == 'fn_cache.tarbin' else ''))
            bad += n != 'fn_cache.tarbin'
            continue
        got = md5(p)
        state = 'matches the published version' if want.get(n) == got else 'NOT in MANIFEST.md5' if n not in want else 'DIFFERS from the published version'
        bad += want.get(n) != got and n != 'fn_cache.tarbin'
        print(f'  {n:22s} {got}  {state}')
    print(f"wrote {out} ({a.release}): {sum((1 for _ in out.iterdir()))} files; {('all match' if not bad else f'{bad} differ')}")
    sys.exit(1 if bad and a.strict else 0)
if __name__ == '__main__':
    main()
