#!/usr/bin/env python3
import argparse, json, os, re, shutil, subprocess, sys, tarfile, tempfile
from pathlib import Path
ENV_TAR = 'arc_py312.tarbin'
CKPT_OLD = 'Path(find(IN + "/**/" + MODE.get("ckpt_marker", "model.safetensors.index.json"))).parent'
CKPT_FN = '\n\ndef _ckpt_dir():\n    """checkpoint directory: ARC_CKPT_DIR when set (the notebook launcher sets it after checking every file: the Kaggle\n    Models mount, a local directory of links merging the checkpoint\'s Model parts, or the checkpoint dataset), else the\n    first sorted IN/**/ckpt_marker match (v27). A set but unusable ARC_CKPT_DIR is an error, never a fallback to a\n    marker search (which could find one part of a split checkpoint)."""\n    marker = MODE.get("ckpt_marker", "model.safetensors.index.json")\n    d = os.environ.get("ARC_CKPT_DIR")\n    if d:\n        if not (Path(d) / marker).is_file():\n            raise FileNotFoundError(f"ARC_CKPT_DIR={d} has no {marker}")\n        return Path(d)\n    return Path(find(IN + "/**/" + marker)).parent\n'
CKPT_ANCHOR = '\n\ndef _mount_fstype(p):'

def patch_run_l4(src):
    if '\ndef _ckpt_dir():' in src:
        return src
    assert src.count(CKPT_OLD) == 3, 'run_l4: expected 3 checkpoint lookups'
    assert src.count(CKPT_ANCHOR) == 1
    return src.replace(CKPT_OLD, '_ckpt_dir()').replace(CKPT_ANCHOR, CKPT_FN + CKPT_ANCHOR)

def patch_so(b, roots):
    n = 0
    for root in roots:
        pat = re.compile(re.escape(root.encode()))

        def rep(m):
            nonlocal n
            n += 1
            s = b'/opt/build'
            return s + b'_' * (len(m.group(0)) - len(s))
        b = pat.sub(rep, b)
    return (b, n)

def main():
    ap = argparse.ArgumentParser(description="Finalise a raw runtime bundle (pack_runtime_dataset.py output or any flat bundle) into an upload folder: kernels without the local build path, the CUDA toolkit tar with numeric owner 0:0 (extracted tree checked equal), the launcher's ARC_CKPT_DIR lookup, dataset-metadata.json, and a scan for local paths. Writes OUT/../<OUT name>_build_report.json; exit 1 if a check fails.")
    ap.add_argument('src', help='raw bundle folder')
    ap.add_argument('out', help='upload folder (replaced)')
    ap.add_argument('--build-root', action='append', default=[], help='path prefix to remove from the compiled kernels (repeatable; default: your home directory)')
    ap.add_argument('--forbid', action='append', default=[], help='regex that must not remain in any file but the env archive (default: your home directory)')
    ap.add_argument('--dataset-id', help="OWNER/SLUG for dataset-metadata.json (default: keep the bundle's, if any)")
    ap.add_argument('--title', default='Dots3-Note ARC runtime')
    a = ap.parse_args()
    src, out = (Path(a.src).resolve(), Path(a.out).resolve())
    home = os.path.expanduser('~')
    roots = a.build_root or [home]
    bad = re.compile('|'.join(a.forbid or [re.escape(home)]).encode(), re.I)
    keep = out / ENV_TAR
    held = None
    if keep.exists() and (not (src / ENV_TAR).exists()):
        held = out.parent / f'.{out.name}_{ENV_TAR}'
        os.replace(keep, held)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    if held:
        os.replace(held, keep)
    rep = {'src': str(src), 'out': str(out), 'files': {}, 'fail': []}
    for f in sorted(src.iterdir()):
        if f.is_dir():
            continue
        d, name = (out / f.name, f.name)
        if name == 'dataset-metadata.json' and a.dataset_id:
            continue
        if name == 'run_l4.py':
            s = f.read_text()
            new = patch_run_l4(s)
            if new != s:
                d.write_text(new)
                rep['files'][name] = {'ckpt_patch': True}
                continue
        if name.endswith('.so'):
            b = f.read_bytes()
            nb, n = patch_so(b, roots)
            if n:
                d.write_bytes(nb)
                rep['files'][name] = {'paths_overwritten': n, 'size_equal': len(b) == len(nb), 'bytes_changed': sum((1 for x, y in zip(b, nb) if x != y))}
                continue
        if name == 'cuda_toolkit.tarbin':
            own = {(m.uid, m.gid, m.uname, m.gname) for m in tarfile.open(f)}
            if own != {(0, 0, '', '')} and own != {(0, 0, 'root', 'root')}:
                tmp = Path(tempfile.mkdtemp(dir=str(out.parent), prefix='.cudax_'))
                tmp2 = Path(tempfile.mkdtemp(dir=str(out.parent), prefix='.cudax2_'))
                try:
                    subprocess.run(['tar', '-C', str(tmp), '-xf', str(f)], check=True)
                    members = [m.name for m in tarfile.open(f)]
                    subprocess.run(['tar', '--numeric-owner', '--owner=0', '--group=0', '--no-recursion', '-C', str(tmp), '-cf', str(d), '--files-from=-'], input='\n'.join(members).encode(), check=True)
                    subprocess.run(['tar', '-C', str(tmp2), '-xf', str(d)], check=True)
                    r = subprocess.run(['diff', '-r', '--no-dereference', str(tmp), str(tmp2)], capture_output=True, text=True)
                    m_old = [(m.name, m.size, m.mode, m.type, m.linkname) for m in tarfile.open(f)]
                    m_new = [(m.name, m.size, m.mode, m.type, m.linkname) for m in tarfile.open(d)]
                    rep['files'][name] = {'extracted_trees_identical': r.returncode == 0, 'members_identical': m_old == m_new, 'owners': sorted(map(str, {(m.uname, m.gname, m.uid, m.gid) for m in tarfile.open(d)}))}
                    if r.returncode != 0 or m_old != m_new:
                        rep['fail'].append(f'{name}: tree differs {r.stdout[:300]}')
                finally:
                    shutil.rmtree(tmp)
                    shutil.rmtree(tmp2)
                continue
        os.link(f, d)
    if a.dataset_id:
        json.dump({'id': a.dataset_id, 'title': a.title, 'licenses': [{'name': 'other'}]}, open(out / 'dataset-metadata.json', 'w'), indent=1)
        rep['files']['dataset-metadata.json'] = 'new'
    left = {}
    for f in sorted(out.iterdir()):
        if f.name == ENV_TAR:
            continue
        with open(f, 'rb') as fh:
            hits = set()
            for blk in iter(lambda: fh.read(64 << 20), b''):
                hits |= {m.group(0).decode(errors='replace').lower() for m in bad.finditer(blk)}
        if hits:
            left[f.name] = sorted(hits)
    rep['leftover_terms'] = left
    if left:
        rep['fail'].append(f'leftover terms: {left}')
    out_rep = out.parent / f'{out.name}_build_report.json'
    json.dump(rep, open(out_rep, 'w'), indent=1)
    print(json.dumps(rep, indent=1)[:6000])
    print('FAIL' if rep['fail'] else 'OK', out_rep)
    sys.exit(1 if rep['fail'] else 0)
if __name__ == '__main__':
    main()
