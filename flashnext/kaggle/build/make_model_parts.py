#!/usr/bin/env python3
import argparse, fnmatch, hashlib, json, os, sys
from pathlib import Path
SMALL_MAX = 64 << 20
LICENSE_NAMES = ('LICENSE', 'LICENSE.txt', 'NOTICE')

def sha256(p, blk=64 << 20):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while True:
            b = f.read(blk)
            if not b:
                return h.hexdigest()
            h.update(b)

def plan(files, part_gb, ple_glob, tag='w', small_max=None, max_files=48):
    cap = int(part_gb * 1000000000.0)
    small_max = SMALL_MAX if small_max is None else small_max
    small = sorted((n for n, s in files.items() if s < small_max))
    big = sorted((n for n, s in files.items() if s >= small_max))
    max_big = max_files - len(small)
    groups = {tag: [n for n in big if not fnmatch.fnmatch(n, ple_glob)], 'ple': [n for n in big if fnmatch.fnmatch(n, ple_glob)]}
    parts = []
    for tag, names in groups.items():
        cur, cur_b, k = ([], 0, 1)
        for n in names:
            if cur and (cur_b + files[n] > cap or len(cur) >= max_big):
                parts.append((f'{tag}-p{k}', cur))
                k += 1
                cur, cur_b = ([], 0)
            cur.append(n)
            cur_b += files[n]
        if cur:
            parts.append((f'{tag}-p{k}', cur))
    return ([(name, sorted(names + small)) for name, names in parts], small)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt')
    ap.add_argument('out')
    ap.add_argument('--owner', required=True)
    ap.add_argument('--slug', required=True)
    ap.add_argument('--title', default='Qwen3.8-Flash-Next ARC build')
    ap.add_argument('--subtitle', default='Quantized Qwen3.8-Flash-Next checkpoint for 4xL4 inference')
    ap.add_argument('--summary', default='A quantized build of Qwen3.8-Flash-Next, split into several variations. Attach all of them.')
    ap.add_argument('--part-gb', type=float, default=15.0)
    ap.add_argument('--tag', default='w', help='variation name prefix of the non-PLE parts (e.g. k384-q3)')
    ap.add_argument('--small-mb', type=float, default=64, help='files below this go into every part')
    ap.add_argument('--max-files', type=int, default=48, help='files per part (Kaggle allows 50 per version)')
    ap.add_argument('--ple-glob', default='model-ple*')
    ap.add_argument('--sha256', action='store_true', help='hash every file into the manifest (slow on 100+ GB)')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--symlink', action='store_true', help='symlink instead of hard-link (source on another filesystem)')
    a = ap.parse_args()
    ck, out = (Path(a.ckpt), Path(a.out))
    files = {p.name: p.stat().st_size for p in ck.iterdir() if p.is_file() and (not p.name.startswith('.'))}
    assert files, f'no files in {ck}'
    lic = [n for n in LICENSE_NAMES if n in files]
    assert lic, "the checkpoint has no LICENSE file: copy the base model's licence in before splitting"
    parts, small = plan(files, a.part_gb, a.ple_glob, a.tag, int(a.small_mb * 2 ** 20), a.max_files)
    total = sum(files.values())
    print(f'{len(files)} files, {total / 2 ** 30:.1f} GiB -> {len(parts)} parts ({len(small)} small files in each)')
    for name, names in parts:
        print(f'  {name:8s} {sum((files[n] for n in names)) / 1000000000.0:6.2f} GB  {len(names)} files')
    if a.dry_run:
        return
    out.mkdir(parents=True, exist_ok=True)
    manifest = {'files': {n: files[n] for n in sorted(files)}, 'parts': {}, 'slug': a.slug, 'owner': a.owner}
    if a.sha256:
        manifest['sha256'] = {n: sha256(ck / n) for n in sorted(files)}
    for name, names in parts:
        d = out / f'{a.slug}-{name}'
        d.mkdir(exist_ok=True)
        for n in names:
            dst = d / n
            if not dst.exists():
                if a.symlink:
                    os.symlink((ck / n).resolve(), dst)
                else:
                    os.link(ck / n, dst)
        manifest['parts'][name] = names
        json.dump({'ownerSlug': a.owner, 'modelSlug': a.slug, 'instanceSlug': name, 'framework': 'pyTorch', 'overview': f'{a.title}, part {name}. Attach all parts and place their files in one directory.', 'usage': '# Model Format\n\nHugging Face style directory split into several variations. Attach every variation; the files of all variations together form one checkpoint directory.\n\n# License\n\nQwen Community License 1.0 (LICENSE included), as the base Qwen3.8-Flash-Next weights (https://huggingface.co/Qwen/Qwen3.8-Flash-Next).\n', 'licenseName': 'Other', 'fineTunable': False, 'trainingData': [], 'modelInstanceType': 'Unspecified', 'baseModelInstance': '', 'externalBaseModelUrl': ''}, open(d / 'model-instance-metadata.json', 'w'), indent=2)
    meta = out / f'{a.slug}-meta'
    meta.mkdir(exist_ok=True)
    json.dump({'ownerSlug': a.owner, 'title': a.title, 'slug': a.slug, 'isPrivate': True, 'subtitle': a.subtitle, 'description': '# Model Summary\n\n' + a.summary + '\n\n# License\n\nQwen Community License 1.0 (see LICENSE in every part). Derived from Qwen/Qwen3.8-Flash-Next. Private model.\n', 'publishTime': '', 'provenanceSources': ''}, open(meta / 'model-metadata.json', 'w'), indent=2)
    json.dump(manifest, open(out / 'manifest.json', 'w'), indent=1)
    k = '$K'
    print('\n# upload commands (run only after approval; K = the kaggle CLI)')
    print(f'{k} models create -p {meta}')
    for name, _ in parts:
        print(f"{k} models instances create -p {out / (a.slug + '-' + name)}")
    for name, _ in parts:
        print(f'{k} models instances versions files {a.owner}/{a.slug}/pyTorch/{name}/1')
    print(f"\nmanifest: {out / 'manifest.json'}")
if __name__ == '__main__':
    sys.exit(main())
