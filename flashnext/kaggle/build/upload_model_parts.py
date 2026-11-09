#!/usr/bin/env python3
import argparse, csv, io, json, os, shutil, subprocess, sys, time
from pathlib import Path
K = os.environ.get('KAGGLE_CLI', 'kaggle')

def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)

def run(cmd, check=True, capture=True, timeout=None):
    r = subprocess.run(cmd, capture_output=capture, text=True, timeout=timeout)
    if check and r.returncode:
        raise RuntimeError(f"{' '.join(cmd)[:200]} -> rc {r.returncode}: {(r.stdout or '')[-800:]} {(r.stderr or '')[-800:]}")
    return r

def kaggle_create(cmd):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    bad = []
    for ln in p.stdout:
        if 'Upload successful' in ln or 'Starting upload' in ln:
            continue
        print('   ', ln.rstrip()[-300:], flush=True)
        if 'error' in ln.lower() or 'failed' in ln.lower():
            bad.append(ln.strip())
    rc = p.wait()
    if rc or bad:
        raise RuntimeError(f"{' '.join(cmd[:4])}: rc {rc}; {bad[:3]}")

def kaggle_csv(args):
    rows, token = ([], None)
    for _ in range(50):
        r = run([K] + args + ['--format', 'csv'] + (['--page-token', token] if token else []), check=False)
        if r.returncode:
            return None if not rows else rows
        token = None
        body = []
        for ln in r.stdout.splitlines():
            if ln.startswith('Next Page Token = '):
                token = ln.split('=', 1)[1].strip()
            elif ln and (not ln.startswith('Warning')):
                body.append(ln)
        rows += list(csv.DictReader(io.StringIO('\n'.join(body))))
        if not token:
            break
    return rows

def version_files(owner, slug, part, version):
    rows = kaggle_csv(['models', 'instances', 'versions', 'files', f'{owner}/{slug}/pyTorch/{part}/{version}', '--page-size', '100'])
    return None if rows is None else {r.get('name'): int(r.get('size') or -1) for r in rows}

def latest_version(owner, slug, part):
    rows = kaggle_csv(['models', 'instances', 'versions', 'list', f'{owner}/{slug}/pyTorch/{part}'])
    nums = [int(r['version']) for r in rows or [] if r.get('variation') == part and str(r.get('version', '')).isdigit()]
    return max(nums) if nums else None

def part_complete(owner, slug, part, want):
    v = latest_version(owner, slug, part)
    if v is None:
        return (None, False)
    got = version_files(owner, slug, part, v) or {}
    return (v, all((got.get(n) == s for n, s in want.items())))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, help='make_model_parts.py output dir (parts are uploaded in place)')
    ap.add_argument('--parts', default='', help='comma list (default: all parts in the manifest)')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--keep-local', action='store_true')
    ap.add_argument('--force', action='store_true', help='upload a new version even if the latest one is complete')
    ap.add_argument('--poll-timeout-s', type=int, default=7200)
    a = ap.parse_args()
    stage = Path(a.stage).resolve()
    assert (stage / 'manifest.json').is_file(), f'no manifest.json in {stage} (run make_model_parts.py first)'
    man = json.load(open(stage / 'manifest.json'))
    owner, slug = (man['owner'], man['slug'])
    parts = [p for p in (a.parts.split(',') if a.parts else man['parts']) if p]
    assert all((p in man['parts'] for p in parts)), f"unknown parts: {set(parts) - set(man['parts'])}"
    state_p = stage / 'upload_state.json'
    state = json.load(open(state_p)) if state_p.exists() else {}
    biggest = max((sum((man['files'][n] for n in man['parts'][p])) for p in parts))
    log(f'model {owner}/{slug}: {len(parts)} parts, largest {biggest / 1000000000.0:.1f} GB')
    meta_dir = stage / f'{slug}-meta'
    (stage / '_probe').mkdir(exist_ok=True)
    probe = run([K, 'models', 'get', f'{owner}/{slug}', '-p', str(stage / '_probe')], check=False)
    if probe.returncode or not (stage / '_probe' / 'model-metadata.json').exists():
        meta = json.load(open(meta_dir / 'model-metadata.json'))
        assert meta.get('isPrivate') is True and meta['ownerSlug'] == owner and (meta['slug'] == slug), meta
        log('create model', f'{owner}/{slug}', '(private)')
        if not a.dry_run:
            kaggle_create([K, 'models', 'create', '-p', str(meta_dir)])
    else:
        log('model exists:', f'{owner}/{slug}')
    for part in parts:
        want = {n: man['files'][n] for n in man['parts'][part]}
        v, ok = part_complete(owner, slug, part, want)
        if ok and (not a.force):
            log(f'{part}: version {v} already complete ({len(want)} files) - skip')
            state[part] = {'version': v, 'status': 'done'}
            json.dump(state, open(state_p, 'w'), indent=1)
            continue
        ldir = stage / f'{slug}-{part}'
        bad = [n for n, s in want.items() if not (ldir / n).is_file() or (ldir / n).stat().st_size != s]
        assert not bad, f'{part}: local files missing or wrong size: {bad[:5]}'
        extra = sorted(set(os.listdir(ldir)) - set(want) - {'model-instance-metadata.json'})
        assert not extra, f'{part}: unexpected files in the part dir: {extra}'
        log(f'{part}: {sum(want.values()) / 1000000000.0:.2f} GB, {len(want)} files checked')
        if a.dry_run:
            log(f"{part}: dry run - would upload {('a new version' if v else 'a new variation')}")
        else:
            t1 = time.time()
            if v is None:
                kaggle_create([K, 'models', 'instances', 'create', '-p', str(ldir), '-r', 'skip', '-q'])
            else:
                kaggle_create([K, 'models', 'instances', 'versions', 'create', f'{owner}/{slug}/pyTorch/{part}', '-p', str(ldir), '-r', 'skip', '-q', '--ignore-patterns', 'model-instance-metadata.json', '-n', f"rolling upload {time.strftime('%Y-%m-%d %H:%M')}"])
            log(f'{part}: uploaded in {time.time() - t1:.0f} s; waiting until Kaggle lists every file')
            deadline = time.time() + a.poll_timeout_s
            while True:
                v2, ok = part_complete(owner, slug, part, want)
                if ok and (v is None or v2 > v):
                    break
                assert time.time() < deadline, f'{part}: not complete on Kaggle after {a.poll_timeout_s} s'
                time.sleep(30)
            log(f'{part}: version {v2} verified ({len(want)} files, sizes match)')
            state[part] = {'version': v2, 'status': 'done', 'bytes': sum(want.values())}
            json.dump(state, open(state_p, 'w'), indent=1)
        if not a.keep_local and (not a.dry_run):
            shutil.rmtree(ldir)
            log(f'{part}: local copy deleted')
    log('all requested parts done' + (' (dry run: nothing uploaded)' if a.dry_run else ''))
    print(json.dumps(state, indent=1))
if __name__ == '__main__':
    main()
