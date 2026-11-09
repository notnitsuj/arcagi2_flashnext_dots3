import json, os, re, subprocess, sys, tarfile, tempfile
from collections import defaultdict
RESTORE = '/tmp/fn/vcache/triton'
_HOME = os.path.expanduser('~')
_FORBID = ([re.escape(_HOME)] if _HOME not in ('', '/') else []) + ([os.environ['FORBID']] if os.environ.get('FORBID') else [])
BAD = re.compile('|'.join(_FORBID).encode(), re.I) if _FORBID else None
KDIR = re.compile('^(?:\\./)?triton/([A-Z0-9]{52})/([^/]+)$')
base, out, deltas = (sys.argv[1], sys.argv[2], sys.argv[3:])
assert deltas, 'usage: merge_fn_cache.py BASE_TARBIN OUT_TARBIN DELTA_TAR [DELTA_TAR ...]'
d = tempfile.mkdtemp(prefix='fncache_')
subprocess.run(['tar', '-C', d, '-xf', base], check=True)
have = set(os.listdir(os.path.join(d, 'triton')))
n0 = len(have)
added, skipped = ([], defaultdict(int))
for dt in deltas:
    with tarfile.open(dt) as t:
        mem = [m for m in t.getmembers() if m.isfile()]
        dirs = defaultdict(dict)
        for m in mem:
            k = KDIR.match(m.name)
            if not k:
                skipped['not a triton kernel file: ' + m.name.split('/')[1 if m.name.startswith('./') else 0]] += 1
                continue
            dirs[k.group(1)][k.group(2)] = m
        for h, fs in sorted(dirs.items()):
            if h in have:
                skipped['hash already present'] += 1
                continue
            grp = [f for f in fs if f.startswith('__grp__') and f.endswith('.json')]
            if len(grp) != 1:
                skipped['no single group file (partial dir)'] += 1
                continue
            g = json.loads(t.extractfile(fs[grp[0]]).read())
            kids = {os.path.basename(v) for v in g.get('child_paths', {}).values()}
            if not kids or not kids <= set(fs):
                skipped['group lists missing files (partial dir)'] += 1
                continue
            os.makedirs(os.path.join(d, 'triton', h))
            for f, m in fs.items():
                data = t.extractfile(m).read()
                if f == grp[0]:
                    g['child_paths'] = {c: f'{RESTORE}/{h}/{os.path.basename(v)}' for c, v in g['child_paths'].items()}
                    data = json.dumps(g).encode()
                hit = BAD.search(data) if BAD is not None else None
                assert not hit, (dt, h, f, hit.group(0))
                with open(os.path.join(d, 'triton', h, f), 'wb') as w:
                    w.write(data)
                os.utime(os.path.join(d, 'triton', h, f), (m.mtime, m.mtime))
            have.add(h)
            added.append((h, next((f for f in fs if f.endswith('.cubin')), sorted(fs)[0])))
subprocess.run(['tar', '-C', d, '--owner=0', '--group=0', '--sort=name', '-cf', out, '.'], check=True)
names = defaultdict(int)
for h, f in added:
    names[re.sub('\\.\\w+$', '', f)] += 1
print(f'base {n0} kernel dirs + {len(added)} new = {len(have)} -> {out} ({os.path.getsize(out) / 2 ** 20:.1f} MiB)')
print('new kernels:', dict(sorted(names.items(), key=lambda x: -x[1])))
print('skipped:', dict(skipped) or 'none')
subprocess.run(['rm', '-rf', d], check=True)
