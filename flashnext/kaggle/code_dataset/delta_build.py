import hashlib, json, os, re, shutil, sys
import torch
L = range(49)

def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(64 << 20), b''):
            h.update(b)
    return h.hexdigest()

def make(base, der, out):
    os.makedirs(out, exist_ok=True)
    for l in L:
        d = torch.load(f'{der}/lowbit-L{l:02d}.pt', map_location='cpu', weights_only=False)
        sc = {e: {'gu_s': d[e]['gu_s'].clone(), 'd_s': d[e]['d_s'].clone()} for e in d['experts']}
        torch.save(sc, f'{out}/scales-L{l:02d}.pt')
        shutil.copy2(f'{der}/router-L{l:02d}.pt', f'{out}/router-L{l:02d}.pt')
    shutil.copy2(f'{der}/keep.json', f'{out}/keep.json')
    if os.path.exists(f'{der}/norms.safetensors'):
        shutil.copy2(f'{der}/norms.safetensors', f'{out}/norms.safetensors')

def verify_manifest(delta):
    mp = os.path.join(delta, 'manifest.json')
    if not os.path.exists(mp):
        return
    for n, v in json.load(open(mp)).items():
        if n.startswith('_') or not isinstance(v, dict):
            continue
        p = os.path.join(delta, n)
        assert os.path.getsize(p) == v['bytes'] and md5(p) == v['md5'], f'{p}: does not match manifest.json'

def build(base, *rest):
    *deltas, out = rest
    assert deltas, 'need at least one delta dir'
    os.makedirs(out, exist_ok=True)
    for d in deltas:
        verify_manifest(d)
    for l in L:
        b = torch.load(f'{base}/lowbit-L{l:02d}.pt', map_location='cpu', weights_only=False)
        router = f'{base}/router-L{l:02d}.pt'
        for d in deltas:
            sp = f'{d}/scales-L{l:02d}.pt'
            if os.path.exists(sp):
                sc = torch.load(sp, map_location='cpu', weights_only=False)
                assert list(sc) == list(b['experts']), f'L{l}: expert ids differ'
                for e, v in sc.items():
                    b[e]['gu_s'], b[e]['d_s'] = (v['gu_s'], v['d_s'])
            if os.path.exists(f'{d}/router-L{l:02d}.pt'):
                router = f'{d}/router-L{l:02d}.pt'
        torch.save(b, f'{out}/lowbit-L{l:02d}.pt')
        shutil.copy2(router, f'{out}/router-L{l:02d}.pt')
    keep = next((f'{d}/keep.json' for d in reversed(deltas) if os.path.exists(f'{d}/keep.json')), f'{base}/keep.json')
    shutil.copy2(keep, f'{out}/keep.json')
    skip = re.compile('^(scales|router|lowbit)-L\\d+\\.pt$|^(keep\\.json|manifest\\.json|LICENSE)$')
    for d in deltas:
        for n in sorted(os.listdir(d)):
            if not skip.match(n) and os.path.isfile(f'{d}/{n}'):
                shutil.copy2(f'{d}/{n}', f'{out}/{n}')

def check(reb, der):
    same_md5 = same_t = 0
    bad = []
    names = [f'lowbit-L{l:02d}.pt' for l in L] + [f'router-L{l:02d}.pt' for l in L] + ['keep.json']
    for n in names:
        if md5(f'{reb}/{n}') == md5(f'{der}/{n}'):
            same_md5 += 1
            continue
        if n.endswith('.pt'):
            x = torch.load(f'{reb}/{n}', map_location='cpu', weights_only=False)
            y = torch.load(f'{der}/{n}', map_location='cpu', weights_only=False)
            if eq(x, y):
                same_t += 1
                continue
        bad.append(n)
    print(f'{len(names)} files: {same_md5} md5-identical, {same_t} tensor-identical (bytes differ), {len(bad)} DIFFERENT {bad[:5]}')
    return not bad

def eq(x, y):
    if isinstance(x, dict):
        return isinstance(y, dict) and list(x) == list(y) and all((eq(x[k], y[k]) for k in x))
    if isinstance(x, (list, tuple)):
        return type(x) == type(y) and len(x) == len(y) and all((eq(a, b) for a, b in zip(x, y)))
    if torch.is_tensor(x):
        return torch.is_tensor(y) and x.dtype == y.dtype and (x.shape == y.shape) and torch.equal(x, y)
    return x == y
if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'make':
        make(*sys.argv[2:5])
    elif cmd == 'build':
        build(*sys.argv[2:])
    elif cmd == 'check':
        sys.exit(0 if check(*sys.argv[2:4]) else 1)
