import json, sys, collections
import torch
A, B = (sys.argv[1], sys.argv[2])

def leaves(x, y, path=''):
    if isinstance(x, dict):
        for k in x:
            yield from leaves(x[k], y.get(k) if isinstance(y, dict) else None, f'{path}.{k}' if path else str(k))
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            yield from leaves(v, y[i] if isinstance(y, (list, tuple)) and i < len(y) else None, f'{path}[{i}]')
    else:
        yield (path, x, y)
tot = collections.Counter()
diff = collections.Counter()
nd = collections.Counter()
other = []
for l in range(49):
    a = torch.load(f'{A}/lowbit-L{l:02d}.pt', map_location='cpu', mmap=True, weights_only=False)
    b = torch.load(f'{B}/lowbit-L{l:02d}.pt', map_location='cpu', mmap=True, weights_only=False)
    if set(a) != set(b) or list(a['experts']) != list(b['experts']):
        other.append(f'L{l}: expert sets differ')
    for e in a['experts']:
        e = int(e) if not isinstance(e, int) else e
        for k, va, vb in leaves(a[e], b[e]):
            if torch.is_tensor(va):
                sz = va.numel() * va.element_size()
                tot[k] += sz
                same = torch.is_tensor(vb) and va.dtype == vb.dtype and (va.shape == vb.shape) and torch.equal(va, vb)
                if not same:
                    diff[k] += sz
                    nd[k] += 1
            elif va != vb:
                other.append(f'L{l} e{e} {k}: {va!r} != {vb!r}')
    ra = torch.load(f'{A}/router-L{l:02d}.pt', map_location='cpu', weights_only=False)
    rb = torch.load(f'{B}/router-L{l:02d}.pt', map_location='cpu', weights_only=False)
    tot['router'] += ra['weight'].numel() * 2
    if not torch.equal(ra['weight'], rb['weight']):
        diff['router'] += ra['weight'].numel() * 2
        nd['router'] += 1
    if not torch.equal(torch.as_tensor(ra['keep']), torch.as_tensor(rb['keep'])):
        other.append(f'L{l} router keep differs')
for k in sorted(tot):
    print(f'{k:8s} total {tot[k] / 1000000000.0:7.3f} GB  differ {diff[k] / 1000000000.0:7.3f} GB  in {nd[k]} tensors')
print('keep.json equal:', json.load(open(f'{A}/keep.json')) == json.load(open(f'{B}/keep.json')))
print('other:', other[:10], len(other))
