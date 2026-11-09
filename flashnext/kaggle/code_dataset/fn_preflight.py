import json, os, sys
code, base, *deltas = sys.argv[1:]
assert deltas, 'need at least one delta dir'
sys.path.insert(0, os.path.join(code, 'fnext', 'moe'))
sys.path.insert(0, code)
import torch
import fnmoe
import delta_build
for d in deltas:
    delta_build.verify_manifest(d)
for l in (0, 48):
    pk = torch.load(f'{base}/lowbit-L{l:02d}.pt', map_location='cpu', mmap=True, weights_only=False)
    fnmoe.apply_scale_deltas(pk, l, deltas)
    r = torch.load(fnmoe.router_path(l, base, deltas), map_location='cpu', weights_only=False)
    assert r['weight'].shape[0] == len(pk['experts']), f'L{l}: router rows {r['weight'].shape[0]} != experts {len(pk['experts'])}'
keep = next((f'{d}/keep.json' for d in reversed(deltas) if os.path.exists(f'{d}/keep.json')), f'{base}/keep.json')
k = json.load(open(keep))
assert len(k) == 49 and len({len(v) for v in k.values()}) == 1, f'{keep}: unexpected layout'
print(f'PREFLIGHT OK: {len(deltas)} delta(s), K={len(k['0'])}, keep from {os.path.basename(os.path.dirname(keep))}')
