import json, os, random, shutil, sys
from pathlib import Path
D = Path(os.environ.get('ARC_DATA', 'data/ARC-AGI-2/data'))
FT = os.path.join(os.environ.get('D3_WORK', 'work'), 'ft')
split = json.load(open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'split.json')))
tr = sorted((os.path.splitext(f)[0] for f in os.listdir(D / 'training') if f.endswith('.json')))
ev = set((os.path.splitext(f)[0] for f in os.listdir(D / 'evaluation') if f.endswith('.json')))
pilot = set(split['pilot'])
pool = [t for t in tr if t not in pilot and t not in ev]
rng = random.Random(20261013)
val = sorted(rng.sample(pool, 200))
gen = sorted(set(pool) - set(val))
assert val == sorted(split['val']) and gen == sorted(split['gen']), 'split.json does not match this ARC-AGI-2 training set'
assert not set(val) & ev and (not set(gen) & ev) and (not set(gen) & set(val))

def link(t, out):
    os.makedirs(out, exist_ok=True)
    d = os.path.join(out, f'{t}.json')
    if not os.path.exists(d):
        try:
            os.link(D / 'training' / f'{t}.json', d)
        except OSError:
            shutil.copy(D / 'training' / f'{t}.json', d)
for s in ('easy', 'hard', 'unk'):
    for t, st in split['gen'].items():
        if st == s:
            link(t, os.path.join(FT, f'gen_{s}'))
for t in val:
    link(t, os.path.join(FT, 'val'))
for s in ('easy', 'hard', 'unk'):
    print(f'gen_{s}:', sum((v == s for v in split['gen'].values())), 'tasks;', 'val:', sum((v == s for v in split['val'].values())))
print('val', len(val), 'gen', len(gen), 'pilot (excluded)', len(pilot), '->', FT)
