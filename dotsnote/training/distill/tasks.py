import argparse, json, os, random
from pathlib import Path
DATA = Path(os.environ.get('ARC_DATA', 'data/ARC-AGI-2/data'))

def input_grids(path):
    j = json.load(open(path))
    return {json.dumps(p['input']) for p in j['train'] + j['test']}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(os.environ.get('D3_WORK', 'work'), 'kd'))
    ap.add_argument('--train', type=int, default=560)
    ap.add_argument('--val', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    tr = sorted((p.stem for p in (DATA / 'training').glob('*.json')))
    ev = sorted((p.stem for p in (DATA / 'evaluation').glob('*.json')))
    assert tr and ev and (not set(tr) & set(ev)), 'training and evaluation ids must be disjoint and non-empty'
    eg = set()
    for e in ev:
        eg |= input_grids(DATA / 'evaluation' / f'{e}.json')
    shared = [t for t in tr if input_grids(DATA / 'training' / f'{t}.json') & eg]
    ok = [t for t in tr if t not in shared]
    random.Random(a.seed).shuffle(ok)
    train, val = (sorted(ok[:a.train]), sorted(ok[a.train:a.train + a.val]))
    assert not (set(train) | set(val)) & set(ev)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'tasks_train.txt').write_text('\n'.join(train) + '\n')
    (out / 'tasks_val.txt').write_text('\n'.join(val) + '\n')
    (out / 'tasks_all.txt').write_text('\n'.join(val + train) + '\n')
    print(f'{len(tr)} training / {len(ev)} evaluation tasks; {len(shared)} training tasks share an input grid with an evaluation task (excluded); wrote {len(train)} train + {len(val)} validation tasks to {out}')
if __name__ == '__main__':
    main()
