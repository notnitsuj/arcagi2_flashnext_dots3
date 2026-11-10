import argparse, json, os, re
from pathlib import Path
DATA = Path(os.environ.get('ARC_DATA', 'data/ARC-AGI-2/data'))
W = os.environ.get('D3_WORK', 'work')

def grid_index(split):
    idx = {}
    for f in sorted((DATA / split).glob('*.json')):
        j = json.load(open(f))
        for p in j['train'] + j['test']:
            idx.setdefault(json.dumps(p['input']), f.stem)
    return idx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mix', default=os.path.join(W, 'calib', 'calib_mix.jsonl'))
    ap.add_argument('--direct', default=os.path.join(W, 'calib', 'direct_text.jsonl'))
    ap.add_argument('--val-tasks', default=os.path.join(W, 'kd', 'tasks_val.txt'))
    ap.add_argument('--out', default=os.path.join(W, 'kd', 'calib_text.jsonl'))
    a = ap.parse_args()
    ev, tr = (grid_index('evaluation'), grid_index('training'))
    assert ev and tr, f'no ARC-AGI-2 tasks under {DATA}'
    val = set(open(a.val_tasks).read().split())

    def find(prompt):
        for g in re.findall('\\[\\[[0-9, \\[\\]]*\\]\\]', prompt):
            key = json.dumps(json.loads(g))
            if key in ev:
                return 'EVAL:' + ev[key]
            if key in tr:
                return tr[key]
        return None
    n = skip = 0
    with open(a.out, 'w') as out:
        for i, l in enumerate(open(a.mix)):
            r = json.loads(l)['full']
            t = find(r['prompt'])
            if t is None or t.startswith('EVAL') or t in val:
                print('skip', t, flush=True)
                skip += 1
                continue
            out.write(json.dumps({'task': t, 'test': 0, 'attempt': 100 + i, 'prompt': r['prompt'], 'completion': r['completion']}) + '\n')
            n += 1
        for l in open(a.direct):
            r = json.loads(l)
            t = find(r['prompt'])
            if t is None or t.startswith('EVAL') or t in val or (t != r['task']):
                print('skip', t, r['task'], flush=True)
                skip += 1
                continue
            out.write(json.dumps({'task': t, 'test': r['test'], 'attempt': 200 + r['attempt'], 'prompt': r['prompt'], 'completion': r['completion']}) + '\n')
            n += 1
    print(f'wrote {n} traces to {a.out}, skipped {skip}')
if __name__ == '__main__':
    main()
