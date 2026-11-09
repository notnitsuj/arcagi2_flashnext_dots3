import argparse, collections, hashlib, json, os, statistics as st, sys
from pathlib import Path
import numpy as np
FT = Path(os.environ.get('D3_WORK', 'work')) / 'ft'
sys.path.insert(0, os.environ.get('D3_LOOP_DETECT_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'kaggle', 'runtime_dataset')))
from loop_detect import first_fire
D = Path(os.environ.get('ARC_DATA', 'data/ARC-AGI-2/data'))
EOS = 151668

def ghash(g):
    return hashlib.md5(json.dumps(g).encode()).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--maxlen', type=int, default=40960, help='drop sequences longer than this (prompt + completion)')
    ap.add_argument('--hard-w', type=int, default=2)
    ap.add_argument('--out', default=str(FT / 'sft'))
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.environ.get('D3_LB144R', os.path.join(os.environ.get('D3_WORK', 'work'), 'lb144r')))
    split = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'split.json')))
    val = set(split['val'])
    ev = {p.stem for p in (D / 'evaluation').glob('*.json')}
    held = set()
    for t in val:
        js = json.load(open(D / 'training' / f'{t}.json'))
        held |= {ghash(p['input']) for p in js['train'] + js['test']}
    for t in ev:
        js = json.load(open(D / 'evaluation' / f'{t}.json'))
        held |= {ghash(p['input']) for p in js['train'] + js['test']}
    unk_r1 = collections.defaultdict(list)
    for l in open(FT / 'gen_unk.jsonl'):
        r = json.loads(l)
        unk_r1[r['task']].append(bool(r['correct']))
    hard_tasks = {t for t, s in split['gen'].items() if s == 'hard'}
    hard_tasks |= {t for t, v in unk_r1.items() if not (len(v) >= 2 and all(v))}
    stats = collections.Counter()
    cand = collections.defaultdict(list)
    n_correct_by_key = collections.Counter()
    for s in ('hard', 'easy', 'unk', 'unk2'):
        txt = {}
        for l in open(FT / f'gen_{s}_text.jsonl'):
            r = json.loads(l)
            txt[r['task'], r['test'], r['attempt']] = r
        for l in open(FT / f'gen_{s}.jsonl'):
            r = json.loads(l)
            stats['rows'] += 1
            if not r['correct']:
                continue
            stats['correct'] += 1
            k = (r['task'], r['test'], r['attempt'])
            assert r['task'] not in val and r['task'] not in ev, r['task']
            t = txt.get(k)
            if t is None:
                stats['drop_no_text'] += 1
                continue
            c = t['completion']
            if r['finish'] != 'stop' or not r['closed_think']:
                stats['drop_not_natural'] += 1
                continue
            if 'Time is up' in c or 'time is up' in c.lower():
                stats['drop_forced_text'] += 1
                continue
            if first_fire(c) is not None:
                stats['drop_loop'] += 1
                continue
            js = json.load(open(D / 'training' / f"{r['task']}.json"))
            if any((ghash(p['input']) in held for p in js['train'] + js['test'])):
                stats['drop_grid_overlap'] += 1
                continue
            stats['clean'] += 1
            n_correct_by_key[r['task'], r['test']] += 1
            cand[r['task'], r['test']].append((r['ct'], t['prompt'], c, s, r['attempt']))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    man, saving = ([], [])
    for (task, ti), lst in sorted(cand.items()):
        lst.sort(key=lambda x: x[0])
        ct, prompt, comp, src, att = lst[0]
        if len(lst) >= 2:
            saving.append(ct / st.mean((x[0] for x in lst)))
        p_ids = tok(prompt, add_special_tokens=False).input_ids
        c_ids = tok(comp, add_special_tokens=False).input_ids + [EOS]
        if len(p_ids) + len(c_ids) > a.maxlen:
            stats['drop_too_long'] += 1
            continue
        hard = task in hard_tasks
        w = a.hard_w if hard else 1
        name = f'{task}_{ti}'
        np.savez(out / f'{name}.npz', ids=np.array(p_ids + c_ids, dtype=np.int32), plen=len(p_ids), w=w)
        man.append(dict(name=name, task=task, test=ti, src=src, attempt=att, hard=hard, w=w, plen=len(p_ids), clen=len(c_ids), ct_server=ct, n_clean=len(lst)))
    clen = [m['clen'] for m in man]
    summ = dict(filter=dict(stats), selected=len(man), tasks=len({m['task'] for m in man}), hard=sum((m['hard'] for m in man)), easy=sum((not m['hard'] for m in man)), completion_tok=sum(clen), prompt_tok=sum((m['plen'] for m in man)), weighted_tok_per_epoch=sum((m['w'] * (m['clen'] + m['plen']) for m in man)), clen_mean=round(st.mean(clen)), clen_median=round(st.median(clen)), clen_max=max(clen), clen_mean_hard=round(st.mean((m['clen'] for m in man if m['hard']))), clen_mean_easy=round(st.mean((m['clen'] for m in man if not m['hard']))), shortest_over_mean_correct=round(st.mean(saving), 3), n_multi=len(saving), maxlen=a.maxlen, hard_w=a.hard_w)
    json.dump(dict(summary=summ, seqs=man), open(out / 'manifest.json', 'w'), indent=0)
    print(json.dumps(summ, indent=1))
if __name__ == '__main__':
    main()
