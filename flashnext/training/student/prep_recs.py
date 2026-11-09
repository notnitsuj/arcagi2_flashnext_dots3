import argparse, gzip, json, os, sys
from pathlib import Path
EVAL_DIR = os.path.join(os.environ.get('ARC_DATA', 'data/ARC-AGI-2/data'), 'evaluation')
TOK = os.environ.get('FN_BASE_MODEL', 'models/Qwen3.8-Flash-Next-FP8')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('srcs', nargs='+')
    ap.add_argument('--val-tasks', default=os.path.join(os.environ.get('FN_WORK', 'work'), 'cap', 'ho_tasks.txt') + ',' + os.path.join(os.environ.get('FN_WORK', 'work'), 'cap', 'ho_tasks_c.txt'))
    ap.add_argument('--max-len', type=int, default=65536)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK)
    ev = {p.stem for p in Path(EVAL_DIR).glob('*.json')}
    assert len(ev) == 120, len(ev)
    val = set()
    for f in a.val_tasks.split(','):
        if f and Path(f).exists():
            val |= set(open(f).read().split())
    im_end = tok.convert_tokens_to_ids('<|im_end|>')
    im_start = tok.convert_tokens_to_ids('<|im_start|>')
    asst = tok('assistant\n', add_special_tokens=False).input_ids
    think = tok('<think>\n', add_special_tokens=False).input_ids
    print('special', im_end, im_start, asst, think, file=sys.stderr)
    n = ntok = ngen = 0
    with open(a.out, 'w') as fo:
        for src in a.srcs:
            tag = Path(src).parent.name
            for li, line in enumerate(gzip.open(src, 'rt')):
                r = json.loads(line)
                assert r['task'] not in ev, 'evaluation-split task in KD data'
                p_ids = tok(r['prompt'], add_special_tokens=False).input_ids
                ids = tok(r['prompt'] + r['completion'], add_special_tokens=False).input_ids
                plen = len(p_ids)
                if ids[:plen] != p_ids:
                    plen = next((i for i, (x, y) in enumerate(zip(ids, p_ids)) if x != y))
                ids = ids[:a.max_len]
                T = len(ids)
                if T <= plen + 16:
                    continue
                gen, i, s = ([], plen, plen)
                while i < T:
                    if ids[i] == im_end:
                        gen.append((s, i + 1))
                        j = i + 1
                        while j < T:
                            if ids[j] == im_start and ids[j + 1:j + 1 + len(asst)] == asst:
                                j = j + 1 + len(asst)
                                if ids[j:j + len(think)] == think:
                                    j += len(think)
                                break
                            j += 1
                        i = s = j
                        continue
                    i += 1
                if s < T:
                    gen.append((s, T))
                gen = [(x, y) for x, y in gen if y > x]
                key = f"{tag}{a.tag}_{r['mode']}_{r['task']}_{r['test']}_{r['attempt']}"
                rec = {'key': key, 'task': r['task'], 'mode': r['mode'], 'split': 'val' if r['task'] in val else 'train', 'plen': plen, 'gen': gen, 'ids': ids}
                fo.write(json.dumps(rec) + '\n')
                n += 1
                ntok += T
                ngen += sum((y - x for x, y in gen))
    print(f'{n} records, {ntok / 1000000.0:.2f}M tokens, {ngen / 1000000.0:.2f}M generated', file=sys.stderr)
if __name__ == '__main__':
    main()
