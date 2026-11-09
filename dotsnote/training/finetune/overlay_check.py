import json, os, sys
from pathlib import Path
TASKS = ['0520fde7', '05a7bcf2', '0a2355a6']

def prompts():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'harness'))
    from direct import make_prompt, DATA
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.environ.get('D3_LB144R', os.path.join(os.environ.get('D3_WORK', 'work'), 'lb144r')))
    out = []
    for t in TASKS:
        task = json.load(open(DATA / 'training' / f'{t}.json'))
        out.append((t, tok.apply_chat_template([{'role': 'user', 'content': make_prompt(task, 0)}], tokenize=False, add_generation_prompt=True)))
    return out

def dump(port, out, pfile=None):
    import httpx
    host = '127.0.0.1'
    if ':' in str(port):
        host, port = str(port).rsplit(':', 1)
    res = []
    for t, p in json.load(open(pfile)) if pfile else prompts():
        r = httpx.post(f'http://{host}:{port}/v1/completions', timeout=600, json={'model': 'm', 'prompt': p, 'max_tokens': 64, 'temperature': 0.0, 'logprobs': 5, 'prompt_logprobs': 5, 'skip_special_tokens': False}).json()
        c = r['choices'][0]
        res.append({'task': t, 'text': c['text'], 'prompt_logprobs': c.get('prompt_logprobs'), 'logprobs': c.get('logprobs')})
    json.dump(res, open(out, 'w'))
    print('dumped', out, [len(x['prompt_logprobs'] or []) for x in res])

def cmp(a, b):
    A, B = (json.load(open(a)), json.load(open(b)))
    npos = ndiff = 0
    maxd = 0.0
    for x, y in zip(A, B):
        for u, v in zip(x['prompt_logprobs'][1:], y['prompt_logprobs'][1:]):
            npos += 1
            if u != v:
                ndiff += 1
                common = set(u) & set(v)
                maxd = max([maxd] + [abs(u[k]['logprob'] - v[k]['logprob']) for k in common])
    same_text = sum((x['text'] == y['text'] for x, y in zip(A, B)))
    same_lp = sum((x['logprobs'] == y['logprobs'] for x, y in zip(A, B)))
    print(f'{a} vs {b}: prompt positions {npos}, differing {ndiff} (max |dlogprob| {maxd:.3g}); greedy text equal {same_text}/{len(A)}, completion logprobs equal {same_lp}/{len(A)}')
if __name__ == '__main__':
    {'dump': lambda: dump(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None), 'prompts': lambda: json.dump(prompts(), open(sys.argv[2], 'w')), 'cmp': lambda: cmp(sys.argv[2], sys.argv[3])}[sys.argv[1]]()
