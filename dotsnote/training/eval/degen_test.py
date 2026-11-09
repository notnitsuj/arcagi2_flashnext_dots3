import asyncio, json, os, re, sys
import httpx
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'harness'))
from direct import make_prompt, DATA
from transformers import AutoTokenizer
port = int(sys.argv[1])
N = int(sys.argv[2]) if len(sys.argv) > 2 else 16
G = int(sys.argv[3]) if len(sys.argv) > 3 else 12000
tok = AutoTokenizer.from_pretrained(os.environ.get('D3_BASE_MODEL', 'models/dots3-note-prev-fp8'))
ids = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'holdout40.txt')).read().split()[:N]
prompts = [tok.apply_chat_template([{'role': 'user', 'content': make_prompt(json.load(open(DATA / 'evaluation' / f'{t}.json')), 0)}], tokenize=False, add_generation_prompt=True) for t in ids]

def degenerate(txt):
    tail = txt[-3000:]
    return len(re.sub('[\\d,\\s\\[\\]]', '', tail)) < 30

def runaway(txt):
    tail = txt[-3000:]
    return len(tail) == 3000 and (not re.search('[^\\d, ]', tail))

async def main():
    async with httpx.AsyncClient(timeout=7200) as c:
        rs = await asyncio.gather(*[c.post(f'http://127.0.0.1:{port}/v1/completions', json={'model': 'm', 'prompt': p, 'max_tokens': G, 'temperature': 0.7, 'top_p': 1.0, 'seed': i}) for i, p in enumerate(prompts)])
    txt = [r.json()['choices'][0]['text'] for r in rs]
    d = [degenerate(t) for t in txt]
    r = [runaway(t) for t in txt]
    print(f'degenerate {sum(d)}/{len(d)} runaway {sum(r)}/{len(r)}', ''.join(('R' if y else 'X' if x else '.' for x, y in zip(d, r))), flush=True)
    if len(sys.argv) > 4:
        json.dump(txt, open(sys.argv[4], 'w'))
asyncio.run(main())
