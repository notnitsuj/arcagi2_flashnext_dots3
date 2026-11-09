import argparse, asyncio, json, os, random, re, sys, tempfile, time
from pathlib import Path
import httpx
import numpy as np
DATA = Path(os.environ.get('ARC_DATA', 'data/ARC-AGI-2/data'))
PROMPT = 'You are an expert at solving ARC (Abstraction and Reasoning Corpus) puzzles by writing Python code.\nBelow are input/output example pairs of integer grids (colors 0-9), and one or more test inputs.\nFind the single transformation rule that maps every example input to its output, and implement it as\n\n```python\nimport numpy as np\ndef transform(grid: np.ndarray) -> np.ndarray:\n    ...\n```\n\nThe function must reproduce every example output exactly and generalize to the test inputs.\nOnly numpy and the Python standard library are available. Reply with a short explanation of the rule\nfollowed by the complete code in a single ```python block. Do not include example usage or a __main__ block.\n\n$PROBLEM'
FEEDBACK = '\n\nHere are previous attempts at this puzzle with their results on the examples (best last). They are not fully\ncorrect. Study where they fail, rethink the rule if needed, and write an improved, fully correct program.\n$ATTEMPTS'

def g2s(g):
    return '\n'.join((' '.join((str(int(v)) for v in row)) for row in g))

def format_problem(task, rng=None):
    train = list(task['train'])
    if rng is not None:
        rng.shuffle(train)
    parts = []
    for i, p in enumerate(train, 1):
        ri, ci = (len(p['input']), len(p['input'][0]))
        ro, co = (len(p['output']), len(p['output'][0]))
        parts.append(f"## Example {i}\nInput ({ri}x{ci}):\n{g2s(p['input'])}\nOutput ({ro}x{co}):\n{g2s(p['output'])}")
    for i, p in enumerate(task['test'], 1):
        ri, ci = (len(p['input']), len(p['input'][0]))
        parts.append(f"## Test input {i} ({ri}x{ci}):\n{g2s(p['input'])}")
    return '\n\n'.join(parts)
RUNNER = '\nimport json, sys, signal, numpy as np\nsignal.alarm(int(sys.argv[2]))\nsrc = open(sys.argv[1]).read()\ninputs = json.load(sys.stdin)\nns = {}\nout = []\ntry:\n    exec(src, ns)\n    f = ns["transform"]\nexcept Exception as e:\n    print(json.dumps({"fatal": f"{type(e).__name__}: {e}"[:500]})); sys.exit(0)\nfor g in inputs:\n    try:\n        r = np.asarray(f(np.array(g, dtype=int)))\n        if r.ndim != 2 or r.size == 0 or r.shape[0] > 30 or r.shape[1] > 30:\n            out.append({"err": f"bad output shape {r.shape}"})\n        else:\n            out.append({"grid": r.astype(int).tolist()})\n    except Exception as e:\n        out.append({"err": f"{type(e).__name__}: {e}"[:300]})\nprint(json.dumps({"results": out}))\n'
RUNNER_PATH = Path(tempfile.gettempdir()) / f'arc_runner_{os.getpid()}.py'
RUNNER_PATH.write_text(RUNNER)
SANDBOX_SEM = None

async def run_code(code, grids, timeout=10):
    async with SANDBOX_SEM:
        with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
            f.write(code)
            path = f.name
        try:
            proc = await asyncio.create_subprocess_exec(sys.executable, str(RUNNER_PATH), path, str(timeout), stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            try:
                out, err = await asyncio.wait_for(proc.communicate(json.dumps(grids).encode()), timeout + 5)
            except asyncio.TimeoutError:
                proc.kill()
                return [{'err': 'timeout'}] * len(grids)
            try:
                res = json.loads(out.decode().strip().splitlines()[-1])
            except Exception:
                return [{'err': 'crash: ' + err.decode()[-300:] or 'timeout'}] * len(grids)
            if 'fatal' in res:
                return [{'err': res['fatal']}] * len(grids)
            return res['results']
        finally:
            os.unlink(path)

def parse_code(text):
    blocks = re.findall('```python\\s*\\n(.*?)```', text, re.S)
    blocks = [b for b in blocks if 'def transform' in b]
    return blocks[-1] if blocks else None

def diff_feedback(task, results):
    lines, score = ([], 0.0)
    for i, (p, r) in enumerate(zip(task['train'], results), 1):
        exp = np.array(p['output'])
        if 'err' in r:
            lines.append(f"Example {i}: ERROR {r['err']}")
            continue
        got = np.array(r['grid'])
        if got.shape != exp.shape:
            lines.append(f'Example {i}: wrong output shape {got.shape}, expected {exp.shape}. Output was:\n{g2s(got)}')
            continue
        acc = float((got == exp).mean())
        score += acc
        if acc == 1:
            lines.append(f'Example {i}: correct')
        else:
            d = '\n'.join((' '.join((str(a) if a == b else f'{a}/{b}' for a, b in zip(ra, rb))) for ra, rb in zip(got.tolist(), exp.tolist())))
            lines.append(f'Example {i}: {acc:.0%} cells correct. Cells shown as got/expected where wrong:\n{d}')
    return ('\n'.join(lines), score / len(task['train']))

class Backend:

    def __init__(self, kind, ports, effort, think_budget, answer_budget, temperature, tokenizer=None):
        self.kind, self.effort = (kind, effort)
        self.think_budget, self.answer_budget, self.temperature = (think_budget, answer_budget, temperature)
        self.urls = [f'http://127.0.0.1:{p}/v1' for p in ports]
        self.rr = 0
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(7200.0))
        if kind == 'qwen':
            from transformers import AutoTokenizer
            root = tokenizer or httpx.get(self.urls[0] + '/models').json()['data'][0]['root']
            self.tok = AutoTokenizer.from_pretrained(root)
            vocab = self.tok.get_vocab()
            self.eos_ids = [vocab[t] for t in ['<|im_end|>', '<|endoftext|>', '<|endofassistant|>'] if t in vocab]

    def url(self):
        self.rr += 1
        return self.urls[self.rr % len(self.urls)]

    async def complete(self, prompt):
        if self.kind == 'oss':
            body = {'model': 'm', 'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': self.think_budget + self.answer_budget, 'temperature': self.temperature, 'reasoning_effort': self.effort}
            r = (await self.client.post(self.url() + '/chat/completions', json=body)).json()
            u = r['usage']
            msg = r['choices'][0]['message']
            self.last_full = (msg.get('reasoning_content') or '') + '\n' + (msg['content'] or '')
            return (msg['content'] or '', u['prompt_tokens'], u['completion_tokens'])
        kw = {'reasoning_effort': self.effort} if self.effort else {}
        text = self.tok.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True, **kw)
        url = self.url()
        body = {'model': 'm', 'prompt': text, 'max_tokens': self.think_budget, 'temperature': self.temperature, 'top_p': 0.95, 'stop': ['</think>'], 'logit_bias': {str(i): -100 for i in self.eos_ids}}
        r = (await self.client.post(url + '/completions', json=body)).json()
        think = r['choices'][0]['text']
        pt, ct = (r['usage']['prompt_tokens'], r['usage']['completion_tokens'])
        if r['choices'][0]['finish_reason'] == 'length':
            think += '\n\nI am out of thinking budget, so I will now write my best program.\n'
        text2 = text + think + '</think>\n\n'
        body = {'model': 'm', 'prompt': text2, 'max_tokens': self.answer_budget, 'temperature': self.temperature, 'top_p': 0.95}
        r = (await self.client.post(url + '/completions', json=body)).json()
        ans = r['choices'][0]['text']
        self.last_full = {'prompt': text, 'completion': think + '</think>\n\n' + ans}
        return (ans, pt + r['usage']['prompt_tokens'], ct + r['usage']['completion_tokens'])

async def expert(task_id, task, eid, args, be, log):
    rng = random.Random(hash((task_id, eid)) & 4294967295)
    tests = [p['input'] for p in task['test']]
    sols = []
    for it in range(args.iters):
        prompt = PROMPT.replace('$PROBLEM', format_problem(task, rng if args.shuffle else None))
        if sols and args.feedback:
            best = sorted(sols, key=lambda s: s['score'])[-args.max_fb:]
            att = '\n'.join((f"<attempt>\n```python\n{s['code']}\n```\nResults:\n{s['fb']}\nScore: {s['score']:.2f}\n</attempt>" for s in best))
            prompt += FEEDBACK.replace('$ATTEMPTS', att)
        t0 = time.time()
        try:
            text, pt, ct = await be.complete(prompt)
            full = be.last_full
        except Exception as e:
            print('LLM error', task_id, repr(e)[:200], flush=True)
            await asyncio.sleep(5)
            continue
        code = parse_code(text)
        rec = {'task': task_id, 'expert': eid, 'iter': it, 'pt': pt, 'ct': ct, 't0': t0, 't1': time.time(), 'has_code': code is not None, 'train_pass': False, 'score': 0.0, 'test': None}
        if code:
            res = await run_code(code, [p['input'] for p in task['train']] + tests)
            tr, te = (res[:len(task['train'])], res[len(task['train']):])
            fb, score = diff_feedback(task, tr)
            rec['train_pass'] = all(('grid' in r and r['grid'] == p['output'] for r, p in zip(tr, task['train'])))
            rec['score'] = score
            rec['test'] = [r.get('grid') for r in te]
            rec['code'] = code
            sols.append({'code': code, 'fb': fb, 'score': score})
        if args.save_text:
            rec['full'] = full
        log.write(json.dumps(rec) + '\n')
        log.flush()
        if rec['train_pass']:
            return

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kind', required=True, choices=['qwen', 'oss'])
    ap.add_argument('--ports', required=True)
    ap.add_argument('--effort', default=None)
    ap.add_argument('--think', type=int, default=24000)
    ap.add_argument('--answer', type=int, default=6000)
    ap.add_argument('--temp', type=float, default=1.0)
    ap.add_argument('--experts', type=int, default=4)
    ap.add_argument('--iters', type=int, default=4)
    ap.add_argument('--max_fb', type=int, default=3)
    ap.add_argument('--no_feedback', dest='feedback', action='store_false')
    ap.add_argument('--shuffle', action='store_true', default=True)
    ap.add_argument('--split', default='evaluation')
    ap.add_argument('--tasks', default=None, help='comma list or @file')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--concurrency', type=int, default=96)
    ap.add_argument('--out', required=True)
    ap.add_argument('--save_text', action='store_true')
    ap.add_argument('--tokenizer', default=None)
    args = ap.parse_args()
    global SANDBOX_SEM
    SANDBOX_SEM = asyncio.Semaphore(48)
    ids = sorted((p.stem for p in (DATA / args.split).glob('*.json')))
    if args.tasks:
        sel = open(args.tasks[1:]).read().split() if args.tasks.startswith('@') else args.tasks.split(',')
        ids = [i for i in ids if i in set(sel)]
    if args.limit:
        ids = ids[:args.limit]
    tasks = {i: json.load(open(DATA / args.split / f'{i}.json')) for i in ids}
    be = Backend(args.kind, [int(p) for p in args.ports.split(',')], args.effort, args.think, args.answer, args.temp, args.tokenizer)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        for l in open(args.out):
            r = json.loads(l)
            if r['train_pass'] or r['iter'] == args.iters - 1:
                done.add((r['task'], r['expert']))
    log = open(args.out, 'a')
    json.dump(vars(args), open(args.out + '.args.json', 'w'))
    sem = asyncio.Semaphore(args.concurrency)

    async def run(tid, e):
        async with sem:
            await expert(tid, tasks[tid], e, args, be, log)
    jobs = [run(t, e) for e in range(args.experts) for t in ids if (t, e) not in done]
    t0 = time.time()
    print(f'{len(jobs)} expert chains over {len(ids)} tasks', flush=True)
    await asyncio.gather(*jobs)
    print(f'done in {time.time() - t0:.0f}s', flush=True)
if __name__ == '__main__':
    asyncio.run(main())
