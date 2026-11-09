# Prompt template and answer parsers (TEMPLATE, backscan_json_parser, extract_from_boxed) from
# github.com/arcprize/arc-agi-benchmarking (MIT License, Copyright (c) 2025 ARC Prize Foundation).
import argparse, asyncio, json, os, re, sys, time
from pathlib import Path
import httpx
DATA = Path(os.environ.get('ARC_DATA', 'data/ARC-AGI-2/data'))
TEMPLATE = 'You are participating in a puzzle solving competition. You are an expert at solving puzzles.\n\nBelow is a list of input and output pairs with a pattern. Your goal is to identify the pattern or transformation in the training examples that maps the input to the output, then apply that pattern to the test input to give a final output.\n\nRespond in the format of the training output examples\n\n--Training Examples--\n{training_examples}\n--End of Training Examples--\n\n--Test Input--\n{test_input}\n--End of Test Input--\n\nYour response:'

def make_prompt(task, test_idx):
    ex = ''
    for i, p in enumerate(task['train']):
        ex += f'--Example {i}-- \n\n INPUT: \n\n' + json.dumps(p['input']) + '\n\n'
        ex += 'OUTPUT: \n\n' + json.dumps(p['output']) + '\n\n'
    return TEMPLATE.format(training_examples=ex, test_input=json.dumps(task['test'][test_idx]['input']))

def backscan_json_parser(log_str):
    last_bracket_idx = -1
    closing_bracket = None
    for i in range(len(log_str) - 1, -1, -1):
        char = log_str[i]
        if char in (']', '}'):
            last_bracket_idx = i
            closing_bracket = char
            break
    if last_bracket_idx == -1:
        return None
    opening_bracket = '[' if closing_bracket == ']' else '{'
    bracket_counter = 1
    start_idx = -1
    for i in range(last_bracket_idx - 1, -1, -1):
        char = log_str[i]
        if char == closing_bracket:
            bracket_counter += 1
        elif char == opening_bracket:
            bracket_counter -= 1
            if bracket_counter == 0:
                start_idx = i
                break
    if start_idx == -1:
        return None
    json_candidate = log_str[start_idx:last_bracket_idx + 1]
    try:
        parsed_json = json.loads(json_candidate)
        if isinstance(parsed_json, list) and parsed_json and all((isinstance(row, list) for row in parsed_json)):
            return parsed_json
        else:
            return None
    except json.JSONDecodeError:
        return None

def extract_from_boxed(log_str):
    match = re.search('\\\\boxed\\{(.*?)\\}', log_str, re.DOTALL)
    if match:
        content = match.group(1).strip()
        try:
            parsed_json = json.loads(content)
            if isinstance(parsed_json, list) and all((isinstance(i, list) for i in parsed_json)):
                return parsed_json
        except json.JSONDecodeError:
            pass
    return None

def parse(answer):
    for f in (extract_from_boxed, backscan_json_parser):
        try:
            r = f(answer)
        except Exception:
            r = None
        if r is not None:
            return r
    return None

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ports', required=True)
    ap.add_argument('--tokenizer', default=os.environ.get('D3_BASE_MODEL', 'models/dots3-note-prev-fp8'))
    ap.add_argument('--split', default='evaluation')
    ap.add_argument('--tasks', default=None, help='comma list or @file')
    ap.add_argument('--attempts', type=int, default=2)
    ap.add_argument('--max_tokens', type=int, default=384000)
    ap.add_argument('--max_model_len', type=int, default=400000)
    ap.add_argument('--temp', type=float, default=0.7)
    ap.add_argument('--top_p', type=float, default=1.0)
    ap.add_argument('--concurrency', type=int, default=64)
    ap.add_argument('--out', required=True)
    ap.add_argument('--save_text', default=None, help='jsonl path for full completions')
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    ids = sorted((p.stem for p in (DATA / a.split).glob('*.json')))
    if a.tasks:
        sel = open(a.tasks[1:]).read().split() if a.tasks.startswith('@') else a.tasks.split(',')
        ids = [i for i in ids if i in set(sel)]
    tasks = {i: json.load(open(DATA / a.split / f'{i}.json')) for i in ids}
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            r = json.loads(l)
            done.add((r['task'], r['test'], r['attempt']))
    log = open(a.out, 'a')
    txt = open(a.save_text, 'a') if a.save_text else None
    urls = [f'http://127.0.0.1:{p}/v1' for p in a.ports.split(',')]
    sem = asyncio.Semaphore(a.concurrency)
    client = httpx.AsyncClient(timeout=httpx.Timeout(6 * 3600.0))
    n = [0]

    async def one(tid, ti, att):
        async with sem:
            prompt = tok.apply_chat_template([{'role': 'user', 'content': make_prompt(tasks[tid], ti)}], tokenize=False, add_generation_prompt=True)
            n_prompt = len(tok(prompt, add_special_tokens=False).input_ids)
            body = {'model': 'm', 'prompt': prompt, 'temperature': a.temp, 'top_p': a.top_p, 'max_tokens': min(a.max_tokens, a.max_model_len - n_prompt - 16), 'skip_special_tokens': False}
            url = urls[n[0] % len(urls)]
            n[0] += 1
            t0 = time.time()
            for retry in range(3):
                try:
                    r = (await client.post(url + '/completions', json=body)).json()
                    c = r['choices'][0]
                    break
                except Exception as e:
                    print('error', tid, ti, att, repr(e)[:200], flush=True)
                    await asyncio.sleep(30)
            else:
                return
            text = c['text']
            think, _, answer = text.rpartition('</think>') if '</think>' in text else ('', '', text)
            grid = parse(answer) if '</think>' in text else None
            gt = tasks[tid]['test'][ti].get('output')
            rec = {'task': tid, 'test': ti, 'attempt': att, 'pt': r['usage']['prompt_tokens'], 'ct': r['usage']['completion_tokens'], 'finish': c['finish_reason'], 'closed_think': '</think>' in text, 't0': t0, 't1': time.time(), 'parsed': grid is not None, 'correct': grid == gt if gt is not None else None, 'pred': grid}
            log.write(json.dumps(rec) + '\n')
            log.flush()
            if txt:
                txt.write(json.dumps({'task': tid, 'test': ti, 'attempt': att, 'prompt': prompt, 'completion': text}) + '\n')
                txt.flush()
    jobs = [one(t, ti, at) for at in range(a.attempts) for t in ids for ti in range(len(tasks[t]['test'])) if (t, ti, at) not in done]
    print(f'{len(jobs)} requests', flush=True)
    t0 = time.time()
    await asyncio.gather(*jobs)
    print(f'done in {time.time() - t0:.0f}s', flush=True)
if __name__ == '__main__':
    asyncio.run(main())
