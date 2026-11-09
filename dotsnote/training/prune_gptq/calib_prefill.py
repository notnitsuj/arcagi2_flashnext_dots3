import json, os, sys, argparse, httpx
from transformers import AutoTokenizer
ap = argparse.ArgumentParser()
ap.add_argument('port', type=int)
ap.add_argument('--n', type=int, default=16)
ap.add_argument('--len', type=int, default=16384)
ap.add_argument('--src', default=os.path.join(os.environ.get('D3_WORK', 'work'), 'calib', 'calib_mix.jsonl'))
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(os.environ.get('D3_BASE_MODEL', 'models/dots3-note-prev-fp8'))
rs = [json.loads(l) for l in open(a.src)]
rs = sorted(rs, key=lambda r: -len(r['full']['prompt'] + r['full']['completion']))[:a.n]
tot = 0
for r in rs:
    ids = tok(r['full']['prompt'] + r['full']['completion'], add_special_tokens=False).input_ids[:a.len]
    httpx.post(f'http://127.0.0.1:{a.port}/v1/completions', json={'model': 'm', 'prompt': ids, 'max_tokens': 1, 'temperature': 0}, timeout=7200).raise_for_status()
    tot += len(ids)
    print(tot, flush=True)
