"""Serving-fidelity A/B on neutral content: top-20 prompt logprobs over fixed token sequences, in dots3/fidelity.py's npz
format (compare two dumps with `python dots3/fidelity.py cmp A.npz B.npz`).

  python logprob_dump.py --port 8000 --tokenizer CKPT --gen IDS.json --out A.npz    # generate sequences, then dump
  python logprob_dump.py --port 8000 --tokenizer CKPT --ids IDS.json --out B.npz    # dump the same sequences
Generation: N neutral prompts (arithmetic, logic, code, simulation), --len new tokens each at T=0.7, concurrently.
"""
import argparse, asyncio, json, time

import httpx
import numpy as np
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--tokenizer", required=True)
ap.add_argument("--gen", default="")
ap.add_argument("--ids", default="")
ap.add_argument("--len", type=int, default=12000)
ap.add_argument("--out", required=True)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.tokenizer)
K = 20
PROMPTS = [
    "A warehouse ships boxes of 12, 18 and 30 items. Find every way to ship exactly 1000 items using at most 60 boxes, "
    "and explain your reasoning step by step, checking each candidate.",
    "Five people sit in a row. Ann is not next to Ben, Cara is left of Dan, Eve is at an end, and Ben is in the middle. "
    "List all valid seatings, reasoning carefully about each constraint.",
    "Write a Python function that returns the longest palindromic subsequence of a string, prove its correctness, "
    "analyse its complexity, and test it by hand on five examples.",
    "Simulate Conway's Game of Life on this 8x8 toroidal board for 10 generations, printing each board and the number "
    "of live cells:\n........\n...#....\n....#...\n..###...\n........\n........\n.....##.\n.....##.",
    "A 7x7 grid has some cells coloured. Describe a general method to detect whether one shape is a rotated or "
    "reflected copy of another, then apply it to several small examples you construct yourself.",
    "Solve the equation x^4 - 10x^2 + 9 = 0, then generalise to x^4 - (a^2+b^2) x^2 + a^2 b^2 = 0 and verify each "
    "root by substitution.",
]


def chat(p):
    return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)


async def gen():
    async with httpx.AsyncClient(timeout=7200) as c:
        async def one(i, p):
            pr = chat(p)
            r = (await c.post(f"http://127.0.0.1:{a.port}/v1/completions",
                              json={"model": "m", "prompt": pr, "max_tokens": a.len, "temperature": 0.7, "seed": i,
                                    "ignore_eos": True})).json()
            plen = len(tok(pr, add_special_tokens=False).input_ids)
            ids = tok(pr + r["choices"][0]["text"], add_special_tokens=False).input_ids
            return {"name": f"p{i}", "ids": ids, "plen": plen}
        return await asyncio.gather(*[one(i, p) for i, p in enumerate(PROMPTS)])


def dump(seqs):
    res = {}
    for s in seqs:
        t0, ids, name = time.time(), s["ids"], s["name"]
        r = httpx.post(f"http://127.0.0.1:{a.port}/v1/completions",
                       json={"model": "m", "prompt": ids, "max_tokens": 1, "prompt_logprobs": K, "temperature": 0},
                       timeout=7200).json()
        pl = r["choices"][0]["prompt_logprobs"]
        L = len(ids)
        top_ids = np.full((L, K), -1, np.int32); top_lp = np.full((L, K), -1e9, np.float32); act = np.zeros(L, np.float32)
        for i, d in enumerate(pl):
            if d is None:
                continue
            items = sorted(((int(k), v["logprob"], v.get("rank", 99)) for k, v in d.items()), key=lambda x: x[2])
            tops = [x for x in items if x[2] <= K][:K]
            top_ids[i, :len(tops)] = [x[0] for x in tops]; top_lp[i, :len(tops)] = [x[1] for x in tops]
            act[i] = d[str(ids[i])]["logprob"] if str(ids[i]) in d else tops[-1][1]
        res[f"{name}_ids"] = np.asarray(ids, np.int32)
        res[f"{name}_top_ids"], res[f"{name}_top_lp"], res[f"{name}_act"] = top_ids, top_lp, act
        res[f"{name}_plen"] = np.asarray(s["plen"])
        print(f"[logprob_dump] {name}: {L} tok in {time.time() - t0:.0f}s, NLL {-act[s['plen']:].mean():.4f}", flush=True)
    np.savez_compressed(a.out, **res)


if a.gen:
    seqs = asyncio.run(gen())
    json.dump(seqs, open(a.gen, "w"))
else:
    seqs = json.load(open(a.ids))
dump(seqs)
print("[logprob_dump] done", flush=True)
