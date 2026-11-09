"""Decode throughput vs concurrency on the live server, in one ramp: start LEVELS[0] long generations (real ARC prompts,
ignore_eos), measure steady decode tok/s from /metrics while exactly that many run with no prefill, then add requests up
to the next level, and so on. One prefill per request, so a 40->56->64 ramp costs ~15 min on 4x L4.

  python load_sweep.py --port 8000 --challenges C.json --template T.txt --tokenizer CKPT --levels 40,56,64 \
      --measure-s 180 --out load_sweep.json
"""
import argparse, asyncio, json, re, statistics, time

import httpx
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--challenges", default="")
ap.add_argument("--template", default="")
ap.add_argument("--synthetic", type=int, default=0, help="N > 0: synthetic prompts of ~N tokens instead of task files")
ap.add_argument("--ctx-file", default="", help="JSON {ctx: [...]}: per-request decode context lengths (tokens) to emulate; "
                "requests are prefilled with a cellular-automaton log of that length and keep simulating")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tokenizer", required=True)
ap.add_argument("--levels", default="40,56,64")
ap.add_argument("--measure-s", type=float, default=180)
ap.add_argument("--max-tokens", type=int, default=30000)
ap.add_argument("--timeout-s", type=float, default=2400, help="give up on the whole sweep after this long")
ap.add_argument("--out", required=True)
a = ap.parse_args()
LEVELS = [int(x) for x in a.levels.split(",")]
T0 = time.time()
tok = AutoTokenizer.from_pretrained(a.tokenizer)
TEMPLATE = open(a.template).read() if a.template else ""
chal = json.load(open(a.challenges)) if a.challenges else {}


def make_prompt(task, ti):
    ex = ""
    for i, p in enumerate(task["train"]):
        ex += f"--Example {i}-- \n\n INPUT: \n\n" + json.dumps(p["input"]) + "\n\n"
        ex += "OUTPUT: \n\n" + json.dumps(p["output"]) + "\n\n"
    body = TEMPLATE.format(training_examples=ex, test_input=json.dumps(task["test"][ti]["input"]))
    return tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False, add_generation_prompt=True)


def synthetic_prompt(i, n):
    """long-generation request of ~n prompt tokens, distinct per request: a list of arithmetic word problems"""
    import random
    rnd = random.Random(i)
    qs = []
    while len(qs) * 30 < n:
        a, b, c = rnd.randrange(10, 999), rnd.randrange(2, 99), rnd.randrange(1, 50)
        qs.append(f"{len(qs) + 1}. A store has {a} items, sells them in boxes of {b}, and {c} boxes are returned. "
                  f"How many items remain unsold, and what fraction of boxes were returned?")
    body = "Solve every problem below, showing all intermediate steps, then summarize the answers in a table.\n\n" + \
        "\n".join(qs)
    return tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False, add_generation_prompt=True)


def life_prompt(i, n):
    """~n-token request mid-way through a long, repetitive reasoning trace: a Game of Life simulation log (random
    board per request, real successive generations) followed by the instruction to keep simulating"""
    import random
    rnd = random.Random(1000 * a.seed + i)
    H, W = rnd.randrange(14, 26), rnd.randrange(14, 26)
    g = [[int(rnd.random() < 0.35) for _ in range(W)] for _ in range(H)]

    def step(g):
        return [[int((n == 3) or (g[y][x] and n == 2)) for x in range(W)
                 for n in [sum(g[(y + dy) % H][(x + dx) % W] for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)]]
                for y in range(H)]
    head = (f"Simulate Conway's Game of Life on a {H}x{W} toroidal board ('#' alive, '.' dead). For every generation, "
            "print the full board, then one line stating how many cells are alive and what changed. Continue until "
            "told to stop.\n\n")
    parts, gen = [], 0
    board = lambda g: "\n".join("".join("#" if c else "." for c in row) for row in g)
    have, cnt = 0, 0   # tokens measured so far: later (sparser) boards tokenize smaller, so measure as we go
    while have < n:
        alive = sum(map(sum, g))
        parts.append(f"Generation {gen}:\n{board(g)}\nAlive: {alive}.")
        cnt += 1
        if cnt == 1 or cnt % 32 == 0:
            have = len(tok("\n\n".join(parts))["input_ids"])
        g = step(g); gen += 1
        if sum(map(sum, g)) < 6:   # extinct: reseed so the log stays non-trivial
            g = [[int(rnd.random() < 0.35) for _ in range(W)] for _ in range(H)]
    msgs = [{"role": "user", "content": head + "Begin with this board.\n\n" + parts[0]},
            {"role": "assistant", "content": "\n\n".join(parts)},
            {"role": "user", "content": "Continue the simulation from the last generation, same format, as long as you can."}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


if a.ctx_file:
    import random as _r
    CTX = json.load(open(a.ctx_file))["ctx"]
    rr = _r.Random(a.seed)
    prompts = [life_prompt(i, rr.choice(CTX)) for i in range(max(LEVELS))]
    print("[load_sweep] context lengths (tokens):", sorted(len(tok(p)["input_ids"]) for p in prompts), flush=True)
else:
    prompts = ([synthetic_prompt(i, a.synthetic) for i in range(max(LEVELS))] if a.synthetic
               else [make_prompt(t, 0) for t in chal.values()])
prompts = (prompts * (1 + max(LEVELS) // max(1, len(prompts))))[:max(LEVELS)]


def log(*x):
    print(f"[load_sweep {time.time() - T0:6.0f}s]", *x, flush=True)


async def metrics(c):
    m = (await c.get(f"http://127.0.0.1:{a.port}/metrics", timeout=10)).text
    f = lambda name: sum(float(x) for x in re.findall(rf"^vllm:{name}(?:\{{[^}}]*\}})? ([0-9.e+-]+)", m, re.M))
    return dict(t=time.time(), run=f("num_requests_running"), wait=f("num_requests_waiting"),
                gen=f("generation_tokens_total"), prompt=f("prompt_tokens_total"), kv=f("kv_cache_usage_perc"),
                acc=f("spec_decode_num_accepted_tokens_total"), drafts=f("spec_decode_num_drafts_total"),
                preempt=f("num_preemptions_total"))


async def main():
    res = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(a.timeout_s), limits=httpx.Limits(max_connections=max(LEVELS) + 8)) as c:
        body = lambda p: dict(model="m", prompt=p, max_tokens=a.max_tokens, temperature=0.7, ignore_eos=True)
        first = {}   # request index -> time of its first streamed token (= its prefill is done)

        async def run_req(i, p):
            async with c.stream("POST", f"http://127.0.0.1:{a.port}/v1/completions", json=dict(body(p), stream=True)) as r:
                async for line in r.aiter_lines():
                    if line.startswith("data:") and i not in first:
                        first[i] = time.time()

        tasks, started = [], 0
        for N in LEVELS:
            tasks += [asyncio.create_task(run_req(i, prompts[i])) for i in range(started, N)]
            started = N
            log(f"level {N}: {len(tasks)} requests in flight, waiting for prefill")
            t_lv = time.time()
            prev, rates, kvs, t_level, accs = None, [], [], time.time(), []
            while time.time() - t_level < a.measure_s + 900 and time.time() - T0 < a.timeout_s:
                await asyncio.sleep(10)
                try:
                    m = await metrics(c)
                except Exception:
                    continue
                # pure decode: every request of this level has streamed a token (vLLM counts requests still in chunked
                # prefill as running, and its prompt-token counter is not updated in this build)
                if prev and m["run"] == N and m["wait"] == 0 and len(first) >= N and prev.get("all_first"):
                    rates.append((m["gen"] - prev["gen"]) / (m["t"] - prev["t"])); kvs.append(m["kv"])
                    if m["drafts"] > prev["drafts"]:   # mean acceptance length = 1 + accepted / drafts
                        accs.append(1 + (m["acc"] - prev["acc"]) / (m["drafts"] - prev["drafts"]))
                    if len(rates) * 10 >= a.measure_s:
                        break
                elif rates and m["run"] < N:
                    break   # requests finished: measure what we have
                if len(first) >= N and not (prev or {}).get("all_first"):
                    open(a.out + ".decode", "w").write(str(N))   # marker: this level is in pure decode (nsys trigger)
                m["all_first"] = len(first) >= N
                prev = m
            log(f"level {N}: prefill done for {len(first)}/{N} requests, last first-token at "
                f"{max(first.values()) - t_lv if first else -1:.0f} s into the level")
            r = dict(level=N, windows=len(rates), tok_s=round(statistics.median(rates), 1) if rates else None,
                     per_seq=round(statistics.median(rates) / N, 2) if rates else None,
                     kv=round(statistics.median(kvs), 3) if kvs else None,
                     accept_len=round(statistics.median(accs), 2) if accs else None,
                     running=m.get("run"), waiting=m.get("wait"), preempt=m.get("preempt"))
            res.append(r); log(json.dumps(r))
            json.dump(res, open(a.out, "w"), indent=1)
        for t in tasks:
            t.cancel()
    log("done")

asyncio.run(main())
