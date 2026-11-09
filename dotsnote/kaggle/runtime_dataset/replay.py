"""Closed-loop replay of a long-generation request mix on the live server, scheduled exactly like solve.py's main loop
(max --concurrency in flight, new requests only while server KV usage < --kv-admit or fewer than --min-running run, KV
polled every 20 s), to check steady-state throughput, KV pressure and preemptions over a long window.

Request lengths come from --lengths (JSON {prompt: [...], completion: [...]}, numeric token counts). Content is a
cellular-automaton simulation log (ignore_eos, max_tokens = sampled completion). --warm N: the first N
requests start mid-way (size-biased by completion length, uniform progress), so the mix is near steady state at once.

  python replay.py --port 8000 --tokenizer CKPT --lengths req_lengths.json --concurrency 40 --duration-s 5400 --out r.json
"""
import argparse, asyncio, json, random, re, statistics, time

import httpx
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--tokenizer", required=True)
ap.add_argument("--lengths", required=True)
ap.add_argument("--concurrency", type=int, default=40)
ap.add_argument("--kv-admit", type=float, default=0.85)
ap.add_argument("--min-running", type=int, default=8)
ap.add_argument("--duration-s", type=float, default=5400)
ap.add_argument("--warm", type=int, default=28, help="this many first requests start mid-way (keep their total "
                "context below the KV pool: they are admitted back to back before KV usage is visible)")
ap.add_argument("--max-model-len", type=int, default=262144)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()
T0 = time.time()
END = T0 + a.duration_s
tok = AutoTokenizer.from_pretrained(a.tokenizer)
L = json.load(open(a.lengths))
PAIRS = list(zip(L["prompt"], L["completion"]))
rnd = random.Random(a.seed)
KV = [0.0]


def log(*x):
    print(f"[replay {time.time() - T0:6.0f}s]", *x, flush=True)


def life_prompt(i, n):
    """~n-token chat request whose assistant turn is a Game of Life simulation log (same as load_sweep.py)"""
    r = random.Random(1000 * a.seed + i)
    H, W = r.randrange(14, 26), r.randrange(14, 26)
    g = [[int(r.random() < 0.35) for _ in range(W)] for _ in range(H)]

    def step(g):
        return [[int((n == 3) or (g[y][x] and n == 2)) for x in range(W)
                 for n in [sum(g[(y + dy) % H][(x + dx) % W] for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)]]
                for y in range(H)]
    head = (f"Simulate Conway's Game of Life on a {H}x{W} toroidal board ('#' alive, '.' dead). For every generation, "
            "print the full board, then one line stating how many cells are alive and what changed. Continue until "
            "told to stop.\n\n")
    parts, gen, have, cnt = [], 0, 0, 0
    board = lambda g: "\n".join("".join("#" if c else "." for c in row) for row in g)
    while have < n:
        parts.append(f"Generation {gen}:\n{board(g)}\nAlive: {sum(map(sum, g))}.")
        cnt += 1
        if cnt == 1 or cnt % 32 == 0:
            have = len(tok("\n\n".join(parts))["input_ids"])
        g = step(g); gen += 1
        if sum(map(sum, g)) < 6:
            g = [[int(r.random() < 0.35) for _ in range(W)] for _ in range(H)]
    msgs = [{"role": "user", "content": head + "Begin with this board.\n\n" + parts[0]},
            {"role": "assistant", "content": "\n\n".join(parts)},
            {"role": "user", "content": "Continue the simulation from the last generation, same format, as long as you can."}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def make_req(i, warm):
    if warm:   # steady state: a running request is size-biased by its length, at uniform progress
        pt, ct = rnd.choices(PAIRS, weights=[c for _, c in PAIRS])[0]
        done = int(rnd.random() * ct)
    else:
        (pt, ct), done = rnd.choice(PAIRS), 0
    ct = min(ct, a.max_model_len - pt - 4096)
    return dict(i=i, ctx=pt + done, gen=max(256, ct - done), prompt=life_prompt(i, pt + done))


stats = dict(started=0, finished=0, errors=0, ttft=[], gen_done=0)


async def run_req(c, r):
    t = time.time()
    body = dict(model="m", prompt=r["prompt"], max_tokens=r["gen"], temperature=0.7, ignore_eos=True, stream=True)
    try:
        async with c.stream("POST", f"http://127.0.0.1:{a.port}/v1/completions", json=body) as resp:
            first = False
            async for line in resp.aiter_lines():
                if line.startswith("data:") and not first:
                    first = True
                    stats["ttft"].append((round(t - T0), r["ctx"], round(time.time() - t, 1)))
                if time.time() > END:
                    break
        stats["finished"] += 1
    except Exception as e:
        stats["errors"] += 1
        log("error", r["i"], repr(e)[:160])


async def metrics(c):
    m = (await c.get(f"http://127.0.0.1:{a.port}/metrics", timeout=10)).text
    f = lambda name: sum(float(x) for x in re.findall(rf"^vllm:{name}(?:\{{[^}}]*\}})? ([0-9.e+-]+)", m, re.M))
    kv = re.findall(r"^vllm:kv_cache_usage_perc(?:\{[^}]*\})? ([0-9.e+-]+)", m, re.M)
    return dict(t=time.time(), run=f("num_requests_running"), wait=f("num_requests_waiting"),
                gen=f("generation_tokens_total"), kv=max([float(x) for x in kv] or [0.0]),
                acc=f("spec_decode_num_accepted_tokens_total"), drafts=f("spec_decode_num_drafts_total"),
                preempt=f("num_preemptions_total"))


async def monitor(c, series):
    prev = None
    while time.time() < END:
        try:
            m = await metrics(c)
            KV[0] = m["kv"]
            if prev:
                dt = m["t"] - prev["t"]
                d = lambda k: m[k] - prev[k]
                series.append(dict(t=round(m["t"] - T0), run=m["run"], wait=m["wait"], kv=round(m["kv"], 3),
                                   tps=round(d("gen") / dt, 1), preempt=m["preempt"],
                                   accept=round(1 + d("acc") / d("drafts"), 2) if d("drafts") else None))
                s = series[-1]
                log(f"run {s['run']:.0f} wait {s['wait']:.0f} kv {s['kv']:.2f} {s['tps']:.0f} tok/s acc {s['accept']} "
                    f"preempt {s['preempt']:.0f} started {stats['started']} finished {stats['finished']}")
            prev = m
        except Exception as e:
            log("metrics error", repr(e)[:120])
        await asyncio.sleep(20)


async def main():
    series = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(a.duration_s + 600),
                                 limits=httpx.Limits(max_connections=a.concurrency + 8)) as c:
        mt = asyncio.create_task(monitor(c, series))
        running, i = set(), 0
        nxt = asyncio.create_task(asyncio.to_thread(make_req, i, i < a.warm))
        while time.time() < END:
            while len(running) < a.concurrency and (KV[0] < a.kv_admit or len(running) < a.min_running or i < a.warm):
                r = await nxt
                i += 1
                nxt = asyncio.create_task(asyncio.to_thread(make_req, i, i < a.warm))
                running.add(asyncio.create_task(run_req(c, r)))
                stats["started"] += 1
            if running:
                _, running = await asyncio.wait(running, timeout=15, return_when=asyncio.FIRST_COMPLETED)
            else:
                await asyncio.sleep(15)
        for t in running:
            t.cancel()
        mt.cancel()
    steady = [s for s in series if s["t"] > 0.25 * a.duration_s]
    res = dict(args=vars(a), stats={k: v for k, v in stats.items() if k != "ttft"}, ttft=stats["ttft"], series=series,
               steady_tps=round(statistics.mean(s["tps"] for s in steady), 1) if steady else None,
               steady_run=round(statistics.mean(s["run"] for s in steady), 1) if steady else None,
               preemptions=series[-1]["preempt"] - series[0]["preempt"] if series else None,
               steady_preemptions=steady[-1]["preempt"] - steady[0]["preempt"] if steady else None)
    json.dump(res, open(a.out, "w"), indent=1)
    log("RESULT steady tok/s", res["steady_tps"], "steady running", res["steady_run"], "preemptions", res["preemptions"],
        "(steady", res["steady_preemptions"], ")",
        "finished", stats["finished"], "errors", stats["errors"])

asyncio.run(main())
