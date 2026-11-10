# Training: from Qwen3.8-Flash-Next-FP8 to the shipped model

This folder holds the code that turned the public `Qwen/Qwen3.8-Flash-Next-FP8` checkpoint into the model the Kaggle
notebook serves, plus the evaluation protocol used to accept or reject every change. The writeup's appendix
describes the method: compression is Appendix A, RL on the quantization scales is Appendix B, and the draft head is
Appendix C. The serving engine and the notebook are in `../kaggle/`. The submission's code dataset is unpacked in
`../kaggle/code_dataset/`, and this README refers to it instead of copying files from it.

```
FP8 model ──(1) own traces on ARC-AGI-2 training tasks──(2) teacher-forced routing capture
          ──(3) keep score: 320 of 512 experts per layer──(4) GPTQ: 160 experts at 4-bit + 160 at 3-bit  = k320q34
k320q34   ──(6) RL round 1 → rl1 ──round 2 → rl2 ──round 3 → rl3          (trains fp16 group scales + router rows)
rl3       ──(7) delta against k320q34 (1.3 GB)       rl1 rollouts ──(8) draft head (MTP layer) delta (0.2 GB)
every stage ──(5) serve on H200 + (9) pass@2 on the 120 public evaluation tasks, paired against a same-time control
```

## What you need

**Hardware.** The runs used H200 GPUs (141 GB). Each stage needs 1 to 24 of them:

| Stage | GPUs |
|---|---|
| traces, evaluation pools, RL rollouts | 2 per vLLM server; the shipped RL rounds ran 10-12 servers |
| routing capture | 2 |
| keep score | 1; GPTQ build: 2 (one per half of the layers) |
| RL training | 1 per data-parallel replica (about 118 GB peak at 32k tokens); the shipped rounds used 4-8 |
| draft head (target dump and training) | 1 (80 GB is enough) |

Plan for 1-2 TB of scratch space. The large items are the routing captures, the BF16 `ovr/` copy that every build and
RL export carries (about 150 GB), and the draft-head target dumps (about 20 KB per token).

**Software** (the versions used):
- Python 3.12 (the code also parses with 3.11).
- PyTorch 2.13 (CUDA 13 build) and transformers 5.18. transformers provides the Qwen4Exp model code used by the
  exactness check.
- safetensors, numpy and httpx.
- `fla-core` 0.6.0 (flash-linear-attention kernels). The student uses its chunked gated delta rule. Without it, the
  student falls back to the much slower reference implementation in transformers.
- vLLM 0.30.0 (stock, from PyPI) with flashinfer 0.6.18 for every H200 step: traces, capture, evaluation and
  rollout servers. The plugin in `plugin/` adds capture, pruning and BF16 expert overrides.
- vLLM 0.32.0 with flashinfer 0.7.0.post1 plus the serving overlay built from `../kaggle/code_dataset/fnext/`. Only
  the draft-head target dump and the closed-loop acceptance check need this. It is the same stack as on Kaggle.

**Model and data.**
- `Qwen/Qwen3.8-Flash-Next-FP8` from Hugging Face goes in `$FN_BASE_MODEL`.
- ARC-AGI-2 (github.com/arcprize/ARC-AGI-2): `$ARC_DATA` must contain `training/` and `evaluation/`. Only training
  tasks are used for calibration and RL. `rl/rollout.py`, `rl/prep_rl.py` and `draft_head/prep.py` assert that no
  evaluation task id is present (keep `$ARC_DATA` correct, or the guard in `prep_rl.py` has nothing to check against).
- RL rounds 2 and 3 also use:
  - ARC-AGI-1 (github.com/fchollet/ARC-AGI, `data/` with `training/` and `evaluation/`);
  - RE-ARC (github.com/michaelhodel/re-arc): the generated examples (`tasks/<id>.json` + `metadata.json`) and the
    source (`generators.py`, `verifiers.py`);
  - optionally, Kaggle's `arc-agi_evaluation_challenges.json` as an extra overlap reference.

  Every grid of these pools is checked against the evaluation set with a hash that ignores the 8 symmetries and
  colour permutations. The README commands pass these locations as `$ARC1_DATA`, `$REARC_DATA`, `$REARC_SRC` and
  `$KAGGLE_EVAL_JSON`.

### Environment variables

Every script reads these. The Python defaults equal the shell defaults in `env.sh`. Relative paths resolve against
the directory you run from, and all commands below run from the `flashnext/` folder of this repository after `source training/env.sh`.

| Variable | Default | Meaning |
|---|---|---|
| `FN_BASE_MODEL` | `models/Qwen3.8-Flash-Next-FP8` | local copy of the FP8 checkpoint (weights + tokenizer) |
| `FN_WORK` | `work` | all outputs: `traces*/`, `cap/`, `keep/`, `builds/`, `dense/`, `rl/`, `draft/`, `pools/`, `evals/`, `logs/` |
| `ARC_DATA` | `data/ARC-AGI-2/data` | ARC-AGI-2 `training/` and `evaluation/` |
| `LOGDIR` | `$FN_WORK/logs` | logs of the shell scripts |
| `FN_DRAFT_BUILD` | `$FN_WORK/builds/rl1` | draft head: the build whose layer-48 codes and router it loads (identical in all builds) |
| `FN_HOT_VOCAB` | `$FN_WORK/hot32k_tr.pt` | draft head: the 32,768-token draft vocabulary |
| `QR_ARCKIT` | unset; `block` in `rl/rollout.sh` | harness: `block` (the shipped setting) loads the `ak` library, and a program that breaks a training-pair invariant on a test output does not count as verified |
| `QR_ARCKIT_PATH` | `kaggle/code_dataset/arckit.py` of this repository | the `ak` library (the exact file the submission ships) |
| `MTP_KV_FP8` | `0` | draft-head training: emulate the FP8 KV cache (`1` in the shipped runs) |
| `KD_PAD`, `KD_PLE_SHM` | `2048` (`4096` in `rl/train_rl.sh` and `kd/train_kd.sh`), `/dev/shm/ple` | student: sequence padding; RAM copy of the n-gram table |

Script-specific variables (GPU ids, ports, inputs) are listed in each script's header and in the reference table at
the end.

## Recipe

Settings below are those of the shipped artifacts. Sampling on GPUs is not bit-reproducible, so a rerun gives
statistically equivalent rather than identical traces, rollouts and checkpoints.

### 0. Setup

```bash
source training/env.sh                       # FN_BASE_MODEL, FN_WORK, ARC_DATA, LOGDIR
W=$FN_WORK; mkdir -p $W/cap $W/keep $W/builds $W/deltas $W/student $W/rl/roll $W/rl/prep $W/rl/runs $W/draft $W/pools $W/evals
```

### 1. Self-generated traces on training tasks

The FP8 model answers ARC-AGI-2 training tasks itself, once in direct mode and once in a simple python-tool mode
(`harness/probe/run.py`), one attempt per test input. The shipped calibration set has 1,040 sequences on 480 tasks:
688 on 320 random tasks at a 64k-token cap, and 352 on 160 harder tasks at a 96k cap.

```bash
# FP8 servers: TP2 + expert parallel, MTP 3, 131k context (the shipped traces used 128 sequences per server)
for i in 0 1 2 3; do SEQS=128 UTIL=0.92 bash training/serve/serve_build.sh fp8 $((2*i)),$((2*i+1)) $((8000+i)) \
  --seed $((2000+i)) > $LOGDIR/fp8_$i.log 2>&1 & done
EPS=127.0.0.1:8000,127.0.0.1:8001,127.0.0.1:8002,127.0.0.1:8003
ENDPOINTS=$EPS bash training/prune_gptq/traces.sh                      # 320 tasks = random.Random(2027).sample(...)
# 160 harder tasks, not among the 320, written to $W/traces_long/tasks.txt first (see "Provenance" below)
ENDPOINTS=$EPS OUT=$W/traces_long MAXTOK=98304 bash training/prune_gptq/traces.sh
```

Without `ENDPOINTS`, `traces.sh` starts its own servers on `GPU_PAIRS` (`"0,1 2,3"`). Use `SERVE_LEN=131072` for the
96k set.

### 2. Routing capture by teacher-forced replay

An eager FP8 server with `plugin/fnplug.py` (`FN_CAP_DIR`) records routing for every token of every request. The
MTP layer is captured as row 48. For each layer and expert it sums top-10 counts and routing weights over all tokens,
and it keeps a random 8% of tokens with their gate inputs and router logits. Each trace is sent as one prompt with
`max_tokens=1`.

Tasks are split into a calibration set (`tr*`) and a held-out set (`ho*`): 32 of the 320 and 16 of the 160. These
48 tasks are also excluded from RL.

```bash
python3 - <<'PY'
import os, random
W = os.environ["FN_WORK"]
for src, n, dst in (("traces", 32, "ho_tasks.txt"), ("traces_long", 16, "ho_tasks_c.txt")):
    t = open(f"{W}/{src}/tasks.txt").read().split()
    open(f"{W}/cap/{dst}", "w").write("\n".join(random.Random(7).sample(sorted(t), n)) + "\n")
PY
# 320-task traces -> cap/tr_a, cap/ho_a (12.6M calibration tokens per layer in the shipped run)
TRACES=$W/traces/traces.jsonl.gz HO=$W/cap/ho_tasks.txt TAG=_a bash training/prune_gptq/capture.sh
# long traces -> cap/tr_c, cap/ho_c; with SPLIT the long ones are also replayed up to 30k (cap/*_c_pre), so that
# routing beyond 30k = tr_c - tr_c_pre can be checked
CAP_LEN=131072 REPLAY_LEN=128000 SPLIT=30000 TRACES=$W/traces_long/traces.jsonl.gz HO=$W/cap/ho_tasks_c.txt TAG=_c \
  bash training/prune_gptq/capture.sh
# tool spans: prefixes ending before <tool_call>, after </tool_call> and after </tool_response> (cap/tc_pre,
# tc_call, tc_resp); tool-call routing = tc_call - tc_pre, tool-response routing = tc_resp - tc_call
CAP_LEN=131072 SERVE_ONLY=1 bash training/prune_gptq/capture.sh &      # wait for SERVER_READY in its output
python3 training/prune_gptq/replay_tc.py --traces $W/traces/traces.jsonl.gz $W/traces_long/traces.jsonl.gz \
  --endpoint 127.0.0.1:8000 --model-dir $FN_BASE_MODEL --cap $W/cap --max-calls 3 --max-len 120000 --conc 16
# optional, for header protection (keep_hdr.py): cap/hd_pre, hd_post around "<tool_call>...<parameter=code>"
python3 training/prune_gptq/replay_tc.py --header --traces $W/traces/traces.jsonl.gz $W/traces_long/traces.jsonl.gz \
  --endpoint 127.0.0.1:8000 --model-dir $FN_BASE_MODEL --cap $W/cap --max-calls 2 --max-len 120000 --conc 16
```

The shipped run replayed the 320-task traces in two incremental passes while they were still being written
(`TAG=_a`, then `TAG=_b`, sharing `cap/replayed.txt`). The statistics add up, so one pass is equivalent. Wherever
the shipped commands list `tr_a,tr_b`, use `tr_a`.

### 3. Keep score: 320 of 512 experts per layer

For each expert *e* of a layer:

score(*e*) = S(*e*) / Σ_e′ S(*e*′) + 0.5 · C(*e*) / Σ_e′ C(*e*′) + 0.25 · T(*e*) / Σ_e′ T(*e*′),  with S(*e*) = Σ_t g_{t,e}·‖y_{t,e}‖₂

S is a REAP-style saliency: gate weight times expert output norm, summed over calibration tokens. C and T are the
expert's routing mass on tool-call and tool-response tokens. Each term is normalized per layer. The
top 320 are kept in every layer, the MTP layer included. Routing becomes softmax over the kept logits, top 10,
renormalized.

```bash
python3 training/prune_gptq/gptq_fn.py keep --cap $W/cap/tr_a,$W/cap/tr_c --out $W/keep --ks 384,320,256 \
  --crit reapsum --gpu 0 --batch 32                     # -> keep/scores_reapsum.pt, keep/keep_reapsum_{K}.pt
python3 training/prune_gptq/keep_protect.py --cap $W/cap --scores $W/keep/scores_reapsum.pt --ks 256,320,384 \
  --out $W/keep                                         # -> keep/keep_mix0.5_320.pt  (the shipped keep set)
```

`keep_protect.py` prints tool-call and tool-response coverage, and writes three blends (`mix0.5`, `mix1`, `mix2`)
plus a hard-protection variant (`keep_tool_{K}`). The shipped build uses `keep_mix0.5_320.pt` (weights 0.5 and
0.25).

`keep_hdr.py --php 0.9 --tag hd9` also forces in the experts that carry 90% of the routing of the
`<parameter=code>` header. This is what fixed the tool format at K = 256 (Appendix A). The shipped K = 320 build does
not need it.

Check every keep set in the agent loop before trusting it (step 5).

### 4. GPTQ build: 160 experts at 4-bit, 160 at 3-bit

- Group size 128, asymmetric (fp16 scale, uint8 zero), with an MSE clip search over 10 ratios from 1.0 to 0.55.
- Activation order; damping mean(diag H)·(0.01 + 4/n).
- Hessians come from router-weighted gate inputs under the pruned routing. Gate and up are quantized jointly, and the
  down projection's Hessian uses the quantized activations.
- The 160 experts with the largest routing-weight sum per layer get 4 bits; the rest get 3 bits.

```bash
CAP=$W/cap/tr_a,$W/cap/tr_c,$W/cap/tc_resp G0=0 G1=1 bash training/prune_gptq/buildq.sh "k320q34:mix0.5_320:160:4_all:3"
```

`$W/builds/k320q34/` then holds the following, with non-expert weights taken unchanged from the FP8 checkpoint:
- `lowbit-L{00..48}.pt`: experts in planar packing, keyed by original id; `gu` rows are [gate 640; up 640]; the
  dequantized weight is w = fp16 scale · (q − zero).
- `router-L*.pt`: router rows of the kept experts.
- `keep.pt` and `keep.json`; `gptq-L*.json` with per-layer error statistics.
- `ovr/`: the same codes dequantized to BF16, used for H200 serving.
- `README.md`.

The shipped build is 35.6 GB of low-bit files, 8.7 GB per GPU when spread over 4.

Optional checks:
- `gptq_fn.py eval --cap $W/cap/ho_a --keep KEEP --cand name=BUILD_DIR --out $W/heldout_err.jsonl` measures held-out
  MoE output error against the full 512-expert FP8 layer.
- `eval/cmp_builds.py A B` compares two builds tensor by tensor.

### 5. Serving a build for evaluation and rollouts

H200 servers use stock vLLM 0.30 with `plugin/fnplug.py`:
- the experts of the FP8 checkpoint outside the keep set are dropped at load time;
- the kept ones are replaced by the BF16 dequantized copies in `ovr/`;
- the router rows are sliced to the keep set, or taken from `ovr/` when the build carries trained rows (RL exports);
- MTP 3, TP2 with expert parallelism.

The numerics are those of the low-bit codes. The L4 engine in `../kaggle/` runs the same codes with low-bit kernels.

```bash
bash training/serve/serve_build.sh $W/builds/k320q34 0,1 8000               # one server
BUILD=$W/builds/k320q34 NGPU=8 bash training/serve/serve_node.sh            # four servers, ports 8000-8003
```

**Tool-format check for a new keep set.** Run at least 30 REPL sessions on held-out training tasks and count
malformed calls:

```bash
QR_ARCKIT=block python3 training/harness/pool.py --endpoints 127.0.0.1:8000 --model-dir $FN_BASE_MODEL \
  --split $ARC_DATA/training --tasks $(head -24 $W/cap/ho_tasks.txt | paste -sd,) --arms repl:2 --prompt ak \
  --out $W/pools/fmt_k320q34
python3 training/eval/fmt_check.py $W/pools/fmt_k320q34        # sessions, verified rate, no-code errors per call
```

### 6. RL on the quantization scales and router rows

**What is trained.** A log-offset t on every per-group scale of the kept experts (s = s₀·e^t, 589.8M parameters),
plus the kept router rows (39.3M). Codes, zeros, the keep set and all other weights are frozen.

**Gradient path.** `student/student_fn.py` is a plain-PyTorch Qwen3.8-Flash-Next:
- Gated DeltaNet through fla;
- sparse attention through flex attention with the indexer's block selection (no grad);
- hyper-connections and the per-layer n-gram embedding;
- each expert weight rebuilt from its packed codes in forward and backward by a custom autograd function.

One full model fits on one H200.

**Objective.** REINFORCE with a leave-one-out baseline: A_i = R_i − mean_{j≠i} R_j over the task's samples, with no
std scaling, clipped to [−2, 2]. The loss is −Σ_i A_i Σ_t log π(y_{i,t}) / (n · 16,384), with one on-policy pass per
round. Only tokens the model sampled at T = 1 are trained; tool output, nudges and forced answers are masked.
Sequences are cut at 32,768 tokens. Optimizer:
- Adam with β = (0.9, 0.99) and ε = 1e-10 on fp32 masters;
- learning rate 3e-4 for the scales and 1.5e-5 for the router;
- 3 warm-up steps, then cosine;
- 64 sequences per step.

**Rollouts.** These use the evaluation-pool harness (`harness/pool.py`, through `rl/rollout.py`). It shares the
submission's prompt, `python` tool protocol, `ak` library and verification rules. The Kaggle solver in `../kaggle/`
adds runtime fixes on top, such as carrying imports and definitions between cells and a longer execution limit (10 s
here). The settings:
- `QR_ARCKIT=block`, `--prompt ak`;
- T 1.0, top-p 0.95, top-k 20;
- 98,304-token cap, 24 tool calls, a nudge after 49,152 thinking tokens;
- a stop at the first verified program.

`rollout.py` also records the character span of every sampled completion, so the trainer masks exactly the sampled
tokens.

**Rewards** (`prep_rl.py`) are scored per session:

| Outcome | Reward |
|---|---|
| verified and right | +1 |
| verified and wrong | −0.6 |
| unverified and right | +0.5 |
| unverified and wrong | −0.6 |

`prep_rl.py` defaults to the first round-1 reward (+1, −1, +0.5, −0.2), which taught the policy to abstain from
verifying. The shipped rounds always pass `--rewards 1,-0.6,0.5,-0.6`.

`trained.pt` stores cumulative offsets relative to the original k320q34 build. Each round resumes the previous
`trained.pt` and exports against `$W/builds/k320q34` (`RL_BUILD`'s default). Never train or export on top of an
exported build, or the offsets would be applied twice.

#### 6.0 One-time setup

```bash
python3 training/student/make_dense.py $W/dense      # dense-only copy (no routed experts, MTP or n-gram table)
python3 training/student/ple_stage.py                # n-gram table -> /dev/shm/ple (needs about 51 GB of RAM)
# optional exactness check of the student against transformers (2 GPUs; logit KL about 1.5e-4)
python3 training/student/prep_recs.py $W/student/recs.jsonl $W/traces/traces.jsonl.gz $W/traces_long/traces.jsonl.gz
python3 training/student/check_hf.py --layers 4 --T 3000
# task lists: 48 held-out capture tasks, the other 952 training tasks, and one task that shares a concept with an
# evaluation task (found by the overlap audit; removed from rounds 2 and 3)
cat $W/cap/ho_tasks.txt $W/cap/ho_tasks_c.txt > $W/rl/ho48.txt
ls $ARC_DATA/training | sed 's/\.json$//' | grep -vxFf $W/rl/ho48.txt > $W/rl/train_all.txt
echo b7999b51 > $W/rl/excl_eval_overlap.txt
H="--repl-cap 98304 --turn-tokens 49152 --max-tool-calls 24 --force-tokens 3072 --prompt ak"   # = the shipped harness
TR="--split $ARC_DATA/training --exclude $W/rl/ho48.txt"
R=$W/rl/roll
```

`rl/rollout.sh` runs one rollout set. In the commands below, `$EPS` lists the servers (`host:port,...`) of the policy
being sampled: k320q34 in round 1, rl1 in round 2, rl2 in round 3. `$EPS1` lists rl1 servers.

#### 6.1 Round 1 → rl1 (1,346 samples, 21 steps)

Sample from k320q34 servers (`BUILD=$W/builds/k320q34 ... serve_node.sh`).

The training split is easy for the build: 99% of sessions verify, and only 124 of 948 tasks gave mixed reward groups.
Leave-one-pair-out variants, in which one demonstration becomes the test, add more groups. A second sampling stage
adds 4 samples on every task with a non-right outcome.

```bash
RL_EPS=$EPS RL_OUT=$R/r1_train RL_ARGS="$H $TR --tasks @$W/rl/train_all.txt --arms repl:4 --concurrency 44 --seed 7001" bash training/rl/rollout.sh
RL_EPS=$EPS RL_OUT=$R/r1_loo RL_ARGS="$H $TR --tasks @$W/rl/train_all.txt --loo --loo-one --arms repl:4 --concurrency 24 --seed 7201" bash training/rl/rollout.sh
python3 training/rl/roll_stats.py $R/r1_train --informative $W/rl/r1_mixed.txt          # tasks not all verified-right
python3 training/rl/roll_stats.py $R/r1_loo --informative $W/rl/r1_loo_mixed.txt; sed -i 's/~.*//' $W/rl/r1_loo_mixed.txt
RL_EPS=$EPS RL_OUT=$R/r1_train_more RL_ARGS="$H $TR --tasks @$W/rl/r1_mixed.txt --arms repl:4 --concurrency 16 --seed 7301" bash training/rl/rollout.sh
RL_EPS=$EPS RL_OUT=$R/r1_loo_more RL_ARGS="$H $TR --tasks @$W/rl/r1_loo_mixed.txt --loo --loo-one --arms repl:4 --concurrency 12 --seed 7401" bash training/rl/rollout.sh
python3 training/rl/prep_rl.py $W/rl/prep/r1 $R/r1_train $R/r1_train_more $R/r1_loo $R/r1_loo_more --max-len 32768 \
  --exclude $W/rl/ho48.txt --rewards 1,-0.6,0.5,-0.6
RL_OUT=$W/rl/runs/r1 RL_NREP=4 RL_EXPORT=$W/builds/rl1 \
  RL_ARGS="--data $W/rl/prep/r1 --per-step 64 --lr-scales 3e-4 --lr-router 1.5e-5 --warmup 3" bash training/rl/train_rl.sh
```

Later rounds reuse the informative task lists of round 1. Those lists came from a prep with the default rewards
(the first round-1 reward), which marks slightly different groups as informative:

```bash
python3 training/rl/prep_rl.py $W/rl/prep/r1_lists $R/r1_train $R/r1_train_more $R/r1_loo $R/r1_loo_more --max-len 32768 --exclude $W/rl/ho48.txt
python3 - <<'PY'
import json, os
W = os.environ["FN_WORK"]
ts = {json.loads(l)["task"] for l in open(f"{W}/rl/prep/r1_lists/index.jsonl")}
open(f"{W}/rl/informative.txt", "w").write("\n".join(sorted(t for t in ts if "~" not in t)) + "\n")      # 153 tasks
open(f"{W}/rl/informative_loo.txt", "w").write("\n".join(sorted(t.split("~")[0] for t in ts if "~" in t)) + "\n")
PY
grep -vxFf $W/rl/excl_eval_overlap.txt $W/rl/informative.txt > $W/rl/informative_clean.txt  # 152 tasks
```

#### 6.2 Round 2 → rl2 (2,068 samples in 222 groups, 32 steps)

Sample from rl1 servers. The reward is budget-aware: each positive reward is multiplied by
max(0.2, 1 − tokens/98,304). Only groups whose unshaped rewards disagree are kept, so the budget term re-weights
within groups that carry a correctness signal.

```bash
RL_EPS=$EPS RL_OUT=$R/r2_train RL_ARGS="$H $TR --tasks @$W/rl/informative.txt --arms repl:8 --concurrency 40 --seed 9701" bash training/rl/rollout.sh
RL_EPS=$EPS RL_OUT=$R/r2_loo RL_ARGS="$H $TR --tasks @$W/rl/informative_loo.txt --loo --loo-one --arms repl:8 --concurrency 12 --seed 9801" bash training/rl/rollout.sh
# public pool: ARC-AGI-1 tasks that are not ARC-AGI-2 training tasks + 2 RE-ARC instances per generator,
# overlap-filtered against every evaluation grid (826 tasks kept in the shipped run)
python3 training/rl/make_pool.py $W/rl/tasks/pool_public --arc1 $ARC1_DATA --rearc $REARC_DATA \
  --arc2-train $ARC_DATA/training --arc2-eval $ARC_DATA/evaluation --eval-json $KAGGLE_EVAL_JSON --rearc-per-task 2
PUB="--split $W/rl/tasks/pool_public"
RL_EPS=$EPS RL_OUT=$R/screen_pool_public RL_ARGS="$H $PUB --tasks @$W/rl/tasks/pool_public/_tasks.txt --arms repl:4 --concurrency 32 --seed 9981" bash training/rl/rollout.sh
python3 training/rl/roll_stats.py $R/screen_pool_public --informative $W/rl/pool_public_informative.txt  # 109 tasks
RL_EPS=$EPS RL_OUT=$R/r2_pool_public RL_ARGS="$H $PUB --tasks @$W/rl/pool_public_informative.txt --arms repl:8 --concurrency 40 --seed 9991" bash training/rl/rollout.sh
python3 training/rl/prep_rl.py $W/rl/prep/r2 $R/r2_train $R/r2_loo $R/screen_pool_public $R/r2_pool_public --max-len 32768 \
  --exclude $W/rl/ho48.txt,$W/rl/excl_eval_overlap.txt --rewards 1,-0.6,0.5,-0.6 --budget-aware 1.0 --require-mixed --min-group 3
RL_OUT=$W/rl/runs/r2 RL_NREP=8 RL_EXPORT=$W/builds/rl2 RL_ARGS="--data $W/rl/prep/r2 --resume $W/rl/runs/r1/trained.pt \
  --per-step 64 --lr-scales 3e-4 --lr-router 1.5e-5 --warmup 3" bash training/rl/train_rl.sh
```

#### 6.3 Round 3 → rl3 (903 samples in 468 groups, 30 steps)

Sample from rl2 servers. Each task's 2 shortest right samples get A = +1, and everything else is dropped. The two
extra RE-ARC pools are built first and screened on rl1.

```bash
python3 training/rl/make_pool_hard.py $W/rl/tasks/pool_hard --rearc $REARC_DATA --arc2-eval $ARC_DATA/evaluation \
  --eval-json $KAGGLE_EVAL_JSON --inform $W/rl/pool_public_informative.txt --per-gen 1 --per-gen-inform 4     # hardest decile per generator
python3 -I training/rl/make_pool_comp.py $W/rl/tasks/pool_comp --rearc-src $REARC_SRC --arc2-eval $ARC_DATA/evaluation \
  --eval-json $KAGGLE_EVAL_JSON --n-tasks 600                                               # rule compositions B(A(x))
# screen both on rl1 servers (x4)
RL_EPS=$EPS1 RL_OUT=$R/screen_pool_hard RL_ARGS="$H --split $W/rl/tasks/pool_hard --tasks @$W/rl/tasks/pool_hard/_tasks.txt --arms repl:4 --concurrency 12 --seed 10202" bash training/rl/rollout.sh
RL_EPS=$EPS1 RL_OUT=$R/screen_pool_comp RL_ARGS="$H --split $W/rl/tasks/pool_comp --tasks @$W/rl/tasks/pool_comp/_tasks.txt --arms repl:4 --concurrency 12 --seed 10201" bash training/rl/rollout.sh
python3 training/rl/roll_stats.py $R/screen_pool_hard --informative $W/rl/pool_hard_informative.txt  # 163 tasks
python3 - <<'PY'                                                                     # 91 tasks with mixed correctness
import collections, json, os
W = os.environ["FN_WORK"]; r = collections.defaultdict(list)
for l in open(f"{W}/rl/roll/screen_pool_comp/samples.jsonl"):
    d = json.loads(l)
    if d.get("status") == "error": continue
    v = [p for p in d["progs"] if p.get("verified")]
    r[d["task"]].append(bool(all(v[0]["tests_correct"]) if v else all(d.get("answer_correct") or [False])))
open(f"{W}/rl/pool_comp_mixed.txt", "w").write("\n".join(sorted(t for t, v in r.items() if any(v) and not all(v))) + "\n")
PY
# round-3 rollouts on rl2 servers (the shipped collection was cut after 3,759 samples)
X="--exclude $W/rl/ho48.txt,$W/rl/excl_eval_overlap.txt"
RL_EPS=$EPS RL_OUT=$R/r3_train RL_ARGS="$H --split $ARC_DATA/training $X --tasks @$W/rl/informative_clean.txt --arms repl:8 --concurrency 12 --seed 11001" bash training/rl/rollout.sh
RL_EPS=$EPS RL_OUT=$R/r3_pool_public RL_ARGS="$H $PUB --tasks @$W/rl/pool_public_informative.txt --arms repl:8 --concurrency 8 --seed 11002" bash training/rl/rollout.sh
RL_EPS=$EPS RL_OUT=$R/r3_pool_hard RL_ARGS="$H --split $W/rl/tasks/pool_hard --tasks @$W/rl/pool_hard_informative.txt --arms repl:8 --concurrency 10 --seed 11003" bash training/rl/rollout.sh
RL_EPS=$EPS RL_OUT=$R/r3_pool_comp RL_ARGS="$H --split $W/rl/tasks/pool_comp --tasks @$W/rl/pool_comp_mixed.txt --arms repl:8 --concurrency 8 --seed 11004" bash training/rl/rollout.sh
RL_EPS=$EPS RL_OUT=$R/r3_pool_hard_b RL_ARGS="$H --split $W/rl/tasks/pool_hard --tasks @$W/rl/pool_hard_informative.txt --arms repl:4 --concurrency 24 --seed 11013" bash training/rl/rollout.sh
RL_EPS=$EPS RL_OUT=$R/r3_pool_comp_b RL_ARGS="$H --split $W/rl/tasks/pool_comp --tasks @$W/rl/pool_comp_mixed.txt --arms repl:4 --concurrency 16 --seed 11014" bash training/rl/rollout.sh
python3 training/rl/prep_rl.py $W/rl/prep/r3 $R/r3_train $R/r3_pool_public $R/r3_pool_hard $R/r3_pool_comp $R/r3_pool_hard_b $R/r3_pool_comp_b \
  --max-len 32768 --exclude $W/rl/ho48.txt,$W/rl/excl_eval_overlap.txt --concise 2 --min-group 3
RL_OUT=$W/rl/runs/r3 RL_NREP=8 RL_EXPORT=$W/builds/rl3 RL_ARGS="--data $W/rl/prep/r3 --resume $W/rl/runs/r2/trained.pt \
  --per-step 64 --steps 30 --lr-scales 3e-4 --lr-router 1.5e-5 --warmup 3" bash training/rl/train_rl.sh
```

Notes:
- `train_rl.sh` runs `RL_NREP` single-GPU replicas on one machine. The shipped rounds used 4 replicas (round 1) and
  8 over two machines (rounds 2 and 3). Set `RL_NNODES`, `RL_NODE_RANK` and `MASTER_ADDR` (NCCL over TCP) for
  several machines, and run the script on each.
- The gradient is averaged over the 64 sequences of a step, so the replica count does not change the update.
- On a network filesystem, copy `$W/dense` and the build's `lowbit-*`, `router-*` and `keep.*` files to local disk
  first; loading from a shared filesystem with many ranks was slow.
- Round 3 cycles its 903 records about twice (30 × 64).
- `rl/audit_lp.py` compares, on prepared records, the serving stack's teacher-forced log-probs with the trainer's.
  The shipped audit measured 0.053 nats per token, unbiased, and the sampler and staleness effects. Run it with
  `--recs $W/rl/prep/rN --endpoint HOST:PORT --build $W/builds/k320q34 --model $W/dense --resume <trained.pt> --out audit.json`.
- Levers that failed are documented in Appendix B and are not part of the recipe: augmentation, PPO clip,
  trainable norms, 64k credit and relabelling. Some still exist as `prep_rl.py` / `train_rl.py` options that are off
  by default: `--len-penalty`, `--concise-neg`, `--ppo-clip`, `--kl-anchor`, `--trainable ...,norms`.

### 7. Delta against the base build

An RL export differs from k320q34 only in fp16 scales and router rows. The submission ships k320q34 plus this delta
and rebuilds the rest at start-up with `../kaggle/code_dataset/delta_build.py`:

```bash
python3 kaggle/code_dataset/delta_build.py make  $W/builds/k320q34 $W/builds/rl3 $W/deltas/rl3        # 1.3 GB
python3 kaggle/code_dataset/delta_build.py build $W/builds/k320q34 $W/deltas/rl3 $W/builds/rl3_rebuilt
python3 kaggle/code_dataset/delta_build.py check $W/builds/rl3_rebuilt $W/builds/rl3   # 99 files, all identical
python3 training/eval/cmp_builds.py $W/builds/k320q34 $W/builds/rl3                     # which tensors changed
```

`build` accepts several deltas in order. The draft head (step 8) is the second: `build BASE RL_DELTA DRAFT_DELTA OUT`.

### 8. Draft head (MTP layer)

Only layer 48 (the MTP layer) is trained: its dense tensors, norms, router rows and expert scale offsets, 100.7M
parameters in all. The target model is unchanged, and rejection sampling keeps the output distribution exact.

At draft step j, α_j = Σ_v min(p_j(v), q_j(v)) over the target's processed top-20 tokens (top-k 20, top-p 0.95).
Here q is the draft over the hot vocabulary, restricted to its own top 20. The objective is the expected accepted
length E = Σ_s Π_{j≤s} α_j, with the loss −E + 0.3·KL. Training unrolls 4 draft steps, emulates the FP8 KV cache,
and weights prose positions ×2.

**8.1 Hot vocabulary.** The 32,768 most frequent token ids of the model's own completions, topped up with special
tokens and the lowest unseen ids. The shipped file is `../kaggle/code_dataset/hot32k_tr.pt`. It was counted over
13.2M completion tokens of direct and REPL attempts on 96 training tasks (five `harness/probe/run.py` runs at a 24k
cap, served from a community 4-bit copy of the same model). Any such traces of the model work:

```bash
python3 training/draft_head/vocab_stats.py --traces RUN1/traces.jsonl.gz RUN2/traces.jsonl.gz ... \
  --model-dir $FN_BASE_MODEL --out $W/hot32k_tr.pt --size 32768 --holdout-frac 0.2
```

**8.2 Data.** The data is rl1's own rollouts: `r2_train`, `r2_loo` and every leave-one-pair-out variant of the 153
informative tasks, 3,328 sequences or 99M tokens in all. 10% of base tasks are held out by crc32 (409 sequences, 26
tasks).

```bash
RL_EPS=$EPS1 RL_OUT=$R/draft_loo_all RL_ARGS="$H $TR --tasks @$W/rl/informative.txt --loo --arms repl:4 --concurrency 56 --seed 9951" bash training/rl/rollout.sh
python3 training/draft_head/prep.py $W/draft/recs $R/r2_train $R/r2_loo $R/draft_loo_all
```

**8.3 Target dump.** The deployed low-bit stack (vLLM 0.32 + overlay, one GPU) prefills each transcript. It records
the hidden stream that enters the MTP layer and the target's raw top-20 log-probs at every position.

```bash
export VLLM032_SP=/path/to/vllm032-env/lib/python3.12/site-packages DRAFT_TARGET=$W/builds/rl1
source training/draft_head/overlay_env.sh       # first use: builds the overlay in $W/overlay (fnext/apply.sh)
python3 training/draft_head/dump.py --recs $W/draft/recs --out $W/draft/dump/ho  --split ho --n 60 --bt 65600 --util 0.88
python3 training/draft_head/dump.py --recs $W/draft/recs --out $W/draft/dump/tr  --split tr --n 700 --bt 65600 --util 0.88 --kv-skip 48 --seed 2
python3 training/draft_head/dump.py --recs $W/draft/recs --out $W/draft/dump/tr2 --split tr --n 500 --skip 700 --bt 65600 --util 0.88 --kv-skip 48 --seed 2
```

**8.4 Training** on one GPU in two stages. The warm start uses KL plus the acceptance term. The shipped stage uses
the expected accepted length:
- 300 updates of 4 sequences;
- AdamW at 2e-5 for dense tensors and router, 3e-4 for scales;
- 40,960-token crops and gradient clip 1.0;
- position weights: prose ×2, and 1 / 1.6 / 4 for context below 16k / 16k-32k / above 32k;
- the best held-out checkpoint is kept.

```bash
export MTP_KV_FP8=1 FN_DRAFT_BUILD=$W/builds/rl1 FN_HOT_VOCAB=$W/hot32k_tr.pt
D=$W/draft/dump
python3 training/draft_head/train.py --dump $D/tr $D/tr2 --ho-dump $D/ho --out $W/draft/runs/warm --steps 4 \
  --lr 6e-5 --lr-scales 1e-3 --lr-router 6e-5 --tv 1.0 --eval-every 50 --max-train-len 32768
python3 training/draft_head/train.py --dump $D/tr $D/tr2 --ho-dump $D/ho --out $W/draft/runs/elen --steps 4 \
  --init $W/draft/runs/warm/best.pt --obj elen --kl-w 0.3 --prose-w 2 --ctx-bins 16384:1,32768:1.6,40960:4 \
  --lr 2e-5 --lr-scales 3e-4 --lr-router 2e-5 --eval-every 50 --max-train-len 40960
```

The shipped warm start saw the first 832 dumped training sequences (208 updates; best at update 200). The main run
saw all 1,200 (300 updates; best at update 250, about 1.5 h).

**8.5 Export and checks.**

```bash
python3 training/draft_head/accept.py --dump $D/ho --k 4 --delta $W/draft/runs/elen/best.pt --out $W/draft/accept.json
python3 training/draft_head/export.py make  $W/draft/runs/elen/best.pt $W/builds/rl1 $FN_BASE_MODEL $W/deltas/draft  # 202 MB
python3 training/draft_head/export.py build $W/builds/rl1 $W/deltas/draft $W/builds/rl1_draft
python3 training/draft_head/export.py check $W/builds/rl1_draft $W/draft/runs/elen/best.pt $W/builds/rl1 $FN_BASE_MODEL
# closed-loop acceptance in real vLLM with the deployed draft settings (int8 hot-vocabulary head, top-20 drafts)
export FN_MTP_HEAD=int8 FN_MTP_HEAD_VOCAB=$W/hot32k_tr.pt FN_MTP_DRAFT_TOPK=20 DRAFT_TARGET=$W/builds/rl1_draft
source training/draft_head/overlay_env.sh; unset VLLM_ENABLE_V1_MULTIPROCESSING
FN_MTP_OVR=$W/builds/rl1_draft/mtp_ovr.safetensors python3 training/draft_head/vacc.py --recs $W/draft/recs --n 48 \
  --out $W/draft/vacc.json                                      # shipped: 2.79 -> 3.13 tokens per step
```

`export.py make` writes three files plus `manifest.json`, the format `delta_build.py build` takes as a second delta:
- `mtp_ovr.safetensors`: the changed dense MTP tensors, served with `FN_MTP_OVR`;
- `scales-L48.pt`;
- `router-L48.pt`.

The draft head was trained on rl1 but deployed with rl3. Its offline acceptance is the same on rl1, rl2 and rl3
targets: 3.159 / 3.158 / 3.161 tokens per step. Further checks:
- `check_exact.py` compares the PyTorch MTP layer with vLLM's drafter on dumps made with `--draft-check N`.
- `test_mtpft.py` holds CPU unit tests. It needs vLLM 0.32 and the overlay's `mtp/` package on `PYTHONPATH`
  (`kaggle/code_dataset/fnext`).

### 9. Evaluation protocol

**Local set.** The 120 public evaluation tasks in two halves: `eval/holdout40.txt` and the other 80. The score is
pass@2 under a global budget of 100k generated tokens per test input.

**Pools.** Sample 8 independent REPL sessions per task with the shipped harness, from the build under test and
from a control sampled at the same time (pools a day apart drifted by +3.3 points):

```bash
comm -23 <(ls $ARC_DATA/evaluation | sed 's/\.json$//' | sort) <(sort training/eval/holdout40.txt) > $W/rest80.txt
for half in holdout40:training/eval/holdout40.txt rest80:$W/rest80.txt; do
  QR_ARCKIT=block python3 training/harness/pool.py --endpoints $EPS --model-dir $FN_BASE_MODEL \
    --split $ARC_DATA/evaluation --tasks @${half#*:} --arms repl:8 --prompt ak --seed 8101 --out $W/pools/${half%%:*}_rl3
done
```

**Paired comparison.** `policy_sim.py` replays a budgeted run on a pool:
- sessions go to the task with the fewest so far while the global budget lasts, and each session is capped (`--cap`);
- every session votes with its verified program; forced answers and partial programs count less;
- the top two grids are the attempts.

`paired_diff.py` averages each task's expected pass@2 over 100 sample orders and bootstraps the build-minus-control
difference over tasks.

```bash
python3 training/eval/paired_diff.py --build $W/pools/holdout40_rl3 $W/pools/rest80_rl3 \
  --ref $W/pools/holdout40_ctl $W/pools/rest80_ctl --budgets 60000,80000,100000 --policy repl_nostop --cap 81920
```

Rules that held up:
- compare at equal tokens;
- require the same sign on both halves;
- fix thresholds before the run;
- validate on the shipped build in the agent loop.

At ×8, a pool carries about ±3.5 points of noise.

Other tools:
- `eval/score.py` scores `harness/probe/run.py` results.
- `prune_gptq/eval_direct.sh` with `evstat.py` and `evci.py` runs direct-mode arms against an FP8 arm. Name the
  holdout40 FP8 arm `fp8ref`, and prefix 120-task arms with `e120_` (reference `e120_fp8`) and REPL arms with `r_`
  (reference `r_fp8`).
- `eval/fmt_check.py` reports tool-format errors and verified precision of a pool.

### 10. Distillation toward the FP8 model (not used)

Before RL, distillation was tried as a direct repair of k320q34's compression error: token-level KL toward the FP8
model on its own step-1 traces, with the same trainable set as RL. It removed 36% of the KL above the student's
forward-mismatch floor, but REPL pass@2 at 80k tokens was −3.9 [−11.1, +2.6] points against k320q34. The distilled
build verified a little more often (0.825 vs 0.788 of sessions) at lower precision (0.485 vs 0.536), after fewer tool
calls (7.3 vs 9.7 per session): it committed earlier to wrong programs. The bar, set before the run, was +2.5 points,
so the shipped model does not use it.

**Data.** The step-1 traces (`$W/traces`, `$W/traces_long`): 1,040 sequences on 480 training tasks, one direct and
one REPL attempt per test input. `student/prep_recs.py` (step 6.0) turns them into records:
- only generated tokens are trained: the prompt, tool responses and the `<|im_start|>assistant\n<think>\n` headers of
  later turns are masked;
- it stops if a record comes from an ARC-AGI-2 evaluation task;
- records of the 48 held-out capture tasks (step 2) form the validation split.

Training uses all 938 training-split records (469 REPL and 469 direct on 432 tasks; 6.1M and 11.8M generated
tokens), cut at 24,576 tokens. Validation uses the first 8 REPL and the first 8 direct held-out records in key order,
cut at 16,384 tokens (86k and 66k generated tokens).

**Teacher.** The FP8 model on stock vLLM 0.30, TP2 with expert parallelism and no MTP (`kd/serve_teacher.sh`).
`kd/dump.py` sends each record as token ids with `max_tokens=1` and `prompt_logprobs=20`, and saves the raw top-20
log-probs of every position after the prompt.

**Student and trainable set.** The step-6 student (`student/student_fn.py`) on k320q34's codes. As in RL, only the
log-offsets of the per-group scales of the kept experts (589.8M) and the kept router rows (39.3M) train. Norms are
supported (`--trainable scales,router,norms`), but serving them needs a loader hook, so they were off.

**Loss.** At every generated position, the forward KL(teacher ‖ student) over 21 bins: the teacher's top-20 tokens
plus the remaining mass. Tool-call spans (`<tool_call>` to `</tool_call>`) weigh ×2 and direct-mode sequences ×0.5.
Gradients are summed over the step and divided by its total weight. Validation reports the unweighted KL per
generated token and the top-1 agreement, REPL and direct separately.

**The run's settings.**
- 16 single-GPU data-parallel replicas with one sequence each: 16 sequences and about 180k generated tokens per step;
- sequences cut at 24,576 tokens; 150 steps (2.6 epochs);
- Adam with β = (0.9, 0.99) and ε = 1e-10 on fp32 masters; learning rate 5e-4 for the scales and 2e-5 for the router;
- 10 linear warm-up steps, then cosine with a floor at 5% of the peak;
- validation every 25 steps; seed 0.

`train_kd.py`'s defaults are these values except `--len` (32,768), `--steps` (300), `--val-n` (16) and
`--eval-every` (50), so the command passes the run's values. The batch order depends only on `--seed`: 8 replicas
with `--accum 2` see the same 16 sequences per step as the run's 16 replicas.

```bash
# 10.1 teacher dumps: four TP2 servers on 8 GPUs (wait until they answer); rerun dump.py to fill in failed requests
NGPU=8 bash training/kd/serve_teacher.sh &
python3 training/kd/dump.py $W/student/recs.jsonl $W/kd/teacher \
  --endpoints 127.0.0.1:8000,127.0.0.1:8001,127.0.0.1:8002,127.0.0.1:8003
# 10.2 floor: the student's forward with the original 512 FP8 experts per layer (2 GPUs, full checkpoint)
python3 training/kd/train_kd.py --out $W/kd/floor --src fp8 --gpus 0,1 --val-n 8 --eval-only
# 10.3 training on top of step 6.0 ($W/dense, n-gram table, records); every run logs k320q34 at step 0
KD_OUT=$W/kd/run KD_NREP=8 KD_ARGS="--len 24576 --steps 150 --accum 2 --val-n 8 --eval-every 25" \
  bash training/kd/train_kd.sh
python3 training/student/export_build.py $W/kd/run/trained.pt $W/builds/k320q34 $W/builds/k320q34_kd   # or KD_EXPORT
```

| Validation KL (held-out) | step 0 = k320q34 | 25 | 50 | 75 | 100 | 150 | floor |
|---|---|---|---|---|---|---|---|
| REPL | 0.0444 | 0.0348 | 0.0334 | 0.0327 | 0.0324 | 0.0324 | 0.0112 |
| direct | 0.0098 | 0.0078 | 0.0074 | 0.0072 | 0.0071 | 0.0071 | 0.0022 |

The floor is the student's own forward with the original FP8 experts. Above it, the KL fell by 36% on both REPL
(0.0332 → 0.0212) and direct sequences, with a plateau after about 75 steps. REPL top-1 agreement went from 93.1% to
93.9% (floor 96.4%).

**Evaluation.** The export changes only the trunk's fp16 scales and router rows, and its `ovr/` carries the trained
router rows that `plugin/fnplug.py` loads. Two checks:
- **Served KL.** No-MTP servers of both builds dump the held-out records, and `served_kl.py` compares each dump with
  the teacher's. A teacher token outside the build's top 20 gets the build's 20th log-prob, so the KL is a lower
  bound, computed the same way for both builds. On 10 held-out REPL records (71k generated tokens) it fell from
  0.0390 to 0.0271 (−31%, top-1 93.9% → 94.8%), in line with the −27% of the validation KL. Export and serving work
  as intended.
- **REPL gate.** Step-5 servers (MTP 3) and the step-9 tools with the harness settings of that time: default prompt,
  no `ak` library, holdout40 only, ×4 per seed. Seeds 2 and 3 for the distilled build, seeds 0 and 1 for k320q34.

```bash
# served KL (the published numbers: 10 records, see "Provenance and caveats")
bash training/kd/serve_nomtp.sh $W/builds/k320q34 0,1 8000 > $LOGDIR/nomtp_base.log 2>&1 &
bash training/kd/serve_nomtp.sh $W/builds/k320q34_kd 2,3 8001 > $LOGDIR/nomtp_kd.log 2>&1 &
V="--split val --max-len 16384"
python3 training/kd/dump.py $W/student/recs.jsonl $W/kd/served_base --endpoints 127.0.0.1:8000 $V
python3 training/kd/dump.py $W/student/recs.jsonl $W/kd/served_kd --endpoints 127.0.0.1:8001 $V
python3 training/kd/served_kl.py $W/kd/teacher $W/kd/served_base $W/kd/served_kd
# REPL gate (servers of the distilled build; the same pools from k320q34 servers go to $W/pools/h40_base_s0, _s1)
BUILD=$W/builds/k320q34_kd NGPU=4 bash training/serve/serve_node.sh &
for s in 2 3; do
  python3 training/harness/pool.py --endpoints 127.0.0.1:8000,127.0.0.1:8001 --model-dir $FN_BASE_MODEL \
    --split $ARC_DATA/evaluation --tasks @training/eval/holdout40.txt --arms repl:4 --concurrency 48 --seed $s \
    --out $W/pools/h40_kd_s$s
done
python3 training/eval/paired_diff.py --build $W/pools/h40_kd_s2 $W/pools/h40_kd_s3 \
  --ref $W/pools/h40_base_s0 $W/pools/h40_base_s1 --budgets 50000,80000,100000,118000
python3 training/eval/fmt_check.py $W/pools/h40_kd_s2 $W/pools/h40_kd_s3 $W/pools/h40_base_s0 $W/pools/h40_base_s1
```

| Distilled minus k320q34, holdout40 REPL ×8 (`repl_nostop`, cap 81,920) | 50k | 80k | 100k | 118k |
|---|---|---|---|---|
| pass@2 difference, points [95% CI] | −2.6 [−9.0, +3.2] | **−3.9 [−11.1, +2.6]** | −4.5 [−12.2, +2.6] | −4.3 [−12.3, +2.9] |

`fmt_check.py`: no-code tool-format errors fell from 0.9% to 0.1% of calls. Rest80 was not run.

**Hardware and time.**
- Teacher dumps: four TP2 servers on 8 H200, about 25 min for the 25.5M tokens (16-18k tokens/s).
- Training: 16 H200, one replica each. Peak memory 94-101 GB at 24,576 tokens. About 22 s per step, 63 min for the
  run including validation (2 replicas with `--accum 8`: about 2.5 min per step). A replica loaded in 27 s from
  node-local copies of `$W/dense` and the build's `lowbit-*`, `router-*` and `keep.*` files (the note in step 6
  applies). The n-gram table needs about 51 GB of RAM per machine.
- Export: CPU only, a few minutes. The served-KL check and the gate use 2-4 GPUs per build.

## Script reference

The Python files had their comments and docstrings removed for release, so this table is their documentation.
`--help` lists every option.

| Script | What it does / main arguments |
|---|---|
| `env.sh` | shared defaults; sourced by every shell script |
| `plugin/fnplug.py` (+ `fnplug-0.1.dist-info`) | vLLM general plugin, active when `plugin/` is on `PYTHONPATH`. `FN_CAP_DIR`: routing capture while `FN_CAP_DIR/on` exists (its text names the output subdir; `FN_CAP_P` = sampled share of tokens). `FN_KEEP`: load-time pruning to a keep set. `FN_OVR`: BF16 expert overrides (and trained router rows) per layer |
| `harness/pool.py` | sample pools: `--endpoints H:P,... --model-dir M --split DIR --tasks @FILE --arms repl:8[,direct:4] --prompt ak --seed N --out DIR` → `samples.jsonl` (per session: tokens, status, every executed program with its train/test results, answer) and `traces.jsonl.gz`. Resumable |
| `harness/qrepl/{session,sandbox,prompt}.py` | the REPL session (native tool-call format, sandboxed execution, nudge, forced answer), the sandbox (`python -I`, 2 GiB, CPU limit, no network; `QR_ARCKIT=lib|block|review`), prompts (`--prompt ak` = instructions + `ak` API sheet). The `QR_H2` review modes are off by default; they tested negative (Appendix D) |
| `harness/probe/run.py` | calibration harness: `--modes direct,repl --attempts 1 --max-tokens 65536 --out DIR` → `results.jsonl`, `traces.jsonl.gz` (prompt + completion); `direct.py` (prompt template and answer parser), `sandbox.py` (its tool sandbox) |
| `prune_gptq/traces.sh` | step 1 (env: `ENDPOINTS`, `OUT`, `NTASK`, `MAXTOK`, `CONC`, `GPU_PAIRS`, `SERVE_LEN`) |
| `prune_gptq/capture.sh` | step 2 server + replay (env: `TRACES`, `HO`, `TAG`, `SPLIT`, `CAP`, `CAP_LEN`, `REPLAY_LEN`, `GPUS`, `PORT`, `SERVE_ONLY`) |
| `prune_gptq/replay.py` | `--traces F... --endpoint H:P --model-dir M --cap DIR --holdout-file F [--tag T] [--split-pos P] [--max-len N] [--done-file F]` |
| `prune_gptq/replay_tc.py` | tool-call / tool-response (or `--header`) prefix replays: `--traces F... --endpoint --model-dir --cap [--max-calls 3] [--max-len]` |
| `prune_gptq/gptq_fn.py` | `keep --cap DIRS --out DIR --ks K,... --crit reapsum|wsum|freq`; `build --cap DIRS --keep FILE --tiers 160:4,all:3 --out DIR --layers 0-48 --gpu G [--rtn] [--no-ovr]`; `eval --cap DIRS --keep FILE --cand NAME=DIR,... --out stats.jsonl` |
| `prune_gptq/keep_protect.py` | `--cap CAPDIR --scores scores_reapsum.pt --ks 256,320,384 --out KEEPDIR [--pcall 0.98 --presp 0.95]` → `keep_mix{0.5,1,2}_{K}.pt`, `keep_tool_{K}.pt` |
| `prune_gptq/keep_hdr.py` | header protection: `--cap --scores --ks --php 0.9 --out KEEPDIR --tag hd9` |
| `prune_gptq/buildq.sh`, `mkreadme.py`, `lowbit_pack.py` | build queue (`"NAME:KEEP:TIERS ..."`; env `CAP`, `KEEP_DIR`, `BUILD_DIR`, `G0`, `G1`, `BATCH`); build README; planar packing used by `gptq_fn.py` |
| `prune_gptq/eval_direct.sh`, `evstat.py`, `evci.py` | direct-mode arms (env `ARMS="name:BUILD ..."`, `TASKS`, `MODES`, `ATT`) and their summary / paired bootstrap against FP8 (read `$FN_WORK/evals`) |
| `serve/serve_build.sh` | `serve_build.sh BUILD|fp8 GPUS PORT [vllm args]` (env `MTP`, `UTIL`, `LEN`, `SEQS`) |
| `serve/serve_node.sh` | env `BUILD`, `NGPU`, `TP`, `GPU0`, `PORT0`: `NGPU/TP` servers with seeds 4000+i |
| `student/student_fn.py` | the differentiable model (`Student(model_dir, build, gpus, trainable=("scales","router"))`, `.hidden(ids)`) |
| `student/make_dense.py [OUT]`, `ple_stage.py [DIR]` | dense-only checkpoint copy; n-gram table to RAM |
| `student/export_build.py TRAINED.pt SRC_BUILD OUT [--no-ovr]` | new build with fp16 scales s₀·e^t and trained router rows; L48 and `keep.*` hard-linked; `ovr/` with BF16 experts + router rows; `DONE` last |
| `student/prep_recs.py OUT.jsonl TRACES... `, `check_hf.py --layers 4 --T 3000 [--recs F]` | tokenized trace records; student-vs-transformers exactness check (2 GPUs) |
| `rl/rollout.py` | `pool.py` options plus `--exclude FILES`, `--loo` (all leave-one-pair-out variants, tasks with ≥ 3 pairs), `--loo-one` (one variant per task), `--aug N` / `--aug-offset K` (D8 + colour-permuted copies); records the sampled spans |
| `rl/rollout.sh`, `rl/train_rl.sh` | rollout wrapper (env `RL_EPS`, `RL_OUT`, `RL_ARGS`); trainer launcher + export (env `RL_OUT`, `RL_ARGS`, `RL_NREP`, `RL_GPR`, `RL_BUILD`, `RL_EXPORT`, `DENSE`, `RL_NNODES`, `RL_NODE_RANK`, `MASTER_ADDR`) |
| `rl/prep_rl.py OUT ROLL...` | `--rewards vr,vw,ur,uw --max-len --min-group --budget-aware BA --ba-floor --require-mixed --concise K --exclude FILES --only FILES` → `<key>.npz` (ids, trained positions, advantage), `index.jsonl`, `stats.json` |
| `rl/train_rl.py` | `--data DIRS --build DIR --model DENSE --out DIR --per-step 64 [--steps N] [--resume trained.pt] --lr-scales --lr-router --warmup` → `trained.pt`, `log.jsonl` |
| `rl/roll_stats.py DIRS [--informative OUT]` | verified rate, precision, solve rate and outcome classes; writes tasks not all verified-right |
| `rl/make_pool.py`, `make_pool_hard.py`, `make_pool_comp.py` | public RL pools with the overlap filter (instances; hardest decile; rule compositions; `make_pool_comp.py` imports RE-ARC's generators and verifiers, run with `python3 -I`) |
| `rl/audit_lp.py` | serving-vs-trainer log-prob audit on prepared records |
| `draft_head/prep.py OUT ROLL...` | rollout dirs → token records with a sampled-token mask; held-out split by base task |
| `draft_head/dump.py`, `overlay_env.sh` | target dump in the deployed stack (`--recs --out --split tr|ho --n --skip --bt --util --kv-skip --draft-check`) |
| `draft_head/mtp_torch.py`, `accept.py`, `train.py` | the PyTorch MTP layer; offline acceptance (`--dump DIR --k 3|4 --delta CKPT`); training (options in step 8.4) |
| `draft_head/export.py make|build|check`, `check_exact.py`, `vacc.py`, `test_mtpft.py`, `vocab_stats.py` | delta export/rebuild/check; exactness against vLLM's drafter; closed-loop acceptance; unit tests; hot vocabulary |
| `eval/score.py`, `policy_sim.py`, `paired_diff.py`, `fmt_check.py`, `cmp_builds.py`, `holdout40.txt` | see step 9 and step 4 |
| `kd/serve_teacher.sh` | step 10 teacher: `NGPU/TP` servers of the FP8 model without MTP, `--max-logprobs 20`, seeds 3000+i (env `NGPU`, `TP`, `GPU0`, `PORT0`, `SEQS`, `MBT`) |
| `kd/dump.py RECS OUT --endpoints H:P,...` | top-20 prompt log-probs of each record → `OUT/<key>.npz` (`ids`, `plen`, `gen` spans, `top_ids`, fp16 `top_lp` for every position after the prompt); `--split train,val --mode repl,direct --max-len 65536 --conc 3` (per endpoint); skips existing outputs |
| `kd/train_kd.py` | `--data DUMPS --recs RECS --build DIR --model DENSE --out DIR --len --steps --accum --val-n --eval-every [--resume trained.pt] [--eval-only] [--src fp8]` → `trained.pt` (saved at every validation), `log.jsonl` |
| `kd/train_kd.sh` | launcher + export (env `KD_OUT`, `KD_ARGS`, `KD_NREP`, `KD_GPR`, `KD_BUILD`, `KD_EXPORT`, `DENSE`, `KD_NNODES`, `KD_NODE_RANK`, `MASTER_ADDR`) |
| `kd/serve_nomtp.sh BUILD GPUS PORT`, `kd/served_kl.py TEACHER_DIR BUILD_DIR...` | served-KL check: one no-MTP server of a build; per-mode KL and top-1 agreement of each build's dump against the teacher's |

## Provenance and caveats

- Part of the record cannot be reproduced exactly:
  - **The 160 harder calibration tasks.** These were the highest-weighted training tasks (outside the first 320) of
    a per-task difficulty list kept from earlier work; that list is not part of this release. Their role is
    long-context calibration, so any 160 hard training tasks outside the first set serve it.
  - **The round-3 rollouts** were stopped after a fixed time (3,759 samples).
  - **Sampling** on GPU servers is not deterministic.
- The shipped artifacts are:
  - the base build k320q34;
  - the round-3 delta (rl3 against k320q34);
  - the draft-head delta (made against rl1; only layer-48 files change, and they are identical in every build);
  - `hot32k_tr.pt`.

  `../kaggle/` explains how the notebook assembles them.
- Names on Kaggle: the base build k320q34 is `k320-q34-p1`…`p4`. rl1, rl2 and rl3 are the variations named `r1b`,
  `r8` and `r11c` (`k320-q34-r1b-d-*`, `k320-q34-r8-d-*`, `k320-q34-r11c-d-*`; the final submissions RC8 and RC9 use r11c), and
  the draft head is `k320-q34-r1b-mtp2-p1`.
- A delta only applies to the build it was made against (`delta_build.py` checks the expert ids). The traces and
  captures are sampled, so a rebuilt k320q34 differs from the published one and needs its own RL rounds and draft
  head. To start from the shipped weights, use the published base build instead (see `../kaggle/`).
- Step 10 (distillation, not shipped):
  - This code was checked against the run's files. `prep_recs.py` rebuilds the records byte for byte, and
    `export_build.py` on the run's final `trained.pt` rebuilds every file of the distilled build except
    `README.md`. `served_kl.py`, `paired_diff.py` and `fmt_check.py` give the numbers above from the run's dumps
    and pools. Serving the distilled build with `serve_nomtp.sh` and dumping the same 10 records again gave a served
    KL of 0.0270 (published: 0.0271). Two replicas with `--accum 8` train on the run's 16 sequences per step: the
    same token counts per step, and a step-1 KL of 0.02789 against the run's 0.02795.
  - The student's forward is not bit-reproducible on GPU, so the validation KL of the same weights moves in the
    fourth decimal between runs. Reruns of this code and of the run's own code gave a REPL KL of 0.04411-0.04445
    for k320q34 (logged: 0.04436) and 0.03224-0.03234 for the final weights (logged: 0.03236). With PyTorch's
    deterministic algorithms on (`torch.use_deterministic_algorithms(True, warn_only=True)` and
    `CUBLAS_WORKSPACE_CONFIG=:4096:8`), four reruns on k320q34, two with each code, gave identical numbers.
  - Teacher dumps are not bit-reproducible either. Re-dumping 4 records (2 REPL, 2 direct) on a fresh server gave
    97.8% top-1 agreement with the original dumps and a KL of 0.0078 (REPL) and 0.0020 (direct) between the two.
    That is close to the student's floor, which suggests that much of the floor is the teacher's own noise.
  - The gate's k320q34 pools were sampled 10-12 hours before the distilled build's, not at the same time as step 9
    asks.
  - All four gate pools used Kaggle's copy of the evaluation tasks (the competition's evaluation challenges with their
    solutions, one file per task). It differs from the GitHub copy in 3 of the 40 holdout tasks: `abc82100`,
    `d8e07eb2` and `faa9f03d`.
  - The published served-KL numbers cover the first 10 held-out records that both dumps finished: the run's
    `dump.py` stopped at a record whose prompt is longer than `--max-len`. The released `dump.py` skips such records,
    so a rerun covers the others too.
- Third-party code: `harness/probe/direct.py` contains the prompt template and two answer parsers of
  github.com/arcprize/arc-agi-benchmarking (MIT License). `rl/make_pool_comp.py` imports RE-ARC's generators and
  verifiers at run time (Apache-2.0; not included).
