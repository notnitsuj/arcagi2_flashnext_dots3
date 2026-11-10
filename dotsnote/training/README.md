# Training: from the dots3-note preview to lb144r and p1s

This folder holds the code that turned the public dots3-note preview (`dots-studio/dots3-note-prev-fp8`, Apache-2.0)
into the two artifacts the Dots3-Note notebook loads, plus the evaluation used to accept them:

- **lb144r**, the checkpoint: 144 of the 256 routed experts kept in each of the 45 MoE layers, GPTQ-quantized to 3 bits
  (96 experts per layer) and 2 bits (48 per layer), 54.6 GB of experts. All other weights are the preview's own,
  unchanged.
- **p1s**, the fine-tune: masked-cross-entropy SFT of lb144r on its own shortest correct answers to ARC-AGI-2 training
  tasks. It trains only the GPTQ scales, the RMSNorms and the router, and ships as a 1.18 GiB overlay,
  `p1s_delta.safetensors`, that reproduces the fine-tuned checkpoint bit for bit at load time.

The writeup describes this model in section 3. The engine (the `arcstats` vLLM plugin, its kernels and the vLLM 0.30
patch) is in [`../engine/`](../engine/), and the notebook, its runtime dataset and the tools that package the Kaggle
artifacts are in [`../kaggle/`](../kaggle/README.md). This folder refers to them instead of copying them.

```
FP8 preview ──(1) its own answers to 70 ARC-AGI-2 training tasks: 66 traces, 0.92M tokens
            ──(2) MoE inputs of the first 16k tokens of the 16 longest traces (253,573 rows per layer)
              ──> reapsum score per expert: sum over calibration tokens of gate weight × ‖expert output‖
            ──(3) keep the top 144 of 256 per layer ──(4) GPTQ: top 96 at 3 bits, next 48 at 2 bits, group 128
            ──(5) assemble: dense FP8 safetensors + lowbit/lowbit-L01..45.pt + pruning.json            = lb144r
lb144r      ──(7) its own direct answers to 720 training tasks ──> shortest correct trace per test input (638)
            ──> SFT of GPTQ scales + RMSNorms + router, 2 epochs ──> 1.18 GiB overlay, checked bit-exact = p1s
every stage ──(6) served with the 4×L4 numerics on H100s ──(8) holdout40 / training-split validation, paired runs
```

## What you need

**Hardware.** We used H100 80 GB GPUs (and H100 NVL 94 GB for the unpruned model).

| Step | GPUs |
|---|---|
| 1, 2: serving the unpruned FP8 preview (traces, MoE-input capture) | one server for the whole model: 4 GPUs with at least 94 GB (TP4), or 8 × 80 GB (TP8) |
| 2: reapsum scores; 4: GPTQ | 1 GPU per process (layer ranges can run in parallel processes) |
| 3, 5: pruning and assembly | CPU only |
| 6, 8: evaluation servers (lb144r with the 4×L4 numerics) | holdout40: 4 GPUs (TP4, 26 GiB budget each); validation runs and fine-tune data: 2 GPUs (TP2, 66 GiB budget) |
| 7: fine-tune training | 3 × H100 80 GB, one layer pipeline (peak 61–67 GB per GPU): 9.8 h for the shipped run |
| 10 (not used): distillation | teacher and on-policy student servers: 4 × H100 80 GB (TP4); training: one layer pipeline, 5 × H100 80 GB in our rounds, 4 with this folder's student |

- **Host memory.** The evaluation server keeps the sparse-attention KV rows in shared, pinned host memory
  (`/dev/shm`). We capped each server at 300 GB of host RAM.
- **Disk.** The FP8 preview is 279 GB as downloaded (vision and audio towers included). The captured MoE inputs take
  117 GB, lb144r 66 GB, and an optional full copy of the fine-tuned checkpoint (for the bit-exact check) another 66 GB.

**Software.** Use the environment of the submission, built as described in [`../kaggle/README.md`](../kaggle/README.md)
§5.2 (`build/build_env.sh`). It holds Python 3.12, vLLM 0.30.0 with [`../engine/vllm_patch`](../engine/vllm_patch),
PyTorch 2.13.0 for CUDA 13.0, Triton 3.7.1, FlashInfer 0.6.18.post1, transformers 5.18.0, safetensors 0.8.0, numpy
2.3.5 and httpx 0.28.1.
- The serving steps need the `arcstats` plugin installed with its compiled kernels. Build them with
  `../engine/build_kernels.sh`, which also produces the sm_90 code that H100s run, then install
  `../engine/arcstats_pkg` (`pip install -e`, so that the kernels next to the modules are found). vLLM loads the plugin
  through its entry point.
- `python3` and `vllm` on the `PATH` must be that environment's. `serve/serve.sh` needs a CUDA 13 toolkit in
  `CUDA_HOME` (else it looks in `/usr/local/cuda*`) for the kernels that vLLM compiles at start.
- The fine-tune trainer (`finetune/train.py`, `finetune/student.py`) needs only PyTorch, safetensors and numpy. It does
  not import vLLM.

**Model and data.**
- `dots-studio/dots3-note-prev-fp8` from Hugging Face goes in `$D3_BASE_MODEL`.
- ARC-AGI-2 (github.com/arcprize/ARC-AGI-2; we used commit `f3283f7`): `$ARC_DATA` must contain `training/` and
  `evaluation/`.
- **Only training tasks are used for calibration and fine-tuning.** This is checked as follows:
  - The calibration task lists come from the training directory: they are the first 40 and the next 30 ids of
    `random.Random(7).sample(sorted(training ids), 100)`. The samplers read them with `--split training`, which only
    opens `$ARC_DATA/training/<id>.json`. Step 1 starts with a one-line check that no listed id is an evaluation id.
  - `finetune/split.py` re-derives the fine-tune task split from the training ids and asserts that it contains no
    evaluation id and that its validation and generation sets are disjoint.
  - `finetune/select_sft.py` asserts that no selected trace belongs to a validation or evaluation task. It also drops
    every trace whose task shares an input grid (exact hash) with a validation or evaluation task; 9 traces were
    dropped.
  - The distillation of step 10 (not used) samples and scores training tasks only. `distill/tasks.py` and
    `distill/teacher_dump.py` assert that no evaluation id is used.
  - Apart from these exclusion checks, evaluation tasks are read only by the evaluation tools: holdout40, the prompts
    of the degeneration test, and the optional KL bench, whose traces the trainer can also score for measurement
    (`--fid`, step 10).

### Environment variables

Every script reads these. The Python defaults equal the shell defaults in `env.sh`. Relative paths resolve against
the directory you run from, and all commands below run from the `dotsnote/` folder after `source training/env.sh`.

| Variable | Default | Meaning |
|---|---|---|
| `D3_BASE_MODEL` | `models/dots3-note-prev-fp8` | local copy of the FP8 preview (weights, config, tokenizer) |
| `D3_WORK` | `work` | all outputs: `calib/`, `calib_acts/`, `reap_scores.pt`, `stats_reapsum.pt`, `lb144r/`, `gptq/`, `ft/`, `gate/`, `hold/`, `overlay/`, `fid/`, `masks/`, `w8a8_cfg/`, `kd/`, `logs/` |
| `ARC_DATA` | `data/ARC-AGI-2/data` | ARC-AGI-2 `training/` and `evaluation/` |
| `D3_LB144R` | `$D3_WORK/lb144r` | the lb144r checkpoint that the fine-tune, the overlay export and the evaluation commands start from (the one built in step 5, or the published one: see step 5) |
| `D3_ENGINE` | `../engine` (next to this folder) | the folder that holds `arcstats_pkg/`; `gptq_experts.py` imports the packer and `export_delta.py` the overlay from it |
| `LOGDIR` | `$D3_WORK/logs` | logs of the shell scripts |
| `D3_LOOP_DETECT_DIR` | `../kaggle/runtime_dataset` | folder with `loop_detect.py`, the solver's loop detector, which `select_sft.py` applies to candidate traces |
| `D3_FID_TRACES` | `$D3_WORK/fid/fp8_text.jsonl` | KL bench only: the FP8 preview's own reference traces |
| `ARC_CALIB_DIR` | unset | engine plugin: record MoE inputs and routing (step 2) |
| `ARC_DELTA_FILE` | unset | engine plugin: load a fine-tune overlay on top of the checkpoint being served |
| `ARC_EXPERT_MASK` | unset | engine plugin: emulate a keep set at run time on the unpruned model (KL bench of pruning alone) |
| `DISTILL_NESTED_CKPT`, `DISTILL_NESTED_MIN` | `0`, `0` (`1`, `24576` in `finetune/train.sh`) | trainer: nested activation checkpointing (attention, MLP, each expert) for sequences of at least this many tokens |

Script-specific variables (GPU ids, ports, budgets) are listed in each script's header and in the reference table at
the end.

## Recipe

Settings below are those of the shipped artifacts. Sampling on GPUs is not bit-reproducible, so a rerun gives
statistically equivalent rather than identical traces, rankings and checkpoints. Wherever we could, we checked the
tools against the shipped files; the results are given in each step.

### 0. Setup

```bash
cd dotsnote
source training/env.sh                   # D3_BASE_MODEL, D3_WORK, ARC_DATA, D3_LB144R, D3_ENGINE, LOGDIR
W=$D3_WORK; T=training
# wait until a server log says it is up
up() { until grep -a -q -E 'Application startup complete|Traceback' $1; do sleep 15; done; grep -a -q 'startup complete' $1; }
```

### 1. Calibration traces

The FP8 preview answers 70 training tasks once each.
- **30 tasks in direct mode** (`harness/direct.py`): the ARC Prize benchmark prompt, thinking on, T 0.7, top-p 1.0, no
  practical cap (384k tokens). Our run stopped with 26 of the 32 test inputs answered (25 tasks); the longest trace has
  103,717 completion tokens.
- **40 tasks with a program-synthesis prompt** (`calib/synth.py`): think for up to 12,000 tokens, then answer in up to
  4,000; T 1.0, top-p 0.95, one attempt per task.
- `calib/make_mix.py` concatenates the 26 + 40 traces (prompt + completion) into `calib_mix.jsonl`: 922,665 tokens. On
  our two trace files it reproduces the shipped file byte for byte.

```bash
for t in $(cat $T/calib/tasks_*.txt); do [ -e $ARC_DATA/evaluation/$t.json ] && echo "EVALUATION TASK $t"; done
mkdir -p $W/calib
bash $T/serve/serve.sh 0,1,2,3 8010 $D3_BASE_MODEL --tensor-parallel-size 4 --language-model-only --trust-remote-code \
  --disable-custom-all-reduce --max-model-len 400000 --attention-config '{"flash_attn_version": 2}' > $LOGDIR/fp8.log 2>&1 &
up $LOGDIR/fp8.log
python3 $T/harness/direct.py --ports 8010 --split training --tasks @$T/calib/tasks_direct30.txt --attempts 1 \
  --out $W/calib/direct.jsonl --save_text $W/calib/direct_text.jsonl
python3 $T/calib/synth.py --kind qwen --ports 8010 --split training --tasks @$T/calib/tasks_synth40.txt --think 12000 \
  --answer 4000 --experts 1 --iters 1 --save_text --tokenizer $D3_BASE_MODEL --out $W/calib/synth.jsonl
python3 $T/calib/make_mix.py $W/calib/calib_mix.jsonl $W/calib/direct_text.jsonl $W/calib/synth.jsonl
bash $T/serve/kill_server.sh 8010 0,1,2,3
```

- `--kind qwen` selects `synth.py`'s two-phase completion path (think until `</think>`, then answer), which works with
  any chat template.
- The server flags of our calibration run were not recorded. The command uses the flags of our other FP8 servers and a
  context long enough for `direct.py`'s default 400,000-token budget. On 80 GB GPUs, serve on 8 GPUs
  (`0,1,2,3,4,5,6,7`) with `--tensor-parallel-size 8`.
- Our traces were sampled before we found that FlashAttention 3's sliding-window path is wrong for head sizes above 128
  on H100. The command forces FlashAttention 2, which is exact.

### 2. MoE-input capture and reapsum ranking

**Capture.** An eager server of the unpruned FP8 preview runs the capture hook of the engine plugin (`ARC_CALIB_DIR`).
On tensor-parallel rank 0 it saves, for every MoE layer and every prefill chunk longer than 64 tokens, the layer's
input rows (bf16) and their top-8 expert ids. `prune_gptq/calib_prefill.py` sends the first 16,384 tokens of the 16
longest traces of `calib_mix.jsonl` as prompts with `max_tokens=1`. That is 253,573 rows per layer and 117 GB on disk.
Prefix caching is off, so every token is computed and recorded; the shipped capture holds all 253,573 tokens.

```bash
ARC_CALIB_DIR=$W/calib_acts bash $T/serve/serve.sh 0,1,2,3 8040 $D3_BASE_MODEL --tensor-parallel-size 4 \
  --language-model-only --trust-remote-code --disable-custom-all-reduce --enforce-eager --no-enable-prefix-caching \
  > $LOGDIR/capture.log 2>&1 &
up $LOGDIR/capture.log
python3 $T/prune_gptq/calib_prefill.py 8040 --src $W/calib/calib_mix.jsonl     # prints the running token count
bash $T/serve/kill_server.sh 8040 0,1,2,3
```

**Ranking.** `reap_scores.py` recomputes the routing of every captured row from the preview's own router: bf16 logits,
sigmoid scores, top 8 of score + correction bias, weights renormalized over the 8. It agrees with the captured ids on
every row. It then runs every routed expert (FP8 weights dequantized to bf16) on the rows routed to it, and stores per
expert the count, the gate-weight sum, the summed output norm and REAP = the mean of gate weight × ‖output‖₂ over the
expert's rows.

The shipped ranking is **reapsum = REAP × count = Σ_t g_{t,e} · ‖f_e(x_t)‖₂** over the calibration tokens routed to
expert e. `reapsum_stats.py` writes it as `wsum` in the routing-statistics format that `prune.py` and
`gptq_experts.py` rank by.

```bash
python3 $T/prune_gptq/reap_scores.py --calib $W/calib_acts --out $W/reap_scores.pt --gpu 0    # resumable, layer by layer
python3 $T/prune_gptq/reapsum_stats.py --reap $W/reap_scores.pt --out $W/stats_reapsum.pt
```

The shipped ranking file also carried the routing counts of an earlier routing-statistics capture. `--template FILE`
keeps those fields, but neither tool reads them. On the shipped `reap_scores.pt`, `reapsum_stats.py` reproduces the
shipped ranking exactly, with or without the template.

### 3. Keep 144 experts per layer

```bash
python3 $T/prune_gptq/prune.py --stats $W/stats_reapsum.pt --keep 144 --no-experts --out $W/lb144r
```

- `prune.py` keeps the top 144 experts by reapsum in each of the 45 MoE layers (layer 0 is dense). It prints the kept
  share of reapsum: minimum 0.845, mean 0.959.
- It writes the dense part of the checkpoint and `pruning.json` (`{"keep": {layer: [kept ids]}, "coverage": {layer:
  kept share}}`). On the shipped ranking it reproduces the shipped `pruning.json` byte for byte.
- At serving time, routing is the model's own (sigmoid, correction bias, top 8, renormalized) over the 144 kept
  experts.
- The same uniform 144 was chosen with the KL bench (step 8): at equal bytes, 144 and 160 experts were equally good,
  176 and 192 worse.

### 4. GPTQ: 96 experts at 3 bits, 48 at 2 bits

```bash
python3 $T/prune_gptq/gptq_experts.py $W/gptq --tiers 96:3,144:2 --stats $W/stats_reapsum.pt --calib $W/calib_acts \
  --pack $W/lb144r/lowbit --no-fp8 --layers 1-45 --gpu 0
# or in parallel: --layers 1-23 --gpu 0 and --layers 24-45 --gpu 1 (finished layers are skipped)
```

- **Bits by reapsum rank.** In every layer the 96 experts with the highest reapsum get 3 bits and the next 48 get 2
  bits; ranks 144 and above are not quantized, because they are pruned. We checked all 45 shipped layers: 4,320
  experts at 3 bits and 2,160 at 2 bits, exactly by this rule.
- **Grid.** Static groups of 128 input columns, asymmetric (fp16 scale, uint8 zero). The clip ratio is searched by MSE
  over 10 values from 1.0 to 0.55, on the original weights.
- **GPTQ.** Activation order; damping mean(diag H) · (0.01 + 4 / n_rows); blocks of 128 columns. H = XᵀX over the
  calibration rows routed to the expert (unweighted). Gate and up share one Hessian and are quantized as one
  [3,072 × 5,120] matrix. The down projection's Hessian is built from the activations of the already-quantized
  gate/up. 32 experts are quantized as one batch.
- **Packing** (`--pack`) writes `lowbit-L{01..45}.pt`, one file per layer (1,212,384,361 bytes each, 54.6 GB in all):
  - `{"experts": [kept ids, ascending], id: {"bits", "gu", "gu_s", "gu_z", "d", "d_s", "d_z"}}`, where `gu` holds the
    rows [gate 1,536; up 1,536] × 5,120 and `d` holds 5,120 × 1,536. The weight is w = scale · (q − zero), with scales
    and zeros of shape [rows, columns/128].
  - The codes are in the planar layout of `../engine/arcstats_pkg/arcstats/lowbit_moe.py`. Per row and group of 128
    columns, `p2` holds 16 int16 words; field j (2 bits, j < 8) of word w is column 16j + w. For 3-bit experts, `p1`
    holds 16 bytes; bit j of byte w is the third bit of column 16j + w.
- `--no-fp8` skips the fake-quantized FP8 copies of the experts, which are only for evaluation with the stock loader.
  Without `--pack`, the script writes those copies and the unpacked codes, and `--assemble` turns them into a
  checkpoint that stock vLLM loads.

### 5. The assembled checkpoint

Steps 3 and 4 produce lb144r:

```
lb144r/
  model-00000..00002.safetensors, model.safetensors.index.json   11.2 GB: every weight except the routed experts
  config.json                     the preview's config with n_routed_experts = 144 and arc_lowbit = "lowbit"
  pruning.json                    kept expert ids and kept reapsum share per layer
  lowbit/lowbit-L01..L45.pt       54.6 GB of 3/2-bit experts (step 4)
  tokenizer and chat-template files, generation_config.json, Dots Studio's README and LICENSE (copied unchanged)
```

- Every non-expert tensor is the preview's own, byte for byte: block-FP8 linears with their `weight_scale_inv`, bf16
  norms and embeddings, the shared expert, and the MTP layer (index 46, which has a dense MLP).
- The router rows (`mlp.gate.weight`, `mlp.gate.e_score_correction_bias`) are sliced to the kept experts.
- The vision and audio towers are dropped; serve with `--language-model-only`.
- The `arcstats` plugin builds the MoE layers of an `arc_lowbit` checkpoint from the low-bit files.

The published Kaggle Model has the same files in a flat layout: the low-bit files sit next to `config.json`, and
`arc_lowbit` is `"."`; with that one change, `config.json` is byte-identical to the published 5,001-byte file. The
commands below produce that layout; `../kaggle/build/split_model_parts.py` then splits it into the four published
variations (`../kaggle/README.md` §5.3):

```bash
python3 - <<'PY'
import json, os
W = os.environ["D3_WORK"]; src, dst = f"{W}/lb144r", f"{W}/lb144r_flat"
os.makedirs(dst, exist_ok=True)
for n in os.listdir(src):
    if os.path.isfile(f"{src}/{n}") and n != "config.json": os.link(f"{src}/{n}", f"{dst}/{n}")
for n in os.listdir(f"{src}/lowbit"): os.link(f"{src}/lowbit/{n}", f"{dst}/{n}")
cfg = json.load(open(f"{src}/config.json")); cfg["arc_lowbit"] = "."
json.dump(cfg, open(f"{dst}/config.json", "w"), indent=2)
PY
```

**Starting from the published checkpoint instead.** The four variations, merged into one folder, are a valid
`$D3_LB144R` for the fine-tune (`student.py` and `export_delta.py` follow `arc_lowbit`). `finetune/build_ckpt.sh`
expects the `lowbit/` layout, so move the low-bit files into `lowbit/` and set `arc_lowbit` back to `"lowbit"` before
using it. The traces and captures are sampled, so a rebuilt lb144r differs from the published one, and the shipped
p1s overlay only applies to the published one (step 7).

### 6. Serving a checkpoint for evaluation

`serve/serve_eval.sh` serves lb144r on H100s with the numerics of the 4×L4 submission:
- portable sparse attention with the sparse-aware host KV offload;
- W8A8 block-FP8 dense GEMMs (tiles in `serve/w8a8_tiles.json`, the ones our evaluation runs used) and packed
  sliding-window attention;
- the low-bit MoE kernels, the host all-reduce, the SM89 code paths, and FP8 KV;
- a per-GPU memory budget (`L4_GIB`).

MTP-3 drafts tokens unless `MTP=0`. `ARC_DELTA_FILE` adds a fine-tune overlay.

```bash
L4_GIB=26 EMULATE_GPU_GIB= bash $T/serve/serve_eval.sh 0,1,2,3 8155 $D3_LB144R > $LOGDIR/eval.log 2>&1 &     # holdout setting
ARC_DELTA_FILE=$W/ft/run/delta.safetensors L4_GIB=66 KLOC=2048 EMULATE_GPU_GIB= \
  bash $T/serve/serve_eval.sh 2,3 8496 $D3_LB144R --seed 6161 > $LOGDIR/eval_p1s.log 2>&1 &                    # lb144r + p1s
```

The evaluation scripts in step 8 start and stop their own servers with these settings.

### 7. The p1s fine-tune

**7.1 Task split.** [`finetune/split.json`](finetune/split.json) is the shipped split of the 1,000 training tasks:
- 80 tasks excluded (the tasks of an earlier pilot run);
- a 200-task validation shard, `val` = `sorted(random.Random(20261013).sample(pool, 200))`, where the pool is the
  other 920 tasks;
- the other 720 tasks for generation, `gen`.

Each task carries a difficulty stratum: `easy` if a sampling run of an earlier lb144r variant solved it at least once
at a 14k-token cap, `hard` if that run never did, and `unk` if it was not sampled. That run is not part of this
release, so the strata are shipped as data. `split.py` re-derives `val` and `gen` from the training ids and the 80
excluded tasks, asserts that they equal `split.json`, and links the task files into `$W/ft/{gen_easy,gen_hard,gen_unk,val}/`.

```bash
python3 $T/finetune/split.py                          # 196 easy / 205 hard / 319 unk generation tasks; 200 validation
```

**7.2 Generation.** lb144r (TP2, evaluation numerics, server seed 5151) answers the generation tasks in direct mode,
on their real test inputs, at T 0.7 with a 32,768-token cap: hard tasks ×4, easy ×2, unk ×2, and 2 more for every unk
task that was not solved 2/2. The shipped run made 2,240 attempts, 1,393 of them correct.

```bash
bash $T/finetune/gen.sh                               # GPUS=0,1; -> $W/ft/gen_{hard,easy,unk,unk2}.jsonl (+ _text.jsonl)
```

**7.3 Selection** (`finetune/select_sft.py`). A correct attempt is a candidate if:
- it finished naturally (`finish == stop` and `</think>` closed);
- it contains no "time is up" text;
- the loop detector does not fire on it;
- its task shares no input grid with a validation or evaluation task.

The selection then keeps the single shortest candidate per (task, test input) and drops sequences over 40,960 tokens.
Each sequence is the prompt, the completion and `<|endofassistant|>`. The loss covers the completion tokens only.
Sequences of the harder half get weight 2: `hard` tasks plus `unk` tasks not solved 2/2 in the first round.

| | Sequences | Mean completion tokens | Weight |
|---|---|---|---|
| harder half | 214 | 16,834 | 2 |
| easier half | 424 | 5,731 | 1 |
| **total** | **638** on 599 tasks | 9,455 (median 6,132, max 32,177) | |

The shipped set has 6,032,341 completion tokens and 2,046,955 prompt tokens. 1,384 of the 1,393 correct attempts were
candidates; 9 were dropped for grid overlap and none for loops or forced text. Where an input has 2 or more candidates
(573 inputs), the shortest is 0.77× their mean length. On the shipped generation files, `select_sft.py` reproduces all
638 sequences and the manifest exactly.

```bash
python3 $T/finetune/select_sft.py                     # -> $W/ft/sft/<task>_<test>.npz (ids, plen, w) + manifest.json
```

**7.4 Training** (`finetune/train.sh` → `train.py` + `student.py`).

`student.py` re-implements the Dots3-Note language model in plain PyTorch, with lb144r's exact formats:
- MLA attention with the sparse indexer's top-2,048 selection, as a token mask over flex attention; the selection
  itself gets no gradient;
- sliding-window MLA (window 513);
- the sigmoid router with the correction bias;
- the 144 low-bit experts, rebuilt from the packed codes in forward and backward by a custom autograd function;
- the frozen block-FP8 linears with FP8-quantized activations and FP8-rounded KV latents, as served.

The settings:
- **Trainable** (1,228,137,856 parameters): a log-offset t on every expert GPTQ scale (s = s₀·eᵗ; 1,194,393,600
  values), the RMSNorms (566,656) and the router rows (33,177,600). Norms and router get fp32 masters. Codes, zeros and
  every other weight are frozen.
- **Loss:** cross-entropy summed over completion tokens, times the sequence weight.
- **Optimizer:** Adam, β = (0.9, 0.99), ε = 1e-12. Learning rates 1.6e-4 for the scales and 5e-6 for norms and router;
  10 warm-up steps, then cosine to 0.
  - 1.6e-4 on log-scales is a relative step about equal to 5e-6 on the raw expert weights (rms ≈ 0.031).
  - A smaller step would be rounded away when the scales are exported to fp16.
- **Steps:** one sequence per step, 2 epochs.
  - The trainer holds out the 17 sequences of 16 tasks (`--val-n 16`, seed 0) for a cross-entropy curve.
  - That leaves 621 training sequences, 1,242 steps and 5,897,527 completion tokens per epoch.
- **Sequences:** up to 40,960 tokens, right-padded to multiples of 2,048. Nested activation checkpointing applies
  from 24,576 tokens.

The shipped run took 9.8 h on 3 H100 80 GB GPUs (about 450 tokens/s, 28 s per step). Held-out cross-entropy went from
0.1355 to 0.1291 in the first 100 steps and stayed flat (0.1293 at the end); train cross-entropy averaged 0.117 and
0.105 over the two epochs. p1s is the final state (step 1,242). Run with `--eval-only` on the shipped SFT set, this
folder's trainer reproduces the shipped run's held-out split and its step-0 cross-entropy to the last digit
(0.13551295847211323 over 134,814 tokens).

```bash
GPUS=0,1,2 bash $T/finetune/train.sh $W/ft/run        # -> $W/ft/run/{log.jsonl, val_tasks.txt, trained.pt (4.9 GB), ckpt/}
```

`train.sh` ends with `build_ckpt.sh`, which writes a full checkpoint, `$W/ft/run/ckpt` (66 GB):
- new fp16 scales s₀·eᵗ, computed one expert at a time in fp32;
- the trained norms and router rows written into copies of the shards.

**7.5 Overlay export.**

```bash
python3 $T/finetune/export_delta.py $W/ft/run          # -> $W/ft/run/delta.safetensors (1.18 GiB); shipped as p1s_delta.safetensors
```

- **Scales:** per layer, the bit difference between the new and the base fp16 scales, in fp16 ulps, as int8.
  1,090,095,875 of the 1,194,393,600 scales change (91.3%).
- **Dense tensors:** the 276 trained norm and router tensors, as full bf16 replacements.
- **Guards:** per layer, the expert ids and a checksum of the base scales, so that the engine (`arcstats/delta_overlay.py`,
  `ARC_DELTA_FILE`) refuses any other base.
- The engine applies the scales right after loading each `lowbit-L*.pt`, and swaps the dense tensors in vLLM's weight
  stream.

**7.6 Bit-exact checks.**

```bash
python3 $T/finetune/export_delta.py $W/ft/run --check $W/ft/run/ckpt          # file level, CPU: "OK (bit-identical)"
bash $T/finetune/overlay_check.sh full $W/ft/run/ckpt                         # served, MTP off: prompt logprobs + greedy
bash $T/finetune/overlay_check.sh ovl $D3_LB144R $W/ft/run/delta.safetensors
bash $T/finetune/overlay_check.sh base $D3_LB144R
python3 $T/finetune/overlay_check.py cmp $W/overlay/full.json $W/overlay/ovl.json     # shipped: 0 of 24,399 positions differ
python3 $T/finetune/overlay_check.py cmp $W/overlay/full.json $W/overlay/base.json    # every position differs
```

- The file-level check re-applies the overlay with the engine's own code. It compares every scale with the
  `build_ckpt.sh` formula, and every dense tensor of the weight stream with the full checkpoint.
- The served check takes top-5 prompt log-probabilities at every position of 3 validation prompts (24,399 positions),
  plus a 64-token greedy continuation.
- We re-ran this folder's `export_delta.py` on the shipped run's `trained.pt` and the shipped lb144r. It wrote a file
  byte-identical to the published `p1s_delta.safetensors` (sha256 `a1227ff7…`), and `--check` passed.

The MTP layer is not trained. Its acceptance rose anyway, from 3.448 to 3.501 tokens per step.

### 8. Evaluation protocol

All accuracy numbers below come from one or two servers on our H100s with `serve_eval.sh`, sampling at T 0.7. A
single 40-task holdout run has a 90% confidence interval of about ±11 points, and repeated runs of one checkpoint vary
as well: 11 runs of lb144r averaged 61.7%, with a standard deviation of about 5 points per run. So every comparison
was paired (same tasks, both arms run at the same time) and repeated.

**lb144r: holdout40.** The 40 public evaluation tasks of [`eval/holdout40.txt`](eval/holdout40.txt) (61 test
inputs; `random.Random(2026).sample(sorted(evaluation ids), 40)`). Each run makes one attempt per test input with a
240,000-token cap, and reports pass@1 averaged per task. A degeneration test comes first: 16 sequences of 12,000
tokens, flagged if the tail is a runaway digit loop.

```bash
bash $T/eval/holdout.sh lb144r:$D3_LB144R                # degeneration 0/16; score 62.9% (shipped run)
SFX=_r2 bash $T/eval/holdout.sh lb144r:$D3_LB144R        # second run: 62.5%
python3 $T/eval/paired_diff.py "$W/hold/A.jsonl,$W/hold/A_r2.jsonl" "$W/hold/B.jsonl,$W/hold/B_r2.jsonl"
python3 $T/eval/direct_score.py $W/hold/lb144r.jsonl --attempts 1                     # pass@k under per-attempt caps
```

For reference, the full FP8 preview scored about 74–76% on the same tasks with an uncapped budget. `paired_diff.py`
bootstraps the per-task difference of two arms. It rejected a cheaper variant (step 9) at −8.5 points [−17.5, +0.2].

**KL bench (optional; ranks compression variants with little noise).**
- `eval/fidelity.py` compares top-20 prompt log-probabilities over the first 32,768 tokens of the FP8 preview's own
  direct-mode traces of 6 public evaluation tasks, by context band. These traces are a measurement set only; they
  never fed calibration or training.
- Dump the reference from an FP8 server with FlashAttention 2 (`--attention-config '{"flash_attn_version": 2}'`), and
  the candidates from evaluation servers with `MTP=0`.
- For the effect of pruning alone, serve the unpruned model eagerly with `ARC_EXPERT_MASK` set to a mask from
  `reap_scores.py --masks 144 --metric reapsum`.

```bash
mkdir -p $W/fid
bash $T/serve/serve.sh 0,1,2,3 8010 $D3_BASE_MODEL --tensor-parallel-size 4 --language-model-only --trust-remote-code \
  --disable-custom-all-reduce --max-model-len 400000 --attention-config '{"flash_attn_version": 2}' > $LOGDIR/fp8.log 2>&1 &
up $LOGDIR/fp8.log
python3 $T/harness/direct.py --ports 8010 --attempts 1 --tasks 269e22fb,3a25b0d8,446ef5d2,21897d95,5dbc8537,a32d8b75 \
  --out $W/fid/fp8.jsonl --save_text $W/fid/fp8_text.jsonl
python3 $T/eval/fidelity.py dump 8010 $W/fid/ref.npz && bash $T/serve/kill_server.sh 8010 0,1,2,3
MTP=0 L4_GIB=26 EMULATE_GPU_GIB= bash $T/serve/serve_eval.sh 0,1,2,3 8155 $D3_LB144R > $LOGDIR/fid.log 2>&1 &
up $LOGDIR/fid.log && python3 $T/eval/fidelity.py dump 8155 $W/fid/lb144r.npz && bash $T/serve/kill_server.sh 8155 0,1,2,3
python3 $T/eval/fidelity.py cmp $W/fid/ref.npz $W/fid/lb144r.npz
```

| Configuration (KL to the FP8 preview, all completion positions) | KL |
|---|---|
| FP8 preview on the portable serving stack (floor) | 0.0030 |
| pruning only (FP8 experts), 144 by reapsum | 0.0094 |
| pruning only, 144 by gate-weighted routing frequency / summed output norm / gate-weight sum on the calibration rows | 0.0109 / 0.0106 / 0.0120 |
| pruning only, 144 by mean REAP | 0.0247 |
| pruning only, 128 / 112 by reapsum (by routing frequency) | 0.0125 / 0.0193 (0.0162 / 0.0305) |
| lb144r (serving stack of Oct 6) | 0.0142 |
| 160 / 176 experts by reapsum at the same bytes | 0.0138 / 0.0163 |
| 144 by routing frequency, same GPTQ | 0.0168 |
| lb144r on the submission's serving stack | 0.0154 |

The released `fidelity.py` reproduces every number of this table from our dumps.

**p1s: validation runs on training tasks.** `eval/gate.sh` runs one checkpoint on the 200-task validation shard: 216
test inputs × 4 attempts, a 64k cap, a fixed server seed, then the degeneration test. `eval/gate_analyze.py BASE CAND`
reports, with 90% task-bootstrap intervals:
- accuracy per attempt;
- tokens per correct attempt, raw and "matched" (over inputs both arms solve);
- the cap rate;
- MTP acceptance, from the server log.

On the shipped files, `gate_analyze.py` reproduces our report exactly.

```bash
bash $T/eval/gate.sh base $D3_LB144R                  # GPUS=2,3, SEED=6161; a second base run with SEED=6162 measures noise
ARC_DELTA_FILE=$W/ft/run/delta.safetensors bash $T/eval/gate.sh p1s $D3_LB144R     # or: gate.sh p1s $W/ft/run/ckpt
python3 $T/eval/gate_analyze.py base p1s              # -> $W/gate/cmp_base_p1s.json
TEMP=0.6 bash $T/eval/gate.sh t06 $D3_LB144R          # temperature control
```

| Arm | Accuracy | Tokens / correct | MTP tokens/step | Degeneration |
|---|---|---|---|---|
| lb144r (two seeds) | 82.1% / 81.3% | 17,655 / 17,788 | 3.448 / 3.442 | 0/16 |
| lb144r at T 0.6 / 0.5 | 81.7% / 81.7% | 17,105 / 17,013 | 3.468 / 3.483 | 0/16 |
| **p1s** | **83.8%** | **16,669** | **3.501** | 0/16 |

- **p1s − lb144r**, against the two base seeds:
  - accuracy +1.7 [−0.5, +3.9] and +2.6 [+0.5, +4.7];
  - tokens per correct attempt −5.6% and −6.3%;
  - matched tokens per correct attempt −8.9% [−12.2, −5.5] and −10.0% [−13.6, −6.5].
- **Colder sampling is not the same.** Sampling lb144r colder saved no matched tokens (−0.4% to −2.4%, every interval
  includes 0), and p1s beat it by 2.1 points.
- **p1s is sharper, selectively.** On long eval-like rollouts, its top-20 entropy is 29% lower in every context band.

**Full budget.** p1s's value is mainly token efficiency under the time limit, not per-attempt accuracy.
- On 4×L4 test machines, the full notebook on the 120 public evaluation tasks (each included twice), with a
  14,000 s budget, scored 33.7% with p1s against 29.1% without (4 runs each).
- On Kaggle, the first submission with p1s scored 52.64, against 50.56 for the previous submission, the same build
  without p1s. That submission also changed one offload setting.
- Running the notebook is described in [`../kaggle/README.md`](../kaggle/README.md) §4.

**Rules that held up.**
- Pair every comparison and repeat it.
- Use KL to rank compression variants, but confirm on accuracy. KL rated a variant with 16 more 2-bit experts per
  layer almost equal (+0.0012), yet it lost 8.5 holdout points (step 9).
- Validate a fine-tune on the full solver at the shipped budget. Four times, a later fine-tune round that tied or won
  in the faster validation runs on H200 GPUs lost points at full budget on 4×L4.

### 9. What did not work

- **Distillation toward the FP8 preview** (KL on its top-20, same trainable set as p1s): lower KL, but −4.0 and −0.8
  holdout points in its two rounds (5–11 runs per arm, both intervals include 0). Step 10 has the details and the code.
- **More rounds of shortest-correct self-distillation** (more tasks, 128k caps with forced conclusions,
  below-median-length selection): −2.6 to −7.4 points at full budget on 4×L4. Each round committed earlier and more
  often wrongly.
- **Fine-tuning on hint-rationalised traces, and a policy-gradient round with privileged teacher contexts:** −4.0 and
  −2.5 to −8.3 of 240 at full budget.
- **Dots3-Note as a tool-using agent:** −5.76 of 240 against its own direct mode.
- **Re-running GPTQ with 70× more, long-context calibration data:** 30% lower held-out MoE error, but +0.7
  [−2.7, +4.1] pass@2 alone and −2.9 with p1s, whose scale offsets are tied to lb144r's exact codes.
- **16 more experts per layer at 2 bits** (80 × 3-bit + 64 × 2-bit, freeing 0.49 GiB per GPU for KV): KL only +0.0012,
  but −8.5 holdout points [−17.5, +0.2].
- **Mean REAP** (the average instead of the sum of gate × output norm): more than twice the KL of routing-frequency
  ranking.
- **Mass-greedy per-layer expert counts** and a **knapsack on MoE output error**: worse KL than a uniform 144.
- **Refitting the router rows for the kept experts:** held-out MoE error got worse, because the top-8 selection flips.
- **Lower sampling temperature instead of a fine-tune:** no token saving (step 8).
- **Fine-tuning the MTP layer on p1s rollouts:** +0.2% accepted tokens.

### 10. Distillation toward the FP8 preview (not used)

Two rounds trained the tensors of p1s toward the FP8 preview's next-token distribution instead of toward answers. Both
lowered the KL to the preview, but neither gained solve rate, so no distilled checkpoint was shipped.

| | Teacher-text round | On-policy round |
|---|---|---|
| Training text | the preview's answers to training tasks | the first round's answers to training tasks |
| Validation KL (held-out training tasks) | 0.0146 → 0.0113 (−22%) | 0.01325 → 0.01203 (−9%) |
| KL bench on the serving stack (lb144r: 0.0154) | 0.0128 | 0.0133 |
| holdout40 (lb144r: 11 runs, 61.7%) | 6 runs, 57.7% | 5 runs, 60.9% |
| Difference to lb144r, 90% CI | −4.0 [−11.7, +3.1] | −0.8 [−7.5, +5.9] |
| Mean tokens per attempt (lb144r: 115k) | 122k | 120k |

- The intervals come from `eval/arm_stats.py`, which resamples runs within each arm and then tasks, because a single
  holdout run varies by about 5 points. On our run files it reproduces both intervals.
- Per token, the first round's checkpoint was closer to the preview everywhere we measured, including holdout
  rollouts up to 172k tokens. What changed was style: both rounds took over the preview's habit of marking mismatches
  and re-checking, and wrote 4–6% more tokens per attempt, which costs throughput under the time limit.

**10.1 Data** (training split only).
- `distill/tasks.py` drops the training tasks that share an input grid with an evaluation task (none at commit
  `f3283f7`), shuffles the other training ids with `random.Random(0)`, and takes 560 tasks for training and the next 40
  for validation. It asserts that no listed id is an evaluation id.
- **Teacher generations.** The preview answers each of the 647 test inputs of these 600 tasks once in direct mode
  (`harness/direct.py`: T 0.7, top-p 1.0, 7,168-token cap). Our run made 629 generations on 586 tasks, with 3.55M
  completion tokens (236 correct). One prompt (24,458 tokens) does not fit the 20,480-token context, and we stopped the
  run before the last 17 inputs. The first 62 generations came from a server without CUDA graphs, 37 of them with a
  12,288-token cap.
- **Calibration traces.** `distill/calib_traces.py` adds the preview's traces of step 1. Their prompts carry no task
  id, so it finds the task by grid hash: the first grid of the prompt, in JSON form, that is an input grid of a task (a
  grid shared by several tasks maps to the first id in sorted order). It skips traces that match no task, an evaluation
  task or a validation task. That keeps the 24 direct-mode traces of training tasks; the 40 program-synthesis traces
  write their grids as rows of digits and match nothing. The 24 are in both `calib_mix.jsonl` and `direct_text.jsonl`,
  so they enter twice: 48 sequences.
- **Teacher targets.** `distill/teacher_dump.py` sends each sequence (prompt + completion, truncated to 10,240 tokens)
  to the teacher with `prompt_logprobs=20`, and stores the top-20 token ids and log-probabilities at every position.
  It skips sequences that keep at most 16 completion tokens in the window (33 generations), and it refuses evaluation
  ids. That gives 644 sequences (596 generations and the 48 traces), 4.6 GB.
- **Validation.** The 39 sequences of the validation tasks are held out, and the trainer scores the first 16 (64,670
  positions). That leaves 605 training sequences with 2.82M completion tokens.
- **Leak control.** Only training tasks are sampled and scored, and none shares an input grid with an evaluation task.
  The trainer's `--fid` option scores the six evaluation-task traces of the KL bench (step 8) for measurement only;
  they are never trained on.

**10.2 Teacher.** The FP8 preview on 4 H100s (TP4) with FlashAttention 2, FP8 KV cache, portable sparse attention
(`ARC_DOTS3_PORTABLE_DSA=1`), CUDA graphs, no MTP, memory utilization 0.95 and a 20,480-token context.
- Without CUDA graphs the server was slower (180–250 tokens/s), and at utilization 0.97 it ran out of memory on a
  prompt-logprobs request.
- On H100s, vLLM's DeepGEMM path re-quantizes the block-FP8 weights to power-of-two scales. The teacher of the
  on-policy round ran without it (`VLLM_USE_DEEP_GEMM=0`). On the KL bench's traces, the preview with and without it
  differs by a KL of 0.0043.

**10.3 Student, trainable set and loss.**
- The student is `finetune/student.py` with the trainable set of p1s (step 7.4): the expert scale offsets, the
  RMSNorms and the router rows, 1,228,137,856 parameters. Norms and router are trained in bf16, without the fp32
  masters of p1s (`--fp32-master` is 0 by default with `--loss kd`).
- **Loss** (`train.py --loss kd`): on every completion position, KL(teacher ‖ student) over 21 bins, the teacher's
  top-20 tokens and the remaining mass (1 minus the probability of those 20 tokens, on each side). It is summed over
  the positions of one sequence per step; sequences are truncated to `--len`. Optimizer and schedule are those of p1s:
  Adam (0.9, 0.99), 10 warm-up steps, then cosine to 0.
- **Validation:** the same KL on the held-out sequences and, with `--fid REF.npz`, the KL to a KL-bench reference on
  the first 8,192 tokens of its six traces. Evaluations are at T 1; `--temp` (both rounds: 1) would train both sides
  at another temperature.

**10.4 Settings of the two rounds.**

| | Teacher-text round | On-policy round |
|---|---|---|
| Starts from | lb144r | the teacher-text round's `trained.pt` |
| Sequences: training / validation | 605 / 16, 2.82M completion tokens | 604 / 24, 4.93M completion tokens |
| Sequence length (`--len`) | 10,240 | 16,384 |
| Steps | 450 (0.74 epoch) | 250 scheduled; the state at step 200 is the result |
| Learning rates: scales / norms / router | 5e-4 / 5e-5 / 1e-5 | 2.5e-4 / 2.5e-5 / 5e-6 |
| Evaluation | every 75 steps; `--fid` reference from the preview as in step 8 | every 50 steps; `--fid` reference without DeepGEMM |
| Time on 5 H100 80 GB (one layer pipeline) | 2.5 h, about 19 s per step | 1.6 h to step 200, about 24 s per step |

- **On-policy data** (`distill/onpolicy_data.sh`). The first round's exported checkpoint, served with the evaluation
  numerics (TP4, 26 GiB budget, MTP-3), answers the 647 test inputs once at T 0.7 with a 14,336-token cap (63 min).
  The preview without DeepGEMM (TP4, 172,032-token context, utilization 0.97, 8 sequences, 1,024-token prefill chunks)
  then scores these rollouts, truncated to 22,528 tokens: 646 sequences, 42 of them on validation tasks. The trainer
  truncates them again to 16,384 tokens.
- Our on-policy run stopped at step 202, on a sequence whose prompt filled the whole 16,384-token window. The trainer
  now skips such windows, which leaves the earlier steps unchanged. The result was the `trained.pt` written at the
  step-200 evaluation; a full run overwrites it at step 250, so copy it when `log.jsonl` reaches step 200.

**10.5 Commands.**

```bash
python3 $T/distill/tasks.py                           # -> $W/kd/tasks_{train,val,all}.txt
python3 $T/distill/calib_traces.py                    # -> $W/kd/calib_text.jsonl, 48 sequences (needs step 1's traces)
ARC_DOTS3_PORTABLE_DSA=1 UTIL=0.95 bash $T/serve/serve.sh 0,1,2,3 8461 $D3_BASE_MODEL --tensor-parallel-size 4 \
  --language-model-only --trust-remote-code --disable-custom-all-reduce --kv-cache-dtype fp8 --max-model-len 20480 \
  --max-num-seqs 32 --max-num-batched-tokens 8192 --attention-config '{"flash_attn_version": 2}' \
  --compilation-config '{"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[4,8,16,32]}' > $LOGDIR/teacher.log 2>&1 &
up $LOGDIR/teacher.log
python3 $T/harness/direct.py --ports 8461 --split training --tasks @$W/kd/tasks_all.txt --attempts 1 --max_tokens 7168 \
  --max_model_len 20480 --concurrency 9 --out $W/kd/gen.jsonl --save_text $W/kd/gen_text.jsonl
cat $W/kd/calib_text.jsonl $W/kd/gen_text.jsonl > $W/kd/all_text.jsonl
python3 $T/distill/teacher_dump.py 8461 $W/kd/all_text.jsonl $W/kd/teacher --len 10240 --conc 2 --tag x
bash $T/serve/kill_server.sh 8461 0,1,2,3
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1,2,3,4 python3 $T/finetune/train.py --loss kd \
  --data $W/kd/teacher --val-tasks $W/kd/tasks_val.txt --gpus 0,1,2,3,4 --len 10240 --steps 450 --lr 5e-4 \
  --lr-norm 5e-5 --lr-router 1e-5 --trainable scales,norms,router --eval-every 75 --max-val 16 \
  --fid $W/fid/ref.npz --fid-len 8192 --out $W/kd/run           # -> log.jsonl, trained.pt
bash $T/finetune/build_ckpt.sh $W/kd/run                        # -> $W/kd/run/ckpt
bash $T/distill/onpolicy_data.sh $W/kd/run/ckpt onp 14336       # -> $W/kd/onp_onp.jsonl, $W/kd/teacher_onp/
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python3 $T/finetune/train.py --loss kd --data $W/kd/teacher_onp \
  --val-tasks $W/kd/tasks_val.txt --gpus 0,1,2,3,4 --len 16384 --steps 250 --eval-every 50 --lr 2.5e-4 \
  --lr-norm 2.5e-5 --lr-router 5e-6 --trainable scales,norms,router --resume $W/kd/run/trained.pt \
  --fid $W/fid/ref.npz --fid-len 8192 --out $W/kd/run_onp       # keep the trained.pt of step 200
bash $T/finetune/build_ckpt.sh $W/kd/run_onp
```

- `--fid` needs the reference of the KL bench (step 8); leave it out to skip the measurement. For the on-policy round
  we dumped that reference from a preview server with `VLLM_USE_DEEP_GEMM=0`.
- The KL bench and holdout40 then run as in step 8; for the KL bench we served the checkpoints with `serve_eval.sh`'s
  defaults. Our holdout arms ran interleaved. Give every run its own server seed (`SEED`): two fresh servers with the
  same seed replayed one of our runs exactly.

```bash
bash $T/serve/serve_eval.sh 0,1,2,3 8155 $W/kd/run/ckpt > $LOGDIR/fid_kd.log 2>&1 &
up $LOGDIR/fid_kd.log && python3 $T/eval/fidelity.py dump 8155 $W/fid/kd.npz && bash $T/serve/kill_server.sh 8155 0,1,2,3
python3 $T/eval/fidelity.py cmp $W/fid/ref.npz $W/fid/lb144r.npz $W/fid/kd.npz
for i in 1 2 3 4 5 6; do SEED=$i SFX=_s$i bash $T/eval/holdout.sh kd:$W/kd/run/ckpt lb144r:$D3_LB144R; done
python3 $T/eval/arm_stats.py lb144r=$(ls $W/hold/lb144r_s?.jsonl | paste -sd,) kd=$(ls $W/hold/kd_s?.jsonl | paste -sd,)
```

**10.6 Hardware and time.**
- Teacher: 4 H100 80 GB (TP4). The generations took about 2 h at about 470 tokens/s, the dump 13 min (about 1.5 s per
  sequence).
- Training: 5 H100 80 GB in one layer pipeline. Our rounds ran an earlier version of `student.py` that kept the expert
  codes as uint8 (about 35 GB of weights per GPU). This one keeps them packed: on 4 GPUs it holds 17–19 GB of weights
  per GPU and peaked at 32 GB with 10,240-token sequences.
- On-policy data: 63 min of student rollouts and about 25 min of teacher scoring, on 4 H100s.
- Each export takes 66 GB (`build_ckpt.sh` copies lb144r).

**10.7 Checks of this release.** The data, dumps and checkpoints of the two rounds were not kept. We checked the tools
as follows:
- **Data.** On the ARC-AGI-2 data and our calibration traces, `tasks.py` and `calib_traces.py` match our dump logs:
  the names and token counts of all 48 calibration sequences, and 39 and 42 dumped sequences on validation tasks, which
  leave the 605 and 604 training sequences of our two training logs.
- **Step-0 KL.** On lb144r, against the KL-bench reference of our rounds (`--fid-len 8192`), the trainer's first
  evaluation gives the logged values of both rounds to the last digit: KL 0.01266134075820446 and top-1
  0.9736111111111111 over 14,400 positions.
- **Same steps as our rounds.** With the commands of 10.5 on 4 H100s, we made a small set: 12 teacher generations and
  2 calibration traces, 13 sequences after the dump; `onpolicy_data.sh` added 4 on-policy sequences of 4 training
  tasks. The released `train.py` and `student.py` and the versions that ran the on-policy round (the first round's
  `train.py` lacked only `--temp` and the guards for prompt-only windows) then trained 6 steps with the first round's
  settings, and 4 more from that state with the on-policy round's settings (16,384-token windows). Every training,
  validation and fidelity KL in the logs and every trained tensor (1,228,137,856 values) came out bit-identical.
- **Cross-entropy unchanged.** With the p1s settings, this `train.py` and its earlier cross-entropy-only version give
  the same held-out split, the same step-0 cross-entropy (0.13551295847211323 over 134,814 tokens), and bit-identical
  tensors after 3 training steps.
- **Export.** `build_ckpt.sh` writes the trained norms and router rows exactly and the scales by its formula, and
  `--resume trained.pt` reproduces the trainer's own evaluation exactly. After 20 scales-only steps at lr 1e-3 (mean
  offset 1.5e-3), the exported checkpoint keeps the trained KL: fidelity KL 0.01210 against 0.01210, validation KL
  0.02022 against 0.02025.
- **Offsets below fp16 resolution do not survive the export.** After the 6 warm-up steps above (mean offset 1.8e-4),
  the trained state has validation KL 0.02058 but its export 0.02124 (step 0: 0.02128). The student dequantizes the
  experts to bf16; tiny fp32 offsets tip exact rounding ties there (3.6% of the weights change), which the fp16 scales
  cannot express (0.35%). While the offsets are that small (the first steps, or very small learning rates), the KL in
  `log.jsonl` overstates what the export keeps.
- **Holdout statistics.** On our run files, `eval/arm_stats.py` reproduces −4.0 [−11.7, +3.1] and −0.8 [−7.5, +5.9].

## Script reference

The Python files had their comments and docstrings removed for release, so this table is their documentation.
`--help` lists every option.

| Script | What it does / main arguments |
|---|---|
| `env.sh` | shared defaults; sourced by every shell script |
| `harness/direct.py` | direct-mode sampler: one request per (task, test input, attempt) with the ARC Prize benchmark prompt (thinking on; prompt template and answer parsers from arc-agi-benchmarking, MIT). `--ports P,... --split training\|evaluation\|DIR --tasks @FILE\|a,b --attempts N --max_tokens 384000 --max_model_len 400000 --temp 0.7 --top_p 1.0 --concurrency 64 --out F.jsonl --save_text F_text.jsonl` → per attempt: tokens, finish reason, parsed grid, correctness. Resumable. A `--split` that is not under `$ARC_DATA` must be an absolute path |
| `calib/synth.py` | program-synthesis sampler: `--kind qwen --ports --split --tasks --think 12000 --answer 4000 --experts 1 --iters 1 --save_text --tokenizer DIR --out F.jsonl` (sandboxed `transform` checks against the training pairs; `--iters` > 1 adds refinement with feedback) |
| `calib/make_mix.py OUT DIRECT_TEXT SYNTH` | calibration trace file `{"full": {"prompt", "completion"}}` per line |
| `calib/tasks_direct30.txt`, `tasks_synth40.txt` | the 70 calibration tasks (ARC-AGI-2 training) |
| `prune_gptq/calib_prefill.py PORT [--n 16 --len 16384 --src F]` | prefill of the N longest traces' first LEN tokens (`max_tokens=1`) into a capture server |
| `prune_gptq/reap_scores.py` | `--calib DIR --out F.pt --gpu G [--layers 1-45] [--max-tok N]` → per layer `reap`, `count`, `wsum`, `ean`; `--masks 144,160 --metric reapsum --mask-dir DIR` writes keep masks (`{layer: ids}`) from an existing score file |
| `prune_gptq/reapsum_stats.py` | `--reap F.pt --out F.pt [--template F.pt]` → routing-statistics file with `wsum` = reapsum |
| `prune_gptq/prune.py` | `--stats F --keep 144 --out DIR [--no-experts] [--plan keep.json] [--metric wsum\|count]` → pruned checkpoint (experts kept as FP8 unless `--no-experts`), `pruning.json`, config |
| `prune_gptq/gptq_experts.py OUT` | `--tiers 96:3,144:2 --stats F --calib DIR --pack DIR --no-fp8 --layers A-B --gpu G [--group 128 --damp 0.01 --batch 32] [--rtn]`; without `--pack`: FP8 fake-quant shards + codes, `--assemble` |
| `serve/serve.sh GPUS PORT MODEL [vllm args]` | plain vLLM server (env `UTIL`, `VENV`, `CUDA_HOME`) |
| `serve/serve_eval.sh GPUS PORT CKPT [vllm args]` | evaluation server with the 4×L4 numerics (env `L4_GIB`, `MTP`, `KLOC`, `CHUNK`, `MAXLEN`, `SEQS`, `CACHE_ROWS`, `INDEX_SHARE`, `EMULATE_GPU_GIB`, `EXTRA_ENV`, `ARC_DELTA_FILE`, `ARC_W8A8_CONFIG_DIR`); `serve/w8a8_tiles.json` = its W8A8 tile table |
| `serve/kill_server.sh PORT GPUS` | stops one server and whatever still holds its GPUs |
| `finetune/split.json`, `split.py [split.json]` | the shipped task split with strata; re-derivation, assertions, task folders under `$D3_WORK/ft/` |
| `finetune/gen.sh` | fine-tune data (env `GPUS`, `PORT`, `CKPT`, `L4_GIB`) |
| `finetune/select_sft.py [--maxlen 40960 --hard-w 2 --out DIR]` | the SFT set: `<task>_<test>.npz` (`ids` = prompt + completion + end token, `plen`, `w`) and `manifest.json` |
| `finetune/train.sh OUT` | training + `build_ckpt.sh` (env `GPUS`, `LR`, `LRN`, `LRR`, `DATA`, `CKPT`) |
| `finetune/train.py` | `--ckpt DIR --data DIRS --gpus 0,1,2 --len 40960 --epochs 2 --lr --lr-norm --lr-router --trainable scales,norms,router --val-n 16 --eval-every 100 --out DIR [--resume trained.pt] [--export] [--eval-only]` → `log.jsonl`, `trained.pt`, `val_tasks.txt`. `--loss kd` (default `ce`): distillation on `teacher_dump.py` output, `--val-tasks FILE --steps N [--fid REF.npz --fid-len 8192] [--temp 1.0]`, sequences truncated to `--len`; fp32 masters (`--fp32-master`) are on by default only with `ce` |
| `finetune/student.py` | the differentiable model: `Student(ckpt, gpus, trainable=("scales", "norms", "router"))`, `.hidden(ids)`, `.export_scales(out, src)`; env `DISTILL_NESTED_CKPT`, `DISTILL_NESTED_MIN`, `DISTILL_PACKED` (1), `DISTILL_ACT_FP8` (1), `DISTILL_KV_FP8` (1) |
| `finetune/build_ckpt.sh RUN [GPU]` | full checkpoint `RUN/ckpt` from `RUN/trained.pt` (env `CKPT`) |
| `finetune/export_delta.py RUN [--base DIR] [--out F]`, `--check [CKPT]` | overlay export; bit-exact check against the formula and, if given, a full checkpoint |
| `finetune/overlay_check.sh TAG CKPT [DELTA]`, `overlay_check.py dump PORT OUT \| cmp A B \| prompts P` | served exactness check of an overlay (env `GPUS`, `PORT`) |
| `distill/tasks.py [--out DIR --train 560 --val 40 --seed 0]` | task lists of step 10 (`tasks_train.txt`, `tasks_val.txt`, `tasks_all.txt`): training split, no input grid shared with an evaluation task |
| `distill/calib_traces.py [--mix F --direct F --val-tasks F --out F]` | step 1's calibration traces, mapped to training tasks by grid hash, as `teacher_dump.py` input |
| `distill/teacher_dump.py PORT SRC.jsonl OUT [--len 12288 --conc 4 --tag t --tasks @FILE --tokenizer DIR]` | teacher top-20 prompt log-probabilities of prompt + completion: `OUT/<task>_<test>_<attempt>_<tag>.npz` (`ids`, `plen`, `top_ids`, `top_lp`); refuses evaluation ids; resumable |
| `distill/onpolicy_data.sh CKPT TAG [MAXTOK 12288]` | on-policy data: one attempt per test input from CKPT on the evaluation server, then teacher scoring without DeepGEMM (env `GPUS`, `TASKS`, `PORT`, `TPORT`) |
| `eval/holdout.sh TAG:CKPT ...` | degeneration test + holdout40 (env `GPUS`, `PORT`, `LG`, `SFX`, `SEED`) |
| `eval/score_hold.py RUNS...`, `paired_diff.py "A1,A2" "B1,B2"`, `direct_score.py RUNS [--attempts 1 --caps ...]` | holdout score with a 90% CI; paired difference; pass@k under per-attempt caps |
| `eval/degen_test.py PORT [N G]` | N holdout prompts × G tokens; counts runaway digit loops |
| `eval/arm_stats.py NAME=RUN,RUN,... NAME=...` | holdout arms against the first one: 90% CI of a two-level bootstrap (runs within each arm, then tasks), task-only CI, run-level permutation p |
| `eval/gate.sh TAG CKPT`, `gate_analyze.py BASE CAND` | validation run (env `GPUS`, `PORT`, `ATT`, `SEED`, `TEMP`) and its paired analysis |
| `eval/fidelity.py dump PORT OUT.npz \| cmp REF.npz CAND.npz ...` | KL bench (`--n 6 --len 32768`; traces from `D3_FID_TRACES`) |

## Provenance and caveats

- **What cannot be reproduced exactly.**
  - The 80 excluded tasks and the difficulty strata come from earlier runs, so `split.json` ships them as data.
  - The FP8 server flags of the calibration traces were not recorded (step 1).
  - Sampling on GPUs is not deterministic.
  - The data, teacher dumps and checkpoints of the distillation rounds (step 10) were not kept. Their numbers come
    from our logs and notes; step 10.7 lists what we checked.
- **What was reproduced exactly.** From the shipped intermediate files, the tools of this folder reproduce:
  - the calibration mix;
  - the ranking file and `pruning.json`;
  - the SFT set, and the trainer's held-out split and step-0 cross-entropy;
  - the validation analysis and the KL table;
  - `p1s_delta.safetensors`, byte for byte;
  - for step 10: the calibration sequences and the sequence counts of both distillation rounds, the trainer's step-0
    KL, and the two holdout intervals.

  The keep sets and the 3/2-bit allocation of all 45 shipped layers follow the rules of steps 3 and 4. The captured
  MoE inputs were not kept, so the GPTQ run itself was not repeated.
- **The overlay is tied to its base.** It carries the expert ids and checksums of the published lb144r's scales, and
  the engine refuses it on any other checkpoint. A rebuilt lb144r needs its own fine-tune.
- **Versions.** `finetune/train.py` and `student.py` are the versions that trained p1s; `train.py --loss kd` is their
  distillation mode (step 10). The two distillation rounds ran earlier versions of both files. On the same data these
  give bit-identical training steps (step 10.7).
- **Third-party code.** `harness/direct.py` contains the prompt template and two answer parsers of
  github.com/arcprize/arc-agi-benchmarking (MIT License).
