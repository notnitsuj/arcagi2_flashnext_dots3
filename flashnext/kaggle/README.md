# Kaggle submission

This folder is the submission as it ran on Kaggle, in its two final forms:
- **RC8** (the default here): notebook version 5 with code dataset version 11, submission 57879466. It scored **75.69** on the public leaderboard, our best.
- **RC9**, the final bundle: notebook version 7 with code dataset version 13, submission 57900065. It scored 73.89.

Both use the same Kaggle Models. RC9 is RC8 plus three changes (section 4.1): int8 reduce-scatter on the tensor-parallel group as well (`FN_TP_Q8_RS=1`), a guard in the solver's endgame estimate, and a larger compile cache. The two scores are within the leaderboard's noise of each other.

The folder also holds the tools that rebuild every Kaggle artifact from the training outputs.

| Path | What it holds |
|---|---|
| [`notebook/rc8/`](notebook/rc8/), [`notebook/rc9/`](notebook/rc9/) | `flash-next-arc-eval.ipynb` and `kernel-metadata.json`, byte-identical to what `kaggle kernels pull` returns for versions 5 (RC8) and 7 (RC9). |
| [`code_dataset/`](code_dataset/) | The code dataset, version 11 (RC8): every published file verbatim, with `fnext.tarbin` unpacked into [`fnext/`](code_dataset/fnext/). [`MANIFEST.md5`](code_dataset/MANIFEST.md5) holds the md5 of every published file. `fn_cache.tarbin` (141 MB) is not in git. |
| [`code_dataset/rc9/`](code_dataset/rc9/) | RC9's version of the two files that changed (`repl_solve.py`, `fnext/base/q8_rs.py`) and the md5 of every file of dataset version 13. `build/pack_code_dataset.py --release rc9` overlays them. |
| [`build/`](build/) | Tools for the core split, the model parts and their upload, the offline runtime, the code dataset, the compile cache, the notebook and the validation check. |

Commands below run from this folder (`flashnext/kaggle/`).

Contents: [1 Artifacts](#1-artifacts) · [2 Re-running on Kaggle](#2-re-running-the-submission-on-kaggle) · [3 Running on your own 4×L4](#3-running-the-notebook-on-your-own-4l4-machine) · [4 Launcher settings](#4-launcher-settings-of-rc8-and-rc9) · [5 Rebuilding the artifacts](#5-rebuilding-the-artifacts) · [6 Module map](#6-module-map)

## 1. Artifacts

| Artifact | Kaggle reference | Variation | Holds | Size |
|---|---|---|---|---|
| Notebook | `notnitsuj/flash-next-arc-eval`, versions 5 (RC8) and 7 (RC9) | | 15 cells: inputs, run mode, runtime, server and solver (through `kaggle_fn.sh`), submission | 45 KB |
| Code dataset | `notnitsuj/flash-next-arc-code`, versions 11 (RC8) and 13 (RC9) | | 16 flat files: the engine plugin (`fnext.tarbin`), the agent (`repl_*.py`, `arckit.py`), the launcher, the delta tools, the draft-head vocabulary, the L4 kernel tiles and the compile cache (`fn_cache.tarbin`) | 142.4 MB |
| Model: core | `notnitsuj/qwen38-flash-next-fp8-core` (PyTorch) | `w-p1` | `base-0000{1..3}-of-00003.safetensors`: every tensor of Qwen3.8-Flash-Next-FP8 except the routed experts and the n-gram table, byte-identical and re-sharded. Also config, tokenizer, chat template and LICENSE | 10.9 GB |
| | | `ple-p1` … `ple-p4` | `ngram-000{01..15}-of-00015.safetensors`: the FP8 n-gram embedding table, 4/4/4/3 shards per part | 14.5 / 14.4 / 14.4 / 8.0 GB |
| Model: experts | `notnitsuj/qwen38-flash-next-experts` (PyTorch) | `k320-q34-p1` … `p4` | The base expert build. `lowbit-L00..L48.pt` hold the packed GPTQ codes, fp16 group scales and zeros of the 320 kept experts per layer (160 at 4 bits, 160 at 3 bits, group size 128). `router-L00..L48.pt` hold the router rows of the kept experts, and `keep.json` the kept expert ids. Layers 0–19, 20–39, 40–48 plus routers 0–36, then routers 37–48 | 14.5 / 14.5 / 6.6 / 0.02 GB |
| | | `k320-q34-r11c-d-p1` … `p3` | The RL delta of RC8 and RC9 (three rounds): `scales-L00..L48.pt` (the RL-tuned group scales of every kept expert), `router-L00..L48.pt` and `keep.json` | 0.08 / 1.17 / 0.05 GB |
| | | `k320-q34-r1b-mtp2-p1` | The fine-tuned draft head: `mtp_ovr.safetensors` (BF16 MTP dense tensors), `scales-L48.pt`, `router-L48.pt` and `manifest.json` (md5 and size of each file) | 0.20 GB |
| Model: runtime | `notnitsuj/flash-next-offline-runtime` (Other) | `py312-cu130` | `fn_py312.tarbin` holds CPython 3.12.14, vLLM 0.32.0, PyTorch 2.13.0 (CUDA 13.0), FlashInfer 0.7.0.post1 and its sm89 JIT cache, all unmodified. It unpacks to `/tmp/fn_py312` and comes with `fn_py312.tarbin.sha256` and `LICENSE` | 9.1 GB |

- Every Model variation is at version 1. The model files total 108.4 GB (101 GiB).
- The notebook merges the parts of each Model by file name and checks every file against its exact size (`MODEL_FILES` in the configuration cell).
- On Kaggle, 16 threads read all of them into the page cache in about 110 s.
- The model parts derive from Qwen3.8-Flash-Next and fall under the Qwen Community License ([`../MODEL_LICENSE`](../MODEL_LICENSE)). It ships as `LICENSE` with every Model's variations (the RL delta has it in its first part).

## 2. Re-running the submission on Kaggle

**Inputs.** `notebook/rc8/kernel-metadata.json` lists them (RC9's lists the same):
- the competition `arc-prize-2026-arc-agi-2`;
- the dataset `notnitsuj/flash-next-arc-code`;
- the 14 Model variations above.

It also sets the machine shape `NvidiaL4` (4× L4) with internet off, and records the Docker image digest the run used. The dataset is not pinned to a version: a push attaches the latest one. Notebook version 5 ran dataset version 11, and version 7 ran version 13; *Copy & Edit* of a version keeps the inputs that version ran with. To check a downloaded dataset folder, run `md5sum -c <repo>/flashnext/kaggle/code_dataset/MANIFEST.md5` (RC8) or `.../code_dataset/rc9/MANIFEST.md5` (RC9) inside it.

**How to run it.**
- On kaggle.com: open the notebook, pick version 5 (RC8) or version 7 (RC9) in its version history, choose *Copy & Edit* (the inputs stay attached), then *Submit to competition* or *Save Version*.
- From the CLI: render a notebook folder under your own kernel id with `build/build_notebook.py` (section 5.8), then run `kaggle kernels push -p <folder>`.

**Run modes.** The notebook selects its mode from the environment:

| Mode | When | Tasks | Budget | Output |
|---|---|---|---|---|
| Scored rerun | Kaggle sets `KAGGLE_IS_COMPETITION_RERUN` after *Submit* | hidden test set | `DT` = 42,150 s after container start; solver killed at 42,600 s; launcher stopped at 42,780 s; Kaggle's limit is 43,200 s (12 h) | `submission.json` |
| Save run, eval mode (versions 5 and 7) | *Save Version* or `kaggle kernels push` | the 120 public evaluation tasks twice (240 entries; the copies get a `b` suffix) | rerun settings with `DT` = 7,200 s and `KILLBY` = 7,650 s | `eval_submission.json`, plus `eval_score.json` with pass@2 per task. `submission.json` stays the input echo |
| Save run, smoke mode | a notebook built with `"eval_dt": null` | the public evaluation split | `DT` = 2,400 s | `smoke_submission.json` |

Their own 2-hour save runs scored 122 of 240 at 1,087 generated tokens/s (version 5, RC8; server healthy after 302 s) and 130.5 of 240 at 1,111 tokens/s (version 7, RC9; healthy after 245 s). Single runs differ by a few points from sampling alone. A full-length 12-hour replica of RC8 on our own 4×L4 test machines scored 207.83 of 240 (86.6%) at 1,005 tokens/s.

**Fail-safes.**
- A valid `submission.json` (the input echo) exists from the first minute, and the solver rewrites it atomically after every session.
- If the delta preflight fails, the server falls back to a rebuilt expert build, then to the base build.
- If the server never comes up, the launcher is restarted once on a safer build.
- The last cell re-validates `submission.json` in a `finally` block (any missing attempt becomes the input grid), then kills the server tree.

**Checking a run.** Download the outputs with `kaggle kernels output OWNER/KERNEL -p out/`, then run:

```
python3 build/check_validation.py out --overlay --q8 --tp8 --max-compiles 5     # RC9
python3 build/check_validation.py out --overlay --q8 --max-compiles 100          # RC8
```

These are the criteria version 7 passed. Version 5's output passes the RC8 line: the same checks without the tensor-parallel one, with 81 kernels compiled during its run:
- all stages ran, and the server was healthy with no restarts;
- at least 900 tokens/s, with 0 `No module named 'ak'` errors and no solver tracebacks;
- every overlay module installed and used, with int8 reduce-scatter on the tensor-parallel group but never on the data-parallel one;
- at least 930k KV tokens per engine and at most 5 kernels compiled during the run.

A run leaves the notebook log, `fn/server_log.txt`, `fn/solve.log`, `fn/repl_samples.jsonl` (every session) and `fn_monitor.log` (host memory every 120 s).

## 3. Running the notebook on your own 4×L4 machine

The notebook only orchestrates: its cells use the Python standard library. The engine runs in the runtime archive's own Python.

**Needs.**
- **GPUs:** 4× L4 (24 GB, sm89) and a driver with CUDA 13.0 support (580 or newer).
- **Software:** Linux with glibc ≥ 2.34, plus `bash`, `tar`, `curl` and `pkill`.
- **Memory:** Kaggle's container allows 186 GB (a 173 GiB cgroup limit), and RC9 (version 7) peaked at 137 GB including page cache. The n-gram table and embeddings take 51 GiB of shared memory.
- **Disk:** 109 GB for the inputs, plus 15 GB in `/tmp`, or 50 GB if the expert build has to be rebuilt.

In Docker, use `--gpus all --ipc=host --ulimit memlock=-1`: pinned host buffers are registered with CUDA and shared across ranks through `/dev/shm`. To match Kaggle's limit, add `--memory=173g`.

**1. Build an input tree like `/kaggle/input`.** The notebook searches below `NB_INPUT_DIR` for:
- the competition files by name (`arc-agi_test_challenges.json`, `arc-agi_evaluation_{challenges,solutions}.json`);
- the dataset by its `fn_runtime.json`;
- each Model variation by a `<slug>/<framework>/<variation>/<version>` path (case-insensitive).

```
IN=$PWD/kaggle_input
kaggle competitions download arc-prize-2026-arc-agi-2 -p $IN/arc-prize-2026-arc-agi-2 --unzip
kaggle datasets download notnitsuj/flash-next-arc-code/13 -p $IN/flash-next-arc-code --unzip
for v in w-p1 ple-p1 ple-p2 ple-p3 ple-p4; do
  kaggle models instances versions download notnitsuj/qwen38-flash-next-fp8-core/pyTorch/$v/1 -p $IN/qwen38-flash-next-fp8-core/pytorch/$v/1 --untar; done
for v in k320-q34-p1 k320-q34-p2 k320-q34-p3 k320-q34-p4 k320-q34-r11c-d-p1 k320-q34-r11c-d-p2 k320-q34-r11c-d-p3 k320-q34-r1b-mtp2-p1; do
  kaggle models instances versions download notnitsuj/qwen38-flash-next-experts/pyTorch/$v/1 -p $IN/qwen38-flash-next-experts/pytorch/$v/1 --untar; done
kaggle models instances versions download notnitsuj/flash-next-offline-runtime/other/py312-cu130/1 -p $IN/flash-next-offline-runtime/other/py312-cu130/1 --untar
```

You can also assemble the dataset folder from this repository with `python3 build/pack_code_dataset.py $IN/flash-next-arc-code --fn-cache fn_cache.tarbin` (add `--release rc9` for RC9), using `fn_cache.tarbin` from dataset version 11 (RC8) or 13 (RC9).

**2. Run it.**

```
export NB_INPUT_DIR=$IN NB_WORK_DIR=$PWD/work          # leave NB_TMP_DIR at its default, /tmp/fn
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output-dir work notebook/rc8/flash-next-arc-eval.ipynb   # or notebook/rc9/
```

- **Without `KAGGLE_IS_COMPETITION_RERUN`**, this is the 2-hour eval-mode run, scored into `work/eval_score.json`.
- **With `KAGGLE_IS_COMPETITION_RERUN=1`**, it is the full 11.7-hour scored path on `arc-agi_test_challenges.json`.
- **For a 12-hour replica on public data,** replace that file with the 240-entry evaluation set. Score `work/submission.json` against the matching solutions afterwards:

```
python3 - $IN/arc-prize-2026-arc-agi-2 <<'EOF'
import json, sys
d = sys.argv[1]
ch, so = (json.load(open(f"{d}/arc-agi_evaluation_{k}.json")) for k in ("challenges", "solutions"))
json.dump({**ch, **{t + "b": v for t, v in ch.items()}}, open(f"{d}/arc-agi_test_challenges.json", "w"))
json.dump({**so, **{t + "b": v for t, v in so.items()}}, open("e240_solutions.json", "w"))
EOF
# after the run: pass@2 per task = mean over its test inputs of (attempt_1 or attempt_2 == solution)
python3 -c "import json; s=json.load(open('e240_solutions.json')); b=json.load(open('work/submission.json')); print(sum(sum(y in (b[t][k]['attempt_1'], b[t][k]['attempt_2']) for k, y in enumerate(ys)) / len(ys) for t, ys in s.items()), '/', len(s))"
```

**Caveats.**
- **Clock.** The clock starts at the container start: the start time of PID 1, if that is less than 2 h before the notebook started. Otherwise it starts with the notebook.
- **Teardown.** The final cell runs `pkill -9 -f 'VLLM::'` and `pkill -9 -f 'vllm serve'`. Use a container or a dedicated machine.
- **Fixed paths.** `/tmp/fn_py312`, `/tmp/fnov` (the patched vLLM overlay) and `/tmp/fn/vcache` are fixed, so run one copy per machine.
- **Speed.** Kaggle's L4s share PCIe Gen3 uplinks in pairs. A machine with faster host links generates more tokens per hour, so its scores are optimistic.

## 4. Launcher settings of RC8 and RC9

The notebook passes `LAUNCHER[mode]` to `kaggle_fn.sh` as environment variables. The per-mode values are in `build/notebook_config_rc8.json` and `build/notebook_config_rc9.json` (`launcher`). The release knobs were added to every mode (`knobs`). The two releases differ only in the rows marked RC8 / RC9.

| Setting | Value | Meaning |
|---|---|---|
| `DT` | 42150 / 7200 / 2400 (rerun / eval / save) | Solver deadline in seconds after container start. |
| `KILLBY` | 42600 / 7650 / 2600 | Hard stop for the solver and the keepalive. The notebook stops the launcher at `KILLBY` + 180 s. |
| `MARGIN`, `ENDG` | 900, 600 / 300, 300 / 240, 240 | New sessions are sized to end `MARGIN` seconds before the deadline. `ENDG` is the earliest the endgame cut can come. |
| `MINCAP` | 65536 / 65536 / 16384 | Smallest generation budget a new session may start with. |
| `SKIP48` | 0 | The MTP layer's KV cache is FP8 too (BF16 there cost 4.8–7%). |
| `POLL_S` | 2 | Health-poll interval while the server boots. |
| `FN_CACHE_TAR` | `{code}/fn_cache.tarbin` | Compile cache, restored before the server starts. |
| `KV_PACK` | 1 | Recurrent-state and indexer cache groups packed from 6 to 4: +6.3% tokens/s. |
| `FP8MM` | 1 | Triton FP8 GEMMs for small batches in the online-FP8 dense layers. |
| `L4CFG` | 1 | L4-tuned Marlin and BF16 GEMM tiles from `k2_l4cfg.json`: +2.0%. |
| `SILU_LOCAL` | 1 | SiLU·mul only on the rows routed to this rank's experts. |
| `SAMPLE_SHARD` | 1 | Batch-sharded sampling: +2.6%. |
| `CONC` | 48 | Sessions in flight, 24 per engine (launcher default 44). A session starts only while the projected contexts fit the KV cache. Going to 48 added 3–7% tokens. |
| `IMPORT_FIX` | alias | Imports carried into later cells can no longer break them, and `ak` resolves as an alias of `arckit`: +2.25 points of right answers per session [+0.60, +3.85]. |
| `TB_FIX` | on | Tracebacks of cells that ran with carried code use the model's own line numbers (about 30% pointed at the wrong line before): +0.88 points of verification on the affected sessions. |
| `ANSWER_ORDER` | shape | A session's answer grids are re-assigned to test inputs by predicted output shape when the `ak` size rules agree. |
| `LATE_PF` | 1 | Late prefetch: once the loader signals, read the expert files into the page cache and drop the n-gram shards already in shared memory. With the read order and the compile cache, boot went from 592 s to 245 s. |
| `DRAIN` | 1 | Measured endgame: the cut comes 3–10 min before the deadline, set from the generation rate. Running sessions finish their chunk and answer, and both data-parallel engines are kept busy to the end: +1–3% tokens. |
| `FN_HOST_DP_FUSE` | 1 | One fused host all-gather for all tensors of a step. |
| `FN_HOST_LOGITS` | 1 | The logits gather goes through host memory too. |
| `FN_MOE_ROUTER_FIRST` | 1 | Router gate before the shared-expert fork: about 0.25 ms per step. |
| `FN_GDN_BA_OVERLAP` | 1 | Gated DeltaNet `in_proj_ba` on a side stream, overlapped with `in_proj_qkvz`: about 0.2 ms per step. |
| `FN_MOE_HYB_FUSE` | 2 | The 3-bit expert group runs on a side stream concurrently with the 4-bit Marlin group, with the cast and add fused into the combine: −36 to −51 µs per layer. |
| `FN_MOE_CHUNK` | 2176 | MoE chunk of 2,176 tokens: +2.8% KV. |
| `FN_LOAD_EC` | 1 | Empty the CUDA allocator cache while weights load. Together with `NCCL_CUMEM_ENABLE=0` it gave +2–6% KV. |
| `NCCL_CUMEM_ENABLE` | 0 | NCCL buffers without cuMem allocations. |
| `FN_QSA_SHARE_TOPK` | 1 | The 12 sparse-attention layers share one top-k buffer: +2.7% KV. |
| `FN_EP_Q8` | 1 | Expert dispatch in int8 with per-128 absmax scales: 0.516× the bytes. |
| `FN_EP_Q8_RS` | 1 | Reduce-scatter in int8 (per-128 groups, fp32 scales, local share in BF16, above 24 tokens per rank) on the expert-parallel group: 1.9–4.0% off each step. |
| `FN_TP_Q8_RS` | RC8: unset; RC9: 1 | The same int8 reduce-scatter on the tensor-parallel group: 0.8–1.4% off each step in some pairs, neutral in others. RC9's `q8_rs.py` picks the groups by name, so the data-parallel group (also of size 2) always stays in BF16. |
| `CACHE_EXPORT` | RC8: 1; RC9: unset | Export the kernels compiled during the run to `fn/cache_delta.tar`. RC8's export went into RC9's compile cache. It does not change what the run computes. |

The notebook also sets `ARC_HOST_AR_BLOCKS=16`, which gives the host all-reduce kernel 16 thread blocks.

**Fixed in `kaggle_fn.sh`.** The launcher also sets:
- the low-bit experts, with 4-bit on Marlin and memory-mapped files (`FN_MOE_LOWBIT`, `FN_MOE_MARLIN4`, `FN_MOE_PAIRS`, `FN_MOE_MMAP`), and RL deltas applied at load (`FN_MOE_DELTA`);
- the shared host n-gram table and embedding (`FN_NGRAM_SHARED`, `FN_NGRAM_MEMFD`, `FN_EMBED_HOST`), and online FP8 dense layers (`FN_DENSE_FP8`);
- the host collectives with peer-to-peer off (`FN_HOST_AR`, `FN_HOST_DP`, `FN_HOST_DP_EP`, `FN_HOST_SP`, `NCCL_P2P_DISABLE`), and the lazy recurrent-state fold (`FN_GDN_LAZY`);
- server-side sessions (`FN_SESSION_EP`, `FN_SESSION_EVICT`);
- the int8 draft head over the 32,768 hottest tokens with top-20 drafts (`FN_MTP_HEAD`, `FN_MTP_HEAD_VOCAB`, `FN_MTP_DRAFT_TOPK`), and the fine-tuned MTP layer (`FN_MTP_OVR`);
- `vllm serve` with TP2 × DP2 and expert parallelism, 24 sequences per engine and a 122,880-token context;
- FP8 KV, 3 MTP draft tokens with probabilistic drafts and block verification, and full-decode CUDA graphs up to 128 tokens;
- for the solver, 98,304 generated tokens per session, 24 tool calls, a 49,152-token thinking nudge and admission at 92% KV.

## 5. Rebuilding the artifacts

The build tools need Python ≥ 3.11 and the standard library. `code_dataset/delta_build.py` also needs PyTorch: the runtime's `/tmp/fn_py312/bin/python3.12` has it. The runtime build needs `uv`, `objdump` and internet access. `training/` produces the inputs:
- `$CKPT`: the `Qwen/Qwen3.8-Flash-Next-FP8` checkpoint folder.
- `$BASE_BUILD`: the GPTQ expert build (`lowbit-LNN.pt`, `router-LNN.pt`, `keep.json`).
- `$RL_BUILD`: the same layout after RL.
- `$MTP_DELTA`: the draft-head export (`mtp_ovr.safetensors`, `scales-L48.pt`, `router-L48.pt`, `manifest.json`).

Each Model folder needs the Qwen licence as `LICENSE`: `cp ../MODEL_LICENSE <folder>/LICENSE`. `make_model_parts.py` hard-links the files into parts, writes the Kaggle metadata and `manifest.json`, and prints the `kaggle models` commands. `upload_model_parts.py --stage <parts>` runs those commands one part at a time. It skips parts that are already complete, and polls until Kaggle lists every file at its exact size, because the Kaggle CLI exits 0 even when the server rejects an upload. `--dry-run` shows the plan, and `KAGGLE_CLI` selects the CLI binary. The flags below reproduce the published parts exactly.

**5.1 Core.**
```
python3 build/split_base.py $CKPT core/                      # base-* (all but routed experts and n-gram) + ngram-*, 4 GB shards
cp ../MODEL_LICENSE core/LICENSE                             # if the checkpoint had none
python3 build/make_model_parts.py core/ parts_core/ --owner YOU --slug qwen38-flash-next-fp8-core --tag w --ple-glob 'ngram-*'
python3 build/upload_model_parts.py --stage parts_core/
```
**5.2 Expert build.**
```
python3 build/make_model_parts.py $BASE_BUILD parts_experts/ --owner YOU --slug qwen38-flash-next-experts --tag k320-q34 --small-mb 1
```
**5.3 RL delta.** The delta holds the RL-tuned scales and router rows only (1.3 GB instead of 35.6 GB). Check that base plus delta rebuilds the RL build tensor for tensor, then split it:
```
python3 code_dataset/delta_build.py make $BASE_BUILD $RL_BUILD delta_rl/
python3 code_dataset/delta_build.py build $BASE_BUILD delta_rl/ rebuilt/ && python3 code_dataset/delta_build.py check rebuilt/ $RL_BUILD
cp ../MODEL_LICENSE delta_rl/LICENSE
python3 build/make_model_parts.py delta_rl/ parts_delta/ --owner YOU --slug qwen38-flash-next-experts --tag k320-q34-r11c-d --small-mb 0 --max-files 49
```
**5.4 Draft head.**
```
cp ../MODEL_LICENSE $MTP_DELTA/LICENSE
python3 build/make_model_parts.py $MTP_DELTA parts_mtp/ --owner YOU --slug qwen38-flash-next-experts --tag k320-q34-r1b-mtp2 --small-mb 0
```
**5.5 Offline runtime.**
```
uv python install 3.12.14
python3 build/build_fn_env.py work/        # requirements-fn032.lock.txt + FlashInfer's sm89 JIT cache -> work/fn_py312.tar
```
- The tool rewrites every local build path to `/tmp/fn_py312` and recompiles the `.pyc` files.
- It fails if any local path is left, if a library needs glibc above 2.35, or if vLLM is not 0.32.0.
- To upload, prepare two folders:
  - `meta/`: `build/runtime_model/model-metadata.json`, with your `ownerSlug`.
  - `py312-cu130/`: `fn_py312.tar` renamed to `fn_py312.tarbin`, `fn_py312.tarbin.sha256` (`sha256sum fn_py312.tarbin | cut -d' ' -f1`), `build/runtime_model/LICENSE` and `build/runtime_model/model-instance-metadata.json`, with your `ownerSlug`.

  Then run `kaggle models create -p meta/` and `kaggle models instances create -p py312-cu130/`. A rebuilt archive has a different size, so update the notebook configs.

**5.6 Code dataset.**
```
python3 build/pack_code_dataset.py ds/ --fn-cache fn_cache.tarbin --dataset-id YOU/flash-next-arc-code   # RC8; add --release rc9 for RC9
kaggle datasets create -p ds/            # later versions: kaggle datasets version -p ds/ -m "..."
```
`pack_code_dataset.py` copies the flat files and packs `code_dataset/fnext/` into `fnext.tarbin`:
- in GNU format, sorted by name, owner `root:0`, mtime 0 (RC8's 14 overlay entries keep the timestamp they were published with), modes 0664 for files and 0775 for directories;
- with `--release rc9` it first overlays `code_dataset/rc9/`;
- it compares every file with the release's `MANIFEST.md5`;
- from an unmodified checkout, all 16 files, `fnext.tarbin` included, match dataset version 11 (RC8) or 13 (RC9) byte for byte.

To change the engine, edit `code_dataset/fnext/` and re-pack. Without `--fn-cache` the dataset still works, but the server compiles its Triton kernels at startup.

**5.7 Compile cache.** `fn_cache.tarbin` holds the Triton and vLLM kernels compiled for the L4 shapes this workload hits. RC9's has 425 kernel directories (3,353 files) and compiled only 3 kernels during its run; RC8's smaller cache left 81 to compile during the run. To refresh it after a change:
1. Render a notebook with `"CACHE_EXPORT": "1"` added to `knobs` (section 5.8), and run it once as an eval-mode save run. Kernels compiled during that run land in `fn/cache_delta.tar`. To start cold, ship an empty cache: `mkdir -p empty/triton && tar -C empty -cf fn_cache.tarbin .`
2. Merge the deltas. Only complete kernel directories are taken, group files are re-pointed to `/tmp/fn/vcache/triton`, and files containing your home path are refused (`FORBID=<regex>` adds patterns):
   ```
   python3 build/merge_fn_cache.py fn_cache.tarbin fn_cache_new.tarbin out/fn/cache_delta.tar [more deltas]
   ```
3. Re-pack the dataset with `--fn-cache fn_cache_new.tarbin`. Render the final notebook without `CACHE_EXPORT`.

**5.8 Notebook.** `build/notebook_config_rc8.json` describes version 5 (RC8) and `build/notebook_config_rc9.json` version 7 (RC9):
- the Models with their parts and exact file sizes, and the sizes of the rebuilt expert build (`experts_rebuilt_files`, used only by the rebuild fallback);
- the dataset, the launcher settings, the knobs and `eval_dt` (null turns eval mode off);
- the kernel id and title.

```
python3 build/build_notebook.py build/notebook_config_rc8.json out_nb/ --check     # or notebook_config_rc9.json
kaggle kernels push -p out_nb/
```
- `build_notebook.py` uses `notebook/rc8/flash-next-arc-eval.ipynb` as the template (RC9's gives the same output; only the configuration cell differs). From the config it re-renders only the heading's part count and the `MODELS`, `MODEL_FILES`, `REBUILT_FILES`, `RUNTIME_DATASET` and `LAUNCHER` lines. It validates the knobs and compiles every cell.
- With the shipped configs, the outputs are byte-identical to the notebooks pushed as versions 5 and 7 (Kaggle stores them re-serialized; the cell sources are identical). The `kernel-metadata.json` it writes is byte-identical to the one pushed with each.
- After re-uploading the Models under your account, change `owner`, `runtime_dataset` and `kernel_id`. A model's `parts` and `files` can be replaced by `"manifest": "<parts dir>/manifest.json"` from `make_model_parts.py` (the path is relative to the config file).

**5.9 Validation.** See section 2: `kaggle kernels output`, then `build/check_validation.py`. That script also reads a plain-text notebook log from a local run.

## 6. Module map

### `code_dataset/fnext/`: the vLLM 0.32 plugin

`kaggle_fn.sh` runs `fnext/apply.sh`, which builds a patched vLLM overlay in `/tmp/fnov`. Each subpackage registers its features behind environment flags, so every flag that is unset leaves stock vLLM behaviour. Appendix F of the writeup describes the design.

| File | Role |
|---|---|
| `apply.sh` | Hard-links the installed vLLM into the overlay, applies each subpackage's `*.patch`, copies the modules to `fnext_plugins/` and registers them as a vLLM plugin. |
| `apply_patch.py` | Small unified-diff applier with fuzzy hunk search (no `patch` binary needed). |
| `base/plugin.py` | Registers the base features: host all-reduce, host collectives, int8 transport, lazy recurrent state, side-stream overlap, sharded sampling and the draft model's expert count. |
| `base/host_ar.py` | Tensor-parallel all-reduce as one Triton kernel over `cudaHostRegister`ed shared memory: system-scope flags, fp32 accumulation, safe inside CUDA graphs. |
| `base/host_dp.py` | Host-staged all-gather and reduce-scatter through 32 MB host slots, for expert-parallel dispatch, sequence parallelism and the logits gather; NCCL fallback. |
| `base/host_dp_fused.py` | Fused multi-tensor host all-gather (`FN_HOST_DP_FUSE`). |
| `base/q8_dispatch.py` | int8 expert dispatch with per-128 absmax scales (`FN_EP_Q8`). |
| `base/q8_rs.py` | int8 reduce-scatter, enabled per group: expert-parallel or tensor-parallel, never data-parallel (`FN_EP_Q8_RS`, `FN_TP_Q8_RS`). |
| `base/gdn_lazy.py` | Defers the Gated DeltaNet state fold of accepted draft tokens into the next verify step: −34% state-update time (`FN_GDN_LAZY`). |
| `base/gdn_ba_overlap.py` | `in_proj_ba` on a side stream, concurrently with `in_proj_qkvz` (`FN_GDN_BA_OVERLAP`). |
| `base/load_ec.py` | Empties the CUDA cache during weight loading, so fragmentation does not shrink the KV cache (`FN_LOAD_EC`). |
| `base/qsa_topk.py` | One top-k buffer shared by the 12 sparse-attention layers (`FN_QSA_SHARE_TOPK`). |
| `base/fnplug.py` | Loader hooks used while compressing: keep-set slicing, BF16 overrides and routing capture. Inactive in the submission. |
| `base/link_gate.py`, `base/0001-dev-trunc-loader.patch` | Experiment hooks (a host-link gate word, truncated-layer loading). Off. |
| `boot/fcache.py` | Compile-cache helpers: re-point Triton group files after restore (`fixgrp`), page-cache statistics and drops. |
| `boot/late_prefetch.py` | After the loader signal, reads the expert files into the page cache and drops the n-gram shards (`LATE_PF`). |
| `kv/0001-vllm58863-recoverssm.patch` | Extends vLLM's RecoverSSM speculative decoding (upstream for Kimi-K3's KDA layers) to the model's Gated DeltaNet and n-gram layers. Rejected drafts are undone from one BF16 checkpoint per sequence: 49% more concurrent sequences. |
| `kv/0002-recoverssm-dp-dummy-padding.patch` | RecoverSSM fix for the padded dummy rows of data-parallel batches. |
| `kv/0003-p1-wide-mixed-graphs.patch` | Optional wider mixed CUDA graphs under RecoverSSM (`FN_PW_WIDE`). Off. |
| `kv/plugin.py` | KV-cache options: state packing (on) and the sparse-attention indexer's top-k tie-break and KV-scale calibration tools (off). |
| `kv/state_pack.py` | Packs the recurrent-state and indexer cache groups from 6 to 4 without widening a block (`KV_PACK`). |
| `moe/plugin.py` | Registers the MoE features. |
| `moe/fnmoe.py` | Serves the pruned low-bit build (`FN_MOE_LOWBIT`): memory-maps `lowbit-LNN.pt`, drops the checkpoint's expert tensors, slices routers to the keep set, applies the RL and draft-head deltas at load (`FN_MOE_DELTA`) and places experts across the 4 GPUs. |
| `moe/lowbit_kernels.py` | Low-bit MoE layers: 4-bit experts on Marlin, 3-bit experts on the Triton GEMM, the hybrid that runs both, and the combine kernels. |
| `moe/lowbit2.py` | Triton grouped GEMM for the 3-bit experts (planar layout, `lop3` dequantization). |
| `moe/marlin_part.py` | Marlin wrapper for the 4-bit group (repacks the GPTQ codes) and the local SiLU·mul kernel (`SILU_LOCAL`). |
| `moe/hyb_fuse.py` | Runs the 3-bit part concurrently on a side stream, fuses Marlin's cast and partial sum into the combine, and chunks the MoE (`FN_MOE_HYB_FUSE`, `FN_MOE_CHUNK`). |
| `moe/router_first.py` | Router gate before the shared-expert fork. It refuses to install if vLLM's source differs from the expected one. |
| `moe/dense.py` | Online per-channel FP8 for the Gated DeltaNet projections, shared experts and LM head (`FN_DENSE_FP8`), and Triton small-batch FP8 GEMMs (`FP8MM`). |
| `moe/bf16mm.py`, `moe/l4cfg.py` | Triton BF16 GEMM for small batches, and the loader for L4-tuned Marlin/BF16 tiles (`L4CFG`, `k2_l4cfg.json`). |
| `mtp/plugin.py`, `mtp/fnmtp.py` | Draft-head features: int8 LM head over the 32,768 most-used tokens (`hot32k_tr.pt`), top-20 draft distributions, and the fine-tuned MTP layer override (`FN_MTP_*`). |
| `ngram/plugin.py`, `ngram/host_table.py` | One shared host copy (memfd) of the 47.7 GiB FP8 n-gram table (320M rows × 160) and of the 1.18 GiB token embedding. Each rank loads a quarter; a Triton zero-copy gather runs on a side stream. |
| `session/plugin.py`, `session/session_ep.py` | Server-side sessions: `/arc/session`, `/arc/session/push`, `/arc/session/close`, `/arc/session/stats`. A session keeps its KV on the server and appends only the tool output: prompt processing went from 6.1 to 0.2 tokens per generated token. Includes a context guard and eviction on KV deadlock. |
| `session/resume_fix.py` | Exact session resumes: caps the recurrent-state fold when a session continues. |
| `session/detok_rope.py` | Incremental detokenizer that keeps the output text in pieces, so long outputs do not cost quadratic string copies (`FN_DETOK_ROPE`). |

### `code_dataset/`: agent, launcher and helpers

| File | Role |
|---|---|
| `kaggle_fn.sh` | Launcher. It builds the overlay, restores the compile cache and starts the supervised `vllm serve` (up to 3 restarts). It then starts the late prefetch and runs `repl_solve.py` against the deadline. When the endgame flag appears, it keeps both data-parallel engines busy. It fills any missing task with the input. |
| `fn_runtime.json` | Marker the notebook uses to find the dataset, plus the launcher's build options. |
| `repl_solve.py` | The solver (`--backend fnext`). It schedules tasks (fewest sessions first, settled tasks last, at most 2 in flight per task) and admits sessions by KV and projected context. It sizes budgets, adds carry notes, handles the deadline and the drain, votes, breaks ties and writes the submission atomically. |
| `repl_session.py` | One multi-turn tool session in Qwen's native tool-call format: chunked generation over the session API, tool calls, the thinking nudge and the forced answer. |
| `repl_prompt.py` | Task prompt (grids as digit rows), the `python` tool schema, the `ak` API sheet, and parsing of code and answer grids. |
| `repl_sandbox.py` | Runs model programs in a fresh `python -I` process as an unprivileged user (2 GiB, CPU and wall-clock limits, an audit hook against processes, sockets and outside writes). Checks `transform` on every training pair and returns a masked diff. |
| `repl_vote.py` | Votes the verified programs' test outputs into `attempt_1` and `attempt_2`. |
| `repl_d3fmt.py`, `repl_dsession.py` | Dots3-Note's tool-call format and session (`--backend dots3`). Not used by this submission. |
| `arckit.py` | The `ak` library: describe, check, objects, diffs, symmetries, periodicity, size rules, colour maps and the 11 content invariants that gate verification. |
| `delta_build.py` | `make` / `build` / `check` an expert delta (scales and router rows) against the base build. The notebook uses `build` in its fallback path. |
| `fn_preflight.py` | Checks the deltas with the server's own loader code before the server applies them at load. |
| `k2_l4cfg.json` | L4-tuned Marlin and BF16 GEMM tiles. |
| `hot32k_tr.pt` | The 32,768 most-used token ids, for the int8 draft head. |
| `fn_cache.tarbin` | Not in git: the compile cache from dataset version 11 or 13 (md5 in the release's `MANIFEST.md5`). |
