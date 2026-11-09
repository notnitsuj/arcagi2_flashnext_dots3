# Dots3-Note on Kaggle

This folder is the Dots3-Note submission as it ran on Kaggle: the notebook in two configurations, the runtime dataset it ran with (version 5), and the tools that rebuild every Kaggle artifact. The engine source (the vLLM plugin `arcstats`, its CUDA kernels and the vLLM 0.30 patch) is in [`../engine/`](../engine/).

| Path | What it holds |
|---|---|
| [`notebook/best_throughput/`](notebook/best_throughput/) | Notebook version 4 and its `kernel-metadata.json`, byte-identical to what `kaggle kernels pull` returns. The fastest Dots3-Note build (v29). |
| [`notebook/best_score/`](notebook/best_score/) | The same notebook running the v22c configuration, our best Dots3-Note score, on the same runtime. Built by `build/build_notebook.py`. |
| [`runtime_dataset/`](runtime_dataset/) | The 68 launcher, solver, mode, table and gate files of runtime dataset version 5, verbatim. [`MANIFEST.md5`](runtime_dataset/MANIFEST.md5) holds the md5 of all 119 published files. |
| [`build/`](build/) | Notebook builder with presets, dataset packer, bundle and environment builders, Model splitter, mocked notebook tests. |

Commands below run from this folder (`kaggle/`).

Contents: [1 Artifacts](#1-artifacts) · [2 The two presets](#2-the-two-presets) · [3 Running on Kaggle](#3-running-on-kaggle) · [4 Running on your own 4×L4 machine](#4-running-on-your-own-4l4-machine) · [5 Rebuilding the artifacts](#5-rebuilding-the-artifacts) · [6 Module map](#6-module-map)

## 1. Artifacts

**Notebook** `notnitsuj/dots3-note-arc-agi-2` (46 KB, 20 cells: configuration, helpers, six stages):

| Version | Build | Runtime dataset | Public score |
|---|---|---|---|
| 1 | v27: the configuration of the earlier script notebook, now loading the checkpoint from Kaggle Models | 3 | — |
| 2 | version 1 plus a storage-diagnostics cell | 3 | 58.89 (an identical rerun: 53.19) |
| 3 | v29 with 408,700-row caches | 4 | not submitted: the server failed at start, because the row-cache sizes were not multiples of the 8 cache ways |
| 4 | v29: v27 plus the compact GPU row cache, the preallocated prefill output, larger row caches, solver deadline 42,150 s and budget margin 300 s; 5–8% more tokens/s than v27 in paired runs | 5 | 55.28 |

Our best Dots3-Note score, 63.47, came from v22c, submitted from an earlier notebook that is not published. The `best_score` preset runs its configuration on runtime version 5 (section 2).

**Runtime dataset** `notnitsuj/dots3-note-arc-runtime`:

| Version | Content | Used by |
|---|---|---|
| 1, 2 | the v27 code bundle without the Python environment (118 files, 2.05 GB) | earlier uploads |
| 3 | the same plus `arc_py312.tarbin` (119 files, 10.27 GB) | notebook versions 1 and 2 |
| 4 | the v29 code with 408,700-row caches | notebook version 3 |
| 5 | the v29 code (119 files, 10.27 GB = 9.57 GiB) | notebook version 4 and both presets here |

**Model** `notnitsuj/dots3-note-lb144r` (PyTorch, version 1 of every variation): Dots3-Note pruned to 144 of 256 experts per MoE layer, experts quantised with GPTQ to 3 and 2 bits. 63 files, 61.3 GiB, split into four variations:

| Variation | Files | Size |
|---|---|---|
| `lb144r-p1` | config, tokenizer and chat template, `README*.md`, `pruning.json`, `model.safetensors.index.json`, 14 expert files `lowbit-LNN.pt` | 16.99 GB |
| `lb144r-p2` | `model-00000.safetensors`, 10 expert files | 16.74 GB |
| `lb144r-p3` | `model-00001.safetensors`, 10 expert files | 16.29 GB |
| `lb144r-p4` | `model-00002.safetensors`, 11 expert files | 15.80 GB |
| `lb144r`, `probe` | placeholders left from the upload (`config.json`, `LICENSE`); not used | |

Every variation carries the Apache-2.0 `LICENSE` of the base weights. The notebook links the four parts into one directory and checks all 63 files against their exact sizes (`CKPT_FILES`) before anything starts. [`build/model_parts.json`](build/model_parts.json) lists the files, sizes and parts.

Every published notebook version calls the runtime a "private dataset" in its overview. The dataset is public; the text was left unchanged so that the files stay byte-identical to what ran. The `best_score` notebook says "dataset".

## 2. The two presets

Both presets run the same notebook code against the same runtime (dataset version 5). They differ only in the mode file that the launcher (`run_l4.py`) reads.

- **`best_throughput`** passes the runtime's own `mode_submit.json` (the release mode, sha256 `f97ba865…` without its note) as is.
- **`best_score`** checks that the runtime's `mode_submit.json` is that release mode (the release check and `RELEASE_MODE_SHA256`). It then applies `PRESET_CHANGES` from its configuration cell and checks the result against every pinned value and `MODE_SHA256` (`e0c4bf7a…`). It writes the result to `mode_best_score.json` in the working directory, reads it back, and passes it to the launcher through `ARC_MODE_FILE`. The save-run mode is derived from it as in version 4: drop `save_overrides`, apply the smoke-test settings.

The changes are in [`build/presets/best_score.json`](build/presets/best_score.json), and the full mode the notebook writes is in [`build/presets/best_score_mode.json`](build/presets/best_score_mode.json):

| Setting | best_score | best_throughput | What it does | Outputs |
|---|---|---|---|---|
| `ARC_AR_LB`, `ARC_AR_LB_MAIN`, `ARC_AR_LB_EF`, `ARC_AR_LB_HAD` | off | `1`, `4f/5`, `1`, `8` | Lower-bit host all-reduce: 4-bit FP4 (e2m1) peer payloads and a 5-bit reduced sum, with cross-layer error feedback and an 8-point Hadamard rotation. Off, every all-reduce is int6 (`ARC_HOST_AR_BITS=6`, both presets). | changes the numerics (the only such setting) |
| `ARC_OFFLOAD_CCACHE` | off | `1` | GPU row-cache lines keep the 400-byte compact host row instead of the 576-byte FP8 row, so 1.42× the lines fit in the same memory. | bit-exact |
| `ARC_OFFLOAD_CACHE_ROWS_L` | layer 0: 142,000; others: 232,000 | layer 0: 202,000; others: 362,600 | GPU row-cache lines per offloaded layer (multiples of the 8 cache ways). | bit-exact (hit rate only) |
| `ARC_OFFLOAD_PF_OUT` | off | `1` | The offloaded layers' prefill attention writes into one preallocated output (−32 MiB transient). | bit-exact |
| `ARC_KVB_REBUILD` | off | `1` | Keeps the block-FP8 `kv_b_proj` weights off the GPU and rebuilds each one right before use. | bit-exact |
| `ARC_PENALTY_LAZY` | off | `1` | Allocates vLLM's penalty counters (37 MiB) only when a request uses penalties; the solver never does. | bit-exact |
| `ARC_MLA_WS_TOKENS` | off | `16384` | Sizes the MLA prefill workspace for 16,384 tokens instead of 65,536 (−102 MiB). | bit-exact |
| `ARC_PARAM_COMPACT` | off | `1` | Re-packs the weights after load, so the holes freed during loading are returned (about 215 MiB). | bit-exact |
| `deadline_total_s` | 41,400 | 42,150 | Solver budget from the launcher start. | schedule only |
| `startup_guard.deadline_abs_s` | 41,700 | 42,150 | Cap on the solver deadline, from the container start (S0). | schedule only |
| `--budget-margin-s` (solver) | not passed: 600 | 300 | The deadline planner sizes the remaining work to end this long before the deadline. | schedule only |

Everything else is identical: the other 38 environment flags, the `vllm serve` arguments (TP4, MTP-3, FP8 KV, 262k context, CUDA graphs up to 256), the solver caps and concurrency, the KV pool, the pinned kernel choices and the fine-tuned delta. "Bit-exact" settings change memory or speed, never tokens. "Schedule only" settings change how long the solver runs, not what a request returns.

**Why each value.** The builder checks the derived mode against [`build/presets/v22c_mode.json`](build/presets/v22c_mode.json) (the v22c mode file as it shipped; only its note is new). Every key is equal except those listed here:

| Key | best_score value | Source | Reason |
|---|---|---|---|
| all serving and solver keys | v22c's | v22c | the configuration that scored 63.47 |
| `deadline_total_s`, no `--budget-margin-s` | 41,400; margin 600 s | v22c | v22c's solver budget. Its solver had the 600 s margin built in; the later `solve.py` takes it as `--budget-margin-s`, default 600. |
| `startup_guard.deadline_abs_s` | 41,700 | notebook versions 1–2 | The cap that went with a 41,400 s budget. The binding deadline stays launcher start + 41,400 s, as in v22c, unless the launcher starts more than 300 s after the container. |
| `startup_guard`, other keys | session anchor, one server restart before ready, hang 2,400 s, hard stop S0 + 42,600 s | v27 onward | start-up safety; changes nothing in a normal run |
| `prefetch` | local staging with the robust copier (32 threads, 64 MB segments, retries), read block 8 MB | v27 onward | storage robustness (Kaggle's input mounts can drop to 10–100 MB/s); the same bytes are loaded |
| `triton_cache_tar`, `ready_poll_s` | `triton_cache_lv5.tarbin`, 2 s | v27 onward | start-up time only; a kernel missing from the cache is compiled at start |

**Evidence that the serving numerics are v22c's.** On a 64-request test (8k–32k-token prompts, 6,144 new tokens each), the v27 runtime with the lower-bit all-reduce off returned the same token ids and log-probabilities as v22c for all 393,216 generated tokens. The memory reclaims were on in that test. The clean notebook with runtime version 3 passed the same test. The compact row cache, the preallocated prefill output and the memory reclaims are each bit-exact by their own kernel and serving-path tests. The `best_score` notebook itself has not been run on Kaggle; v22c's 63.47 came from the earlier notebook.

## 3. Running on Kaggle

**Inputs** (listed in each `kernel-metadata.json`):
- the dataset `notnitsuj/dots3-note-arc-runtime` (version 5);
- the four Model variations `notnitsuj/dots3-note-lb144r/PyTorch/lb144r-p1..p4/1`;
- the competition `arc-prize-2026-arc-agi-2`.

The machine shape is `NvidiaL4` (4× L4), with internet off. `kernel-metadata.json` records the Docker image digest that version 4 ran with. The dataset is not pinned to a version: a push attaches the latest one, and the notebook refuses any runtime whose `mode_submit.json` is not version 5's.

**best_throughput.** Open notebook version 4 and choose *Copy & Edit*; the inputs stay attached. Then *Save Version* (a save run) or *Submit to competition*.

**best_score.** Use either of these routes:
- On kaggle.com, *Copy & Edit* version 4, then *File → Import Notebook* with `notebook/best_score/dots3-note-arc-agi-2.ipynb`.
- From the CLI, render it under your own id and push it:
  ```
  python3 build/build_notebook.py --preset best_score --kernel-id YOU/dots3-note-arc-agi-2-best-score --out nb_best_score/
  kaggle kernels push -p nb_best_score/
  ```

**Run modes.** The notebook selects its mode from the environment:

| Mode | When | Tasks | Deadlines | Output |
|---|---|---|---|---|
| Scored rerun | Kaggle sets `KAGGLE_IS_COMPETITION_RERUN` after *Submit* | hidden test set | best_score: launcher start + 41,400 s, at most S0 + 41,700 s; best_throughput: S0 + 42,150 s; both: hard stop at S0 + 42,600 s (Kaggle's limit is 43,200 s) | `submission.json` |
| Save run | *Save Version* or `kaggle kernels push` | public evaluation split | 1,200 s solver deadline; the launcher is stopped after 1,800 s | `smoke_evaluation.json` and a local score line; `submission.json` stays the input echo |

S0 is the container start, taken as the start time of PID 1 if that is less than an hour before the notebook started.

**Fail-safes.**
- A valid `submission.json` (the input echo) exists before the server starts.
- The server is restarted once if it dies or hangs before it is ready.
- At the hard stop, the launcher kills its process tree and repairs the submission: every missing or malformed attempt becomes the input grid, and valid attempts are kept.
- The notebook kills the launcher itself if it is still alive 180 s after its hard stop, then checks `submission.json` once more.

Version 4's save run: the server was ready 308 s after it started (S0 + 581 s), the short smoke solve averaged 882 generated tokens/s, and at least 931 MiB of GPU memory stayed free.

## 4. Running on your own 4×L4 machine

The notebook only orchestrates: its cells use the Python standard library. The launcher re-executes itself under the runtime's own Python 3.12 environment.

**Needs.**
- **GPUs:** 4× L4 (24 GB, sm89), with a driver that supports CUDA 13.0.
- **Memory:** host RAM comparable to Kaggle's 188 GB (a 173 GiB cgroup). The offloaded KV rows (about 83 GiB) live in `/dev/shm`, so it must hold at least that (Kaggle's holds 86 GB). In Docker, use `--gpus all --ipc=host --shm-size=96g --ulimit memlock=-1`.
- **Disk:** about 90 GB free in `/tmp`, for the staged checkpoint and tars (70.8 GiB) and the unpacked environment (7.8 GiB).

**1. Build an input tree like `/kaggle/input`.** The notebook searches below `ARC_INPUT` for:
- the runtime, by its `run_l4.py` (exactly one);
- the environment archive `arc_py312.tarbin`;
- `arc-agi_test_challenges.json`;
- each Model variation, by a `<slug>/pytorch/<variation>/<version>/` path (case-insensitive).

```
IN=$PWD/kaggle_input
kaggle competitions download arc-prize-2026-arc-agi-2 -p $IN/competitions/arc-prize-2026-arc-agi-2 && (cd $IN/competitions/arc-prize-2026-arc-agi-2 && unzip -q *.zip)
kaggle datasets download notnitsuj/dots3-note-arc-runtime -p $IN/datasets/notnitsuj/dots3-note-arc-runtime --unzip
for p in 1 2 3 4; do kaggle models instances versions download notnitsuj/dots3-note-lb144r/pyTorch/lb144r-p$p/1 \
  -p $IN/models/notnitsuj/dots3-note-lb144r/pytorch/lb144r-p$p/1 --untar; done
```

If the whole-dataset download fails, fetch the files one by one with `kaggle datasets download notnitsuj/dots3-note-arc-runtime -f <name>`. Check the folder with `(cd <folder> && md5sum -c <this folder>/runtime_dataset/MANIFEST.md5)`. You can also assemble it from this repository (section 5.2).

**2. Run it.**
```
export ARC_INPUT=$IN ARC_WORK=$PWD/work            # default /kaggle/input and /kaggle/working
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output-dir work notebook/best_score/dots3-note-arc-agi-2.ipynb
```

| Variable | Read by | Meaning |
|---|---|---|
| `ARC_INPUT`, `ARC_WORK` | notebook and launcher | input tree, and the working directory for logs and `submission.json` |
| `KAGGLE_IS_COMPETITION_RERUN` | notebook and launcher | set: the scored path on `arc-agi_test_challenges.json`. Unset: the 1,200 s save run on the evaluation split, scored against `arc-agi_evaluation_solutions.json` if present. |
| `ARC_CKPT_MERGE_DIR` | notebook | where the four parts are linked into one checkpoint directory (default `/tmp/arc_ckpt_parts`) |
| `NB_DRY_RUN=1` | notebook | resolve and check everything, write the mode files, launch nothing (used by the tests) |

For a 12-hour run on public data, copy `arc-agi_evaluation_challenges.json` over `arc-agi_test_challenges.json`, set `KAGGLE_IS_COMPETITION_RERUN=1`, and score `work/submission.json` against the evaluation solutions afterwards.

**Caveats.**
- **Fixed paths.** `/tmp/arc_py312` (the environment), `/tmp/cuda13`, `/tmp/arc_ckpt` (the staged copy), `/tmp/culink`, `~/.triton/cache` and `/dev/shm/arcoff_*` are fixed. Run one copy per machine.
- **Clock.** On a machine that has been up for more than an hour, S0 is the notebook start.
- **Speed.** Kaggle's L4s share PCIe Gen3 uplinks in pairs. A machine with faster host links generates more tokens per hour, so its scores are optimistic.

## 5. Rebuilding the artifacts

The tools need Python ≥ 3.11 and the standard library, plus `uv`, `gcc` and internet for the environment and kernel builds.

**5.1 Notebook.**
```
python3 build/build_notebook.py --preset best_throughput --check   # identical to notebook version 4, byte for byte
python3 build/build_notebook.py --preset best_score                # notebook/best_score/ and build/presets/best_score_mode.json
```
- The builder generates every cell from the runtime's `mode_submit.json`, [`build/model_parts.json`](build/model_parts.json) and the preset. The configuration cell pins every value and the hash of the whole mode, so the notebook refuses any other runtime.
- Kaggle stores a pushed notebook re-serialized (compact JSON, one source string per cell), and the builder writes that form. For `best_throughput`, its output is byte-identical to version 4:
  - the notebook's sha256 is `e43fe1ae…`;
  - `kernel-metadata.json`'s is `6984c655…`.
- `--kernel-id`, `--title` and `--owner` re-target the notebook and its sources.

**5.2 Runtime dataset.** From this repository plus the published binaries:
```
python3 build/pack_runtime_dataset.py ds/ --binaries <folder with the dataset version 5 binaries> --dataset-id YOU/dots3-note-arc-runtime --strict
kaggle datasets create -p ds/      # later versions: kaggle datasets version -p ds/ -m "..."
```
- `pack_runtime_dataset.py` copies the 68 files of `runtime_dataset/`. It flattens `../engine/arcstats_pkg/arcstats/*.py` into `arcstats__<name>.py`, and the patched vLLM files into `vllm__<path>.py`.
- It adds the binaries and compares every file with `MANIFEST.md5`. Every text file comes out byte-identical. With the published binaries, all 119 files match dataset version 5.
- To rebuild the binaries from source as well, use `build/build_bundle.sh ds/ ENV_TAR [TRITON_CACHE [DELTA]]`.

| Binary | Size | Origin | Rebuild |
|---|---|---|---|
| `arc_py312.tarbin` | 8.22 GB | The offline environment: standalone CPython 3.12.15 unpacked to `/tmp/arc_py312`, with the 200 packages of [`build/requirements-arc-py312.lock.txt`](build/requirements-arc-py312.lock.txt) (vLLM 0.30.0, PyTorch 2.13.0 for CUDA 13.0, Triton 3.7.1, FlashInfer 0.6.18.post1, …) and the `arcstats` plugin. | `build/build_env.sh env.tar`, then `build/build_nb_env.py env.tar ds/arc_py312.tarbin`. The re-pack sets owner 0:0 and makes the interpreter's build prefix and the plugin's install path neutral, then checks that the environment reports the same distributions, imports and sysconfig. A rebuilt archive holds the same packages but not the same bytes (timestamps, compiled files). |
| `cuda_toolkit.tarbin` | 653 MB | CUDA 13.0 headers, `nvcc` and runtime from pip wheels, without the math libraries' shared objects (the launcher links against the environment's own). It unpacks to `/tmp/cuda13`. | `build/build_bundle.sh` step 2 (the wheels are pinned there). |
| `triton_cache_lv5.tarbin` | 105 MB | Triton kernels (1,801 files) compiled for L4 by an earlier run of this stack; restored before the server starts. | Run with `"triton_cache_save": true` in the mode; the launcher writes `triton_cache.tar` to the working directory. Rename it to `.tarbin`, because Kaggle unpacks uploaded `.tar` files. Without the cache, the server compiles these kernels at start. |
| `p1s_delta.safetensors` | 1.26 GB | The fine-tuned delta (expert GPTQ scales, RMSNorms, router), overlaid at load by `arcstats/delta_overlay.py`. | The fine-tuning recipe in [`../training/`](../training/). |
| `arcstats__{host_arc,offload_attn,offload_attn2,swa_attn,topk_select}.so` | 27 MB in all | The compiled kernels of [`../engine/arcstats_pkg/arcstats/csrc/`](../engine/arcstats_pkg/arcstats/csrc/): `nvcc` 13.4 for sm_89 and sm_90 with static cudart, and `gcc` for the host reducer. | `../engine/build_kernels.sh OUT` (`CU13_HOME` = a pip CUDA 13 toolkit; by default the one inside the unpacked environment), then `build/build_nb_bundle.py`, which overwrites the build path recorded in each kernel with a neutral one of the same length. A rebuild gives an identical `host_arc.so`. The four CUDA modules come out at the same size with identical host and device code; they differ only in compiler-generated names (temp-file ids and per-file hashes in symbol names). |

**5.3 Model variations.**
```
python3 build/split_model_parts.py <lb144r checkpoint folder> parts/ --owner YOU    # hard links; LICENSE and metadata in each part
kaggle models create -p parts/ && for p in 1 2 3 4; do kaggle models instances create -p parts/lb144r-p$p; done
```
- The split reproduces the four published variations file for file.
- A single 62 GB variation could not be created: the API returned HTTP 200 with an empty body. Variations of about 15 GB work.
- Push the notebook only when every part lists all its files: check with `kaggle models instances versions files YOU/dots3-note-lb144r/pyTorch/lb144r-p1/1`.
- The checkpoint itself (pruning, GPTQ) comes from [`../training/`](../training/).

**5.4 Tests.**
```
PY=<python with papermill and ipykernel> python3 build/test_notebook.py
```
- The suite runs 95 checks on synthetic inputs: sparse checkpoint files of the exact sizes, random tasks, and a mock launcher that execs itself and ends with the hard stop. Both presets are executed headless with papermill, as Kaggle runs notebooks.
- It checks:
  - the builder output and the 4-part merge;
  - the save-run and scored-rerun modes, and what the launcher reads through `ARC_MODE_FILE`;
  - the release guard and the preset's pins;
  - refusals: a missing part, two versions of a part, two runtimes, a missing file, a wrong file size, a runtime mode that is not version 5's;
  - submission repair, the backstop, the storage diagnostics, and the launcher's checkpoint and mode-file lookups.
- No GPU is used.

## 6. Module map

### `../engine/`

| Path | Role |
|---|---|
| `arcstats_pkg/` | The vLLM plugin package (`pip install`able; entry point `vllm.general_plugins: arcstats = arcstats:register`). Its modules are dataset version 5's `arcstats__*.py` files, byte for byte. Every feature sits behind an environment flag and is off when the flag is unset. |
| `arcstats_pkg/arcstats/csrc/` | CUDA and C sources of the five compiled modules (comments edited only). |
| `build_kernels.sh`, `build_*.sh` | Build the compiled modules (one script each, same nvcc flags as the shipped ones). |
| `vllm_patch/vllm-0.30.0.patch` | The changes to the vLLM 0.30.0 wheel as a unified diff (7 files, 48 lines). The patched files are under `vllm_patch/vllm/`, byte-identical to the dataset's `vllm__*.py`. The launcher copies them over the installed vLLM at every start. |

The patch touches these files:
- `models/dots3_note/nvidia/model.py`: FP8 or BF16 sliding-window KV (`ARC_SWA_FP8`, FP8 in both presets);
- `models/dots3_note/nvidia/mtp.py`: the MTP head keeps its own embedding;
- `models/dots3_note/nvidia/attention.py`: the prefill input is cast only for floating-point `kv_b_proj` weights (FP8 Marlin on L4 repacks them to int32);
- `v1/attention/ops/xpu_mla_sparse.py`: 64-bit row offsets and a KV scale in the portable sparse-attention kernel;
- `model_executor/layers/attention/sparse_mla_attention.py`, `v1/hisparse/runtime.py` and `distributed/kv_transfer/kv_connector/v1/hisparse/worker.py`: sparse-row width and CUDA-graph warm-up tolerance.

| `arcstats` module | Role |
|---|---|
| `__init__.py` | Plugin entry point: installs each feature behind its flag (attention backend, offload, host collectives, low-bit MoE, kernels, sessions, delta, memory reclaims); also the routing-statistics hooks used for pruning. |
| `dsa_sm89.py` | Portable DeepSeek-style sparse attention for Dots3-Note on GPUs without sm90 kernels (L4). |
| `sparse_offload.py` | Sparse-aware KV offload of the sparse-attention layers: KV rows in shared host memory, per-layer W-way GPU row caches, fill, spill, mirror and overlap. |
| `offload_rowfmt.py` | Compact host rows (e2m2: 400 B instead of 576 B per row) and the compact GPU row cache (`ARC_OFFLOAD_CCACHE`). |
| `offload_fill.py` | Alternative scan, fill and resolve variants of the row cache (same cache state). |
| `offload_cuda.py`, `offload_cuda2.py` | Loaders for the CUDA offload decode attention: split-K over FP8 rows (`offload_attn.so`), and version 2 with warp-specialized tiles and FP16 or FP8 P·V (`offload_attn2.so`). |
| `link_gate.py` | Pauses background row copies while a host all-reduce runs (`ARC_OFFLOAD_AR_GATE`). |
| `host_ar.py` | Two-shot all-reduce through shared pinned host memory for GPUs without peer-to-peer (int8/int6 payloads, CUDA-graph safe), and host all-gathers. |
| `ar_lowbit.py` | Sub-int6 all-reduce payloads with error feedback and Hadamard rotation (`ARC_AR_LB`). |
| `ag_pack.py` | 16-bit ids in the sharded indexer's all-gather (`ARC_AG_U16`). |
| `shard_indexer.py` | Tensor-parallel sharded sparse-attention indexer keys. |
| `shard_logits_v2.py`, `shard_logits_pk.py` | Variants of the sharded indexer's decode logits for the L4 (v2 is pinned on in `GATE_PINS`; the request-packed one is off). |
| `fast_topk.py` | Exact radix top-k for the sharded indexer (`topk_select.so`). |
| `index_share.py` | Top-k sharing between sparse-attention layers (`ARC_INDEX_SHARE`, on in both presets: the layers marked `S` reuse the previous layer's top-k) and overlap statistics. |
| `lowbit_moe.py` | Mixed 2/3/4-bit MoE for the routed experts: one Triton GEMM launch per projection for all experts of a layer. |
| `lowbit2.py`, `lowbit_fuse.py` | Expert-stationary low-bit GEMM for the pair layout, and fused small kernels around the MoE GEMMs. |
| `w8a8_v2.py`, `w8a8_ws.py` | Block-FP8 W8A8 GEMMs with L4-tuned tiles per shape, and a decode-sized variant. |
| `lmhead_fp8.py`, `mla_fp8.py` | Block-FP8 LM head; FP8 weights for MLA's absorbed per-head matmuls. |
| `gmerge.py`, `qcat.py` | The attention-gate projection merged into the fused q/kv-a GEMM; the MLA decode query built in place (both bit-identical). |
| `swa_fused.py` | One-pass FP8 sliding-window decode attention over the paged KV cache (`swa_attn.so`). |
| `shard_sample.py` | Speculative-decoding verification on the vocab-parallel logits shard, without the all-gather. |
| `draft_head.py` | Greedy MTP drafts from a frequent-token subset of the LM head (off in both presets). |
| `session_ep.py` | HTTP routes for streaming-input sessions: one attempt keeps its KV across continuation chunks (`ARC_SESSION_EP`). |
| `delta_overlay.py` | Load-time overlay of the fine-tuned delta (`p1s_delta.safetensors`) on the checkpoint. |
| `kvb_rebuild.py`, `mem_reclaim.py` | Exact GPU-memory reclaims: `kv_b_proj` rebuilt on use; lazy penalty buffers, MLA workspace cap, compact parameters. |
| `stage_wait.py` | Lets the ranks wait for each checkpoint file while the staging copier writes it. |
| `steplog.py` | Read-only per-step speculative-decoding log (off). |
| `fake_tp.py` | Single-GPU emulation of one tensor-parallel rank, for local debugging. |

### `runtime_dataset/`

| File(s) | Role |
|---|---|
| `run_l4.py` | The launcher. It runs the staging copier (environment, CUDA and Triton tars first), unpacks the environment and re-executes itself under it. It installs the vLLM patch, the plugin and the CUDA toolkit, and restores the Triton cache. It runs the startup gates, then `vllm serve` under the startup guard, then the solver; at the hard stop it repairs the submission. It reads `ARC_MODE_FILE`, `ARC_CKPT_DIR`, `ARC_INPUT` and `ARC_WORK`. |
| `solve.py` | The solver. Two attempts per test input, 64 requests in flight, token caps that shrink as the deadline nears, an endgame that forces conclusions, and continuation chunks in server-side sessions. It writes `submission.json` atomically. |
| `gridfmt.py`, `loop_detect.py`, `conf_gate.py` | Solver helpers: grid formats for prompts and answers, repetition-loop detection (on), and confidence-gated termination (off unless `ARC_CONF_KILL` is set). |
| `system_prompt.txt` | The prompt template. |
| `mode_submit.json`, `mode.json` | The release mode (`best_throughput`); the two files are identical. |
| `mode_submit_v6.json`, `mode_submit_v7.json`, `mode_submit_v8.json`, `mode_tput.json`, `mode_tput5.json`, `mode_real_holdout.json`, `sweep_random_lb144.json` | Older modes kept in the dataset; not used. |
| `w8a8_configs.json`, `w8a8_ws.json`, `lowbit_tune_l4.json`, `lowbit_tune_pairs_sm89emu.json`, `lowbit2_comb_m256.json`, `lowbit_routes_mtp5.npz` | Kernel tile tables measured on L4, and recorded routing for the MoE tile tuner. |
| `bench_*.py`, `tune_*.py`, `test_ag_u16.py`, `test_gmerge.py`, `nccl_pick.py`, `l4_microbench.py` | Startup gates, tuners and microbenchmarks the launcher runs when a mode asks for them. Both presets pin the gates' choices (`GATE_PINS`) and measure only the host all-reduce gate (`bench_host_ar.py`). |
| `replay.py`, `load_sweep.py`, `logprob_dump.py`, `link_hog.py`, `ncu_probe.py`, `bench_dram.py`, `ctx_lengths.json`, `req_lengths.json`, `draft_vocab.json` | Diagnostics and their data: request replay, load sweeps, log-prob dumps, PCIe link emulation, a profiler probe, and the token list of the unused draft head; not used by the submission. |
| `holdout40.txt`, `smoke5.txt` | 40 and 5 public-evaluation task ids for local checks. |
| `serve_l4.sh.txt` | An early stand-alone `vllm serve` script for 4×L4, kept for reference; the launcher builds its own command line from the mode. |

All of these files are dataset version 5's bytes, so some of their comments still name internal version labels or old repository paths. For example, `dots3/build_offload_attn.sh` is `../engine/build_offload_attn.sh` here.
