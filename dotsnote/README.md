# Dots3-Note on 4×L4

The model we ran from Oct 6 to 26, before Qwen3.8-Flash-Next: the dots3-note preview by Dots Studio, a 280B mixture-of-experts model (16B active) with sparse plus sliding-window attention. It answers ARC tasks directly, with two attempts per test input.

- **Fit:** 144 of 256 routed experts kept per layer, ranked on traces of ARC-AGI-2 training tasks and quantized with GPTQ to 3 and 2 bits (54.6 GB), tensor-parallel across the four GPUs. A small fine-tuned delta (`p1s`: GPTQ scales, norms and router) is overlaid at load.
- **Memory:** the 82.7 GiB sparse-attention KV cache lives in pinned host memory and streams back every step, through split-K attention over host rows, a GPU cache for hot rows and compact 400-byte rows.
- **Speed:** host-staged all-reduce in int8/int6, 1.7× faster than NCCL on Kaggle's PCIe-only L4s; FP8 and low-bit kernels tuned for L4; MTP-3 speculative decoding; and a server boot cut from 1,648 s to 413 s.
- **Solver:** 64 requests in flight, two attempts per test input, token caps that shrink as the deadline nears, an endgame that forces open attempts to conclude, and multi-turn server sessions.

## Two configurations of one notebook

| | `best_score` | `best_throughput` |
|---|---|---|
| What it is | v22c, the configuration of our best Dots3-Note score | v29, our fastest Dots3-Note build |
| Public leaderboard | **63.47** (submitted from an earlier notebook that is not published) | 55.28 (notebook version 4) |
| Kaggle notebook | built from [`kaggle/notebook/best_score/`](kaggle/notebook/best_score/) | `notnitsuj/dots3-note-arc-agi-2`, version 4 |
| Runtime dataset | `notnitsuj/dots3-note-arc-runtime`, version 5 | the same |
| Serving numerics | int6 all-reduce everywhere | a lower-bit (4/5-bit) all-reduce on the main path; everything else identical |
| Speed | the v22c build | faster: memory reclaims spent on larger row caches, a compact GPU row cache and the lower-bit all-reduce (the last step, v27 to v29, added 5–8% tokens/s in paired runs), plus a later solver deadline |

Both use the same Kaggle Model, `notnitsuj/dots3-note-lb144r` (variations `lb144r-p1` to `lb144r-p4`), and the same code. They differ only in the flags of the mode file the launcher reads ([`kaggle/README.md`](kaggle/README.md) §2 lists every flag). A single leaderboard score moves by about ±3 points from sampling alone: an identical rerun of notebook version 2 scored 58.89 and then 53.19.

## Folders

| Path | What it holds |
|---|---|
| [`kaggle/`](kaggle/README.md) | The two notebook presets, the runtime dataset's text files (launcher `run_l4.py`, solver `solve.py`, mode files, L4 tables and start-up gates) with the md5 of every published file, and the tools that rebuild every Kaggle artifact: notebook builder, dataset packer, bundle and environment builders, Model splitter and a mocked notebook test. |
| [`engine/`](engine/) | The engine source: the vLLM plugin package `arcstats` (host-memory KV rows, host collectives, low-bit MoE and FP8 kernels, sessions, the delta overlay), its CUDA kernels with build scripts, and our patch to vLLM 0.30.0. |
| [`training/`](training/README.md) | How the checkpoint and the delta were made: calibration traces, expert ranking and pruning, GPTQ, the fine-tune, and the evaluation protocol. |

## Reproducing

1. **Re-run on Kaggle.** For `best_throughput`, open `dots3-note-arc-agi-2`, pick version 4 and choose *Copy & Edit*: Kaggle keeps its inputs attached. For `best_score`, create a notebook from [`kaggle/notebook/best_score/`](kaggle/notebook/best_score/) with the same inputs (`kaggle kernels push -p` after changing the id to your own). See [`kaggle/README.md`](kaggle/README.md) §3.
2. **Run the same notebook on your own 4×L4 machine** from a `/kaggle/input`-like tree. See [`kaggle/README.md`](kaggle/README.md) §4.
3. **Rebuild everything.** Start from the dots3-note preview checkpoint (`dots-studio/dots3-note-prev-fp8`). Follow [`training/README.md`](training/README.md) for the pruned low-bit checkpoint and the delta, then [`kaggle/README.md`](kaggle/README.md) §5 for the kernels, the environment, the runtime dataset, the Model parts and the notebooks.
   - The packer reproduces every file of runtime dataset version 5 (byte for byte for the text files), and the notebook builder reproduces version 4 byte for byte.

The `best_score` notebook has not been run on Kaggle in this form. The evidence that it serves exactly like v22c: with the lower-bit all-reduce off, the later runtime returned the same tokens and log-probabilities as v22c on a 64-request test, and each memory and cache flag that `best_score` turns off is bit-exact by its own tests.

Only ARC-AGI-2 *training* tasks were used for calibration and fine-tuning.

## Licences

- Code: Apache-2.0 ([`../LICENSE`](../LICENSE)). The vLLM patch and the patched vLLM files are under vLLM's Apache-2.0 licence.
- Model parts: they derive from the dots3-note preview weights and are under the Apache License 2.0, like the original (the LICENSE is in every part).
- Third-party packages in the offline environment keep their own licences.
