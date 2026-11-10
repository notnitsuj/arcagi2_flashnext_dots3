# Scaling Qwen 3.8 Flash-Next and Dots-3 Note to >1,100 tok/s on 4×L4: an ARC Prize 2026 (ARC-AGI-2) solution

Code, notebooks and build recipes for our ARC Prize 2026 solution. We ran two large mixture-of-experts models inside Kaggle's 12-hour, offline, four-L4 budget: first Dots3-Note, then Qwen3.8-Flash-Next, our final solution. Each has its own folder with everything needed to re-run its Kaggle submissions and rebuild every artifact.

| Folder | Model | Public leaderboard | Contents |
|---|---|---|---|
| [`flashnext/`](flashnext/README.md) | Qwen3.8-Flash-Next (125B MoE): a program-writing agent, Oct 27 to Nov 2 | **75.69** (RC8), 15th of 3,626 teams at the close; the final bundle RC9 scored 73.89 | RC8 and RC9 notebooks, code dataset, engine plugin, agent, build tools, training recipe (pruning, GPTQ, RL, draft head; also the distillation we tried and did not use) |
| [`dotsnote/`](dotsnote/README.md) | Dots3-Note (280B MoE, 16B active): direct answers, Oct 6 to 26 | **63.47** | notebook presets for our best-scoring and our fastest build, runtime dataset, engine plugin, build tools, compression and fine-tuning recipe (also the distillation we tried and did not use) |
| [`docs/`](docs/) | | | the writeup as published on Kaggle ([`docs/writeup.md`](docs/writeup.md)), the appendix with recipes, hyperparameters and negative results ([`docs/appendix.pdf`](docs/appendix.pdf), source [`docs/appendix.tex`](docs/appendix.tex)), and the figures |

Read the writeup first.

**Added after the deadline (2026-11-10):** the distillation trainers of both models, which neither submission uses. See step 10 of the training READMEs: [Flash-Next](flashnext/training/README.md#10-distillation-toward-the-fp8-model-not-used) and [Dots3-Note](dotsnote/training/README.md#10-distillation-toward-the-fp8-preview-not-used). The repository as of the open-source deadline is commit [`67ebe69`](https://github.com/notnitsuj/arcagi2_flashnext_dots3/tree/67ebe69).

## Flash-Next in brief

- **Compression:** the model is pruned from 512 to 320 experts per layer and quantized to 3/4 bits so it fits.
- **RL repair:** RL in the agent loop repairs the compression loss. It trains only the quantization scales and router rows.
- **Serving:** a vLLM plugin rebuilt for four PCIe-only L4s generates about 40M tokens per run, at 1,087–1,120 generated tokens/s on Kaggle.
- **Answers:** verified programs vote for the answers. A 12-hour local replica of RC8 scored 86.6% pass@2 on the 120 public evaluation tasks.

## Dots3-Note in brief

- **Fit:** 144 of 256 experts per layer kept, at 3 and 2 bits (54.6 GB), tensor-parallel across the four GPUs. The 82.7 GiB sparse-attention cache lives in pinned host memory and streams back every step.
- **Speed:** host-staged all-reduce in int8/int6 (1.7× faster than NCCL on Kaggle), split-K attention over host rows with a GPU cache for hot rows, and a server boot cut from 1,648 s to 413 s.
- **Solver:** two attempts per test input with deadline-aware token caps and multi-turn server sessions.
- **Two presets** of one notebook: the configuration of our best Dots3-Note score (63.47) and our fastest build (notebook version 4).

## Kaggle artifacts

| Model | Artifact | Kaggle reference |
|---|---|---|
| Flash-Next | Notebook | `notnitsuj/flash-next-arc-eval`: version 5 (RC8, submission 57879466) and version 7 (RC9, submission 57900065) |
| | Code dataset | `notnitsuj/flash-next-arc-code`: version 11 (RC8) and version 13 (RC9) |
| | Models | `notnitsuj/qwen38-flash-next-fp8-core` (non-expert FP8 weights and the n-gram table), `notnitsuj/qwen38-flash-next-experts` (pruned 3/4-bit experts, RL deltas, draft heads), `notnitsuj/flash-next-offline-runtime` (Python 3.12 with vLLM 0.32.0) |
| Dots3-Note | Notebook | `notnitsuj/dots3-note-arc-agi-2`: version 2 (public 58.89) and version 4 (the fastest build, public 55.28) |
| | Runtime dataset | `notnitsuj/dots3-note-arc-runtime`: version 3 (notebook version 2) and version 5 (notebook version 4) |
| | Model | `notnitsuj/dots3-note-lb144r`: variations `lb144r-p1` to `lb144r-p4` |

## Reproducing

Each folder's README has three levels:
1. **Re-run a submission on Kaggle:** *Copy & Edit* the notebook version; Kaggle keeps its inputs attached.
2. **Run the same notebook on your own 4×L4 machine** from a `/kaggle/input`-like tree.
3. **Rebuild the artifacts** from the public base checkpoints: `Qwen/Qwen3.8-Flash-Next-FP8` and the dots3-note preview (`dots-studio/dots3-note-prev-fp8`).

Only ARC-AGI-2 *training* tasks (plus ARC-AGI-1 and RE-ARC-derived tasks for Flash-Next's later RL rounds) were used for calibration and training. The tools assert that no evaluation task id is used.

## Licences

- Our code: Apache-2.0 ([`LICENSE`](LICENSE)).
- Flash-Next model parts: they derive from Qwen3.8-Flash-Next and are under the Qwen Community License 1.0 ([`flashnext/MODEL_LICENSE`](flashnext/MODEL_LICENSE)).
- Dots3-Note model parts: they derive from the dots3-note preview weights and are under the Apache License 2.0, like the original (the LICENSE is in every part).
- Third-party packages in the offline runtimes keep their own licences.

## Acknowledgements

- The Qwen team, for Qwen3.8-Flash-Next, and Dots Studio, for Dots3-Note.
- ARC Prize and Kaggle, for the competition.
- vLLM, FlashInfer, Triton and flash-linear-attention, on which the engines are built.
