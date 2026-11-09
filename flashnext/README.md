# Flash-Next: Qwen3.8-Flash-Next on 4×L4

Our final solution. A program-writing agent runs on Qwen3.8-Flash-Next, a 125B mixture-of-experts model, inside Kaggle's 12-hour, offline, four-L4 budget:
- **Compression:** the model is pruned from 512 to 320 experts per layer and quantized to 3/4 bits so it fits.
- **RL repair:** RL in the agent loop repairs the compression loss. It trains only the quantization scales and the router rows.
- **Draft head:** a fine-tuned MTP layer and an int8 head over the 32,768 most-used tokens speed up decoding.
- **Serving:** a vLLM 0.32 plugin rebuilt for four PCIe-only L4s generates about 40M tokens per run.
- **Answers:** verified programs vote for the answers.

| | RC8 (default here) | RC9 (final bundle) |
|---|---|---|
| Kaggle notebook `flash-next-arc-eval` | version 5 | version 7 |
| Code dataset `flash-next-arc-code` | version 11 | version 13 |
| Submission | 57879466 | 57900065 |
| Public leaderboard | **75.69** (15th of 3,626 teams at the close) | 73.89 |
| Kaggle 2-hour save run (240 entries) | 122 at 1,087 tokens/s | 130.5 at 1,111 tokens/s |

- Both use the same Kaggle Models: `qwen38-flash-next-fp8-core`, `qwen38-flash-next-experts` (base build, RL delta, draft head) and `flash-next-offline-runtime`.
- RC9 is RC8 plus three changes: int8 reduce-scatter on the tensor-parallel group too (`FN_TP_Q8_RS=1`), a guard in the solver's endgame estimate, and a larger compile cache. The two scores are within the leaderboard's noise of each other.
- A full-length 12-hour replica of RC8 on our own 4×L4 test machines scored 86.6% pass@2 (207.83 of 240 entries: the 120 public evaluation tasks, each entered twice), at 1,005 tokens/s.

## Folders

| Path | What it holds |
|---|---|
| [`kaggle/`](kaggle/README.md) | The submission: the RC8 and RC9 notebooks and the exact code dataset they ran, with the engine plugin `fnext`, the agent, `ak` and the launcher. RC8's files are the default; [`kaggle/code_dataset/rc9/`](kaggle/code_dataset/rc9/) holds RC9's two changed files. Also the tools that build every Kaggle artifact: core split, model parts, offline runtime, code dataset, compile cache, notebook and validation check. |
| [`training/`](training/README.md) | How the model parts were made: self-generated traces, routing capture, keep set, GPTQ, RL in the agent loop, the RL delta, the draft head, and the evaluation protocol. |
| [`MODEL_LICENSE`](MODEL_LICENSE) | The Qwen Community License 1.0, which covers every model part. |

## Reproducing

1. **Re-run a submission on Kaggle.** Open `flash-next-arc-eval`, pick version 5 (RC8) or 7 (RC9) and choose *Copy & Edit*: Kaggle keeps the inputs that version ran with. Submit it, or run it as a save run, which scores itself on the public evaluation tasks. See [`kaggle/README.md`](kaggle/README.md) §2.
2. **Run the same notebook on your own 4×L4 machine.** Download the dataset and the Model variations into a `/kaggle/input`-like tree, point the notebook at it with `NB_INPUT_DIR`, `NB_WORK_DIR` and `NB_TMP_DIR`, and run it with `jupyter nbconvert --execute`. See [`kaggle/README.md`](kaggle/README.md) §3.
3. **Rebuild everything.** Start from `Qwen/Qwen3.8-Flash-Next-FP8`. Follow [`training/README.md`](training/README.md) for the expert build, the RL rounds and the draft head, then [`kaggle/README.md`](kaggle/README.md) §5 for the Kaggle parts, the code dataset (`build/pack_code_dataset.py`, `--release rc8|rc9`) and the notebook (`build/build_notebook.py build/notebook_config_rc8.json` or `_rc9.json`).
   - From an unmodified checkout, both notebooks and both code datasets come out byte-identical to what was pushed to Kaggle.
   - Our runs used H200 GPUs: RL took about 2.5 days on up to 24 GPUs (at most about 1,500 GPU-hours), and the draft head 1.5 hours on one GPU.

Only ARC-AGI-2 *training* tasks, ARC-AGI-1 and RE-ARC-derived tasks were used for calibration and training. The tools assert that no evaluation task id is used.

## Licences

- Code: Apache-2.0 ([`../LICENSE`](../LICENSE)).
- Model parts: they derive from Qwen3.8-Flash-Next and are under the Qwen Community License 1.0 ([`MODEL_LICENSE`](MODEL_LICENSE)).
- Third-party packages in the offline runtime keep their own licences.
