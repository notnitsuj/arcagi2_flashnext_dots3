"""ARC_LMHEAD_FP8=1: the (untied, vocab-parallel) LM head in block FP8 instead of bf16.

The head is read once per verify step and once per MTP draft step (4x per step, ~389 MB bf16 per rank at TP4): pure
weight bandwidth at decode batch sizes. After weight load the bf16 weight is quantized to e4m3 with one scale per
128x128 block (the checkpoint's own scheme for its dense linears) and the bf16 copy is freed (-195 MB/GPU); logits
then come from vLLM's Triton W8A8 block GEMM (activations quantized per 1x128 group). The MTP layer shares this head.
"""
import os

import torch

BLK = 128


def quantize(w):
    """bf16 [N, K] -> (e4m3 [N, K], fp32 scales [N/128, K/128]); N, K multiples of 128."""
    N, K = w.shape
    wb = w.float().view(N // BLK, BLK, K // BLK, BLK)
    s = wb.abs().amax(dim=(1, 3)).clamp(min=1e-8) / 448.0
    q = (wb / s[:, None, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn).view(N, K)
    return q, s.contiguous()


def _convert(model):
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
    n = 0
    for name, m in model.named_modules():
        if isinstance(m, ParallelLMHead) and not hasattr(m, "_arc_fp8") and m.weight.dtype == torch.bfloat16 \
                and m.weight.is_cuda and m.weight.shape[0] % BLK == 0 and m.weight.shape[1] % BLK == 0:
            q, s = quantize(m.weight.data)
            m._arc_fp8 = (q, s)
            m.weight.data = torch.empty(0, m.weight.shape[1], dtype=m.weight.dtype, device=m.weight.device)
            n += 1
            print(f"[lmhead_fp8] {name}: {tuple(q.shape)} -> e4m3 + {tuple(s.shape)} block scales", flush=True)
    return n


def install():
    from vllm.model_executor.layers import vocab_parallel_embedding as vpe
    from vllm.model_executor.layers.quantization.utils import fp8_utils as fu
    from vllm.v1.worker.gpu import model_runner as mr

    orig_apply = vpe.UnquantizedEmbeddingMethod.apply

    def apply(self, layer, x, bias=None):
        f = getattr(layer, "_arc_fp8", None)
        if f is None:
            return orig_apply(self, layer, x, bias)
        q, s = f
        xq, xs = fu.per_token_group_quant_fp8(x.reshape(-1, x.shape[-1]), BLK)
        y = fu.w8a8_triton_block_scaled_mm(xq, q, xs, s, [BLK, BLK], output_dtype=x.dtype)
        y = y.view(*x.shape[:-1], q.shape[0])
        return y if bias is None else y + bias
    vpe.UnquantizedEmbeddingMethod.apply = apply

    orig_load = mr.GPUModelRunner.load_model

    def load_model(self, *a, **kw):
        r = orig_load(self, *a, **kw)
        n = _convert(self.model)
        for attr in ("drafter", "speculator"):   # the MTP drafter holds its own reference to the model's head
            d = getattr(self, attr, None)
            if d is not None and getattr(d, "model", None) is not None:
                n += _convert(d.model)
        torch.cuda.empty_cache()
        print(f"[lmhead_fp8] {n} head(s) converted", flush=True)
        return r
    mr.GPUModelRunner.load_model = load_model
