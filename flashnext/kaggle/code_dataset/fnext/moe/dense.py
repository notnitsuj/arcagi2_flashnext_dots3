import os
import re
_ST = {'done': False}

def _log(msg):
    print(f'[fndense] {msg}', flush=True)

def targets_for(ckpt_quant_method):
    fp8_ckpt = (ckpt_quant_method or '').lower() in ('fp8', 'fbgemm_fp8', 'modelopt', 'compressed-tensors')
    t = {'re:.*linear_attn\\.(in_proj_qkvz|in_proj_qkv|in_proj_z|out_proj)$': 'fp8_per_channel', 're:.*mlp\\.shared_expert\\.(gate_up_proj|down_proj)$': 'fp8_per_channel'}
    if not fp8_ckpt:
        t['re:.*self_attn\\.o_proj$'] = 'fp8_per_channel'
        if os.environ.get('FN_DENSE_FP8_QKV') == '1':
            t['re:.*self_attn\\.(qkv_proj|q_proj|k_proj|v_proj|indexer\\.index_qk_proj)$'] = 'fp8_per_channel'
    if os.environ.get('FN_DENSE_FP8_HC') == '1':
        t['re:.*hyper_connection\\.input_mix_weight_up$'] = 'fp8_per_channel'
    return t
FP8MM_TUNE = {(2560, 8192): {32: (16, 32, 64), 64: (16, 32, 64)}, (3072, 2560): {32: (16, 32, 128), 64: (64, 64, 128)}, (2560, 1280): {32: (32, 128, 256), 64: (16, 32, 64)}, (640, 2560): {32: (32, 32, 128), 64: (16, 32, 128)}, (2560, 124160): {32: (16, 32, 256), 64: (32, 32, 64)}}

def fp8mm_cfg(M, K, N):
    t = FP8MM_TUNE.get((K, N))
    if not t:
        return None
    for m in sorted(t):
        if M <= m:
            return t[m]
    return None

def fp8_scaled_mm(A, B, As, Bs, out_dtype, bias=None):
    from vllm import _custom_ops as ops
    cfg = fp8mm_cfg(A.shape[0], A.shape[1], B.shape[1]) if _FP8MM and bias is None else None
    if cfg is None:
        return ops.cutlass_scaled_mm(A, B, As, Bs, out_dtype, bias=bias)
    from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import triton_scaled_mm
    bs = Bs.reshape(-1, 1) if Bs.numel() > 1 else Bs
    return triton_scaled_mm(A, B, As, bs, out_dtype, block_size_m=cfg[0], block_size_n=cfg[1], block_size_k=cfg[2])
_FP8MM = os.environ.get('FN_FP8MM_TRITON') == '1'

def _install_fp8mm():
    from vllm.model_executor.kernels.linear.scaled_mm import cutlass as cm
    orig = cm.CutlassFP8ScaledMMLinearKernel.apply_scaled_mm

    def apply_scaled_mm(self, *, A, B, out_dtype, As, Bs, bias, output_shape):
        out_n = self.logical_output_size
        if bias is None and A.shape[1] == B.shape[0] and (out_n == B.shape[1]) and fp8mm_cfg(A.shape[0], A.shape[1], B.shape[1]):
            return fp8_scaled_mm(A, B, As, Bs, out_dtype).view(*output_shape)
        return orig(self, A=A, B=B, out_dtype=out_dtype, As=As, Bs=Bs, bias=bias, output_shape=output_shape)
    cm.CutlassFP8ScaledMMLinearKernel.apply_scaled_mm = apply_scaled_mm
    _log('FN_FP8MM_TRITON: small-M FP8 GEMMs via triton_scaled_mm (L4-tuned shapes)')

def register():
    if os.environ.get('FN_DENSE_FP8') != '1' or _ST['done']:
        return
    _ST['done'] = True
    _install_online_linear()
    if _FP8MM:
        _install_fp8mm()
    if os.environ.get('FN_DENSE_FP8_LMHEAD', '1') == '1':
        _install_lmhead()

def _install_online_linear():
    from vllm.model_executor.model_loader import weight_utils as wu
    orig = wu.get_quant_config

    def get_quant_config(model_config, load_config):
        cfg = orig(model_config, load_config)
        if getattr(cfg, 'online_quantization_config', None) is not None:
            _log('checkpoint already composed with an online quantization config; FN_DENSE_FP8 linears not added')
            return cfg
        from vllm.config.quantization import QuantizationConfigArgs
        from vllm.model_executor.layers.quantization.online.base import OnlineQuantizationConfig
        qm = getattr(model_config, 'quantization', None)
        t = targets_for(qm)
        cfg.online_quantization_config = OnlineQuantizationConfig(QuantizationConfigArgs(targets=t))
        _log(f'checkpoint quant={qm}: online fp8_per_channel targets {list(t)}')
        return cfg
    wu.get_quant_config = get_quant_config

def _install_lmhead():
    import torch
    from vllm.model_executor.layers import vocab_parallel_embedding as vpe
    skip = os.environ.get('FN_DENSE_FP8_LMHEAD_SKIP')

    class Fp8LMHeadMethod(vpe.UnquantizedEmbeddingMethod):

        def process_weights_after_loading(self, layer):
            w = layer.weight.data
            if w.dtype == torch.float8_e4m3fn:
                return
            s = (w.float().abs().amax(1, keepdim=True) / 448.0).clamp_min(1e-12)
            q = (w.float() / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            layer.weight = torch.nn.Parameter(q, requires_grad=False)
            layer.register_buffer('fn_fp8_scale', s.view(1, -1).contiguous(), persistent=False)
            torch.cuda.empty_cache()

        def apply(self, layer, x, bias=None):
            from vllm import _custom_ops as ops
            shp = x.shape
            x2 = x.reshape(-1, shp[-1])
            if x2.shape[0] == 0:
                return x.new_empty(*shp[:-1], layer.weight.shape[0])
            xq, xs = ops.scaled_fp8_quant(x2, use_per_token_if_dynamic=True)
            out = fp8_scaled_mm(xq, layer.weight.t(), xs, layer.fn_fp8_scale, x.dtype, bias=bias)
            return out.reshape(*shp[:-1], out.shape[-1])

        def embedding(self, layer, input_):
            raise RuntimeError('FN_DENSE_FP8 LM head is not an embedding (tie_word_embeddings must be False)')
    orig_init = vpe.ParallelLMHead.__init__

    def init(self, *a, **kw):
        orig_init(self, *a, **kw)
        prefix = kw.get('prefix', '')
        if type(self.quant_method) is not vpe.UnquantizedEmbeddingMethod:
            return
        if skip and re.search(skip, prefix):
            return
        if getattr(self, 'bias', None) is not None:
            return
        self.quant_method = Fp8LMHeadMethod()
        _log(f'LM head {prefix or '(no prefix)'}: FP8 per-row W8A8')
    vpe.ParallelLMHead.__init__ = init
