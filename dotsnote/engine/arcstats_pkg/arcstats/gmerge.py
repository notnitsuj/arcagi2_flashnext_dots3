"""ARC_GATE_MERGE=1: the headwise attention-gate projection (g_proj, hidden -> 64, block FP8) computed by the fused
q/kv-a projection's GEMM instead of its own (bit-identical).

Every Dots3 attention layer runs two W8A8 block-FP8 GEMMs on the same normed hidden state: fused_qkv_a_proj
(N = 2112 SWA / 1600 DSA) and g_proj (N = 64), each with its own per-token-group activation quantization. On L4 at 64
sequences x MTP-4 the g_proj pair costs ~13-17 us (quant) + ~20 us (GEMM) per layer, ~1.6 ms per decode step over 46
layers + the MTP layer (L4 profile). After weight loading the g_proj rows are appended to the fused weight,
after zero rows that pad the fused part to a whole 128-row scale block (2112 -> 2176 + 64 = 2240, 1600 -> 1664 + 64 =
1728), so both keep their own block scales; the merged GEMM uses the fused shape's tile config (`w8a8` config lookup
aliased), and each output column sees the same K loop as before, so q/kv-a and gate outputs are unchanged bit for bit.
The gate is then sliced from the merged output in `_forward_note_mla`.
"""
import torch

BLK = 128
_ALIAS = {}   # merged (N, K) -> original fused (N, K) for the tile-config lookup


def _merge(att, gp):
    qp = att.fused_qkv_a_proj
    wq, wg = qp.weight, gp.weight
    sq, sg = getattr(qp, "weight_scale_inv", None), getattr(gp, "weight_scale_inv", None)
    if (sq is None or sg is None or wq.dtype != torch.float8_e4m3fn or wg.dtype != wq.dtype or wq.dim() != 2
            or wq.shape[1] != wg.shape[1] or type(qp.quant_method) is not type(gp.quant_method)
            or sq.shape[0] != -(-wq.shape[0] // BLK) or sg.shape[0] != -(-wg.shape[0] // BLK)
            or sq.shape[1] != sg.shape[1] or wg.shape[0] > BLK):
        return None
    nq, ng, K = wq.shape[0], wg.shape[0], wq.shape[1]
    goff = -(-nq // BLK) * BLK
    w = torch.zeros((goff + ng, K), dtype=wq.dtype, device=wq.device)
    w[:nq].copy_(wq)
    w[goff:].copy_(wg)
    s = torch.cat([sq, sg]).contiguous()
    _ALIAS[(goff + ng, K)] = (nq, K)
    qp.weight.data = w
    qp.weight_scale_inv.data = s
    gp.weight.data = torch.empty((0, K), dtype=wg.dtype, device=wg.device)   # free; g_proj is not called any more
    att._arc_gm = (nq, goff, ng)
    return (nq, ng, goff + ng, K)


def convert(model):
    shapes = {}
    for name, m in model.named_modules():
        gp = getattr(m, "g_proj", None)
        if gp is None or getattr(m, "attention_gate_type", None) != "headwise":
            continue
        att = m.mla_attn if hasattr(getattr(m, "mla_attn", None), "fused_qkv_a_proj") else m
        if getattr(att, "_arc_gm", None) is not None or not hasattr(att, "fused_qkv_a_proj"):
            continue
        r = _merge(att, gp)
        if r is None:
            print(f"[gmerge] {name}: layout not supported, left as is", flush=True)
            continue
        m._arc_gm_att = att
        shapes[r] = shapes.get(r, 0) + 1
    return shapes


def install():
    from vllm.model_executor.layers.quantization.utils import fp8_utils as fu
    from vllm.models.dots3_note.nvidia import model as dm
    from vllm.v1.worker.gpu import model_runner as mr

    orig_get = fu.get_w8a8_block_fp8_configs

    def get(N, K, block_n, block_k):
        return orig_get(*_ALIAS.get((N, K), (N, K)), block_n, block_k)
    fu.get_w8a8_block_fp8_configs = get

    orig_fwd = dm._forward_note_mla

    def forward_note_mla(attention, positions, hidden_states, *, g_proj, k_rope_only_layernorm, attention_gate_type,
                         q_lora_scale, kv_lora_scale, llama_4_scaling=None):
        gm = getattr(attention, "_arc_gm", None)
        if gm is None:
            return orig_fwd(attention, positions, hidden_states, g_proj=g_proj,
                            k_rope_only_layernorm=k_rope_only_layernorm, attention_gate_type=attention_gate_type,
                            q_lora_scale=q_lora_scale, kv_lora_scale=kv_lora_scale, llama_4_scaling=llama_4_scaling)
        nq, goff, ng = gm
        merged = attention.fused_qkv_a_proj(hidden_states)[0]
        qkv_lora = merged[..., :nq]
        q_c, kv_lora = qkv_lora.split([attention.q_lora_rank, attention.kv_lora_rank + attention.qk_rope_head_dim],
                                      dim=-1)
        q_c = attention.q_a_layernorm(q_c) * q_lora_scale
        kv_c, k_pe = kv_lora.split([attention.kv_lora_rank, attention.qk_rope_head_dim], dim=-1)
        kv_c_normed = attention.kv_a_layernorm(kv_c) * kv_lora_scale
        k_pe = k_rope_only_layernorm(k_pe).unsqueeze(1)

        q = attention.q_b_proj(q_c)[0]
        heads = attention.num_heads
        if attention.dcp_q_replicate:
            heads *= attention.q_b_proj.group_size
        q = q.view(-1, heads, attention.qk_head_dim)
        q[..., attention.qk_nope_head_dim:], k_pe = attention.rotary_emb(
            positions, q[..., attention.qk_nope_head_dim:], k_pe)

        if attention.indexer and attention.is_sparse and not attention.skip_topk:
            attention.indexer(hidden_states, q_c, positions, attention.indexer_rope_emb)
        if llama_4_scaling is not None:
            q *= llama_4_scaling

        q_dcp_replicated = None
        if attention.dcp_q_replicate:
            q_dcp_replicated, q = q, attention.q_b_proj._local_view(q)
        attn_out = attention.mla_attn(
            q, kv_c_normed, k_pe,
            output_shape=(hidden_states.shape[0], attention.num_heads * attention.v_head_dim),
            q_dcp_replicated=q_dcp_replicated)

        gate = merged[..., goff:goff + ng]
        if gate.shape[-1] != attention.num_heads:
            rank = dm.get_tensor_model_parallel_rank()
            gate = gate.narrow(-1, rank * attention.num_heads, attention.num_heads)
        attn_out = attn_out.view(-1, attention.num_heads, attention.v_head_dim)
        gate = torch.sigmoid(gate.float()).to(attn_out.dtype)
        attn_out = (attn_out * gate.unsqueeze(-1)).flatten(-2)
        return attention.o_proj(attn_out)[0]
    dm._forward_note_mla = forward_note_mla

    orig_load = mr.GPUModelRunner.load_model

    def load_model(self, *a, **kw):
        r = orig_load(self, *a, **kw)
        shapes = convert(self.model)
        for attr in ("drafter", "speculator"):
            d = getattr(self, attr, None)
            if d is not None and getattr(d, "model", None) is not None:
                for k, v in convert(d.model).items():
                    shapes[k] = shapes.get(k, 0) + v
        torch.cuda.empty_cache()
        print(f"[gmerge] g_proj merged into fused_qkv_a_proj: {shapes} (nq, ng, N, K): layers", flush=True)
        return r
    mr.GPUModelRunner.load_model = load_model
    print("[gmerge] installed", flush=True)
