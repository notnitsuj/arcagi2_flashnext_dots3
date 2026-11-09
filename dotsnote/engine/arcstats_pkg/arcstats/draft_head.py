"""ARC_DRAFT_VOCAB=<file>: greedy MTP draft tokens from a frequent-token subset of the (shared) LM head.

With use_local_argmax_reduction the drafter asks the MTP model for get_top_tokens(): every TP rank multiplies the
hidden states by its whole vocab shard (152k/TP rows of 5120, ~1.6 ms per draft step on an L4) and only (max, index)
pairs are gathered. Real generations use a few thousand distinct tokens (top 4096 cover 99.85%), so here each rank
multiplies only by its rows of the subset (a contiguous bf16 copy, ~5 MB per rank for 8k tokens). Drafts can then
only propose subset tokens; verification by the target model is unchanged, so outputs are exact and only the
acceptance of the rare out-of-subset tokens is lost.

<file>: JSON list of token ids (or {"ids": [...]}). Needs use_local_argmax_reduction in the speculative config (the
drafter only calls get_top_tokens then). The subset weights are built right after the MTP weights load: the v2 runner
captures the draft step in CUDA graphs, so a lazily built path would never be entered.
"""
import json
import os

import torch

_SUB = {}      # id(head) -> (sub_weight [n, H], global ids [n])


def _ids():
    d = json.load(open(os.environ["ARC_DRAFT_VOCAB"]))
    return sorted(set(int(x) for x in (d["ids"] if isinstance(d, dict) else d)))


def _build(head):
    """This rank's draft-head rows. The frequent tokens cluster in the low vocab ids (rank 0's shard), so the subset rows
    are all-gathered once and re-split evenly: each rank then multiplies by ~n/TP rows instead of its own shard's."""
    si = head.shard_indices
    lo, hi = si.org_vocab_start_index, si.org_vocab_end_index
    ids = torch.tensor(_ids(), dtype=torch.long)
    loc = ids[(ids >= lo) & (ids < hi)]
    w = head.weight
    sub = w[(loc - lo).to(w.device)].contiguous()
    gid = loc.to(w.device)
    tp = head.tp_size
    if tp > 1 and os.environ.get("ARC_DRAFT_VOCAB_BALANCE", "1") == "1":
        import torch.distributed as dist
        from vllm.distributed import get_tp_group
        g = get_tp_group()
        n = torch.tensor([sub.shape[0]], device=w.device)
        ns = [torch.zeros_like(n) for _ in range(tp)]
        dist.all_gather(ns, n, group=g.device_group)
        ns = [int(x) for x in ns]
        m = max(ns)
        pad = torch.zeros((m, sub.shape[1]), dtype=sub.dtype, device=w.device); pad[:sub.shape[0]] = sub
        pid = torch.full((m,), -1, dtype=torch.long, device=w.device); pid[:gid.shape[0]] = gid
        rows = [torch.empty_like(pad) for _ in range(tp)]
        rid = [torch.empty_like(pid) for _ in range(tp)]
        dist.all_gather(rows, pad, group=g.device_group)
        dist.all_gather(rid, pid, group=g.device_group)
        allw = torch.cat([r[:k] for r, k in zip(rows, ns)])
        allid = torch.cat([r[:k] for r, k in zip(rid, ns)])
        per = -(-allw.shape[0] // tp)
        r0 = g.rank_in_group * per
        sub, gid = allw[r0:r0 + per].contiguous(), allid[r0:r0 + per].contiguous()
        del rows, allw
    print(f"[arc] draft head: {sub.shape[0]} of {hi - lo} vocab rows on this rank "
          f"({sub.numel() * sub.element_size() / 2**20:.1f} MiB)", flush=True)
    return sub, gid


def install():
    from vllm.distributed import tensor_model_parallel_all_gather
    from vllm.models.deepseek_v32.nvidia import mtp as dmtp

    orig = dmtp.DeepseekV32MultiTokenPredictor.get_top_tokens
    orig_load = dmtp.DeepseekV32MTP.load_weights

    def load_weights(self, weights):
        r = orig_load(self, weights)
        m = self.model
        for i in range(m.num_mtp_layers):
            head = m.layers[str(m.mtp_start_layer_idx + i)].shared_head.head
            if id(head) not in _SUB:
                _SUB[id(head)] = _build(head)
        return r

    def get_top_tokens(self, hidden_states, spec_step_idx=0):
        step = spec_step_idx % self.num_mtp_layers
        head = self.layers[str(self.mtp_start_layer_idx + step)].shared_head.head
        ent = _SUB.get(id(head))
        if ent is None:
            if torch.cuda.is_current_stream_capturing():
                return orig(self, hidden_states, spec_step_idx)
            ent = _SUB[id(head)] = _build(head)
        sub, gid = ent
        lp = self.logits_processor
        n = hidden_states.shape[0]
        if sub.shape[0]:
            logits = hidden_states @ sub.t()
            if lp.soft_cap is not None:
                logits = torch.tanh(logits / lp.soft_cap) * lp.soft_cap
            if lp.scale != 1.0:
                logits = logits * lp.scale
            v, i = logits.max(dim=-1)
            pair = torch.stack([v.float(), gid[i].float()], dim=-1)
        else:
            pair = torch.tensor([[-float("inf"), 0.0]], device=hidden_states.device).expand(n, 2).contiguous()
        tp = head.tp_size
        if tp == 1:
            return pair[:, 1].to(torch.int64)
        g = tensor_model_parallel_all_gather(pair, dim=-1).view(n, tp, 2)
        r = g[:, :, 0].argmax(dim=-1, keepdim=True)
        return g[:, :, 1].gather(dim=-1, index=r).squeeze(-1).to(torch.int64)

    dmtp.DeepseekV32MultiTokenPredictor.get_top_tokens = get_top_tokens
    dmtp.DeepseekV32MTP.load_weights = load_weights
    print(f"[arc] MTP drafts from vocab subset {os.environ['ARC_DRAFT_VOCAB']}", flush=True)
