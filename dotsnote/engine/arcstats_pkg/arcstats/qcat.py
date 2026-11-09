"""ARC_MLA_QCAT=1: MLA decode query built in place instead of torch.cat (bit-identical).

vLLM's MLA decode writes the absorbed query q_nope @ W_UK into a fresh [T, H, 1024] buffer and hands (ql_nope, q_pe)
to the backend, which concatenates them into [T, H, 1088] (`CatArrayBatchedCopy`): on L4 at 64 sequences x MTP-4
that is ~67 us per SWA layer and ~70 us per DSA layer, ~3.4 ms per decode step (L4 profile), re-reading and
re-writing 9 MB per layer. Here the bmm writes straight into the first 1024 columns of the [T, H, 1088] buffer (cuBLAS
takes the strided output: no extra copy) and only q_pe (64 columns) is copied, for backends that accept a
concatenated tensor (the Dots3 SWA Triton/CUDA impl and the portable DSA impl; others keep the tuple).

Installed by patching the source of MLAAttention.forward_impl (two anchors, each must match exactly once; otherwise
nothing is patched and a warning is printed).
"""
import inspect
import os
import textwrap

IMPLS = set(os.environ.get("ARC_MLA_QCAT_IMPLS", "Dots3NoteTritonMLAImpl,PortableSparseImpl").split(","))

A1 = """                    mqa_ql_nope = mqa_q_nope.new_empty((B, N, L))
                    torch.bmm(mqa_q_nope, W_UK_T, out=mqa_ql_nope.transpose(0, 1))
"""
B1 = """                    if _arc_qcat:
                        _arc_qbuf = mqa_q_nope.new_empty((B, N, L + mqa_q_pe.shape[-1]))
                        mqa_ql_nope = _arc_qbuf[..., :L]
                        torch.bmm(mqa_q_nope, W_UK_T, out=mqa_ql_nope.transpose(0, 1))
                        _arc_qbuf[..., L:].copy_(mqa_q_pe)
                    else:
                        mqa_ql_nope = mqa_q_nope.new_empty((B, N, L))
                        torch.bmm(mqa_q_nope, W_UK_T, out=mqa_ql_nope.transpose(0, 1))
"""
A2 = """                mqa_q = (mqa_ql_nope, mqa_q_pe)
"""
B2 = """                mqa_q = _arc_qbuf if _arc_qbuf is not None else (mqa_ql_nope, mqa_q_pe)
"""
A3 = """            mqa_q_nope, mqa_q_pe = mqa_q.split(
"""
B3 = """            _arc_qbuf = None
            _arc_qcat = (type(self.impl).__name__ in _ARC_QCAT_IMPLS and self.impl.dcp_world_size <= 1
                         and self.qk_rope_head_dim > 0)
            mqa_q_nope, mqa_q_pe = mqa_q.split(
"""


def install():
    from vllm.model_executor.layers.attention import mla_attention as m
    fn = m.MLAAttention.forward_impl
    src = inspect.getsource(fn)
    for a in (A1, A2, A3):
        if src.count(a) != 1:
            print(f"[qcat] NOT installed: anchor found {src.count(a)}x: {a.strip()[:60]!r}", flush=True)
            return False
    src = src.replace(A1, B1).replace(A2, B2).replace(A3, B3)
    g = dict(m.__dict__)
    g["_ARC_QCAT_IMPLS"] = IMPLS
    code = compile(textwrap.dedent(src), inspect.getsourcefile(fn) + ":qcat", "exec")
    exec(code, g)
    m.MLAAttention.forward_impl = g["forward_impl"]
    print(f"[qcat] MLA decode query concat in place for {sorted(IMPLS)}", flush=True)
    return True
