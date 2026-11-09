import hashlib
import inspect
import os
ON = os.environ.get('FN_MOE_ROUTER_FIRST') == '1'
EXPECTED = os.environ.get('FN_MOE_ROUTER_FIRST_SRC', '')

def _log(msg):
    print(f'[router_first] {msg}', flush=True)

def _forward_impl(self, hidden_states, router_logits, shared_experts_input, input_ids=None):
    self.routed_experts._ensure_moe_quant_config_init()
    if self.gate is not None:
        if self._fse_fuse_gate:
            self._maybe_fuse_gate_weights()
            from vllm.model_executor.layers.utils import dispatch_unquantized_gemm
            router_logits = dispatch_unquantized_gemm()(self, hidden_states, self._combined_gate_weight, None)
        else:
            router_logits, _ = self.gate(hidden_states)
    shared_experts_overlapping = False
    if self._shared_experts is not None:
        shared_experts_overlapping = self._shared_experts.maybe_forward_async(shared_experts_input)
    with self._sequence_parallel_context():
        hidden_states, router_logits = self._maybe_dispatch(hidden_states, router_logits)
        shared_output, hidden_states = self._apply_quant_method(hidden_states=hidden_states, router_logits=router_logits, shared_experts_input=shared_experts_input, input_ids=input_ids, shared_experts_overlapping=shared_experts_overlapping)
        return self._maybe_combine(shared_output, hidden_states)

def source_sha():
    from vllm.model_executor.layers.fused_moe.runner import moe_runner as mr
    return hashlib.sha256(inspect.getsource(mr.MoERunner._forward_impl).encode()).hexdigest()

def install():
    if not ON:
        return False
    from vllm.model_executor.layers.fused_moe.runner import moe_runner as mr
    src = inspect.getsource(mr.MoERunner._forward_impl)
    sha = hashlib.sha256(src.encode()).hexdigest()
    want = EXPECTED or _KNOWN
    if sha != want:
        _log(f'MoERunner._forward_impl source {sha[:16]} != expected {want[:16]}: NOT installed')
        return False
    mr.MoERunner._forward_impl = _forward_impl
    _log('router gate before the shared-expert fork (FN_MOE_ROUTER_FIRST)')
    return True
_KNOWN = '2d358e9a2a7a5b4c4591b5d62d29f9ab59e6983f1664e724143b028361477d96'
