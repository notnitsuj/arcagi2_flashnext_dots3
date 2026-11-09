import os

def _compatible(a, b):
    return a.block_size == b.block_size and a.num_speculative_blocks == b.num_speculative_blocks and (a.num_prefill_checkpoint_blocks == b.num_prefill_checkpoint_blocks) and (a.prefill_checkpoint_alignment == b.prefill_checkpoint_alignment) and (a.mamba_cache_mode == b.mamba_cache_mode) and (a.dcp_sharded == b.dcp_sharded)

def _ring_host_ok(m):
    return m.mamba_cache_mode != 'align' and m.num_speculative_blocks == 0 and (m.num_prefill_checkpoint_blocks == 0)

def pack_groups(groups, bytes_per_block, log=print, ring=False):
    from dataclasses import replace
    from vllm.v1.kv_cache_interface import CircularBufferSpec, MambaSpec, UniformTypeKVCacheSpecs

    def specs_of(g):
        s = g.kv_cache_spec
        if isinstance(s, UniformTypeKVCacheSpecs):
            return dict(s.kv_cache_specs)
        return {name: s for name in g.layer_names}

    def plain(g):
        if g.host_resident or getattr(g, 'is_eagle_group', False):
            return False
        role = getattr(g, 'role', None)
        return role is None or getattr(role, 'name', str(role)) in ('DEFAULT', 'default')

    def is_plain_mamba(g):
        sp = specs_of(g)
        return plain(g) and bool(sp) and all((isinstance(x, MambaSpec) for x in sp.values()))

    def is_ring(g):
        sp = specs_of(g)
        return ring and plain(g) and bool(sp) and all((isinstance(x, CircularBufferSpec) for x in sp.values()))

    def can_host(h, d0):
        h0 = next(iter(specs_of(h).values()))
        if isinstance(d0, MambaSpec):
            return _compatible(h0, d0)
        return isinstance(d0, CircularBufferSpec) and _ring_host_ok(h0)

    def used(g):
        return sum((x.page_size_bytes for x in specs_of(g).values()))
    out = list(groups)
    merged_any = True
    while merged_any:
        merged_any = False
        cands = [g for g in out if is_plain_mamba(g)]
        donors = cands + [g for g in out if is_ring(g)]
        for donor in sorted(donors, key=used):
            dspecs = specs_of(donor)
            d0 = next(iter(dspecs.values()))
            hosts = [h for h in cands if h is not donor and can_host(h, d0) and (used(h) + used(donor) <= bytes_per_block) and (used(h) >= used(donor))]
            if not hosts:
                continue
            host = min(hosts, key=lambda h: bytes_per_block - used(h))
            hspecs = specs_of(host)
            new_specs = {**hspecs, **dspecs}
            first = next(iter(hspecs.values()))
            spec = UniformTypeKVCacheSpecs(block_size=first.block_size, kv_cache_specs=new_specs)
            new_host = replace(host, layer_names=list(host.layer_names) + list(donor.layer_names), kv_cache_spec=spec)
            log(f'[kv_pack_state] merged group {list(donor.layer_names)[:2]}{('...' if len(donor.layer_names) > 2 else '')} ({used(donor)} B) into {list(host.layer_names)[:1]}... ({used(host)} -> {used(new_host)} B of {bytes_per_block} B per block)')
            out = [new_host if g is host else g for g in out if g is not donor]
            merged_any = True
            break
    return out

def _patch_mamba_groups():
    from vllm.v1.kv_cache_interface import MambaSpec, UniformTypeKVCacheSpecs
    from vllm.v1.worker import mamba_utils as mu
    if getattr(mu, '_fn_pack_ring', False):
        return
    mu._fn_pack_ring = True

    def get_mamba_groups(kv_cache_config):
        mamba_groups = {}
        for i, g in enumerate(kv_cache_config.kv_cache_groups):
            spec = g.kv_cache_spec
            per = spec.kv_cache_specs if isinstance(spec, UniformTypeKVCacheSpecs) else {n: spec for n in g.layer_names}
            first = next(iter(per.values()), None)
            if not isinstance(first, MambaSpec):
                continue
            for name in g.layer_names:
                ls = per[name]
                if isinstance(ls, MambaSpec):
                    mamba_groups.setdefault(ls, set()).add(i)
        assert len(mamba_groups) > 0, 'no mamba layers in the model'
        return {spec: sorted(ids) for spec, ids in mamba_groups.items()}
    mu.get_mamba_groups = get_mamba_groups
    try:
        from vllm.v1.worker.gpu.model_states import mamba_hybrid as mh
        mh.get_mamba_groups = get_mamba_groups
    except Exception:
        pass
    _patch_ring_kernel_block()

def _ring_kbs(spec, kbs):
    from vllm.v1.kv_cache_interface import CircularBufferSpec
    if kbs is not None and isinstance(spec, CircularBufferSpec) and (spec.block_size % kbs != 0):
        return None
    return kbs

def _patch_ring_kernel_block():
    import vllm.v1.kv_cache_interface as kci
    from vllm.v1.worker import utils as wu
    if getattr(kci, '_fn_pack_ring', False):
        return
    kci._fn_pack_ring = True
    shape0 = kci.compute_layer_kv_cache_shape_bytes

    def compute_layer_kv_cache_shape_bytes(spec, num_blocks, kernel_block_size=None):
        return shape0(spec, num_blocks, _ring_kbs(spec, kernel_block_size))
    kci.compute_layer_kv_cache_shape_bytes = compute_layer_kv_cache_shape_bytes
    lay0 = kci.compute_layout_strides

    def compute_layout_strides(spec, num_blocks, num_layers, layout, kernel_block_size=None, **kw):
        return lay0(spec, num_blocks, num_layers, layout, _ring_kbs(spec, kernel_block_size), **kw)
    kci.compute_layout_strides = compute_layout_strides
    views0 = kci.create_kv_cache_views

    def create_kv_cache_views(raw, spec, num_blocks, layout, kv_cache_tensor, kernel_block_size=None):
        return views0(raw, spec, num_blocks, layout, kv_cache_tensor, _ring_kbs(spec, kernel_block_size))
    kci.create_kv_cache_views = create_kv_cache_views
    for mod in (wu,):
        for name, fn in (('create_kv_cache_views', create_kv_cache_views), ('compute_layer_kv_cache_shape_bytes', compute_layer_kv_cache_shape_bytes), ('compute_layout_strides', compute_layout_strides)):
            if hasattr(mod, name):
                setattr(mod, name, fn)
    AG = wu.AttentionGroup
    cmb0 = AG.create_metadata_builders

    def create_metadata_builders(self, vllm_config, device, kernel_block_size=None, num_metadata_builders=1):
        return cmb0(self, vllm_config, device, _ring_kbs(self.kv_cache_spec, kernel_block_size), num_metadata_builders)
    AG.create_metadata_builders = create_metadata_builders
    Z = wu.KVBlockZeroer
    z0 = Z.__init__

    def zinit(self, device, attn_groups_iter, kernel_block_sizes, *a, **kw):
        groups = [g for g in attn_groups_iter if _ring_kbs(g.kv_cache_spec, kernel_block_sizes[g.kv_cache_group_id] if g.kv_cache_group_id < len(kernel_block_sizes) else None) == (kernel_block_sizes[g.kv_cache_group_id] if g.kv_cache_group_id < len(kernel_block_sizes) else None)]
        return z0(self, device, iter(groups), kernel_block_sizes, *a, **kw)
    Z.__init__ = zinit

def install():
    if os.environ.get('FN_KV_PACK_STATE') != '1':
        return
    from vllm.v1.core import kv_cache_utils as ku
    if getattr(ku, '_fn_pack_state', False):
        return
    ku._fn_pack_state = True
    orig = ku._get_packed_kv_cache_groups
    verbose = os.environ.get('FN_KV_PACK_STATE_LOG', '1') == '1'
    ring = os.environ.get('FN_KV_PACK_RING') == '1'
    if ring:
        _patch_mamba_groups()

    def _get_packed_kv_cache_groups(vllm_config, kv_cache_spec):
        groups = orig(vllm_config, kv_cache_spec)
        if not groups:
            return groups
        layout = vllm_config.cache_config.get_resolved_kv_cache_layout()
        if not layout.is_block_outermost:
            return groups
        bpb = ku._get_kv_cache_bytes_per_block(groups, layout)
        new = pack_groups(groups, bpb, log=(lambda s: print(s, flush=True)) if verbose else lambda s: None, ring=ring)
        assert ku._get_kv_cache_bytes_per_block(new, layout) == bpb, 'state packing must not widen the block'
        if verbose:
            print(f'[kv_pack_state] groups {len(groups)} -> {len(new)}; block {bpb} B', flush=True)
        return new
    ku._get_packed_kv_cache_groups = _get_packed_kv_cache_groups
