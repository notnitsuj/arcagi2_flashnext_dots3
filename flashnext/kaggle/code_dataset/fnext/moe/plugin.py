def register():
    import os
    if os.environ.get('FN_MOE_LOWBIT'):
        from . import fnmoe
        fnmoe.register()
    if os.environ.get('FN_MOE_L4CFG'):
        from . import l4cfg
        l4cfg.register()
    if os.environ.get('FN_DENSE_FP8') == '1':
        from . import dense
        dense.register()
    if os.environ.get('FN_MOE_ROUTER_FIRST') == '1':
        from . import router_first
        router_first.install()
    if os.environ.get('FN_MOE_HYB_FUSE', '0') in ('1', '2') or os.environ.get('FN_MOE_CHUNK', '0') not in ('', '0'):
        from . import hyb_fuse
        hyb_fuse.install()
