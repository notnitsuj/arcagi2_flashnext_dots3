import os

def register():
    if os.environ.get('FN_SESSION_EP', '0') == '1':
        from . import session_ep
        session_ep.install()
    if os.environ.get('FN_STEPPROF', '0') not in ('', '0'):
        from . import stepprof
        stepprof.install()
    if os.environ.get('FN_SAMPPROF', '0') == '1':
        from . import sampprof
        sampprof.install()
    if os.environ.get('FN_DETOK_ROPE', '0') == '1':
        from . import detok_rope
        detok_rope.install()
