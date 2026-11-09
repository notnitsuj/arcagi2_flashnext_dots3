def register():
    from . import fnmtp
    if fnmtp.flags_on():
        fnmtp.install()
