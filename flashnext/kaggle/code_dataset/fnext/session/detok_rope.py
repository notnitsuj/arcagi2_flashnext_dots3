def install():
    from vllm.v1.engine import detokenizer as dk
    B = dk.BaseIncrementalDetokenizer
    if getattr(B, '_fn_rope', False):
        return
    B._fn_rope = True
    upd0, next0 = (B.update, B.get_next_output_text)

    def _get(self):
        p = self.__dict__.get('_pcs')
        if p is None:
            return ''
        if len(p) > 1:
            s = ''.join(p)
            p[:] = [s]
        return p[0] if p else ''

    def _set(self, v):
        self.__dict__['_pcs'] = [v] if v else []
        self.__dict__['_plen'] = len(v)
        self.__dict__['_unsent'] = None
    B.output_text = property(_get, _set)

    def update(self, new_token_ids, stop_terminated):
        if self.stop or not new_token_ids:
            return upd0(self, new_token_ids, stop_terminated)
        if stop_terminated and (not self.include_stop_str_in_output):
            skipped = new_token_ids[-1]
            new_token_ids = new_token_ids[:-1]
        else:
            skipped = None
        d = self.__dict__
        pcs = d.setdefault('_pcs', [])
        un = d.get('_unsent')
        if un is None and d.get('_plen', 0) == self._last_output_text_offset:
            un = d['_unsent'] = []
        n = d.get('_plen', 0)
        for t in new_token_ids:
            self.token_ids.append(t)
            s = self.decode_next(t)
            if s:
                pcs.append(s)
                n += len(s)
                if un is not None:
                    un.append(s)
        d['_plen'] = n
        if len(pcs) > 4096:
            pcs[:] = [''.join(pcs)]
        if skipped is not None:
            self.token_ids.append(skipped)
        return None

    def get_next_output_text(self, finished, delta):
        d = self.__dict__
        un = d.get('_unsent')
        if delta and (not self.stop) and (un is not None):
            n = d.get('_plen', 0)
            if self._last_output_text_offset < n:
                self._last_output_text_offset = n
                s = ''.join(un)
                un.clear()
                return s
            return ''
        d['_unsent'] = None
        return next0(self, finished, delta)
    B.update = update
    B.get_next_output_text = get_next_output_text
