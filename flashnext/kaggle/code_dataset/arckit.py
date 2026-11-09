from collections import Counter, deque
import inspect
import numpy as np
__all__ = ['grid', 'show', 'bg', 'task_bg', 'palette', 'counts', 'd8', 'D8', 'crop', 'trim', 'bbox', 'scale', 'downscale', 'tile', 'find_sub', 'paste', 'line', 'ray', 'move', 'flood', 'enclosed', 'split_grid', 'period', 'repair_period', 'symmetries', 'complete_symmetry', 'objects', 'Obj', 'diff', 'obj_diff', 'size_rule', 'color_map', 'invariants', 'violations', 'flag', 'check', 'equivariance', 'gravity', 'describe', 'help']
D8 = ('id', 'rot90', 'rot180', 'rot270', 'flipud', 'fliplr', 'transpose', 'antitranspose')

def grid(x):
    g = np.array(x, dtype=np.int64)
    if g.ndim != 2:
        raise ValueError(f'expected a 2-D grid, got shape {g.shape}')
    return g

def show(g, name=None, coords=False):
    g = grid(g)
    rows = [''.join((str(int(v)) if 0 <= v <= 9 else '?' for v in r)) for r in g]
    if coords:
        rows = ['   ' + ''.join((str(j % 10) for j in range(g.shape[1])))] + [f'{i:2d} {r}' for i, r in enumerate(rows)]
    return (f'{name} ({g.shape[0]}x{g.shape[1]}):\n' if name else '') + '\n'.join(rows)

def counts(g):
    v, n = np.unique(grid(g), return_counts=True)
    return dict(sorted(zip(v.tolist(), n.tolist()), key=lambda t: (-t[1], t[0])))

def palette(g):
    return sorted(np.unique(grid(g)).tolist())

def bg(g):
    c = counts(g)
    top = max(c.values())
    tied = [k for k, n in c.items() if n == top]
    return 0 if 0 in tied else min(tied)

def task_bg(train):
    tot = Counter()
    for p in train:
        for k in ('input', 'output'):
            if k in p:
                tot.update(counts(p[k]))
    top = max(tot.values())
    tied = [k for k, n in tot.items() if n == top]
    return 0 if 0 in tied else min(tied)

def d8(g, name=None):
    g = grid(g)
    f = {'id': lambda x: x, 'rot90': lambda x: np.rot90(x, -1), 'rot180': lambda x: np.rot90(x, 2), 'rot270': lambda x: np.rot90(x, 1), 'flipud': np.flipud, 'fliplr': np.fliplr, 'transpose': lambda x: x.T, 'antitranspose': lambda x: np.rot90(x, 2).T}
    if name is not None:
        return f[name](g).copy()
    return {k: f[k](g).copy() for k in D8}

def bbox(mask):
    m = np.asarray(mask).astype(bool)
    rs, cs = np.nonzero(m)
    if rs.size == 0:
        return None
    return (int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max()))

def crop(g, box):
    r0, c0, r1, c1 = box
    return grid(g)[r0:r1 + 1, c0:c1 + 1].copy()

def trim(g, bgc=None):
    g = grid(g)
    b = bbox(g != (bg(g) if bgc is None else bgc))
    return g.copy() if b is None else crop(g, b)

def scale(g, ky, kx=None):
    kx = ky if kx is None else kx
    return np.kron(grid(g), np.ones((ky, kx), dtype=np.int64))

def downscale(g, ky, kx=None, how='mode'):
    g = grid(g)
    kx = ky if kx is None else kx
    H, W = g.shape
    if H % ky or W % kx:
        raise ValueError(f'grid {H}x{W} is not divisible into {ky}x{kx} blocks')
    out = np.zeros((H // ky, W // kx), dtype=np.int64)
    b = bg(g)
    for i in range(H // ky):
        for j in range(W // kx):
            blk = g[i * ky:(i + 1) * ky, j * kx:(j + 1) * kx]
            c = Counter(blk.ravel().tolist())
            if how == 'any' and any((k != b for k in c)):
                c.pop(b, None)
            out[i, j] = c.most_common(1)[0][0]
    return out

def tile(g, ny, nx=None):
    return np.tile(grid(g), (ny, ny if nx is None else nx))

def find_sub(big, small, wildcard=None, limit=100):
    B, s = (grid(big), grid(small))
    H, W = B.shape
    h, w = s.shape
    if h > H or w > W:
        return []
    care = np.ones_like(s, dtype=bool) if wildcard is None else s != wildcard
    win = np.lib.stride_tricks.sliding_window_view(B, (h, w))
    ok = np.all((win == s) | ~care, axis=(2, 3))
    rs, cs = np.nonzero(ok)
    return [(int(r), int(c)) for r, c in zip(rs[:limit], cs[:limit])]

def paste(dst, src, r, c, skip=None):
    D, S = (grid(dst), grid(src))
    H, W = D.shape
    h, w = S.shape
    r0, c0, r1, c1 = (max(r, 0), max(c, 0), min(r + h, H), min(c + w, W))
    if r0 >= r1 or c0 >= c1:
        return D
    sub = S[r0 - r:r1 - r, c0 - c:c1 - c]
    if skip is None:
        D[r0:r1, c0:c1] = sub
    else:
        m = sub != skip
        D[r0:r1, c0:c1][m] = sub[m]
    return D

def line(g, r0, c0, r1, c1, colour):
    g = grid(g)
    n = max(abs(r1 - r0), abs(c1 - c0))
    if n and (not (r0 == r1 or c0 == c1 or abs(r1 - r0) == abs(c1 - c0))):
        raise ValueError('line: only horizontal, vertical or 45-degree segments')
    dr = (r1 > r0) - (r1 < r0)
    dc = (c1 > c0) - (c1 < c0)
    for k in range(n + 1):
        r, c = (r0 + k * dr, c0 + k * dc)
        if 0 <= r < g.shape[0] and 0 <= c < g.shape[1]:
            g[r, c] = colour
    return g

def ray(g, r, c, dr, dc, colour, stop=None, overwrite=True):
    g = grid(g)
    b = bg(g)
    stop = () if stop is None else (stop,) if isinstance(stop, (int, np.integer)) else tuple(stop)
    r, c = (r + dr, c + dc)
    while 0 <= r < g.shape[0] and 0 <= c < g.shape[1]:
        if g[r, c] in stop:
            break
        if overwrite or g[r, c] == b:
            g[r, c] = colour
        r, c = (r + dr, c + dc)
    return g

def move(g, obj, dr, dc, bgc=None):
    g = grid(g)
    b = obj.bg if bgc is None else bgc
    for r, c in obj.cells:
        g[r, c] = b
    for r, c in obj.cells:
        rr, cc = (r + dr, c + dc)
        if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]:
            g[rr, cc] = obj.sub[r - obj.r0, c - obj.c0]
    return g

def _nbrs(diag):
    return ((-1, 0), (1, 0), (0, -1), (0, 1)) + (((-1, -1), (-1, 1), (1, -1), (1, 1)) if diag else ())

def flood(g, r, c, colour, diag=False):
    g = grid(g)
    m = _region(g, r, c, diag)
    g[m] = colour
    return g

def _region(g, r, c, diag=False):
    H, W = g.shape
    v = g[r, c]
    m = np.zeros((H, W), dtype=bool)
    m[r, c] = True
    q = deque([(r, c)])
    nb = _nbrs(diag)
    while q:
        i, j = q.popleft()
        for di, dj in nb:
            a, b = (i + di, j + dj)
            if 0 <= a < H and 0 <= b < W and (not m[a, b]) and (g[a, b] == v):
                m[a, b] = True
                q.append((a, b))
    return m

def enclosed(g, bgc=None, diag=False):
    g = grid(g)
    b = bg(g) if bgc is None else bgc
    H, W = g.shape
    free = g == b
    seen = np.zeros_like(free)
    q = deque()
    for i in range(H):
        for j in (0, W - 1):
            if free[i, j] and (not seen[i, j]):
                seen[i, j] = True
                q.append((i, j))
    for j in range(W):
        for i in (0, H - 1):
            if free[i, j] and (not seen[i, j]):
                seen[i, j] = True
                q.append((i, j))
    nb = _nbrs(diag)
    while q:
        i, j = q.popleft()
        for di, dj in nb:
            a, c = (i + di, j + dj)
            if 0 <= a < H and 0 <= c < W and free[a, c] and (not seen[a, c]):
                seen[a, c] = True
                q.append((a, c))
    return free & ~seen

def split_grid(g):
    g = grid(g)
    H, W = g.shape
    best = None
    b = bg(g)
    for col in np.unique(g).tolist():
        if col == b:
            continue
        rows = [i for i in range(H) if np.all(g[i] == col)]
        cols = [j for j in range(W) if np.all(g[:, j] == col)]
        if not rows and (not cols):
            continue
        if len(rows) == H or len(cols) == W:
            continue
        rb = _bands(rows, H)
        cb = _bands(cols, W)
        if len(rb) * len(cb) < 2:
            continue
        cand = (len(rb) * len(cb), rows, cols, col, rb, cb)
        if best is None or cand[0] > best[0]:
            best = cand
    if best is None:
        return None
    _, rows, cols, col, rb, cb = best
    boxes = [[(r0, c0, r1, c1) for c0, c1 in cb] for r0, r1 in rb]
    return {'rows': rows, 'cols': cols, 'colour': col, 'boxes': boxes, 'cells': [[crop(g, bx) for bx in row] for row in boxes]}

def _bands(lines, n):
    out, start = ([], 0)
    s = set(lines)
    for i in range(n + 1):
        if i == n or i in s:
            if i > start:
                out.append((start, i - 1))
            start = i + 1
    return out

def period(g, ignore=None):
    g = grid(g)
    H, W = g.shape
    care = np.ones_like(g, dtype=bool) if ignore is None else g != ignore

    def ok_r(p):
        for off in range(p):
            sl, m = (g[off::p], care[off::p])
            for j in range(W):
                v = sl[:, j][m[:, j]]
                if v.size and np.any(v != v[0]):
                    return False
        return True

    def ok_c(p):
        for off in range(p):
            sl, m = (g[:, off::p], care[:, off::p])
            for i in range(H):
                v = sl[i][m[i]]
                if v.size and np.any(v != v[0]):
                    return False
        return True
    prs = [p for p in range(1, H // 2 + 1) if ok_r(p)] + [H]
    pcs = [p for p in range(1, W // 2 + 1) if ok_c(p)] + [W]
    for pr_, pc_ in sorted(((a, b) for a in prs for b in pcs), key=lambda t: (t[0] * t[1], t[0])):
        if (pr_, pc_) == (H, W) or _joint_ok(g, care, pr_, pc_):
            return (pr_, pc_)
    return (H, W)

def _joint_ok(g, care, pr, pc):
    H, W = g.shape
    key = (np.arange(H)[:, None] % pr * pc + np.arange(W)[None, :] % pc)[care]
    v = g[care]
    lo = np.full(pr * pc, 99, dtype=np.int64)
    hi = np.full(pr * pc, -1, dtype=np.int64)
    np.minimum.at(lo, key, v)
    np.maximum.at(hi, key, v)
    seen = hi >= 0
    return bool(np.all(lo[seen] == hi[seen]))

def repair_period(g, ignore):
    g = grid(g)
    pr, pc = period(g, ignore)
    if (pr, pc) == g.shape:
        return None
    H, W = g.shape
    care = g != ignore
    ref = {}
    for i in range(H):
        for j in range(W):
            if care[i, j]:
                ref.setdefault((i % pr, j % pc), g[i, j])
    out = g.copy()
    for i in range(H):
        for j in range(W):
            if not care[i, j]:
                v = ref.get((i % pr, j % pc))
                if v is None:
                    return None
                out[i, j] = v
    return out

def symmetries(g, ignore=None):
    g = grid(g)
    out = []
    for k, v in d8(g).items():
        if k == 'id' or v.shape != g.shape:
            continue
        if ignore is None:
            if np.array_equal(v, g):
                out.append(k)
        else:
            m = (g != ignore) & (v != ignore)
            if np.array_equal(v[m], g[m]):
                out.append(k)
    return out

def complete_symmetry(g, ignore, syms=None):
    g = grid(g)
    syms = symmetries(g, ignore) if syms is None else syms
    if not syms:
        return None
    out = g.copy()
    for _ in range(3):
        for k in syms:
            v = d8(out, k)
            m = (out == ignore) & (v != ignore)
            out[m] = v[m]
    return None if np.any(out == ignore) else out

class Obj:
    __slots__ = ('colour', 'colours', 'size', 'r0', 'c0', 'r1', 'c1', 'h', 'w', 'cells', 'mask', 'sub', 'bg', 'shape_key', 'holes', 'border', 'rect', '_sym', 'idx')

    @property
    def box(self):
        return (self.r0, self.c0, self.r1, self.c1)

    @property
    def slice(self):
        return np.s_[self.r0:self.r1 + 1, self.c0:self.c1 + 1]

    @property
    def center(self):
        return ((self.r0 + self.r1) / 2, (self.c0 + self.c1) / 2)

    @property
    def sym(self):
        if self._sym is None:
            self._sym = symmetries(self.mask.astype(np.int64))
        return self._sym

    def __repr__(self):
        cs = '' if len(self.colours) == 1 else f' colours={self.colours}'
        extra = (' rect' if self.rect else '') + (f' holes={self.holes}' if self.holes else '') + (' border' if self.border else '')
        return f'Obj(colour={self.colour}{cs} size={self.size} box=({self.r0},{self.c0},{self.r1},{self.c1}) {self.h}x{self.w}{extra})'

def objects(g, bgc=None, diag=False, multicolour=False, min_size=1):
    g = grid(g)
    b = bg(g) if bgc is None else bgc
    H, W = g.shape
    lab = np.full((H, W), -1, dtype=np.int64)
    nb = _nbrs(diag)
    objs = []
    for i in range(H):
        for j in range(W):
            if lab[i, j] >= 0 or g[i, j] == b:
                continue
            col = g[i, j]
            k = len(objs)
            lab[i, j] = k
            q = deque([(i, j)])
            cells = []
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                for dr, dc in nb:
                    a, e = (r + dr, c + dc)
                    if 0 <= a < H and 0 <= e < W and (lab[a, e] < 0) and (g[a, e] != b) and (multicolour or g[a, e] == col):
                        lab[a, e] = k
                        q.append((a, e))
            objs.append(_mkobj(g, cells, b, H, W))
    objs = [o for o in objs if o.size >= min_size]
    objs.sort(key=lambda o: (-o.size, o.r0, o.c0))
    for k, o in enumerate(objs):
        o.idx = k
    return objs

def _mkobj(g, cells, b, H, W):
    o = Obj()
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    o.r0, o.r1, o.c0, o.c1 = (min(rs), max(rs), min(cs), max(cs))
    o.h, o.w = (o.r1 - o.r0 + 1, o.c1 - o.c0 + 1)
    o.cells = sorted(cells)
    o.size = len(cells)
    o.bg = b
    m = np.zeros((o.h, o.w), dtype=bool)
    for r, c in cells:
        m[r - o.r0, c - o.c0] = True
    o.mask = m
    sub = np.full((o.h, o.w), b, dtype=np.int64)
    blk = g[o.r0:o.r1 + 1, o.c0:o.c1 + 1]
    sub[m] = blk[m]
    o.sub = sub
    cnt = Counter(sub[m].tolist())
    o.colour = cnt.most_common(1)[0][0]
    o.colours = sorted(cnt)
    o.shape_key = (o.h, o.w, m.tobytes())
    o.rect = bool(m.all())
    o.border = o.r0 == 0 or o.c0 == 0 or o.r1 == H - 1 or (o.c1 == W - 1)
    o.holes = _count_holes(m)
    o._sym = None
    o.idx = -1
    return o

def _count_holes(m):
    if m.all():
        return 0
    pad = np.pad(~m, 1, constant_values=True)
    H, W = pad.shape
    seen = np.zeros_like(pad)
    n = 0
    for i in range(H):
        for j in range(W):
            if pad[i, j] and (not seen[i, j]):
                q = deque([(i, j)])
                seen[i, j] = True
                while q:
                    r, c = q.popleft()
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        a, e = (r + dr, c + dc)
                        if 0 <= a < H and 0 <= e < W and pad[a, e] and (not seen[a, e]):
                            seen[a, e] = True
                            q.append((a, e))
                n += 1
    return n - 1

class Diff(dict):

    def __str__(self):
        return self['text']
    __repr__ = __str__

def diff(a, b, max_list=6):
    A, B = (grid(a), grid(b))
    d = Diff(same_shape=A.shape == B.shape, shape_a=A.shape, shape_b=B.shape)
    if A.shape != B.shape:
        lines = [f'shape {A.shape[0]}x{A.shape[1]} -> {B.shape[0]}x{B.shape[1]}']
        rel = _shape_relation(A, B)
        if rel:
            lines.append(rel)
        ca, cb = (set(palette(A)), set(palette(B)))
        if cb - ca:
            lines.append(f'new colours {sorted(cb - ca)}')
        if ca - cb:
            lines.append(f'dropped colours {sorted(ca - cb)}')
        d.update(n_changed=None, transitions={}, box=None, text='; '.join(lines))
        return d
    m = A != B
    n = int(m.sum())
    tr = Counter(zip(A[m].tolist(), B[m].tolist()))
    d.update(n_changed=n, transitions=dict(tr.most_common()), box=bbox(m), mask=m)
    if n == 0:
        d['text'] = 'identical'
        return d
    items = ', '.join((f'{x}->{y}:{k}' for (x, y), k in tr.most_common(max_list)))
    more = f' (+{len(tr) - max_list} more)' if len(tr) > max_list else ''
    d['text'] = f'same shape; {n}/{A.size} cells change in box {d['box']}; {items}{more}'
    return d

def _shape_relation(A, B):
    (ha, wa), (hb, wb) = (A.shape, B.shape)
    for k, v in d8(A).items():
        if v.shape == B.shape and np.array_equal(v, B):
            return f'output = input {k}'
    if hb % ha == 0 and wb % wa == 0:
        fy, fx = (hb // ha, wb // wa)
        if np.array_equal(scale(A, fy, fx), B):
            return f'output = input scaled x{fy}x{fx}'
        if np.array_equal(tile(A, fy, fx), B):
            return f'output = input tiled {fy}x{fx}'
        lay = []
        for i in range(fy):
            row = []
            for j in range(fx):
                blk = B[i * ha:(i + 1) * ha, j * wa:(j + 1) * wa]
                nm = next((k for k, v in d8(A).items() if v.shape == blk.shape and np.array_equal(v, blk)), None)
                row.append(nm)
            lay.append(row)
        if all((x is not None for r in lay for x in r)):
            return f'output = {fy}x{fx} blocks of the input under {lay}'
    if ha % hb == 0 and wa % wb == 0:
        fy, fx = (ha // hb, wa // wb)
        if np.array_equal(scale(B, fy, fx), A):
            return f'output = input downscaled /{fy}x{fx}'
    if hb <= ha and wb <= wa:
        hits = find_sub(A, B, limit=4)
        if hits:
            return f'output = crop of input at {hits[:3]}'
        for k, v in d8(B).items():
            if v.shape[0] <= ha and v.shape[1] <= wa and find_sub(A, v, limit=1):
                return f'output = {k} of a crop of the input'
    return None

def obj_diff(a, b, bgc=None, diag=False, multicolour=False):
    A, B = (grid(a), grid(b))
    b0 = bg(A) if bgc is None else bgc
    oa = objects(A, b0, diag, multicolour)
    ob = objects(B, b0, diag, multicolour)
    used_a, used_b, ev = (set(), set(), [])

    def take(keyf, kind):
        buck = {}
        for y in ob:
            if y.idx not in used_b:
                buck.setdefault(keyf(y), []).append(y)
        for x in oa:
            if x.idx in used_a:
                continue
            cand = [y for y in buck.get(keyf(x), ()) if y.idx not in used_b]
            if not cand:
                continue
            y = min(cand, key=lambda y: abs(x.r0 - y.r0) + abs(x.c0 - y.c0))
            used_a.add(x.idx)
            used_b.add(y.idx)
            if kind == 'exact':
                if (x.r0, x.c0) == (y.r0, y.c0):
                    ev.append(('same', x, y))
                else:
                    ev.append(('moved', x, y, (y.r0 - x.r0, y.c0 - x.c0)))
            else:
                ev.append(('recoloured', x, y, (x.colour, y.colour)))
    take(lambda o: (o.shape_key, o.sub.tobytes()), 'exact')
    take(lambda o: (o.shape_key, o.r0, o.c0), 'shape')
    ra = [x for x in oa if x.idx not in used_a]
    rb = [y for y in ob if y.idx not in used_b]
    if len(ra) * len(rb) <= 40000:
        for x in ra:
            best = None
            for y in rb:
                if y.idx in used_b or x.r1 < y.r0 or y.r1 < x.r0 or (x.c1 < y.c0) or (y.c1 < x.c0):
                    continue
                dist = abs(x.r0 - y.r0) + abs(x.c0 - y.c0)
                if best is None or dist < best[0]:
                    best = (dist, y)
            if best is not None:
                used_a.add(x.idx)
                used_b.add(best[1].idx)
                ev.append(('reshaped', x, best[1]))
    ev += [('deleted', x) for x in oa if x.idx not in used_a]
    ev += [('created', y) for y in ob if y.idx not in used_b]
    return ev

def obj_diff_text(a, b, **kw):
    ev = obj_diff(a, b, **kw)
    c = Counter((e[0] for e in ev))
    parts = [', '.join((f'{k} {c[k]}' for k in ('same', 'moved', 'recoloured', 'reshaped', 'deleted', 'created') if c[k]))]
    for e in ev:
        if e[0] == 'moved':
            parts.append(f'moved {e[1]!r} by {e[3]}')
        elif e[0] == 'recoloured':
            parts.append(f'recoloured {e[1]!r} -> {e[3][1]}')
        elif e[0] == 'reshaped':
            parts.append(f'reshaped {e[1]!r} -> {e[2]!r}')
        elif e[0] == 'deleted':
            parts.append(f'deleted {e[1]!r}')
        elif e[0] == 'created':
            parts.append(f'created {e[1]!r}')
    return '\n'.join(parts[:13])

def _pairs(train):
    return [(grid(p['input']), grid(p['output'])) for p in train]

class SizeRule:

    def __init__(self, kind, fn):
        self.kind, self.fn = (kind, fn)

    def predict(self, g):
        try:
            return self.fn(grid(g))
        except Exception:
            return None

    def __repr__(self):
        return f'SizeRule({self.kind})'

def size_rule(train, all_rules=False):
    P = _pairs(train)
    cands = [('same as input', lambda g: g.shape), ('transposed input', lambda g: g.shape[::-1])]
    shp = {o.shape for _, o in P}
    if len(shp) == 1:
        s = next(iter(shp))
        cands.append((f'constant {s[0]}x{s[1]}', lambda g, s=s: s))
    i0, o0 = P[0]
    if o0.shape[0] % i0.shape[0] == 0 and o0.shape[1] % i0.shape[1] == 0:
        fy, fx = (o0.shape[0] // i0.shape[0], o0.shape[1] // i0.shape[1])
        cands.append((f'input x{fy} rows, x{fx} cols', lambda g, a=fy, b=fx: (g.shape[0] * a, g.shape[1] * b)))
    if i0.shape[0] % o0.shape[0] == 0 and i0.shape[1] % o0.shape[1] == 0:
        fy, fx = (i0.shape[0] // o0.shape[0], i0.shape[1] // o0.shape[1])
        cands.append((f'input /{fy} rows, /{fx} cols', lambda g, a=fy, b=fx: (g.shape[0] // a, g.shape[1] // b)))
    cands.append(('input size squared (h*h x w*w)', lambda g: (g.shape[0] ** 2, g.shape[1] ** 2)))
    cands.append(('one cell of the separator split', lambda g: _cellshape(g)))
    cands.append(('box of non-background cells', lambda g: _bbshape(trim(g))))
    for diag in (False, True):
        for mc in (False, True):
            tag = ('8' if diag else '4') + ('-multicolour' if mc else '')
            cands.append((f'largest object box ({tag})', lambda g, d=diag, m=mc: _objshape(g, d, m, 'largest')))
            cands.append((f'smallest object box ({tag})', lambda g, d=diag, m=mc: _objshape(g, d, m, 'smallest')))
    cands.append(('n_colours x n_colours', lambda g: (len(palette(g)) - 1,) * 2))
    dy, dx = (o0.shape[0] - i0.shape[0], o0.shape[1] - i0.shape[1])
    cands.append((f'input {dy:+d} rows, {dx:+d} cols', lambda g, a=dy, b=dx: (g.shape[0] + a, g.shape[1] + b)))
    fits = []
    for kind, fn in cands:
        try:
            if all((tuple(fn(i)) == o.shape for i, o in P)):
                fits.append(SizeRule(kind, lambda g, f=fn: tuple((int(x) for x in f(g)))))
        except Exception:
            continue
    if all_rules:
        return fits
    return fits[0] if fits else None

def _bbshape(g):
    return g.shape

def _objshape(g, diag, mc, which):
    os_ = objects(g, diag=diag, multicolour=mc)
    if not os_:
        return None
    o = os_[0] if which == 'largest' else os_[-1]
    if which == 'largest' and len(os_) > 1 and (os_[1].size == o.size):
        return None
    return (o.h, o.w)

def _cellshape(g):
    s = split_grid(g)
    if s is None:
        return None
    shp = {c.shape for row in s['cells'] for c in row}
    return next(iter(shp)) if len(shp) == 1 else None

def color_map(train):
    m = {}
    for i, o in _pairs(train):
        if i.shape != o.shape:
            return None
        for x, y in zip(i.ravel().tolist(), o.ravel().tolist()):
            if m.setdefault(x, y) != y:
                return None
    return m

def _inv_defs(train):
    P = _pairs(train)
    out_pal = set().union(*(set(palette(o)) for _, o in P))
    new_cols = [set(palette(o)) - set(palette(i)) for i, o in P]
    const_new = set.intersection(*new_cols) if new_cols else set()
    any_new = set().union(*new_cols)
    defs = [('output shape == input shape', lambda i, o: i.shape == o.shape), ('output colours subset of input colours', lambda i, o: set(palette(o)) <= set(palette(i))), ('output colours subset of input colours + colours added in training outputs', lambda i, o: set(palette(o)) <= set(palette(i)) | any_new), ('output colours subset of training-output colours + input colours', lambda i, o: set(palette(o)) <= out_pal | set(palette(i))), ('output background == input background', lambda i, o: bg(o) == bg(i)), ('output differs from input', lambda i, o: i.shape != o.shape or not np.array_equal(i, o)), ('output has >= 2 colours', lambda i, o: len(palette(o)) >= 2), ('non-background input cells kept', lambda i, o: i.shape == o.shape and np.array_equal(o[i != bg(i)], i[i != bg(i)])), ('only background cells change', lambda i, o: i.shape == o.shape and bool(np.all((i == o) | (i == bg(i))))), ('background input cells kept', lambda i, o: i.shape == o.shape and bool(np.all(o[i == bg(i)] == bg(i)))), ('output is a sub-grid of the input', lambda i, o: o.shape[0] <= i.shape[0] and o.shape[1] <= i.shape[1] and bool(find_sub(i, o, limit=1))), ('output colours in every training output', lambda i, o: set(palette(o)) <= set.intersection(*(set(palette(oo)) for _, oo in P)) | set(palette(i))), ('colour counts never decrease', lambda i, o: i.shape == o.shape and all((counts(o).get(c, 0) >= n for c, n in counts(i).items() if c != bg(i)))), ('colour counts never increase', lambda i, o: i.shape == o.shape and all((n <= counts(i).get(c, 0) for c, n in counts(o).items() if c != bg(i)))), ('same number of colours', lambda i, o: len(palette(o)) == len(palette(i))), ('same object count (4-conn)', lambda i, o: len(objects(i)) == len(objects(o, bgc=bg(i)))), ('output smaller than input', lambda i, o: o.size < i.size), ('output larger than input', lambda i, o: o.size > i.size)]
    if const_new:
        defs.append((f'output contains colour(s) {sorted(const_new)}', lambda i, o, c=frozenset(const_new): c <= set(palette(o))))
    for s in ('fliplr', 'flipud', 'transpose', 'rot180', 'rot90'):
        defs.append((f'output symmetric ({s})', lambda i, o, s=s: s in symmetries(o)))
    srs = size_rule(train, all_rules=True)
    if srs:
        defs.append((f'output shape follows rule: {srs[0].kind}', lambda i, o, rs=tuple(srs): any((r.predict(i) == o.shape for r in rs))))
    return defs
SAFE = ('output colours subset of training-output colours + input colours', 'output colours subset of input colours + colours added in training outputs', 'output differs from input', 'output has >= 2 colours', 'output colours in every training output', 'output shape follows rule', 'output colours subset of input colours', 'output shape == input shape', 'same number of colours', 'colour counts never decrease', 'colour counts never increase', 'non-background input cells kept', 'only background cells change', 'output smaller than input', 'output larger than input', 'output is a sub-grid of the input')

def invariants(train, names_only=True):
    P = _pairs(train)
    out = []
    for name, f in _inv_defs(train):
        try:
            if all((bool(f(i, o)) for i, o in P)):
                out.append((name, f))
        except Exception:
            continue
    return [n for n, _ in out] if names_only else out
SIZE_INV = ('output shape follows rule', 'output smaller than input', 'output larger than input', 'output shape == input shape', 'output is a sub-grid of the input')

def flag(train, test_inputs, test_outputs):
    bad = set()
    for v in violations(train, test_inputs, test_outputs):
        bad.update((x for x in v if not x.startswith(SIZE_INV)))
    return sorted(bad)

def violations(train, test_inputs, test_outputs, safe_only=True):
    inv = invariants(train, names_only=False)
    if safe_only and SAFE is not None:
        inv = [(n, f) for n, f in inv if any((n.startswith(s) for s in SAFE))]
    res = []
    for ti, to in zip(test_inputs, test_outputs):
        if to is None:
            res.append(['no output'])
            continue
        i, o = (grid(ti), grid(to))
        bad = []
        for n, f in inv:
            try:
                if not f(i, o):
                    bad.append(n)
            except Exception:
                bad.append(n)
        res.append(bad)
    return res

def _call(f, g, **kw):
    try:
        r = f([list(map(int, row)) for row in grid(g).tolist()], **kw)
    except Exception:
        r = f(grid(g), **kw)
    return grid(r)

def _masked(exp, got, max_cells=400):
    if exp.size > max_cells:
        return ''
    return '\n'.join((''.join(('.' if e == x else str(int(x)) for e, x in zip(re_, rg))) for re_, rg in zip(exp, got)))

def check(transform, train, test_inputs=None, verbose=True, show_diff=True, equiv=True):
    P = _pairs(train)
    loo = 'train' in _params(transform)
    lines, rep = ([], {'pairs': [], 'tests': [], 'loo': None})
    for k, (i, o) in enumerate(P):
        r = {'ok': False}
        try:
            got = _call(transform, i, **{'train': train} if loo else {})
            r['out'] = got
            if got.shape != o.shape:
                r['why'] = f'shape {got.shape[0]}x{got.shape[1]}, expected {o.shape[0]}x{o.shape[1]}'
            elif np.array_equal(got, o):
                r['ok'] = True
            else:
                m = got != o
                conf = Counter(zip(o[m].tolist(), got[m].tolist())).most_common(4)
                r['why'] = f'{int(m.sum())}/{o.size} cells wrong in box {bbox(m)}; expected->got ' + ', '.join((f'{a}->{b}:{n}' for (a, b), n in conf))
                if show_diff:
                    r['masked'] = _masked(o, got)
        except Exception as e:
            r['why'] = f'raised {type(e).__name__}: {str(e)[:150]}'
        rep['pairs'].append(r)
        lines.append(f'pair {k}: ' + ('ok' if r['ok'] else r['why']))
        if r.get('masked') and sum((not x['ok'] for x in rep['pairs'])) <= 2:
            lines.append(r['masked'])
    n_ok = sum((r['ok'] for r in rep['pairs']))
    lines.insert(0, f'check: {n_ok}/{len(P)} training pairs correct')
    if loo and len(P) > 1:
        lo = []
        for k, (i, o) in enumerate(P):
            sub = [p for j, p in enumerate(train) if j != k]
            try:
                lo.append(bool(np.array_equal(_call(transform, i, train=sub), o)))
            except Exception:
                lo.append(False)
        rep['loo'] = lo
        lines.append(f'leave-one-out: {sum(lo)}/{len(lo)} pairs reproduced from the other pairs ' + str(['ok' if x else 'FAIL' for x in lo]))
    if test_inputs:
        outs = []
        for t, g in enumerate(test_inputs):
            try:
                outs.append(_call(transform, g, **{'train': train} if loo else {}))
            except Exception as e:
                outs.append(None)
                lines.append(f'test {t}: raised {type(e).__name__}: {str(e)[:150]}')
        srs = size_rule(train, all_rules=True)
        viol = violations(train, test_inputs, outs)
        eq = equivariance(transform, train, test_inputs, outs) if equiv and n_ok == len(P) else None
        for t, (g, out) in enumerate(zip(test_inputs, outs)):
            strong = [v for v in viol[t] if v != 'no output' and (not v.startswith(SIZE_INV))]
            soft = [v for v in viol[t] if v.startswith(SIZE_INV)]
            notes = []
            if out is not None:
                pred = [r.predict(g) for r in srs]
                if pred and out.shape not in pred and (None not in pred):
                    soft = [v for v in soft if not v.startswith('output shape follows')]
                    soft.insert(0, f"shape {out.shape[0]}x{out.shape[1]} but the training size rule '{srs[0].kind}' predicts {pred[0][0]}x{pred[0][1]}")
                if any((np.array_equal(out, o) for _, o in P)) and len({o.tobytes() for _, o in P}) == len(P):
                    soft.append('output equals a training output')
                if eq and eq[t]:
                    notes.append('transform is equivariant under ' + ', '.join(eq[t]) + ' on every training pair but not on this test input')
            rep['tests'].append({'out': out, 'warnings': strong, 'soft': soft, 'notes': notes})
            if out is not None:
                txt = f'test {t}: output {out.shape[0]}x{out.shape[1]}'
                if strong:
                    txt += '; WARNING (every training pair satisfies this, the test output does not): ' + '; '.join(strong)
                if soft:
                    txt += '; size check (test inputs may legitimately differ): ' + '; '.join(soft)
                if notes:
                    txt += '; note (weak signal): ' + '; '.join(notes)
                if not (strong or soft or notes):
                    txt += '; no warnings'
                lines.append(txt)
    rep['text'] = '\n'.join(lines)
    rep['ok'] = n_ok == len(P)
    if verbose:
        print(rep['text'])
    return rep
_PERMS = ((1, 2, 3, 4, 5, 6, 7, 8, 9), (2, 3, 4, 5, 6, 7, 8, 9, 1), (5, 6, 7, 8, 9, 1, 2, 3, 4), (9, 8, 7, 6, 5, 4, 3, 2, 1))

def _cperm(k):
    lut = np.arange(10)
    lut[1:] = _PERMS[k]
    return lambda g: lut[grid(g)]

def equivariance(transform, train, test_inputs, test_outputs=None, budget_s=2.0):
    import time
    t0 = time.time()
    P = _pairs(train)
    kw = {'train': train} if 'train' in _params(transform) else {}
    maps = [(k, lambda g, k=k: d8(g, k)) for k in D8[1:]] + [(f'colour perm {k}', _cperm(k)) for k in (1, 2, 3)]
    if test_outputs is None:
        test_outputs = []
        for g in test_inputs:
            try:
                test_outputs.append(_call(transform, g, **kw))
            except Exception:
                test_outputs.append(None)
    res = [[] for _ in test_inputs]
    for name, m in maps:
        if time.time() - t0 > budget_s:
            break
        try:
            if not all((np.array_equal(_call(transform, m(i), **kw), m(o)) for i, o in P)):
                continue
        except Exception:
            continue
        for t, (g, out) in enumerate(zip(test_inputs, test_outputs)):
            if out is None:
                continue
            try:
                ok = np.array_equal(_call(transform, m(g), **kw), m(out))
            except Exception:
                ok = False
            if not ok:
                res[t].append(name)
    return res

def gravity(g, direction='down', rigid=False, bgc=None, diag=False):
    g = grid(g)
    b = bg(g) if bgc is None else bgc
    k = {'down': 0, 'left': 1, 'up': 2, 'right': 3}[direction]
    r = np.rot90(g, k).copy()
    H, W = r.shape
    if not rigid:
        out = np.full_like(r, b)
        for j in range(W):
            col = [v for v in r[:, j] if v != b]
            if col:
                out[H - len(col):, j] = col
        return np.rot90(out, -k).copy()
    objs = sorted(objects(r, b, diag=diag), key=lambda o: -o.r1)
    for o in objs:
        cells = [(rr, cc) for rr, cc in o.cells]
        vals = [r[rr, cc] for rr, cc in cells]
        own = set(cells)
        d = 0
        while all((rr + d + 1 < H and ((rr + d + 1, cc) in own or r[rr + d + 1, cc] == b) for rr, cc in cells)):
            d += 1
        if d:
            for rr, cc in cells:
                r[rr, cc] = b
            for (rr, cc), v in zip(cells, vals):
                r[rr + d, cc] = v
    return np.rot90(r, -k).copy()

def _params(f):
    try:
        return inspect.signature(f).parameters
    except (TypeError, ValueError):
        return {}

def describe(train, test_inputs=(), max_obj=4, budget=1800):
    P = _pairs(train)
    out = []
    sr = size_rule(train)
    cm = color_map(train)
    b = task_bg(train)
    out.append(f'{len(P)} train pairs; task background {b}; output size rule: {(sr.kind if sr else 'none found')}' + (f'; consistent cell colour map {cm}' if cm and any((k != v for k, v in cm.items())) else ''))
    for k, (i, o) in enumerate(P):
        d = diff(i, o)
        line = f'pair {k}: {i.shape[0]}x{i.shape[1]} -> {o.shape[0]}x{o.shape[1]}; {d['text']}'
        oi = objects(i, b)
        line += f'; input objects {len(oi)}'
        if d['same_shape'] and d['n_changed']:
            ev = obj_diff(i, o, b)
            c = Counter((e[0] for e in ev))
            line += ' (' + ', '.join((f'{x} {c[x]}' for x in ('same', 'moved', 'recoloured', 'reshaped', 'deleted', 'created') if c[x])) + ')'
        s = split_grid(i)
        if s:
            line += f'; separators colour {s['colour']} -> {len(s['boxes'])}x{len(s['boxes'][0])} cells'
        pi = period(i)
        if pi != i.shape:
            line += f'; input period {pi}'
        sy = symmetries(o)
        if sy:
            line += f'; output symmetric {sy}'
        out.append(line)
        if k == 0 and oi:
            out.append('  objects of pair 0 input: ' + '; '.join((repr(x)[4:-1] for x in oi[:max_obj])) + (f'; +{len(oi) - max_obj} more' if len(oi) > max_obj else ''))
    inv = [n for n in invariants(train) if not n.startswith('output shape follows')]
    if 'output colours subset of input colours' in inv:
        inv = [n for n in inv if not n.startswith(('output colours subset of input colours +', 'output colours subset of training-output'))]
    if inv:
        out.append('holds on all pairs: ' + '; '.join(inv[:10]))
    for t, g in enumerate(test_inputs):
        g = grid(g)
        nov = []
        pal = set().union(*(set(palette(i)) for i, _ in P))
        if set(palette(g)) - pal:
            nov.append(f'unseen colours {sorted(set(palette(g)) - pal)}')
        if sr is not None:
            nov.append(f'predicted output size {sr.predict(g)}')
        out.append(f'test {t}: {g.shape[0]}x{g.shape[1]}, {len(objects(g, b))} objects' + (f'; {'; '.join(nov)}' if nov else ''))
    txt = '\n'.join(out)
    if len(txt) > budget:
        txt = txt[:budget - 20].rsplit('\n', 1)[0] + '\n[... truncated]'
    return txt
HELP = "arckit is pre-loaded as `ak` (numpy only; scipy is not available: use ak.objects for labelling and\nak.enclosed for hole filling). Grids in: lists or arrays; grids out: new numpy int arrays (.tolist() for lists).\nBoxes are (r0, c0, r1, c1) inclusive; coordinates (row, col).\n ak.describe(train, test_inputs) -> str: size rule, per-pair diffs, objects, invariants holding on all pairs\n ak.check(transform, train, test_inputs): per-pair diffs + warnings on test outputs (shape rule, palette,\n     output == input, ...); with def transform(grid, train=None) it also runs leave-one-out\n ak.objects(g, bgc=None, diag=False, multicolour=False) -> [Obj] largest first; Obj fields: colour colours size\n     box r0 c0 r1 c1 h w cells mask sub shape_key holes border rect sym center slice\n ak.diff(a, b) -> changed cells, (from,to) counts, box, shape relation\n ak.obj_diff(a, b) -> [('same'|'moved'|'recoloured'|'reshaped'|'deleted'|'created', obj_a, obj_b, ...)]\n ak.bg(g) ak.task_bg(train) ak.palette(g) ak.counts(g) ak.bbox(mask) ak.crop(g, box) ak.trim(g)\n ak.d8(g) -> {id rot90 rot180 rot270 flipud fliplr transpose antitranspose: grid}; ak.d8(g, 'rot90')\n ak.scale(g, k) ak.downscale(g, k) ak.tile(g, ny, nx) ak.paste(dst, src, r, c, skip=None)\n ak.find_sub(big, small, wildcard=None) -> [(r, c)]  ak.flood(g, r, c, colour)  ak.enclosed(g) -> hole mask\n ak.line(g, r0, c0, r1, c1, colour)  ak.ray(g, r, c, dr, dc, colour, stop=None)  ak.move(g, obj, dr, dc)\n ak.gravity(g, 'down'|'up'|'left'|'right', rigid=False)  (rigid=True: whole objects fall until blocked)\n ak.show(g, coords=True) -> str with row/col indices\n ak.split_grid(g) -> {rows cols colour boxes cells} for full separator lines, or None\n ak.period(g, ignore=None) -> (pr, pc)  ak.repair_period(g, ignore)  ak.symmetries(g, ignore=None)\n ak.complete_symmetry(g, ignore)  ak.size_rule(train).predict(g)  ak.color_map(train)  ak.invariants(train)"

def help():
    print(HELP)
