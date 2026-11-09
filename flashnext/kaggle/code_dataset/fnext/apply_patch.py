import os, re, sys, tempfile

def parse(text):
    files, cur, hunk = ([], None, None)
    for line in text.splitlines(keepends=True):
        if line.startswith('diff --git'):
            cur = None
            continue
        if line.startswith('--- '):
            src = line[4:].strip().split('\t')[0]
            cur = {'src': src, 'dst': None, 'hunks': []}
            files.append(cur)
            hunk = None
            continue
        if line.startswith('+++ ') and cur is not None and (cur['dst'] is None):
            cur['dst'] = line[4:].strip().split('\t')[0]
            continue
        m = re.match('@@ -(\\d+)(?:,(\\d+))? \\+(\\d+)(?:,(\\d+))? @@', line)
        if m and cur is not None:
            hunk = {'old_start': int(m.group(1)), 'lines': []}
            cur['hunks'].append(hunk)
            continue
        if hunk is not None and line[:1] in (' ', '-', '+'):
            hunk['lines'].append(line)
        elif hunk is not None and line.startswith('\\'):
            pass
    return files

def strip(p):
    return None if p == '/dev/null' else p.split('/', 1)[1]

def apply_file(root, f, check):
    src, dst = (strip(f['src']), strip(f['dst']))
    path = os.path.join(root, dst or src)
    old = [] if src is None else open(os.path.join(root, src)).read().splitlines(keepends=True)
    if dst is None:
        if not check:
            os.remove(path)
        return f'deleted {src}'
    out, pos = ([], 0)
    for h in f['hunks']:
        before = [l[1:] for l in h['lines'] if l[0] in ' -']
        after = [l[1:] for l in h['lines'] if l[0] in ' +']
        want = max(0, h['old_start'] - 1)
        found = None
        for d in range(0, 201):
            for s in (want + d, want - d):
                if s >= pos and old[s:s + len(before)] == before:
                    found = s
                    break
            if found is not None:
                break
        if found is None:
            raise SystemExit(f'FAIL {dst}: hunk at line {h['old_start']} does not match')
        out += old[pos:found] + after
        pos = found + len(before)
    out += old[pos:]
    if not check:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path))
        with os.fdopen(fd, 'w') as fh:
            fh.write(''.join(out))
        if os.path.exists(path):
            os.chmod(tmp, os.stat(path).st_mode & 511)
            os.remove(path)
        os.rename(tmp, path)
    return f'{('new' if src is None else 'patched')} {dst} ({len(f['hunks'])} hunks)'

def main():
    root, patch = (sys.argv[1], sys.argv[2])
    check = '--check' in sys.argv
    files = parse(open(patch).read())
    for f in files:
        print('  ' + apply_file(root, f, check))
if __name__ == '__main__':
    main()
