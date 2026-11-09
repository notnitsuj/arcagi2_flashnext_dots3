import gzip, json, sys
from pathlib import Path
for d in map(Path, sys.argv[1:]):
    n = calls = err = 0
    try:
        for l in gzip.open(d / 'traces.jsonl.gz', 'rt'):
            try:
                r = json.loads(l)
            except Exception:
                break
            if r['arm'] != 'repl':
                continue
            n += 1
            calls += r['text'].count('<tool_call>')
            err += r['text'].count('Error: no `code` parameter')
    except (EOFError, OSError):
        pass
    sp = d / 'samples.jsonl'
    rows = [json.loads(l) for l in open(sp) if l.strip().endswith('}')] if sp.exists() else []
    rows = [r for r in rows if r['arm'] == 'repl' and r['status'] != 'error']
    ver = sum((any((p['verified'] for p in r['progs'])) for r in rows))
    vc = [all([p for p in r['progs'] if p['verified']][-1]['tests_correct']) for r in rows if any((p['verified'] for p in r['progs']))]
    print(f'{d.name}: sessions {len(rows)}, verified {ver / max(1, len(rows)):.3f}, verified precision {sum(vc) / max(1, len(vc)):.3f}, tool calls {calls} ({calls / max(1, n):.1f}/session), no-code errors {err} ({err / max(1, calls):.3f} of calls)')
