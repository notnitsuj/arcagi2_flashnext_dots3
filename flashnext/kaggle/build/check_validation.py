import json, re, sys
from pathlib import Path
out = Path(sys.argv[1])
MIN_TOK = float(sys.argv[sys.argv.index('--min-tok') + 1]) if '--min-tok' in sys.argv else 900.0
MIN_PASS = float(sys.argv[sys.argv.index('--min-pass') + 1]) if '--min-pass' in sys.argv else 20.0
nblog = next(out.glob('*.log'))
txt = ''
for line in open(nblog, errors='replace'):
    line = line.strip().lstrip('[,').rstrip(']')
    if line.startswith('{'):
        try:
            txt += json.loads(line).get('data', '')
        except Exception:
            pass
if not txt:
    txt = open(nblog, errors='replace').read()
fn = out / 'fn'
rd = lambda p: (fn / p).read_text(errors='replace') if (fn / p).exists() else ''
solve, server = (rd('solve.log'), rd('server_log.txt'))
samples = rd('repl_samples.jsonl')
res = {}
res['all stages ran'] = 'stage status: all stages ran' in txt and 'STAGE FAILED' not in txt
res['server healthy'] = 'server healthy' in txt or 'server healthy' in solve
res['no server restart'] = '[supervisor] restart' not in server
m = re.findall('generated (\\d+) tokens in (\\d+)s = (\\d+) tok/s', solve)
tok = int(m[-1][2]) if m else 0
res[f'tok/s >= {MIN_TOK:.0f} (got {tok})'] = tok >= MIN_TOK
ak = (solve + samples + server).count("No module named 'ak'") + (solve + samples).count('No module named \\"ak\\"')
res[f"0 'No module named ak' (got {ak})"] = ak == 0
pm = re.findall('EVAL pass@2: ([\\d.]+) / (\\d+) tasks = ([\\d.]+)%', txt)
pct = float(pm[-1][2]) if pm else -1
res[f"EVAL pass@2 >= {MIN_PASS}% (got {pct}%, {(pm[-1][0] + '/' + pm[-1][1] if pm else 'missing')})"] = pct >= MIN_PASS
res['fused host AG installed'] = 'FN_HOST_DP_FUSE: fused multi-tensor host all-gather installed' in server
first = len(re.findall('first fused host all-gather', server))
res[f'fused host AG used (lines {first} >= 4 ranks)'] = first >= 4
res['router_first installed'] = 'router gate before the shared-expert fork' in server and 'NOT installed' not in server
res['no spin-wait timeout'] = 'spin-wait timeout' not in server.lower() and 'spin wait timeout' not in server.lower()
res['late_pf.log exists'] = (fn / 'late_pf.log').exists()
res['fcache_stat.log exists'] = (fn / 'fcache_stat.log').exists()
cm = re.search('concurrency (\\d+)', solve)
res[f'concurrency 48 (got {(cm.group(1) if cm else None)})'] = bool(cm) and cm.group(1) == '48'
tb = len(re.findall('Traceback', solve))
res[f'solver tracebacks 0 (got {tb})'] = tb == 0
res['solver exit 0'] = 'solver exit 0' in txt or 'solver exit 0' in solve
if '--overlay' in sys.argv:
    res['qsa_topk shared (12 layers)'] = '[qsa_topk] 12 QSA layers share one top-k buffer' in server
    res['load_ec not refused'] = not any(('[load_ec]' in l and ('NOT installed' in l or 'not applied' in l) for l in server.splitlines()))
    res['hyb_fuse installed'] = '[hyb_fuse] HybridMoE' in server
    res['gdn_ba_overlap installed'] = '[gdn_ba_overlap] in_proj_ba forked' in server
    if '--q8' in sys.argv:
        res['int8 EP dispatch installed + used'] = '[q8_dispatch] FN_EP_Q8: int8' in server and 'first int8 EP dispatch' in server
        res['int8 reduce-scatter installed + used'] = 'int8 reduce-scatter installed' in server and 'first int8 reduce-scatter' in server
    ck = sys.argv[sys.argv.index('--chunk') + 1] if '--chunk' in sys.argv else '2176'
    res[f'hyb_fuse chunked path (M<=512, chunks of {ck})'] = f'chunks of {ck} tokens' in server
    res['DRAIN active (late measured cut)'] = '(drain:' in solve or 'drain: no new sessions' in solve
    if '--cache-export' in sys.argv:
        res['compile cache delta exported'] = 'compile cache delta exported' in txt or 'compile cache delta exported' in rd('../fn/solve.log')
    kv = [int(x.replace(',', '')) for x in re.findall('GPU KV cache size: ([\\d,]+) tokens', server)]
    res[f'KV per engine >= 930k (got {kv[:2]})'] = len(kv) >= 2 and min(kv[:2]) >= 930000
if '--tp8' in sys.argv:
    res['int8 reduce-scatter on the tp group'] = 'int8 reduce-scatter enabled for the tp group' in server
    res['int8 reduce-scatter never on the dp group'] = 'int8 reduce-scatter enabled for the dp group' not in server
cc = re.findall('compile cache: (\\d+) kernels compiled', txt)
if '--max-compiles' in sys.argv:
    mx = int(sys.argv[sys.argv.index('--max-compiles') + 1])
    res[f"in-run compiles <= {mx} (got {(cc[-1] if cc else 'n/a')})"] = bool(cc) and int(cc[-1]) <= mx
for k, v in res.items():
    print(('PASS ' if v else 'FAIL ') + k)
print('info: in-run compiles', cc[-1] if cc else 'n/a', '| final line:', (re.findall('final: [^\\n]*', solve) or ['n/a'])[-1][:160])
print('VERDICT:', 'ALL PASS' if all(res.values()) else 'FAIL')
sys.exit(0 if all(res.values()) else 1)
