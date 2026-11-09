import json, sys
out, direct_text, synth = sys.argv[1:4]
with open(out, 'w') as f:
    for l in open(direct_text):
        r = json.loads(l)
        f.write(json.dumps({'full': {'prompt': r['prompt'], 'completion': r['completion']}}) + '\n')
    n = 0
    for l in open(synth):
        r = json.loads(l)
        if r.get('full'):
            f.write(json.dumps({'full': r['full']}) + '\n')
            n += 1
print('wrote', out, sum((1 for _ in open(out))), 'traces,', n, 'of them program-synthesis traces')
