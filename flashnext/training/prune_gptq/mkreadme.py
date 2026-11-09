import json, sys, glob, os, torch
d = sys.argv[1]
name = os.path.basename(d.rstrip('/'))
k = torch.load(f'{d}/keep.pt')
js = [json.load(open(f)) for f in sorted(glob.glob(f'{d}/gptq-L*.json'))]
bits = {}
for j in js:
    for b, n in j['bits'].items():
        bits[b] = bits.get(b, 0) + n
lb = sum((os.path.getsize(f) for f in glob.glob(f'{d}/lowbit-L*.pt')))
trunk = sum((os.path.getsize(f) for f in glob.glob(f'{d}/lowbit-L*.pt') if not f.endswith('L48.pt')))
imb = max((j.get('ep4_imbalance', 0) for j in js))
open(f'{d}/README.md', 'w').write(f'''# {name} (compression build, Qwen3.8-Flash-Next)\n- Experts: K={int(k['keep'][0].sum())} per layer (all 48 trunk layers + MTP layer L48), keep set {os.path.basename(k['src'])}\n  (reapsum ranking on Flash-Next FP8 routing over teacher-forced replays of its own ARC-AGI-2 *training*-task traces).\n- Quantization: GPTQ (act order, static g128 asymmetric grids with MSE clip from the FP8 originals, damp 1% + 4/n),\n  {('routing-weighted' if k['weighted'] else 'unweighted')} Hessians from the same capture; tiers "{k['tiers']}" over the kept experts\n  ranked by routing-weight sum; expert counts by bits over all 49 layers: {bits}; rtn={k['rtn']}.\n- Bytes: lowbit files {lb / 1000000000.0:.2f} GB total, trunk {trunk / 1000000000.0:.2f} GB (= {trunk / 4 / 1000000000.0:.2f} GB per EP4 rank streamed per step when all\n  kept experts are touched). Max EP4 linear-placement imbalance {imb:.3f}.\n- Files: lowbit-L{{l:02d}}.pt (planar low-bit packing: {{"experts": kept original ids ascending, id: {{bits, gu, gu_s, gu_z,\n  d, d_s, d_z}}}}, gu rows = [gate 640; up 640], w = fp16 s * (q - z), z int in [0, 2^b-1]); router-L{{l:02d}}.pt ({{"weight": bf16\n  [K, 2560] router rows in kept order, "keep"}}); keep.pt / keep.json (all 49 rows); gptq-L*.json (per-layer error stats);\n  ovr/ = the same codes dequantized to BF16 (evaluation serving with plugin/fnplug.py FN_OVR; not needed on L4).\n- Non-expert tensors: Qwen/Qwen3.8-Flash-Next-FP8 unchanged. Routing: softmax over the K kept logits, top-10, renormalize.\n- Evaluation serving: serve/serve_build.sh {d} GPUS PORT (stock vLLM 0.30 + plugin/fnplug.py).\n''')
print(open(f'{d}/README.md').read())
