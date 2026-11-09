#!/usr/bin/env python3
import argparse, json, os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
WORDS = {2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six'}
USAGE = '# Model Format\n\nHugging Face style directory: config.json, tokenizer files, model-0000*.safetensors (dense weights) and lowbit-L01.pt .. lowbit-L45.pt (packed low-bit expert weights, loaded by a custom vLLM model plugin).\n\n# License\n\nApache License 2.0, as the base Dots3-Note weights (LICENSE included).'
DESCRIPTION = '# Model Summary\n\nDots3-Note (the dots3-note preview by Dots Studio) with its mixture-of-experts layers pruned to 144 of 256 routed experts per layer, the experts ranked on traces of ARC-AGI-2 training tasks, and the kept expert weights GPTQ-quantised to 3 and 2 bits. Dense and attention weights are unchanged (FP8 safetensors). Expert weights are stored per layer in lowbit-L01.pt to lowbit-L45.pt next to config.json (arc_lowbit = "."), and pruning.json lists the kept experts of every layer.\n\n# Variations\n\n- {first} to {last}: the checkpoint ({n_files} files, {gib:.1f} GiB) split into {n_words} parts of about 15 GB. Attach all {n_words} and place their files in one directory.\n\n# License\n\nDerived from the dots3-note preview weights, which are licensed under the Apache License 2.0 (LICENSE in every part). Changes: experts pruned and quantised as described above.\n'

def main():
    ap = argparse.ArgumentParser(description='Split the lb144r checkpoint folder into the Kaggle Model variations of model_parts.json: OUT/<variation>/ holds hard links to its files (LICENSE in every part) and model-instance-metadata.json; OUT/model-metadata.json describes the Model. Prints the kaggle commands that create them.')
    ap.add_argument('ckpt', help='checkpoint folder (the 63 files of model_parts.json)')
    ap.add_argument('out', help='output folder')
    ap.add_argument('--parts', default=str(HERE / 'model_parts.json'), help='file list, sizes and part layout')
    ap.add_argument('--owner', help="Kaggle owner (default: the manifest's)")
    ap.add_argument('--slug', help="Model slug (default: the manifest's)")
    ap.add_argument('--title', default='Dots3-Note lb144r')
    ap.add_argument('--subtitle', default='Expert-pruned, low-bit quantised Dots3-Note checkpoint for 4xL4 inference')
    ap.add_argument('--public', action='store_true', help='create the Model as public (default: private)')
    ap.add_argument('--symlink', action='store_true', help='symlinks instead of hard links (checkpoint on another filesystem)')
    ap.add_argument('--no-size-check', action='store_true', help='accept files whose sizes differ from the manifest')
    a = ap.parse_args()
    man = json.load(open(a.parts))
    owner, slug = (a.owner or man['owner'], a.slug or man['slug'])
    ck, out = (Path(a.ckpt), Path(a.out))
    missing = [n for n in man['files'] if not (ck / n).is_file()]
    assert not missing, f'{ck}: missing {missing}'
    wrong = [n for n, s in man['files'].items() if (ck / n).stat().st_size != s]
    assert not wrong or a.no_size_check, f'{ck}: sizes differ from the manifest for {wrong} (--no-size-check to accept)'
    extra = sorted((p.name for p in ck.iterdir() if p.is_file() and p.name not in man['files']))
    if extra:
        print(f'not part of the checkpoint (left out): {extra}')
    parts = man['parts']
    assert set().union(*map(set, parts.values())) == set(man['files']), 'the parts do not cover the checkpoint'
    assert all(('LICENSE' in names for names in parts.values())), 'every part needs the LICENSE'
    out.mkdir(parents=True, exist_ok=True)
    n = len(parts)
    for k, (variation, names) in enumerate(parts.items(), 1):
        d = out / variation
        d.mkdir(exist_ok=True)
        for name in names:
            dst = d / name
            if not dst.exists():
                if a.symlink:
                    os.symlink((ck / name).resolve(), dst)
                else:
                    os.link(ck / name, dst)
        meta = {'ownerSlug': owner, 'modelSlug': slug, 'instanceSlug': variation, 'framework': man['framework'], 'overview': f'Dots3-Note lb144r checkpoint, part {k} of {n}. Attach all {WORDS.get(n, n)} parts and place their files in one directory.', 'usage': USAGE, 'licenseName': 'Apache 2.0', 'fineTunable': False, 'trainingData': [], 'modelInstanceType': 'Unspecified', 'baseModelInstance': '', 'externalBaseModelUrl': ''}
        json.dump(meta, open(d / 'model-instance-metadata.json', 'w'), indent=2)
        size = sum(((ck / name).stat().st_size for name in names))
        print(f'  {variation:10s} {len(names):3d} files {size / 1000000000.0:6.2f} GB  {d}')
    first, last = (list(parts)[0], list(parts)[-1])
    gib = sum(man['files'].values()) / 2 ** 30
    json.dump({'ownerSlug': owner, 'slug': slug, 'title': a.title, 'subtitle': a.subtitle, 'isPrivate': not a.public, 'description': DESCRIPTION.format(first=first, last=last, n_files=len(man['files']), gib=gib, n_words=WORDS.get(n, n)), 'publishTime': '', 'provenanceSources': ''}, open(out / 'model-metadata.json', 'w'), indent=2)
    print(f'\nkaggle models create -p {out}')
    for variation in parts:
        print(f'kaggle models instances create -p {out / variation}')
    for variation in parts:
        print(f"kaggle models instances versions files {owner}/{slug}/{man['framework']}/{variation}/{man['version']}")
if __name__ == '__main__':
    main()
