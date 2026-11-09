#!/bin/bash
# Fine-tune data: lb144r (TP2, evaluation numerics) answers the 720 generation tasks of finetune/split.json in direct
# mode, on their real test inputs, at T 0.7 with a 32k-token cap: hard x4, easy x2, unk x2, then 2 more samples for every
# unk task that was not solved 2/2. Run finetune/split.py first.
# Output: $D3_WORK/ft/gen_{hard,easy,unk,unk2}.jsonl (+ _text.jsonl). Starts and stops its own server.
# env: GPUS (0,1), PORT (8495), CKPT ($D3_LB144R), L4_GIB (66)
source "$(dirname "$0")/../env.sh"
GPUS=${GPUS:-0,1}; PORT=${PORT:-8495}; CK=${CKPT:-$D3_LB144R}
FT=$(realpath -m "$D3_WORK/ft"); L=$LOGDIR/ft_gen.srv.log
(setsid env L4_GIB=${L4_GIB:-66} KLOC=2048 MTP=3 EMULATE_GPU_GIB= bash "$TRAINING_DIR/serve/serve_eval.sh" $GPUS $PORT \
  $CK --seed 5151 > $L 2>&1 &)
sleep 10; n=0; until grep -a -q -E 'Application startup complete|Traceback|core.py:1366' $L || [ $n -gt 200 ]; do sleep 15; n=$((n+1)); done
grep -a -q 'Application startup complete' $L || { echo "gen: server failed"; bash "$TRAINING_DIR/serve/kill_server.sh" $PORT $GPUS; exit 1; }
gen() {
  timeout 36000 python3 "$TRAINING_DIR/harness/direct.py" --ports $PORT --split $FT/gen_$1 --attempts $2 \
    --max_tokens 32768 --max_model_len 65536 --concurrency $3 --out $FT/gen_$1.jsonl \
    --save_text $FT/gen_$1_text.jsonl > $LOGDIR/ft_gen_$1.log 2>&1
  echo "gen $1: rc=$? rows=$(grep -c '"task"' $FT/gen_$1.jsonl 2>/dev/null)"
}
gen hard 4 40 & gen easy 2 12 & gen unk 2 12 & wait
mkdir -p $FT/gen_unk2
python3 - "$FT" <<'EOF'
import json, os, sys, collections
ft = sys.argv[1]
c = collections.defaultdict(list)
for l in open(f"{ft}/gen_unk.jsonl"):
    r = json.loads(l); c[r["task"]].append(bool(r.get("correct")))
for t, v in c.items():
    if not (len(v) >= 2 and all(v)):
        d = f"{ft}/gen_unk2/{t}.json"
        if not os.path.exists(d): os.link(f"{ft}/gen_unk/{t}.json", d)
print("gen unk2 tasks:", len(os.listdir(f"{ft}/gen_unk2")))
EOF
gen unk2 2 64
bash "$TRAINING_DIR/serve/kill_server.sh" $PORT $GPUS
echo GENDONE
