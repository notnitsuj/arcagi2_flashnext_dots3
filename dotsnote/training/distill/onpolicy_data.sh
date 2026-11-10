#!/bin/bash
# On-policy distillation data: a student checkpoint answers the training tasks (one direct-mode attempt per test input,
# evaluation numerics with MTP-3 under a 26 GiB per-GPU budget, TP over GPUS), then the FP8 preview, served without
# DeepGEMM, scores those rollouts: top-20 prompt log-probabilities of prompt + completion, truncated to MAXTOK + 8192.
# Usage: onpolicy_data.sh CKPT TAG [MAXTOK 12288]
# env: GPUS (0,1,2,3), TASKS ($D3_WORK/kd/tasks_all.txt), PORT (8486, student server), TPORT (8487, teacher server)
# Output: $D3_WORK/kd/onp_TAG.jsonl (+ _text.jsonl), $D3_WORK/kd/teacher_TAG/*.npz; logs in $LOGDIR
source "$(dirname "$0")/../env.sh"
CK=$1; TAG=$2; MT=${3:-12288}; GPUS=${GPUS:-0,1,2,3}; TASKS=${TASKS:-$D3_WORK/kd/tasks_all.txt}
PORT=${PORT:-8486}; TPORT=${TPORT:-8487}; K=$D3_WORK/kd
n=$(echo $GPUS | tr ',' '\n' | wc -l)
mkdir -p $K
L=$LOGDIR/kd_onp_$TAG.srv.log
if [ ! -s $K/onp_${TAG}_text.jsonl ]; then
  (setsid env L4_GIB=26 EMULATE_GPU_GIB= MTP=3 bash "$TRAINING_DIR/serve/serve_eval.sh" $GPUS $PORT $CK > $L 2>&1 &)
  sleep 10; i=0; until grep -a -q -E 'Application startup complete|Traceback|core.py:1366' $L || [ $i -gt 200 ]; do sleep 15; i=$((i+1)); done
  grep -a -q 'Application startup complete' $L || { echo "student server failed"; bash "$TRAINING_DIR/serve/kill_server.sh" $PORT $GPUS; exit 1; }
  timeout 14400 python3 "$TRAINING_DIR/harness/direct.py" --ports $PORT --split training --tasks @$TASKS --attempts 1 \
    --max_tokens $MT --max_model_len 262144 --concurrency 64 --out $K/onp_$TAG.jsonl --save_text $K/onp_${TAG}_text.jsonl \
    > $LOGDIR/kd_onp_gen_$TAG.log 2>&1
  bash "$TRAINING_DIR/serve/kill_server.sh" $PORT $GPUS; sleep 10
fi
L=$LOGDIR/kd_onp_teacher_$TAG.srv.log
(setsid env VLLM_USE_DEEP_GEMM=0 ARC_DOTS3_PORTABLE_DSA=1 UTIL=0.97 bash "$TRAINING_DIR/serve/serve.sh" $GPUS $TPORT \
  $D3_BASE_MODEL --tensor-parallel-size $n --language-model-only --trust-remote-code --disable-custom-all-reduce \
  --kv-cache-dtype fp8 --max-num-batched-tokens 1024 --max-num-seqs 8 --max-model-len 172032 \
  --attention-config '{"flash_attn_version": 2}' \
  --compilation-config '{"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[4,8,16,32]}' > $L 2>&1 &)
sleep 10; i=0; until grep -a -q -E 'Application startup complete|Traceback' $L || [ $i -gt 200 ]; do sleep 15; i=$((i+1)); done
grep -a -q 'Application startup complete' $L || { echo "teacher server failed"; bash "$TRAINING_DIR/serve/kill_server.sh" $TPORT $GPUS; exit 1; }
timeout 7200 python3 "$TRAINING_DIR/distill/teacher_dump.py" $TPORT $K/onp_${TAG}_text.jsonl $K/teacher_$TAG \
  --len $((MT + 8192)) --conc 4 --tag onp > $LOGDIR/kd_onp_dump_$TAG.log 2>&1
bash "$TRAINING_DIR/serve/kill_server.sh" $TPORT $GPUS
echo "onp $TAG done: $(ls $K/teacher_$TAG | wc -l) seqs"
