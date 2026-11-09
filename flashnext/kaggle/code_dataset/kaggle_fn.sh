#!/bin/bash
set -u
T0=${T0:-$(date +%s)}; DT=${DT:-42150}; KILLBY=${KILLBY:-42600}
PY=${PY:-/tmp/fn_py312/bin/python3.12}; SITE=${SITE:-$(dirname $(dirname $PY))/lib/python3.12/site-packages}
CODE=${CODE:?code dir}; MODEL=${MODEL:-/tmp/fn/ckpt}; EXPERTS=${EXPERTS:-/tmp/fn/experts}
CHAL=${CHAL:-/kaggle/input/arc-prize-2026/arc-agi_test_challenges.json}; OUT=${OUT:-/kaggle/working/submission.json}
WORK=${WORK:-/kaggle/working/fn}; mkdir -p $WORK
log() { echo "[fn $(( $(date +%s) - T0 ))s] $*"; }
BIN=$(dirname $PY)
export PATH=$BIN:$PATH LD_LIBRARY_PATH=$SITE/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
OV=/tmp/fnov
FN_APPLY_ONLY="base kv mtp ngram moe session" FN_PY=$PY bash $CODE/fnext/apply.sh $OV $SITE > $WORK/apply.log 2>&1 \
  || { log "apply.sh failed"; tail -20 $WORK/apply.log; exit 1; }
export PYTHONPATH=$OV${PYTHONPATH:+:$PYTHONPATH}
KEEPF=$EXPERTS/keep.json; MTPF=$EXPERTS/mtp_ovr.safetensors
if [ -n "${EXPERTS_DELTA:-}" ]; then
  export FN_MOE_DELTA=$EXPERTS_DELTA
  for d in $(echo $EXPERTS_DELTA | tr ':' ' '); do
    [ -f $d/keep.json ] && KEEPF=$d/keep.json; [ -f $d/mtp_ovr.safetensors ] && MTPF=$d/mtp_ovr.safetensors
  done
  log "experts: base $EXPERTS + deltas $EXPERTS_DELTA applied at load"
fi
if [ -n "${FN_CACHE_TAR:-}" ] && [ -f "$FN_CACHE_TAR" ]; then
  VC=${FN_CACHE_DIR:-/tmp/fn/vcache}; mkdir -p $VC
  if tar -C $VC -xf "$FN_CACHE_TAR" && $PY $CODE/fnext/boot/fcache.py fixgrp $VC/triton > /dev/null; then
    export VLLM_CACHE_ROOT=$VC TRITON_CACHE_DIR=$VC/triton
    log "compile cache restored: $(find $VC -type f | wc -l) files"
    touch $VC/.restored   # marker: kernels newer than this were really compiled in this run (vLLM's "JIT compilation
  else log "compile cache restore failed (JIT as usual)"; fi
fi
NEXP=$($PY -c "import json;k=json.load(open('$KEEPF'));print(len(k['0']))")
log "overlay ready; experts K=$NEXP"
[ -f "$MTPF" ] && export FN_MTP_OVR=$MTPF && log "MTP layer override: $FN_MTP_OVR"
log "KV cache skip layer 48 (BF16 MTP KV): ${SKIP48:-1}"
export FN_MOE_LOWBIT=$EXPERTS VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128 FN_MOE_PAIRS=1 FN_MOE_MARLIN4=1 FN_NGRAM_SHARED=1 \
  FN_EMBED_HOST=1 FN_NGRAM_MEMFD=1 FN_SESSION_EP=1 FN_SESSION_EVICT=1 FN_HOST_AR=1 FN_MTP_HEAD=int8 \
  FN_MTP_HEAD_VOCAB=$CODE/hot32k_tr.pt FN_DENSE_FP8=1 NCCL_P2P_DISABLE=1 FN_HOST_DP_EP=1 FN_HOST_SP=1 $([ "${KV_PACK:-0}" = 1 ] && echo FN_KV_PACK_STATE=1 FN_KV_PACK_RING=1) $([ "${FP8MM:-0}" = 1 ] && echo FN_FP8MM_TRITON=1) $([ "${L4CFG:-0}" = 1 ] && echo FN_MOE_L4CFG=$CODE/k2_l4cfg.json) $([ "${SILU_LOCAL:-0}" = 1 ] && echo FN_MOE_SILU_LOCAL=1) $([ "${SAMPLE_SHARD:-0}" = 1 ] && echo FN_SAMPLE_SHARD=1) FN_HOST_DP=1 FN_GDN_LAZY=1 FN_MOE_MMAP=1 FN_MTP_DRAFT_TOPK=20 FN_DETOK_ROPE=1
SERVE=(vllm serve $MODEL --served-model-name m --host 127.0.0.1 --port 8000
  --tensor-parallel-size 2 --data-parallel-size 2 --enable-expert-parallel --api-server-count 1
  --language-model-only --kv-cache-dtype fp8 --attention-config.indexer_kv_dtype fp8 --mamba-ssm-cache-dtype bfloat16
  --max-num-seqs 24 --max-model-len 122880 --gpu-memory-utilization 0.92 --max-num-batched-tokens ${BT:-2048}
  --no-enable-prefix-caching
  --speculative-config '{"method":"mtp","num_speculative_tokens":3, "draft_sample_method": "probabilistic", "rejection_sample_method": "block"}'
  --hf-overrides "{\"text_config\": {\"num_experts\": $NEXP}}"
  --use-replayssm --mamba-backend triton --max-cudagraph-capture-size 128 -cc '{"cudagraph_mode":"FULL_DECODE_ONLY"}' $([ "${SAMPLE_SHARD:-0}" = 1 ] && echo --enable-batch-sharded-sampling) $([ "${SKIP48:-1}" = 1 ] && echo --kv-cache-dtype-skip-layers 48))
(n=0
 while true; do
   left=$(( T0 + DT - $(date +%s) ))
   [ $left -lt 1800 ] && { echo "[supervisor] not restarting: ${left}s left"; break; }
   [ $n -gt 0 ] && echo "[supervisor] restart $n at $(( $(date +%s) - T0 ))s"
   "${SERVE[@]}"
   echo "[supervisor] server exited rc=$? at $(( $(date +%s) - T0 ))s"
   n=$((n + 1)); [ $n -gt 3 ] && break
   sleep 20
 done) > $WORK/server_log.txt 2>&1 &
SUP=$!
($PY $CODE/fnext/boot/fcache.py stat launcher; for i in $(seq 1 1800); do grep -q "Loading weights took" $WORK/server_log.txt 2>/dev/null && break; sleep 1; done; $PY $CODE/fnext/boot/fcache.py stat weights_loaded) >> $WORK/fcache_stat.log 2>&1 &
if [ "${LATE_PF:-0}" = 1 ]; then $PY $CODE/fnext/boot/late_prefetch.py --log $WORK/server_log.txt --count 4 --drop-glob "$MODEL/ngram-*.safetensors" $EXPERTS ${EXPERTS_DELTA:+${EXPERTS_DELTA//:/ }} > $WORK/late_pf.log 2>&1 & fi
if [ "${FN_DROP_NGRAM:-0}" = 1 ]; then
  (for i in $(seq 1 1800); do
     [ $(grep -c "\[fnmoe\] loader: dropped" $WORK/server_log.txt 2>/dev/null) -ge 4 ] && break; sleep 1
   done
   $PY $CODE/fnext/boot/fcache.py drop $(ls -d $MODEL/ngram-*.safetensors 2>/dev/null)) > $WORK/fcache.log 2>&1 &
fi
SRV=http://127.0.0.1:8000
until curl -sf $SRV/health > /dev/null; do
  sleep ${POLL_S:-15}
  if ! kill -0 $SUP 2>/dev/null; then
    curl -sf $SRV/health > /dev/null && break
    log "SERVER FAILED: supervisor gave up"; tail -50 $WORK/server_log.txt; exit 3
  fi
  [ $(( $(date +%s) - T0 )) -gt $(( DT - 900 )) ] && { log "server never healthy"; tail -50 $WORK/server_log.txt; break; }
done
log "server healthy"
$PY $CODE/fnext/boot/fcache.py stat healthy >> $WORK/fcache_stat.log 2>&1 &
[ "${FN_DROP_AFTER_BOOT:-0}" = 1 ] && $PY $CODE/fnext/boot/fcache.py drop $MODEL $EXPERTS ${EXPERTS_DELTA:+${EXPERTS_DELTA//:/ }} >> $WORK/fcache.log 2>&1 &
ENDG=${ENDG:-600}
# DRAIN=1: the solver cuts late (--drain-fill: measured guard, at most ENDG before the deadline) and creates
# $WORK/endgame.flag at the cut; the DP keepalive starts on that flag (fallback: DRAIN_KA_S before the deadline)
[ "${DRAIN:-0}" = 1 ] && rm -f $WORK/endgame.flag
(if [ "${DRAIN:-0}" = 1 ]; then
   while [ ! -f $WORK/endgame.flag ] && [ $(date +%s) -lt $(( T0 + DT - ${DRAIN_KA_S:-240} )) ]; do sleep 2; done
 else sleep $(( T0 + DT - ENDG - 180 - $(date +%s) )) 2>/dev/null; fi
 for r in 0 1; do
   (while [ $(date +%s) -lt $(( T0 + KILLBY )) ]; do
      curl -s -m 900 -o /dev/null -H "X-data-parallel-rank: $r" -H "Content-Type: application/json" $SRV/v1/completions \
        -d '{"model": "m", "prompt": [1, 2, 3], "max_tokens": 4096, "temperature": 0, "ignore_eos": true}' || sleep 5
    done) &
 done
 wait) > $WORK/keepalive.log 2>&1 &
LEFT=$(( T0 + DT - $(date +%s) ))
log "solver deadline ${LEFT}s"
timeout $(( T0 + KILLBY - $(date +%s) )) $PY $CODE/repl_solve.py --backend fnext --host 127.0.0.1 --port 8000 \
  --challenges $CHAL --tokenizer $MODEL --out $OUT --log $WORK/solve.log --samples-out $WORK/repl_samples.jsonl \
  --deadline-s $LEFT \
  --concurrency ${CONC:-44} --sample-cap 98304 --turn-tokens 49152 --max-tool-calls 24 --exec-timeout 20 --force-tokens 1536 \
  --force-skip-verified --policy keep --agree 99 --kmax 99 --task-cap-per-input 100000000 --max-model-len 122880 \
  --ctx-typ 36000 --ctx-frac 1.0 --kv-admit 0.92 --kv-agg ${KV_AGG:-max} --import-fix ${IMPORT_FIX:-off} --tb-fix ${TB_FIX:-off} --answer-order ${ANSWER_ORDER:-off} --rate0 800 --session-chunk 2048 --max-inflight-per-task 2 \
  --min-cap ${MINCAP:-65536} --budget-margin-s ${MARGIN:-900} --endgame-s $ENDG $([ "${DRAIN:-0}" = 1 ] && echo --drain-fill on --endgame-flag $WORK/endgame.flag) --arckit block --settle-last on --att2-fill on --tie-break flags --carry on --tail-fill on --start-stops on --tok-cache on
log "solver exit $?"
[ -f "${VC:-/nonexistent}/.restored" ] && log "compile cache: $(find $VC/triton -name '*.cubin' -newer $VC/.restored | wc -l) kernels compiled during the run (cache misses): $(find $VC/triton -name '*.cubin' -newer $VC/.restored -printf '%f ' | head -c 600)"
if [ "${CACHE_EXPORT:-0}" = 1 ] && [ -f "${VC:-/nonexistent}/.restored" ]; then (cd $VC && find . -type f -newer .restored ! -name .restored | tar -cf $WORK/cache_delta.tar -T -) 2>/dev/null; log "compile cache delta exported: $(tar -tf $WORK/cache_delta.tar 2>/dev/null | wc -l) files"; fi
$PY - $CHAL $OUT <<'P'
import json, os, sys
ch = json.load(open(sys.argv[1]))
sub = json.load(open(sys.argv[2])) if os.path.exists(sys.argv[2]) else {}
for t, v in ch.items():
    if t not in sub or len(sub[t]) != len(v["test"]):
        sub[t] = [{"attempt_1": x["input"], "attempt_2": x["input"]} for x in v["test"]]
json.dump(sub, open(sys.argv[2] + ".tmp", "w")); os.replace(sys.argv[2] + ".tmp", sys.argv[2])
print("submission tasks", len(sub))
P
kill $SUP 2>/dev/null; pkill -f "vllm serve" 2>/dev/null
