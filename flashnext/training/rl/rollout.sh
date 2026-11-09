#!/bin/bash
# REPL rollouts with the shipped harness (rl/rollout.py over harness/pool.py, QR_ARCKIT=block) against running servers.
# Usage: RL_EPS=host:port,... RL_OUT=dir RL_ARGS="--prompt ak --split DIR --tasks @FILE --arms repl:8 ..." \
#        bash training/rl/rollout.sh
source "$(dirname "$0")/../env.sh"
export QR_ARCKIT=${QR_ARCKIT:-block}
mkdir -p "$RL_OUT"
echo "rollout: eps=$RL_EPS out=$RL_OUT args=$RL_ARGS"
for e in ${RL_EPS//,/ }; do
  for t in $(seq 1 360); do curl -sf -m 5 $e/v1/models >/dev/null && break; sleep 10; done
  curl -sf -m 5 $e/v1/models >/dev/null || echo "endpoint $e not up (continuing without it is NOT supported)"
done
echo "endpoints up"
python3 "$TRAINING_DIR/rl/rollout.py" --endpoints $RL_EPS --model-dir $FN_BASE_MODEL --out $RL_OUT $RL_ARGS
echo "rollout done rc=$?"
