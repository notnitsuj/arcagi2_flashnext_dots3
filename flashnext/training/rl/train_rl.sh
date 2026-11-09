#!/bin/bash
# RL training: rl/train_rl.py data-parallel on this machine (RL_NREP replicas x RL_GPR GPUs each), then (RL_EXPORT=dir)
# export of the trained scales and router rows on top of the original build with student/export_build.py.
# env: RL_OUT, RL_ARGS (train_rl.py args), RL_NREP (8), RL_GPR (1), RL_BUILD ($FN_WORK/builds/k320q34), RL_EXPORT,
#      DENSE ($FN_WORK/dense, from student/make_dense.py); several machines: RL_NNODES, RL_NODE_RANK, MASTER_ADDR.
source "$(dirname "$0")/../env.sh"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True KD_PAD=${KD_PAD:-4096} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
NREP=${RL_NREP:-8}; GPR=${RL_GPR:-1}; BLD=${RL_BUILD:-$FN_WORK/builds/k320q34}; DENSE=${DENSE:-$FN_WORK/dense}
mkdir -p $RL_OUT/logs "$LOGDIR"
python3 "$TRAINING_DIR/student/ple_stage.py" || exit 1    # n-gram table -> /dev/shm/ple (once per machine)
NP=${RL_NNODES:-1}; PI=${RL_NODE_RANK:-0}; MA=${MASTER_ADDR:-127.0.0.1}
echo "train_rl: node $PI/$NP master=$MA nrep=$NREP gpr=$GPR out=$RL_OUT args=$RL_ARGS"
pids=()
for r in $(seq 0 $((NREP - 1))); do
  G=$(seq -s, $((r * GPR)) $((r * GPR + GPR - 1)))
  LG=$(seq -s, 0 $((GPR - 1)))
  RK=$((PI * NREP + r))
  CUDA_VISIBLE_DEVICES=$G WORLD_SIZE=$((NP * NREP)) RANK=$RK MASTER_ADDR=$MA MASTER_PORT=${MASTER_PORT:-29533} \
    python3 "$TRAINING_DIR/rl/train_rl.py" --gpus $LG --out $RL_OUT --model $DENSE --build $BLD $RL_ARGS \
    > $RL_OUT/logs/rank$RK.log 2>&1 &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait $p || rc=1; done
echo "train done rc=$rc"
if [ $rc = 0 ] && [ -n "$RL_EXPORT" ] && [ $PI = 0 ]; then
  python3 "$TRAINING_DIR/student/export_build.py" $RL_OUT/trained.pt $BLD $RL_EXPORT > $RL_OUT/export.log 2>&1
  echo "export done rc=$? -> $RL_EXPORT"
fi
