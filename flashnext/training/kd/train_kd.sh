#!/bin/bash
# Distillation training: kd/train_kd.py on KD_NREP local replicas of KD_GPR GPUs, then export (KD_EXPORT=dir).
# env: KD_OUT, KD_ARGS (train_kd.py args), KD_NREP (8), KD_GPR (1), KD_BUILD ($FN_WORK/builds/k320q34), KD_EXPORT,
#      DENSE ($FN_WORK/dense); several machines: KD_NNODES, KD_NODE_RANK, MASTER_ADDR.
source "$(dirname "$0")/../env.sh"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True KD_PAD=${KD_PAD:-4096} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
NREP=${KD_NREP:-8}; GPR=${KD_GPR:-1}; BLD=${KD_BUILD:-$FN_WORK/builds/k320q34}; DENSE=${DENSE:-$FN_WORK/dense}
mkdir -p $KD_OUT/logs "$LOGDIR"
python3 "$TRAINING_DIR/student/ple_stage.py" || exit 1
NP=${KD_NNODES:-1}; PI=${KD_NODE_RANK:-0}; MA=${MASTER_ADDR:-127.0.0.1}
echo "train_kd: node $PI/$NP master=$MA nrep=$NREP gpr=$GPR out=$KD_OUT args=$KD_ARGS"
pids=()
for r in $(seq 0 $((NREP - 1))); do
  G=$(seq -s, $((r * GPR)) $((r * GPR + GPR - 1)))
  LG=$(seq -s, 0 $((GPR - 1)))
  RK=$((PI * NREP + r))
  CUDA_VISIBLE_DEVICES=$G WORLD_SIZE=$((NP * NREP)) RANK=$RK MASTER_ADDR=$MA MASTER_PORT=${MASTER_PORT:-29511} \
    python3 "$TRAINING_DIR/kd/train_kd.py" --gpus $LG --out $KD_OUT --model $DENSE --build $BLD $KD_ARGS \
    > $KD_OUT/logs/rank$RK.log 2>&1 &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait $p || rc=1; done
echo "train done rc=$rc"
if [ $rc = 0 ] && [ -n "$KD_EXPORT" ] && [ $PI = 0 ]; then
  python3 "$TRAINING_DIR/student/export_build.py" $KD_OUT/trained.pt $BLD $KD_EXPORT > $KD_OUT/export.log 2>&1
  echo "export done rc=$?"
fi
