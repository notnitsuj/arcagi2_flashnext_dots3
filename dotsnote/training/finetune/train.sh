#!/bin/bash
# Fine-tune p1s: masked-CE SFT of lb144r on the selected set (expert GPTQ scales in log space, RMSNorms, router),
# 2 epochs, one sequence per step, layer-pipelined over the GPUs; then build_ckpt.sh writes a full checkpoint OUT/ckpt.
# Usage: train.sh OUT    env: GPUS (2,3,4), LR (1.6e-4, expert scales), LRN (5e-6, norms), LRR (5e-6, router),
#                             DATA ($D3_WORK/ft/sft), CKPT ($D3_LB144R)
source "$(dirname "$0")/../env.sh"
OUT=$1; GPUS=${GPUS:-2,3,4}
n=$(echo $GPUS | tr ',' '\n' | wc -l); idx=$(seq -s, 0 $((n - 1)))
DISTILL_NESTED_CKPT=1 DISTILL_NESTED_MIN=24576 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$GPUS \
  python3 "$TRAINING_DIR/finetune/train.py" --ckpt ${CKPT:-$D3_LB144R} --data ${DATA:-$D3_WORK/ft/sft} --val-tasks '' \
  --val-n 16 --max-val 24 --gpus $idx --len 40960 --epochs 2 --accum 1 --lr ${LR:-1.6e-4} --lr-norm ${LRN:-5e-6} \
  --lr-router ${LRR:-5e-6} --trainable scales,norms,router --eval-every 100 --seed 0 --out $OUT || exit 1
CKPT=${CKPT:-$D3_LB144R} bash "$TRAINING_DIR/finetune/build_ckpt.sh" $OUT $(echo $GPUS | cut -d, -f1)
echo TRAINDONE
