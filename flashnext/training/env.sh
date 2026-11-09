# Shared defaults for the training scripts (sourced by every script; override any variable in the environment).
# Relative paths resolve against the directory you run from; the README runs everything from the flashnext/ folder.
TRAINING_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export FN_BASE_MODEL=${FN_BASE_MODEL:-models/Qwen3.8-Flash-Next-FP8}
export FN_WORK=${FN_WORK:-work}
export ARC_DATA=${ARC_DATA:-data/ARC-AGI-2/data}
export LOGDIR=${LOGDIR:-$FN_WORK/logs}
mkdir -p "$FN_WORK" "$LOGDIR"
