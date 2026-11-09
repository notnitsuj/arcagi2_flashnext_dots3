# Shared defaults for the Dots3-Note training scripts (sourced by every shell script; override any variable in the
# environment). Relative paths resolve against the directory you run from; the README runs everything from dotsnote/.
TRAINING_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export D3_BASE_MODEL=${D3_BASE_MODEL:-models/dots3-note-prev-fp8}
export D3_WORK=${D3_WORK:-work}
export ARC_DATA=${ARC_DATA:-data/ARC-AGI-2/data}
export D3_LB144R=${D3_LB144R:-$D3_WORK/lb144r}
export D3_ENGINE=${D3_ENGINE:-$TRAINING_DIR/../engine}
export LOGDIR=${LOGDIR:-$D3_WORK/logs}
mkdir -p "$D3_WORK" "$LOGDIR"
