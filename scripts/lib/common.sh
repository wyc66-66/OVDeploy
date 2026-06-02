#!/usr/bin/env bash
# OVDeploy shared WSL paths (论文2 project, 论文1 YOLO-World).
set -euo pipefail

export PROJ="${PROJ:-/mnt/d/ccfa/论文2}"
export YOLO="${YOLO:-/mnt/d/ccfa/论文1/YOLO-World}"
if [[ ! -d "$PROJ" ]]; then
  export PROJ="/mnt/c/Users/34186/yolo_eval/论文2"
fi
if [[ ! -d "$YOLO" ]]; then
  export YOLO="/mnt/d/ccfa/论文1/YOLO-World"
fi
export ENV_NAME="${ENV_NAME:-yoloworld5070}"
export LOG_DIR="${LOG_DIR:-/tmp/ovdeploy_logs}"
export HF_HOME="${HF_HOME:-/mnt/c/Users/34186/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

mkdir -p "$LOG_DIR" "$HF_HOME"

activate_conda() {
  if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
  elif [[ -n "${CONDA_EXE:-}" ]]; then
    # shellcheck disable=SC1091
    source "$(dirname "$CONDA_EXE")/../etc/profile.d/conda.sh"
  else
    echo "ERROR: conda not found." >&2
    exit 1
  fi
  conda activate "$ENV_NAME"
  export PIP_USER=0
  export PYTHONNOUSERSITE=1
  export PYTHON="$(which python)"
}

write_8gb_config() {
  local cfg="$YOLO/configs/pretrain/yolo_world_v2_s_lvis_minival_8gb.py"
  cat > "$cfg" <<'EOF'
_base_ = './yolo_world_v2_s_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py'
val_dataloader = dict(batch_size=1, num_workers=2)
test_dataloader = val_dataloader
EOF
}

ensure_lf_scripts() {
  find "$PROJ/scripts" -name '*.sh' -type f -exec sed -i 's/\r$//' {} +
}
