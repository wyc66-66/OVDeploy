#!/usr/bin/env bash
# Clone/install microsoft/GLIP for native GLIP-T (.pth) inference.
set -euo pipefail
ROOT=/mnt/d/ccfa/submission-a
cd "$ROOT"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate yoloworld5070

GLIP_DIR="$ROOT/third_party/GLIP"
if [ ! -f "$GLIP_DIR/setup.py" ]; then
  mkdir -p "$ROOT/third_party"
  if ! GIT_TERMINAL_PROMPT=0 git clone --depth 1 https://github.com/microsoft/GLIP.git "$GLIP_DIR" 2>/dev/null; then
    echo "git clone failed; download GLIP-main.zip on Windows into third_party/GLIP"
    exit 1
  fi
fi

pip install -q einops shapely timm yacs tensorboardX ftfy prettytable pymongo inflect
cd "$GLIP_DIR"
python setup.py build develop --user 2>&1 | tail -30

echo "GLIP repo ready at $GLIP_DIR"
python -c "import maskrcnn_benchmark; print('maskrcnn_benchmark OK')"
