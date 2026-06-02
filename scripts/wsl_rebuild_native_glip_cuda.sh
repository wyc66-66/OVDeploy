#!/usr/bin/env bash
# Rebuild microsoft/GLIP maskrcnn_benchmark with CUDA ops (WSL).
set -eo pipefail
ROOT=/mnt/d/ccfa/论文2
cd "$ROOT"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate yoloworld5070

# Install nvcc if missing (PyTorch cu128 bundle)
if ! command -v nvcc >/dev/null 2>&1; then
  echo "Installing cuda-nvcc via conda..."
  conda install -y -c nvidia cuda-nvcc cuda-toolkit || true
fi

export CUDA_HOME="${CUDA_HOME:-$(python scripts/_check_cuda_home.py 2>/dev/null | awk '/CUDA_HOME/ {print $2}')}"
if [ -z "${CUDA_HOME}" ] || [ "${CUDA_HOME}" = "None" ]; then
  for d in \
    "$CONDA_PREFIX" \
    "/usr/local/cuda" \
    "/usr/lib/wsl/lib"; do
    if [ -x "$d/bin/nvcc" ] 2>/dev/null || [ -d "$d/include" ]; then
      export CUDA_HOME="$d"
      break
    fi
  done
fi

export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"

# CUDA 12.8 requires g++ < 14; prefer conda gxx 12 if present
if [ -x "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++" ]; then
  export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
  export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
fi
echo "CXX=$CXX"
echo "CUDA_HOME=$CUDA_HOME"
python scripts/_check_cuda_home.py || true
which nvcc || true
nvcc --version 2>/dev/null || echo "WARN: nvcc not on PATH"

GLIP_DIR="$ROOT/third_party/GLIP"
pip install -q einops shapely timm yacs tensorboardX ftfy prettytable pymongo inflect
python "$ROOT/scripts/patch_glip_pytorch2.py"
cd "$GLIP_DIR"
rm -rf build maskrcnn_benchmark.egg-info
# Force CUDA build even if setup.py heuristics fail
FORCE_CUDA=1 python setup.py build develop --user 2>&1 | tee "$ROOT/reports/wsl_native_glip_cuda_build.log" | tail -60

echo "=== post-build CUDA op test ==="
python "$ROOT/scripts/_test_cuda_op.py"
