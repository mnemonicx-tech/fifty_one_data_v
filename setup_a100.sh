#!/usr/bin/env bash
set -Eeuo pipefail

# A100 setup script for the FiftyOne + COCO S3 pipeline
# Usage:
#   bash setup_a100.sh
#   bash setup_a100.sh --python python3.10 --venv .venv-a100 --torch-cuda cu121

PYTHON_BIN="python3"
VENV_DIR=".venv"
TORCH_CUDA="cu121"
SKIP_TORCH=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --venv)
      VENV_DIR="$2"
      shift 2
      ;;
    --torch-cuda)
      TORCH_CUDA="$2"
      shift 2
      ;;
    --skip-torch)
      SKIP_TORCH=1
      shift
      ;;
    -h|--help)
      echo "Usage: bash setup_a100.sh [--python python3.10] [--venv .venv] [--torch-cuda cu121] [--skip-torch]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Error: $PYTHON_BIN not found"
  exit 1
fi

echo "==> Python binary: $PYTHON_BIN"
"$PYTHON_BIN" --version

echo "==> Creating virtual environment: $VENV_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

echo "==> Upgrading packaging tools"
python -m pip install --upgrade pip setuptools wheel

if [[ "$SKIP_TORCH" -eq 0 ]]; then
  echo "==> Installing PyTorch with CUDA wheel index: $TORCH_CUDA"
  python -m pip install --upgrade \
    --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" \
    torch torchvision torchaudio
else
  echo "==> Skipping PyTorch installation (--skip-torch)"
fi

echo "==> Installing pipeline dependencies"
python -m pip install --upgrade \
  boto3 \
  botocore \
  fiftyone \
  ijson \
  numpy \
  pycocotools \
  tqdm

echo "==> Running environment checks"
python - <<'PY'
import importlib

packages = [
    "boto3",
    "botocore",
    "fiftyone",
    "ijson",
    "numpy",
    "pycocotools",
    "tqdm",
]

for pkg in packages:
    importlib.import_module(pkg)

print("Python package imports: OK")

try:
    import torch
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU count:", torch.cuda.device_count())
        print("GPU name:", torch.cuda.get_device_name(0))
except Exception as e:
    print("Torch check skipped/failed:", e)
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "==> NVIDIA GPU check"
  nvidia-smi
else
  echo "Warning: nvidia-smi not found. Install NVIDIA drivers/CUDA on the server first."
fi

echo ""
echo "Setup complete."
echo "Activate environment with: source ${VENV_DIR}/bin/activate"
echo "Run pipeline with: python pipeline.py --sample-size 800 --mode random --workers 4 --output-dir ./subset_gpu"
