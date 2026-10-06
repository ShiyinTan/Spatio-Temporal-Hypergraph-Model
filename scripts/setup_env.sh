#!/usr/bin/env bash
# Create .venv and install a PyTorch 2.x + PyG stack.
# Uses a CUDA wheel when nvidia-smi can see a GPU, otherwise a CPU wheel.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required" >&2
  exit 1
fi

python3 - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit(
        f"Python 3.10+ is required (found {sys.version.split()[0]}). "
        "Create a newer environment, then re-run this script."
    )
print(f"Using {sys.version.split()[0]}")
PY

if [[ ! -x .venv/bin/python ]]; then
  rm -rf .venv
  if ! python3 -m venv .venv; then
    echo "Could not create .venv. On Ubuntu install python3-venv, then re-run this script:" >&2
    echo "  sudo apt-get install -y python3-venv" >&2
    exit 1
  fi
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip setuptools wheel

# Newest PyTorch release that still has official torch-scatter / torch-sparse wheels.
# A newer torch (2.13+) currently has no matching PyG extension wheel.
choose_stack() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "cpu 2.12.1"
    return
  fi
  local ver
  ver="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1 || true)"
  if [[ -z "${ver}" ]]; then
    echo "cpu 2.12.1"
    return
  fi
  python - "${ver}" <<'PY'
import sys
major, minor = (int(part) for part in sys.argv[1].split(".")[:2])
# nvidia-smi reports the newest CUDA runtime this driver can load.
# Each pair is a published PyTorch build that also has PyG wheels.
if (major, minor) >= (13, 0):
    print("cu130 2.12.1")
elif (major, minor) >= (12, 6):
    print("cu126 2.12.1")
elif (major, minor) >= (12, 4):
    print("cu124 2.6.0")
elif (major, minor) >= (11, 8):
    print("cu118 2.7.1")
else:
    print("cpu 2.12.1")
PY
}

read -r CUDA_INDEX TORCH_VERSION < <(choose_stack)
echo "Installing PyTorch ${TORCH_VERSION} from the ${CUDA_INDEX} index"
python -m pip install "torch==${TORCH_VERSION}" --index-url "https://download.pytorch.org/whl/${CUDA_INDEX}"

read -r TORCH_TAG CUDA_TAG <<EOF
$(python - <<'PY'
import torch
parts = torch.__version__.split("+")[0].split(".")
torch_tag = f"{parts[0]}.{parts[1]}.0"
cuda = torch.version.cuda
cuda_tag = "cpu" if not cuda else "cu" + cuda.replace(".", "")
print(torch_tag, cuda_tag)
PY
)
EOF

echo "Installing torch-scatter and torch-sparse for torch-${TORCH_TAG}+${CUDA_TAG}"
python -m pip install torch_scatter torch_sparse \
  -f "https://data.pyg.org/whl/torch-${TORCH_TAG}+${CUDA_TAG}.html"
python -m pip install -r requirements.txt

python - <<'PY'
import torch
import torch_geometric
import torch_scatter
import torch_sparse
print(f"torch {torch.__version__}")
print(f"torch_geometric {torch_geometric.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu: {torch.cuda.get_device_name(0)}")
else:
    print("gpu: none (CPU build)")
print(f"torch_scatter {torch_scatter.__version__}")
print(f"torch_sparse {torch_sparse.__version__}")
PY

echo
echo "Environment is ready. Activate it with: source .venv/bin/activate"
