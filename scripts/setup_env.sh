#!/usr/bin/env bash
# Create the conda env "sthgcn" and install PyTorch 2.x + PyG.
# Uses a CUDA wheel when nvidia-smi can see a GPU, otherwise a CPU wheel.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV_NAME="sthgcn"
export CONDA_PLUGINS_AUTO_ACCEPT_TOS="${CONDA_PLUGINS_AUTO_ACCEPT_TOS:-yes}"

ensure_conda() {
  if command -v conda >/dev/null 2>&1; then
    return
  fi
  local prefix
  for prefix in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "/opt/conda"; do
    if [[ -f "${prefix}/etc/profile.d/conda.sh" ]]; then
      # shellcheck disable=SC1091
      source "${prefix}/etc/profile.d/conda.sh"
      return
    fi
  done

  local installer="/tmp/miniconda.sh"
  local url="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
  case "$(uname -s)-$(uname -m)" in
    Linux-x86_64) url="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" ;;
    Linux-aarch64) url="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh" ;;
    Darwin-arm64) url="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh" ;;
    Darwin-x86_64) url="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh" ;;
    *)
      echo "Install Miniconda yourself, then re-run this script. Unsupported platform: $(uname -s)-$(uname -m)" >&2
      exit 1
      ;;
  esac
  echo "conda was not found. Installing Miniconda into ${HOME}/miniconda3"
  curl -fsSL -o "${installer}" "${url}"
  bash "${installer}" -b -p "${HOME}/miniconda3"
  # shellcheck disable=SC1091
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda init bash >/dev/null
}

ensure_conda
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

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
  python3 - "${ver}" <<'PY'
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

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Reusing conda env ${ENV_NAME}"
else
  echo "Creating conda env ${ENV_NAME} from environment.yml"
  conda env create -f environment.yml -y
fi
conda activate "${ENV_NAME}"
python -m pip install -U pip setuptools wheel

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

CONDA_BASE="$(conda info --base)"
echo
echo "Conda env ${ENV_NAME} is ready. Activate it with:"
echo "  source \"${CONDA_BASE}/etc/profile.d/conda.sh\""
echo "  conda activate ${ENV_NAME}"
