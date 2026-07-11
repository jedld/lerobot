#!/usr/bin/env bash
# Bootstrap a Python 3.12 ROCm venv for Pi0/Pi0.5 inference on AMD Strix Halo.
#
# Prerequisite (once, same as MolmoAct2):
#   sudo ~/workspace/molmoact2/examples/so101/install_rocm_system.sh
#
# Usage:
#   ./examples/molmoact_so101_eval/setup_amd_pi.sh
#   ./examples/molmoact_so101_eval/run_eval.sh --policy pi --local

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$SCRIPT_DIR"

# shellcheck source=rocm_env.sh
source "${SCRIPT_DIR}/rocm_env.sh"

PYTHON_MINOR="${PYTHON_MINOR:-3.12}"
VENV_DIR="${VENV_DIR:-${SCRIPT_DIR}/.venv-rocm}"
ROCM_REL="${ROCM_REL:-7.2.1}"
WHEEL_BASE="https://repo.radeon.com/rocm/manylinux/rocm-rel-${ROCM_REL}"
MOLMOACT2_HOME="${MOLMOACT2_HOME:-${HOME}/workspace/molmoact2}"
MOLMO_SITE="${MOLMOACT2_HOME}/.venv/lib/python${PYTHON_MINOR}/site-packages"

export_rocm_ld_library_path
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="${TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

if ! command -v rocm-smi >/dev/null 2>&1; then
  echo "error: rocm-smi not found. Install the AMDGPU/ROCm stack first." >&2
  echo "  sudo ${MOLMOACT2_HOME}/examples/so101/install_rocm_system.sh" >&2
  exit 1
fi

if ! groups | tr ' ' '\n' | grep -qxE 'video|render'; then
  echo "warning: user is not in the video/render group; GPU access may fail." >&2
  echo "         sudo usermod -aG render,video \"\$USER\" && re-login" >&2
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is required. Install from https://astral.sh/uv" >&2
  exit 1
fi

echo "Installing CPython ${PYTHON_MINOR}..."
uv python install "${PYTHON_MINOR}"

if [[ -d "$VENV_DIR" && -x "$VENV_DIR/bin/python" ]]; then
  echo "Using existing venv at $VENV_DIR"
else
  echo "Creating venv at $VENV_DIR"
  uv venv --python "${PYTHON_MINOR}" --seed "$VENV_DIR"
fi

PYTHON="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"
DST_SITE="$VENV_DIR/lib/python${PYTHON_MINOR}/site-packages"

copy_rocm_torch_from_molmoact2() {
  [[ -d "$MOLMO_SITE/torch" ]] || return 1
  echo "Reusing ROCm torch from ${MOLMOACT2_HOME}/.venv ..."
  local pattern path base
  for pattern in \
    torch torch-*.dist-info torchvision torchvision-*.dist-info \
    triton triton-*.dist-info pytorch_triton_rocm* functorch torchgen \
    filelock filelock-*.dist-info sympy sympy-*.dist-info mpmath mpmath-*.dist-info \
    networkx networkx-*.dist-info jinja2 jinja2-*.dist-info markupsafe markupsafe-*.dist-info \
    typing_extensions.py typing_extensions-*.dist-info fsspec fsspec-*.dist-info
  do
    for path in ${MOLMO_SITE}/${pattern}; do
      [[ -e "$path" ]] || continue
      base="$(basename "$path")"
      rm -rf "${DST_SITE}/${base}"
      cp -a "$path" "${DST_SITE}/${base}"
    done
  done
}

need_torch=1
if "$PYTHON" - <<'PY' 2>/dev/null
import torch
assert getattr(torch.version, "hip", None)
assert torch.cuda.is_available()
print(torch.__version__, torch.cuda.get_device_name(0))
PY
then
  echo "ROCm PyTorch already usable in $VENV_DIR"
  need_torch=0
fi

if [[ "$need_torch" -eq 1 ]]; then
  if copy_rocm_torch_from_molmoact2 && "$PYTHON" - <<'PY' 2>/dev/null
import torch
assert getattr(torch.version, "hip", None)
assert torch.cuda.is_available()
print(torch.__version__, torch.cuda.get_device_name(0))
PY
  then
    echo "ROCm torch copy OK"
    need_torch=0
  fi
fi

if [[ "$need_torch" -eq 1 ]]; then
  PY_TAG="cp${PYTHON_MINOR/./}"
  ARCH="linux_x86_64"
  TMPDIR="$(mktemp -d)"
  trap 'rm -rf "$TMPDIR"' EXIT
  TORCH_WHL="$TMPDIR/torch-2.9.1+rocm${ROCM_REL}.lw.gitff65f5bc-${PY_TAG}-${PY_TAG}-${ARCH}.whl"
  VISION_WHL="$TMPDIR/torchvision-0.24.0+rocm${ROCM_REL}.gitb919bd0c-${PY_TAG}-${PY_TAG}-${ARCH}.whl"
  TRITON_WHL="$TMPDIR/triton-3.5.1+rocm${ROCM_REL}.gita272dfa8-${PY_TAG}-${PY_TAG}-${ARCH}.whl"

  echo "Fetching ROCm PyTorch wheels (rocm-rel-${ROCM_REL})..."
  wget -q -O "$TORCH_WHL" \
    "${WHEEL_BASE}/torch-2.9.1%2Brocm${ROCM_REL}.lw.gitff65f5bc-${PY_TAG}-${PY_TAG}-${ARCH}.whl"
  wget -q -O "$VISION_WHL" \
    "${WHEEL_BASE}/torchvision-0.24.0%2Brocm${ROCM_REL}.gitb919bd0c-${PY_TAG}-${PY_TAG}-${ARCH}.whl"
  wget -q -O "$TRITON_WHL" \
    "${WHEEL_BASE}/triton-3.5.1%2Brocm${ROCM_REL}.gita272dfa8-${PY_TAG}-${PY_TAG}-${ARCH}.whl"

  echo "Installing ROCm PyTorch (AMD repo.radeon wheels for gfx1150)..."
  "$PIP" install --upgrade pip wheel
  "$PIP" install --force-reinstall "$TORCH_WHL" "$VISION_WHL" "$TRITON_WHL"
fi

echo "Installing LeRobot + Pi inference dependencies (keeping ROCm torch)..."
"$PIP" install --upgrade pip wheel
"$PIP" install \
  "typing-extensions" \
  "draccus==0.10.0" \
  "einops>=0.8.0,<0.9.0" \
  "huggingface-hub>=1.0,<2" \
  "hf-transfer>=0.1.8" \
  "accelerate>=1.10.0" \
  "safetensors>=0.4" \
  "transformers==5.5.4" \
  "tokenizers" \
  "sentencepiece>=0.2" \
  "protobuf>=4.25" \
  "pillow>=10" \
  "numpy>=2.0,<2.3" \
  "opencv-python-headless>=4.9,<4.14" \
  "requests>=2.32" \
  "packaging>=24.2,<26" \
  "setuptools>=71,<81" \
  "cmake>=3.29,<4.2" \
  "pyyaml" \
  "tqdm" \
  "regex" \
  "scipy" \
  "diffusers" \
  "pyserial" \
  "deepdiff" \
  "jsonlines" \
  "termcolor" \
  "av>=15,<16" \
  "rerun-sdk>=0.23" \
  "gymnasium>=1.0" \
  "wandb" \
  "datasets" \
  "imageio" \
  "imageio-ffmpeg" \
  "cloudpickle" \
  "json-numpy" \
  "fastapi<1.0" \
  "uvicorn" \
  "pydantic" \
  "rich" \
  "typer"

# Editable install without deps so CUDA torch from PyPI cannot overwrite ROCm.
"$PIP" install --no-deps -e "${REPO_ROOT}"

echo "Verifying ROCm GPU access..."
"$PYTHON" - <<'PY'
import os
import torch

from lerobot.policies.factory import make_pre_post_processors  # noqa: F401
from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # noqa: F401

print("torch", torch.__version__)
print("hip", getattr(torch.version, "hip", None))
print("LD_LIBRARY_PATH", os.environ.get("LD_LIBRARY_PATH", ""))
print("cuda available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit(
        "PyTorch has no GPU backend. Check ROCm install, /dev/kfd access, "
        "and video/render group membership."
    )
print("device", torch.cuda.get_device_name(0))
x = torch.randn(4, device="cuda")
print("tensor ok", x.shape, x.device)
print("PI05Policy import OK")
PY

echo
echo "AMD Pi setup complete."
echo "  Venv:   $VENV_DIR"
echo "  Start:  ./examples/molmoact_so101_eval/run_eval.sh --policy pi --local"
echo "  Or:     ./examples/molmoact_so101_eval/start_server_pi.sh"
