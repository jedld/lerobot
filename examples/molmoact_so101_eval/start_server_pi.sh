#!/usr/bin/env bash
# Start LeRobot Pi0/Pi0.5 /act inference for SO-ARM101.
#
# NVIDIA: uses the active LeRobot venv CUDA torch.
# AMD Strix Halo (Radeon 880M/890M): uses .venv-rocm from setup_amd_pi.sh.
#
# First-time AMD setup:
#   ./examples/molmoact_so101_eval/setup_amd_pi.sh
#
# Health check:
#   curl http://127.0.0.1:8102/act

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$SCRIPT_DIR"

# Prefer local helpers; fall back to molmoact2 if present.
if [[ -f "${SCRIPT_DIR}/rocm_env.sh" ]]; then
  # shellcheck source=rocm_env.sh
  source "${SCRIPT_DIR}/rocm_env.sh"
elif [[ -f "${MOLMOACT2_HOME:-$HOME/workspace/molmoact2}/examples/so101/rocm_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${MOLMOACT2_HOME:-$HOME/workspace/molmoact2}/examples/so101/rocm_env.sh"
fi

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="${TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

if declare -F export_rocm_ld_library_path >/dev/null 2>&1; then
  export_rocm_ld_library_path
fi

HOST="${PI_LOCAL_HOST:-127.0.0.1}"
PORT="${PI_LOCAL_PORT:-8102}"
CHECKPOINT="${PI_CHECKPOINT:-lerobot/pi05_base}"
POLICY_TYPE="${PI_POLICY_TYPE:-pi05}"
DEVICE="${PI_DEVICE:-auto}"
ROBOT_TYPE="${PI_ROBOT_TYPE:-}"
MAX_ACTIONS="${PI_MAX_ACTIONS:-}"
TOP_IMAGE_KEY="${PI_TOP_IMAGE_KEY:-}"
SIDE_IMAGE_KEY="${PI_SIDE_IMAGE_KEY:-}"
LOG_LEVEL="${EVAL_LOG_LEVEL:-INFO}"

if declare -F is_amd_strix >/dev/null 2>&1 && is_amd_strix; then
  # Fast defaults for the 890M unless the user already set them.
  PI_DTYPE="${PI_DTYPE:-bfloat16}"
  PI_NUM_INFERENCE_STEPS="${PI_NUM_INFERENCE_STEPS:-5}"
fi
DTYPE="${PI_DTYPE:-auto}"
NUM_INFERENCE_STEPS="${PI_NUM_INFERENCE_STEPS:-5}"

resolve_python() {
  if [[ -n "${PI_PYTHON:-}" && -x "${PI_PYTHON}" ]]; then
    echo "${PI_PYTHON}"
    return 0
  fi
  if [[ -x "${SCRIPT_DIR}/.venv-rocm/bin/python" ]]; then
    echo "${SCRIPT_DIR}/.venv-rocm/bin/python"
    return 0
  fi
  # On Strix Halo, refuse the CUDA-only LeRobot venv — it silently falls back to CPU.
  if declare -F is_amd_strix >/dev/null 2>&1 && is_amd_strix; then
    echo "error: AMD Strix Halo detected but ${SCRIPT_DIR}/.venv-rocm is missing." >&2
    echo "Run once:" >&2
    echo "  ${SCRIPT_DIR}/setup_amd_pi.sh" >&2
    echo "Or set PI_PYTHON to a ROCm-enabled interpreter." >&2
    exit 1
  fi
  if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    echo "${REPO_ROOT}/.venv/bin/python"
    return 0
  fi
  if command -v uv >/dev/null 2>&1; then
    echo "uv"
    return 0
  fi
  command -v python3
}

PYTHON_BIN="$(resolve_python)"

if [[ "${PYTHON_BIN}" != "uv" ]]; then
  PIP="${PYTHON_BIN%python}pip"
  if [[ -x "${PIP}" || "${PIP}" == pip ]]; then
    echo "[setup] Installing HTTP server deps (fastapi, uvicorn, json-numpy) ..."
    "${PIP}" install -q "fastapi<1.0" "uvicorn" "json-numpy" 2>/dev/null || \
      "${PYTHON_BIN}" -m pip install -q "fastapi<1.0" "uvicorn" "json-numpy"
  fi
fi

ARGS=(
  "${SCRIPT_DIR}/host_server_pi.py"
  --host "${HOST}"
  --port "${PORT}"
  --checkpoint "${CHECKPOINT}"
  --policy-type "${POLICY_TYPE}"
  --device "${DEVICE}"
  --dtype "${DTYPE}"
  --num-inference-steps "${NUM_INFERENCE_STEPS}"
  --log-level "${LOG_LEVEL}"
)

if [[ -n "${ROBOT_TYPE}" ]]; then
  ARGS+=(--robot-type "${ROBOT_TYPE}")
fi
if [[ -n "${MAX_ACTIONS}" ]]; then
  ARGS+=(--max-actions "${MAX_ACTIONS}")
fi
if [[ -n "${TOP_IMAGE_KEY}" ]]; then
  ARGS+=(--top-image-key "${TOP_IMAGE_KEY}")
fi
if [[ -n "${SIDE_IMAGE_KEY}" ]]; then
  ARGS+=(--side-image-key "${SIDE_IMAGE_KEY}")
fi
if [[ -n "${PI_TOKENIZER_NAME:-}" ]]; then
  ARGS+=(--tokenizer-name "${PI_TOKENIZER_NAME}")
fi

if declare -F is_amd_strix >/dev/null 2>&1 && is_amd_strix; then
  echo "AMD Strix iGPU detected — launching Pi with ROCm (${PYTHON_BIN}, dtype=${DTYPE}, steps=${NUM_INFERENCE_STEPS})."
fi

if [[ "${PYTHON_BIN}" == "uv" ]]; then
  exec uv run --extra pi python "${ARGS[@]}" "$@"
fi
exec "${PYTHON_BIN}" "${ARGS[@]}" "$@"
