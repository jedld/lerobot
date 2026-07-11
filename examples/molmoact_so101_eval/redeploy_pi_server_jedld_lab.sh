#!/usr/bin/env bash
# Run this script ON jedld-lab (RTX 3090), not on the robot host.
#
# Replaces the wrong pi05 "raw 32-dim state" server with the SO-101 server
# from this repo (expects state shape (6,), returns actions (N, 6)).
#
# Usage on jedld-lab:
#   cd /path/to/lerobot
#   git pull   # or copy host_server_pi.py + start_server_pi.sh from the robot repo
#   ./examples/molmoact_so101_eval/redeploy_pi_server_jedld_lab.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

export PI_LOCAL_HOST=0.0.0.0
export PI_LOCAL_PORT=8102
export PI_CHECKPOINT="${PI_CHECKPOINT:-lerobot/pi05_base}"
export PI_POLICY_TYPE="${PI_POLICY_TYPE:-pi05}"
export PI_DEVICE=cuda
export PI_DTYPE=bfloat16
export PI_NUM_INFERENCE_STEPS=5

echo "Stopping any existing Pi server on port ${PI_LOCAL_PORT}..."
if command -v fuser >/dev/null 2>&1; then
  fuser -k "${PI_LOCAL_PORT}/tcp" 2>/dev/null || true
elif command -v lsof >/dev/null 2>&1; then
  pid="$(lsof -ti ":${PI_LOCAL_PORT}" 2>/dev/null || true)"
  [[ -n "${pid}" ]] && kill "${pid}" 2>/dev/null || true
fi
sleep 1

echo "Health check BEFORE redeploy (expect state_dim=32 if wrong build is running):"
curl -s "http://127.0.0.1:${PI_LOCAL_PORT}/act" 2>/dev/null | python3 -m json.tool || echo "(no server yet)"

echo
echo "Starting SO-101 Pi server from ${REPO_ROOT} ..."
exec "${SCRIPT_DIR}/start_server_pi.sh"
