#!/usr/bin/env bash
# Train with posttrain using configs/posttrain_from_custom.example.json
#
# --- Run under PM2 (keeps process alive if SSH disconnects; central logs) ---
#
# From the repo root (custom-ballspotting/):
#
#   chmod +x scripts/run-posttrain-pm2.sh
#   pm2 start scripts/run-posttrain-pm2.sh --name ballspot-posttrain --no-autorestart
#
# Follow logs:
#   pm2 logs ballspot-posttrain
#
# After training exits successfully, PM2 will leave the process "stopped" (--no-autorestart
# avoids spinning the job again). Remove when done:
#   pm2 delete ballspot-posttrain
#
# Optional: persist PM2's process list across reboots (usually not needed for one-off training):
#   pm2 save
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${REPO_ROOT}/configs/posttrain_from_custom.example.json"
export PYTHONUNBUFFERED=1

if [[ -x "${REPO_ROOT}/.venv/bin/custom-ballspotting" ]]; then
  exec "${REPO_ROOT}/.venv/bin/custom-ballspotting" posttrain --config "${CONFIG}"
elif command -v custom-ballspotting >/dev/null 2>&1; then
  exec custom-ballspotting posttrain --config "${CONFIG}"
else
  echo "custom-ballspotting not found. Activate .venv or: pip install -e ." >&2
  exit 1
fi
