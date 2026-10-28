#!/usr/bin/env bash
set -euo pipefail
BASE=/home/ubuntu/workspace/nvidia-v2d
N=$BASE/runs/track3-production/episode_000020_rebuild/native
RECON_PID=$(cat "$N/reconstruction_chain.pid")
READY=0
for ((i=0; i<3600; i++)); do
  if grep -q 'E20_RECONSTRUCTIONS_COMPLETE' "$N/reconstruction_chain.log"; then READY=1; break; fi
  state=$(ps -o stat= -p "$RECON_PID" 2>/dev/null || true)
  if [[ -z "$state" || "$state" == Z* ]]; then echo "RECONSTRUCTION_FAILED state=$state"; tail -n 100 "$N/reconstruction_chain.log"; exit 1; fi
  sleep 10
done
if [[ "$READY" != 1 ]]; then echo RECONSTRUCTION_TIMEOUT; exit 1; fi
"$N/run_e20_loaders.sh"
echo E20_LOADERS_AND_MANIFESTS_COMPLETE
