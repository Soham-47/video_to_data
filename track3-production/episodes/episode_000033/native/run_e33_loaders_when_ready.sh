#!/usr/bin/env bash
set -euo pipefail
BASE=/home/ubuntu/workspace/nvidia-v2d
ROOT=$BASE/runs/track3-production/episode_000033_rebuild
N=$ROOT/native
RECON_PID=$(cat "$N/reconstruction_chain.pid")
READY=0
for ((i=0; i<3600; i++)); do
  if grep -q 'E33_RECONSTRUCTIONS_COMPLETE' "$N/reconstruction_chain.log"; then READY=1; break; fi
  state=$(ps -o stat= -p "$RECON_PID" 2>/dev/null || true)
  if [[ -z "$state" || "$state" == Z* ]]; then echo "RECONSTRUCTION_FAILED state=$state"; tail -n 100 "$N/reconstruction_chain.log"; exit 1; fi
  sleep 10
done
if [[ "$READY" != 1 ]]; then echo RECONSTRUCTION_TIMEOUT; exit 1; fi
for object in blue_cup wooden_spoon; do
  rm -rf "$N/input_$object/episode_000033/outputs/result"
  mkdir -p "$N/input_$object/episode_000033/outputs"
  cp -a "$ROOT/$object/result_slam_gravity_aligned" "$N/input_$object/episode_000033/outputs/result"
  if [[ "$object" != "wooden_spoon" ]]; then
    "/tmp/v2d-submission-venv/bin/python" "$BASE/runs/track3-production/build_video_mesh_asset.py" "$ROOT/$object/mesh_scaled.obj" "$object" "$N"
  fi
done
"/tmp/v2d-submission-venv/bin/python" "$N/fix_e33_manifest_flags.py" > "$N/manifest_corrections.log" 2>&1
"$N/run_e33_loaders.sh"
echo E33_LOADERS_AND_MANIFESTS_COMPLETE
