#!/usr/bin/env bash
set -euo pipefail
N=/home/ubuntu/workspace/nvidia-v2d/runs/track3-production/episode_000019_rebuild/native
"$N/run_e19_mini_sweeper.sh" > "$N/reconstruction_mini_sweeper.log" 2>&1
"$N/run_e19_mini_dust_pan.sh" > "$N/reconstruction_mini_dust_pan.log" 2>&1
echo "E19_RECONSTRUCTIONS_COMPLETE $(date -Is)"
