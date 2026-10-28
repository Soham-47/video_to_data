#!/usr/bin/env bash
set -euo pipefail
source /home/ubuntu/workspace/nvidia-v2d/runs/track3-production/wait_for_rebuild_disk.sh
wait_for_rebuild_disk E20 /home/ubuntu/workspace/nvidia-v2d
N=/home/ubuntu/workspace/nvidia-v2d/runs/track3-production/episode_000020_rebuild/native
"$N/run_e20_blue_cup.sh" > "$N/reconstruction_blue_cup.log" 2>&1
"$N/run_e20_water_pitcher.sh" > "$N/reconstruction_water_pitcher.log" 2>&1
echo "E20_RECONSTRUCTIONS_COMPLETE $(date -Is)"
