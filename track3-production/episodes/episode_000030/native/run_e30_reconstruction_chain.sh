#!/usr/bin/env bash
set -euo pipefail
source /home/ubuntu/workspace/nvidia-v2d/runs/track3-production/wait_for_rebuild_disk.sh
wait_for_rebuild_disk E30 /home/ubuntu/workspace/nvidia-v2d
N=/home/ubuntu/workspace/nvidia-v2d/runs/track3-production/episode_000030_rebuild/native
"$N/run_e30_mini_sweeper.sh" > "$N/reconstruction_mini_sweeper.log" 2>&1
"$N/run_e30_plastic_dish_rack.sh" > "$N/reconstruction_plastic_dish_rack.log" 2>&1
echo "E30_RECONSTRUCTIONS_COMPLETE $(date -Is)"
