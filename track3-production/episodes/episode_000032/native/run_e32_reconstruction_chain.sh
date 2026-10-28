#!/usr/bin/env bash
set -euo pipefail
source /home/ubuntu/workspace/nvidia-v2d/runs/track3-production/wait_for_rebuild_disk.sh
wait_for_rebuild_disk E32 /home/ubuntu/workspace/nvidia-v2d
N=/home/ubuntu/workspace/nvidia-v2d/runs/track3-production/episode_000032_rebuild/native
"$N/run_e32_blue_cup.sh" > "$N/reconstruction_blue_cup.log" 2>&1
"$N/run_e32_wooden_spoon.sh" > "$N/reconstruction_wooden_spoon.log" 2>&1
echo "E32_RECONSTRUCTIONS_COMPLETE $(date -Is)"
