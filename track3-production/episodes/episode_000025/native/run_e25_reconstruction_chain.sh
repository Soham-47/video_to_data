#!/usr/bin/env bash
set -euo pipefail
source /home/ubuntu/workspace/nvidia-v2d/runs/track3-production/wait_for_rebuild_disk.sh
wait_for_rebuild_disk E25 /home/ubuntu/workspace/nvidia-v2d
N=/home/ubuntu/workspace/nvidia-v2d/runs/track3-production/episode_000025_rebuild/native
"$N/run_e25_wooden_piece_1.sh" > "$N/reconstruction_wooden_piece_1.log" 2>&1
"$N/run_e25_wooden_piece_2.sh" > "$N/reconstruction_wooden_piece_2.log" 2>&1
echo "E25_RECONSTRUCTIONS_COMPLETE $(date -Is)"
