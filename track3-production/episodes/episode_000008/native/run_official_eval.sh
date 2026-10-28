#!/usr/bin/env bash
set -euo pipefail
BASE=/home/ubuntu/workspace/nvidia-v2d
ROOT=$BASE/runs/track3-production/episode_000008_rebuild
docker run --rm --gpus all \
  --mount type=bind,src="$ROOT/native",dst=/workspace/native \
  --mount type=bind,src="$ROOT/native/object_assets",dst=/workspace/object_assets,readonly \
  --mount type=bind,src="$BASE/runs/track3-production",dst=/v2d/runs/track3-production,readonly \
  -e XLA_PYTHON_CLIENT_PREALLOCATE=false -e MUJOCO_GL=egl \
  --entrypoint bash flash_chord:8c778f9 -lc '
    set -euo pipefail
    cd /workspace
    python -u scripts/evaluate_policy.py \
      "evaluation.checkpoint=/v2d/runs/track3-production/episode_000008_rebuild/native/checkpoint_njmax4096/policy_65536.safetensors" \
      evaluation.world_count=1 \
      evaluation.start_frame=0 \
      evaluation.reset_mode=explicit \
      evaluation.motion_start_frame=0 \
      evaluation.motion_end_frame=-1 \
      evaluation.metrics_output=/workspace/native/official_results_mounted/episode_000008.json \
      evaluation.object_trajectories_output=/workspace/native/official_results_mounted/episode_000008.parquet \
      logging.mode=disabled \
      hydra.run.dir=/tmp/flash_chord_evaluate_episode_000008
  '
