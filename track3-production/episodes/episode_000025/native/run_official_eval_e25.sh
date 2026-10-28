#!/usr/bin/env bash
set -euo pipefail
test -s /workspace/native/checkpoint_njmax4096/policy_65536.safetensors
cd /workspace
python -u scripts/evaluate_policy.py \
  evaluation.checkpoint=/v2d/runs/track3-production/episode_000025_rebuild/native/checkpoint_njmax4096/policy_65536.safetensors \
  evaluation.world_count=1 \
  evaluation.start_frame=0 \
  evaluation.reset_mode=explicit \
  evaluation.motion_start_frame=0 \
  evaluation.motion_end_frame=-1 \
  evaluation.metrics_output=/workspace/native/official_results_mounted/episode_000025.json \
  evaluation.object_trajectories_output=/workspace/native/official_results_mounted/episode_000025.parquet \
  logging.mode=disabled \
  hydra.run.dir=/tmp/flash_chord_evaluate_episode_000025
