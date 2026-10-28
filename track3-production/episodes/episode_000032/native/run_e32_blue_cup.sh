#!/usr/bin/env bash
set -euo pipefail
BASE=/home/ubuntu/workspace/nvidia-v2d
ROOT=$BASE/runs/track3-production/episode_000032_rebuild
RECON=$BASE/upstream/video_to_data/reconstruction
VIDEO=$BASE/data/raw/video_to_data_challenge/track_3/evaluation/videos/chunk-000/observation.images.ego_cam_a/episode_000032.mp4
cd "$RECON"
export V2D_SKIP_OVERLAYS=1
export PYTHONPATH=/home/ubuntu/workspace/nvidia-v2d/upstream/video_to_data/reconstruction/modules
exec /home/ubuntu/workspace/nvidia-v2d/runs/bootstrap/reconstruction-venv/bin/python \
  /home/ubuntu/workspace/nvidia-v2d/upstream/video_to_data/reconstruction/modules/v2d_pipelines/run_ego_reconstruction.py \
  --video "$VIDEO" \
  --object_prompt "a blue cup" \
  --output_dir "$ROOT/blue_cup" \
  --hand_tracking hamer \
  --sam2_hand_prompt_source box \
  --run_droid_slam \
  --run_gravity_alignment \
  --gravity_align_target z_up \
  --geocalib_weights pinhole \
  --geocalib_camera_model pinhole \
  --geocalib_num_samples 8
