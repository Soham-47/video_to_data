#!/usr/bin/env bash
set +e
cd "/home/ubuntu/workspace/nvidia-v2d/upstream/video_to_data/reconstruction"
export PYTHONPATH="/home/ubuntu/workspace/nvidia-v2d/upstream/video_to_data/reconstruction/modules${PYTHONPATH:+:$PYTHONPATH}"
"/home/ubuntu/workspace/nvidia-v2d/runs/bootstrap/reconstruction-venv/bin/python" "/home/ubuntu/workspace/nvidia-v2d/upstream/video_to_data/reconstruction/modules/v2d_pipelines/run_ego_wilor.py" --video_path "/home/ubuntu/workspace/nvidia-v2d/data/raw/video_to_data_challenge/track_3/evaluation/videos/chunk-000/observation.images.ego_cam_a/episode_000001.mp4" --output_dir "/home/ubuntu/workspace/nvidia-v2d/runs/track3-production/episode_000001_rebuild/white_pot_lid" --object_prompt "a white pot lid" --hand_pose_source hamer --run_slam --gravity_align > "/home/ubuntu/workspace/nvidia-v2d/runs/track3-production/episode_000001_rebuild/white_pot_lid/runner.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "/home/ubuntu/workspace/nvidia-v2d/runs/track3-production/episode_000001_rebuild/white_pot_lid/runner.exit"
exit "$rc"
