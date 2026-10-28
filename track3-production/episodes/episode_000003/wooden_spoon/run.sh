#!/usr/bin/env bash
set +e
cd /home/ubuntu/workspace/nvidia-v2d/upstream/video_to_data/reconstruction
export PYTHONPATH=/home/ubuntu/workspace/nvidia-v2d/upstream/video_to_data/reconstruction/modules${PYTHONPATH:+:$PYTHONPATH}
export V2D_SKIP_OVERLAYS=1
/home/ubuntu/workspace/nvidia-v2d/runs/bootstrap/reconstruction-venv/bin/python /home/ubuntu/workspace/nvidia-v2d/upstream/video_to_data/reconstruction/modules/v2d_pipelines/run_ego_wilor.py --video_path /home/ubuntu/workspace/nvidia-v2d/data/raw/video_to_data_challenge/track_3/evaluation/videos/chunk-000/observation.images.ego_cam_a/episode_000003.mp4 --output_dir /home/ubuntu/workspace/nvidia-v2d/runs/track3-production/episode_000003_rebuild/wooden_spoon --object_prompt "a wooden spoon" --object_mesh_path /home/ubuntu/workspace/nvidia-v2d/runs/track3-production/episode_000015/reconstruction/wooden_spoon/mesh_scaled.obj --hand_pose_source hamer --run_slam --gravity_align > /home/ubuntu/workspace/nvidia-v2d/runs/track3-production/episode_000003_rebuild/wooden_spoon/runner.log 2>&1
rc=$?
printf "%s\n" "$rc" > /home/ubuntu/workspace/nvidia-v2d/runs/track3-production/episode_000003_rebuild/wooden_spoon/runner.exit
exit "$rc"
