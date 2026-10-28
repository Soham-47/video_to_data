#!/usr/bin/env bash
set +e
cd /home/ubuntu/workspace/nvidia-v2d/upstream/video_to_data/reconstruction
export PYTHONPATH=/home/ubuntu/workspace/nvidia-v2d/upstream/video_to_data/reconstruction/modules${PYTHONPATH:+:$PYTHONPATH}
/home/ubuntu/workspace/nvidia-v2d/runs/bootstrap/reconstruction-venv/bin/python modules/v2d_pipelines/run_ego_wilor.py --video_path /home/ubuntu/workspace/nvidia-v2d/data/raw/video_to_data_challenge/track_3/evaluation/videos/chunk-000/observation.images.ego_cam_a/episode_000003.mp4 --output_dir /home/ubuntu/workspace/nvidia-v2d/runs/track3-production/episode_000003_rebuild/white_pot --object_prompt "a white pot" --object_mesh_path /home/ubuntu/workspace/nvidia-v2d/runs/track3-production/episode_000001_rebuild/white_pot/mesh_scaled.obj --hand_pose_source hamer --run_slam --gravity_align > /home/ubuntu/workspace/nvidia-v2d/runs/track3-production/episode_000003_rebuild/white_pot/runner.log 2>&1
rc=$?
echo $rc > /home/ubuntu/workspace/nvidia-v2d/runs/track3-production/episode_000003_rebuild/white_pot/runner.exit
exit $rc
