#!/usr/bin/env bash
set -euo pipefail
BASE=/home/ubuntu/workspace/nvidia-v2d
N=$BASE/runs/track3-production/episode_000030_rebuild/native
RECON=$BASE/upstream/video_to_data/reconstruction
PYTHON=$BASE/runs/bootstrap/reconstruction-venv/bin/python
export PYTHONPATH=$RECON/modules
cd "$RECON"
for object in mini_sweeper plastic_dish_rack; do
  rm -rf "$N/loaded_$object"
  processed="$N/human_motion_data/ego_recon/processed"
  rm -f "$processed/$object.obj" "$processed/$object.mtl" "$processed/${object}_rigid.urdf" "$processed/${object}_visual.stl" "$processed/${object}_material_"*.png
  "$PYTHON" "$RECON/modules/v2d_task_library_loader/docker/run_loader.py" \
    --dataset ego_recon \
    --output_dir "$N/loaded_$object" \
    --mano_model_dir "$N/human_motion_data/mano" \
    --human_motion_data_dir "$N/human_motion_data" \
    --dataset_root "$N/input_$object/episode_000030" \
    --object_name "$object" \
    --sequence_name episode_000030 \
    --no_ground_align --save > "$N/loader_$object.log" 2>&1
done
echo "E30_LOADERS_COMPLETE $(date -Is)"
