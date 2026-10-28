#!/usr/bin/env bash
set -euo pipefail
BASE=/home/ubuntu/workspace/nvidia-v2d
ROOT=$BASE/runs/track3-production/episode_000020_rebuild
N=$ROOT/native
RECON=$BASE/upstream/video_to_data/reconstruction
PYTHON=$BASE/runs/bootstrap/reconstruction-venv/bin/python
export PYTHONPATH=$RECON/modules
cd "$RECON"
for object in blue_cup water_pitcher; do
  rm -rf "$N/loaded_$object"
  processed="$N/human_motion_data/ego_recon/processed"
  rm -f "$processed/$object.obj" "$processed/$object.mtl" "$processed/${object}_rigid.urdf" "$processed/${object}_visual.stl" "$processed/${object}_material_"*.png
  output="$N/input_$object/episode_000020/outputs"
  rm -rf "$output/result"
  mkdir -p "$output"
  cp -a "$ROOT/$object/result_slam_gravity_aligned" "$output/result"
  "/tmp/v2d-submission-venv/bin/python" "$BASE/runs/track3-production/build_video_mesh_asset.py" "$ROOT/$object/mesh_scaled.obj" "$object" "$N"
done
"$PYTHON" "$N/fix_e20_manifest_flags.py"
for object in blue_cup water_pitcher; do
  "$PYTHON" "$RECON/modules/v2d_task_library_loader/docker/run_loader.py" \
    --dataset ego_recon \
    --output_dir "$N/loaded_$object" \
    --mano_model_dir "$N/human_motion_data/mano" \
    --human_motion_data_dir "$N/human_motion_data" \
    --dataset_root "$N/input_$object/episode_000020" \
    --object_name "$object" \
    --sequence_name episode_000020 \
    --no_ground_align --save > "$N/loader_$object.log" 2>&1
done
echo "E20_LOADERS_COMPLETE $(date -Is)"
