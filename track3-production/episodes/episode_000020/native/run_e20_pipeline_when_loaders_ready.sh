#!/usr/bin/env bash
set -euo pipefail
BASE=/home/ubuntu/workspace/nvidia-v2d
PROD=$BASE/runs/track3-production
ROOT=$PROD/episode_000020_rebuild
N=$ROOT/native
POST_PID=$(cat "$N/post_recon_supervisor.pid")
READY=0
for ((i=0; i<3600; i++)); do
  if grep -q E20_LOADERS_AND_MANIFESTS_COMPLETE "$N/post_recon_supervisor.log"; then READY=1; break; fi
  state=$(ps -o stat= -p "$POST_PID" 2>/dev/null || true)
  if [[ -z "$state" || "$state" == Z* ]]; then echo "LOADER_STAGE_FAILED state=$state"; tail -n 100 "$N/post_recon_supervisor.log"; exit 1; fi
  sleep 10
done
if [[ "$READY" != 1 ]]; then echo LOADER_STAGE_TIMEOUT; exit 1; fi
VIDEO_DIR=$BASE/data/raw/video_to_data_challenge/track_3/evaluation/videos/chunk-000/observation.images.ego_cam_a
if [[ ! -s "$N/source_fps_correction.json" ]]; then
docker run --rm --mount type=bind,src="$N",dst=/workspace/native --mount type=bind,src="$VIDEO_DIR",dst=/workspace/video,readonly -e PYTHONPATH=/workspace/native/compat_pyarrow20 --entrypoint python robotic-grounding:latest /workspace/native/apply_e20_fps_correction.py > "$N/fps_correction.log" 2>&1
fi
docker run --rm --mount type=bind,src="$N",dst=/workspace/native --mount type=bind,src="$PROD",dst=/v2d/runs/track3-production,readonly -e PYTHONPATH=/workspace/native/compat_pyarrow20 --entrypoint python robotic-grounding:latest /workspace/native/merge_e20_shared_frame.py > "$N/merge.log" 2>&1
docker run --rm --gpus all --mount type=bind,src="$N",dst=/workspace/native --mount type=bind,src="$N/object_assets",dst=/workspace/object_assets,readonly --mount type=bind,src="$N/human_motion_data",dst=/data/human_motion_data --mount type=bind,src="$PROD",dst=/v2d/runs/track3-production,readonly -e PYTHONPATH=/workspace/native/compat_pyarrow20 -e XLA_PYTHON_CLIENT_PREALLOCATE=false --entrypoint bash robotic-grounding:latest -lc '/workspace/native/run_retarget_e20.sh && python /workspace/native/finalize_e20_reference.py' > "$N/retarget.log" 2>&1
docker run --rm --gpus all --mount type=bind,src="$N",dst=/workspace/native --mount type=bind,src="$N/object_assets",dst=/workspace/object_assets,readonly --mount type=bind,src="$N/human_motion_data",dst=/data/human_motion_data -e MUJOCO_GL=egl -e XLA_PYTHON_CLIENT_PREALLOCATE=false --entrypoint bash flash_chord:8c778f9 -lc '/workspace/native/run_train_e20.sh' > "$N/train_njmax4096.log" 2>&1
test -s "$N/checkpoint_njmax4096/policy_65536.safetensors"
grep -q 'completed 65536 environment steps' "$N/train_njmax4096.log"
docker run --rm --gpus all --mount type=bind,src="$N",dst=/workspace/native --mount type=bind,src="$N/object_assets",dst=/workspace/object_assets,readonly --mount type=bind,src="$PROD",dst=/v2d/runs/track3-production,readonly -e XLA_PYTHON_CLIENT_PREALLOCATE=false -e MUJOCO_GL=egl --entrypoint bash flash_chord:8c778f9 -lc '/workspace/native/run_official_eval_e20.sh && python -u /workspace/native/validate_official_e20.py' > "$N/official_eval.log" 2>&1
docker run --rm --mount type=bind,src="$N",dst=/workspace/native --mount type=bind,src="$PROD",dst=/v2d/runs/track3-production,readonly --mount type=bind,src="$BASE/data",dst=/source_data,readonly -e PYTHONPATH=/workspace/native/compat_pyarrow20 --entrypoint python robotic-grounding:latest /workspace/native/build_e20_provenance.py > "$N/provenance_build.log" 2>&1
BACKUP=$PROD/episode_000020_backup
mkdir -p "$BACKUP"
cp "$PROD/build_episode_ego_recon_urdfs.sh" "$N/build_episode_ego_recon_urdfs.sh"
tar -I 'gzip -1' -cf "$BACKUP/episode_000020_compliant_20261028_evidence.tgz" -C "$PROD" episode_000020_rebuild
(cd "$ROOT" && find . -type f -print0 | sort -z | xargs -0 sha256sum) > "$BACKUP/SHA256SUMS"
(cd "$ROOT" && sha256sum -c "$BACKUP/SHA256SUMS") > "$BACKUP/verify_files.log"
sha256sum "$BACKUP/episode_000020_compliant_20261028_evidence.tgz" > "$BACKUP/episode_000020_compliant_20261028_evidence.tgz.sha256"
gzip -t "$BACKUP/episode_000020_compliant_20261028_evidence.tgz"
printf 'E20_PIPELINE_COMPLETE archive='; sha256sum "$BACKUP/episode_000020_compliant_20261028_evidence.tgz"
printf 'files='; wc -l < "$BACKUP/SHA256SUMS"
tail -5 "$N/official_eval.log"
