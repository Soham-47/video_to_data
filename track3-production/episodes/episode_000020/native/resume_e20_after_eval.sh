#!/usr/bin/env bash
set -euo pipefail

BASE=/home/ubuntu/workspace/nvidia-v2d
PROD=$BASE/runs/track3-production
ROOT=$PROD/episode_000020_rebuild
N=$ROOT/native
BACKUP=$PROD/episode_000020_backup
ARCHIVE=$BACKUP/episode_000020_compliant_20261028_evidence.tgz
PYTHON=$BASE/runs/bootstrap/reconstruction-venv/bin/python

# Preserve the already successful evaluation and checkpoint; this resume only
# repairs evidence metadata, rebuilds provenance, and creates a new archive.
test -s "$N/checkpoint_njmax4096/policy_65536.safetensors"
grep -Eq "valid=true rows=404 frames=202 .*environment_valid=true fps=20 world_count=1" "$N/official_eval.log"
test -s "$N/episode_000020.parquet"
test -s "$N/episode_000020_retargeted_source.parquet"
test -s "$N/merge_shared_frame.json"
test -s "$N/reference_validation.json"

# The loader consumes result.npz and mesh data; this corrects only manifests,
# preserving each original manifest beside its corrected copy.
"$PYTHON" "$N/fix_e20_manifest_flags.py"
for object in blue_cup water_pitcher; do
  test -s "$ROOT/$object/result_slam_gravity_aligned/manifest.pre_source_flag_correction.json"
  test -s "$N/input_$object/episode_000020/outputs/result/manifest.pre_source_flag_correction.json"
done

# Use the same provenance container and mounts as the original post-evaluation pipeline.
docker run --rm \
  --mount type=bind,src="$N",dst=/workspace/native \
  --mount type=bind,src="$PROD",dst=/v2d/runs/track3-production,readonly \
  --mount type=bind,src="$BASE/data",dst=/source_data,readonly \
  -e PYTHONPATH=/workspace/native/compat_pyarrow20 \
  --entrypoint python robotic-grounding:latest \
  /workspace/native/build_e20_provenance.py > "$N/provenance_build.log" 2>&1
test -s "$N/episode_000020_provenance.json"

mkdir -p "$BACKUP"
cp "$PROD/build_episode_ego_recon_urdfs.sh" "$N/build_episode_ego_recon_urdfs.sh"
for path in "$ARCHIVE" "$ARCHIVE.sha256" "$BACKUP/SHA256SUMS" "$BACKUP/verify_files.log"; do
  test ! -e "$path"
done

archive_tmp="$ARCHIVE.tmp.$$"
sha_tmp="$BACKUP/SHA256SUMS.tmp.$$"
verify_tmp="$BACKUP/verify_files.log.tmp.$$"
archive_sha_tmp="$ARCHIVE.sha256.tmp.$$"
trap 'rm -f "$archive_tmp" "$sha_tmp" "$verify_tmp" "$archive_sha_tmp"' EXIT

tar -I 'gzip -1' -cf "$archive_tmp" -C "$PROD" episode_000020_rebuild
mv "$archive_tmp" "$ARCHIVE"
(cd "$ROOT" && find . -type f -print0 | sort -z | xargs -0 sha256sum) > "$sha_tmp"
(cd "$ROOT" && sha256sum -c "$sha_tmp") > "$verify_tmp"
mv "$sha_tmp" "$BACKUP/SHA256SUMS"
mv "$verify_tmp" "$BACKUP/verify_files.log"
sha256sum "$ARCHIVE" > "$archive_sha_tmp"
gzip -t "$ARCHIVE"
mv "$archive_sha_tmp" "$ARCHIVE.sha256"

archive_sha=$(sha256sum "$ARCHIVE" | awk '{print $1}')
printf 'E20_PIPELINE_COMPLETE archive=%s %s\n' "$archive_sha" "$ARCHIVE"
