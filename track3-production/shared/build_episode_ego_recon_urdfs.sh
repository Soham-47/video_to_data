#!/usr/bin/env bash
set -euo pipefail
target=/data/human_motion_data/ego_recon/processed
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
mkdir -p "$stage/ego_recon/processed"
for object in "$@"; do
  test -s "$target/$object.obj"
  ln -s "$target/$object.obj" "$stage/ego_recon/processed/$object.obj"
done
HUMAN_MOTION_DATA_DIR="$stage" python scripts/generate_rigid_urdfs.py --dataset ego_recon
for object in "$@"; do
  for suffix in _visual.stl _rigid.urdf; do
    test -s "$stage/ego_recon/processed/$object$suffix"
    cp "$stage/ego_recon/processed/$object$suffix" "$target/$object$suffix"
  done
done
