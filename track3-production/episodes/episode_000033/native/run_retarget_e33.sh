#!/usr/bin/env bash
set -euo pipefail
cd /workspace/video_to_data/robotic_grounding
export PYTHONPATH=/workspace/native/compat_pyarrow20${PYTHONPATH:+:${PYTHONPATH}}
export HUMAN_MOTION_DATA_DIR=/data/human_motion_data
/v2d/runs/track3-production/build_episode_ego_recon_urdfs.sh blue_cup wooden_spoon
python -c 'import numpy, pyarrow; print("NUMPY", numpy.__version__, "PYARROW", pyarrow.__version__)'
python -u scripts/retarget/ego_recon_to_sharpa.py \
  --input_dir /workspace/native/merged_loaded20_flat \
  --output_dir /workspace/native/processed \
  --save --device cuda:0
