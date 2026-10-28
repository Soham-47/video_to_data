#!/usr/bin/env bash
set -euo pipefail
cd /workspace/video_to_data/robotic_grounding
COMPAT=/workspace/native/compat_pyarrow20
mkdir -p "$COMPAT"
python -m pip install --disable-pip-version-check --no-deps --target "$COMPAT" "pyarrow==20.0.0"
export PYTHONPATH="$COMPAT${PYTHONPATH:+:$PYTHONPATH}"
python -c 'import numpy, pyarrow; print("NUMPY",numpy.__version__,"PYARROW",pyarrow.__version__)'
python -u scripts/retarget/ego_recon_to_sharpa.py \
  --input_dir /workspace/native/merged_loaded20_flat \
  --output_dir /workspace/native/processed \
  --save --device cuda:0
