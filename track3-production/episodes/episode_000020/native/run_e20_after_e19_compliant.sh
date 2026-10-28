#!/usr/bin/env bash
set -euo pipefail
BASE=/home/ubuntu/workspace/nvidia-v2d
PROD=$BASE/runs/track3-production
N19=$PROD/episode_000019_rebuild/native
N=$PROD/episode_000020_rebuild/native
ROOT=$PROD/episode_000020_rebuild
PID=$(cat "$N19/pipeline_supervisor.pid")
READY=0
for ((i=0; i<7200; i++)); do
  if grep -q E19_PIPELINE_COMPLETE /tmp/episode_000019_pipeline.log; then READY=1; break; fi
  state=$(ps -o stat= -p "$PID" 2>/dev/null || true)
  if [[ -z "$state" || "$state" == Z* ]]; then echo "E19_COMPLIANT_PIPELINE_FAILED state=$state"; tail -n 100 /tmp/episode_000019_pipeline.log; exit 1; fi
  sleep 5
done
if [[ "$READY" != 1 ]]; then echo E19_COMPLIANT_PIPELINE_TIMEOUT; exit 1; fi
sudo rm -rf "$ROOT/blue_cup" "$ROOT/water_pitcher" "$N/input_blue_cup" "$N/input_water_pitcher" "$N/loaded_blue_cup" "$N/loaded_water_pitcher" "$N/loaded_20fps_blue_cup" "$N/loaded_20fps_water_pitcher" "$N/merged_loaded20_flat" "$N/processed" "$N/object_assets" "$N/mesh_budget" "$N/checkpoint_njmax4096" "$N/official_results_mounted"
sudo rm -f "$N/episode_000020.parquet" "$N/episode_000020_retargeted_source.parquet" "$N/episode_000020_checkpoint_65536.json" "$N/episode_000020_provenance.json" "$N/reference_validation.json" "$N/merge_shared_frame.json" "$N/source_fps_correction.json" "$N/reconstruction_chain.log" "$N/post_recon_supervisor.log" "$N/train_njmax4096.log" "$N/official_eval.log" "$N/retarget.log" "$N/merge.log" "$N/provenance_build.log" "$N/reconstruction_blue_cup.log" "$N/reconstruction_water_pitcher.log" "$N/loader_blue_cup.log" "$N/loader_water_pitcher.log" "$N/reconstruction_chain.pid" "$N/post_recon_supervisor.pid" "$N/pipeline_supervisor.pid" "$PROD/episode_000020_backup/SHA256SUMS" "$PROD/episode_000020_backup/verify_files.log"
: > "$N/reconstruction_chain.log"
: > "$N/post_recon_supervisor.log"
: > /tmp/episode_000020_pipeline.log
nohup "$N/run_e20_reconstruction_chain.sh" > "$N/reconstruction_chain.log" 2>&1 < /dev/null & echo $! > "$N/reconstruction_chain.pid"
nohup "$N/run_e20_loaders_when_ready.sh" > "$N/post_recon_supervisor.log" 2>&1 < /dev/null & echo $! > "$N/post_recon_supervisor.pid"
nohup "$N/run_e20_pipeline_when_loaders_ready.sh" > /tmp/episode_000020_pipeline.log 2>&1 < /dev/null & echo $! > "$N/pipeline_supervisor.pid"
echo "E20_COMPLIANT_RERUN_LAUNCHED recon=$(cat "$N/reconstruction_chain.pid") loaders=$(cat "$N/post_recon_supervisor.pid") pipeline=$(cat "$N/pipeline_supervisor.pid")"
