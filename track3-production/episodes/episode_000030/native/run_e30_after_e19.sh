#!/usr/bin/env bash
set -euo pipefail
BASE=/home/ubuntu/workspace/nvidia-v2d
N19=$BASE/runs/track3-production/episode_000019_rebuild/native
E19_PID=$(cat "$N19/pipeline_supervisor.pid")
READY=0
for ((i=0; i<7200; i++)); do
  if grep -q E19_PIPELINE_COMPLETE /tmp/episode_000019_pipeline.log; then READY=1; break; fi
  state=$(ps -o stat= -p "$E19_PID" 2>/dev/null || true)
  if [[ -z "$state" || "$state" == Z* ]]; then echo "E19_PIPELINE_FAILED state=$state"; tail -n 100 /tmp/episode_000019_pipeline.log; exit 1; fi
  sleep 10
done
if [[ "$READY" != 1 ]]; then echo E19_PIPELINE_TIMEOUT; exit 1; fi
N30=$BASE/runs/track3-production/episode_000030_rebuild/native
nohup "$N30/run_e30_reconstruction_chain.sh" > "$N30/reconstruction_chain.log" 2>&1 < /dev/null & echo $! > "$N30/reconstruction_chain.pid"
nohup "$N30/run_e30_loaders_when_ready.sh" > "$N30/post_recon_supervisor.log" 2>&1 < /dev/null & echo $! > "$N30/post_recon_supervisor.pid"
nohup "$N30/run_e30_pipeline_when_loaders_ready.sh" > /tmp/episode_000030_pipeline.log 2>&1 < /dev/null & echo $! > "$N30/pipeline_supervisor.pid"
echo E30_PIPELINE_LAUNCHED
