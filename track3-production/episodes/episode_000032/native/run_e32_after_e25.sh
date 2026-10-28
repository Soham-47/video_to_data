#!/usr/bin/env bash
set -euo pipefail
BASE=/home/ubuntu/workspace/nvidia-v2d
P=$BASE/runs/track3-production
N25=$P/episode_000025_rebuild/native
N32=$P/episode_000032_rebuild/native
N33=$P/episode_000033_rebuild/native
READY=0
E25_PID=
for ((i=0; i<7200; i++)); do
  if grep -q E25_PIPELINE_COMPLETE /tmp/episode_000025_pipeline.log 2>/dev/null; then READY=1; break; fi
  if test -s "$N25/pipeline_supervisor.pid"; then E25_PID=$(cat "$N25/pipeline_supervisor.pid"); fi
  if test -n "$E25_PID"; then
    state=$(ps -o stat= -p "$E25_PID" 2>/dev/null || true)
    if [[ -z "$state" || "$state" == Z* ]]; then echo "E25_PIPELINE_FAILED state=$state"; tail -n 100 /tmp/episode_000025_pipeline.log; exit 1; fi
  fi
  sleep 10
done
if [[ "$READY" != 1 ]]; then echo E25_PIPELINE_TIMEOUT; exit 1; fi
nohup "$N32/run_e32_reconstruction_chain.sh" > "$N32/reconstruction_chain.log" 2>&1 < /dev/null & echo $! > "$N32/reconstruction_chain.pid"
nohup "$N32/run_e32_loaders_when_ready.sh" > "$N32/post_recon_supervisor.log" 2>&1 < /dev/null & echo $! > "$N32/post_recon_supervisor.pid"
nohup "$N32/run_e32_pipeline_when_loaders_ready.sh" > /tmp/episode_000032_pipeline.log 2>&1 < /dev/null & echo $! > "$N32/pipeline_supervisor.pid"
nohup "$N33/run_e33_after_e32.sh" > "$N33/after_e32_launcher.log" 2>&1 < /dev/null & echo $! > "$N33/after_e32_launcher.pid"
echo E32_PIPELINE_LAUNCHED
