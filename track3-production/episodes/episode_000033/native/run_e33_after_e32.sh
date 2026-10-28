#!/usr/bin/env bash
set -euo pipefail
BASE=/home/ubuntu/workspace/nvidia-v2d
P=$BASE/runs/track3-production
N32=$P/episode_000032_rebuild/native
N33=$P/episode_000033_rebuild/native
READY=0
E32_PID=
for ((i=0; i<7200; i++)); do
  if grep -q E32_PIPELINE_COMPLETE "$N32/pipeline_supervisor_retry.log" 2>/dev/null || grep -q E32_PIPELINE_COMPLETE /tmp/episode_000032_pipeline.log 2>/dev/null; then READY=1; break; fi
  if test -s "$N32/pipeline_supervisor.pid"; then E32_PID=$(cat "$N32/pipeline_supervisor.pid"); fi
  if test -n "$E32_PID"; then
    state=$(ps -o stat= -p "$E32_PID" 2>/dev/null || true)
    if [[ -z "$state" || "$state" == Z* ]]; then echo "E32_PIPELINE_FAILED state=$state"; tail -n 100 /tmp/episode_000032_pipeline.log; exit 1; fi
  fi
  sleep 10
done
if [[ "$READY" != 1 ]]; then echo E32_PIPELINE_TIMEOUT; exit 1; fi
# Keep enough room for E33's rebuild files and final compressed evidence archive.
for ((i=0; i<3600; i++)); do
  avail_kb=$(df -Pk "$BASE" | awk 'NR==2 {print $4}')
  if [[ "$avail_kb" =~ ^[0-9]+$ ]] && (( avail_kb >= 9*1024*1024 )); then break; fi
  if (( i == 3599 )); then echo "E33_DISK_SPACE_TIMEOUT available_kb=$avail_kb"; exit 1; fi
  sleep 10
done
nohup "$N33/run_e33_reconstruction_chain.sh" > "$N33/reconstruction_chain.log" 2>&1 < /dev/null & echo $! > "$N33/reconstruction_chain.pid"
nohup "$N33/run_e33_loaders_when_ready.sh" > "$N33/post_recon_supervisor.log" 2>&1 < /dev/null & echo $! > "$N33/post_recon_supervisor.pid"
nohup "$N33/run_e33_pipeline_when_loaders_ready.sh" > /tmp/episode_000033_pipeline.log 2>&1 < /dev/null & echo $! > "$N33/pipeline_supervisor.pid"
echo E33_PIPELINE_LAUNCHED
