#!/usr/bin/env bash
set -euo pipefail
BASE=/home/ubuntu/workspace/nvidia-v2d
N30=$BASE/runs/track3-production/episode_000030_rebuild/native
N25=$BASE/runs/track3-production/episode_000025_rebuild/native
LAUNCHER_PID=$(cat "$N30/after_e19_launcher.pid")
READY=0
for ((i=0; i<7200; i++)); do
  if grep -q E30_PIPELINE_COMPLETE /tmp/episode_000030_pipeline.log 2>/dev/null; then READY=1; break; fi
  if grep -q E30_PIPELINE_LAUNCHED "$N30/after_e19_launcher.log" 2>/dev/null; then
    E30_PID=$(cat "$N30/pipeline_supervisor.pid")
    break
  fi
  state=$(ps -o stat= -p "$LAUNCHER_PID" 2>/dev/null || true)
  if [[ -z "$state" || "$state" == Z* ]]; then echo "E30_LAUNCH_FAILED state=$state"; tail -n 100 "$N30/after_e19_launcher.log"; exit 1; fi
  sleep 10
done
if [[ "$READY" != 1 && -z "${E30_PID:-}" ]]; then echo E30_LAUNCH_TIMEOUT; exit 1; fi
if [[ "$READY" != 1 ]]; then
  for ((i=0; i<7200; i++)); do
    if grep -q E30_PIPELINE_COMPLETE /tmp/episode_000030_pipeline.log; then READY=1; break; fi
    state=$(ps -o stat= -p "$E30_PID" 2>/dev/null || true)
    if [[ -z "$state" || "$state" == Z* ]]; then echo "E30_PIPELINE_FAILED state=$state"; tail -n 100 /tmp/episode_000030_pipeline.log; exit 1; fi
    sleep 10
  done
fi
if [[ "$READY" != 1 ]]; then echo E30_PIPELINE_TIMEOUT; exit 1; fi
nohup "$N25/run_e25_reconstruction_chain.sh" > "$N25/reconstruction_chain.log" 2>&1 < /dev/null & echo $! > "$N25/reconstruction_chain.pid"
nohup "$N25/run_e25_loaders_when_ready.sh" > "$N25/post_recon_supervisor.log" 2>&1 < /dev/null & echo $! > "$N25/post_recon_supervisor.pid"
nohup "$N25/run_e25_pipeline_when_loaders_ready.sh" > /tmp/episode_000025_pipeline.log 2>&1 < /dev/null & echo $! > "$N25/pipeline_supervisor.pid"
N32=$BASE/runs/track3-production/episode_000032_rebuild/native
nohup "$N32/run_e32_after_e25.sh" > "$N32/after_e25_launcher.log" 2>&1 < /dev/null & echo $! > "$N32/after_e25_launcher.pid"
echo E25_PIPELINE_LAUNCHED
