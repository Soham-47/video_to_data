#!/usr/bin/env bash
set -euo pipefail
cd /workspace
python -u scripts/train_flash_sac.py \
  experiment=sharpa_flash_sac curriculum=reference \
  task.parquet=/workspace/native/episode_000020.parquet \
  task.motion_speed=1.0 task.source_frame_playback=false \
  task.motion_start_frame=0 task.motion_end_frame=-1 \
  sim.fps=20.0 sim.substeps=5 sim.njmax=4096 sim.nconmax=512 \
  scene.world_count=128 scene.support=false scene.decompose_objects=true \
  reset.seed=42 logging.mode=disabled logging.run_name=episode_000020 \
  training.total_environment_steps=65536 \
  training.updates_per_collection=1.0 \
  training.replay.capacity=131072 training.replay.minimum_size=8192 \
  training.replay.batch_size=512 \
  training.checkpoint.save_interval_environment_steps=32768 \
  training.network.actor_hidden_dim=256 \
  training.network.critic_hidden_dim=512 \
  training.network.initial_normalized_std=0.1 \
  training.network.log_std_max=-1.8971199848858813 \
  output_dir=/workspace/native/checkpoint_njmax4096 \
  hydra.run.dir=/tmp/flash_chord_train_episode_000020
