# Track 3 reproducibility source bundle

This is a sanitized, source-only supplement for the Track 3 production runs.
It is not a submission package and it does not contain videos, meshes,
checkpoints, trajectories, logs, credentials, derived JSON, or other generated
outputs.

## Public producing code

The producing framework changes are already publicly accessible at:

`https://github.com/Soham-47/video_to_data/commit/b24c77fb30771d72d9212698f7f390979cbd229e`

The producing checkout was inspected read-only at
`/home/ubuntu/workspace/nvidia-v2d/upstream/video_to_data`; its three modified
framework files matched the public commit byte-for-byte. They are intentionally
not duplicated here because publication will use the existing framework
repository and this commit URL. No new public commit is needed for these
framework changes.

The exact paths are recorded in `SOURCE_COMMIT.txt` for the maintainer.

## Coverage audit

The complete submission roster has 20 episodes. The bundle does not invent
missing source; coverage is recorded explicitly:

| Episodes | Source coverage |
| --- | --- |
| 1, 3 | Partial legacy reconstruction/evaluation scripts; shared two-object merge/validation helpers included. |
| 5, 6 | Missing episode source; local/HPC evidence archives contain generated outputs/config fragments only. |
| 8 | Partial: official-evaluation wrapper and input-path manifest only. |
| 10 | Missing episode source; production directory contains generated outputs/logs only. |
| 14 | Partial: recorded loader, retarget, training, and official-evaluation command fragments; no wrapper source. |
| 15 | Missing episode source; local evidence archive contains generated outputs only. |
| 16 | Missing episode source; local/HPC evidence archive contains generated outputs only. |
| 19, 20, 25, 30, 32, 33 | Complete episode-specific pipeline source/config for the compliant runs. |
| 28 | Partial: reconstruction, retarget, training, evaluation, validation, and provenance scripts/config; no episode loader/merge adapter source was present. |
| 35 | Resolved training config only; no episode execution source. |
| 36 | Partial: one recorded failed loader command fragment; no wrapper source. |
| 37 | Partial: recorded loader, retarget, and training command fragments; no wrapper source. |
| 38 | Missing episode source; only generated evidence was present. |

The six complete compliant pipelines preserve the observed order E19 -> E20
and E30 -> E25 -> E32 -> E33. `shared/legacy/` contains the identical
two-object merge and validation helpers used by E1 and E3.

## Included execution material

- `episodes/episode_XXXXXX/native/` contains the episode-specific Python and
  shell stages for reconstruction, loading, FPS correction, shared-frame
  merge, retarget/finalization, training, official evaluation/validation, and
  provenance construction.
- Resolved training YAML is included where present as a configuration snapshot.
  Computed FPS/merge JSON, provenance JSON, manifest corrections, checkpoints,
  and validation outputs were deliberately removed as derived results.
- `shared/build_episode_ego_recon_urdfs.sh` builds the rigid URDF assets;
  `shared/build_video_mesh_asset.py` prepares loader mesh assets; and
  `shared/wait_for_rebuild_disk.sh` is the small disk gate used by later
  reconstruction chains.
- `episodes/episode_000014/commands/`,
  `episodes/episode_000036/commands/`, and
  `episodes/episode_000037/commands/` contain small recorded command fragments
  recovered from backup text files/logs. The E36 loader fragment is explicitly
  marked failed; these fragments are provenance, not runnable complete stages.
- The handoff scripts preserve the observed execution order: E19 -> E20 and
  E30 -> E25 -> E32 -> E33.
- `episodes/episode_000020/native/resume_e20_after_eval.sh` is a
  non-rerunning post-evaluation helper: it applies the evidence-based manifest
  correction, rebuilds provenance, and creates/verifies a distinct compliant
  archive before emitting its completion marker.

The included handoff/loader scripts contain the original `nohup`, polling, and
cleanup operations used on Brev. They were copied for provenance only and were
not run during assembly; do not execute them against a live workspace without
reviewing their paths and cleanup scope.

## Deliberate exclusions

Excluded are all videos, raw frames, reconstructions, meshes, checkpoints,
parquet/JSON results, computed transforms, provenance and manifest outputs,
logs, PID files,
`.pre_compliance_*` snapshots, and archive/cleanup wrappers. Diagnostic-only
scripts (`analyze_scene_alignment.py`, `compare_scene_offsets.py`, and the
pre-residual-calibration E19 merge) are also excluded. Alternate
`run_*_eval_when_ready.sh` launchers and evidence-archive scripts are omitted;
the explicitly documented non-rerunning E20 post-evaluation resume helper is
the only archive-related source included.
the direct official-evaluation and validation stages used by the completed
pipelines are included.

## Runtime caveat

The scripts retain their recorded Brev paths and container names, including
`/home/ubuntu/workspace/nvidia-v2d`, `/v2d`, `/tmp/v2d-submission-venv`,
`robotic-grounding:latest`, and `flash_chord:8c778f9`. Reproduction therefore
requires the original dataset, MANO/object assets, container images, and the
episode inputs/checkpoints; this bundle alone is intentionally not executable.
Review and adapt paths before running any script. Nothing in this bundle was
executed while assembling it.

## Sanitization and integrity

Candidate source/config files were scanned for credential patterns without
printing matched values. No credential-bearing file was found; the only
`sk-` matches were the words `mask-based`/`mask-based crop` in comments. The
file list and SHA-256 values are in `SHA256SUMS`.
