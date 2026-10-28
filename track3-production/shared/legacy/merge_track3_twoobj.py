"""Merge two genuine single-object native loader rows for a Track 3 clip.

The loader emits one trajectory row per object.  This helper keeps the first
row's hand/robot fields and concatenates the two independent object pose
trajectories, metadata, and asset paths in the requested kit order.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def read_one(root: Path) -> tuple[pa.Table, dict]:
    files = sorted(root.rglob("*.parquet"))
    if len(files) != 1:
        raise RuntimeError(f"expected one parquet under {root}, found {files}")
    table = pq.read_table(files[0])
    rows = table.to_pylist()
    if len(rows) != 1:
        raise RuntimeError(f"expected one trajectory row in {files[0]}, found {len(rows)}")
    return table, rows[0]


def main(first_root: str, second_root: str, output_root: str, sequence_id: str, first_name: str, second_name: str) -> None:
    first_table, first = read_one(Path(first_root))
    second_table, second = read_one(Path(second_root))
    if first_table.schema != second_table.schema:
        raise ValueError("source parquet schemas differ")
    if float(first["fps"]) != float(second["fps"]):
        raise ValueError(f"fps mismatch: {first['fps']} vs {second['fps']}")

    first_pos = np.asarray(first["object_body_position"], dtype=np.float64)
    second_pos = np.asarray(second["object_body_position"], dtype=np.float64)
    first_q = np.asarray(first["object_body_wxyz"], dtype=np.float64)
    second_q = np.asarray(second["object_body_wxyz"], dtype=np.float64)
    frames = first_pos.shape[0]
    if first_pos.shape != (frames, 1, 3) or second_pos.shape != (frames, 1, 3):
        raise ValueError(f"expected one body per source: {first_pos.shape}, {second_pos.shape}")
    if first_q.shape != (frames, 1, 4) or second_q.shape != (frames, 1, 4):
        raise ValueError(f"unexpected quaternion shapes: {first_q.shape}, {second_q.shape}")
    for label, arr in (("first positions", first_pos), ("second positions", second_pos), ("first quaternions", first_q), ("second quaternions", second_q)):
        if not np.isfinite(arr).all():
            raise ValueError(f"non-finite {label}")
    qnorm = np.linalg.norm(np.concatenate([first_q[:, 0], second_q[:, 0]], axis=0), axis=1)
    qnorm_maxerr = float(np.max(np.abs(qnorm - 1.0)))
    if qnorm_maxerr > 1e-4:
        raise ValueError(f"source quaternion norm out of tolerance: {qnorm_maxerr}")

    row = dict(first)
    row.update(
        sequence_id=sequence_id,
        raw_motion_file=f"{first['raw_motion_file']}+{second['raw_motion_file']}",
        object_name=f"{first_name}+{second_name}",
        safe_object_name=f"{first_name}_{second_name}",
        object_body_names=[first_name, second_name],
        safe_object_body_names=[first_name, second_name],
        object_mesh_paths=[first["object_mesh_paths"][0], second["object_mesh_paths"][0]],
        object_urdf_paths=[first["object_urdf_paths"][0], second["object_urdf_paths"][0]],
        object_mesh_radius=[first["object_mesh_radius"][0], second["object_mesh_radius"][0]],
        object_body_position=np.concatenate([first_pos, second_pos], axis=1).tolist(),
        object_body_wxyz=np.concatenate([first_q, second_q], axis=1).tolist(),
    )
    out_file = Path(output_root) / f"sequence_id={sequence_id}" / "robot_name=sharpa_wave" / f"{sequence_id}_twoobj.parquet"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([row], schema=first_table.schema), str(out_file), compression="zstd")
    separation = np.linalg.norm(second_pos[:, 0] - first_pos[:, 0], axis=1)
    print(f"merged frames={frames} fps={row['fps']} bodies={row['object_body_names']} mean_separation={separation.mean():.6f} max_separation={separation.max():.6f} finite=true qnorm_maxerr={qnorm_maxerr:.3e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("first_root")
    parser.add_argument("second_root")
    parser.add_argument("output_root")
    parser.add_argument("--sequence-id", default="episode_000014")
    parser.add_argument("--first-name", default="blue_cup")
    parser.add_argument("--second-name", default="plastic_dish_rack")
    args = parser.parse_args()
    main(args.first_root, args.second_root, args.output_root, args.sequence_id, args.first_name, args.second_name)
