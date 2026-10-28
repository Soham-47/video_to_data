from pathlib import Path
import glob
import hashlib
import json
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

native = Path("/workspace/native")
episode_root = Path("/v2d/runs/track3-production/episode_000019_rebuild")
names = ("mini_sweeper", "mini_dust_pan")
rows, tables, input_paths, offsets = [], [], [], []
source_matrices, cameras, gravity = [], [], []
for name in names:
    paths = sorted(glob.glob(str(native / f"loaded_20fps_{name}" / "**" / "*.parquet"), recursive=True))
    assert len(paths) == 1, (name, paths)
    table = pq.read_table(paths[0])
    assert table.num_rows == 1
    row = table.to_pylist()[0]
    assert row["fps"] == 20.0
    assert row["object_body_names"] == [name]
    positions = np.asarray(row["object_body_position"], dtype=np.float64)
    orientations = np.asarray(row["object_body_wxyz"], dtype=np.float64)
    bundle = episode_root / name / "result_slam_gravity_aligned"
    with np.load(bundle / "result.npz", allow_pickle=True) as data:
        matrix = data["object_to_world_transform"].astype(np.float64)
        source_matrices.append(matrix)
        cameras.append(data["camera_to_world_transform"].astype(np.float64))
        gravity.append(data["gravity_alignment_transform"].astype(np.float64))
    source_pos = matrix[:, :3, 3]
    delta = positions[:, 0, :] - source_pos
    offset = delta.mean(0)
    assert float(np.max(np.abs(delta - offset))) < 1e-5
    source_rot = Rotation.from_matrix(matrix[:, :3, :3])
    loaded_q = orientations[:, 0, :]
    loaded_rot = Rotation.from_quat(loaded_q[:, [1, 2, 3, 0]])
    quat_error = float((source_rot.inv() * loaded_rot).magnitude().max())
    assert quat_error < 1e-5
    assert positions.shape == (235, 1, 3) and orientations.shape == (235, 1, 4)
    assert np.isfinite(positions).all() and np.isfinite(orientations).all()
    assert np.allclose(np.linalg.norm(orientations, axis=-1), 1.0, atol=1e-5)
    rows.append(row)
    tables.append(table)
    input_paths.append(Path(paths[0]))
    offsets.append(offset)

assert cameras[0].shape == cameras[1].shape == (235, 4, 4)
camera_maps = np.einsum("tij,tjk->tik", cameras[0], np.linalg.inv(cameras[1]))
world2_to_world1 = np.eye(4)
world2_to_world1[:3, :3] = Rotation.from_matrix(camera_maps[:, :3, :3]).mean().as_matrix()
world2_to_world1[:3, 3] = camera_maps[:, :3, 3].mean(0)
aligned_cameras = np.einsum("ij,tjk->tik", world2_to_world1, cameras[1])
camera_translation_error = np.linalg.norm(aligned_cameras[:, :3, 3] - cameras[0][:, :3, 3], axis=1)
camera_rotation_error = (
    Rotation.from_matrix(cameras[0][:, :3, :3]).inv()
    * Rotation.from_matrix(aligned_cameras[:, :3, :3])
).magnitude()
assert float(camera_translation_error.max()) < 0.012
assert float(camera_rotation_error.max()) < np.deg2rad(0.3)

rotation_map = Rotation.from_matrix(world2_to_world1[:3, :3])
first_pos = np.asarray(rows[0]["object_body_position"], dtype=np.float64)
second_pos = np.asarray(rows[1]["object_body_position"], dtype=np.float64)
second_in_world1 = rotation_map.apply(second_pos[:, 0, :]) + world2_to_world1[:3, 3]
translation_adjustment = offsets[0] - rotation_map.apply(offsets[1])
second_shared = (second_in_world1 + translation_adjustment)[:, None, :]
aligned_source2 = np.einsum("ij,tjk->tik", world2_to_world1, source_matrices[1])
source_distance = np.linalg.norm(
    source_matrices[0][:, :3, 3] - aligned_source2[:, :3, 3], axis=1
)
shared_distance = np.linalg.norm(first_pos[:, 0, :] - second_shared[:, 0, :], axis=1)
assert float(np.max(np.abs(source_distance - shared_distance))) < 1e-5

q1 = np.asarray(rows[0]["object_body_wxyz"], dtype=np.float64)
q2 = np.asarray(rows[1]["object_body_wxyz"], dtype=np.float64)
q2_xyzw = q2[:, 0, :][:, [1, 2, 3, 0]]
q2_aligned_xyzw = (rotation_map * Rotation.from_quat(q2_xyzw)).as_quat()
q2_aligned_wxyz = q2_aligned_xyzw[:, [3, 0, 1, 2]][:, None, :]
aligned_source_rot2 = rotation_map * Rotation.from_matrix(source_matrices[1][:, :3, :3])
q2_source_xyzw = aligned_source_rot2.as_quat()
rotation_error = float(
    (aligned_source_rot2.inv() * Rotation.from_quat(q2_aligned_xyzw)).magnitude().max()
)
assert rotation_error < 1e-5

row = dict(rows[0])
row["raw_motion_file"] = rows[0]["raw_motion_file"] + "+" + rows[1]["raw_motion_file"]
row["object_name"] = rows[0]["object_name"] + "+" + rows[1]["object_name"]
row["safe_object_name"] = rows[0]["safe_object_name"] + "_" + rows[1]["safe_object_name"]
for key in ("object_body_names", "safe_object_body_names", "object_mesh_paths",
            "object_urdf_paths", "object_mesh_radius"):
    row[key] = list(rows[0][key]) + list(rows[1][key])
row["object_body_position"] = np.concatenate((first_pos, second_shared), axis=1).tolist()
row["object_body_wxyz"] = np.concatenate((q1, q2_aligned_wxyz), axis=1).tolist()
row.pop("sequence_id", None)
row.pop("robot_name", None)
schema = tables[0].schema
for key in ("sequence_id", "robot_name"):
    if key in schema.names:
        schema = schema.remove(schema.get_field_index(key))
assert row["object_body_names"] == ["mini_sweeper", "mini_dust_pan"]
assert np.asarray(row["object_body_position"]).shape == (235, 2, 3)
assert np.asarray(row["object_body_wxyz"]).shape == (235, 2, 4)
assert np.isfinite(np.asarray(row["object_body_position"])).all()
assert np.isfinite(np.asarray(row["object_body_wxyz"])).all()

out_dir = native / "merged_loaded20_flat" / "sequence_id=episode_000019" / "robot_name=sharpa_wave"
out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / "episode_000019_twoobj.parquet"
pq.write_table(pa.Table.from_pylist([row], schema=schema), out)
written = pq.read_table(out).to_pylist()[0]
written_pos = np.asarray(written["object_body_position"], dtype=np.float64)
assert written["object_body_names"] == ["mini_sweeper", "mini_dust_pan"]
assert np.max(np.abs(written_pos[:, 0, :] - first_pos[:, 0, :])) == 0.0
assert float(np.max(np.abs(
    np.linalg.norm(written_pos[:, 0, :] - written_pos[:, 1, :], axis=1) - source_distance
))) < 1e-5

gravity_map = gravity[0] @ np.linalg.inv(gravity[1])
gravity_alignment_residual = float(np.max(np.abs(world2_to_world1 - gravity_map)))
report = {
    "episode_id": "episode_000019",
    "ordered_objects": list(names),
    "frames": 235,
    "fps": 20.0,
    "source_camera_frames_equal": bool(np.allclose(cameras[0], cameras[1], atol=1e-6)),
    "source_camera_frames_aligned_by_rigid_transform": True,
    "world2_to_world1_transform": world2_to_world1.tolist(),
    "camera_alignment_max_translation_residual_m": float(camera_translation_error.max()),
    "camera_alignment_max_rotation_residual_deg": float(np.rad2deg(camera_rotation_error.max())),
    "gravity_transform_mapping_residual_max_abs": gravity_alignment_residual,
    "translation_offsets_from_source_world": {name: offsets[i].tolist() for i, name in enumerate(names)},
    "second_object_translation_adjustment_into_first_object_frame": translation_adjustment.tolist(),
    "source_pair_distance_mean_max_m": [float(source_distance.mean()), float(source_distance.max())],
    "merged_pair_distance_mean_max_m": [
        float(np.linalg.norm(written_pos[:, 0, :] - written_pos[:, 1, :], axis=1).mean()),
        float(np.linalg.norm(written_pos[:, 0, :] - written_pos[:, 1, :], axis=1).max()),
    ],
    "body_pose_rotations_match_source_after_alignment": True,
    "body_positions_match_source_pairwise_distances": True,
    "shared_hand_and_contact_fields_from_first_object_loader": True,
    "inputs": [{"path": str(p).replace("/workspace/native", "/home/ubuntu/workspace/nvidia-v2d/runs/track3-production/episode_000019_rebuild/native"),
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in input_paths],
    "output": str(out).replace("/workspace/native", "/home/ubuntu/workspace/nvidia-v2d/runs/track3-production/episode_000019_rebuild/native"),
    "output_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
}
(native / "merge_shared_frame.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
