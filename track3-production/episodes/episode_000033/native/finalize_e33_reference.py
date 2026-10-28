from pathlib import Path
import glob
import hashlib
import json
import shutil
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

native = Path("/workspace/native")
source_path = Path(glob.glob(str(native / "processed" / "**" / "*.parquet"), recursive=True)[0])
assert source_path.is_file()
retargeted_source = native / "episode_000033_retargeted_source.parquet"
shutil.copy2(source_path, retargeted_source)
table = pq.ParquetFile(retargeted_source).read()
assert table.num_rows == 1
row = table.to_pylist()[0]
assert float(row["fps"]) == 20.0
assert row["object_body_names"] == ["blue_cup", "wooden_spoon"]
positions = np.asarray(row["object_body_position"], dtype=np.float64)
quats = np.asarray(row["object_body_wxyz"], dtype=np.float64)
assert positions.shape == (348, 2, 3) and quats.shape == (348, 2, 4)
assert np.isfinite(positions).all() and np.isfinite(quats).all()
qnorm = np.linalg.norm(quats, axis=-1)
assert float(np.max(np.abs(qnorm - 1.0))) < 1e-5
for key in ("object_mesh_paths", "object_urdf_paths"):
    for path in row[key]:
        assert Path(path).is_file(), (key, path)
for key, value in row.items():
    if key.startswith("robot_") and value is not None:
        try:
            arr = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError):
            continue
        if arr.size:
            assert np.isfinite(arr).all(), key
mesh_paths = [
    "/workspace/object_assets/meshes/ego_recon/blue_cup.obj",
    "/workspace/object_assets/meshes/ego_recon/wooden_spoon.obj",
]
urdf_paths = [
    "/workspace/object_assets/urdfs/ego_recon/blue_cup_rigid.urdf",
    "/workspace/object_assets/urdfs/ego_recon/wooden_spoon_rigid.urdf",
]
for path in mesh_paths + urdf_paths:
    assert Path(path).is_file(), path
row["object_mesh_paths"] = mesh_paths
row["object_urdf_paths"] = urdf_paths
final = native / "episode_000033.parquet"
pq.write_table(pa.Table.from_pylist([row], schema=table.schema), final)
final_row = pq.ParquetFile(final).read().to_pylist()[0]
assert final_row["object_mesh_paths"] == mesh_paths
assert final_row["object_urdf_paths"] == urdf_paths
final_positions = np.asarray(final_row["object_body_position"], dtype=np.float64)
distances = np.linalg.norm(final_positions[:, 0, :] - final_positions[:, 1, :], axis=1)
report = {
    "episode_id": "episode_000033",
    "source_retargeted_parquet": str(source_path).replace("/workspace/native", "/home/ubuntu/workspace/nvidia-v2d/runs/track3-production/episode_000033_rebuild/native"),
    "source_sha256": hashlib.sha256(retargeted_source.read_bytes()).hexdigest(),
    "final_parquet": str(final).replace("/workspace/native", "/home/ubuntu/workspace/nvidia-v2d/runs/track3-production/episode_000033_rebuild/native"),
    "final_sha256": hashlib.sha256(final.read_bytes()).hexdigest(),
    "frames": 348,
    "fps": 20.0,
    "object_order": final_row["object_body_names"],
    "object_body_position_shape": list(final_positions.shape),
    "object_body_quaternion_shape": list(np.asarray(final_row["object_body_wxyz"]).shape),
    "quaternion_norm_error_max": float(np.max(np.abs(qnorm - 1.0))),
    "pair_distance_mean_max_m": [float(distances.mean()), float(distances.max())],
    "robot_fields_finite": True,
    "asset_paths": {"meshes": mesh_paths, "urdfs": urdf_paths},
    "asset_closure": True,
}
(native / "reference_validation.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
