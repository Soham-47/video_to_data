#!/usr/bin/env python3
# Budget one video-derived metric mesh and refresh simulator asset files.
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, "/home/ubuntu/workspace/nvidia-v2d/runs/bootstrap/submission_kit/v2d_submission_kit")
from v2dlb.mesh_budget import budget_mesh

parser = argparse.ArgumentParser()
parser.add_argument("source", type=Path)
parser.add_argument("name")
parser.add_argument("native", type=Path)
args = parser.parse_args()
source = args.source.resolve()
assert source.is_file(), source
budget_dir = args.native / "mesh_budget" / args.name
asset_meshes = args.native / "object_assets/meshes/ego_recon"
asset_urdfs = args.native / "object_assets/urdfs/ego_recon"
for directory in (budget_dir, asset_meshes, asset_urdfs):
    directory.mkdir(parents=True, exist_ok=True)
raw = trimesh.load(source, force="mesh", process=False)
vertices, faces = budget_mesh(source, faces=4096, vertices=4096)
vertices, faces = np.asarray(vertices), np.asarray(faces)
assert vertices.shape == (4096, 3) and faces.shape == (4096, 3), (vertices.shape, faces.shape)
assert np.isfinite(vertices).all() and faces.min() >= 0 and faces.max() < len(vertices)
budget = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
obj_path = budget_dir / "mesh_budget.obj"
stl_path = asset_urdfs / (args.name + "_visual.stl")
asset_obj = asset_meshes / (args.name + ".obj")
obj_data = budget.export(file_type="obj")
obj_path.write_bytes(obj_data.encode() if isinstance(obj_data, str) else obj_data)
stl_path.write_bytes(budget.export(file_type="stl"))
asset_obj.write_bytes(obj_path.read_bytes())
np.savez(budget_dir / "mesh_budget.npz", vertices=vertices, faces=faces)
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
report = {
    "source_path": str(source),
    "source_sha256": sha(source),
    "source_vertices": int(len(raw.vertices)),
    "source_faces": int(len(raw.faces)),
    "source_extents_m": raw.bounding_box.extents.tolist(),
    "budget_vertices": int(len(vertices)),
    "budget_faces": int(len(faces)),
    "budget_degenerate_faces": int(np.sum(np.all(faces == 0, axis=1))),
    "budget_obj_sha256": sha(obj_path),
    "asset_obj_sha256": sha(asset_obj),
    "asset_stl_sha256": sha(stl_path),
}
(budget_dir / "mesh_budget_report.json").write_text(json.dumps(report, indent=2) + "\n")
urdf = [
    '<?xml version="1.0" ?>',
    '<robot name="' + args.name + '">',
    '   <mujoco><compiler meshdir="./" balanceinertia="true" discardvisual="false"/></mujoco>',
    '   <link name="object">',
    '      <inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="0.3"/>',
    '         <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial>',
    '      <visual><origin rpy="0 0 0" xyz="0 0 0"/><geometry><mesh filename="./' + args.name + '_visual.stl" scale="1 1 1"/></geometry></visual>',
    '      <collision><origin rpy="0 0 0" xyz="0 0 0"/><geometry><mesh filename="./' + args.name + '_visual.stl" scale="1 1 1"/></geometry></collision>',
    '   </link>',
    '</robot>',
]
(asset_urdfs / (args.name + "_rigid.urdf")).write_text("\n".join(urdf) + "\n")
assert len(stl_path.read_bytes()) == 84 + 50 * 4096, len(stl_path.read_bytes())
assert asset_obj.read_bytes() == obj_path.read_bytes()
print(json.dumps(report, sort_keys=True))

