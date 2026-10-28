"""Write an evaluator-visible copy of a two-object trajectory with absolute assets.

The native loader intentionally stores portable relative ``object_assets/...``
paths.  The pinned evaluator resolves paths from its checkpoint/runtime root,
so this runtime-only helper creates a byte-distinct copy whose mesh and URDF
references point at the mounted ``/v2d`` closure.  The original Parquet is not
modified.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq


def main(source: str, output: str, asset_root: str) -> None:
    source_path = Path(source)
    output_path = Path(output)
    table = pq.read_table(source_path)
    if table.num_rows != 1:
        raise ValueError(f"expected one trajectory row, got {table.num_rows}")
    row = table.to_pylist()[0]
    root = Path(asset_root)
    row["object_mesh_paths"] = [str(root / "meshes" / "ego_recon" / "white_pot.obj"), str(root / "meshes" / "ego_recon" / "white_pot_lid.obj")]
    row["object_urdf_paths"] = [str(root / "urdfs" / "ego_recon" / "white_pot_rigid.urdf"), str(root / "urdfs" / "ego_recon" / "white_pot_lid_rigid.urdf")]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.from_pylist([row], schema=table.schema), output_path, compression="zstd")
    print(f"wrote={output_path} rows={table.num_rows} meshes={row['object_mesh_paths']} urdfs={row['object_urdf_paths']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("asset_root")
    args = parser.parse_args()
    main(args.source, args.output, args.asset_root)
