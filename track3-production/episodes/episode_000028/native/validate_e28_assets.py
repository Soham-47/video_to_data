from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
p=Path("/workspace/native/episode_000028.parquet")
row=pq.read_table(p).to_pylist()[0]
for key in ("object_mesh_paths","object_urdf_paths"):
    for value in row[key]:
        if not Path(value).is_file():
            raise SystemExit(f"missing {key}: {value}")
for key,value in row.items():
    if key.startswith("robot_") and value is not None:
        try:
            arr=np.asarray(value,dtype=np.float64)
        except (TypeError,ValueError):
            continue
        if arr.size and not np.isfinite(arr).all():
            raise SystemExit(f"non-finite {key}")
        print(key,arr.shape,"finite=true")
print("robot_fields",sum(k.startswith("robot_") for k in row),"asset_closure=true")
