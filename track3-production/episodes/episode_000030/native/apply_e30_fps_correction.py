from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

native = Path("/workspace/native")
host_native = "/home/ubuntu/workspace/nvidia-v2d/runs/track3-production/episode_000030_rebuild/native"
video = Path("/workspace/video/episode_000030.mp4")
video_host = "/home/ubuntu/workspace/nvidia-v2d/data/raw/video_to_data_challenge/track_3/evaluation/videos/chunk-000/observation.images.ego_cam_a/episode_000030.mp4"
probe = json.loads(subprocess.check_output([
    "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
    "stream=r_frame_rate,avg_frame_rate,duration,nb_frames", "-of", "json", str(video)
], text=True))["streams"][0]
assert probe["avg_frame_rate"] == "20/1" and int(probe["nb_frames"]) == 263, probe
records = []
for name in ("mini_sweeper", "plastic_dish_rack"):
    source_root = native / f"loaded_{name}"
    corrected_root = native / f"loaded_20fps_{name}"
    source_files = sorted(source_root.rglob("*.parquet"))
    assert len(source_files) == 1, (name, source_files)
    assert not corrected_root.exists(), corrected_root
    shutil.copytree(source_root, corrected_root)
    target = corrected_root / source_files[0].relative_to(source_root)
    original = pq.read_table(source_files[0])
    assert original.num_rows == 1
    index = original.schema.get_field_index("fps")
    assert index >= 0 and original["fps"].to_pylist() == [30.0]
    frames = np.asarray(original["object_body_position"][0].as_py()).shape[0]
    assert frames == 263, (name, frames)
    updated = original.set_column(index, original.schema.field(index),
                                  pa.array([20.0], type=original.schema.field(index).type))
    pq.write_table(updated, target)
    corrected = pq.read_table(target)
    assert corrected["fps"].to_pylist() == [20.0]
    assert original.drop(["fps"]).equals(corrected.drop(["fps"]))
    records.append({
        "object": name,
        "source_parquet": str(source_files[0]).replace("/workspace/native", host_native),
        "corrected_parquet": str(target).replace("/workspace/native", host_native),
        "source_sha256": hashlib.sha256(source_files[0].read_bytes()).hexdigest(),
        "corrected_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "source_fps": 30.0,
        "corrected_fps": 20.0,
        "frame_count": frames,
        "operation": "fps metadata only; no temporal resampling and all source rows retained",
    })
report = {
    "episode_id": "episode_000030",
    "video": video_host,
    "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
    "ffprobe": probe,
    "verified_fps": 20.0,
    "frame_count": 263,
    "loader_declared_fps_before_correction": 30.0,
    "corrected_fps": 20.0,
    "operation": "metadata correction only; the loader retained all source frames and no temporal resampling was applied",
    "original_outputs_preserved": ["loaded_mini_sweeper", "loaded_plastic_dish_rack"],
    "corrected_outputs": ["loaded_20fps_mini_sweeper", "loaded_20fps_plastic_dish_rack"],
    "parquet_files": records,
}
(native / "source_fps_correction.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
