from pathlib import Path
import hashlib
import json
import shutil
import numpy as np

base = Path("/home/ubuntu/workspace/nvidia-v2d/runs/track3-production/episode_000032_rebuild")
native = base / "native"
records = []
for name in ("blue_cup", "wooden_spoon"):
    source_dir = base / name / "result_slam_gravity_aligned"
    calibration = base / name / "geocalib" / "calibration.json"
    npz_path = source_dir / "result.npz"
    assert calibration.is_file() and npz_path.is_file()
    with np.load(npz_path, allow_pickle=True) as data:
        assert bool(data["gravity_alignment_applied"].item())
        assert data["object_is_valid"].shape == (259,)
        assert np.all(data["object_is_valid"])
        assert np.isfinite(data["object_to_world_transform"]).all()
    manifest_paths = [source_dir / "manifest.json"]
    staged = native / f"input_{name}" / "episode_000032" / "outputs" / "result" / "manifest.json"
    if staged.is_file():
        manifest_paths.append(staged)
    for manifest_path in manifest_paths:
        backup = manifest_path.with_name("manifest.pre_source_flag_correction.json")
        if backup.exists():
            original = backup.read_bytes()
        else:
            original = manifest_path.read_bytes()
            shutil.copy2(manifest_path, backup)
        manifest = json.loads(original)
        sources = manifest.setdefault("sources", {})
        sources["run_geocalib"] = True
        sources["geocalib_calibration_path"] = str(calibration)
        sources["gravity_align"] = True
        assert manifest.get("gravity_alignment", {}).get("enabled") is True
        assert sources["gravity_align_target"] == "z_up"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        corrected = json.loads(manifest_path.read_text())
        assert corrected["sources"]["run_geocalib"] is True
        assert corrected["sources"]["gravity_align"] is True
        records.append({
            "object": name,
            "manifest": str(manifest_path),
            "original_sha256": hashlib.sha256(original).hexdigest(),
            "preserved_original": str(backup),
            "preserved_original_sha256": hashlib.sha256(backup.read_bytes()).hexdigest(),
            "corrected_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "calibration_path": str(calibration),
            "calibration_sha256": hashlib.sha256(calibration.read_bytes()).hexdigest(),
            "result_npz_sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
            "npz_gravity_alignment_applied": True,
            "valid_object_frames": 259,
        })
report = native / "episode_000032_manifest_corrections.json"
report.write_text(json.dumps({
    "episode_id": "episode_000032",
    "reason": "The original sources flags contradicted the selected NPZ gravity transform and present GeoCalib calibration. Only those source flags and calibration paths are corrected; original manifests are preserved.",
    "evidence": records,
}, indent=2) + "\n")
print(json.dumps(records, indent=2))

