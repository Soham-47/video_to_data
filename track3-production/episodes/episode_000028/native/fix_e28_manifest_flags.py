from pathlib import Path
import hashlib
import json
import shutil
import numpy as np

base = Path("/v2d/runs/track3-production/episode_000028_rebuild")
native = Path("/workspace/native")
records = []
for name in ("plastic_dish_rack", "white_pot"):
    source_dir = base / name / "result_slam_gravity_aligned"
    calibration = base / name / "geocalib" / "calibration.json"
    npz_path = source_dir / "result.npz"
    assert calibration.is_file() and npz_path.is_file()
    with np.load(npz_path, allow_pickle=True) as data:
        assert bool(data["gravity_alignment_applied"].item())
        assert np.all(data["object_is_valid"])
    bundle_paths = [
        source_dir / "manifest.json",
        native / f"input_{name}" / "episode_000028" / "outputs" / "result" / "manifest.json",
    ]
    for manifest_path in bundle_paths:
        original = manifest_path.read_bytes()
        backup = manifest_path.with_name("manifest.pre_source_flag_correction.json")
        if not backup.exists():
            shutil.copy2(manifest_path, backup)
        manifest = json.loads(original)
        src = manifest.setdefault("sources", {})
        src["run_geocalib"] = True
        src["geocalib_calibration_path"] = str(calibration)
        src["gravity_align"] = True
        assert manifest.get("gravity_alignment", {}).get("enabled") is True
        assert src["gravity_align_target"] == "z_up"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        corrected = manifest_path.read_bytes()
        records.append({
            "object": name,
            "manifest": str(manifest_path),
            "original_sha256": hashlib.sha256(original).hexdigest(),
            "preserved_original": str(backup),
            "preserved_original_sha256": hashlib.sha256(backup.read_bytes()).hexdigest(),
            "corrected_sha256": hashlib.sha256(corrected).hexdigest(),
            "calibration_path": str(calibration),
            "calibration_sha256": hashlib.sha256(calibration.read_bytes()).hexdigest(),
            "result_npz_sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
            "npz_gravity_alignment_applied": True,
            "geocalib_calibration_exists": True,
        })
out = native / "episode_000028_manifest_corrections.json"
out.write_text(json.dumps({
    "episode_id": "episode_000028",
    "reason": "Selected manifests said GeoCalib and gravity alignment were disabled, contradicting the selected NPZ and its embedded gravity transform. Corrected only those source flags and calibration paths.",
    "evidence": records,
}, indent=2) + "\n")
print(json.dumps(records, indent=2))
