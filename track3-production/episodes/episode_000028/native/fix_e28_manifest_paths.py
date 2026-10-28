from pathlib import Path
import hashlib, json
native=Path("/workspace/native")
episode=Path("/v2d/runs/track3-production/episode_000028_rebuild")
report_path=native/"episode_000028_manifest_corrections.json"
report=json.loads(report_path.read_text())
for record in report["evidence"]:
    name=record["object"]
    if "/input_" in record["manifest"]:
        actual=native/f"input_{name}"/"episode_000028"/"outputs"/"result"/"manifest.json"
    else:
        actual=episode/name/"result_slam_gravity_aligned"/"manifest.json"
    calibration=episode/name/"geocalib"/"calibration.json"
    host_calibration=Path("/home/ubuntu/workspace/nvidia-v2d/runs/track3-production/episode_000028_rebuild")/name/"geocalib"/"calibration.json"
    manifest=json.loads(actual.read_text())
    manifest["sources"]["run_geocalib"]=True
    manifest["sources"]["geocalib_calibration_path"]=str(host_calibration)
    manifest["sources"]["gravity_align"]=True
    actual.write_text(json.dumps(manifest,indent=2)+"\n")
    record["manifest"]=str(actual)
    record["calibration_path"]=str(host_calibration)
    record["container_calibration_path"]=str(calibration)
    record["corrected_sha256"]=hashlib.sha256(actual.read_bytes()).hexdigest()
report_path.write_text(json.dumps(report,indent=2)+"\n")
print("updated",len(report["evidence"]),"manifest records")
for r in report["evidence"]: print(r["manifest"],r["corrected_sha256"],r["calibration_path"])
