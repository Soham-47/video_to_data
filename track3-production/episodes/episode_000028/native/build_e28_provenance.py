from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, re, subprocess
import numpy as np
import pyarrow.parquet as pq

project = Path("/home/ubuntu/workspace/nvidia-v2d")
track = Path("/v2d/runs/track3-production")
episode = track / "episode_000028_rebuild"
native = Path("/workspace/native")
template = json.loads((track / "episode_000008_rebuild/native/episode_000008_provenance.json").read_text())
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def host(path):
    return str(path).replace("/v2d/runs/track3-production", str(project / "runs/track3-production")).replace("/workspace/native", str(project / "runs/track3-production/episode_000028_rebuild/native"))
source_video_rel = "data/raw/video_to_data_challenge/track_3/evaluation/videos/chunk-000/observation.images.ego_cam_a/episode_000028.mp4"
video = Path("/source_data") / source_video_rel.removeprefix("data/")
probe = json.loads(subprocess.check_output([
    "ffprobe","-v","error","-select_streams","v:0","-show_entries","stream=avg_frame_rate,nb_frames,duration","-of","json",str(video)
], text=True))["streams"][0]
from fractions import Fraction
fps = float(Fraction(probe["avg_frame_rate"]))
frames = int(probe["nb_frames"])
assert fps == 20.0 and frames == 220, (probe, fps, frames)
# Normalize correction-report paths to host paths before hashing it.
correction_path = native / "episode_000028_manifest_corrections.json"
correction = json.loads(correction_path.read_text())
for record in correction["evidence"]:
    name = record["object"]
    if "/input_" in record["manifest"] or "/workspace/native/" in record["manifest"]:
        manifest_path = native / f"input_{name}" / "episode_000028/outputs/result/manifest.json"
    else:
        manifest_path = episode / name / "result_slam_gravity_aligned/manifest.json"
    backup_path = manifest_path.with_name("manifest.pre_source_flag_correction.json")
    record["manifest"] = host(manifest_path)
    record["preserved_original"] = host(backup_path)
    record["corrected_sha256"] = sha(manifest_path)
correction_path.write_text(json.dumps(correction, indent=2) + "\n")
objects = {}
for name, prior_episode, prompt in (
    ("plastic_dish_rack", "episode_000014", "a plastic dish rack"),
    ("white_pot", "episode_000001", "a white pot"),
):
    obj_root = episode / name
    source_manifest = obj_root / "result_slam_gravity_aligned/manifest.json"
    source_npz = obj_root / "result_slam_gravity_aligned/result.npz"
    staged = native / f"input_{name}/episode_000028/outputs/result"
    with np.load(source_npz, allow_pickle=True) as z:
        object_valid_frames = int(np.asarray(z["object_is_valid"], dtype=bool).sum())
        gravity_applied = bool(z["gravity_alignment_applied"].item())
    assert object_valid_frames == frames and gravity_applied
    scale_path = obj_root / "provided_mesh_scale.json"
    scale = float(json.loads(scale_path.read_text())["scale"])
    run_script = native / ("run_e28_plastic_dish_rack.sh" if name == "plastic_dish_rack" else "run_e28_white_pot.sh")
    mesh_arg = re.search(r"^MESH=(.+)$", run_script.read_text(), re.M).group(1)
    source_host_path = mesh_arg.replace("$BASE", str(project))
    source_container_path = mesh_arg.replace("$BASE/runs/track3-production", "/v2d/runs/track3-production")
    source_container = Path(source_container_path)
    assert source_container.is_file(), source_container
    calibration = obj_root / "geocalib/calibration.json"
    report_path = native / "mesh_budget" / name / "mesh_budget_report.json"
    budget_obj = native / "mesh_budget" / name / "mesh_budget.obj"
    report = json.loads(report_path.read_text())
    budget = {
        **report,
        "report_path": str(report_path.relative_to(native)),
        "report_sha256": sha(report_path),
        "budget_obj_path": str(budget_obj.relative_to(native)),
        "budget_obj_sha256": sha(budget_obj),
    }
    objects[name] = {
        "frames": frames,
        "object_valid_frames": object_valid_frames,
        "prompt": prompt,
        "source_mesh_episode": prior_episode,
        "input_mesh_path": source_host_path,
        "input_mesh_sha256": sha(source_container),
        "provided_mesh_scale": scale,
        "scale_file_sha256": sha(scale_path),
        "provided_mesh_path": str(obj_root / "provided_mesh_scaled.obj"),
        "provided_mesh_sha256": sha(obj_root / "provided_mesh_scaled.obj"),
        "geocalib_calibration_path": host(calibration),
        "geocalib_calibration_sha256": sha(calibration),
        "gravity_alignment_enabled": gravity_applied,
        "gravity_target": "z_up",
        "result_bundle": str(Path(f"input_{name}") / "episode_000028/outputs/result"),
        "result_manifest_sha256": sha(staged / "manifest.json"),
        "result_npz_sha256": sha(staged / "result.npz"),
        "mesh_budget": budget,
    }
ref_path = native / "episode_000028.parquet"
source_ref_path = native / "episode_000028_retargeted_source.parquet"
ref_table = pq.read_table(ref_path)
assert ref_table.num_rows == 1
ref = ref_table.to_pylist()[0]
assert ref["fps"] == 20.0 and len(ref["object_body_position"]) == frames
assert ref["object_body_names"] == ["plastic_dish_rack","white_pot"]
checkpoint_dir = native / "checkpoint_njmax4096"
checkpoint_map_path = native / "episode28_checkpoint_65536.json"
checkpoint_map_path.write_text(json.dumps({"28":"/v2d/runs/track3-production/episode_000028_rebuild/native/checkpoint_njmax4096/policy_65536.safetensors"},indent=2)+"\n")
train_log = native / "train_njmax4096.log"
train_text = train_log.read_text(errors="replace")
assert "completed 65536 environment steps" in train_text
official_json_path = native / "official_results_mounted/episode_000028.json"
official_parquet_path = native / "official_results_mounted/episode_000028.parquet"
evaluation_json = json.loads(official_json_path.read_text())
evaluation_table = pq.read_table(official_parquet_path)
eval_rows = evaluation_table.to_pylist()
assert evaluation_table.num_rows == frames*2 and all(r["environment_valid"] for r in eval_rows)
achieved_q = np.array([[r[f"achieved_quaternion_{c}"] for c in "xyzw"] for r in eval_rows], dtype=np.float64)
reference_q = np.array([[r[f"reference_quaternion_{c}"] for c in "xyzw"] for r in eval_rows], dtype=np.float64)
runtime = dict(template["runtime"])
runtime.update({
    "native_loader_image":"v2d_task_library_loader:7eb9721",
    "native_loader_image_sha256":"sha256:4e6a014956dfc73265d5c75621be159a22d350c12c5970f260da3ba9391b382c",
    "robotic_grounding_image":"robotic-grounding:latest",
    "robotic_grounding_image_sha256":"sha256:87d68718502e873898e7df2cc430a50197974349c4e37ab0ae7891db5f0ea2a9",
    "retarget_python":"3.11.13",
    "retarget_numpy":"1.26.4",
    "retarget_pyarrow":"20.0.0",
    "retarget_pyarrow_override_path":"native/compat_pyarrow20",
})
runtime["official_evaluation"] = {
    **runtime["official_evaluation"],
    "official_one_world":True,
    "checkpoint_sha256":evaluation_json["checkpoint_sha256"],
}
training = {
    "algorithm":"FlashSAC",
    "total_environment_steps":65536,
    "checkpoint_interval_environment_steps":32768,
    "updates_per_collection":1.0,
    "world_count":128,
    "sim_fps":20.0,
    "physics_fps":100.0,
    "njmax":4096,
    "nconmax":512,
    "seed":42,
    "support":False,
    "decompose_objects":True,
    "curriculum":"reference",
    "curriculum_thresholds":[2000,3500,5000,6500,8000,9500,11000,12500,14000,15500],
    "unassisted_final_stage_start":15500,
    "replay":{"capacity":131072,"minimum_size":8192,"batch_size":512},
    "network":{"actor_hidden_dim":256,"critic_hidden_dim":512,"initial_normalized_std":0.1,"log_std_max":-1.8971199848858813},
    "overflow_warning_count":sum("overflow" in line.lower() for line in train_text.splitlines()),
    "checkpoint_map":checkpoint_map_path.name,
    "policy_32768_sha256":sha(checkpoint_dir/"policy_32768.safetensors"),
    "policy_65536_sha256":sha(checkpoint_dir/"policy_65536.safetensors"),
    "state_32768_sha256":sha(checkpoint_dir/"state_32768.safetensors"),
    "state_65536_sha256":sha(checkpoint_dir/"state_65536.safetensors"),
}
asset_root = native / "object_assets"
assets = {}
for directory in (asset_root/"meshes/ego_recon", asset_root/"urdfs/ego_recon"):
    for p in sorted(directory.iterdir()):
        if p.is_file():
            assets[str(p.relative_to(native))] = sha(p)
artifact_files = [
    "loader_plastic_dish_rack.log","loader_white_pot.log","retarget_compat.log","run_retarget_e28.sh",
    "run_train_e28.sh","resolved_train_e28.yaml","train_njmax4096.log","run_official_eval_e28.sh",
    "official_eval.log","source_fps_correction.json","episode_000028_manifest_corrections.json",
    "episode_000028_retargeted_source.parquet","episode_000028.parquet","episode28_checkpoint_65536.json",
    "official_results_mounted/episode_000028.json","official_results_mounted/episode_000028.parquet",
    "checkpoint_njmax4096/policy_32768.safetensors","checkpoint_njmax4096/policy_65536.safetensors",
    "checkpoint_njmax4096/state_32768.safetensors","checkpoint_njmax4096/state_65536.safetensors",
    "mesh_budget/plastic_dish_rack/mesh_budget_report.json","mesh_budget/plastic_dish_rack/mesh_budget.obj",
    "mesh_budget/white_pot/mesh_budget_report.json","mesh_budget/white_pot/mesh_budget.obj",
]
artifacts = {p:sha(native/p) for p in artifact_files if (native/p).is_file()}
provenance = {
    "episode_id":"episode_000028",
    "created_at_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
    "source_video":{"path":source_video_rel,"sha256":sha(video),"fps":int(fps),"frames":frames,"avg_frame_rate":probe["avg_frame_rate"]},
    "source_code":template["source_code"],
    "runtime":runtime,
    "objects":objects,
    "native_reference":{
        "path":"episode_000028.parquet","sha256":sha(ref_path),
        "source_retargeted_parquet_sha256":sha(source_ref_path),
        "fps":20,"frames":frames,"object_order":["plastic_dish_rack","white_pot"],
        "asset_root_in_runtime":"/workspace/object_assets",
    },
    "training":training,
    "evaluation":{
        "official_one_world":True,"rows":evaluation_table.num_rows,"frames":frames,
        "finite":True,"environment_valid":all(r["environment_valid"] for r in eval_rows),
        "quaternion_max_error_achieved":float(np.max(np.abs(np.linalg.norm(achieved_q,axis=1)-1.0))),
        "quaternion_max_error_reference":float(np.max(np.abs(np.linalg.norm(reference_q,axis=1)-1.0))),
        "reset_mode":evaluation_json["reset_mode"],"world_count":evaluation_json["world_count"],
        "metrics":evaluation_json["metrics"],"reference_tracking":evaluation_json["reference_tracking"],
        "json_sha256":sha(official_json_path),"parquet_sha256":sha(official_parquet_path),
        "checkpoint_sha256":evaluation_json["checkpoint_sha256"],
    },
    "assets":assets,
    "artifacts":artifacts,
    "notes":[
        "Object geometry was reused from verified reconstructed physical objects in episodes 000014 and 000001, with fresh per-video FoundationPose metric registration/scales for episode 000028.",
        "The selected bundles use GeoCalib and gravity alignment. Their stale source flags were corrected from the embedded NPZ transforms and calibration files; raw manifests are preserved and hashed.",
        "The pinned loader hard-codes 30 FPS. The loader outputs were preserved unchanged, and corrected copies use 20 FPS after ffprobe confirmed 20/1 and 220 frames; no temporal resampling was applied.",
        "The cached IsaacLab retarget runtime combines NumPy 1.26.4 with PyArrow 26. PyArrow 20.0.0 was installed only under this episode's native directory to run the upstream retargeter.",
        "The learned rollout is structurally valid but has low tracking metrics and zero success rates; it is not evidence of successful task completion.",
    ],
}
out=native/"episode_000028_provenance.json"
out.write_text(json.dumps(provenance,indent=2,sort_keys=True)+"\n")
print("PROVENANCE",out,"sha256",sha(out),"bytes",out.stat().st_size)
print("METRICS",json.dumps(provenance["evaluation"]["metrics"],sort_keys=True))
print("TRAINING_OVERFLOW_LINES",training["overflow_warning_count"])

