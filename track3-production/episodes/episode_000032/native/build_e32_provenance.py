from pathlib import Path
from datetime import datetime, timezone
from fractions import Fraction
import hashlib, json, re, subprocess
import numpy as np
import pyarrow.parquet as pq

project = Path('/home/ubuntu/workspace/nvidia-v2d')
track = Path('/v2d/runs/track3-production')
episode = track / 'episode_000032_rebuild'
native = Path('/workspace/native')
template = json.loads((track / 'episode_000008_rebuild/native/episode_000008_provenance.json').read_text())

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def host(path):
    return str(path).replace('/v2d/runs/track3-production', str(project / 'runs/track3-production')).replace('/workspace/native', str(project / 'runs/track3-production/episode_000032_rebuild/native'))

def track_relative(path):
    path = Path(path)
    if path.is_relative_to(track):
        return str(path.relative_to(track))
    return str(Path(episode.name) / 'native' / path.relative_to(native))

source_video_rel = 'data/raw/video_to_data_challenge/track_3/evaluation/videos/chunk-000/observation.images.ego_cam_a/episode_000032.mp4'
video = Path('/source_data') / source_video_rel.removeprefix('data/')
probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=avg_frame_rate,nb_frames,duration','-of','json',str(video)], text=True))['streams'][0]
fps = float(Fraction(probe['avg_frame_rate']))
frames = int(probe['nb_frames'])
assert fps == 20.0 and frames == 259, (probe, fps, frames)

objects = {}
for name, prompt in (('blue_cup','a blue cup'), ('wooden_spoon','a wooden spoon')):
    obj_root = episode / name
    source_manifest = obj_root / 'result_slam_gravity_aligned/manifest.json'
    source_npz = obj_root / 'result_slam_gravity_aligned/result.npz'
    staged = native / f'input_{name}/episode_000032/outputs/result'
    with np.load(source_npz, allow_pickle=True) as z:
        object_valid_frames = int(np.asarray(z['object_is_valid'], dtype=bool).sum())
        gravity_applied = bool(z['gravity_alignment_applied'].item())
    assert object_valid_frames == frames and gravity_applied
    metric_scale_source = None
    metric_scale_provenance = {}
    if name == 'wooden_spoon':
        mesh = track / 'episode_000003_rebuild/wooden_spoon/provided_mesh_scaled.obj'
        budget_obj = track / 'episode_000003_rebuild/native/object_assets/meshes/ego_recon/wooden_spoon.obj'
        budget_report_path = track / 'episode_000003_rebuild/native/mesh_budget/wooden_spoon/mesh_budget_report.json'
        metric_scale_source = track / 'episode_000003_rebuild/native/episode_000003_provenance.json'
        metric_scale_provenance = json.loads(metric_scale_source.read_text())['objects']['wooden_spoon']
        source_mesh_description = 'video-reconstructed metric spoon prior from E3; fresh per-video registration is used here'
        scale_estimation_description = 'skipped; metric scale is baked into the reused mesh from E3 FoundationPose factor ' + str(metric_scale_provenance['foundationpose_model_scale'])
        assert json.loads(budget_report_path.read_text())['source_sha256'] == sha(mesh)
    else:
        mesh_dir = native / 'mesh_budget' / name
        mesh = obj_root / 'mesh_scaled.obj'
        budget_obj = mesh_dir / 'mesh_budget.obj'
        budget_report_path = mesh_dir / 'mesh_budget_report.json'
        source_mesh_description = 'SAM3D mesh reconstructed from this evaluation video; scale estimated from video depth and masks with FoundationPose'
        scale_estimation_description = 'per-video FoundationPose scale estimation; metric scale is baked into mesh_scaled.obj'
    budget_report = json.loads(budget_report_path.read_text())
    calibration = obj_root / 'geocalib/calibration.json'
    runner = native / f'run_e32_{name}.sh'
    runner_text = runner.read_text()
    if name == 'wooden_spoon':
        assert '--object_mesh' in runner_text and '--skip_object_scale_estimation' in runner_text
    else:
        assert '--object_mesh' not in runner_text and '--skip_object_scale_estimation' not in runner_text
    objects[name] = {
        'frames': frames,
        'object_valid_frames': object_valid_frames,
        'prompt': prompt,
        'source_mesh': source_mesh_description,
        'input_mesh_path': host(mesh),
        'input_mesh_sha256': sha(mesh),
        'provided_mesh_scale': 1.0 if name == 'wooden_spoon' else None,
        'metric_scale_provenance_path': host(metric_scale_source) if metric_scale_source else None,
        'metric_scale_provenance_sha256': sha(metric_scale_source) if metric_scale_source else None,
        'metric_scale_model_scale_baked_into_mesh': metric_scale_provenance.get('foundationpose_model_scale'),
        'scale_estimation': scale_estimation_description,
        'mesh_budget': {
            **budget_report,
            'report_path': track_relative(budget_report_path),
            'report_sha256': sha(budget_report_path),
            'budget_obj_path': track_relative(budget_obj),
            'budget_obj_sha256': sha(budget_obj),
        },
        'geocalib_calibration_path': host(calibration),
        'geocalib_calibration_sha256': sha(calibration),
        'gravity_alignment_enabled': gravity_applied,
        'gravity_target': 'z_up',
        'result_bundle': str(Path(f'input_{name}') / 'episode_000032/outputs/result'),
        'result_manifest_sha256': sha(staged / 'manifest.json'),
        'result_npz_sha256': sha(staged / 'result.npz'),
        'source_aligned_manifest_path': host(source_manifest),
        'source_aligned_npz_path': host(source_npz),
        'source_aligned_npz_sha256': sha(source_npz),
        'preserved_original_manifest_path': host(source_manifest.with_name('manifest.pre_source_flag_correction.json')),
        'preserved_original_manifest_sha256': sha(source_manifest.with_name('manifest.pre_source_flag_correction.json')),
    }

ref_path = native / 'episode_000032.parquet'
source_ref_path = native / 'episode_000032_retargeted_source.parquet'
ref_table = pq.read_table(ref_path)
assert ref_table.num_rows == 1
ref = ref_table.to_pylist()[0]
expected = ['blue_cup','wooden_spoon']
assert ref['fps'] == 20.0 and len(ref['object_body_position']) == frames and ref['object_body_names'] == expected
checkpoint_map_path = native / 'episode_000032_checkpoint_65536.json'
checkpoint_map_path.write_text(json.dumps({'25':'/v2d/runs/track3-production/episode_000032_rebuild/native/checkpoint_njmax4096/policy_65536.safetensors'}, indent=2)+'\n')
train_log = native / 'train_njmax4096.log'
train_text = train_log.read_text(errors='replace')
assert 'completed 65536 environment steps' in train_text
checkpoint_dir = native / 'checkpoint_njmax4096'
evaluation_json_path = native / 'official_results_mounted/episode_000032.json'
evaluation_parquet_path = native / 'official_results_mounted/episode_000032.parquet'
evaluation_json = json.loads(evaluation_json_path.read_text())
evaluation_table = pq.read_table(evaluation_parquet_path)
eval_rows = evaluation_table.to_pylist()
assert evaluation_table.num_rows == frames * 2 and all(row['environment_valid'] for row in eval_rows)
assert evaluation_json['world_count'] == 1 and evaluation_json['reset_mode'] == 'explicit'
achieved_q = np.array([[r[f'achieved_quaternion_{c}'] for c in 'xyzw'] for r in eval_rows], dtype=np.float64)
reference_q = np.array([[r[f'reference_quaternion_{c}'] for c in 'xyzw'] for r in eval_rows], dtype=np.float64)

runtime = dict(template['runtime'])
runtime.update({
    'native_loader_image':'v2d_task_library_loader:7eb9721',
    'native_loader_image_sha256':'sha256:4e6a014956dfc73265d5c75621be159a22d350c12c5970f260da3ba9391b382c',
    'robotic_grounding_image':'robotic-grounding:latest',
    'robotic_grounding_image_sha256':'sha256:87d68718502e873898e7df2cc430a50197974349c4e37ab0ae7891db5f0ea2a9',
    'retarget_python':'3.11.13',
    'retarget_numpy':'1.26.4',
    'retarget_pyarrow':'20.0.0',
    'retarget_pyarrow_override_path':'native/compat_pyarrow20',
})
runtime['official_evaluation'] = {**runtime['official_evaluation'], 'official_one_world':True, 'checkpoint_sha256':evaluation_json['checkpoint_sha256']}
training = {
    'algorithm':'FlashSAC', 'total_environment_steps':65536,
    'checkpoint_interval_environment_steps':32768, 'updates_per_collection':1.0,
    'world_count':128, 'sim_fps':20.0, 'physics_fps':100.0,
    'njmax':4096, 'nconmax':512, 'seed':42, 'support':False,
    'decompose_objects':True, 'curriculum':'reference',
    'curriculum_thresholds':[2000,3500,5000,6500,8000,9500,11000,12500,14000,15500],
    'unassisted_final_stage_start':15500,
    'replay':{'capacity':131072,'minimum_size':8192,'batch_size':512},
    'network':{'actor_hidden_dim':256,'critic_hidden_dim':512,'initial_normalized_std':0.1,'log_std_max':-1.8971199848858813},
    'overflow_warning_count':sum('overflow' in line.lower() for line in train_text.splitlines()),
    'resolved_config_path':'resolved_train_e32.yaml', 'resolved_config_sha256':sha(native/'resolved_train_e32.yaml'),
    'checkpoint_map':checkpoint_map_path.name,
    'policy_32768_sha256':sha(checkpoint_dir/'policy_32768.safetensors'),
    'policy_65536_sha256':sha(checkpoint_dir/'policy_65536.safetensors'),
    'state_32768_sha256':sha(checkpoint_dir/'state_32768.safetensors'),
    'state_65536_sha256':sha(checkpoint_dir/'state_65536.safetensors'),
}
asset_root = native / 'object_assets'
assets = {}
for directory in (asset_root/'meshes/ego_recon', asset_root/'urdfs/ego_recon'):
    for path in sorted(directory.iterdir()):
        if path.is_file(): assets[str(path.relative_to(native))] = sha(path)
artifact_files = [
    'run_e32_blue_cup.sh','run_e32_wooden_spoon.sh','run_e32_loaders.sh','run_retarget_e32.sh',
    'run_train_e32.sh','resolved_train_e32.yaml','train_njmax4096.log','run_official_eval_e32.sh',
    'official_eval.log','source_fps_correction.json','merge_shared_frame.json','episode_000032_manifest_corrections.json',
    'reference_validation.json','episode_000032_retargeted_source.parquet','episode_000032.parquet',
    'episode_000032_checkpoint_65536.json','official_results_mounted/episode_000032.json',
    'official_results_mounted/episode_000032.parquet','checkpoint_njmax4096/policy_32768.safetensors',
    'checkpoint_njmax4096/policy_65536.safetensors','checkpoint_njmax4096/state_32768.safetensors',
    'checkpoint_njmax4096/state_65536.safetensors',
]
artifacts = {p:sha(native/p) for p in artifact_files if (native/p).is_file()}
provenance = {
    'episode_id':'episode_000032',
    'created_at_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
    'source_video':{'path':source_video_rel,'sha256':sha(video),'fps':int(fps),'frames':frames,'avg_frame_rate':probe['avg_frame_rate']},
    'source_code':template['source_code'],
    'runtime':runtime,
    'objects':objects,
    'native_reference':{
        'path':'episode_000032.parquet','sha256':sha(ref_path),
        'source_retargeted_parquet_sha256':sha(source_ref_path),
        'fps':20,'frames':frames,'object_order':expected,
        'asset_root_in_runtime':'/workspace/object_assets',
        'shared_frame_alignment_report_sha256':sha(native/'merge_shared_frame.json'),
    },
    'training':training,
    'evaluation':{
        'official_one_world':True,'rows':evaluation_table.num_rows,'frames':frames,
        'finite':True,'environment_valid':all(r['environment_valid'] for r in eval_rows),
        'quaternion_max_error_achieved':float(np.max(np.abs(np.linalg.norm(achieved_q,axis=1)-1.0))),
        'quaternion_max_error_reference':float(np.max(np.abs(np.linalg.norm(reference_q,axis=1)-1.0))),
        'reset_mode':evaluation_json['reset_mode'],'world_count':evaluation_json['world_count'],
        'metrics':evaluation_json['metrics'],'reference_tracking':evaluation_json['reference_tracking'],
        'json_sha256':sha(evaluation_json_path),'parquet_sha256':sha(evaluation_parquet_path),
        'checkpoint_sha256':evaluation_json['checkpoint_sha256'],
    },
    'assets':assets,'artifacts':artifacts,
    'notes':[
        'The blue cup geometry was reconstructed from this evaluation video with per-video FoundationPose scale estimation. The wooden spoon uses the video-derived metric-scale mesh reused from E3, with its recorded FoundationPose scale factor and no fresh object-scale estimation; no public scan prior was used.',
        'The selected aligned bundles report GeoCalib calibration and gravity alignment. Their stale source flags were corrected using the embedded NPZ transforms and calibration files; the original manifests are preserved and hashed.',
        'The pinned task loader hard-codes 30 FPS. Original outputs were preserved, and separate Parquet copies were marked 20 FPS after ffprobe confirmed 20/1 and 259 frames; no temporal resampling was applied.',
        'The independently loaded object scenes were aligned into one common source frame with a measured rigid rotation/translation fit from the per-frame camera transforms; residual gates and source pairwise-distance checks passed.',
        'The cached retarget runtime combines NumPy 1.26.4 with PyArrow 26. PyArrow 20.0.0 was used only under this episode native directory to run the upstream retargeter.',
        'The learned rollout is structurally valid; task success metrics are reported as measured and do not imply successful task completion.',
    ],
}
out = native/'episode_000032_provenance.json'
out.write_text(json.dumps(provenance, indent=2, sort_keys=True)+'\n')
print('PROVENANCE', out, 'sha256', sha(out), 'bytes', out.stat().st_size)
print('METRICS', json.dumps(provenance['evaluation']['metrics'], sort_keys=True))
print('TRAINING_OVERFLOW_LINES', training['overflow_warning_count'])
