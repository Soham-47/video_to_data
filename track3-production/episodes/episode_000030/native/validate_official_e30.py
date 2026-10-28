from pathlib import Path
import json
import numpy as np
import pyarrow.parquet as pq

root = Path('/workspace/native')
metrics_path = root / 'official_results_mounted/episode_000030.json'
parquet_path = root / 'official_results_mounted/episode_000030.parquet'
expected = ['mini_sweeper', 'plastic_dish_rack']
frames = 263
report = json.loads(metrics_path.read_text())
table = pq.read_table(parquet_path)
rows = table.to_pylist()
assert table.num_rows == frames * len(expected), (table.num_rows, frames, expected)
assert report['step_count'] == frames
assert report['sim_control_fps'] == 20.0
assert report['world_count'] == 1
assert report['start_frame'] == 0 and report['motion_start_frame'] == 0 and report['motion_end_frame'] == -1
assert report['reset_mode'] == 'explicit'
assert report['non_finite_worlds'] == []
assert report['metrics']['object_body_names'] == expected
assert len(rows) == frames * len(expected)
assert all(row['environment_valid'] for row in rows)
for i in range(frames):
    pair = rows[i * 2 : i * 2 + 2]
    assert [(r['step'], r['reference_frame'], r['object_body_id'], r['object_body_name']) for r in pair] == [
        (i, i, 0, expected[0]), (i, i, 1, expected[1])
    ]
    assert all(r['environment_id'] == 0 for r in pair)
    assert all(abs(r['time_s'] - i / 20.0) < 1e-8 for r in pair)
    for field in ('achieved', 'reference'):
        q = np.array([[r[f'{field}_quaternion_{c}'] for c in 'xyzw'] for r in pair], dtype=np.float64)
        p = np.array([[r[f'{field}_position_{c}'] for c in 'xyz'] for r in pair], dtype=np.float64)
        assert np.isfinite(q).all() and np.isfinite(p).all()
        assert np.max(np.abs(np.linalg.norm(q, axis=1) - 1.0)) < 1e-4
print(f'valid=true rows={len(rows)} frames={frames} bodies={expected} environment_valid=true fps=20 world_count=1')
print('metrics', json.dumps(report['metrics'], sort_keys=True))
