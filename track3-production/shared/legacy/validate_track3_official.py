"""Validate a pinned official Track 3 rollout for one episode."""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import pyarrow.parquet as pq


def main(path: str, frames: int, names: list[str]) -> None:
    rows = pq.read_table(path).to_pylist()
    if len(rows) != frames * len(names):
        raise ValueError(f"expected {frames * len(names)} rows, got {len(rows)}")
    steps = sorted({int(row["step"]) for row in rows})
    if steps != list(range(frames)):
        raise ValueError(f"unexpected step range: {steps[:3]}..{steps[-3:]}")
    for row in rows:
        if int(row["reference_frame"]) != int(row["step"]):
            raise ValueError("reference_frame does not equal step")
        if abs(float(row["time_s"]) - float(row["step"]) / 20.0) > 1e-9:
            raise ValueError("time_s does not equal step/20")
    counts = Counter(row["object_body_name"] for row in rows)
    if counts != Counter({name: frames for name in names}):
        raise ValueError(f"unexpected body counts: {counts}")
    if not all(bool(row["environment_valid"]) for row in rows):
        raise ValueError("environment_valid is false for at least one row")
    keys = ("position_x", "position_y", "position_z", "quaternion_x", "quaternion_y", "quaternion_z", "quaternion_w")
    achieved = np.asarray([[row[f"achieved_{key}"] for key in keys] for row in rows], dtype=np.float64)
    reference = np.asarray([[row[f"reference_{key}"] for key in keys] for row in rows], dtype=np.float64)
    if not np.isfinite(achieved).all() or not np.isfinite(reference).all():
        raise ValueError("non-finite trajectory value")
    achieved_qerr = float(np.max(np.abs(np.linalg.norm(achieved[:, 3:], axis=1) - 1.0)))
    reference_qerr = float(np.max(np.abs(np.linalg.norm(reference[:, 3:], axis=1) - 1.0)))
    if achieved_qerr > 1e-4 or reference_qerr > 1e-4:
        raise ValueError(f"quaternion norm error: achieved={achieved_qerr}, reference={reference_qerr}")
    print(f"valid=true rows={len(rows)} frames={frames} bodies={dict(counts)} reference_frame=step time_s=step/20 environment_valid=true finite=true achieved_qnorm_maxerr={achieved_qerr:.3e} reference_qnorm_maxerr={reference_qerr:.3e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--names", nargs="+", required=True)
    args = parser.parse_args()
    main(args.path, args.frames, args.names)
