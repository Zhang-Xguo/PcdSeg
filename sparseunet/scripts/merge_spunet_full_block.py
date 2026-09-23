#!/usr/bin/env python3
"""Backfill full-block SpUNet tile predictions into the original LAS."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import laspy
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--result-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-missing-fallback", type=int, default=10)
    ap.add_argument("--prediction-field", default="classif")
    ap.add_argument("--timing-output", type=Path)
    args = ap.parse_args()
    post_start = time.perf_counter()
    meta = json.loads(args.metadata.read_text())
    source = Path(meta["source_las"])
    n = int(meta["num_points"])
    votes = np.zeros((n, 7), dtype=np.uint8)
    coverage = np.zeros(n, dtype=np.uint8)
    for row in meta["chunks"]:
        name = row["chunk"]
        idx = np.load(args.data_root / "test_final" / name / "original_index.npy", mmap_mode="r")
        pred = np.load(args.result_dir / f"{name}_pred.npy", mmap_mode="r")
        if len(idx) != len(pred) or len(idx) != row["points"]:
            raise ValueError(f"{name}: tile index/prediction count mismatch")
        if np.any(idx >= n) or np.any(pred > 6):
            raise ValueError(f"{name}: invalid index or class")
        np.add.at(votes, (idx, pred), 1)
        np.add.at(coverage, idx, 1)
    missing = int(np.count_nonzero(coverage == 0))
    if missing > args.max_missing_fallback:
        raise ValueError(f"{missing} original points have no tile prediction")
    if missing:
        missing_idx = np.flatnonzero(coverage == 0)
        covered_idx = np.flatnonzero(coverage != 0)
        pos = np.searchsorted(covered_idx, missing_idx)
        left = covered_idx[np.maximum(pos - 1, 0)]
        right = covered_idx[np.minimum(pos, len(covered_idx) - 1)]
        nearest = np.where(missing_idx - left <= right - missing_idx, left, right)
        votes[missing_idx] = votes[nearest]
    labels = votes.argmax(axis=1).astype(np.uint8)
    with laspy.open(source) as reader:
        if reader.header.point_count != n:
            raise ValueError("Original LAS point count changed")
        original_dims = list(reader.header.point_format.dimension_names)
    las = laspy.read(source)
    if args.prediction_field in original_dims:
        raise ValueError(f"Original LAS already has {args.prediction_field}; refusing to overwrite it")
    las.add_extra_dim(laspy.ExtraBytesParams(
        name=args.prediction_field, type=np.uint8,
        description="SpUNet 7-class prediction 0-6"))
    las[args.prediction_field] = labels
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_name(args.output.stem + ".partial.las")
    post_seconds = time.perf_counter() - post_start
    write_start = time.perf_counter()
    las.write(temp)
    write_seconds = time.perf_counter() - write_start
    del las
    # Read back and compare every original dimension, including RGB, intensity,
    # standard classification, and any source ExtraBytes, in source point order.
    with laspy.open(source) as raw_reader, laspy.open(temp) as out_reader:
        if raw_reader.header.point_count != out_reader.header.point_count:
            raise ValueError("Output point count changed")
        if not set(original_dims).issubset(out_reader.header.point_format.dimension_names):
            raise ValueError("Output lost an original dimension")
        for raw, out in zip(raw_reader.chunk_iterator(500_000),
                            out_reader.chunk_iterator(500_000)):
            for dim in original_dims:
                if not np.array_equal(np.asarray(raw[dim]), np.asarray(out[dim])):
                    raise ValueError(f"Original dimension changed: {dim}")
            if np.any(np.asarray(out[args.prediction_field]) > 6):
                raise ValueError(f"Output {args.prediction_field} outside 0-6")
    verify_seconds = time.perf_counter() - write_start - write_seconds
    temp.replace(args.output)
    report = {
        "source": str(source), "output": str(args.output),
        "points": n, "tiles": len(meta["chunks"]),
        "boundary_fallback_points": missing, "original_dimensions_verified": original_dims,
        "class_field": args.prediction_field,
        "class_counts": np.bincount(labels, minlength=7).tolist(),
    }
    args.output.with_suffix(".audit.json").write_text(json.dumps(report, indent=2) + "\n")
    timing = {
        "T_postprocess_ms": 1000.0 * post_seconds,
        "T_write_ms": 1000.0 * write_seconds,
        "T_verify_ms": 1000.0 * verify_seconds,
    }
    if args.timing_output:
        args.timing_output.parent.mkdir(parents=True, exist_ok=True)
        args.timing_output.write_text(json.dumps(timing, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
