#!/usr/bin/env python3
"""Backfill full-block SpUNet tile predictions into the original LAS."""
from __future__ import annotations

import argparse
import json
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
    args = ap.parse_args()
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
    if "classif" in original_dims:
        raise ValueError("Original LAS already has classif; refusing to overwrite it")
    las.add_extra_dim(laspy.ExtraBytesParams(
        name="classif", type=np.uint8, description="SpUNet 7-class prediction 0-6"))
    las.classif = labels
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_name(args.output.stem + ".partial.las")
    las.write(temp)
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
            if np.any(np.asarray(out["classif"]) > 6):
                raise ValueError("Output classif outside 0-6")
    temp.replace(args.output)
    report = {
        "source": str(source), "output": str(args.output),
        "points": n, "tiles": len(meta["chunks"]),
        "boundary_fallback_points": missing, "original_dimensions_verified": original_dims,
        "class_field": "classif", "class_counts": np.bincount(labels, minlength=7).tolist(),
    }
    args.output.with_suffix(".audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
