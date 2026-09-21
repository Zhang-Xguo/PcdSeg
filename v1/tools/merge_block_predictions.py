#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import laspy
import numpy as np


CLASS_NAMES = [
    "Pylone",
    "Conductor cable",
    "Structural cable",
    "Insulator",
    "High vegetation",
    "Low vegetation",
    "Herbaceous vegetation",
    "Rock, gravel, soil",
    "Impervious soil (Road)",
    "Water",
    "Building",
]

CLASS_COLORS = np.asarray(
    [
        [243, 214, 171],  # 0 Pylone
        [70, 115, 66],    # 1 Conductor cable
        [233, 50, 239],   # 2 Structural cable
        [243, 238, 0],    # 3 Insulator
        [190, 153, 153],  # 4 High vegetation
        [214, 0, 54],     # 5 Low vegetation
        [243, 90, 171],   # 6 Herbaceous
        [70, 45, 66],     # 7 Rock/gravel/soil
        [233, 60, 239],   # 8 Road
        [243, 140, 0],    # 9 Water
        [190, 100, 153],  # 10 Building
    ],
    dtype=np.uint16,
)

CLASS_NAMES_7 = [
    "Tower", "Conductor", "Insulator", "Vegetation", "Building", "Ground", "Other"
]
CLASS_COLORS_7 = np.asarray(
    [
        [243, 214, 171],  # 0 tower: beige
        [255, 140, 0],    # 1 conductor: orange
        [243, 238, 0],    # 2 insulator: bright yellow
        [46, 125, 50],    # 3 vegetation: green
        [190, 100, 153],  # 4 building: purple-gray
        [190, 160, 90],   # 5 ground: earth yellow
        [70, 140, 190],   # 6 other: blue-gray
    ],
    dtype=np.uint16,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--metadata", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-classes", type=int, choices=(7, 11), default=11)
    parser.add_argument(
        "--max-missing-fallback", type=int, default=0,
        help="Allow at most this many uncovered boundary points; copy the vote "
             "of the nearest covered source-order point (default: fail).",
    )

    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    data_root = Path(args.data_root)
    result_dir = Path(args.result_dir)
    output_path = Path(args.output)
    class_names = CLASS_NAMES_7 if args.num_classes == 7 else CLASS_NAMES
    class_colors = CLASS_COLORS_7 if args.num_classes == 7 else CLASS_COLORS

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    source_las = Path(metadata["source_las"])
    n_points = int(metadata["num_points"])
    chunks = metadata["chunks"]

    print("=" * 100)
    print("source      :", source_las)
    print("points      :", f"{n_points:,}")
    print("chunks      :", len(chunks))
    print("result dir  :", result_dir)
    print("=" * 100)

    # 20m chunk / 10m stride:
    # 一个点通常最多被几个相邻块覆盖。
    votes = np.zeros(
        (n_points, args.num_classes),
        dtype=np.uint8,
    )

    coverage = np.zeros(
        n_points,
        dtype=np.uint8,
    )

    for i, chunk in enumerate(chunks, 1):
        name = chunk["chunk"]

        idx_path = (
            data_root
            / "test_final"
            / name
            / "original_index.npy"
        )

        pred_path = (
            result_dir
            / f"{name}_pred.npy"
        )

        if not idx_path.exists():
            raise FileNotFoundError(
                f"Missing index file: {idx_path}"
            )

        if not pred_path.exists():
            raise FileNotFoundError(
                f"Missing prediction file: {pred_path}"
            )

        idx = np.load(
            idx_path,
            mmap_mode="r",
        )

        pred = np.load(
            pred_path,
            mmap_mode="r",
        )

        idx = np.asarray(idx, dtype=np.int64)
        pred = np.asarray(pred, dtype=np.int64).reshape(-1)

        if len(idx) != len(pred):
            raise RuntimeError(
                f"{name}: index/pred mismatch: "
                f"{len(idx)} vs {len(pred)}"
            )

        if pred.min() < 0 or pred.max() >= args.num_classes:
            raise RuntimeError(
                f"{name}: unexpected prediction range "
                f"{pred.min()} ~ {pred.max()}"
            )

        np.add.at(
            votes,
            (idx, pred),
            1,
        )

        np.add.at(
            coverage,
            idx,
            1,
        )

        if i == 1 or i % 10 == 0 or i == len(chunks):
            print(
                f"[{i:3d}/{len(chunks)}] "
                f"{name}: {len(idx):,} pts"
            )

    missing = int(
        np.sum(coverage == 0)
    )

    print()
    print("coverage:")
    print("  min       :", int(coverage.min()))
    print("  max       :", int(coverage.max()))
    print("  missing   :", f"{missing:,}")

    if missing > args.max_missing_fallback:
        raise RuntimeError(
            f"{missing:,} original points were not covered "
            "by any PTv3 chunk."
        )

    if missing:
        missing_idx = np.flatnonzero(coverage == 0)
        covered_idx = np.flatnonzero(coverage != 0)
        pos = np.searchsorted(covered_idx, missing_idx)
        left = covered_idx[np.maximum(pos - 1, 0)]
        right = covered_idx[np.minimum(pos, len(covered_idx) - 1)]
        nearest = np.where(missing_idx - left <= right - missing_idx, left, right)
        votes[missing_idx] = votes[nearest]
        coverage[missing_idx] = coverage[nearest]
        print(
            f"fallback  : copied nearest covered source-order vote for "
            f"{missing:,} boundary point(s)"
        )

    final_pred = np.argmax(
        votes,
        axis=1,
    ).astype(np.uint8)

    print()
    print("Final prediction statistics:")
    print("-" * 80)

    ids, counts = np.unique(
        final_pred,
        return_counts=True,
    )

    for cid, count in zip(ids, counts):
        cid = int(cid)
        count = int(count)

        print(
            f"class {cid:2d}  "
            f"{class_names[cid]:28s} "
            f"{count:12,d} "
            f"{100.0 * count / n_points:8.3f}%"
        )

    print("-" * 80)

    print("\nReading original LAS...")
    las = laspy.read(source_las)

    if len(las.points) != n_points:
        raise RuntimeError(
            f"Original LAS point count changed: "
            f"{len(las.points)} != {n_points}"
        )

    dims = set(
        las.point_format.dimension_names
    )

    if "classif" not in dims:
        las.add_extra_dim(
            laspy.ExtraBytesParams(
                name="classif",
                type=np.uint8,
                description="PTv3 GridNet-HD predicted class",
            )
        )

    las.classif = final_pred
    las.classification = final_pred

    rgb = class_colors[final_pred]

    # GridNet官方颜色是8 bit RGB
    # LAS RGB字段是uint16
    las.red = (
        rgb[:, 0] * 257
    ).astype(np.uint16)

    las.green = (
        rgb[:, 1] * 257
    ).astype(np.uint16)

    las.blue = (
        rgb[:, 2] * 257
    ).astype(np.uint16)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nWriting LAS...")
    las.write(output_path)

    print()
    print("=" * 100)
    print("saved:", output_path)
    print("points:", f"{len(las.points):,}")
    print("=" * 100)


if __name__ == "__main__":
    main()
