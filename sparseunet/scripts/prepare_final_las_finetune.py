#!/usr/bin/env python3
"""Build Pointcept tiles from a final label LAS and original sensor LAS blocks.

Labels come from final ``classif`` while colors always come from the original
sensor clouds. Point order and integer XYZ are checked block by block.
"""
import argparse
import json
import shutil
from pathlib import Path

import laspy
import numpy as np


RECORD = np.dtype([
    ("coord", "<f4", (3,)), ("color", "u1", (3,)),
    ("segment", "u1"), ("source", "u1"), ("supervision_weight", "<f4"),
])

MAP_11_TO_7 = np.asarray([0, 1, 1, 2, 3, 3, 3, 5, 5, 6, 4], dtype=np.uint8)


def rgb8(points):
    dims = set(points.point_format.dimension_names)
    if not {"red", "green", "blue"}.issubset(dims):
        return np.zeros((len(points), 3), dtype=np.uint8)
    rgb = np.column_stack((points.red, points.green, points.blue)).astype(np.uint32)
    return rgb.astype(np.uint8) if rgb.max(initial=0) <= 255 else (rgb >> 8).astype(np.uint8)


def append(path, coord, color, segment, source, supervision_weight):
    rec = np.empty(len(segment), dtype=RECORD)
    rec["coord"], rec["color"] = coord, color
    rec["segment"], rec["source"] = segment, source
    rec["supervision_weight"] = supervision_weight
    with path.open("ab") as stream:
        rec.tofile(stream)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled-las", required=True)
    parser.add_argument("--original-las", nargs="+", required=True,
                        help="Original blocks in source_block order")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--scene-name", required=True)
    parser.add_argument("--split", choices=("train_final", "val_final"), required=True)
    parser.add_argument("--tile-size", type=float, default=20.0)
    parser.add_argument("--pre-grid-size", type=float, default=.05)
    parser.add_argument("--min-points", type=int, default=3000)
    parser.add_argument("--min-manual-points", type=int, default=100)
    parser.add_argument("--read-points", type=int, default=1_000_000)
    parser.add_argument("--num-classes", type=int, choices=(7, 11), default=11)
    parser.add_argument("--manual-weight", type=float, default=1.0)
    parser.add_argument("--postprocess-weight", type=float, default=0.5)
    parser.add_argument("--pseudo-weight", type=float, default=0.25)
    args = parser.parse_args()

    labeled_path = Path(args.labeled_las).resolve()
    originals = [Path(x).resolve() for x in args.original_las]
    root = Path(args.output_root).resolve()
    split_root = root / args.split
    split_root.mkdir(parents=True, exist_ok=True)
    spool = root / f".{args.scene_name}_{args.split}_spool"
    if spool.exists():
        raise RuntimeError(f"Stale spool exists: {spool}")
    if any(x.name.startswith(args.scene_name + "__") for x in split_root.iterdir()):
        raise RuntimeError(f"Scene already prepared in {split_root}: {args.scene_name}")
    spool.mkdir(parents=True)

    with laspy.open(labeled_path) as labels:
        dims = set(labels.header.point_format.dimension_names)
        required = {"classif", "label_source", "source_block"}
        if args.num_classes == 7:
            required.add("ptv3_classif")
        if not required.issubset(dims):
            raise RuntimeError(f"Missing dimensions: {sorted(required - dims)}")
        raw_points = [laspy.open(x).header.point_count for x in originals]
        if sum(raw_points) != labels.header.point_count:
            raise RuntimeError("Original block point sum differs from labeled LAS")
        mins = np.min([laspy.open(x).header.mins for x in originals], axis=0)
        tile_stats = {}
        label_cursor = 0
        class_counts = np.zeros(args.num_classes, dtype=np.int64)
        manual_counts = np.zeros(args.num_classes, dtype=np.int64)
        supervision_counts = {"manual": 0, "postprocess": 0, "pseudo": 0}
        for block_id, original_path in enumerate(originals):
            with laspy.open(original_path) as raw:
                remaining = int(raw.header.point_count)
                while remaining:
                    n = min(args.read_points, remaining)
                    rp = raw.read_points(n)
                    lp = labels.read_points(n)
                    if len(lp) != n:
                        raise RuntimeError("Labeled LAS ended before original blocks")
                    tolerance = max(float(np.max(raw.header.scales)),
                                    float(np.max(labels.header.scales))) + 1e-9
                    xyz_error = max(
                        np.max(np.abs(np.asarray(rp.x) - np.asarray(lp.x)), initial=0),
                        np.max(np.abs(np.asarray(rp.y) - np.asarray(lp.y)), initial=0),
                        np.max(np.abs(np.asarray(rp.z) - np.asarray(lp.z)), initial=0),
                    )
                    if xyz_error > tolerance:
                        raise RuntimeError(f"XYZ/order mismatch in block {block_id}")
                    sb = np.asarray(lp.source_block, dtype=np.uint8)
                    if not np.all(sb == block_id):
                        raise RuntimeError(f"source_block mismatch in block {block_id}")
                    xyz = np.column_stack((rp.x, rp.y, rp.z)).astype(np.float64)
                    coord = (xyz - mins).astype(np.float32)
                    segment11 = np.asarray(lp.classif, dtype=np.uint8)
                    source = np.asarray(lp.label_source, dtype=np.uint8)
                    if args.num_classes == 7:
                        ptv3 = np.asarray(lp.ptv3_classif, dtype=np.uint8)
                        postprocessed = (source == 0) & (segment11 != ptv3)
                        segment = MAP_11_TO_7[segment11]
                    else:
                        postprocessed = np.zeros(len(source), dtype=bool)
                        segment = segment11
                    supervision_weight = np.full(
                        len(source), args.pseudo_weight, dtype=np.float32)
                    supervision_weight[postprocessed] = args.postprocess_weight
                    supervision_weight[source == 1] = args.manual_weight
                    color = rgb8(rp)
                    if segment.max(initial=0) >= args.num_classes:
                        raise RuntimeError("Training label outside configured class range")
                    class_counts += np.bincount(segment, minlength=args.num_classes)
                    manual_counts += np.bincount(segment[source == 1], minlength=args.num_classes)
                    supervision_counts["manual"] += int((source == 1).sum())
                    supervision_counts["postprocess"] += int(postprocessed.sum())
                    supervision_counts["pseudo"] += int(((source == 0) & ~postprocessed).sum())
                    ij = np.floor(coord[:, :2] / args.tile_size).astype(np.int32)
                    keys, inverse = np.unique(ij, axis=0, return_inverse=True)
                    for k, (ix, iy) in enumerate(keys):
                        keep = inverse == k
                        key = f"x{ix}_y{iy}"
                        path = spool / f"{key}.bin"
                        append(path, coord[keep], color[keep], segment[keep], source[keep],
                               supervision_weight[keep])
                        stat = tile_stats.setdefault(key, {"raw": 0, "manual": 0})
                        stat["raw"] += int(keep.sum())
                        stat["manual"] += int(source[keep].sum())
                    label_cursor += n
                    remaining -= n
        if label_cursor != labels.header.point_count:
            raise RuntimeError("Labeled LAS has trailing points")

    saved = []
    for bin_path in sorted(spool.glob("*.bin")):
        key = bin_path.stem
        stat = tile_stats[key]
        if stat["raw"] < args.min_points or stat["manual"] < args.min_manual_points:
            bin_path.unlink()
            continue
        raw = np.memmap(bin_path, mode="r", dtype=RECORD)
        if args.pre_grid_size > 0:
            grid = np.floor(raw["coord"] / args.pre_grid_size).astype(np.int64)
            priority = np.argsort(raw["source"] == 0, kind="stable")
            _, first = np.unique(grid[priority], axis=0, return_index=True)
            ids = priority[first]
        else:
            ids = np.arange(len(raw))
        out = split_root / f"{args.scene_name}__{key}"
        out.mkdir()
        np.save(out / "coord.npy", np.asarray(raw["coord"][ids]))
        np.save(out / "color.npy", np.asarray(raw["color"][ids]))
        np.save(out / "segment.npy", np.asarray(raw["segment"][ids], dtype=np.int64))
        np.save(out / "label_source.npy", np.asarray(raw["source"][ids]))
        np.save(out / "supervision_weight.npy",
                np.asarray(raw["supervision_weight"][ids], dtype=np.float32))
        u, n = np.unique(raw["segment"][ids], return_counts=True)
        saved.append({"tile": out.name, "raw_points": stat["raw"],
                      "manual_points": stat["manual"], "saved_points": len(ids),
                      "class_counts": {str(int(a)): int(b) for a, b in zip(u, n)}})
        del raw
        bin_path.unlink()
    shutil.rmtree(spool)
    if not saved:
        raise RuntimeError("No tile passed manual/point thresholds")
    report = {
        "scene": args.scene_name, "split": args.split,
        "labeled_las": str(labeled_path),
        "original_las": [str(x) for x in originals],
        "label_policy": "full final classif; manual and repaired key classes plus PTv3 pseudo labels",
        "color_policy": "original sensor RGB only; never visualization RGB",
        "point_order_and_xyz_verified": True,
        "num_classes": args.num_classes,
        "supervision_policy": {
            "manual": args.manual_weight, "postprocess": args.postprocess_weight,
            "pseudo": args.pseudo_weight, "raw_point_counts": supervision_counts,
        },
        "class_counts": {str(i): int(x) for i, x in enumerate(class_counts)},
        "manual_class_counts": {str(i): int(x) for i, x in enumerate(manual_counts)},
        "tiles": saved, "parameters": vars(args),
    }
    report_path = root / f"{args.scene_name}_{args.split}_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"scene": args.scene_name, "split": args.split,
                      "tiles": len(saved), "saved_points": sum(x["saved_points"] for x in saved),
                      "report": str(report_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
