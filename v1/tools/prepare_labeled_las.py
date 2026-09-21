#!/usr/bin/env python3
"""Convert labeled LAS files into LitePT/Pointcept training or validation tiles."""
import argparse
import json
from pathlib import Path

import laspy
import numpy as np


def rgb8(las):
    dims = set(las.point_format.dimension_names)
    if not {"red", "green", "blue"}.issubset(dims):
        return np.zeros((len(las.points), 3), dtype=np.uint8)
    rgb = np.column_stack((las.red, las.green, las.blue)).astype(np.uint32)
    return rgb.astype(np.uint8) if rgb.max(initial=0) <= 255 else (rgb >> 8).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", default="train_final")
    parser.add_argument("--manifest", default=None,
                        help="Manifest filename; defaults to <split>.json")
    parser.add_argument("--label-field", default="classification")
    parser.add_argument("--tile-size", type=float, default=20.0)
    parser.add_argument("--min-points", type=int, default=3000)
    parser.add_argument("--ignore-label", type=int, default=255)
    parser.add_argument("--num-classes", type=int, default=7)
    args = parser.parse_args()

    split_root = args.output_root / args.split
    split_root.mkdir(parents=True, exist_ok=True)
    names = []
    for source in args.inputs:
        las = laspy.read(source)
        dims = set(las.point_format.dimension_names)
        if args.label_field not in dims:
            raise RuntimeError(f"{source}: missing label field {args.label_field!r}")
        xyz = np.column_stack((las.x, las.y, las.z)).astype(np.float64)
        coord = (xyz - xyz.min(axis=0)).astype(np.float32)
        color = rgb8(las)
        segment = np.asarray(las[args.label_field], dtype=np.int64)
        valid = (segment == args.ignore_label) | ((segment >= 0) & (segment < args.num_classes))
        if not valid.all():
            bad = np.unique(segment[~valid]).tolist()
            raise RuntimeError(f"{source}: labels outside 0..{args.num_classes - 1}: {bad}")
        tile_id = np.floor(coord[:, :2] / args.tile_size).astype(np.int64)
        keys, inverse = np.unique(tile_id, axis=0, return_inverse=True)
        for k, (ix, iy) in enumerate(keys):
            keep = np.flatnonzero(inverse == k)
            if len(keep) < args.min_points:
                continue
            name = f"{source.stem}__x{ix:04d}_y{iy:04d}"
            out = split_root / name
            if out.exists():
                raise RuntimeError(f"output tile already exists: {out}")
            out.mkdir()
            np.save(out / "coord.npy", coord[keep])
            np.save(out / "color.npy", color[keep])
            np.save(out / "segment.npy", segment[keep])
            names.append(name)
        print(f"{source}: saved {sum(n.startswith(source.stem + '__') for n in names)} tiles")

    manifest = args.output_root / (args.manifest or f"{args.split}.json")
    manifest.write_text(json.dumps(names, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"split": args.split, "tiles": len(names), "manifest": str(manifest)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
