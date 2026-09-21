#!/usr/bin/env python3
"""Stream a large LAS into overlapping Pointcept inference tiles.

Each source point is written to at most four 20 m / 10 m XY windows without
loading the whole cloud or scanning it once per window.  Every tile carries an
``original_index.npy`` for exact voting back to source LAS order.
"""

import argparse
import json
from pathlib import Path

import laspy
import numpy as np


RECORD_DTYPE = np.dtype([
    ("coord", "<f4", (3,)),
    ("color", "u1", (3,)),
    ("index", "<u8"),
])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--chunk-size", type=float, default=20.0)
    p.add_argument("--stride", type=float, default=10.0)
    p.add_argument("--read-points", type=int, default=2_000_000)
    return p.parse_args()


def append_records(path, coord, color, index):
    rec = np.empty(len(index), dtype=RECORD_DTYPE)
    rec["coord"] = coord
    rec["color"] = color
    rec["index"] = index
    with open(path, "ab") as f:
        rec.tofile(f)


def convert_color(points, has_rgb):
    if not has_rgb:
        return np.zeros((len(points), 3), dtype=np.uint8)
    rgb = np.column_stack((points.red, points.green, points.blue)).astype(np.uint32)
    if rgb.max(initial=0) <= 255:
        return rgb.astype(np.uint8)
    return (rgb >> 8).astype(np.uint8)


def finalize_bin(bin_path, out_dir, copy_block=2_000_000):
    n = bin_path.stat().st_size // RECORD_DTYPE.itemsize
    raw = np.memmap(bin_path, mode="r", dtype=RECORD_DTYPE, shape=(n,))
    out_dir.mkdir(parents=True, exist_ok=False)
    coord = np.lib.format.open_memmap(
        out_dir / "coord.npy", mode="w+", dtype=np.float32, shape=(n, 3)
    )
    color = np.lib.format.open_memmap(
        out_dir / "color.npy", mode="w+", dtype=np.uint8, shape=(n, 3)
    )
    index = np.lib.format.open_memmap(
        out_dir / "original_index.npy", mode="w+", dtype=np.int64, shape=(n,)
    )
    segment = np.lib.format.open_memmap(
        out_dir / "segment.npy", mode="w+", dtype=np.int64, shape=(n,)
    )
    for s in range(0, n, copy_block):
        e = min(s + copy_block, n)
        coord[s:e] = raw["coord"][s:e]
        color[s:e] = raw["color"][s:e]
        index[s:e] = raw["index"][s:e]
        segment[s:e] = 255
    for arr in (coord, color, index, segment):
        arr.flush()
    del raw, coord, color, index, segment
    bin_path.unlink()
    return int(n)


def main():
    args = parse_args()
    src = Path(args.input).resolve()
    root = Path(args.output_root).resolve()
    split_root = root / "test_final"
    spool = root / "_spool"
    if split_root.exists() or spool.exists():
        raise RuntimeError(f"Output root is not clean: {root}")
    split_root.mkdir(parents=True)
    spool.mkdir(parents=True)

    with laspy.open(src) as reader:
        h = reader.header
        mins = np.asarray(h.mins, dtype=np.float64)
        maxs = np.asarray(h.maxs, dtype=np.float64)
        span = maxs - mins
        n_source = int(h.point_count)
        dims = set(h.point_format.dimension_names)
        has_rgb = {"red", "green", "blue"}.issubset(dims)
        nx = max(0, int(np.ceil(max(0.0, span[0] - args.chunk_size) / args.stride)))
        ny = max(0, int(np.ceil(max(0.0, span[1] - args.chunk_size) / args.stride)))
        cursor = 0
        touched = set()
        for points in reader.chunk_iterator(args.read_points):
            n = len(points)
            xyz = np.column_stack((points.x, points.y, points.z)).astype(np.float64)
            coord = (xyz - mins).astype(np.float32)
            color = convert_color(points, has_rgb)
            source_index = np.arange(cursor, cursor + n, dtype=np.uint64)
            cursor += n
            # A float32-rounded point on the global maximum can fall one grid
            # index beyond the last valid window.  Clipping to ``n + 1`` keeps
            # the existing two-window assignment while ensuring that the last
            # window receives those boundary points.
            qx = np.clip(
                np.floor(coord[:, 0] / args.stride).astype(np.int32), 0, nx + 1
            )
            qy = np.clip(
                np.floor(coord[:, 1] / args.stride).astype(np.int32), 0, ny + 1
            )
            for dix in (0, -1):
                ix = qx + dix
                valid_x = (ix >= 0) & (ix <= nx)
                x0 = ix.astype(np.float64) * args.stride
                valid_x &= (coord[:, 0] >= x0) & (coord[:, 0] <= x0 + args.chunk_size)
                for diy in (0, -1):
                    iy = qy + diy
                    valid = valid_x & (iy >= 0) & (iy <= ny)
                    y0 = iy.astype(np.float64) * args.stride
                    valid &= (coord[:, 1] >= y0) & (coord[:, 1] <= y0 + args.chunk_size)
                    if not valid.any():
                        continue
                    keys = ix[valid].astype(np.int64) * (ny + 1) + iy[valid]
                    order = np.argsort(keys, kind="stable")
                    keys = keys[order]
                    src_pos = np.flatnonzero(valid)[order]
                    starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1]
                    ends = np.r_[starts[1:], len(keys)]
                    for s, e in zip(starts, ends):
                        key = int(keys[s])
                        tile_ix, tile_iy = divmod(key, ny + 1)
                        name = f"{src.stem}__x{tile_ix:04d}_y{tile_iy:04d}"
                        pos = src_pos[s:e]
                        append_records(
                            spool / f"{name}.bin", coord[pos], color[pos], source_index[pos]
                        )
                        touched.add(name)
            print(f"\rstreamed {cursor:,}/{n_source:,}", end="", flush=True)
    print()
    if cursor != n_source:
        raise RuntimeError(f"Read count mismatch: {cursor} != {n_source}")

    chunks = []
    total_tile_points = 0
    for i, name in enumerate(sorted(touched), 1):
        bin_path = spool / f"{name}.bin"
        n = finalize_bin(bin_path, split_root / name)
        total_tile_points += n
        parts = name.rsplit("__x", 1)[1]
        ix = int(parts.split("_y")[0]); iy = int(parts.split("_y")[1])
        chunks.append(dict(
            chunk=name, scene=src.stem, points=n,
            x0=ix * args.stride, x1=ix * args.stride + args.chunk_size,
            y0=iy * args.stride, y1=iy * args.stride + args.chunk_size,
        ))
        if i == 1 or i % 100 == 0 or i == len(touched):
            print(f"finalized {i:,}/{len(touched):,} tiles")
    spool.rmdir()

    meta = dict(
        scene=src.stem, source_las=str(src), num_points=n_source,
        min_xyz=mins.tolist(), max_xyz=maxs.tolist(), span_xyz=span.tolist(),
        chunk_size=args.chunk_size, stride=args.stride,
        tile_point_copies=total_tile_points,
        chunks=chunks,
    )
    meta_path = root / f"{src.stem}_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"tiles={len(chunks):,} copies={total_tile_points:,} ratio={total_tile_points/n_source:.3f}")
    print(f"metadata={meta_path}")


if __name__ == "__main__":
    main()
