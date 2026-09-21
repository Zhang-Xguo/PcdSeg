#!/usr/bin/env python3
"""Export LitePT validation predictions and references as colored LAS tiles."""
import argparse
import json
from pathlib import Path
import laspy
import numpy as np

PALETTE = np.asarray([
    [243, 214, 171], [255, 140, 0], [243, 238, 0], [46, 125, 50],
    [190, 100, 153], [190, 160, 90], [70, 140, 190],
], dtype=np.uint16)

def write_las(tile, pred, ref, out):
    coord = np.load(tile / "coord.npy").astype(np.float64)
    color = np.load(tile / "color.npy").astype(np.uint16)
    pred = np.asarray(pred).reshape(-1).astype(np.uint8)
    ref = np.asarray(ref).reshape(-1).astype(np.uint8)
    if not (len(coord) == len(color) == len(pred) == len(ref)):
        raise RuntimeError(f"point count mismatch: {tile}")
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.asarray([0.001, 0.001, 0.001])
    header.offsets = np.floor(coord.min(axis=0))
    header.add_extra_dim(laspy.ExtraBytesParams(name="reference7", type=np.uint8))
    header.add_extra_dim(laspy.ExtraBytesParams(name="prediction7", type=np.uint8))
    pts = laspy.ScaleAwarePointRecord.zeros(len(coord), header=header)
    pts.x, pts.y, pts.z = coord.T
    pts.red, pts.green, pts.blue = (PALETTE[pred] * 257).T
    pts.prediction7 = pred
    pts.reference7 = ref
    las = laspy.LasData(header, pts)
    out.parent.mkdir(parents=True, exist_ok=True)
    las.write(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--prediction-root", type=Path, required=True)
    ap.add_argument("--manifest", default="val_fixed_las_blocks.json")
    ap.add_argument("--output-root", type=Path, required=True)
    args = ap.parse_args()
    names = json.loads((args.data_root / args.manifest).read_text())
    for rel in names:
        tile = args.data_root / rel
        name = tile.name
        pred = np.load(args.prediction_root / f"{name}_pred.npy")
        ref = np.load(tile / "segment.npy")
        write_las(tile, pred, ref, args.output_root / f"{name}_prediction.las")
        write_las(tile, ref, ref, args.output_root / f"{name}_reference.las")
    print(f"exported {len(names)} tiles to {args.output_root}")

if __name__ == "__main__":
    main()
