#!/usr/bin/env python3
"""Verify that a model prediction is aligned one-to-one with a raw block."""
import argparse
import json
from pathlib import Path

import laspy
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args()
    report = {"raw": str(args.raw.resolve()),
              "prediction": str(args.prediction.resolve())}
    coordinate_match = True
    max_error = 0.0
    with laspy.open(args.raw) as raw, laspy.open(args.prediction) as pred:
        report["raw_points"] = int(raw.header.point_count)
        report["prediction_points"] = int(pred.header.point_count)
        report["point_count_match"] = report["raw_points"] == report["prediction_points"]
        report["prediction_dimensions"] = list(pred.header.point_format.dimension_names)
        if report["point_count_match"]:
            raw_iter = raw.chunk_iterator(args.chunk_size)
            pred_iter = pred.chunk_iterator(args.chunk_size)
            for raw_points, pred_points in zip(raw_iter, pred_iter):
                delta = np.max(np.abs(np.column_stack((raw_points.x, raw_points.y, raw_points.z)) -
                                      np.column_stack((pred_points.x, pred_points.y, pred_points.z))))
                max_error = max(max_error, float(delta))
                if delta > args.atol:
                    coordinate_match = False
        else:
            coordinate_match = False
    report["coordinate_order_match"] = coordinate_match
    report["max_xyz_abs_error"] = max_error
    report["valid"] = report["point_count_match"] and coordinate_match
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()
