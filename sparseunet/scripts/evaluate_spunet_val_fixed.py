#!/usr/bin/env python3
"""Aggregate per-point IoU, precision and recall over fixed validation tiles."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


NAMES = ("tower", "conductor", "insulator", "vegetation", "building", "ground", "other")


def evaluate(data_root: Path, split: Path, results: list[Path]) -> dict:
    rows = json.loads(split.read_text(encoding="utf-8"))
    confusion = np.zeros((7, 7), dtype=np.int64)
    total_points = 0
    ignored_points = 0
    for row in rows:
        tile = data_root / row
        name = tile.name
        matches = [p / f"{name}_pred.npy" for p in results if (p / f"{name}_pred.npy").exists()]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one prediction for {row}, found {len(matches)}")
        truth = np.load(tile / "segment.npy").reshape(-1)
        pred = np.load(matches[0]).reshape(-1)
        if pred.shape != truth.shape:
            raise ValueError(f"Point count mismatch for {row}: {len(pred)} != {len(truth)}")
        if np.any((pred < 0) | (pred > 6)):
            raise ValueError(f"Prediction outside 0..6: {row}")
        valid = (truth >= 0) & (truth <= 6)
        total_points += len(truth)
        ignored_points += int((~valid).sum())
        confusion += np.bincount(truth[valid] * 7 + pred[valid], minlength=49).reshape(7, 7)
    tp = confusion.diagonal()
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    def ratio(a, b):
        return float(a / b) if b else None
    per_class = {}
    for i, name in enumerate(NAMES):
        per_class[name] = {
            "id": i, "support": int(support[i]), "predicted": int(predicted[i]),
            "iou": ratio(tp[i], support[i] + predicted[i] - tp[i]),
            "precision": ratio(tp[i], predicted[i]),
            "recall": ratio(tp[i], support[i]),
        }
    def macro(key):
        # A class with no predictions contributes zero precision to the
        # seven-class macro average; its per-class value stays null/undefined.
        return float(np.mean([v[key] if v[key] is not None else 0.0
                              for v in per_class.values()]))
    valid_count = int(confusion.sum())
    true_positive = int(tp.sum())
    false_positive = int(predicted.sum() - true_positive)
    false_negative = int(support.sum() - true_positive)
    return {
        "tiles": len(rows), "points": total_points, "valid_points": valid_count,
        "ignored_points": ignored_points,
        "overall": {
            "accuracy_micro": ratio(true_positive, valid_count),
            "iou_macro": macro("iou"),
            "precision_macro": macro("precision"),
            "recall_macro": macro("recall"),
            "iou_micro": ratio(true_positive, true_positive + false_positive + false_negative),
            "precision_micro": ratio(true_positive, true_positive + false_positive),
            "recall_micro": ratio(true_positive, true_positive + false_negative),
        },
        "per_class": per_class,
        "confusion_matrix_gt_rows_pred_columns": confusion.tolist(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--result-dir", type=Path, required=True, nargs="+")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    report = evaluate(a.data_root, a.split, a.result_dir)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["overall"], indent=2))
    print(f"tiles={report['tiles']} valid_points={report['valid_points']} output={a.output}")


if __name__ == "__main__":
    main()
