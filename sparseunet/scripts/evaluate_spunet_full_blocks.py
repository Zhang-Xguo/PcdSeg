#!/usr/bin/env python3
"""Evaluate full-block LAS predictions against aligned combined ground truth."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import laspy
import numpy as np


NAMES = ["tower", "conductor", "insulator", "vegetation", "building", "ground", "other"]
MAP_11_TO_7 = np.asarray([0, 1, 1, 2, 3, 3, 3, 5, 5, 6, 4], dtype=np.int16)


def add_cm(cm: np.ndarray, truth: np.ndarray, pred: np.ndarray) -> None:
    valid = (truth >= 0) & (truth < 7) & (pred >= 0) & (pred < 7)
    cm += np.bincount(7 * truth[valid] + pred[valid], minlength=49).reshape(7, 7)


def metrics(cm: np.ndarray) -> dict:
    tp = np.diag(cm).astype(np.float64)
    gt = cm.sum(1).astype(np.float64)
    predicted = cm.sum(0).astype(np.float64)
    union = gt + predicted - tp
    div = lambda a, b: np.divide(a, b, out=np.full_like(a, np.nan), where=b != 0)
    iou, precision, recall = div(tp, union), div(tp, predicted), div(tp, gt)
    valid = gt > 0
    total = float(cm.sum())
    correct = float(tp.sum())
    return {
        "points": int(total), "correct": int(correct),
        "overall_accuracy": correct / total if total else None,
        "macro_iou": float(np.nanmean(iou[valid])),
        "macro_precision": float(np.nanmean(precision[valid])),
        "macro_recall": float(np.nanmean(recall[valid])),
        "micro_iou": correct / (2 * total - correct) if total else None,
        "micro_precision": correct / total if total else None,
        "micro_recall": correct / total if total else None,
        "per_class": [{
            "id": i, "name": NAMES[i], "iou": float(iou[i]),
            "precision": float(precision[i]), "recall": float(recall[i]),
            "ground_truth_points": int(gt[i]), "predicted_points": int(predicted[i]),
            "true_positive": int(tp[i]),
        } for i in range(7)],
        "confusion_matrix_rows_gt_cols_pred": cm.tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=Path(
        "data/ours_stage_b_7class_v1/collection_manifest.json"))
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    root = args.result_root.resolve()
    manifest = json.loads(args.manifest.read_text())
    total_cm = np.zeros((7, 7), dtype=np.int64)
    manual_cm = np.zeros((7, 7), dtype=np.int64)
    scene_reports = {}

    for scene in manifest:
        if scene["split"] != "val_final":
            continue
        scene_name = scene["scene"]
        selected = {}
        for i, raw in enumerate(scene["original_las"]):
            if scene_name == "las_new_target" and i < 7:
                continue
            pred_path = root / scene_name / Path(raw).stem / "prediction.las"
            if not pred_path.exists():
                raise FileNotFoundError(pred_path)
            las = laspy.read(pred_path)
            selected[i] = {"pred": np.asarray(las.classif, dtype=np.int16), "cursor": 0,
                           "path": str(pred_path), "points": len(las.points)}
            del las
        scene_cm = np.zeros((7, 7), dtype=np.int64)
        scene_manual_cm = np.zeros((7, 7), dtype=np.int64)
        with laspy.open(scene["labeled_las"]) as reader:
            dims = set(reader.header.point_format.dimension_names)
            required = {"source_block", "classif", "label_source"}
            if not required.issubset(dims):
                raise ValueError(f"{scene_name}: missing GT fields {required - dims}")
            for points in reader.chunk_iterator(1_000_000):
                blocks = np.asarray(points.source_block, dtype=np.int64)
                truth11 = np.asarray(points["classif"], dtype=np.int16)
                if np.any((truth11 < 0) | (truth11 >= len(MAP_11_TO_7))):
                    raise ValueError(f"{scene_name}: 11-class GT outside 0-10")
                truth_all = MAP_11_TO_7[truth11]
                sources = np.asarray(points.label_source, dtype=np.int16)
                for block in np.unique(blocks):
                    block = int(block)
                    if block not in selected:
                        continue
                    mask = blocks == block
                    truth = truth_all[mask]
                    entry = selected[block]
                    start, end = entry["cursor"], entry["cursor"] + len(truth)
                    pred = entry["pred"][start:end]
                    if len(pred) != len(truth):
                        raise ValueError(f"{scene_name} block {block}: prediction exhausted")
                    add_cm(scene_cm, truth, pred)
                    manual = sources[mask] == 1
                    add_cm(scene_manual_cm, truth[manual], pred[manual])
                    entry["cursor"] = end
        blocks_report = {}
        for block, entry in selected.items():
            if entry["cursor"] != entry["points"]:
                raise ValueError(f"{scene_name} block {block}: aligned {entry['cursor']} / {entry['points']}")
            blocks_report[str(block)] = {"prediction": entry["path"], "points": entry["points"]}
        total_cm += scene_cm
        manual_cm += scene_manual_cm
        scene_reports[scene_name] = {
            "combined_ground_truth": metrics(scene_cm),
            "manual_only": metrics(scene_manual_cm), "blocks": blocks_report,
        }
        print(scene_name, scene_cm.sum(), "manual", scene_manual_cm.sum(), flush=True)

    report = {
        "schema_version": 1,
        "result_root": str(root),
        "evaluation_scope": "25 complete held-out validation blocks",
        "prediction_protocol": "single_scale_1.0",
        "ground_truth_policy": "final 11-class classif mapped to training 7-class space; manual + repaired/postprocessed + pseudo labels",
        "map_11_to_7": MAP_11_TO_7.tolist(),
        "combined_ground_truth": metrics(total_cm),
        "manual_only": metrics(manual_cm),
        "scenes": scene_reports,
    }
    output = args.output or root / "FULL_BLOCK_METRICS.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
