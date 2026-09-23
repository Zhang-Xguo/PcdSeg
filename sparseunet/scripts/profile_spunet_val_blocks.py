#!/usr/bin/env python3
"""Profile the standard Gridnethd fragment/TTA test path on one GPU."""
from __future__ import annotations

import argparse
import json
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pointcept.datasets import build_dataset, collate_fn
from pointcept.models import build_model
from pointcept.utils.config import Config
from pointcept.utils.env import set_seed

from evaluate_spunet_val_fixed import evaluate


def now() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(
        "configs/stage_b_eval.py"))
    parser.add_argument("--output", type=Path, default=Path(
        "exp/spunet_stage_b_eval"))
    parser.add_argument("--weight", type=Path, help="Checkpoint override")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only for a smoke run; omit for full metrics")
    parser.add_argument("--data-root", type=Path,
                        help="Override the test tile root")
    parser.add_argument("--split", type=str,
                        help="Override the test split JSON within data-root")
    parser.add_argument("--skip-metrics", action="store_true",
                        help="Infer unlabeled full-scene tiles without tile metrics")
    parser.add_argument("--single-scale", action="store_true",
                        help="Use one identity test augmentation for faster full-scene export")
    parser.add_argument("--warmup-iters", type=int, default=0)
    parser.add_argument("--raw-points", type=int,
                        help="Track unique source points reaching the model")
    args = parser.parse_args()
    started = time.perf_counter()
    cfg = Config.fromfile(str(args.config))
    if args.weight is not None:
        cfg.weight = str(args.weight.resolve())
    if args.data_root is not None:
        cfg.data.test.data_root = str(args.data_root.resolve())
    if args.split is not None:
        cfg.data.test.split = args.split
    if args.single_scale:
        cfg.data.test.test_cfg.aug_transform = [[
            dict(type="RandomScale", scale=[1.0, 1.0])]]
    set_seed(cfg.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    result_dir = args.output / "result"
    result_dir.mkdir(exist_ok=True)
    if list(result_dir.glob("*_pred.npy")):
        raise RuntimeError(f"Result directory must be empty: {result_dir}")

    setup_start = time.perf_counter()
    model = build_model(cfg.model).cuda().eval()
    checkpoint = torch.load(cfg.weight, map_location="cpu")
    weight = OrderedDict((k[7:] if k.startswith("module.") else k, v)
                         for k, v in checkpoint["state_dict"].items())
    model.load_state_dict(weight, strict=True)
    dataset = build_dataset(cfg.data.test)
    setup_seconds = now() - setup_start

    times = {key: 0.0 for key in (
        "data_read", "preprocess", "fragment_collate", "h2d",
        "model_forward", "score_accumulate", "label_recovery", "save_prediction")}
    points = 0
    fragments = 0
    processed_voxel_instances = 0
    chunk_voxels = []
    active_source = (np.zeros(args.raw_points, dtype=np.bool_)
                     if args.raw_points is not None else None)
    n = len(dataset) if args.limit is None else min(args.limit, len(dataset))

    # Representative real fragment warm-up. Its preparation, transfer and
    # forwards are intentionally outside every reported inference timer.
    if args.warmup_iters:
        warm = dataset.prepare_test_data(min(n // 2, n - 1))
        fragment = max(warm["fragment_list"], key=lambda x: len(x["index"]))
        warm_input = collate_fn([fragment])
        for key, value in warm_input.items():
            if isinstance(value, torch.Tensor):
                warm_input[key] = value.cuda(non_blocking=True)
        torch.cuda.synchronize()
        with torch.inference_mode():
            for _ in range(args.warmup_iters):
                model(warm_input)
        torch.cuda.synchronize()
        del warm, fragment, warm_input
    torch.cuda.reset_peak_memory_stats()
    inference_started = now()

    # Run the same dataset.prepare_test_data + TTA fragments + probability
    # accumulation + inverse mapping as SemSegTester, but instrument boundaries.
    for idx in range(n):
        read_seconds = [0.0]
        original_get_data = dataset.get_data

        def timed_get_data(i):
            t = time.perf_counter()
            value = original_get_data(i)
            read_seconds[0] += time.perf_counter() - t
            return value

        dataset.get_data = timed_get_data
        t = time.perf_counter()
        try:
            data_dict = dataset.prepare_test_data(idx)
        finally:
            dataset.get_data = original_get_data
        elapsed = time.perf_counter() - t
        times["data_read"] += read_seconds[0]
        times["preprocess"] += elapsed - read_seconds[0]

        fragment_list = data_dict.pop("fragment_list")
        segment = data_dict.pop("segment")
        name = data_dict.pop("name")
        pred = torch.zeros((segment.size, cfg.data.num_classes), device="cuda")
        original_index = None
        if active_source is not None:
            original_index = np.load(
                Path(cfg.data.test.data_root) / str(cfg.data.test.split).rsplit("/", 1)[0]
                / name / "original_index.npy", mmap_mode="r") if "/" in str(cfg.data.test.split) else np.load(
                Path(cfg.data.test.data_root) / "test_final" / name / "original_index.npy",
                mmap_mode="r")
        for fragment in fragment_list:
            t = time.perf_counter()
            input_dict = collate_fn([fragment])
            times["fragment_collate"] += time.perf_counter() - t
            cpu_index = np.asarray(input_dict["index"], dtype=np.int64)
            if original_index is not None:
                active_source[np.asarray(original_index[cpu_index], dtype=np.int64)] = True
            t = now()
            for key, value in input_dict.items():
                if isinstance(value, torch.Tensor):
                    input_dict[key] = value.cuda(non_blocking=True)
            t1 = now()
            times["h2d"] += t1 - t

            index = input_dict["index"]
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            with torch.inference_mode():
                logits = model(input_dict)["seg_logits"]
            end_event.record()
            end_event.synchronize()
            times["model_forward"] += start_event.elapsed_time(end_event) / 1000.0
            t2 = time.perf_counter()
            voxels = int(logits.shape[0])
            processed_voxel_instances += voxels
            chunk_voxels.append(voxels)

            probs = F.softmax(logits, dim=-1)
            begin = 0
            for end in input_dict["offset"]:
                pred[index[begin:end], :] += probs[begin:end]
                begin = end
            torch.cuda.synchronize()
            times["score_accumulate"] += time.perf_counter() - t2
        fragments += len(fragment_list)

        t = now()
        labels = pred.argmax(dim=1).cpu().numpy()
        if "origin_segment" in data_dict:
            labels = labels[data_dict["inverse"]]
            segment = data_dict["origin_segment"]
        times["label_recovery"] += now() - t
        points += len(labels)
        t = time.perf_counter()
        np.save(result_dir / f"{name}_pred.npy", labels)
        times["save_prediction"] += time.perf_counter() - t
        print(f"{idx + 1}/{n} {name}: points={len(labels)} "
              f"fragments={len(fragment_list)}", flush=True)

    inference_seconds = now() - inference_started
    metrics_started = time.perf_counter()
    if args.limit is None and not args.skip_metrics:
        data_root = Path(cfg.data.test.data_root)
        metrics = evaluate(data_root, data_root / cfg.data.test.split, [result_dir])
        (args.output / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    else:
        metrics = None
    metrics_seconds = time.perf_counter() - metrics_started
    report = {
        "checkpoint": str(Path(cfg.weight).resolve()),
        "augmentation": "single_scale_1.0" if args.single_scale else "configured_tta",
        "split": cfg.data.test.split,
        "gpu": torch.cuda.get_device_name(),
        "tiles": n,
        "points": points,
        "fragments": fragments,
        "num_chunks": fragments,
        "processed_voxel_instances": processed_voxel_instances,
        "active_voxels": (int(active_source.sum()) if active_source is not None else None),
        "chunk_voxels": {
            "min": int(min(chunk_voxels)) if chunk_voxels else 0,
            "mean": float(np.mean(chunk_voxels)) if chunk_voxels else 0.0,
            "max": int(max(chunk_voxels)) if chunk_voxels else 0,
        },
        "warmup_iters": args.warmup_iters,
        "setup_seconds": setup_seconds,
        "inference_seconds": inference_seconds,
        "phase_seconds": times,
        "phase_sum_seconds": sum(times.values()),
        "unattributed_seconds": inference_seconds - sum(times.values()),
        "metrics_seconds": metrics_seconds,
        "total_wall_seconds": time.perf_counter() - started,
        "points_per_inference_second": points / inference_seconds,
        "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_gpu_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "metrics": metrics["overall"] if metrics else None,
        "timing_note": (
            "Single-process, one-GPU test with CUDA synchronization at phase "
            "boundaries; data_read is npy loading, preprocess includes TTA "
            "fragment generation, and score_accumulate includes softmax. "
            "Total wall time includes startup and Python overhead."
        ),
    }
    path = args.output / "profile_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
