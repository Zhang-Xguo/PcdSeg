#!/usr/bin/env python3
"""Pure SpUNet forward benchmark on one preprocessed complete-block input."""
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from pointcept.datasets import build_dataset, collate_fn
from pointcept.models import build_model
from pointcept.utils.config import Config
from pointcept.utils.env import set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/stage_b_eval.py"))
    ap.add_argument("--weight", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--warmup-iters", type=int, default=5)
    ap.add_argument("--min-iters", type=int, default=10)
    ap.add_argument("--max-iters", type=int, default=50)
    ap.add_argument("--min-total-seconds", type=float, default=10.0)
    args = ap.parse_args()

    cfg = Config.fromfile(str(args.config))
    cfg.weight = str(args.weight.resolve())
    cfg.data.test.data_root = str(args.data_root.resolve())
    cfg.data.test.split = args.split
    cfg.data.test.test_cfg.aug_transform = [[dict(type="RandomScale", scale=[1.0, 1.0])]]
    set_seed(cfg.seed)
    model = build_model(cfg.model).cuda().eval()
    checkpoint = torch.load(cfg.weight, map_location="cpu")
    weights = OrderedDict((k[7:] if k.startswith("module.") else k, v)
                          for k, v in checkpoint["state_dict"].items())
    model.load_state_dict(weights, strict=True)
    dataset = build_dataset(cfg.data.test)
    if len(dataset) < 1:
        raise ValueError("Empty micro split")
    cpu_inputs = []
    names = []
    for idx in range(len(dataset)):
        prepared = dataset.prepare_test_data(idx)
        names.append(prepared["name"])
        cpu_inputs.extend(collate_fn([fragment]) for fragment in prepared["fragment_list"])
    processed = sum(int(x["index"].numel()) for x in cpu_inputs)

    def one_pass() -> float:
        total_ms = 0.0
        with torch.inference_mode():
            for cpu in cpu_inputs:
                gpu = {key: (value.cuda(non_blocking=True)
                             if isinstance(value, torch.Tensor) else value)
                       for key, value in cpu.items()}
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record(); model(gpu); end.record(); end.synchronize()
                total_ms += start.elapsed_time(end)
        return total_ms

    for _ in range(args.warmup_iters):
        one_pass()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    while len(samples) < args.max_iters:
        samples.append(one_pass())
        if len(samples) >= args.min_iters and sum(samples) >= 1000.0 * args.min_total_seconds:
            break
    arr = np.asarray(samples, dtype=np.float64)
    report = {
        "block": names[0].split("__x", 1)[0], "spatial_tiles": len(dataset),
        "warmup_iters": args.warmup_iters,
        "iterations": len(samples), "processed_voxel_instances": processed,
        "forward_ms": samples,
        "statistics": {"mean": float(arr.mean()), "std": float(arr.std()),
                       "p50": float(np.percentile(arr, 50)),
                       "p90": float(np.percentile(arr, 90)),
                       "p95": float(np.percentile(arr, 95)),
                       "min": float(arr.min()), "max": float(arr.max())},
        "voxel_mpts_per_s_mean": processed / arr.mean() / 1000.0,
        "peak_gpu_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_gpu_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3,
        "model_eval": not model.training, "inference_mode": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
