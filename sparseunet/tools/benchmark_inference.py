#!/usr/bin/env python3
"""Reproducible end-to-end and micro benchmark for complete LAS blocks."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import laspy
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


CLASSES = ["tower", "conductor", "insulator", "vegetation", "building", "ground", "other"]
PERCENTILES = [0, 10, 20, 25, 40, 50, 60, 75, 80, 90, 95, 100]


def run(cmd, cwd, env, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        subprocess.run([str(x) for x in cmd], cwd=cwd, env=env, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    if not len(a):
        return {"count": 0, "mean": None, "std": None, "p50": None,
                "p90": None, "p95": None, "min": None, "max": None}
    return {"count": len(a), "mean": float(a.mean()), "std": float(a.std()),
            "p50": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90)),
            "p95": float(np.percentile(a, 95)), "min": float(a.min()),
            "max": float(a.max())}


def distribution(values):
    a = np.asarray(values, dtype=np.float64)
    out = {f"p{p}": float(np.percentile(a, p)) for p in PERCENTILES}
    out.update(mean=float(a.mean()), std=float(a.std()))
    return out


def regression(x, y):
    x = np.asarray(x, dtype=np.float64) / 100000.0
    y = np.asarray(y, dtype=np.float64)
    slope, intercept = np.polyfit(x, y, 1)
    pred = intercept + slope * x
    denom = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - np.sum((y - pred) ** 2) / denom if denom else float("nan")
    return {"intercept_ms": float(intercept), "slope_ms_per_100k": float(slope),
            "r2": float(r2)}


def label_field(path: Path):
    with laspy.open(path) as r:
        dims = set(r.header.point_format.dimension_names)
        field = "classif" if "classif" in dims else "classification"
    return field


def confusion(truth_path, pred_path, pred_field="pred_classif"):
    cm = np.zeros((7, 7), dtype=np.int64); ignored = 0
    truth_field = label_field(truth_path)
    with laspy.open(truth_path) as tr, laspy.open(pred_path) as pr:
        if tr.header.point_count != pr.header.point_count:
            raise ValueError("truth/prediction point count mismatch")
        for t, p in zip(tr.chunk_iterator(1_000_000), pr.chunk_iterator(1_000_000)):
            gt = np.asarray(t[truth_field], dtype=np.int16)
            pred = np.asarray(p[pred_field], dtype=np.int16)
            valid = (gt >= 0) & (gt < 7)
            ignored += int((~valid).sum())
            cm += np.bincount(7 * gt[valid] + pred[valid], minlength=49).reshape(7, 7)
    return cm, ignored, truth_field


def metric_report(cm):
    tp = np.diag(cm).astype(float); gt = cm.sum(1).astype(float); pd = cm.sum(0).astype(float)
    union = gt + pd - tp
    div = lambda a, b: np.divide(a, b, out=np.full_like(a, np.nan), where=b != 0)
    iou, precision, recall = div(tp, union), div(tp, pd), div(tp, gt)
    valid = gt > 0; total = cm.sum(); correct = tp.sum()
    micro_precision = float(correct / pd.sum())
    micro_recall = float(correct / gt.sum())
    micro_iou = float(correct / (gt.sum() + pd.sum() - correct))
    return {"points": int(total), "accuracy": float(correct / total),
            "micro_iou": micro_iou, "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "macro_iou": float(np.nanmean(iou[valid])),
            "macro_precision": float(np.nanmean(precision[valid])),
            "macro_recall": float(np.nanmean(recall[valid])),
            "per_class": [{"id": i, "name": CLASSES[i], "iou": float(iou[i]),
                           "precision": float(precision[i]), "recall": float(recall[i]),
                           "ground_truth_points": int(gt[i])} for i in range(7)],
            "confusion_matrix_rows_gt_cols_pred": cm.tolist()}


def bucket_for(value, bounds):
    for name, upper in zip(["S", "M", "L", "XL"], bounds):
        if value <= upper: return name
    return "XXL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--weight", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--gpu", default="4")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--warmup-iters", type=int, default=5)
    ap.add_argument("--smoke-limit", type=int)
    ap.add_argument("--skip-micro", action="store_true")
    ap.add_argument("--micro-gpus", default=None,
                    help="Comma-separated identical GPUs used for parallel single-GPU micro runs")
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    weight = args.weight.resolve(); inputs = sorted(args.input_dir.resolve().glob("*.las"))
    if args.smoke_limit: inputs = inputs[:args.smoke_limit]
    if not inputs or not weight.is_file(): raise ValueError("missing inputs or checkpoint")
    env = os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=args.gpu, PYTHONPATH=str(repo),
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1", PYTHONHASHSEED="42")
    partial_path = output / "benchmark_rows.partial.jsonl"
    rows = []
    if partial_path.is_file():
        rows = [json.loads(line) for line in partial_path.read_text().splitlines() if line.strip()]
    completed = {(r["round"], r["block"]) for r in rows}

    for round_id in range(1, args.rounds + 1):
        for block_index, source in enumerate(inputs, 1):
            if (round_id, source.stem) in completed:
                print(f"resume: skip round {round_id}/{args.rounds} block {block_index}/{len(inputs)} {source.stem}", flush=True)
                continue
            block_start = time.perf_counter()
            block = source.stem; work = output / "work" / f"round_{round_id}" / block
            data = work / "data"; inference = work / "inference"; result = inference / "result"
            retained_prediction = output / "predictions" / block / "prediction.las"
            prediction = retained_prediction if round_id == 1 else work / "prediction.las"
            work.mkdir(parents=True, exist_ok=True); prediction.parent.mkdir(parents=True, exist_ok=True)
            prep_timing = work / "prepare_timing.json"
            run([sys.executable, repo / "scripts/prepare_las_inference_tiles.py",
                 "--input", source, "--output-root", data, "--timing-output", prep_timing],
                repo, env, work / "prepare.log")
            meta_path = data / f"{block}_metadata.json"; meta = json.loads(meta_path.read_text())
            split = data / "benchmark_split.json"
            split.write_text(json.dumps([f"test_final/{x['chunk']}" for x in meta["chunks"]]) + "\n")
            run([sys.executable, repo / "scripts/profile_spunet_val_blocks.py",
                 "--config", repo / "configs/stage_b_eval.py", "--weight", weight,
                 "--data-root", data, "--split", split.name, "--skip-metrics",
                 "--single-scale", "--warmup-iters", args.warmup_iters,
                 "--raw-points", meta["num_points"], "--output", inference],
                repo, env, work / "inference.log")
            merge_timing = work / "merge_timing.json"
            run([sys.executable, repo / "scripts/merge_spunet_full_block.py",
                 "--metadata", meta_path, "--data-root", data, "--result-dir", result,
                 "--output", prediction, "--prediction-field", "pred_classif",
                 "--timing-output", merge_timing], repo, env, work / "merge.log")
            prep = json.loads(prep_timing.read_text()); prof = json.loads((inference / "profile_report.json").read_text())
            merge = json.loads(merge_timing.read_text()); phase = prof["phase_seconds"]
            t_load = prep["T_load_ms"]
            t_pre = prep["T_preprocess_ms"] + 1000 * (phase["data_read"] + phase["preprocess"] + phase["fragment_collate"])
            t_h2d = 1000 * phase["h2d"]; t_fwd = 1000 * phase["model_forward"]
            t_post = merge["T_postprocess_ms"] + 1000 * (phase["score_accumulate"] + phase["label_recovery"] + phase["save_prediction"])
            t_write = merge["T_write_ms"]; t_pipeline = t_pre + t_h2d + t_fwd + t_post
            t_e2e = t_load + t_pipeline + t_write
            raw = int(meta["num_points"]); active = int(prof["active_voxels"]); processed = int(prof["processed_voxel_instances"])
            cm, ignored, gt_field = confusion(source, prediction)
            row = {"block": block, "round": round_id, "bucket": "PENDING",
                   "ground_truth_field": gt_field, "ignored_ground_truth_points": ignored,
                   "raw_points": raw, "active_voxels": active,
                   "voxel_ratio": active / raw, "processed_voxel_instances": processed,
                   "num_chunks": prof["num_chunks"], "spatial_tiles": prep["spatial_tiles"],
                   "configured_chunk_size": prep["configured_chunk_size"], "stride": prep["stride"],
                   "min_chunk_voxels": prof["chunk_voxels"]["min"],
                   "mean_chunk_voxels": prof["chunk_voxels"]["mean"],
                   "max_chunk_voxels": prof["chunk_voxels"]["max"],
                   "T_init_ms": 1000 * prof["setup_seconds"], "T_load_ms": t_load,
                   "T_preprocess_ms": t_pre, "T_h2d_ms": t_h2d, "T_forward_ms": t_fwd,
                   "T_postprocess_ms": t_post, "T_write_ms": t_write,
                   "T_pipeline_ms": t_pipeline, "T_e2e_ms": t_e2e,
                   "raw_mpts_per_s": raw / t_pipeline / 1000,
                   "voxel_mpts_per_s": processed / t_fwd / 1000,
                   "active_voxel_mpts_per_s": active / t_fwd / 1000,
                   "forward_ms_per_100k_voxels": t_fwd / processed * 100000,
                   "pipeline_ms_per_100k_raw_points": t_pipeline / raw * 100000,
                   "forward_ms_per_standard_tile": t_fwd / prep["spatial_tiles"],
                   "pipeline_ms_per_standard_tile": t_pipeline / prep["spatial_tiles"],
                   "e2e_ms_per_standard_tile": t_e2e / prep["spatial_tiles"],
                   "standard_tiles_per_forward_second": prep["spatial_tiles"] / t_fwd * 1000,
                   "standard_tiles_per_pipeline_second": prep["spatial_tiles"] / t_pipeline * 1000,
                   "peak_gpu_allocated_gb": prof["peak_gpu_memory_mib"] / 1024,
                   "peak_gpu_reserved_gb": prof["peak_gpu_reserved_mib"] / 1024,
                   "status": "OK", "wall_ms_including_init_verify": 1000*(time.perf_counter()-block_start),
                   "confusion_matrix": cm.tolist(),
                   "prediction": str(retained_prediction) if round_id == 1 else ""}
            rows.append(row)
            with partial_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            print(f"round {round_id}/{args.rounds} block {block_index}/{len(inputs)} {block} forward={t_fwd:.1f}ms", flush=True)
            shutil.rmtree(work)

    first = [r for r in rows if r["round"] == 1]
    active_values = [r["active_voxels"] for r in first]
    bounds = [float(np.percentile(active_values, p)) for p in (20, 40, 60, 80)]
    for r in rows: r["bucket"] = bucket_for(r["active_voxels"], bounds)
    representatives = {}
    for bucket in ["S", "M", "L", "XL", "XXL"]:
        members = [r for r in first if r["bucket"] == bucket]
        if not members:
            continue
        target = np.median([x["active_voxels"] for x in members])
        representatives[bucket] = min(members, key=lambda x: abs(x["active_voxels"]-target))["block"]

    manifest_fields = ["block", "path", "raw_points", "active_voxels", "voxel_ratio",
                       "processed_voxel_instances", "num_chunks", "bucket", "ground_truth_field"]
    with (output / "benchmark_manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w=csv.DictWriter(f, fieldnames=manifest_fields); w.writeheader()
        by_name={p.stem:p for p in inputs}
        for r in first: w.writerow({"block":r["block"], "path":str(by_name[r["block"]]),
            **{k:r[k] for k in manifest_fields if k not in ("block","path")}})

    csv_fields=[k for k in rows[0] if k not in ("confusion_matrix",)]
    with (output / "benchmark_per_block.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=csv_fields);w.writeheader();w.writerows({k:r[k] for k in csv_fields} for r in rows)

    timing_keys=["T_preprocess_ms","T_h2d_ms","T_forward_ms","T_postprocess_ms","T_pipeline_ms","T_e2e_ms",
                 "forward_ms_per_standard_tile","pipeline_ms_per_standard_tile","e2e_ms_per_standard_tile",
                 "standard_tiles_per_forward_second","standard_tiles_per_pipeline_second"]
    summary={}
    for group in ["ALL","S","M","L","XL","XXL"]:
        subset=rows if group=="ALL" else [r for r in rows if r["bucket"]==group]
        summary[group]={k:stats([r[k] for r in subset]) for k in timing_keys}
        summary[group]["count"]=len(subset)
    reg_active=regression([r["active_voxels"] for r in rows],[r["T_forward_ms"] for r in rows])
    reg_processed=regression([r["processed_voxel_instances"] for r in rows],[r["T_forward_ms"] for r in rows])
    summary["scaling_regression"]={"active_voxels":reg_active,"processed_voxel_instances":reg_processed}
    total_tiles = sum(r["spatial_tiles"] for r in rows)
    summary["standard_tile_weighted"] = {
        "definition": "20m x 20m XY window with 10m stride",
        "standard_tile_instances": total_tiles,
        "forward_ms_per_tile": sum(r["T_forward_ms"] for r in rows) / total_tiles,
        "pipeline_ms_per_tile": sum(r["T_pipeline_ms"] for r in rows) / total_tiles,
        "e2e_ms_per_tile": sum(r["T_e2e_ms"] for r in rows) / total_tiles,
    }

    total_cm=np.sum([np.asarray(r["confusion_matrix"],dtype=np.int64) for r in first],axis=0)
    accuracy=metric_report(total_cm)
    (output/"accuracy_metrics.json").write_text(json.dumps(accuracy,indent=2)+"\n")

    micro={}
    if not args.skip_micro:
        source_map={p.stem:p for p in inputs}
        micro_jobs = []
        micro_gpus = (args.micro_gpus or args.gpu).split(",")
        for job_index, (bucket,block) in enumerate(representatives.items()):
            source=source_map[block]; work=output/"micro_work"/bucket; data=work/"data"
            run([sys.executable,repo/"scripts/prepare_las_inference_tiles.py","--input",source,
                 "--output-root",data],repo,env,work/"prepare.log")
            meta=json.loads((data/f"{block}_metadata.json").read_text()); split=data/"micro_split.json"
            split.write_text(json.dumps([f"test_final/{x['chunk']}" for x in meta["chunks"]])+"\n")
            report=output/"micro"/f"{bucket}.json"
            micro_jobs.append((bucket, work, data, split, report,
                               micro_gpus[job_index % len(micro_gpus)]))

        def run_micro(job):
            bucket, work, data, split, report, gpu = job
            task_env = env.copy(); task_env["CUDA_VISIBLE_DEVICES"] = gpu
            run([sys.executable,repo/"scripts/micro_benchmark_spunet.py","--weight",weight,
                 "--data-root",data,"--split",split.name,"--output",report],
                repo,task_env,work/"micro.log")
            result = json.loads(report.read_text()); result["physical_gpu"] = gpu
            report.write_text(json.dumps(result, indent=2) + "\n")
            shutil.rmtree(work)
            return bucket, result

        with ThreadPoolExecutor(max_workers=len(micro_gpus)) as pool:
            futures = [pool.submit(run_micro, job) for job in micro_jobs]
            for future in as_completed(futures):
                bucket, result = future.result(); micro[bucket] = result
                print(f"micro bucket {bucket} completed on GPU {result['physical_gpu']}", flush=True)
    (output/"micro_summary.json").write_text(json.dumps(micro,indent=2)+"\n")

    env_info={"timestamp":time.strftime("%Y-%m-%dT%H:%M:%S%z"),
      "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip(),
      "checkpoint":str(weight),"checkpoint_sha256":sha256(weight),"model":"SpUNet-v1m1 7-class",
      "gpu":torch.cuda.get_device_name(0),"gpu_count":torch.cuda.device_count(),"cpu":platform.processor(),
      "os":platform.platform(),"pytorch":str(torch.__version__),"cuda":str(torch.version.cuda),
      "cudnn":int(torch.backends.cudnn.version()),"precision":"FP32",
      "batch_size":1,"voxel_size":0.05,"chunk_size_m":20.0,"stride_m":10.0,
      "overlap":"20m XY windows / 10m stride; source-index vote merge",
      "input_features":"XYZ + original RGB","num_workers":2,"tta":False,"seed":42,
      "warmup_iters":args.warmup_iters,"rounds":args.rounds}
    try:
        import spconv; env_info["spconv"]=str(spconv.__version__)
    except Exception: env_info["spconv"]="unknown"
    try:
        cpu_lines = Path("/proc/cpuinfo").read_text(errors="replace").splitlines()
        env_info["cpu"] = next(x.split(":", 1)[1].strip() for x in cpu_lines if x.startswith("model name"))
        mem_lines = Path("/proc/meminfo").read_text().splitlines()
        env_info["ram_total_gb"] = round(int(next(x.split()[1] for x in mem_lines if x.startswith("MemTotal:"))) / 1024**2, 2)
        env_info["nvidia_driver"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        ).splitlines()[int(args.gpu.split(",")[0])]
    except Exception:
        pass
    config={"protocol_version":1,"environment":env_info,
      "dataset":{"input_dir":str(args.input_dir.resolve()),"files":len(inputs),
                 "raw_points_distribution":distribution([r["raw_points"] for r in first]),
                 "active_voxels_distribution":distribution(active_values)},
      "buckets":{"metric":"active_voxels","upper_bounds":{"S":bounds[0],"M":bounds[1],"L":bounds[2],"XL":bounds[3],"XXL":None}},
      "micro_representatives":representatives}
    (output/"benchmark_config.yaml").write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False))
    (output/"benchmark_summary.json").write_text(json.dumps(summary,indent=2)+"\n")

    plots=[("active_voxels","T_forward_ms","latency_vs_active_voxels.png",reg_active),
           ("processed_voxel_instances","T_forward_ms","latency_vs_processed_voxels.png",reg_processed),
           ("raw_points","T_pipeline_ms","pipeline_vs_raw_points.png",None),
           ("active_voxels","voxel_mpts_per_s","throughput_vs_scale.png",None),
           ("active_voxels","peak_gpu_allocated_gb","gpu_memory_vs_active_voxels.png",None)]
    for xk,yk,name,reg in plots:
        x=np.array([r[xk] for r in rows]);y=np.array([r[yk] for r in rows]);plt.figure(figsize=(7,5));plt.scatter(x,y)
        if reg:
            order=np.argsort(x);plt.plot(x[order],reg["intercept_ms"]+reg["slope_ms_per_100k"]*x[order]/1e5)
        plt.xlabel(xk);plt.ylabel(yk);plt.tight_layout();plt.savefig(output/name,dpi=160);plt.close()

    checks={"pipeline_sum_max_abs_ms":max(abs(r["T_pipeline_ms"]-(r["T_preprocess_ms"]+r["T_h2d_ms"]+r["T_forward_ms"]+r["T_postprocess_ms"])) for r in rows),
      "e2e_sum_max_abs_ms":max(abs(r["T_e2e_ms"]-(r["T_load_ms"]+r["T_pipeline_ms"]+r["T_write_ms"])) for r in rows),
      "active_le_raw":all(r["active_voxels"]<=r["raw_points"] for r in rows),
      "processed_ge_active":all(r["processed_voxel_instances"]>=r["active_voxels"] for r in rows),
      "warmup_excluded":True,"model_eval_inference_mode":True,"pipeline_behavior_changed":False}
    allsum=summary["ALL"]
    report=["# SpUNet 500 kV inference benchmark","",
      f"- Model Forward P50/P95: {allsum['T_forward_ms']['p50']:.2f} / {allsum['T_forward_ms']['p95']:.2f} ms",
      f"- Pipeline P50/P95: {allsum['T_pipeline_ms']['p50']:.2f} / {allsum['T_pipeline_ms']['p95']:.2f} ms",
      f"- End-to-End P50/P95: {allsum['T_e2e_ms']['p50']:.2f} / {allsum['T_e2e_ms']['p95']:.2f} ms",
      f"- Model throughput mean: {np.mean([r['voxel_mpts_per_s'] for r in rows]):.3f} M processed voxels/s",
      f"- Pipeline throughput mean: {np.mean([r['raw_mpts_per_s'] for r in rows]):.3f} M raw points/s",
      f"- Weighted standard-tile forward/pipeline/E2E: {summary['standard_tile_weighted']['forward_ms_per_tile']:.2f} / {summary['standard_tile_weighted']['pipeline_ms_per_tile']:.2f} / {summary['standard_tile_weighted']['e2e_ms_per_tile']:.2f} ms per 20m window",
      f"- Peak GPU allocated/reserved: {max(r['peak_gpu_allocated_gb'] for r in rows):.3f} / {max(r['peak_gpu_reserved_gb'] for r in rows):.3f} GB",
      f"- Scaling slope: {reg_processed['slope_ms_per_100k']:.3f} ms / 100k processed voxels (R²={reg_processed['r2']:.4f})","",
      "## Accuracy","",f"Overall accuracy {accuracy['accuracy']:.4%}; macro IoU {accuracy['macro_iou']:.4%}; macro Precision {accuracy['macro_precision']:.4%}; macro Recall {accuracy['macro_recall']:.4%}.",
      f"Micro IoU {accuracy['micro_iou']:.4%}; micro Precision {accuracy['micro_precision']:.4%}; micro Recall {accuracy['micro_recall']:.4%}.","",
      "| class | IoU | Precision | Recall |","|---|---:|---:|---:|"]
    report += [f"| {x['name']} | {x['iou']:.2%} | {x['precision']:.2%} | {x['recall']:.2%} |" for x in accuracy["per_class"]]
    report += ["","## Size buckets","","| bucket | count | forward P50 | forward P95 | pipeline P50 | e2e P95 |","|---|---:|---:|---:|---:|---:|"]
    for b in ["S","M","L","XL","XXL"]:
        s=summary[b]
        fmt=lambda v: "N/A" if v is None else f"{v:.2f}"
        report.append(f"| {b} | {s['count']} | {fmt(s['T_forward_ms']['p50'])} | {fmt(s['T_forward_ms']['p95'])} | {fmt(s['T_pipeline_ms']['p50'])} | {fmt(s['T_e2e_ms']['p95'])} |")
    report += ["","## Timing boundaries","",
      "- `T_load`: sequential LAS bytes-to-point-record reading in preprocessing.",
      "- `T_preprocess`: tiling, feature construction, dataset transforms, voxelization and CPU collation.",
      "- `T_h2d`: collated tensor transfer to GPU, synchronized around the phase.",
      "- `T_forward`: model calls only, measured with CUDA Events after 5 excluded real-data warmups.",
      "- `T_postprocess`: score accumulation, label recovery, source-index voting and output record construction.",
      "- `T_write`: final LAS disk write. `T_init` is reported separately and excluded from pipeline/E2E.",
      "- `T_pipeline = preprocess + h2d + forward + postprocess`; `T_e2e = load + pipeline + write`.",
      "- A standard tile is one fixed 20m x 20m XY inference window at 10m stride. Weighted ms/tile is total time divided by total tile instances across all block-runs; point/voxel-normalized columns remain available for density-aware comparison.",
      "","## Consistency checks","","```json",json.dumps(checks,indent=2),"```", "", "Values 7 in input ground truth are ignored. Prediction is written to `pred_classif`; original `classif`/`classification` is preserved."]
    (output/"benchmark_report.md").write_text("\n".join(report)+"\n")
    (output/"validation_checks.json").write_text(json.dumps(checks,indent=2)+"\n")
    print(output/"benchmark_report.md")


if __name__ == "__main__": main()
