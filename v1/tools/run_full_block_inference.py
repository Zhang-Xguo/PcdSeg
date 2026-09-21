#!/usr/bin/env python3
"""Run LitePT on complete LAS blocks and fuse overlapping tiles to source order."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


def run(command, log, env):
    started = time.perf_counter()
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        subprocess.run([str(x) for x in command], check=True, env=env,
                       stdout=handle, stderr=subprocess.STDOUT)
    return round(time.perf_counter() - started, 3)


def write_status(path, **values):
    payload = dict(updated_at=time.strftime("%Y-%m-%d %H:%M:%S"), **values)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", type=Path,
                        help="Single raw LAS, e.g. inference/input/input.las")
    inputs.add_argument("--input-list", type=Path,
                        help="TSV with group and raw LAS path per line")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--gpu", default="6")
    parser.add_argument("--worker-id", default="0")
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--config", type=Path,
                        help="Model config (default: Stage B LitePT-S config)")
    parser.add_argument("--keep-work", action="store_true")
    args = parser.parse_args()

    litept = Path(__file__).resolve().parents[1]
    config = (args.config.resolve() if args.config else
              litept / "configs/gridnet/semseg-litept-small-gridnet-stage-b-las-blocks.py")
    verifier = litept / "tools/verify_block_prediction.py"
    if args.input:
        rows = [(args.input.parent.name, args.input.resolve())]
    else:
        rows = []
        for line in args.input_list.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            group, source = line.split("\t", 1)
            rows.append((group, Path(source).resolve()))

    env = os.environ.copy()
    env.update(PYTHONPATH=str(litept), CUDA_VISIBLE_DEVICES=args.gpu,
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    manifest = []
    status_path = args.work_root / f"status_worker_{args.worker_id}.json"
    manifest_path = args.work_root / f"manifest_worker_{args.worker_id}.json"
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.work_root.mkdir(parents=True, exist_ok=True)
    for index, (group, source) in enumerate(rows, 1):
        name = source.stem
        scene = source.parent.name
        work = args.work_root / group / name
        data = work / "data"
        inference = work / "inference"
        output = args.output_root / scene / f"{name}.las"
        record = {"split_group": group, "scene": scene,
                  "source": str(source), "output": str(output),
                  "status": "running", "timings_seconds": {}}
        print(f"[{index}/{len(rows)}] {group}/{name}", flush=True)
        try:
            if not output.exists():
                write_status(status_path, state="running", stage="prepare",
                             current=index, total=len(rows), scene=scene, block=name)
                metadata_files = list(data.glob("*_metadata.json"))
                if not metadata_files or not (data / "test_final").is_dir():
                    # The streaming preparer deliberately refuses a non-empty
                    # output root.  Remove an interrupted partial preparation
                    # before rebuilding it; a complete preparation is reused.
                    if data.exists():
                        shutil.rmtree(data)
                    record["timings_seconds"]["prepare"] = run(
                        [args.python, litept / "tools/prepare_las_inference_tiles.py",
                         "--input", source, "--output-root", data],
                        work / "prepare.log", env)
                else:
                    record["timings_seconds"]["prepare"] = 0.0
                    record["reused_preparation"] = True
                write_status(status_path, state="running", stage="inference",
                             current=index, total=len(rows), scene=scene, block=name)
                record["timings_seconds"]["inference"] = run(
                    [args.python, litept / "tools/test.py", "--config-file", config,
                     "--num-gpus", "1", "--options", f"save_path={inference}",
                     f"weight={args.weight.resolve()}",
                     f"data.test.data_root={data}", "data.test.split=test_final",
                     "batch_size_test=1", "num_worker=1", "enable_wandb=False"],
                    work / "inference.log", env)
                metadata = next(data.glob("*_metadata.json"))
                write_status(status_path, state="running", stage="merge",
                             current=index, total=len(rows), scene=scene, block=name)
                record["timings_seconds"]["merge"] = run(
                    [args.python, litept / "tools/merge_block_predictions.py",
                     "--metadata", metadata, "--data-root", data,
                     "--result-dir", inference / "result", "--output", output,
                     "--num-classes", "7", "--max-missing-fallback", "10"],
                    work / "merge.log", env)
            write_status(status_path, state="running", stage="verify",
                         current=index, total=len(rows), scene=scene, block=name)
            record["timings_seconds"]["verify"] = run(
                [args.python, verifier, "--raw", source, "--prediction", output],
                work / "verify.log", env)
            record["status"] = "complete"
            print(f"COMPLETE {output}", flush=True)
        except Exception as error:
            record["status"] = "failed"
            record["error"] = repr(error)
            write_status(status_path, state="failed", stage="failed",
                         current=index, total=len(rows), scene=scene, block=name,
                         error=repr(error))
            manifest.append(record)
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            raise
        manifest.append(record)
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        # Work tiles are intentionally retained by default for resumability.
        # They can be removed after the complete manifest has been audited.
    write_status(status_path, state="complete", stage="complete",
                 current=len(rows), total=len(rows), scene="", block="")
    print(json.dumps({"complete": len(manifest), "manifest": str(manifest_path)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
