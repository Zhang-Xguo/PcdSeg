#!/usr/bin/env python3
"""Prepare a reproducible multi-scene fine-tuning collection from a manifest."""
import argparse
import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def load_records(path):
    """Load a flat manifest or an object extending another manifest."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if not isinstance(data, dict) or "records" not in data:
        raise RuntimeError("Manifest must be a JSON list or an include/records object")
    records = []
    if data.get("include"):
        included = (path.parent / data["include"]).resolve()
        records.extend(load_records(included))
    records.extend(data["records"])
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--python", required=True)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--resume", action="store_true",
                   help="Keep completed scenes and continue an interrupted collection")
    p.add_argument("--num-classes", type=int, choices=(7, 11), default=11)
    p.add_argument("--manual-weight", type=float, default=1.0)
    p.add_argument("--postprocess-weight", type=float, default=0.5)
    p.add_argument("--pseudo-weight", type=float, default=0.25)
    p.add_argument("--tile-size", type=float, default=20.0)
    p.add_argument("--pre-grid-size", type=float, default=0.05)
    p.add_argument("--min-points", type=int, default=3000)
    p.add_argument("--min-manual-points", type=int, default=100)
    a = p.parse_args()
    records = load_records(a.manifest.resolve())
    if not isinstance(records, list) or not records:
        raise RuntimeError("Manifest must be a non-empty JSON list")
    if a.output_root.exists() and not a.resume:
        raise FileExistsError(f"Refusing to overwrite {a.output_root}")
    a.output_root.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve().parent / "prepare_final_las_finetune.py"

    names = [x["scene"] for x in records]
    if len(names) != len(set(names)):
        raise RuntimeError("Scene names must be unique")

    requested_manifest = a.output_root / "collection_manifest.requested.json"
    if requested_manifest.exists():
        previous = json.loads(requested_manifest.read_text(encoding="utf-8"))
        if previous != records:
            raise RuntimeError(
                "Resume manifest differs from collection_manifest.requested.json"
            )
    else:
        requested_manifest.write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    state_lock = threading.Lock()
    state_path = a.output_root / "prepare_progress.json"
    state = {"completed": [], "failed": {}}
    if a.resume and state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))

    def save_state():
        state_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def completed(record):
        report_path = a.output_root / f"{record['scene']}_{record['split']}_report.json"
        if not report_path.is_file():
            return False
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("num_classes") != a.num_classes:
            return False
        split_root = a.output_root / record["split"]
        return bool(report.get("tiles")) and all(
            (split_root / tile["tile"] / "segment.npy").is_file()
            for tile in report["tiles"]
        )

    def run(record):
        required = {"scene", "split", "labeled_las", "original_las"}
        if not required.issubset(record):
            raise RuntimeError(f"Missing fields for {record}: {required - set(record)}")
        cmd = [a.python, str(script), "--labeled-las", record["labeled_las"],
               "--original-las", *record["original_las"],
               "--output-root", str(a.output_root), "--scene-name", record["scene"],
               "--split", record["split"],
               "--num-classes", str(a.num_classes),
               "--manual-weight", str(a.manual_weight),
               "--postprocess-weight", str(a.postprocess_weight),
               "--pseudo-weight", str(a.pseudo_weight),
               "--tile-size", str(a.tile_size),
               "--pre-grid-size", str(a.pre_grid_size),
               "--min-points", str(a.min_points),
               "--min-manual-points", str(a.min_manual_points)]
        result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        (a.output_root / f"{record['scene']}_prepare.log").write_text(
            result.stdout, encoding="utf-8")
        if result.returncode:
            with state_lock:
                state["failed"][record["scene"]] = (
                    f"return code {result.returncode}; see prepare log"
                )
                save_state()
            raise RuntimeError(f"{record['scene']} failed; see prepare log")
        with state_lock:
            state["failed"].pop(record["scene"], None)
            if record["scene"] not in state["completed"]:
                state["completed"].append(record["scene"])
            save_state()
        return record["scene"]

    pending = []
    for record in records:
        if a.resume and completed(record):
            print("SKIPPED COMPLETE", record["scene"], flush=True)
            if record["scene"] not in state["completed"]:
                state["completed"].append(record["scene"])
        else:
            pending.append(record)
    save_state()

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(run, x) for x in pending]
        for future in as_completed(futures):
            print("PREPARED", future.result(), flush=True)
    (a.output_root / "collection_manifest.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"COLLECTION COMPLETE: {a.output_root}")


if __name__ == "__main__":
    main()
