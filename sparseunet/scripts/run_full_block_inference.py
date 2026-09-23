#!/usr/bin/env python3
"""End-to-end SpUNet inference for complete original LAS blocks."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="*", type=Path)
    ap.add_argument("--input-list", type=Path,
                    help="One LAS path, or group<TAB>LAS path, per line")
    ap.add_argument("--weight", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("configs/stage_b_eval.py"))
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--work-root", type=Path, required=True)
    ap.add_argument("--gpu", default="0", help="Physical CUDA device index")
    ap.add_argument("--single-scale", action="store_true",
                    help="Faster qualitative export; differs from 10-TTA validation")
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    inputs = [(p.parent.name, p.resolve()) for p in (args.inputs or [])]
    if args.input_list:
        for line in args.input_list.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if "\t" in line:
                group, path = line.split("\t", 1)
            else:
                path = line
                group = Path(path).parent.name
            inputs.append((group, Path(path).resolve()))
    if not inputs:
        ap.error("provide --inputs or --input-list")
    weight = args.weight.resolve()
    if not weight.is_file():
        ap.error(f"checkpoint not found: {weight}")
    config = args.config if args.config.is_absolute() else repo / args.config
    if not config.is_file():
        ap.error(f"config not found: {config}")
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, PYTHONPATH=str(repo),
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")

    def run(cmd: list[str], log: Path) -> None:
        with log.open("w", encoding="utf-8") as stream:
            subprocess.run(cmd, cwd=repo, env=env, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)

    for group, source in inputs:
        if not source.is_file() or source.suffix.lower() != ".las":
            raise ValueError(f"Input must be an existing LAS file: {source}")
        out = args.output_root.resolve() / group / source.stem
        work = args.work_root.resolve() / group / source.stem
        out.mkdir(parents=True, exist_ok=True)
        work.mkdir(parents=True, exist_ok=True)
        data = work / "data"
        metadata = data / f"{source.stem}_metadata.json"
        if not metadata.is_file():
            if data.exists() and any(data.iterdir()):
                raise RuntimeError(f"Incomplete preparation at {data}; inspect it before retrying")
            run([sys.executable, str(repo / "scripts/prepare_las_inference_tiles.py"),
                 "--input", str(source), "--output-root", str(data)], work / "prepare.log")
        meta = json.loads(metadata.read_text(encoding="utf-8"))
        if Path(meta["source_las"]).resolve() != source or meta["scene"] != source.stem:
            raise RuntimeError(f"Prepared data belongs to another input: {metadata}")
        names = [row["chunk"] for row in meta["chunks"]]
        split = data / "full_block_split.json"
        split.write_text(json.dumps([f"test_final/{name}" for name in names]) + "\n")
        inference = work / "inference"
        result = inference / "result"
        profile = inference / "profile_report.json"
        existing = list(result.glob("*_pred.npy"))
        expected_mode = "single_scale_1.0" if args.single_scale else "configured_tta"
        reuse = len(existing) == len(names) and profile.is_file()
        if reuse:
            old = json.loads(profile.read_text(encoding="utf-8"))
            reuse = (Path(old["checkpoint"]).resolve() == weight and
                     old.get("augmentation") == expected_mode)
        if not reuse:
            for file in existing:
                file.unlink()
            cmd = [sys.executable, str(repo / "scripts/profile_spunet_val_blocks.py"),
                   "--config", str(config), "--weight", str(weight),
                   "--data-root", str(data), "--split", split.name,
                   "--skip-metrics", "--output", str(inference)]
            if args.single_scale:
                cmd.append("--single-scale")
            run(cmd, work / "inference.log")
        prediction = out / "prediction.las"
        run([sys.executable, str(repo / "scripts/merge_spunet_full_block.py"),
             "--metadata", str(metadata), "--data-root", str(data),
             "--result-dir", str(result), "--output", str(prediction)],
            work / "merge.log")
        print(f"COMPLETE {source} -> {prediction}", flush=True)


if __name__ == "__main__":
    main()
