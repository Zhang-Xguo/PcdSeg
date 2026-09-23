#!/usr/bin/env python3
"""Build and audit the fixed scene-level manifest for seven-class Stage B."""
import argparse
import json
import re
from pathlib import Path


T_SCENE = re.compile(r"^T(\d+)-T(\d+)$")
EXCLUDED_SCENES = {"T7-T8"}  # spatial buffer between validation and training
VAL_SCENES = {
    "T3-T4", "T4-T5", "T5-T6", "T6-T7",
    "las_new_target", "pcd_rgb_highline_target", "transmission_tower_val",
}


def natural_block_key(path):
    match = re.search(r"_Block_(\d+)\.las$", path.name)
    return (0, int(match.group(1))) if match else (1, path.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = json.loads(args.base_manifest.read_text(encoding="utf-8"))
    by_scene = {record["scene"]: record for record in records}
    for record in by_scene.values():
        record["split"] = "val_final" if record["scene"] in VAL_SCENES else "train_final"
        record["provenance"] = record.get("provenance", "") + "; fixed Stage B scene split"

    for start in range(3, 16):
        scene = f"T{start}-T{start + 1}"
        if scene in by_scene:
            continue
        summary_path = args.label_root / scene / "pipeline_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        originals = sorted(
            (p for p in (args.raw_root / scene).glob("*.las")
             if re.search(r"_Block_\d+\.las$", p.name)),
            key=natural_block_key,
        )
        if not originals:
            raise RuntimeError(f"No original blocks found for {scene}")
        by_scene[scene] = {
            "scene": scene,
            "split": "train_final",
            "labeled_las": summary["final"],
            "original_las": [str(path.resolve()) for path in originals],
            "provenance": "stable context pipeline audited; fixed Stage B scene split",
        }

    records = [record for scene, record in by_scene.items()
               if scene not in EXCLUDED_SCENES]
    records.sort(key=lambda r: (0 if r["scene"].startswith("110kv") else
                                1 if T_SCENE.match(r["scene"]) else 2, r["scene"]))
    missing = []
    for record in records:
        for path in [record["labeled_las"], *record["original_las"]]:
            if not Path(path).is_file():
                missing.append(path)
    if missing:
        raise RuntimeError("Missing manifest inputs:\n" + "\n".join(missing))
    if len(records) != 35:
        raise RuntimeError(f"Expected 35 included scenes, got {len(records)}")
    if sum(r["split"] == "train_final" for r in records) != 28:
        raise RuntimeError("Expected 28 training scenes")
    if {r["scene"] for r in records if r["split"] == "val_final"} != VAL_SCENES:
        raise RuntimeError("Validation scene set differs from the frozen split")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"records": len(records), "train": 28, "val": 7,
                      "excluded_buffer_scenes": sorted(EXCLUDED_SCENES),
                      "validation_scenes": sorted(VAL_SCENES),
                      "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
