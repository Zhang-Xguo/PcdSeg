#!/usr/bin/env python3
"""Write fixed Stage-B tile lists and a point/class audit after preparation."""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


INDEPENDENT_SCENES = {
    "las_new_target", "pcd_rgb_highline_target", "transmission_tower_val",
}
LEGACY_SCENES = {"T3-T4", "T4-T5", "T5-T6", "T6-T7", "pcd_rgb_highline_target"}
T3_T7_SCENES = {"T3-T4", "T4-T5", "T5-T6", "T6-T7"}
VAL_SCENES = INDEPENDENT_SCENES | LEGACY_SCENES
GROUPS = {
    "las_new_target": "val_las_new.json",
    "pcd_rgb_highline_target": "val_pcd_rgb.json",
    "transmission_tower_val": "val_transmission_tower.json",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--public-root", type=Path)
    parser.add_argument("--replay-multiplier", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()
    root = args.data_root.resolve()
    manifest = json.loads((root / "collection_manifest.json").read_text(encoding="utf-8"))
    output_lists = defaultdict(list)
    report = {"num_classes": 7, "classes": ["tower", "conductor", "insulator",
              "vegetation", "building", "ground", "other"], "scenes": {},
              "split_totals": {}, "excluded_buffer_scenes": ["T7-T8"]}

    train_scenes, val_scenes = set(), set()
    split_totals = defaultdict(lambda: {"tiles": 0, "saved_points": 0,
                                        "manual_points": 0,
                                        "class_counts": [0] * 7,
                                        "manual_class_counts": [0] * 7})
    for record in manifest:
        scene, split = record["scene"], record["split"]
        scene_report = json.loads(
            (root / f"{scene}_{split}_report.json").read_text(encoding="utf-8")
        )
        if scene_report["num_classes"] != 7:
            raise RuntimeError(f"{scene} is not a direct seven-class scene")
        paths = [f"{split}/{tile['tile']}" for tile in scene_report["tiles"]]
        output_lists["train_self.json" if split == "train_final" else "val_fixed.json"].extend(paths)
        if scene in GROUPS:
            output_lists[GROUPS[scene]].extend(paths)
        if scene in LEGACY_SCENES:
            output_lists["val_legacy_compare.json"].extend(paths)
        if scene in INDEPENDENT_SCENES:
            output_lists["val_independent.json"].extend(paths)
        if scene in T3_T7_SCENES:
            output_lists["val_t3_t7.json"].extend(paths)
        (train_scenes if split == "train_final" else val_scenes).add(scene)
        counts = [scene_report["class_counts"].get(str(i), 0) for i in range(7)]
        manual = [scene_report["manual_class_counts"].get(str(i), 0) for i in range(7)]
        saved = sum(tile["saved_points"] for tile in scene_report["tiles"])
        report["scenes"][scene] = {"split": split, "tiles": len(paths),
                                    "saved_points": saved,
                                    "raw_class_counts": counts,
                                    "raw_manual_class_counts": manual}
        total = split_totals[split]
        total["tiles"] += len(paths)
        total["saved_points"] += saved
        total["manual_points"] += sum(manual)
        total["class_counts"] = [a + b for a, b in zip(total["class_counts"], counts)]
        total["manual_class_counts"] = [a + b for a, b in zip(total["manual_class_counts"], manual)]

    if train_scenes & val_scenes:
        raise RuntimeError(f"Scene leakage: {sorted(train_scenes & val_scenes)}")
    if val_scenes != VAL_SCENES:
        raise RuntimeError(f"Unexpected validation scenes: {sorted(val_scenes)}")
    for filename, paths in output_lists.items():
        (root / filename).write_text(json.dumps(sorted(paths), indent=2), encoding="utf-8")
    if args.public_root is not None:
        public_root = args.public_root.resolve()
        public_paths = sorted(
            f"train_final/{path.name}"
            for path in (public_root / "train_final").iterdir() if path.is_dir()
        )
        replay_size = len(output_lists["train_self.json"]) * args.replay_multiplier
        if replay_size > len(public_paths):
            raise RuntimeError("Requested public replay subset exceeds public training set")
        replay = sorted(random.Random(args.seed).sample(public_paths, replay_size))
        (public_root / "train_stage_b_replay.json").write_text(
            json.dumps(replay, indent=2), encoding="utf-8"
        )
        report["public_replay"] = {
            "file": str(public_root / "train_stage_b_replay.json"),
            "chunks": replay_size, "seed": args.seed,
            "self_tiles": len(output_lists["train_self.json"]),
            "self_loop": args.replay_multiplier,
        }
    report["split_totals"] = dict(split_totals)
    report["leakage_audit"] = {"scene_overlap": [], "passed": True}
    (root / "split_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({name: len(paths) for name, paths in output_lists.items()}, indent=2))


if __name__ == "__main__":
    main()
