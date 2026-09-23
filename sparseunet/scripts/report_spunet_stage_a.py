#!/usr/bin/env python3
"""Write an auditable local status report after Stage A exits."""
import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", type=Path, required=True)
    p.add_argument("--exit-code", type=int, required=True)
    p.add_argument("--stage", choices=("A", "B"), default="A")
    p.add_argument("--expected-validations", type=int, default=10)
    a = p.parse_args()
    log = a.experiment / "train.log"
    text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
    number = r"\d+(?:\.\d+)?"
    validations = [tuple(map(float, row)) for row in re.findall(
        rf"Val result: mIoU/mAcc/allAcc ({number})/({number})/({number})", text)]
    finished = "Best mIoU:" in text
    best_iou = max((x[0] for x in validations), default=None)
    checkpoint = a.experiment / "model" / "model_best.pth"
    report = {
        "stage": a.stage,
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "exit_code": a.exit_code,
        "completed": a.exit_code == 0 and finished and len(validations) == a.expected_validations and checkpoint.exists(),
        "expected_validations": a.expected_validations,
        "validation_cycles": len(validations),
        "best_miou": best_iou,
        "last_validation": validations[-1] if validations else None,
        "best_checkpoint": str(checkpoint.resolve()) if checkpoint.exists() else None,
        "train_log": str(log.resolve()),
    }
    a.experiment.mkdir(parents=True, exist_ok=True)
    path = a.experiment / "TRAIN_STATUS.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Stage {a.stage} status written to {path}: completed={report['completed']}", flush=True)


if __name__ == "__main__":
    main()
