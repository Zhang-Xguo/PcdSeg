#!/usr/bin/env python3
"""Convert the ScanNet20 SpUNet safetensors backbone to Pointcept RGB-3.

The output classifier has seven random, reproducible channels; its predictions
are only a pipeline sanity baseline, not zero-shot power-line recognition.
"""
import argparse
from pathlib import Path

import torch
from safetensors import safe_open

from pointcept.models import build_model


def convert(source: Path, output: Path, seed: int = 42):
    torch.manual_seed(seed)
    model = build_model(dict(
        type="DefaultSegmentor",
        backbone=dict(type="SpUNet-v1m1", in_channels=3, num_classes=7,
                      channels=(32, 64, 128, 256, 256, 128, 96, 96),
                      layers=(2, 3, 4, 6, 2, 2, 2, 2)),
        criteria=[dict(type="CrossEntropyLoss", ignore_index=255)],
    ))
    target = model.state_dict()
    mapped = {}
    missing = []
    with safe_open(str(source), framework="pt", device="cpu") as f:
        for dest, expected in target.items():
            if not dest.startswith("backbone.") or dest.startswith("backbone.final."):
                continue
            name = dest[len("backbone."):]
            if name.startswith("conv_input.") or name.startswith(("down.", "enc.")):
                key = "encoder." + name
            elif name.startswith(("up.", "dec.")):
                key = "decoder." + name
            else:
                missing.append(dest)
                continue
            # torch-pointcloud wraps BatchNorm in a `module` field, and calls
            # the block BatchNorm layers norm1/norm2 rather than bn1/bn2.
            key = key.replace(".bn1.", ".norm1.module.")
            key = key.replace(".bn2.", ".norm2.module.")
            if ".proj.1." in key:
                key = key.replace(".proj.1.", ".proj.1.module.")
            elif name.startswith(("up.", "down.", "conv_input.")):
                parts = key.split(".")
                # ...up.<stage>.1.<bn-field> or ...conv_input.1.<bn-field>
                if parts[-2] == "1":
                    parts.insert(-1, "module")
                    key = ".".join(parts)
            if key not in f.keys():
                missing.append((dest, key))
                continue
            value = f.get_tensor(key)
            if name == "conv_input.0.weight":
                # Reference input is RGB + normal; keep RGB channels only.
                value = value[..., :3].contiguous()
            if value.shape != expected.shape:
                missing.append((dest, key, tuple(value.shape), tuple(expected.shape)))
                continue
            mapped[dest] = value
    if missing:
        raise RuntimeError(f"{len(missing)} incompatible backbone tensors: {missing[:12]}")
    target.update(mapped)
    model.load_state_dict(target, strict=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": 0, "state_dict": model.state_dict(),
                "source": str(source), "seed": seed,
                "backbone_tensors_mapped": len(mapped),
                "classifier": "random 7-class head"}, output)
    print(f"saved {output}, mapped {len(mapped)} backbone tensors")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    convert(a.source, a.output, a.seed)
