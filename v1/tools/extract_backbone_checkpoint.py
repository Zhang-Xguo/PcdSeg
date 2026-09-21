"""Remove a dataset-specific segmentation head from a LitePT checkpoint."""

import argparse
from collections import OrderedDict

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument(
        "--xyz-rgb",
        action="store_true",
        help="expand the NuScenes XYZ+strength stem to XYZ+RGB",
    )
    args = parser.parse_args()

    checkpoint = torch.load(args.source, map_location="cpu")
    prefix = "module.backbone."
    state = OrderedDict(
        (key, value)
        for key, value in checkpoint["state_dict"].items()
        if key.startswith(prefix)
    )
    if args.xyz_rgb:
        key = "module.backbone.embedding.stem.conv.weight"
        source_weight = state[key]
        if source_weight.shape[-1] != 4:
            raise ValueError(f"expected four input channels, got {source_weight.shape}")
        expanded = source_weight.new_zeros(*source_weight.shape[:-1], 6)
        expanded[..., :3] = source_weight[..., :3]
        # Preserve the old luminance response exactly for RGB values in [0, 1].
        luminance = source_weight[..., 3:4]
        expanded[..., 3:] = luminance * source_weight.new_tensor(
            [0.299, 0.587, 0.114]
        )
        state[key] = expanded
    torch.save({"epoch": 0, "state_dict": state}, args.destination)
    print(f"saved {len(state)} backbone tensors to {args.destination}")


if __name__ == "__main__":
    main()
