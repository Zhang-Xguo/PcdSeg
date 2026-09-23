"""Semantic segmentation wrapper used by this SpUNet variant."""

import torch.nn as nn

from pointcept.models.losses import build_criteria
from .builder import MODELS, build_model


@MODELS.register_module()
class DefaultSegmentor(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        seg_logits = self.backbone(input_dict)
        if self.training:
            loss = self.criteria(
                seg_logits, input_dict["segment"],
                input_dict.get("supervision_weight"),
            )
            return dict(loss=loss)
        if "segment" in input_dict:
            loss = self.criteria(
                seg_logits, input_dict["segment"],
                input_dict.get("supervision_weight"),
            )
            return dict(loss=loss, seg_logits=seg_logits)
        return dict(seg_logits=seg_logits)
