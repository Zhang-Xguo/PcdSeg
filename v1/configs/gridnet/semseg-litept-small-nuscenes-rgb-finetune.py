_base_ = ["semseg-litept-small-nuscenes-rgb-linear-probe.py"]

save_path = "exp/gridnet/litept_small_nuscenes_rgb_finetune"
weight = "exp/gridnet/litept_small_nuscenes_rgb_linear_probe/model/model_best.pth"
model = dict(freeze_backbone=False)

optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.005)
param_dicts = [dict(keyword="backbone", lr=0.0001)]
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.001, 0.0001],
    pct_start=0.2,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)
