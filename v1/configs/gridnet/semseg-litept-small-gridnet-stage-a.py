_base_ = ["semseg-litept-small-nuscenes-rgb-linear-probe.py"]

# Formal GridNet-HD public-data Stage A: adapt the NuScenes XYZ+RGB
# initialization, then fine-tune the complete LitePT-S backbone.
save_path = "exp/gridnet/litept_small_gridnet_stage_a"
weight = "checkpoints/nuscenes-semseg-litept-small-v1m1/model/backbone_xyz_rgb.pth"
epoch = 100
eval_epoch = 10
batch_size = 8  # global batch: one sample per GPU on 8 GPUs
num_worker = 8
model = dict(freeze_backbone=False)

optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.005)
param_dicts = [dict(keyword="backbone", lr=0.0001)]
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.001, 0.0001],
    pct_start=0.04,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)

data_root = __import__("os").environ.get("PCDSEG_STAGE_A_DATA", "data/stage_a")
data = dict(
    train=dict(
        split="train_final",
        data_root=data_root,
        loop=1,
    ),
    val=dict(
        split="val_final",
        data_root=data_root,
    ),
    test=dict(
        split="val_final",
        data_root=data_root,
    ),
)
