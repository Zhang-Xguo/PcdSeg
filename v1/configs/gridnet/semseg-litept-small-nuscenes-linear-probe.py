_base_ = ["../_base_/default_runtime.py"]

batch_size = 1
num_worker = 4
# torch21's installed spconv build cannot select an FP16 training kernel on
# A100; FP32 is stable and easily fits while only the 7-class head is trained.
enable_amp = False
enable_wandb = False
empty_cache = True
mix_prob = 0
epoch = 3
eval_epoch = 3
save_path = "exp/gridnet/litept_small_nuscenes_linear_probe"
weight = "checkpoints/nuscenes-semseg-litept-small-v1m1/model/backbone.pth"

model = dict(
    type="DefaultSegmentorV2",
    num_classes=7,
    backbone_out_channels=72,
    freeze_backbone=True,
    backbone=dict(
        type="LitePT",
        in_channels=4,
        order=["z", "z-trans", "hilbert", "hilbert-trans"],
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(36, 72, 144, 252, 504),
        enc_num_head=(2, 4, 8, 14, 28),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        enc_conv=(True, True, True, False, False),
        enc_attn=(False, False, False, True, True),
        enc_rope_freq=(100.0, 100.0, 100.0, 100.0, 100.0),
        dec_depths=(0, 0, 0, 0),
        dec_channels=(72, 72, 144, 252),
        dec_num_head=(4, 4, 8, 14),
        dec_patch_size=(1024, 1024, 1024, 1024),
        dec_conv=(False, False, False, False),
        dec_attn=(False, False, False, False),
        dec_rope_freq=(100.0, 100.0, 100.0, 100.0),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enc_mode=False,
    ),
    criteria=[
        dict(
            type="CrossEntropyLoss",
            loss_weight=1.0,
            ignore_index=255,
            weight=[2.0, 1.3, 3.0, 0.8, 1.0, 1.0, 1.0],
        ),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=0.5, ignore_index=255),
    ],
)

optimizer = dict(type="AdamW", lr=0.01, weight_decay=0.0)
scheduler = dict(
    type="OneCycleLR",
    max_lr=0.01,
    pct_start=0.2,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)
param_dicts = None

dataset_type = "GridNetDataset"
data_root = "../PTv3_GridNet-HD_baseline/data/ours_stage_b_7class_v1"
names = ["tower", "conductor", "insulator", "vegetation", "building", "ground", "other"]

train_transform = [
    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
    dict(type="RandomScale", scale=[0.9, 1.1]),
    dict(type="RandomFlip", p=0.5),
    dict(type="RandomJitter", sigma=0.005, clip=0.02),
    dict(type="GridSample", grid_size=0.05, hash_type="fnv", mode="train", return_grid_coord=True),
    dict(type="SphereCrop", point_max=150000, mode="random"),
    dict(type="ToTensor"),
    dict(type="Update", keys_dict={"grid_size": 0.05}),
    dict(
        type="Collect",
        keys=("coord", "grid_coord", "segment", "grid_size"),
        feat_keys=("coord", "strength"),
    ),
]

val_transform = [
    dict(type="Copy", keys_dict={"segment": "origin_segment"}),
    dict(
        type="GridSample",
        grid_size=0.05,
        hash_type="fnv",
        mode="train",
        return_grid_coord=True,
        return_inverse=True,
    ),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "grid_coord", "segment", "origin_segment", "inverse"),
        feat_keys=("coord", "strength"),
    ),
]

data = dict(
    num_classes=7,
    ignore_index=255,
    names=names,
    train=dict(
        type=dataset_type,
        split="train_self.json",
        data_root=data_root,
        transform=train_transform,
        test_mode=False,
        ignore_index=255,
        loop=1,
    ),
    val=dict(
        type=dataset_type,
        split="val_legacy_compare.json",
        data_root=data_root,
        transform=val_transform,
        test_mode=False,
        ignore_index=255,
    ),
    test=dict(
        type=dataset_type,
        split="val_legacy_compare.json",
        data_root=data_root,
        transform=[],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(type="GridSample", grid_size=0.05, hash_type="fnv", mode="test", return_grid_coord=True),
            crop=None,
            post_transform=[
                dict(type="ToTensor"),
                dict(type="Collect", keys=("coord", "grid_coord", "index"), feat_keys=("coord", "strength")),
            ],
            aug_transform=[[dict(type="RandomScale", scale=[1.0, 1.0])]],
        ),
        ignore_index=255,
    ),
)

hooks = [
    dict(type="CheckpointLoader", strict=False),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=1),
    dict(type="PreciseEvaluator", test_last=False),
]
