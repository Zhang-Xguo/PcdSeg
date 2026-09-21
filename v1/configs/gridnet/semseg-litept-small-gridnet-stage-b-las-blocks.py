_base_ = ["semseg-litept-small-gridnet-stage-a.py"]

# Stage B: fine-tune the public-data Stage A best model on the original
# self-labeled set plus replay and the new spatial-block las_new split.
save_path = "exp/gridnet/litept_small_gridnet_stage_b_las_blocks"
weight = "exp/gridnet/litept_small_gridnet_stage_a_gpus456_b48/model/model_best.pth"
epoch = 30
eval_epoch = 3
batch_size = 24  # global batch: 8 per GPU on physical GPUs 4,5,6
num_worker = 3

model = dict(
    criteria=[
        dict(
            type="CrossEntropyLoss",
            loss_weight=1.0,
            ignore_index=255,
            weight=[2.0, 1.3, 3.0, 0.8, 1.0, 1.0, 1.0],
        ),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=0.5, ignore_index=255),
    ]
)

optimizer = dict(type="AdamW", lr=0.0002, weight_decay=0.005)
param_dicts = [dict(keyword="backbone", lr=0.00002)]
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.0002, 0.00002],
    pct_start=0.1,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)

train_transform = [
    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
    dict(type="RandomScale", scale=[0.9, 1.1]),
    dict(type="RandomFlip", p=0.5),
    dict(type="RandomJitter", sigma=0.005, clip=0.02),
    dict(type="GridSample", grid_size=0.05, hash_type="fnv", mode="train", return_grid_coord=True),
    dict(
        type="ClassAwareSphereCrop",
        point_max=150000,
        class_ids=(0, 1, 2, 4),
        class_weights=(0.25, 0.25, 0.4, 0.1),
        application_ratio=0.5,
    ),
    dict(type="NormalizeColor"),
    dict(type="ToTensor"),
    dict(type="Update", keys_dict={"grid_size": 0.05}),
    dict(type="Collect", keys=("coord", "grid_coord", "segment", "grid_size"), feat_keys=("coord", "color")),
]

val_transform = [
    dict(type="Copy", keys_dict={"segment": "origin_segment"}),
    dict(type="GridSample", grid_size=0.05, hash_type="fnv", mode="train", return_grid_coord=True, return_inverse=True),
    dict(type="NormalizeColor"),
    dict(type="ToTensor"),
    dict(type="Collect", keys=("coord", "grid_coord", "segment", "origin_segment", "inverse"), feat_keys=("coord", "color")),
]

data_root = __import__("os").environ.get("PCDSEG_STAGE_B_DATA", "data/stage_b")
public_root = __import__("os").environ.get("PCDSEG_STAGE_A_DATA", "data/stage_a")
data = dict(
    train=dict(
        _delete_=True,
        type="ConcatDataset",
        datasets=[
            dict(type="GridNetDataset", split="train_stage_b_replay.json", data_root=public_root, transform=train_transform, test_mode=False, ignore_index=255, loop=1),
            dict(type="GridNetDataset", split="train_self.json", data_root=data_root, transform=train_transform, test_mode=False, ignore_index=255, loop=10),
            dict(type="GridNetDataset", split="train_las_new_blocks.json", data_root=data_root, transform=train_transform, test_mode=False, ignore_index=255, loop=10),
        ],
        loop=1,
    ),
    val=dict(_delete_=True, type="GridNetDataset", split="val_fixed_las_blocks.json", data_root=data_root, transform=val_transform, test_mode=False, ignore_index=255),
    test=dict(_delete_=True, type="GridNetDataset", split="val_fixed_las_blocks.json", data_root=data_root, transform=[dict(type="NormalizeColor")], test_mode=True, test_cfg=dict(voxelize=dict(type="GridSample", grid_size=0.05, hash_type="fnv", mode="test", return_grid_coord=True), crop=None, post_transform=[dict(type="ToTensor"), dict(type="Collect", keys=("coord", "grid_coord", "index"), feat_keys=("coord", "color"))], aug_transform=[[dict(type="RandomScale", scale=[1.0, 1.0])]]), ignore_index=255),
)
