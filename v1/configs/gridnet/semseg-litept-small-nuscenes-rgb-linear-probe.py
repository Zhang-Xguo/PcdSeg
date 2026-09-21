_base_ = ["semseg-litept-small-nuscenes-linear-probe.py"]

save_path = "exp/gridnet/litept_small_nuscenes_rgb_linear_probe"
weight = "checkpoints/nuscenes-semseg-litept-small-v1m1/model/backbone_xyz_rgb.pth"
model = dict(backbone=dict(in_channels=6))

train_transform = [
    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
    dict(type="RandomScale", scale=[0.9, 1.1]),
    dict(type="RandomFlip", p=0.5),
    dict(type="RandomJitter", sigma=0.005, clip=0.02),
    dict(type="GridSample", grid_size=0.05, hash_type="fnv", mode="train", return_grid_coord=True),
    dict(type="SphereCrop", point_max=150000, mode="random"),
    dict(type="NormalizeColor"),
    dict(type="ToTensor"),
    dict(type="Update", keys_dict={"grid_size": 0.05}),
    dict(
        type="Collect",
        keys=("coord", "grid_coord", "segment", "grid_size"),
        feat_keys=("coord", "color"),
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
    dict(type="NormalizeColor"),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "grid_coord", "segment", "origin_segment", "inverse"),
        feat_keys=("coord", "color"),
    ),
]

data = dict(
    train=dict(transform=train_transform),
    val=dict(transform=val_transform),
    test=dict(
        transform=[dict(type="NormalizeColor")],
        test_cfg=dict(
            post_transform=[
                dict(type="ToTensor"),
                dict(type="Collect", keys=("coord", "grid_coord", "index"), feat_keys=("coord", "color")),
            ]
        ),
    ),
)
