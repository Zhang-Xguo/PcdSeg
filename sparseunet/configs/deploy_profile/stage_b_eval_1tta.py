# Resolved from the training config; no PTv3 base-file dependency.
weight = 'exp/gridnethd/spunet_7class_stage_b_las_new_blocks/model/model_best.pth'
resume = False
evaluate = True
test_only = False
seed = 42
save_path = 'exp/gridnethd/spunet_stage_b_best_profiled_val_blocks'
num_worker = 2
batch_size = 6
batch_size_val = None
batch_size_test = 1
epoch = 30
eval_epoch = 3
clip_grad = None
sync_bn = False
enable_amp = True
amp_dtype = 'float16'
empty_cache = True
empty_cache_per_epoch = False
find_unused_parameters = False
mix_prob = 0.8
param_dicts = [dict(keyword='block', lr=2e-05)]
hooks = [
    dict(type='CheckpointLoader'),
    dict(type='ModelHook'),
    dict(type='IterationTimer', warmup_iter=2),
    dict(type='InformationWriter'),
    dict(type='SemSegEvaluator'),
    dict(type='CheckpointSaver', save_freq=None),
    dict(type='PreciseEvaluator', test_last=False)
]
train = dict(type='DefaultTrainer')
test = dict(type='SemSegTester', verbose=True)
model = dict(
    type='DefaultSegmentor',
    backbone=dict(
        type='SpUNet-v1m1',
        in_channels=3,
        num_classes=7,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 3, 4, 6, 2, 2, 2, 2)),
    criteria=[
        dict(type='CrossEntropyLoss', loss_weight=1.0, ignore_index=255)
    ])
optimizer = dict(type='AdamW', lr=0.0002, weight_decay=0.005)
scheduler = dict(
    type='OneCycleLR',
    max_lr=[0.0002, 2e-05],
    pct_start=0.1,
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=100.0)
dataset_type = 'Gridnethd'
data_root = 'data/gridnethd/pc7'
ignore_index = 255
names = [
    'tower', 'conductor', 'insulator', 'vegetation', 'building', 'ground',
    'other'
]
data = dict(
    num_classes=7,
    ignore_index=255,
    names=[
        'tower', 'conductor', 'insulator', 'vegetation', 'building', 'ground',
        'other'
    ],
    train=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='Gridnethd',
                split='train_stage_b_replay.json',
                data_root='data/gridnethd/pc7',
                transform=[
                    dict(
                        type='RandomRotate',
                        angle=[-1, 1],
                        axis='z',
                        center=[0, 0, 0],
                        p=0.5),
                    dict(type='RandomScale', scale=[0.9, 1.1]),
                    dict(type='RandomFlip', p=0.5),
                    dict(type='RandomJitter', sigma=0.005, clip=0.02),
                    dict(
                        type='RandomColorJitter',
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.2,
                        hue=0.05,
                        p=0.8),
                    dict(type='RandomColorDrop', p=0.15, color_augment=0.0),
                    dict(
                        type='GridSample',
                        grid_size=0.05,
                        hash_type='fnv',
                        mode='train',
                        return_grid_coord=True),
                    dict(type='SphereCrop', point_max=150000, mode='random'),
                    dict(type='NormalizeColor'),
                    dict(type='ToTensor'),
                    dict(
                        type='Collect',
                        keys=('coord', 'grid_coord', 'segment',
                              'supervision_weight'),
                        feat_keys=('color', ))
                ],
                test_mode=False,
                ignore_index=255,
                default_supervision_weight=1.0,
                loop=1),
            dict(
                type='Gridnethd',
                split='train_self.json',
                data_root='data/ours_stage_b_7class_v1',
                transform=[
                    dict(
                        type='RandomRotate',
                        angle=[-1, 1],
                        axis='z',
                        center=[0, 0, 0],
                        p=0.5),
                    dict(type='RandomScale', scale=[0.9, 1.1]),
                    dict(type='RandomFlip', p=0.5),
                    dict(type='RandomJitter', sigma=0.005, clip=0.02),
                    dict(
                        type='RandomColorJitter',
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.2,
                        hue=0.05,
                        p=0.8),
                    dict(type='RandomColorDrop', p=0.15, color_augment=0.0),
                    dict(
                        type='GridSample',
                        grid_size=0.05,
                        hash_type='fnv',
                        mode='train',
                        return_grid_coord=True),
                    dict(type='SphereCrop', point_max=150000, mode='random'),
                    dict(type='NormalizeColor'),
                    dict(type='ToTensor'),
                    dict(
                        type='Collect',
                        keys=('coord', 'grid_coord', 'segment',
                              'supervision_weight'),
                        feat_keys=('color', ))
                ],
                test_mode=False,
                ignore_index=255,
                default_supervision_weight=1.0,
                loop=10)
        ],
        loop=1),
    val=dict(
        type='Gridnethd',
        split='val_fixed.json',
        data_root='data/ours_stage_b_7class_v1',
        transform=[
            dict(type='Copy', keys_dict=dict(segment='origin_segment')),
            dict(
                type='GridSample',
                grid_size=0.05,
                hash_type='fnv',
                mode='train',
                return_grid_coord=True,
                return_inverse=True),
            dict(type='NormalizeColor'),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'segment', 'origin_segment',
                      'inverse'),
                feat_keys=('color', ))
        ],
        test_mode=False,
        ignore_index=255),
    test=dict(
        type='Gridnethd',
        split='val_fixed_las_blocks.json',
        data_root='data/ours_stage_b_7class_v1',
        transform=[
            dict(type='Copy', keys_dict=dict(segment='origin_segment')),
            dict(type='NormalizeColor'),
            dict(
                type='GridSample',
                grid_size=0.025,
                hash_type='fnv',
                mode='train',
                return_inverse=True)
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type='GridSample',
                grid_size=0.05,
                hash_type='fnv',
                mode='test',
                return_grid_coord=True),
            crop=None,
            post_transform=[
                dict(type='ToTensor'),
                dict(
                    type='Collect',
                    keys=('coord', 'grid_coord', 'index'),
                    feat_keys=('color', ))
            ],
            aug_transform=[[{
                'type': 'RandomScale',
                'scale': [0.9, 0.9]
            }], [{
                'type': 'RandomScale',
                'scale': [0.95, 0.95]
            }], [{
                'type': 'RandomScale',
                'scale': [1, 1]
            }], [{
                'type': 'RandomScale',
                'scale': [1.05, 1.05]
            }], [{
                'type': 'RandomScale',
                'scale': [1.1, 1.1]
            }],
                           [{
                               'type': 'RandomScale',
                               'scale': [0.9, 0.9]
                           }, {
                               'type': 'RandomFlip',
                               'p': 1
                           }],
                           [{
                               'type': 'RandomScale',
                               'scale': [0.95, 0.95]
                           }, {
                               'type': 'RandomFlip',
                               'p': 1
                           }],
                           [{
                               'type': 'RandomScale',
                               'scale': [1, 1]
                           }, {
                               'type': 'RandomFlip',
                               'p': 1
                           }],
                           [{
                               'type': 'RandomScale',
                               'scale': [1.05, 1.05]
                           }, {
                               'type': 'RandomFlip',
                               'p': 1
                           }],
                           [{
                               'type': 'RandomScale',
                               'scale': [1.1, 1.1]
                           }, {
                               'type': 'RandomFlip',
                               'p': 1
                           }]]),
        ignore_index=255))
public_root = 'data/gridnethd/pc7'
self_root = 'data/ours_stage_b_7class_v1'
train_transform = [
    dict(
        type='RandomRotate', angle=[-1, 1], axis='z', center=[0, 0, 0], p=0.5),
    dict(type='RandomScale', scale=[0.9, 1.1]),
    dict(type='RandomFlip', p=0.5),
    dict(type='RandomJitter', sigma=0.005, clip=0.02),
    dict(
        type='RandomColorJitter',
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05,
        p=0.8),
    dict(type='RandomColorDrop', p=0.15, color_augment=0.0),
    dict(
        type='GridSample',
        grid_size=0.05,
        hash_type='fnv',
        mode='train',
        return_grid_coord=True),
    dict(type='SphereCrop', point_max=150000, mode='random'),
    dict(type='NormalizeColor'),
    dict(type='ToTensor'),
    dict(
        type='Collect',
        keys=('coord', 'grid_coord', 'segment', 'supervision_weight'),
        feat_keys=('color', ))
]
val_transform = [
    dict(type='Copy', keys_dict=dict(segment='origin_segment')),
    dict(
        type='GridSample',
        grid_size=0.05,
        hash_type='fnv',
        mode='train',
        return_grid_coord=True,
        return_inverse=True),
    dict(type='NormalizeColor'),
    dict(type='ToTensor'),
    dict(
        type='Collect',
        keys=('coord', 'grid_coord', 'segment', 'origin_segment', 'inverse'),
        feat_keys=('color', ))
]

# ============================================================
# Deployment profiling E2: single-view inference
# ============================================================
save_path = "exp/gridnethd/deploy_profile_spunet_1tta"

data["test"]["test_cfg"]["aug_transform"] = [
    [
        dict(
            type="RandomScale",
            scale=[1.0, 1.0],
        )
    ]
]
