# PcdSeg sparseunet：SpUNet-v1m1 输电点云 7 类分割

本目录是与 `v1/` 并列的独立 Python 方案。模型为 Pointcept 的 SpUNet-v1m1（spconv SparseUNet），输入 XYZ + **原始传感器 RGB**，输出逐点类别 `0–6`。`255` 只用于忽略的真值点。类别顺序和可视化色板见 [`classes.json`](classes.json)；色板不覆盖输出 LAS 的原始 RGB。

目录包含数据预处理、公开数据 Stage A 预训练、自建数据 Stage B 微调、固定验证集指标与计时，以及完整原始 block 的 LAS 推理和字段保真回填。仓库不含数据、权重、实验日志、LAS/NPY 输出或 Windows/C++/MIPOT 工程产物。

## 环境与目录

已验证的原实验环境是 Linux、Python 3.8、PyTorch 2.1.0/CUDA 12.1、spconv 2.3.6。需要 NVIDIA GPU；依赖版本见 `requirements.txt`。先安装与本机驱动匹配的 CUDA 版 PyTorch，再安装其余依赖。运行入口均从本目录执行：

```bash
cd PcdSeg/sparseunet
export PYTHONPATH="$PWD"
python -m pip install -r requirements.txt
```

默认数据与权重路径：

```text
data/gridnethd/pc7/             公开 GridNet-HD 的 7 类 tile
data/ours_stage_b_7class_v1/     自建数据 tile
checkpoints/spunet/              ScanNet20 初始化权重
exp/                             训练和验证输出
```

每个 tile 目录有 `coord.npy`（float32，N×3）、`color.npy`（原始 RGB，N×3）、`segment.npy`（0–6 或忽略值 255）。Stage B 另有 `supervision_weight.npy`。划分 JSON 存储相对于数据根目录的 tile 路径。固定划分的列表副本在 `splits/stage_b/`；它们仅是文件名清单，不含点云。训练前把相应 JSON 放进上面的数据根目录。

## 数据准备

公开 GridNet-HD 原始 LAS 应有 `ground_truth` 字段。下面的脚本把官方类别先映射为项目的 7 类，再按 20 m tile、10 m 步长切分：

```bash
python prepare_gridnethd.py \
  --gridnethd_root /data/GridNet-HD --split_json /data/gridnethd_split.json \
  --out_root data/gridnethd/pc7 --pointcept_root "$PWD" \
  --temporary_root /scratch/gridnethd --dino_projection 0 \
  --target_classes 7
```

当前 Stage B 的标注来自已经整理好的最终标签 LAS；与原始传感器 LAS block **点序一致**，并包含 `classif`（历史 11 类标签）、`ptv3_classif`、`label_source` 和 `source_block`。`prepare_final_las_finetune.py` 按实际训练时的 `11→7` 映射、监督权重及原始 RGB 生成 tile。`prepare_finetune_collection.py` 接收场景清单，逐场景调用它：

```json
[
  {
    "scene": "line_a",
    "split": "train_final",
    "labeled_las": "/data/labels/line_a_final.las",
    "original_las": ["/data/raw/line_a_Block_0.las", "/data/raw/line_a_Block_1.las"]
  }
]
```

```bash
python scripts/prepare_finetune_collection.py \
  --manifest /data/stage_b_manifest.json \
  --output-root data/ours_stage_b_7class_v1 \
  --python "$(command -v python)" --num-classes 7 --workers 2
```

固定实验把 T3–T7、pcd_rgb、输电杆塔以及 las_new 的 Block 7–13 留作验证；las_new Block 0–6 加入训练。`scripts/write_stage_b_splits.py` 和 `scripts/split_las_new_tiles_by_source_block.py` 生成与原实验一致的 tile 清单；`splits/stage_b/` 也保留了此次训练使用的清单副本。增加新数据时必须先按完整 block 或线路划分，再制备 tile，避免同一场景进入训练和验证。

## Stage A：公开数据预训练

初始化使用 ScanNet20 SpUNet 骨干；原权重 6 通道输入、20 类输出。转换脚本仅保留 RGB 输入权重，并用种子 42 初始化 7 类头。原权重及训练后 checkpoint 不提交 Git：

```bash
mkdir -p checkpoints/spunet
curl -L --fail -o checkpoints/spunet/scannet20_model.safetensors \
  https://huggingface.co/torch-pointcloud/spunet-v1m1.scannet20.pointcept/resolve/main/model.safetensors
python scripts/prepare_spunet_pretrained.py \
  --source checkpoints/spunet/scannet20_model.safetensors \
  --output checkpoints/spunet/scannet20_rgb3_7class_random_head.pth

CUDA_VISIBLE_DEVICES=4,5,6 python tools/train.py \
  --config-file configs/stage_a.py --num-gpus 3
```

Stage A 默认总 batch 48、100 次数据遍历、每 10 次验证。最优权重在 `exp/gridnethd/spunet_7class_public_stage_a/model/model_best.pth`。

## Stage B：自建数据微调

Stage B 从 Stage A 最优权重初始化，混合公开数据 replay、自建训练线路和 las_new Block 0–6；验证列表 `val_fixed_las_blocks.json` 不含这些训练 block。总 batch 48、30 次遍历、每 3 次验证：

```bash
cp splits/stage_b/train_stage_b_replay.json data/gridnethd/pc7/
cp splits/stage_b/{train_self,train_las_new_blocks,val_fixed_las_blocks}.json \
  data/ours_stage_b_7class_v1/
CUDA_VISIBLE_DEVICES=4,5,6 python tools/train.py \
  --config-file configs/stage_b.py --num-gpus 3
```

最优权重在 `exp/gridnethd/spunet_7class_stage_b_las_new_blocks/model/model_best.pth`。配置中的 `weight`、`save_path`、`data.*.data_root` 可按新机器路径修改；模型结构和 7 类顺序必须与 checkpoint 一致。

## 固定验证集评测

```bash
CUDA_VISIBLE_DEVICES=4 python tools/test.py \
  --config-file configs/stage_b_eval.py --num-gpus 1 \
  --options save_path=exp/spunet_stage_b_eval
```

默认使用 10 组测试增强并输出 tile 预测 `result/*_pred.npy`。如需逐类 Precision、Recall、IoU 和混淆矩阵：

```bash
python scripts/evaluate_spunet_val_fixed.py \
  --data-root data/ours_stage_b_7class_v1 \
  --split data/ours_stage_b_7class_v1/val_fixed_las_blocks.json \
  --result-dir exp/spunet_stage_b_eval/result \
  --output exp/spunet_stage_b_eval/metrics.json
```

分阶段计时可运行 `scripts/profile_spunet_val_blocks.py`，使用 `--output` 指向**新的空结果目录**。该脚本记录读取、预处理、CPU→GPU、模型前向、累加与保存耗时。已经完成的 Stage B 实验在固定 88 tile 上的 10 组增强 mIoU 为 62.31%；此数值不代表单尺度完整 block 的指标。

如已生成完整 block 的 `prediction.las`，且数据清单中的最终标签 LAS 保留了 `source_block`、`classif`、`label_source` 字段，可按完整原始点集评测：

```bash
python scripts/evaluate_spunet_full_blocks.py \
  --result-root outputs/full_blocks \
  --manifest data/ours_stage_b_7class_v1/collection_manifest.json \
  --output outputs/full_blocks/FULL_BLOCK_METRICS.json
```

脚本使用与 Stage B 预处理完全相同的 11→7 类映射，输出整体、逐类、逐场景的 IoU、Precision、Recall、混淆矩阵，并另外统计仅人工标签点的辅助指标。

## 完整原始 block 的 LAS 输出

单场景示例可将原始点云放到 `inference/input/input.las`，输出写到 `inference/output/`；目录结构和完整命令见 [`inference/README.md`](inference/README.md)。

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/run_full_block_inference.py \
  --inputs /data/raw/line_a_Block_0.las \
  --weight exp/gridnethd/spunet_7class_stage_b_las_new_blocks/model/model_best.pth \
  --output-root outputs/full_blocks --work-root work/full_blocks --gpu 4 \
  --single-scale
```

该入口执行原始 LAS 流式切重叠 tile → SpUNet 推理 → 按 `original_index.npy` 投票合并 → 以**原始 LAS**为底稿新增 `classif:uint8`。输出为 `outputs/full_blocks/<输入父目录>/<block>/prediction.las`，旁边有 `prediction.audit.json`；所有原始维度（包括 XYZ、RGB、intensity 和标准 classification）都逐点读回比对。少量切块边界未覆盖点最多允许 10 个，按源点序最近的已覆盖点补标签，并在审计文件中计数；超过阈值直接失败。省略 `--single-scale` 可运行配置中的 10 组测试增强，速度更慢。多 block 可用 `--input-list`，每行是 LAS 路径或 `分组<TAB>LAS路径`。

`v1/` 的模型与此方案权重不兼容。本目录是研发用 Python 方案，未声称已经具备 ONNX、Windows、MIPOT 或 `dist.tar` 交付能力。Pointcept 衍生源码保留原项目 MIT 许可，见 [`LICENSE`](LICENSE)。
