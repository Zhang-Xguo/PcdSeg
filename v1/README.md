# PcdSeg LitePT v1

LitePT-S 输电线路点云 7 类语义分割版本。模型输入为 `XYZ+RGB`，类别定义固定为：

| ID | 类别 |
|---:|---|
| 0 | 杆塔 tower |
| 1 | 导线 conductor |
| 2 | 绝缘子 insulator |
| 3 | 植被 vegetation |
| 4 | 建筑物 building |
| 5 | 地面 ground |
| 6 | 其他 other |

仓库不包含数据和权重。原始 LAS 的 RGB 必须是传感器颜色，不能使用语义着色后的 RGB
作为模型输入。

## 1. 目录

- `models/`：LitePT-S、7 类分割头和损失；
- `datasets/`：数据读取、增强及类别感知裁剪；
- `configs/gridnet/`：训练和验证配置；
- `tools/prepare_labeled_las.py`：有标签 LAS 转训练/验证 tile；
- `tools/prepare_las_inference_tiles.py`：原始完整 LAS 切重叠推理 tile；
- `tools/train.py`、`tools/test.py`：训练和验证；
- `tools/run_full_block_inference.py`：完整 block 端到端推理；
- `tools/merge_block_predictions.py`：将 tile 预测投票回写原始点序并着色；
- `tools/verify_block_prediction.py`：校验点数、XYZ 和点序；
- `tools/export_validation_las.py`：tile 级预测与真值 LAS 可视化。

## 2. 环境

推荐 Linux、Python 3.8、PyTorch 2.1 + CUDA 12.1。当前验证环境还使用：
`spconv 2.3.6`、`torch-scatter 2.1.2`、`flash-attn` 和 `laspy`。

```bash
git clone git@github.com:Zhang-Xguo/PcdSeg.git
cd PcdSeg/v1
export PYTHONPATH="$PWD"
pip install -r requirements.txt
```

CUDA 扩展版本必须与本机 PyTorch/CUDA 匹配。PointROPE CUDA 扩展不可用时会回退到
PyTorch 实现，但速度较慢。

## 3. 从有标签 LAS 建立数据集

LAS 需要包含 XYZ、RGB，以及取值为 `0..6` 的标签字段。默认读取标准
`classification` 字段，也可以用 `--label-field classif` 指定自定义字段。

```bash
python tools/prepare_labeled_las.py \
  --inputs /data/train/*.las \
  --output-root data/custom_7class \
  --split train_final \
  --manifest train_final.json

python tools/prepare_labeled_las.py \
  --inputs /data/val/*.las \
  --output-root data/custom_7class \
  --split val_final \
  --manifest val_final.json
```

输出结构：

```text
data/custom_7class/
├── train_final.json
├── val_final.json
├── train_final/<tile>/{coord,color,segment}.npy
└── val_final/<tile>/{coord,color,segment}.npy
```

默认以 20 m 网格切分，少于 3000 点的 tile 被过滤。训练和验证应按线路或完整 block
提前分开，不能先切 tile 再随机划分，否则会产生空间泄漏。

## 4. 训练

```bash
export PCDSEG_DATA=/absolute/path/data/custom_7class
export PCDSEG_INIT_WEIGHT=/absolute/path/model_best.pth   # 可为空，或使用兼容的 XYZ+RGB 权重
export PCDSEG_SAVE_PATH=exp/my_litept_v1
export PCDSEG_BATCH_SIZE=16
export PCDSEG_WORKERS=4

CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  --config-file configs/gridnet/semseg-litept-small-custom-7class.py \
  --num-gpus 1 --num-machines 1 --machine-rank 0 --dist-url auto
```

多卡时令 `CUDA_VISIBLE_DEVICES=0,1,2` 且 `--num-gpus 3`。`batch_size` 是全局 batch；
显存不足时优先减小 batch。初始化权重必须与 6 通道 `XYZ+RGB` 输入一致；不兼容权重
应先使用 `tools/extract_backbone_checkpoint.py` 转换或仅加载形状匹配层。

## 5. 验证

```bash
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  --config-file configs/gridnet/semseg-litept-small-custom-7class.py \
  --num-gpus 1 \
  --options weight=/absolute/path/model_best.pth \
            save_path=exp/my_litept_v1_eval
```

逐 tile 预测保存在 `exp/my_litept_v1_eval/result/*_pred.npy`，类别值为 0–6，
并与 tile 原始点序一一对应。

## 6. 新增原始 LAS 并输出完整 block 结果

创建输入清单，每行是 `分组<TAB>LAS绝对路径`：

```text
line_a\t/data/raw/line_a/block_0.las
line_a\t/data/raw/line_a/block_1.las
```

然后运行：

```bash
python tools/run_full_block_inference.py \
  --input-list inputs.tsv \
  --output-root outputs/litept_v1 \
  --work-root work/litept_v1 \
  --python "$(command -v python)" \
  --gpu 0 \
  --worker-id gpu0 \
  --config configs/gridnet/semseg-litept-small-custom-7class.py \
  --weight /absolute/path/model_best.pth
```

该流程会：

1. 将完整 LAS 切成 20 m、步长 10 m 的重叠 tile；
2. 逐 tile 推理；
3. 对重叠预测投票；
4. 按原始点序生成与输入点数完全相同的 LAS；
5. 写入 `classification`、`classif` 和 7 类可视化颜色；
6. 自动校验源文件与结果的点数、XYZ 和点序。

最终文件位于 `outputs/litept_v1/<输入LAS父目录名>/<文件名>.las`。`work/` 包含可续跑的
tile、中间预测、日志和状态 JSON。中断后可以执行同一命令：完整准备结果会复用，半成品
准备目录会自动清理重建。

## 7. 复现实验配置

- `semseg-litept-small-gridnet-stage-a.py`：公共 GridNet-HD Stage A；
- `semseg-litept-small-gridnet-stage-b-las-blocks.py`：当前自建 blocks Stage B；
- `semseg-litept-small-custom-7class.py`：供新数据直接复用的通用配置。

Stage A/B 配置不再包含服务器绝对路径，可通过下列环境变量指定：

```text
PCDSEG_STAGE_A_DATA
PCDSEG_STAGE_B_DATA
PCDSEG_DATA
PCDSEG_INIT_WEIGHT
PCDSEG_SAVE_PATH
```

## 8. 发布范围

v1 只发布 Python 源码、配置、说明和开源许可证，不发布 LAS/NPY 数据、模型权重、日志、
缓存、CUDA 二进制或服务器路径。LitePT 基础代码遵循本目录 `LICENSE`。
