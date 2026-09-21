# 单场景 LAS 推理目录

从 `v1/` 运行时，文件放置如下：

```text
inference/
├── input/
│   ├── input.las          # 必需：原始 XYZ+RGB 点云
│   └── ground_truth.las   # 可选：同一点序的 0–6 类真值
└── output/
    └── input/
        └── input.las      # 推理完成后自动生成的彩色预测
```

`ground_truth.las` 不参与模型推理，也不会被自动当作输入。请确保它与
`input.las` 点数、XYZ 和点序一致；类别放在 LAS `classification` 字段，
或者自行记录所用的自定义字段。没有真值也可以运行推理。

准备好模型权重和环境后，在 `v1/` 下执行：

```bash
export PYTHONPATH="$PWD"
python tools/run_full_block_inference.py \
  --input inference/input/input.las \
  --output-root inference/output \
  --work-root work/litept_v1 \
  --python "$(command -v python)" \
  --gpu 0 \
  --weight /absolute/path/model_best.pth
```

输出保留原始 LAS 的点数与点序，预测类别写入 `classification` 和 `classif`，
RGB 改为七类显示颜色。`work/` 保存切片、日志和状态，方便失败后续跑。
不要把模型预测的语义颜色当作下次训练或推理的原始 RGB。
