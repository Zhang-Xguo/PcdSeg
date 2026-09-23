# 单场景 LAS 推理示例目录

该目录与 `v1/inference/` 使用相同的输入、输出放置方式。从
`sparseunet/` 目录运行时，先放置文件：

```text
inference/
├── input/
│   ├── input.las          # 必需：需要推理的原始 LAS
│   └── ground_truth.las   # 可选：同点数、同点序的 7 类真值
└── output/
    └── input/
        └── input/
            ├── prediction.las
            └── prediction.audit.json
```

`input.las` 可以包含 XYZ、RGB、intensity、标准 classification 以及其他 LAS
维度。输出以原始 LAS 为底稿，保留原有点数、点序和全部字段，只新增
`classif:uint8`，取值为 0–6。原始 RGB 不会被类别色板覆盖。

`ground_truth.las` 不参与推理，仅供后续评测。它必须与输入点云保持相同点数、
XYZ 和点序，7 类真值应为 0–6。完整验证集评测采用带 `source_block` 的场景级
最终标签 LAS，具体见主 README 的完整 block 评测命令。

准备好模型权重后，在 `sparseunet/` 下执行：

```bash
export PYTHONPATH="$PWD"
python scripts/run_full_block_inference.py \
  --inputs inference/input/input.las \
  --weight /absolute/path/model_best.pth \
  --output-root inference/output \
  --work-root work/inference_example \
  --gpu 0 \
  --single-scale
```

结果位于：

```text
inference/output/input/input/prediction.las
inference/output/input/input/prediction.audit.json
```

`prediction.audit.json` 会确认原始字段逐点保留，并报告类别点数和切块边界补点
数量。需要与固定验证协议一致时可去掉 `--single-scale`，此时使用配置中的
10 组测试增强，速度会明显变慢。
