# PcdSeg

输电线路点云语义分割代码版本库，包含两个并列的 7 类 Python 方案：

- [`v1/`](v1/README.md)：LitePT-S，XYZ+RGB；
- [`sparseunet/`](sparseunet/README.md)：Pointcept SpUNet-v1m1，包含公共数据预训练、自建数据微调、固定验证和保留原始属性的完整 LAS block 推理。

本仓库只保存源码与配置，不保存数据集、模型权重、推理结果或编译产物。

具体数据格式、环境和运行命令见各方案目录下的 README。
