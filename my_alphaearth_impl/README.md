# My AlphaEarth-Style Implementation (from scratch)

这是一个**我自己的简化实现版本**，参考了 AlphaEarth Foundations 的核心思路，但代码结构和训练流程按教学和可复现优先重新组织。

> 目标：给你一个可以直接读懂、直接跑起来、直接改造到你自己数据集的起点。

English version: `README_EN.md`

## 1. 这个实现包含什么

- 一个简化的 `STP`（Space-Time-Precision）编码器：
  - **Space**：对时序帧做空间卷积编码。
  - **Time**：对时间维度做轻量聚合（attention pooling）。
  - **Precision**：用残差卷积保留局部细节。
- 一个简单解码器用于重建任务（自监督预训练风格）。
- 一个可运行训练脚本（用随机生成的遥感风格张量演示）。

## 2. 项目结构

```text
my_alphaearth_impl/
├── README.md
├── model.py           # 模型定义（含详细注释）
├── dataset.py         # 伪造时空序列数据集（可替换成真实数据）
└── train_demo.py      # 训练入口（step-by-step）
```

## 3. 环境准备

在仓库根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch
```

## 4. 运行训练 Demo

```bash
python my_alphaearth_impl/train_demo.py \
  --epochs 3 \
  --batch_size 8 \
  --time_steps 6 \
  --image_size 64
```

你会看到每个 epoch 的重建损失下降，并在 `runs/my_alphaearth_impl/` 下保存 checkpoint。

## 5. 如何替换成你自己的数据（关键步骤）

1. 打开 `dataset.py`，把 `SyntheticEarthDataset` 改成读取你自己的样本。
2. 保持返回格式为：
   - `x`: `[T, C, H, W]`（多时相、多通道影像）
   - `target`: `[C, H, W]`（可用最后时刻图像，或你定义的监督目标）
3. 在 `train_demo.py` 中按你的任务替换 `loss_fn`：
   - 重建任务：`MSELoss`
   - 表征学习：可以改为对比损失
   - 监督任务：可改 `CrossEntropyLoss` / `DiceLoss`
4. 调整 `SimpleSTPModel` 的通道数和 embedding 维度适配你的显存。

## 6. 下一步建议

- 接入真实遥感数据（Sentinel-2 / Landsat / MODIS）。
- 增加 mask 机制处理云遮挡与无效像元。
- 增加更强时间建模（Transformer blocks）。
- 增加下游任务头（分类 / 分割 / 回归），将基础表征用于具体业务。

---

如果你愿意，我下一步可以继续帮你把这个 demo 直接改成：
- 可读取 GeoTIFF 的 dataloader；
- 支持多传感器输入；
- 带完整实验配置（YAML + 日志 + 可视化）。
