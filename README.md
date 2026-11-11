# CIFAR-10 残差网络复现与 SE 注意力适用性分析

本仓库是课程论文《基于残差网络的 CIFAR-10 图像分类复现与 SE 注意力适用性分析》的
配套代码、训练日志与模型权重。

## 主要内容

1. **复现评测**：在完整 CIFAR-10 测试集（10 000 张）上重新评测 5 个公开预训练模型
   （ResNet-20/32/56、VGG-16-BN、MobileNetV2），记录 Top-1/Top-5、参数量与推理延迟。
2. **受控对照实验**：以相同的数据增强、优化器与训练轮次，从头训练基线
   **ResNet-20** 与对照模型 **SE-ResNet-20**，检验通道注意力的净效应。
3. **分析**：训练曲线、混淆矩阵、类别级差值、参数量—延迟权衡。

## 主要结果（本仓库实际运行所得）

| 模型 | Top-1/% | Top-5/% | 参数量 | 单张延迟/ms |
|---|---|---|---|---|
| ResNet-20（公开权重） | 92.59 | 99.81 | 272 474 | 1.85 |
| ResNet-32（公开权重） | 93.53 | 99.77 | 466 906 | 2.91 |
| ResNet-56（公开权重） | 94.37 | 99.83 | 855 770 | 4.98 |
| VGG-16-BN（公开权重） | 94.15 | 99.71 | 15 253 578 | 2.23 |
| MobileNetV2（公开权重） | 94.05 | 99.82 | 2 236 682 | 3.92 |
| **ResNet-20（自训练基线）** | **93.00** | 99.73 | 272 474 | 1.89 |
| **SE-ResNet-20（对照模型）** | **93.01** | 99.71 | 275 162 | 3.08 |

结论：SE 模块带来的 Top-1 变化（+0.01 个百分点）远小于训练末段波动
（标准差约 1.11 个百分点），而单张推理延迟增加 62.96%，因此在该规模下
通道注意力不构成有效的精度—开销改进。

## 环境

- Python 3.12，PyTorch 2.11.0+cu128，torchvision 0.26.0+cu128
- NVIDIA GeForce RTX 3060 Laptop GPU（6 GB）
- 详细版本见 `results/environment.json`

## 复现步骤

```bash
# 1) 环境
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt

# 2) 数据集（CIFAR-10）
#    官方源 cs.toronto.edu 在国内常被重置，torchvision 直接 download=True 会卡在几 MB。
#    下面脚本改走 ModelScope 镜像并校验官方 MD5，下完 torchvision 会自动解压。
python get_data.py

# 3) 训练基线与对照模型（各 200 epoch）
python train.py --model resnet20    --epochs 200 --out runs/resnet20_200
python train.py --model se_resnet20 --epochs 200 --out runs/se_resnet20_200

# 4) 评测（精度、混淆矩阵、逐类准确率）
python evaluate_trained.py --checkpoint runs/resnet20_200/best.pt
python evaluate_trained.py --checkpoint runs/se_resnet20_200/best.pt

# 5) 统一测量参数量/延迟/FLOPs
python measure_cost.py

# 6) 复现论文图表
python plot_training.py --run resnet20=runs/resnet20_200/metrics.csv \
                        --run se_resnet20=runs/se_resnet20_200/metrics.csv
python plot_paper_figures.py
python plot_reports.py
```

不想训练时，可直接加载 `weights/` 中的两个权重：

```python
import torch
from se_resnet import MODELS

blob = torch.load("weights/se_resnet20_best.pt", map_location="cpu")
model = MODELS[blob["model"]]()
model.load_state_dict(blob["state_dict"])
model.eval()
```

## 权重文件

| 文件 | 说明 | 大小 |
|---|---|---|
| `weights/resnet20_best.pt` | 基线 ResNet-20（本文训练，Top-1 93.00%） | 1.1 MB |
| `weights/se_resnet20_best.pt` | 对照 SE-ResNet-20（本文训练，Top-1 93.01%） | 1.1 MB |

5 个公开预训练权重体积较大（合计约 76 MB），未纳入本仓库，可在运行时由
`torch.hub` 自动下载：

```python
model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
```

来源：<https://github.com/chenyaofo/pytorch-cifar-models>（BSD-3-Clause）。

`weights/manifest.json` 记录了全部 7 个权重文件的 SHA-256（前 16 位）与来源。

## 目录结构

```
.
├─ train.py                 训练主脚本（ResNet-20 / SE-ResNet-20）
├─ se_resnet.py             模型定义（CIFAR 残差网络 + SE 模块）
├─ evaluate_trained.py      评测：精度、混淆矩阵、逐类准确率
├─ evaluate.py              公开预训练模型批量评测
├─ measure_cost.py          统一的参数量/延迟/FLOPs 测量
├─ plot_*.py                绘图脚本
├─ runs/
│  ├─ resnet20_200/         metrics.csv（逐 epoch）、train_config.json、best.pt
│  └─ se_resnet20_200/
├─ results/                 评测指标、混淆矩阵、逐类准确率、论文插图
└─ weights/                 本文训练的两个权重 + manifest.json
```

## 引用

```
@misc{liu2026cifar,
  author = {Liu Guoqin},
  title  = {Reproduction of ResNets on CIFAR-10 and an Empirical Study of SE Attention},
  year   = {2026},
  url    = {https://github.com/LGQ308/cifar10-resnet-se}
}
```

## 许可

- 本仓库代码与自训练权重：MIT License（见 `LICENSE`）
- 公开预训练权重：遵循上游仓库 BSD-3-Clause（见 `NOTICE`）
