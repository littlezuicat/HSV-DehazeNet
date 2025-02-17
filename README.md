# HSVNet (HSV-DehazeFormer)

## 介绍

HSVNet（HSV-DehazeFormer）是一种针对图像去雾任务的深度学习模型，旨在提高去雾效果的视觉质量。该模型基于传统的图像去雾方法，通过在 HSV 色彩空间中处理图像来增强雾霾中的视觉细节，进而改善图像的质量。尽管在 PSNR 和 SSIM 指标上表现一般，HSVNet 在视觉效果方面表现优秀，尤其适合于那些对图像质量要求较高的应用。

目前，HSVNet 的网络架构正在改进中，优化目标是提升其性能和计算效率。

## 目录结构

```
HSVNet/
│
├── data/                  # 数据集目录
│   ├── hazy/              # 含雾图像
│   └── GT/                # 清晰图像
│
├── models/                # 模型相关代码
│   └── hsvnet.py          # HSVNet 网络结构定义
│
├── utils/                 # 工具函数
│   ├── data_loader.py     # 数据加载相关函数
│   └── metrics.py         # PSNR 和 SSIM 计算
│
├── train.py               # 训练脚本
├── test.py                # 测试脚本
├── requirements.txt       # Python 环境依赖
└── README.md              # 本文件
```

## 安装依赖

在开始使用之前，确保已经安装了以下依赖：

```bash
pip install -r requirements.txt
```

`requirements.txt` 包含以下依赖：

- torch
- torchvision
- opencv-python
- scipy
- numpy

## 数据集

HSVNet 使用自定义的含雾图像和清晰图像数据集进行训练与评估。数据集的结构应如下所示：

```
data/
├── hazy/    # 含雾图像
└── GT/      # 清晰图像
```

每对图像的名称应一致，`hazy/` 目录中的图像为含雾图像，`GT/` 目录中的图像为对应的清晰图像。

## 模型训练

在模型训练时，使用以下命令：

```bash
python train.py --epochs 100 --batch_size 16 --lr 0.0001
```

此命令将开始训练模型，`--epochs` 设置训练轮次，`--batch_size` 设置批次大小，`--lr` 设置学习率。

## 模型测试

在训练完模型后，你可以使用以下命令来评估模型的性能：

```bash
python test.py --model_path path_to_trained_model.pth
```

测试结果会输出 PSNR 和 SSIM 等评估指标，并显示去雾后的图像。

## 网络架构

HSVNet 的核心思想是将图像从 RGB 空间转换为 HSV 空间，然后在该空间中进行去雾处理。通过在 HSV 色彩空间内直接进行融合操作，我们能够在保持图像细节的同时去除雾霾，从而提升图像的视觉质量。最终，模型将结果转换回 RGB 空间以便显示。

### 模型架构概述：

1. **输入层：** 输入图像首先被标准化并转换为 HSV 色彩空间。
2. **特征提取层：** 使用卷积神经网络（CNN）提取图像特征。
3. **去雾层：** 在 HSV 空间中进行雾霾去除操作，通过自定义公式进行颜色空间融合。
4. **输出层：** 将去雾后的图像从 HSV 空间转换回 RGB 空间。

## 评估指标

模型的性能通过以下指标进行评估：

- **PSNR（Peak Signal-to-Noise Ratio）：** 用于衡量重建图像与真实图像之间的差异。
- **SSIM（Structural Similarity Index）：** 用于衡量图像结构的相似性。

此外，我们还会展示去雾后的图像效果，以帮助直观评估模型表现。

## 未来工作

1. **架构改进：** 当前网络架构尚在改进中，未来将尝试引入更多高级的网络层来提高 PSNR 和 SSIM。
2. **数据增强：** 引入更多的图像增强技术，进一步提高模型的泛化能力。
3. **速度优化：** 针对模型的计算效率进行优化，减少推理时间。