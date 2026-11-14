# YOLO Industrial Defect Detection

基于YOLO的工业表面缺陷检测项目，面向工业工件图像中的表面缺陷定位与类别识别。

## 项目简介

本项目使用Ultralytics YOLO构建目标检测流程，覆盖数据集配置、模型训练、模型验证和单张/批量图像推理。可根据具体工业场景替换数据集、类别名称及模型权重。

> 说明：当前仓库提供可复用的项目代码框架。请在完成真实训练和测试后，再补充实际指标、检测截图及实验结论。

## 功能模块

- 数据集配置：采用YOLO标准目录结构和`data.yaml`配置文件。
- 模型训练：支持加载预训练权重进行迁移学习。
- 模型验证：输出Precision、Recall、mAP等目标检测指标。
- 图像推理：对输入图像进行缺陷定位和类别预测，并保存可视化结果。
- 简易应用：提供基于Gradio的图像检测界面。

## 环境配置

建议使用Python 3.10或3.11，并根据本机CUDA版本安装匹配的PyTorch。

```bash
conda create -n yolo-defect python=3.10 -y
conda activate yolo-defect
pip install -r requirements.txt
```

如需GPU训练，请先根据设备和CUDA版本安装对应版本的PyTorch，参考：
https://pytorch.org/get-started/locally/

## 数据集准备

将数据集整理为YOLO检测格式：

```text
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

每张图片对应一个同名`.txt`标签文件。每行标签格式为：

```text
class_id x_center y_center width height
```

坐标采用相对于图像宽高归一化后的数值，类别编号从0开始。

编辑`configs/data.yaml`，将路径和类别名称替换为自己的数据集信息。

## 模型训练

```bash
python scripts/train.py --data configs/data.yaml --model yolo11n.pt --epochs 100 --imgsz 640 --batch 16
```

`--model`可替换为实际使用的YOLO模型权重或模型配置文件。显存不足时可适当减小`--batch`。

## 模型验证

```bash
python scripts/val.py --weights runs/detect/train/weights/best.pt --data configs/data.yaml
```

验证结果包含Precision、Recall、mAP50和mAP50-95等指标，具体以运行输出为准。

## 图像推理

```bash
python scripts/predict.py --weights runs/detect/train/weights/best.pt --source path/to/image.jpg
```

预测可视化结果默认保存至`runs/predict/`。

## 启动简易检测界面

```bash
python app/app.py --weights runs/detect/train/weights/best.pt
```

启动后根据终端提示在浏览器中打开本地界面，上传图片即可查看检测结果。

## 项目结构

```text
YOLO-Defect-Detection/
├── app/
│   └── app.py
├── configs/
│   └── data.yaml
├── scripts/
│   ├── train.py
│   ├── val.py
│   └── predict.py
├── docs/
├── requirements.txt
├── .gitignore
└── README.md
```

## 后续完善建议

1. 补充实际数据集来源、样本数量及缺陷类别说明。
2. 记录训练环境、模型版本和关键超参数。
3. 在`docs/`中添加真实检测效果图和训练曲线。
4. 报告真实测试集上的Precision、Recall、mAP及推理速度。
5. 如进行了模型对比或消融实验，可增加实验记录表。

## License

本项目采用MIT License，使用前请确认所包含代码和数据均具有相应的开源或使用授权。
