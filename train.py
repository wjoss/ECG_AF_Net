# -*- coding: UTF-8 -*-
# ----------------------
# 导入需要的包
# ----------------------
"""第三方库"""
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

"""自定义模块"""
import my_func
from model.CNN import CNN
from data_processing.Dealdata import ECG_Datadeal
from data_processing.Dataset import ECG_Dataset

# ----------------------
# 数据准备与预处理
# ----------------------

# 使用 os.path 自动处理路径分隔符
train_path = os.path.join("data", "train", "traindata.mat")
test_path = os.path.join("data", "test", "testdata.mat")

# 预处理数据并返回保存路径（.npy）
trainset_path = ECG_Datadeal(train_path)
testset_path  = ECG_Datadeal(test_path)

# ----------------------
# 主程序入口
# ----------------------

if __name__ == "__main__":
    # 配置训练参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    num_epochs = 40
    batch_size = 8
    learning_rate = 0.001

    # 构建数据集与加载器
    dataset = ECG_Dataset(trainset_path, mode="train", labeled_only=True)
    train_dataset, test_dataset = dataset.get_splits(0.8)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    model = CNN().to(device)

    # 定义损失函数与优化器
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 可选：保存最佳模型
    save_path = os.path.join("save", "Bestmodel_CNN.pth")

    # 训练模型
    print("开始训练...")
    result = my_func.train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device,
        save_best=True,                     # 保存验证集上最优模型
        save_path=save_path                 # 模型保存路径
    )

    # 提取训练结果
    train_loss = result['train_loss']
    valid_loss = result['valid_loss']
    train_acc  = result['train_acc']
    valid_acc  = result['valid_acc']

    # 可视化训练过程
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(12, 5))
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, valid_loss, label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label='Train Acc')
    plt.plot(epochs, valid_acc, label='Val Acc')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()