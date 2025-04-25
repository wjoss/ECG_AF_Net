# 搭建数据集

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split

# 自定义数据集类
class ECG_Dataset(Dataset):
    def __init__(self, data_path, mode="train", labeled_only=True):
        """
        Args:
            data_path: .npy文件路径
            mode: 数据集模式 ("train"/"test")
            labeled_only: 是否仅加载有标签数据（仅对训练集有效）
        """
        # 加载原始数据
        self.data = np.load(data_path).astype(np.float64)
        self.mode = mode
        self.labeled_only = labeled_only
        
        # 构建标签系统（仅训练集需要）
        if mode == "train":
            # 初始化所有标签为-1（无标签）
            self.labels = np.full(len(self.data), -1, dtype=np.int64)
            
            # 前500条：房颤（标签1）
            self.labels[:500] = 1
            
            # 501-1000条：非房颤（标签0）
            self.labels[500:1000] = 0
            
            # 是否仅使用有标签数据
            if labeled_only:
                self.data = self.data[:1000]
                self.labels = self.labels[:1000]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取信号数据并添加通道维度 (1, 4000)
        signal = torch.from_numpy(self.data[idx]).unsqueeze(0)
        
        if self.mode == "test":
            # 测试集默认返回伪标签-1
            return signal, -1
        else:
            return signal, torch.tensor(self.labels[idx], dtype=torch.long)
    
    def get_splits(self, train_rate=0.85):
        # 拆分数据集确定训练集和测试集的比例
        test_size = round(train_rate * len(self.data))
        train_size = len(self.data) - test_size
        # 根据尺寸划分训练集和测试集并返回
        return random_split(self, [train_size, test_size])


# 使用示例
if __name__ == "__main__":
    # 数据集路径
    train_path = os.path.join("data", "train", "traindata.npy")
    test_path = os.path.join("data", "test", "testdata.npy")

    # 创建数据集实例
    train_dataset = ECG_Dataset(train_path, mode="train", labeled_only=True)
    full_train_dataset = ECG_Dataset(train_path, mode="train", labeled_only=False)
    test_dataset = ECG_Dataset(test_path, mode="test")

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

