# 卷积层

import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_channels=1, seq_length=4000):
        super(CNN, self).__init__()
        
        # 四层卷积
        self.features = nn.Sequential(
            # Layer 1: 快速降维 + 宏观特征提取
            nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=50, stride=3, padding=25),  # 输入: (batch, 1, 4000)
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 输出形状: (batch, 16, 666)
            
            # Layer 2: 中等粒度特征
            nn.Conv1d(16, 32, kernel_size=20, stride=2, padding=10),  # 输出: (batch, 32, 333)
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),  # 输出: (batch, 32, 166)
            
            # Layer 3: 局部细节特征
            nn.Conv1d(32, 64, kernel_size=10, stride=1, padding=5),  # 输出: (batch, 64, 166)
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),  # 输出: (batch, 64, 83)
            
            # Layer 4: 深层抽象特征
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),  # 输出: (batch, 128, 83)
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # 全局平均池化 → (batch, 128, 1)
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Flatten(),  # 展平 → (batch, 128)
            nn.Linear(128, 64),  # 全连接 → (batch, 64)
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),  # 二分类输出
            nn.Sigmoid()  # Sigmoid激活概率输出
        )

    def forward(self, x):
        # 输入形状检查: (batch, channels, length)
        if x.dim() == 2:  # 处理未添加通道维度的输入
            x = x.unsqueeze(1)  # (batch, length) → (batch, 1, length)
            
        x = self.features(x)
        return self.classifier(x)

# 示例用法
if __name__ == "__main__":
    # 模型初始化
    model = CNN()
    print(model)
    
    # 验证输入输出形状
    dummy_input = torch.randn(32, 1, 4000)  # 模拟batch_size=32的输入
    output = model(dummy_input)
    print(f"\n输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}  # 二分类概率")