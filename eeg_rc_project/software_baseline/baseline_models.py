"""
软件基线模型架构定义。
所有模型接收4频带EEG输入 (batch, 4, 512)，输出2类分类结果。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Baseline_CNN1D(nn.Module):
    """
    轻量级1D CNN基线模型。
    输入: (batch, 4, 512)
    输出: (batch, 2)
    """
    def __init__(self, in_channels=4, num_classes=2):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 512 -> 256
            
            # Block 2
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 256 -> 128
            
            # Block 3
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 128 -> 64
            
            # Block 4
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # 64 -> 1
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # x: (batch, 4, 512)
        x = self.features(x)
        x = x.view(x.size(0), -1)  # (batch, 256)
        return self.classifier(x)


class Baseline_CNN_LSTM(nn.Module):
    """
    CNN + LSTM混合模型。
    CNN提取局部特征，LSTM捕捉时序依赖。
    输入: (batch, 4, 512)
    输出: (batch, 2)
    """
    def __init__(self, in_channels=4, num_classes=2, lstm_hidden=64):
        super().__init__()
        
        # CNN特征提取
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 512 -> 256
            
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 256 -> 128
        )
        
        # LSTM时序建模
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            dropout=0
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # x: (batch, 4, 512)
        # CNN特征提取
        x = self.cnn(x)  # (batch, 64, 128)
        
        # 转置为 (batch, seq_len, features) 供LSTM使用
        x = x.transpose(1, 2)  # (batch, 128, 64)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (batch, 128, 64), h_n: (1, batch, 64)
        
        # 取最后时刻的隐藏状态
        x = h_n.squeeze(0)  # (batch, 64)
        
        return self.classifier(x)


class Baseline_Transformer(nn.Module):
    """
    轻量级Transformer模型。
    将EEG分块为patches，添加位置编码，通过Transformer编码器处理。
    输入: (batch, 4, 512)
    输出: (batch, 2)
    """
    def __init__(self, in_channels=4, num_classes=2, 
                 patch_size=32, d_model=64, nhead=4, 
                 num_layers=3, dim_feedforward=128, dropout=0.1):
        super().__init__()
        
        self.patch_size = patch_size
        self.d_model = d_model
        self.in_channels = in_channels
        
        # 计算patches数量: 512 / 32 = 16
        self.num_patches = 512 // patch_size
        
        # Patch嵌入: 将 (4, 32) 映射到 d_model
        self.patch_embed = nn.Linear(in_channels * patch_size, d_model)
        
        # 位置编码 (可学习)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # x: (batch, 4, 512)
        batch_size = x.size(0)
        
        # 分块: (batch, 4, 512) -> (batch, 4, 16, 32)
        x = x.view(batch_size, self.in_channels, self.num_patches, self.patch_size)
        
        # 转置: (batch, 16, 4, 32)
        x = x.permute(0, 2, 1, 3)  # (batch, num_patches, channels, patch_size)
        
        # 展平每个patch: (batch, 16, 4*32=128)
        x = x.reshape(batch_size, self.num_patches, -1)
        
        # Patch嵌入: (batch, 16, d_model)
        x = self.patch_embed(x)
        
        # 添加位置编码
        x = x + self.pos_embed
        
        # Transformer编码
        x = self.transformer(x)  # (batch, 16, d_model)
        
        # 平均池化
        x = x.mean(dim=1)  # (batch, d_model)
        
        return self.classifier(x)


class Baseline_DeepCNN(nn.Module):
    """
    更深的CNN基线模型，用于对比深度增加的影响。
    比Baseline_CNN1D更深，但保持参数量可控。
    输入: (batch, 4, 512)
    输出: (batch, 2)
    """
    def __init__(self, in_channels=4, num_classes=2):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 512 -> 256
            
            # Block 2
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 256 -> 128
            
            # Block 3
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 128 -> 64
            
            # Block 4
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 64 -> 32
            
            # Block 5
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 32 -> 16
            
            # Block 6
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # 16 -> 1
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # x: (batch, 4, 512)
        x = self.features(x)
        x = x.view(x.size(0), -1)  # (batch, 256)
        return self.classifier(x)


def count_parameters(model):
    """计算模型可训练参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 测试模型创建和参数量
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    models = {
        "Baseline_CNN1D": Baseline_CNN1D(),
        "Baseline_CNN_LSTM": Baseline_CNN_LSTM(),
        "Baseline_Transformer": Baseline_Transformer(),
        "Baseline_DeepCNN": Baseline_DeepCNN()
    }
    
    dummy_input = torch.randn(2, 4, 512).to(device)
    
    print("Model Parameter Counts:")
    print("-" * 50)
    for name, model in models.items():
        model = model.to(device)
        params = count_parameters(model)
        output = model(dummy_input)
        print(f"{name:25s}: {params:>10,} params | Output shape: {output.shape}")
    print("-" * 50)
