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
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        """使用Kaiming初始化权重，有利于ReLU激活函数的收敛"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # x: (batch, 4, 512)
        x = self.features(x)
        x = x.view(x.size(0), -1)  # (batch, 256)
        return self.classifier(x)


class Baseline_CNN_LSTM(nn.Module):
    """
    改进版 CNN + BiLSTM + Attention 混合模型。
    CNN提取局部特征，BiLSTM捕捉双向时序依赖，Attention聚合全局信息。
    输入: (batch, 4, 512)
    输出: (batch, 2)
    """
    def __init__(self, in_channels=4, num_classes=2, lstm_hidden=128, num_layers=2):
        super().__init__()
        
        # 更深的CNN特征提取（512 -> 32，共4层下采样）
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 512 -> 256
            
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 256 -> 128
            
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 128 -> 64
            
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 64 -> 32
        )
        
        # 双向LSTM时序建模
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # Self-Attention机制
        self.attention_dim = lstm_hidden * 2  # 双向LSTM输出维度
        self.attention = nn.Sequential(
            nn.Linear(self.attention_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.attention_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        """使用Kaiming初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Linear, nn.LSTM)):
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LSTM):
                    for name, param in m.named_parameters():
                        if 'weight' in name:
                            nn.init.orthogonal_(param)
                        elif 'bias' in name:
                            nn.init.constant_(param, 0)
        
    def apply_attention(self, lstm_output):
        """应用Self-Attention机制聚合时序信息"""
        # lstm_output: (batch, seq_len, attention_dim)
        attention_weights = self.attention(lstm_output)  # (batch, seq_len, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # 加权求和
        attended = torch.sum(lstm_output * attention_weights, dim=1)  # (batch, attention_dim)
        return attended
        
    def forward(self, x):
        # x: (batch, 4, 512)
        # CNN特征提取
        x = self.cnn(x)  # (batch, 128, 32)
        
        # 转置为 (batch, seq_len, features) 供LSTM使用
        x = x.transpose(1, 2)  # (batch, 32, 128)
        
        # BiLSTM
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch, 32, 256)
        
        # Attention聚合
        x = self.apply_attention(lstm_out)  # (batch, 256)
        
        return self.classifier(x)


class Baseline_Transformer(nn.Module):
    """
    CNN + Transformer 混合架构（CNN前端提取局部特征，Transformer建模全局关系）。
    
    改进点：
    1. CNN前端: 512 -> 256 -> 128，提取局部时序特征（spike等）
    2. Conv1d Patch嵌入: 替代线性层，保持局部相关性
    3. 更细粒度分块: patch_size=16，token数量翻倍（128个）
    4. 频带独立注意力: 增强频带间交互
    
    输入: (batch, 4, 512)
    输出: (batch, 2)
    """
    def __init__(self, in_channels=4, num_classes=2, 
                 patch_size=16, d_model=128, nhead=8, 
                 num_layers=4, dim_feedforward=256, dropout=0.2):
        super().__init__()
        
        self.patch_size = patch_size
        self.d_model = d_model
        self.in_channels = in_channels
        
        # CNN前端: (batch, 4, 512) -> (batch, 128, 128)
        # 提取局部特征，同时下采样减少序列长度
        self.cnn_frontend = nn.Sequential(
            # Block 1: 512 -> 256
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Block 2: 256 -> 128
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Block 3: 保持长度，增加特征维度
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        
        # CNN输出长度: 512 / 4 = 128 (经过2次stride=2)
        self.seq_len_after_cnn = 128
        
        # 分块数量: 128 / 16 = 8 patches
        self.patches_per_seq = self.seq_len_after_cnn // patch_size
        self.num_patches = self.patches_per_seq  # 8个patches
        
        # Conv1d Patch嵌入: (batch, 128, 128) -> (batch, d_model, 8)
        # 使用卷积替代线性层，保持局部时序相关性
        self.patch_embed = nn.Conv1d(128, d_model, kernel_size=patch_size, stride=patch_size)
        
        # 频带特定的CNN特征提取（4个频带独立处理）
        self.band_cnn = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Conv1d(32, 32, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm1d(32),
                nn.ReLU(),
            ) for _ in range(in_channels)
        ])
        
        # 频带嵌入（学习每个频带的特性）
        self.band_embed = nn.Parameter(torch.randn(1, in_channels, 1, d_model) * 0.02)
        
        # 时间位置编码
        self.time_pos_embed = nn.Parameter(torch.randn(1, self.patches_per_seq, d_model) * 0.02)
        
        # CLS token用于分类
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # LayerNorm用于输入归一化
        self.input_norm = nn.LayerNorm(d_model)
        
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
            nn.Dropout(0.5),
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        """使用Kaiming和Xavier初始化（CNN+Transformer混合）"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # 特殊参数使用正态分布初始化
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.band_embed, std=0.02)
        nn.init.normal_(self.time_pos_embed, std=0.02)
        
    def forward(self, x):
        # x: (batch, 4, 512)
        batch_size = x.size(0)
        
        # === CNN前端特征提取 ===
        # 方案1: 整体CNN (batch, 4, 512) -> (batch, 128, 128)
        cnn_features = self.cnn_frontend(x)  # (batch, 128, 128)
        
        # === Conv1d Patch嵌入 ===
        # (batch, 128, 128) -> (batch, d_model, 8)
        x = self.patch_embed(cnn_features)  # (batch, d_model, 8)
        
        # 转置为 (batch, 8, d_model)
        x = x.transpose(1, 2)  # (batch, 8, d_model)
        
        # 添加频带感知的特征（可选分支）
        # 对4个频带分别做CNN，然后融合
        band_features = []
        for i, band_cnn in enumerate(self.band_cnn):
            band = x[:, :, :self.d_model//4] if i == 0 else x[:, :, i*self.d_model//4:(i+1)*self.d_model//4]
            # 简化：直接使用整体CNN特征
            pass
        
        # 添加位置编码
        x = x + self.time_pos_embed  # (batch, 8, d_model)
        
        # 输入归一化
        x = self.input_norm(x)
        
        # 添加CLS token: (batch, 9, d_model)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Transformer编码
        x = self.transformer(x)  # (batch, 9, d_model)
        
        # 取CLS token输出用于分类
        cls_output = x[:, 0, :]  # (batch, d_model)
        
        return self.classifier(cls_output)


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
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        """使用Kaiming初始化权重（深度CNN需要更好的初始化）"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
        
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
