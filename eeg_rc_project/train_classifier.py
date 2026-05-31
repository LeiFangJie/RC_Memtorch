"""
此模块负责训练基于全连接层 (FC) 的癫痫发作分类器。
实现了基于患者级别的流式数据集加载器，避免内存溢出，
并计算各类评估指标 (Accuracy, Precision, Recall, F1, 混淆矩阵)。
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from config import *
import random
import copy
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class PatientStreamingDataset(Dataset):
    """
    按患者级别进行数据流式加载的 PyTorch Dataset。
    只在内存中保留当前正在读取的患者特征，以满足大规模 EEG 数据的内存限制要求。
    """
    def __init__(self, patient_ids, data_dir=PROCESSED_DATA_DIR):
        self.patient_ids = patient_ids
        self.data_dir = data_dir
        
        self.file_paths = []
        self.lengths = []
        self.cumulative_lengths = [0]
        
        for pid in patient_ids:
            path = os.path.join(data_dir, f"{pid}_features.pt")
            if os.path.exists(path):
                # Just load to get length, then delete
                features, labels = torch.load(path)
                length = len(labels)
                self.file_paths.append(path)
                self.lengths.append(length)
                self.cumulative_lengths.append(self.cumulative_lengths[-1] + length)
                del features, labels
                
        self.total_length = self.cumulative_lengths[-1]
        
        # Cache for the currently loaded patient
        self.current_file_idx = -1
        self.current_features = None
        self.current_labels = None
        
    def __len__(self):
        return self.total_length
        
    def __getitem__(self, idx):
        # Find which file this idx belongs to
        # A simple binary search could work, but linear is fine for < 24 files
        file_idx = next(i for i, cum_len in enumerate(self.cumulative_lengths[1:]) if idx < cum_len)
        
        # Local index within that file
        local_idx = idx - self.cumulative_lengths[file_idx]
        
        # Load file if not in cache
        if file_idx != self.current_file_idx:
            self.current_features, self.current_labels = torch.load(self.file_paths[file_idx])
            self.current_file_idx = file_idx
            
        feature = self.current_features[local_idx]
        label = self.current_labels[local_idx]
        
        return feature, label

import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss 解决极度不平衡分类问题，将模型注意力集中在难分类的少数类（癫痫）样本上。
    """
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [Batch, NumClasses]
        # targets: [Batch]
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Apply alpha weighting
        # Assuming class 1 is the minority class (seizure)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class CNN1DClassifier(nn.Module):
    """
    1D 卷积神经网络分类器。
    它将展平的 1024 维特征重塑回 [Batch, 4, 256] 的形状，
    然后沿时间维度应用 1D 卷积，以提取跨通道的局部时序模式。
    """
    def __init__(self, input_dim=1024, num_channels=4, seq_len=256, output_dim=2):
        super().__init__()
        self.num_channels = num_channels
        self.seq_len = seq_len
        
        self.features = nn.Sequential(
            # Input: [Batch, 4, 256]
            nn.Conv1d(in_channels=num_channels, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), # [Batch, 32, 128]
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), # [Batch, 64, 64]
            
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) # [Batch, 128, 1]
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim)
        )
        
    def forward(self, x):
        # x is [Batch, 1024]
        batch_size = x.size(0)
        # Reshape to [Batch, Channels, TimeSteps] -> [Batch, 4, 256]
        x = x.view(batch_size, self.num_channels, self.seq_len)
        
        x = self.features(x)
        x = x.view(batch_size, -1) # [Batch, 128]
        return self.classifier(x)

class ComplexFC(nn.Module):
    """
    带有 BatchNorm 和 Dropout 的深度多层感知机。
    相比于 SimpleFC，它具有更强的非线性表达能力，能更好地切分 RC 提取的高维流形特征。
    """
    def __init__(self, input_dim=1024, hidden_dims=[512, 128, 64], output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dims[2], output_dim)
        )
        
        # 权重初始化（Kaiming He初始化配合ReLU激活）
        self._initialize_weights()
        
    def _initialize_weights(self):
        """对网络中的线性层进行Kaiming He初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        return self.net(x)

class SimpleFC(nn.Module):
    """
    用于对 RC 储层特征进行二分类的简单全连接网络。
    """
    def __init__(self, input_dim=1024, hidden_dim=128, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

def plot_confusion_matrix_figure(cm, save_path):
    """
    使用 Seaborn 绘制并保存混淆矩阵热力图。
    
    参数:
        cm (ndarray): 混淆矩阵数据
        save_path (str): 保存图片的路径
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Pre-ictal'], 
                yticklabels=['Normal', 'Pre-ictal'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved confusion matrix plot to {save_path}")

def get_class_weights(dataset):
    print("Calculating class weights...")
    total_0 = 0
    total_1 = 0
    for path in dataset.file_paths:
        _, labels = torch.load(path)
        total_1 += labels.sum().item()
        total_0 += len(labels) - labels.sum().item()
    
    print(f"Total negative (0): {total_0}, Total positive (1): {total_1}")
    weight_0 = 1.0 / total_0 if total_0 > 0 else 0
    weight_1 = 1.0 / total_1 if total_1 > 0 else 0
    
    weights = []
    for path in dataset.file_paths:
        _, labels = torch.load(path)
        w = torch.where(labels == 1, weight_1, weight_0)
        weights.append(w)
        
    return torch.cat(weights).double()

class SinglePatientDataset(Dataset):
    """
    单患者流式数据集。
    接收指定患者的预处理特征和标签，并通过传入的 indices 支持对数据的重排和切分（用于分层抽样和训练集/测试集划分）。
    """
    def __init__(self, patient_id, indices=None):
        feat_path = os.path.join(PROCESSED_DATA_DIR, f"{patient_id}_features.pt")
        self.features, self.labels = torch.load(feat_path)
        
        if indices is not None:
            self.features = self.features[indices]
            self.labels = self.labels[indices]
            
        # Optional: Print distribution
        total_1 = self.labels.sum().item()
        total_0 = len(self.labels) - total_1
        print(f"Dataset Loaded - Size: {len(self.labels)} (Normal: {total_0}, Seizure: {total_1})")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def train_model():
    """
    主训练流程：
    1. 扫描已处理的患者特征文件
    2. 按患者级别进行 8:2 划分
    3. 构建流式 DataLoader 并计算类别权重以解决极度不平衡问题
    4. 训练模型并在每个 Epoch 结束后输出评估指标
    5. 保存训练好的 FC 模型权重
    """
    # Fix random seed
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 确保CUDA运算的确定性（关闭非确定性算法）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Check available files
    available_patients = []
    for f in os.listdir(PROCESSED_DATA_DIR):
        if f.endswith('_features.pt'):
            pid = f.split('_')[0]
            if pid not in available_patients:
                available_patients.append(pid)
                
    if not available_patients:
        print("No processed feature files found.")
        return
        
    available_patients.sort()
    print(f"Available patients: {available_patients}")
    
    # If only 1 patient is available (e.g. testing)
    is_single_patient = len(available_patients) == 1
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 切换为 CNN1DClassifier
    model = CNN1DClassifier(input_dim=1024, output_dim=2).to(device)
    
    # 既然我们用了 WeightedRandomSampler 进行类别平衡采样，模型看到的 batches 已经是接近 1:1 的。
    # 此时不需要再用 Focal Loss 的 alpha 偏袒少数类（这会导致网络走向另一个极端，或者训练不稳定）。
    # 使用标准的 CrossEntropyLoss 配合重采样即可，或者保留 FocalLoss 但设 alpha=0.5
    criterion = FocalLoss(alpha=0.5, gamma=1)
    
    # 降低学习率，加入权重衰减防止过拟合
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    
    epochs = 50
    
    if is_single_patient:
        print("--- Single Patient Mode: Window-based random split ---")
        patient_id = available_patients[0]
        # Load full patient to get length
        temp_path = os.path.join(PROCESSED_DATA_DIR, f"{patient_id}_features.pt")
        _, labels = torch.load(temp_path)
        total_samples = len(labels)
        
        # 窗口级别分层打乱划分 8:2 (Stratified Split)
        # 这保证了训练集和测试集都有相同比例的癫痫发作样本
        normal_indices = np.where(labels.numpy() == 0)[0]
        seizure_indices = np.where(labels.numpy() == 1)[0]
        
        np.random.shuffle(normal_indices)
        np.random.shuffle(seizure_indices)
        
        normal_split = int(len(normal_indices) * (1 - TEST_PATIENTS_RATIO))
        seizure_split = int(len(seizure_indices) * (1 - TEST_PATIENTS_RATIO))
        
        train_indices = np.concatenate([normal_indices[:normal_split], seizure_indices[:seizure_split]])
        test_indices = np.concatenate([normal_indices[normal_split:], seizure_indices[seizure_split:]])
        
        np.random.shuffle(train_indices)
        # 对测试集索引进行排序，以恢复其时间序列的相对顺序，这对于后续的平滑后处理（滑动窗口）至关重要
        test_indices = np.sort(test_indices)
        
        train_dataset = SinglePatientDataset(patient_id, indices=train_indices)
        test_dataset = SinglePatientDataset(patient_id, indices=test_indices)
        
        # 为训练集计算权重并使用 WeightedRandomSampler
        total_1 = train_dataset.labels.sum().item()
        total_0 = len(train_dataset.labels) - total_1
        weight_0 = 1.0 / total_0 if total_0 > 0 else 0
        weight_1 = 1.0 / total_1 if total_1 > 0 else 0
        sample_weights = torch.where(train_dataset.labels == 1, weight_1, weight_0).double()
        
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=256, sampler=sampler)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
        
        # 追踪最佳F1分数
        best_f1 = 0.0
        best_epoch = 0
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            print(f"Epoch {epoch+1}/{epochs} starting...")
            for i, (features, labels) in enumerate(train_loader):
                features, labels = features.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
            print(f"Epoch {epoch+1} Train Loss: {total_loss/len(train_loader):.4f}, Train Acc: {100.*correct/total:.2f}%")
            
            # Evaluate and track best F1
            current_f1 = evaluate_model(model, test_loader, device, epoch, epochs, is_best=False)
            
            # 更新最佳F1
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_epoch = epoch + 1
                # 保存最佳模型状态（深拷贝避免后续训练污染）
                best_model_state = copy.deepcopy(model.state_dict())
                print(f"  *** New best F1: {best_f1:.4f} at Epoch {best_epoch} ***")
            
        # 训练结束，加载并保存最佳模型
        if best_f1 > 0:
            model.load_state_dict(best_model_state)
            print(f"\n{'='*50}")
            print(f"Training Complete - Loading Best Model (Epoch {best_epoch}, F1={best_f1:.4f})")
            print(f"{'='*50}\n")
            # 最终评估并保存最佳模型
            evaluate_model(model, test_loader, device, best_epoch-1, epochs, is_best=True)
            
    else:
        print("--- Full Dataset Mode: Chunk-based Patient Streaming Training ---")
        split_idx = max(1, int(len(available_patients) * (1 - TEST_PATIENTS_RATIO)))
        train_patients = available_patients[:split_idx]
        test_patients = available_patients[split_idx:]
        print(f"Train patients ({len(train_patients)}): {train_patients}")
        print(f"Test patients ({len(test_patients)}): {test_patients}")
        
        # 测试集可以保持流式加载，因为只是前向传播，不保存计算图，OOM 风险低
        test_dataset = PatientStreamingDataset(test_patients)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
        
        # 追踪最佳F1分数
        best_f1 = 0.0
        best_epoch = 0
        
        for epoch in range(epochs):
            model.train()
            # Shuffle patients every epoch for randomness
            random.shuffle(train_patients)
            
            epoch_loss = 0
            epoch_batches = 0
            
            print(f"Epoch {epoch+1}/{epochs} starting...")
            # 逐患者加载并训练（Chunk-based）
            for pid in train_patients:
                # print(f"  Training on patient {pid}...")
                patient_dataset = SinglePatientDataset(pid)
                
                if len(patient_dataset) == 0:
                    continue
                    
                total_1 = patient_dataset.labels.sum().item()
                total_0 = len(patient_dataset.labels) - total_1
                
                # 如果这个患者全是正常样本，为了防止极度失衡，可以进行随机欠采样，或者降低权重
                if total_1 == 0:
                    # 随机抽样部分正常数据训练，避免模型过度拟合0
                    sampler = None
                    train_loader = DataLoader(patient_dataset, batch_size=256, shuffle=True)
                else:
                    weight_0 = 1.0 / total_0 if total_0 > 0 else 0
                    weight_1 = 1.0 / total_1 if total_1 > 0 else 0
                    sample_weights = torch.where(patient_dataset.labels == 1, weight_1, weight_0).double()
                    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
                    train_loader = DataLoader(patient_dataset, batch_size=256, sampler=sampler)
                
                for features, labels in train_loader:
                    features, labels = features.to(device), labels.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    epoch_batches += 1
            
            print(f"Epoch {epoch+1} Train Loss: {epoch_loss/epoch_batches if epoch_batches > 0 else 0:.4f}")
            # Evaluate and track best F1
            current_f1 = evaluate_model(model, test_loader, device, epoch, epochs, is_best=False)
            
            # 更新最佳F1
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_epoch = epoch + 1
                # 保存最佳模型状态（深拷贝避免后续训练污染）
                best_model_state = copy.deepcopy(model.state_dict())
                print(f"  *** New best F1: {best_f1:.4f} at Epoch {best_epoch} ***")
            
        # 训练结束，加载并保存最佳模型
        if best_f1 > 0:
            model.load_state_dict(best_model_state)
            print(f"\n{'='*50}")
            print(f"Training Complete - Loading Best Model (Epoch {best_epoch}, F1={best_f1:.4f})")
            print(f"{'='*50}\n")
            # 最终评估并保存最佳模型
            evaluate_model(model, test_loader, device, best_epoch-1, epochs, is_best=True)
            
    print("Training complete.")

def smooth_predictions(preds, window_size=SMOOTH_WINDOW_SIZE):
    """
    对预测结果进行滑动窗口多数投票平滑处理。
    如果窗口内大部分预测为 1（癫痫），则当前点也判定为 1，否则为 0。
    这能有效消除孤立的假阳性（False Positives）。
    
    参数:
        preds (list or ndarray): 原始预测标签序列
        window_size (int): 滑动窗口大小，必须为奇数
    返回:
        ndarray: 平滑后的预测标签序列
    """
    smoothed = np.copy(preds)
    pad = window_size // 2
    for i in range(len(preds)):
        start = max(0, i - pad)
        end = min(len(preds), i + pad + 1)
        if np.mean(preds[start:end]) > 0.5:
            smoothed[i] = 1
        else:
            smoothed[i] = 0
    return smoothed

def evaluate_model(model, test_loader, device, epoch, epochs, is_best=False):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            
            # Apply standard argmax to see what the model naturally learned
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            


    # 计算平滑后的指标
    smoothed_preds = smooth_predictions(all_preds, window_size=SMOOTH_WINDOW_SIZE)
    s_acc = np.mean(smoothed_preds == np.array(all_labels))
    s_precision = precision_score(all_labels, smoothed_preds, zero_division=0)
    s_recall = recall_score(all_labels, smoothed_preds, zero_division=0)
    s_f1 = f1_score(all_labels, smoothed_preds, zero_division=0)
    s_cm = confusion_matrix(all_labels, smoothed_preds)
    
    print(f"Epoch {epoch+1} Test Metrics (Smoothed, w={SMOOTH_WINDOW_SIZE}):")
    print(f"  Accuracy:  {s_acc*100:.2f}%")
    print(f"  Precision: {s_precision:.4f}")
    print(f"  Recall:    {s_recall:.4f}")
    print(f"  F1 Score:  {s_f1:.4f}")
    print(f"  Smoothed Confusion Matrix:\n{s_cm}\n")
    
    # Plot smoothed confusion matrix at the last epoch or when saving best model
    if epoch == epochs - 1 or is_best:
        cm_save_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
        plot_confusion_matrix_figure(s_cm, cm_save_path)
    
    # Save the trained model (only if it's the best)
    if is_best:
        model_save_path = os.path.join(PROCESSED_DATA_DIR, "fc_model.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Best model saved to {model_save_path} (F1={s_f1:.4f})")
    
    return s_f1

if __name__ == "__main__":
    train_model()
