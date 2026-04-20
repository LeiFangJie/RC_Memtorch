"""
软件基线模型的训练与评估模块。
复用主流程的训练配置和评估逻辑，但使用频带分解数据（无脉冲编码、无RC、无平滑）。
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import random
import time
import csv
from sklearn.metrics import precision_score, recall_score, f1_score
from config import PROCESSED_DATA_DIR, TEST_PATIENTS_RATIO

# FLOPs计算
try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("[Warning] thop not installed. FLOPs calculation will be skipped.")
    print("Install with: pip install thop")


class BandsDataset(Dataset):
    """
    频带分解数据集。
    加载 {patient_id}_bands.pt 文件，支持按indices重排和切分。
    """
    def __init__(self, patient_id, indices=None):
        feat_path = os.path.join(PROCESSED_DATA_DIR, f"{patient_id}_bands.pt")
        if not os.path.exists(feat_path):
            raise FileNotFoundError(f"Bands data not found: {feat_path}")
            
        self.features, self.labels = torch.load(feat_path)
        
        if indices is not None:
            self.features = self.features[indices]
            self.labels = self.labels[indices]
            
        # 打印数据分布
        total_1 = self.labels.sum().item()
        total_0 = len(self.labels) - total_1
        print(f"[Dataset] {patient_id} - Size: {len(self.labels)} (Normal: {total_0}, Seizure: {total_1})")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification.
    与主流程完全一致。
    """
    def __init__(self, alpha=0.5, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Apply alpha weighting
        alpha_t = torch.where(targets == 1, 
                              torch.tensor(self.alpha, device=inputs.device), 
                              torch.tensor(1 - self.alpha, device=inputs.device))
        
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def compute_flops(model, input_shape=(1, 4, 512)):
    """
    计算模型的FLOPs和参数量。
    使用thop库的profile功能。
    """
    if not THOP_AVAILABLE:
        return None, None
    
    try:
        dummy_input = torch.randn(input_shape)
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        return flops, params
    except Exception as e:
        print(f"[Warning] FLOPs calculation failed: {e}")
        return None, None


def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for features, labels in train_loader:
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
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def evaluate(model, test_loader, device):
    """
    评估模型性能。
    返回: f1, precision, recall, predictions, labels
    注意：不使用平滑后处理（与主流程不同）
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 计算指标（无平滑）
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    
    return f1, precision, recall, all_preds, all_labels


def train_baseline(model_class, model_name, patient_id, epochs=15, lr=0.0005, 
                     weight_decay=1e-4, batch_size=256, device=None):
    """
    训练单个基线模型。
    
    参数:
        model_class: 模型类
        model_name: 模型名称
        patient_id: 患者ID
        epochs: 训练轮数
        lr: 学习率
        weight_decay: 权重衰减
        batch_size: 批量大小
        device: 计算设备
        
    返回:
        dict: 包含训练结果的字典
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*60}")
    print(f"Training {model_name} for patient {patient_id}")
    print(f"{'='*60}")
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 加载数据
    bands_path = os.path.join(PROCESSED_DATA_DIR, f"{patient_id}_bands.pt")
    if not os.path.exists(bands_path):
        print(f"[Error] Bands data not found: {bands_path}")
        print(f"Please run: python band_decomposition.py --patient {patient_id}")
        return None
    
    # 加载完整数据以进行分层分割
    full_features, full_labels = torch.load(bands_path)
    total_samples = len(full_labels)
    
    # 分层抽样：8:2划分
    normal_indices = np.where(full_labels.numpy() == 0)[0]
    seizure_indices = np.where(full_labels.numpy() == 1)[0]
    
    np.random.shuffle(normal_indices)
    np.random.shuffle(seizure_indices)
    
    normal_split = int(len(normal_indices) * (1 - TEST_PATIENTS_RATIO))
    seizure_split = int(len(seizure_indices) * (1 - TEST_PATIENTS_RATIO))
    
    train_indices = np.concatenate([
        normal_indices[:normal_split], 
        seizure_indices[:seizure_split]
    ])
    test_indices = np.concatenate([
        normal_indices[normal_split:], 
        seizure_indices[seizure_split:]
    ])
    
    np.random.shuffle(train_indices)
    # 测试集排序（与主流程一致，虽然基线不使用平滑）
    test_indices = np.sort(test_indices)
    
    # 创建数据集
    train_dataset = BandsDataset(patient_id, indices=train_indices)
    test_dataset = BandsDataset(patient_id, indices=test_indices)
    
    # 创建WeightedRandomSampler处理类别不平衡
    train_labels = train_dataset.labels
    total_1 = train_labels.sum().item()
    total_0 = len(train_labels) - total_1
    
    if total_1 > 0 and total_0 > 0:
        weight_0 = 1.0 / total_0
        weight_1 = 1.0 / total_1
        sample_weights = torch.where(train_labels == 1, weight_1, weight_0).double()
        sampler = WeightedRandomSampler(
            weights=sample_weights, 
            num_samples=len(sample_weights), 
            replacement=True
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建模型
    model = model_class().to(device)
    
    # 计算参数量和FLOPs
    params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    flops, _ = compute_flops(model, input_shape=(1, 4, 512))
    
    print(f"[Model] Parameters: {params_count:,}")
    if flops:
        print(f"[Model] FLOPs: {flops/1e6:.2f}M")
    
    # 损失函数和优化器
    criterion = FocalLoss(alpha=0.5, gamma=2)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 训练循环
    start_time = time.time()
    best_f1 = 0
    
    print(f"\n[Training] Starting {epochs} epochs...")
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 评估
        f1, precision, recall, _, _ = evaluate(model, test_loader, device)
        
        # 保存最佳模型
        if f1 > best_f1:
            best_f1 = f1
            checkpoint_path = os.path.join(
                os.path.dirname(__file__), 
                'checkpoints', 
                f"{model_name}_{patient_id}_best.pth"
            )
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:.2f}% | F1: {f1:.4f} | "
                  f"P: {precision:.4f} | R: {recall:.4f}")
    
    training_time = time.time() - start_time
    
    # 最终评估
    print(f"\n[Final Evaluation] Loading best model...")
    model.load_state_dict(torch.load(checkpoint_path))
    final_f1, final_precision, final_recall, _, _ = evaluate(model, test_loader, device)
    
    print(f"[Results] F1: {final_f1:.4f} | Precision: {final_precision:.4f} | Recall: {final_recall:.4f}")
    print(f"[Time] Training completed in {training_time:.2f}s")
    
    # 返回结果字典
    result = {
        "model_name": model_name,
        "patient_id": patient_id,
        "params": params_count,
        "flops": flops if flops else 0,
        "f1": final_f1,
        "precision": final_precision,
        "recall": final_recall,
        "training_time": training_time,
        "best_epoch_f1": best_f1
    }
    
    return result


def save_results_to_csv(results, csv_path=None):
    """
    保存结果到CSV文件。
    
    参数:
        results: 结果字典或列表
        csv_path: CSV文件路径，默认为 software_baseline/results_comparison.csv
    """
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(__file__), 'results_comparison.csv')
    
    # 确保结果是列表
    if isinstance(results, dict):
        results = [results]
    
    # 检查文件是否存在
    file_exists = os.path.exists(csv_path)
    
    with open(csv_path, 'a', newline='') as f:
        fieldnames = ['model_name', 'patient_id', 'params', 'flops', 
                      'f1', 'precision', 'recall', 'training_time']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        for result in results:
            # 只保留fieldnames中的字段
            result_copy = {k: v for k, v in result.items() if k in fieldnames}
            # 格式化flops为M单位
            if result_copy.get('flops'):
                result_copy['flops'] = f"{result_copy['flops']/1e6:.2f}M"
            writer.writerow(result_copy)
    
    print(f"[CSV] Results saved to {csv_path}")


if __name__ == "__main__":
    # 测试训练流程
    from baseline_models import Baseline_CNN1D
    
    print("Testing baseline trainer...")
    result = train_baseline(
        Baseline_CNN1D, 
        "Baseline_CNN1D", 
        "chb01",
        epochs=2  # 测试用短epoch
    )
    
    if result:
        print("\nTest result:")
        for key, value in result.items():
            print(f"  {key}: {value}")
