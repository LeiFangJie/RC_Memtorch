# EEG Baseline Models

纯深度学习基线模型对比实验，用于与主流程（Spike Encoding + RC + CNN）进行性能和复杂度对比。

## 目录结构

```
software_baseline/
├── band_decomposition.py    # 频带分解预处理（无脉冲编码）
├── baseline_models.py         # 4个基线模型架构
├── baseline_trainer.py        # 训练与评估流程
├── run_baselines.py           # 统一入口脚本
├── checkpoints/               # 模型检查点保存目录
├── results_comparison.csv     # 对比结果汇总
└── README.md                  # 本文档
```

## 模型架构

| 模型 | 参数量 | 特点 |
|------|--------|------|
| **Baseline_CNN1D** | ~152K | 4层Conv1D + GAP，轻量级时序特征提取 |
| **Baseline_CNN_LSTM** | ~49K | CNN特征提取 + LSTM时序建模 |
| **Baseline_Transformer** | ~114K | Patch嵌入 + 3层Transformer，最低FLOPs |
| **Baseline_DeepCNN** | ~588K | 6层深度CNN，参数量最大 |

## 输入数据格式

- **输入形状**: `(N, 4, 512)` - 4个频带（delta/theta/alpha/beta），512时间步
- **预处理**: 带通滤波(0.5-40Hz) → Z-score归一化 → Max-Abs Pooling → 窗口化
- **标签**: 0=正常, 1=癫痫发作
- **数据文件**: `processed_data/{patient_id}_bands.pt`

## 依赖安装

```bash
# 使用虚拟环境
& d:/FAFU_work/RC_memtorch/.venv/Scripts/python.exe -m pip install thop
```

## 使用命令

### 1. 单模型单患者训练

```bash
# CNN1D 模型，chb01 患者，15 epochs
& d:/FAFU_work/RC_memtorch/.venv/Scripts/python.exe eeg_rc_project/software_baseline/run_baselines.py --model cnn1d --patient chb01 --epochs 15
```

### 2. 所有模型单患者对比

```bash
# 运行全部4个模型，chb01 患者
& d:/FAFU_work/RC_memtorch/.venv/Scripts/python.exe eeg_rc_project/software_baseline/run_baselines.py --model all --patient chb01 --epochs 15
```

### 3. 数据预处理（首次运行）

```bash
# 自动准备频带分解数据
& d:/FAFU_work/RC_memtorch/.venv/Scripts/python.exe eeg_rc_project/software_baseline/run_baselines.py --model cnn1d --patient chb01 --prepare-data
```

### 4. 命令行参数

```bash
python run_baselines.py [参数]

参数说明:
  --model MODEL        选择模型: cnn1d | cnn_lstm | transformer | deepcnn | all (默认: all)
  --patient PATIENT    患者ID: chb01, chb02, ... | all (默认: all)
  --epochs EPOCHS      训练轮数 (默认: 15)
  --batch-size SIZE    批次大小 (默认: 128)
  --lr LR             学习率 (默认: 0.0005)
  --device DEVICE      设备: cpu | cuda (默认: auto)
  --prepare-data       强制重新生成频带分解数据
```

### 5. 独立运行模块

```bash
# 仅运行频带分解
& d:/FAFU_work/RC_memtorch/.venv/Scripts/python.exe eeg_rc_project/software_baseline/band_decomposition.py --patient chb01

# 仅测试模型架构
& d:/FAFU_work/RC_memtorch/.venv/Scripts/python.exe eeg_rc_project/software_baseline/baseline_models.py

# 单独训练（需已准备数据）
& d:/FAFU_work/RC_memtorch/.venv/Scripts/python.exe -c "
import sys
sys.path.insert(0, 'eeg_rc_project/software_baseline')
from baseline_models import Baseline_CNN1D
from baseline_trainer import train_baseline

result = train_baseline(Baseline_CNN1D, 'Baseline_CNN1D', 'chb01', epochs=15)
print(f'F1: {result[\"f1\"]:.4f}')
"
```

## 输出结果

### 终端输出示例

```
================================================================================
BASELINE MODELS COMPARISON SUMMARY
================================================================================
Model                      Params        FLOPs       F1  Precision   Recall    Time(s)
--------------------------------------------------------------------------------
Baseline_CNN1D            151,778        12.7M   0.5276     0.3644   0.9556       36.4
Baseline_CNN_LSTM          48,738         7.4M   0.2265     0.1293   0.9111       32.1
Baseline_Transformer      113,986         0.9M   0.3916     0.2478   0.9333       32.6
Baseline_DeepCNN          587,874        24.4M   0.1175     0.0625   0.9778       59.5
================================================================================
```

### 文件输出

- **检查点**: `checkpoints/{model_name}_{patient_id}_best.pth`
- **CSV结果**: `results_comparison.csv`

## 特性说明

### 检查点机制
- 自动检测已存在的检查点，存在则**跳过训练直接加载评估**
- 训练过程中保存验证集F1最高的模型
- 使用 `strict=False` 加载以兼容 thop 库添加的额外属性

### 类别不平衡处理
- **Focal Loss** (α=0.5, γ=2)
- **WeightedRandomSampler** 在训练时动态平衡批次

### 评估指标
- **主要指标**: F1-score
- **附加指标**: Precision, Recall
- **复杂度指标**: 参数量, FLOPs (via thop)
- **注意**: 基线模型**不使用时序平滑**（与主流程区分）

### 数据划分
- 分层8:2划分（保持癫痫/正常样本比例）
- 测试集排序（便于后续时序分析）

## 与主流程对比

| 特性 | 主流程 (Spike+RC+CNN) | 软件基线 |
|------|----------------------|----------|
| 输入 | 脉冲编码 (0/1) | 连续值4频带 |
| 预处理 | Spike Encoding | 无（仅频带分解）|
| 特征提取 | Memristor Reservoir | 无（端到端CNN）|
| 后处理 | 时序平滑 | 无 |
| 目标 | 低复杂度高性能 | 纯深度学习基准 |

## 注意事项

1. **首次运行**需确保 CHB-MIT 数据位于正确路径 (`../chb-mit-scalp-eeg-database-1.0.0/`)
2. **CUDA可用时自动使用**，否则回退到CPU
3. **检查点复用**: 删除 `checkpoints/` 目录可强制重新训练
4. **CSV追加模式**: 多次运行会自动追加结果到同一CSV文件

## 快速开始

```bash
# 1. 确保在主项目目录
cd d:\FAFU_work\RC_memtorch\eeg_rc_project

# 2. 准备数据并训练CNN1D（chb01，1 epoch快速测试）
& d:/FAFU_work/RC_memtorch/.venv/Scripts/python.exe software_baseline/run_baselines.py --model cnn1d --patient chb01 --epochs 1 --prepare-data

# 3. 完整对比（所有模型，15 epochs）
& d:/FAFU_work/RC_memtorch/.venv/Scripts/python.exe software_baseline/run_baselines.py --model all --patient chb01 --epochs 15
```
