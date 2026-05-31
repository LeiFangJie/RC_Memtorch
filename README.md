# RC_memtorch: 基于忆阻器储层计算的神经形态识别系统

这是一个围绕**忆阻器储层计算 (Memristor Reservoir Computing, RC)** 展开的神经形态计算研究项目。项目包含三个逻辑模块：一个 MNIST 概念验证子系统、一个面向临床的 EEG 异常预警主系统、以及一套传统深度学习基线对比系统。

---

## 🎯 项目核心目标

将基于忆阻器物理特性的储层计算网络应用于：

1. **MNIST 手写数字识别**（概念验证，验证脉冲编码 + RC 特征提取的可行性）
2. **CHB-MIT 头皮 EEG 癫痫预警 / Pre-ictal-like Warning**（核心临床应用，目标是在器件约束下尽可能提前给出异常告警）

本质上，这是一个**将连续模拟信号 → 脉冲序列 (Spike Train) → 忆阻器物理动态 → 深度神经网络解码**的端到端神经形态计算流水线。

> 当前 EEG 主线的项目定位已经从“严格的 seizure onset 检测”调整为“工程化 pre-ictal 预警”。
> 这里的“pre-ictal”并不强调严格意义上的未来长时预测，而强调：在实际器件与编码能力约束下，尽可能更早地把异常脑电从正常背景中拉开，并将其作为临床可解释的预警信号。

---

## 📂 目录结构概览

```
RC_memtorch/
├── chb-mit-scalp-eeg-database-1.0.0/   # 24位患者的原始EEG数据 (.edf)
├── mnist_rc_project/                   # MNIST验证实验
│   ├── mnist_spike_encoder.py          # 图像→4通道脉冲编码
│   ├── rc_feature_extractor.py         # 忆阻器储层特征提取
│   └── ...
├── eeg_rc_project/                     # EEG 异常预警主系统
│   ├── config.py                       # 全局超参数
│   ├── chb_dataset_parser.py           # CHB-MIT标签解析器
│   ├── eeg_spike_encoder.py            # EEG滤波、频带聚合、脉冲编码
│   ├── rc_feature_extractor.py         # EEG专用忆阻器储层网络
│   ├── train_classifier.py             # 1D CNN + Focal Loss + 滑动窗口平滑
│   ├── run_pipeline.py                 # 统一启动脚本
│   ├── visualization.py                # 可视化与评估
│   └── software_baseline/              # 传统深度学习基线对比
└── requirements.txt                    # Python依赖
```

---

## 🔬 核心技术架构与创新点

| 层级 | 技术方案 | 关键创新 |
|------|---------|---------|
| **输入编码** | 固定绝对阈值 + 局部相对异常门控 + 不应期 | 保留原始“自然密度差”脉冲风格，同时增强对 pre-ictal 异常波动的敏感性 |
| **通道适配** | Max-Abs Pooling 跨通道频带聚合 | 解决不同患者通道数不固定 (23/24) 的问题，固定为 4 频带 |
| **储层动态** | 指数级状态跃迁 + 基线相对衰减 | 防止忆阻器电导快速饱和，保留漏积分器 (Leaky Integrator) 记忆效应 |
| **特征解码** | 1D CNN (替代传统线性层) | 将 1024 维展平特征重塑回时空维度，解开高维流形纠缠 |
| **后处理** | 滑动窗口多数投票 (Window=5, ~6秒) | 利用异常状态的持续性，滤除孤立假阳性，提高工程预警稳定性 |
| **训练策略** | Focal Loss + WeightedRandomSampler | 应对 300:1 的极度类别不平衡 |

---

## 🧩 两个子项目的定位

### 1. `mnist_rc_project/` — 概念验证

- 将 28×28 MNIST 图像切成 4 个 14×14 象限 → 4 通道时间序列 (各 196 步)
- 二值化生成脉冲，输入忆阻器储层，输出 4×14=56 维特征
- 目的：验证"脉冲编码 + 物理储层"这一范式的基本可行性

### 2. `eeg_rc_project/` — 核心研究

- 处理真实世界多通道时间序列 (CHB-MIT, 256Hz, 24患者)
- 2 秒滑动窗口、1 秒重叠 → 512 时间点
- 频带分解 (Delta/Theta/Alpha/Beta) → 4 通道 → 储层采样 256 点 → 1024 维特征 → 1D CNN 分类
- 当前定位：**从严格发作检测转向工程化 pre-ictal 预警**
- 当前策略：**不推翻原有 RC 主体，只在编码规则上做最小修改，保持原版“normal 稀疏 / positive 密集”的自然脉冲特征**

### 3. `eeg_rc_project/software_baseline/` — 对比基准

- 包含传统深度学习模型（如 CNN、LSTM 等）的对比实验
- 用于证明忆阻器 RC 方案在能效或性能上的优势

---

## 🚀 快速开始

### 环境依赖

建议使用 Python 3.8+，主要依赖包括：
- `torch`
- `mne`
- `scipy`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`

### 运行 EEG 预警流水线

```bash
# 快速验证模式（单患者，默认 chb01）
python eeg_rc_project/run_pipeline.py --mode single --patient chb01

# 全量数据模式
python eeg_rc_project/run_pipeline.py --mode all
```

流水线将依次自动执行：
1. 解析标签并进行 0.5-40Hz 滤波、全局 Z-score 归一化
2. Max-Abs 频带降维与“固定绝对阈值 + 局部相对异常门控”脉冲编码
3. 利用 `Memristor Reservoir` 提取物理特征
4. 按患者划分数据集并训练 **1D CNN 分类器**
5. 生成各种可视化图表

---

## 📊 关键技术演进

项目经历了四次关键迭代才达到当前工程形态：

1. **脉冲编码重建**：从"自适应密度阈值" → "固定绝对阈值 + 不应期"（解决正常/异常脉冲形态不可分问题）
2. **解码器升级**：从"单层全连接" → "1D CNN"（解决高维特征空间高度纠缠问题）
3. **时序平滑引入**：从"孤立窗口预测" → "滑动窗口多数投票"（解决类别不平衡导致的 Precision 崩溃问题）
4. **任务定位调整**：从"严格 seizure detection" → "工程化 pre-ictal warning"；实现上保留原始 4 通道 RC 流水线，只在编码阶段加入局部相对异常门控，以增强提前异常响应，同时避免人为规则打点破坏自然脉冲形态

这些迭代使得系统从**完全无法区分 (F1 ≈ 0)** 进化到 **在当前器件约束下具备实际预警价值的工程版本**。

---

## 🛠️ 技术栈

- **PyTorch** (GPU 加速)
- **MNE** (EEG/EDF 神经生理学数据处理)
- **SciPy / NumPy** (信号处理)
- **scikit-learn** (评估指标、t-SNE)
- **matplotlib / seaborn** (可视化)

---

## 📝 详细文档

- **`eeg_rc_project/README.md`**：EEG 预警项目的详细说明、任务重定义与参数速查表
- **`eeg_rc_project/PROJECT_LOG.md`**：项目重大方案修改及迭代日志
