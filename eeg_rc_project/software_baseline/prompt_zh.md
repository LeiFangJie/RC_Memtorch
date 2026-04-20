你正在使用 CHB-MIT 数据集进行基于 EEG 的癫痫检测项目。现有流程包括脉冲编码、频带分解（alpha、beta、theta、delta）、储层计算（RC）和 1D CNN 分类器。

你的任务是实现**纯深度学习基线模型**进行对比，**不修改现有流程**。目标是确保在输入数据、训练协议和评估指标方面的公平对比。

---

## 🎯 目标

实现多个以**频带分解后的 EEG（4 个频带）**为输入的深度学习基线模型，进行二分类（癫痫 vs 非癫痫）。这些基线将与现有流程（Spike + RC + CNN）进行对比。

---

## ⚠️ 关键约束（必须遵守）

1. **不要修改现有的数据加载、分割或评估代码**
2. **使用与主流程相同的训练/验证/测试集划分**
3. **使用相同的评估指标（F1-score 为主要指标）**
4. **基线模型不使用时序平滑后处理**（与主流程使用 window_size=5 平滑不同）
5. **确保公平对比：无数据泄漏，无带来不公平优势的不同预处理**
6. 所有模型必须以模块化方式实现，并通过统一接口调用

---

## 📦 输入格式



- 输入形状：`(batch_size, 4, 512)`，其中：
  - 4 = 频带（delta 0.5-4Hz、theta 4-8Hz、alpha 8-13Hz、beta 13-30Hz）
  - 512 = 时间步（2 秒 @ 256Hz 采样率）

**任何基线模型都不使用脉冲编码、不使用储层计算。**

---

## 🧠 需要实现的模型

在 `software_baseline/` 中实现以下模型：

---

### 1. Baseline_CNN1D

* 输入：4 频带 EEG `(batch, 4, 512)`
* 架构：
  * 3–4 层 Conv1D（卷积核大小 3–7）
  * 每层后接 BatchNorm + ReLU
  * 可选 MaxPooling
  * 全局平均池化
  * 1–2 层全连接层
* 保持参数数量相对较小

---

### 2. Baseline_CNN_LSTM

* 输入：4 频带 EEG `(batch, 4, 512)`
* 架构：
  * 1–2 层 Conv1D（特征提取）
  * LSTM 层（隐藏层大小 ~64–128）
  * 全连接层 → 输出
* 必要时使用 dropout

---

### 3. Baseline_Transformer（轻量级）

* 输入：4 频带 EEG `(batch, 4, 512)`
* 步骤：
  * 将 EEG 分块为 token（例如，16 个块，每块 32 个时间点）
  * 添加位置编码
  * 2–4 层 Transformer 编码器
  * 平均池化
  * FC 分类器
* 保持模型较小（避免过拟合）

---

### 4.（可选）Baseline_DeepCNN

* 比 Baseline_CNN1D 更深的 CNN
* 用于对比"增加深度"策略

---

## ⚙️ 训练设置

使用与主流程相同的训练配置：

* **相同优化器**（Adam，lr=0.0005，weight_decay=1e-4）
* **相同 batch size**（256）
* **相同训练轮数**（15）
* **相同类别不平衡处理**：FocalLoss（alpha=0.5，gamma=2）+ WeightedRandomSampler

---

## 📊 评估

对每个模型报告：

* **F1-score**（主要且唯一的指标）
* Precision / Recall（供参考）

同时计算：

* 参数量
*（必选）近似 FLOPs

**不进行后处理平滑**（与主流程不同）。

---

## 🧾 输出格式

对每个模型返回如下字典：

```python
{
    "model_name": "...",
    "params": ...,
    "f1": ...,
    "precision": ...,
    "recall": ...
}
```

将结果保存到 CSV 文件 `software_baseline/results_comparison.csv`，表头为：
| model_name | params_count | f1 | precision | recall | training_time |

---

## 🧪 实验管理

* 确保可复现性（设置随机种子 42）
* 记录训练曲线（loss、F1）
* 保存最佳模型检查点到 `software_baseline/checkpoints/`

---

## 🧩 集成要求

* 添加标志或配置选项以在以下模式间切换：
  * 现有流程（Spike + RC + CNN）
  * 基线模型

示例：
```python
mode = "baseline_cnn" / "cnn_lstm" / "transformer" / "original_pipeline"
```

创建入口点 `software_baseline/run_baselines.py`，用于：
1. 加载现有处理后的数据（复用 `SinglePatientDataset` 或直接加载 `.pt` 文件）
2. 训练指定的基线模型
3. 评估并保存结果

---

## 🚫 禁止事项

* 不要引入脉冲编码
* 不要使用储层计算
* 不要改变数据集划分
* 不要使用时序平滑（滑动窗口多数投票）
* 不要报告 accuracy（对于此不平衡任务无意义）
* 不要对基线模型进行过度调参，超出合理公平范围

---

## ✅ 最终目标

产生公平且可复现的对比，证明：

* 纯深度学习模型（CNN、CNN+LSTM、Transformer）能达到相似或略好的性能
* 但需要显著更高的模型复杂度（参数量） compared to 现有的 Spike + RC + CNN 流程

所有基线代码必须放置在 `d:\FAFU_work\RC_memtorch\eeg_rc_project\software_baseline\` 目录中。
