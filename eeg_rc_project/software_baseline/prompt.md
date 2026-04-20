You are working on an EEG-based seizure detection project using the CHB-MIT dataset. The existing pipeline includes spike encoding, frequency band decomposition (alpha, beta, theta, delta), reservoir computing (RC), and a 1D CNN classifier.

Your task is to implement **pure deep learning baselines** for comparison, WITHOUT modifying the existing pipeline. The goal is to ensure fair comparison in terms of input data, training protocol, and evaluation metrics.

---

## 🎯 Objective

Implement multiple deep learning baseline models that take **band-decomposed EEG (4 bands)** as input and perform binary classification (seizure vs non-seizure). These baselines will be compared against the existing pipeline (Spike + RC + CNN).

---

## ⚠️ Critical Constraints (MUST FOLLOW)

1. **DO NOT modify existing data loading, splitting, or evaluation code**
2. **USE the same train/val/test splits as the main pipeline**
3. **USE the same evaluation metrics (F1-score PRIMARY)**
4. **NO time-series smoothing post-processing for baselines** (unlike main pipeline which uses window_size=5 smoothing)
5. **Ensure fair comparison: no data leakage, no different preprocessing that gives unfair advantage**
6. All models must be implemented in a modular way and callable via a unified interface

---

## 📦 Input Format


- Input shape: `(batch_size, 4, 512)` where:
  - 4 = frequency bands (delta 0.5-4Hz, theta 4-8Hz, alpha 8-13Hz, beta 13-30Hz)
  - 512 = time steps (2 seconds @ 256Hz sampling rate)

**NO spike encoding, NO reservoir computing in any baseline.**

---

## 🧠 Models to Implement

Implement the following models in `software_baseline/`:

---

### 1. Baseline_CNN1D

* Input: 4-band EEG `(batch, 4, 512)`
* Architecture:
  * 3–4 Conv1D layers (kernel size 3–7)
  * Each followed by BatchNorm + ReLU
  * Optional MaxPooling
  * Global Average Pooling
  * 1–2 Fully Connected layers
* Keep parameter count relatively small

---

### 2. Baseline_CNN_LSTM

* Input: 4-band EEG `(batch, 4, 512)`
* Architecture:
  * 1–2 Conv1D layers (feature extraction)
  * LSTM layer (hidden size ~64–128)
  * Fully Connected layer → output
* Use dropout if necessary

---

### 3. Baseline_Transformer (Lightweight)

* Input: 4-band EEG `(batch, 4, 512)`
* Steps:
  * Patch EEG into tokens (e.g., 16 patches of 32 points)
  * Add positional encoding
  * 2–4 Transformer encoder layers
  * Mean pooling
  * FC classifier
* Keep model small (avoid overfitting)

---

### 4. (Optional) Baseline_DeepCNN

* Slightly deeper CNN than Baseline_CNN1D
* Used to compare with "increased depth" strategy

---

## ⚙️ Training Settings

Use the SAME training configuration as the main pipeline:

* **Same optimizer** (Adam, lr=0.0005, weight_decay=1e-4)
* **Same batch size** (256)
* **Same number of epochs** (15)
* **Same class imbalance handling**: FocalLoss (alpha=0.5, gamma=2) + WeightedRandomSampler

---

## 📊 Evaluation

For each model, report:

* **F1-score** (PRIMARY and ONLY metric)
* Precision / Recall (for reference)

Also compute:

* Number of parameters
* (Must) Approximate FLOPs

**NO post-processing smoothing** (unlike main pipeline).

---

## 🧾 Output Format

For each model, return a dictionary like:

```python
{
    "model_name": "...",
    "params": ...,
    "f1": ...,
    "precision": ...,
    "recall": ...
}
```

Save results into a CSV file `software_baseline/results_comparison.csv` with schema:
| model_name | params_count | f1 | precision | recall | training_time |

---

## 🧪 Experiment Management

* Ensure reproducibility (set random seed 42)
* Log training curves (loss, F1)
* Save best model checkpoint to `software_baseline/checkpoints/`

---

## 🧩 Integration Requirement

* Add a flag or config option to switch between:
  * existing pipeline (Spike + RC + CNN)
  * baseline models

Example:
```python
mode = "baseline_cnn" / "cnn_lstm" / "transformer" / "original_pipeline"
```

Create entry point `software_baseline/run_baselines.py` that:
1. Loads existing processed data (reuse `SinglePatientDataset` or load `.pt` files directly)
2. Trains specified baseline model
3. Evaluates and saves results

---

## 🚫 What NOT to do

* Do NOT introduce spike encoding
* Do NOT use reservoir computing
* Do NOT change dataset split
* Do NOT use time-series smoothing (sliding window majority vote)
* Do NOT report accuracy (not meaningful for this imbalanced task)
* Do NOT tune baselines excessively beyond reasonable fairness

---

## ✅ Final Goal

Produce a fair and reproducible comparison showing that:

* Pure deep learning models (CNN, CNN+LSTM, Transformer) achieve similar or slightly better performance
* BUT require significantly higher model complexity (parameter count) compared to the existing Spike + RC + CNN pipeline

All baseline code must be placed in `d:\FAFU_work\RC_memtorch\eeg_rc_project\software_baseline\` directory.
