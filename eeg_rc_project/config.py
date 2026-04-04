"""
此模块用于配置 EEG 储层计算（RC）项目的全局超参数。
包含路径配置、EEG 信号处理参数、频带划分、脉冲编码参数、物理储层（Memristor）参数以及数据集划分设置。
"""
import os

# Project Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, '../chb-mit-scalp-eeg-database-1.0.0')
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'processed_data')
PLOTS_DIR = os.path.join(PROJECT_ROOT, 'plots')

# EEG Parameters
SAMPLING_RATE = 256  # EEG 采样率（Hz），CHB-MIT 标准为 256Hz
WINDOW_SIZE_SEC = 2  # 滑动窗口时长（秒），每个样本窗口覆盖 2 秒信号
OVERLAP_SEC = 1  # 相邻窗口重叠时长（秒），用于提升时序连续性与样本数量
WINDOW_SAMPLES = SAMPLING_RATE * WINDOW_SIZE_SEC  # 每个窗口对应的采样点数（2*256=512）
OVERLAP_SAMPLES = SAMPLING_RATE * OVERLAP_SEC  # 窗口滑动步长中的重叠采样点数（1*256=256）

# Filter Parameters
LOW_FREQ = 0.5
HIGH_FREQ = 40.0

# Frequency Bands
BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30)
}

# Spike Encoding Parameters
SPIKE_LOCAL_WINDOW = 25  # 脉冲编码局部均值窗口长度（点），约 25/256≈0.1 秒
FIXED_THETA = 0.8  # 固定绝对阈值。由于数据已做全局 Z-score，0.8 代表触发脉冲需要波形发生 0.8 倍全局标准差的突变
REFRACTORY_SAMPLES = 3  # 不应期采样点数，防止连续高频噪声激发

# RC Parameters
R_OFF = 391623.0
R_ON = 42455.0
V_READ = 0.1
DECAY_RATE = 0.95
RC_SAMPLE_INTERVAL = 2  # Sample every 2 steps out of 512, getting 256 states

# Selected patients for training and testing
# For a full run, we would include all 24 patients.
# To allow quick experiments, we provide a subset list as default.
ALL_PATIENTS = [f"chb{str(i).zfill(2)}" for i in range(1, 25)]
TEST_PATIENTS_RATIO = 0.2
