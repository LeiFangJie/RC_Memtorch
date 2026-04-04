"""
此模块负责执行 EEG 数据的核心预处理流，包括：
1. 加载 EDF 文件
2. 0.5 - 40 Hz 带通滤波及独立通道 Z-score 归一化
3. 频带分解 (Delta, Theta, Alpha, Beta) 并跨通道聚合（均值）以适应可变通道数
4. 滑动窗口切分与癫痫发作标注
5. 自适应 Delta 编码，将连续信号转换为适合储层计算的离散脉冲序列
6. 将处理后的数据逐患者保存为 PyTorch 张量文件
"""
import os
import gc
import numpy as np
import torch
import mne
from scipy import signal
from config import *
from chb_dataset_parser import parse_summary_file, get_edf_paths

mne.set_log_level('WARNING')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    使用 Butterworth 滤波器进行带通滤波（零相位滤波，防止时序偏移）。
    
    参数:
        data (ndarray): 输入的连续 EEG 信号
        lowcut (float): 低频截止频率
        highcut (float): 高频截止频率
        fs (int): 采样率
        order (int): 滤波器阶数
        
    返回:
        ndarray: 滤波后的信号
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    y = signal.filtfilt(b, a, data, axis=-1)
    return y

def z_score_normalize(data):
    """
    沿最后一个轴（时间轴）独立对每个通道进行 Z-score 归一化。
    
    参数:
        data (ndarray): 输入信号，形状为 (channels, time)
        
    返回:
        ndarray: 归一化后的信号
    """
    mean = np.mean(data, axis=-1, keepdims=True)
    std = np.std(data, axis=-1, keepdims=True)
    std[std == 0] = 1e-8
    return (data - mean) / std

def adaptive_delta_encoding(window_data, local_w=SPIKE_LOCAL_WINDOW, fixed_theta=FIXED_THETA, refractory_samples=REFRACTORY_SAMPLES):
    """
    对信号窗口应用固定绝对阈值的 Delta 脉冲编码。
    
    【核心逻辑修改说明】
    最初的版本使用了“每个窗口独立 Z-score 归一化” + “自适应密度阈值”。
    但这导致了一个致命错误：正常静息状态下的微小底噪被强行放大，并产生与癫痫发作时一样密集的脉冲，彻底抹平了它们之间的绝对能量差异。
    
    现在的逻辑是：
    1. 依赖预处理阶段的【全局 Z-score 归一化】，保留不同时间段的绝对幅度差异。
    2. 使用【固定绝对阈值】(fixed_theta)。正常信号波动极小，无法越过阈值，产生大面积留白；癫痫信号波动大，产生密集脉冲。
    3. 引入【不应期】(refractory_samples)，防止瞬时高频噪声导致连续激发，保护 RC 层不被迅速饱和。
    
    参数:
        window_data (ndarray): 输入窗口数据，形状为 (4, 512)
        local_w (int): 用于计算局部因果均值的滑动窗口大小
        fixed_theta (float): 固定的激发绝对阈值（倍数）
        refractory_samples (int): 不应期采样点数
        
    返回:
        ndarray: 脉冲序列，形状为 (4, 512)
    """
    channels, time_steps = window_data.shape
    spikes = np.zeros_like(window_data)
    
    for c in range(channels):
        band_data = window_data[c]
        
        # 计算因果滑动均值作为基线
        kernel = np.ones(local_w) / local_w
        causal_mean = np.convolve(band_data, kernel, mode='full')[:time_steps]
        causal_mean = np.roll(causal_mean, 1)
        causal_mean[0] = band_data[0]
        diff = np.abs(band_data - causal_mean)
        
        band_spikes = np.zeros(time_steps)
        last_spike_time = -refractory_samples - 1
        
        # 带有不应期控制的固定阈值脉冲生成
        for t in range(time_steps):
            if t - last_spike_time > refractory_samples:
                if diff[t] > fixed_theta:
                    band_spikes[t] = 1
                    last_spike_time = t
                    
        spikes[c] = band_spikes
            
    return spikes

def process_patient(patient_id):
    """
    对指定的单名患者执行完整的数据预处理流，包括数据加载、滤波、降维、编码及持久化存储。
    
    参数:
        patient_id (str): 患者 ID
    """
    print(f"Processing patient {patient_id}...")
    seizures_dict = parse_summary_file(patient_id)
    edf_paths = get_edf_paths(patient_id)
    
    if not edf_paths:
        print(f"No EDF files found for {patient_id}")
        return
        
    all_spikes = []
    all_labels = []
    
    for edf_path in edf_paths:
        filename = os.path.basename(edf_path)
        seizures = seizures_dict.get(filename, [])
        
        try:
            raw = mne.io.read_raw_edf(edf_path, preload=True)
            data = raw.get_data() # (channels, time)
            fs = int(raw.info['sfreq'])
            
            if fs != SAMPLING_RATE:
                # Resample if not 256Hz (CHB-MIT is 256Hz, but just in case)
                data = mne.filter.resample(data, up=SAMPLING_RATE, down=fs)
                fs = SAMPLING_RATE
                
            channels, total_steps = data.shape
            
            # 1. Bandpass 0.5-40 Hz
            data = butter_bandpass_filter(data, LOW_FREQ, HIGH_FREQ, fs)
            
            # 2. Z-score per channel
            data = z_score_normalize(data)
            
            # 3. Frequency Band Decomposition & Aggregation
            # bands: delta, theta, alpha, beta
            band_signals = []
            for band_name, (low, high) in BANDS.items():
                filtered = butter_bandpass_filter(data, low, high, fs)
                # Aggregate across all channels using Max-Abs Pooling
                # This ensures strong localized seizure bursts are not diluted
                max_indices = np.argmax(np.abs(filtered), axis=0)
                agg_signal = np.take_along_axis(filtered, np.expand_dims(max_indices, axis=0), axis=0)[0]
                band_signals.append(agg_signal)
                
            # shape: (4, time_steps)
            continuous_bands = np.stack(band_signals, axis=0)
            
            # 4. Windowing
            num_windows = (total_steps - WINDOW_SAMPLES) // OVERLAP_SAMPLES + 1
            if num_windows <= 0:
                continue
                
            # Create a label array for the whole file
            # 1 at seizure times, 0 otherwise
            label_array = np.zeros(total_steps)
            for (start_sec, end_sec) in seizures:
                start_idx = start_sec * fs
                end_idx = end_sec * fs
                label_array[start_idx:end_idx] = 1
                
            for w in range(num_windows):
                start_idx = w * OVERLAP_SAMPLES
                end_idx = start_idx + WINDOW_SAMPLES
                
                window_data = continuous_bands[:, start_idx:end_idx]
                window_labels = label_array[start_idx:end_idx]
                
                # Label is 1 if overlap > 50% (1 second = 256 samples)
                is_seizure = 1 if np.sum(window_labels) >= (fs * 1.0) else 0
                
                # 5. 移除窗口级别的独立归一化！
                # 我们必须保留全局的绝对幅度差异。正常窗口的信号波动极小，癫痫窗口波动极大。
                # 如果在这里做归一化，会把正常的微小波动放大，导致产生与癫痫一样密集的脉冲。
                # window_data = z_score_normalize(window_data)
                
                # 6. 使用固定绝对阈值进行 Delta 编码
                spike_train = adaptive_delta_encoding(window_data)
                
                all_spikes.append(spike_train)
                all_labels.append(is_seizure)
                
            # Clean up
            del raw, data, continuous_bands, label_array
            gc.collect()
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    if all_spikes:
        # Save to disk
        spikes_tensor = torch.tensor(np.array(all_spikes), dtype=torch.float32)
        labels_tensor = torch.tensor(np.array(all_labels), dtype=torch.long)
        
        save_path = os.path.join(PROCESSED_DATA_DIR, f"{patient_id}_spikes.pt")
        torch.save((spikes_tensor, labels_tensor), save_path)
        print(f"Saved {patient_id} to {save_path}, shape: {spikes_tensor.shape}")
        
    del all_spikes, all_labels
    gc.collect()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", type=str, default="chb01", help="Patient ID to process or 'all'")
    args = parser.parse_args()
    
    if args.patient == "all":
        for pid in ALL_PATIENTS:
            process_patient(pid)
    else:
        process_patient(args.patient)
