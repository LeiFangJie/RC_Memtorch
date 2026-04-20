"""
此模块负责生成软件基线模型所需的频带分解数据（连续值，无脉冲编码）。
复用主流程的滤波和频带分解逻辑，但跳过脉冲编码步骤。
输出形状: (N, 4, 512) - 4个频带，512个时间步
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    与主流程完全一致。
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
    与主流程完全一致。
    """
    mean = np.mean(data, axis=-1, keepdims=True)
    std = np.std(data, axis=-1, keepdims=True)
    std[std == 0] = 1e-8
    return (data - mean) / std


def process_patient_bands(patient_id):
    """
    对指定患者进行频带分解处理，保存连续值数据（无脉冲编码）。
    
    输出文件: {patient_id}_bands.pt
    内容: (bands_tensor, labels_tensor)
        - bands_tensor: shape (N, 4, 512), dtype float32
        - labels_tensor: shape (N,), dtype long
        - 4个频带: delta, theta, alpha, beta (按此顺序)
    """
    print(f"[BandDecomposition] Processing patient {patient_id}...")
    seizures_dict = parse_summary_file(patient_id)
    edf_paths = get_edf_paths(patient_id)
    
    if not edf_paths:
        print(f"[BandDecomposition] No EDF files found for {patient_id}")
        return
        
    all_bands = []
    all_labels = []
    
    for edf_path in edf_paths:
        filename = os.path.basename(edf_path)
        seizures = seizures_dict.get(filename, [])
        
        try:
            raw = mne.io.read_raw_edf(edf_path, preload=True)
            data = raw.get_data()  # (channels, time)
            fs = int(raw.info['sfreq'])
            
            if fs != SAMPLING_RATE:
                data = mne.filter.resample(data, up=SAMPLING_RATE, down=fs)
                fs = SAMPLING_RATE
                
            channels, total_steps = data.shape
            
            # 1. Bandpass 0.5-40 Hz
            data = butter_bandpass_filter(data, LOW_FREQ, HIGH_FREQ, fs)
            
            # 2. Z-score per channel
            data = z_score_normalize(data)
            
            # 3. Frequency Band Decomposition & Aggregation
            # 使用与主流程相同的Max-Abs Pooling
            band_signals = []
            for band_name, (low, high) in BANDS.items():
                filtered = butter_bandpass_filter(data, low, high, fs)
                # Max-Abs Pooling across channels
                max_indices = np.argmax(np.abs(filtered), axis=0)
                agg_signal = np.take_along_axis(filtered, np.expand_dims(max_indices, axis=0), axis=0)[0]
                band_signals.append(agg_signal)
                
            # shape: (4, time_steps)
            continuous_bands = np.stack(band_signals, axis=0)
            
            # 4. Windowing
            num_windows = (total_steps - WINDOW_SAMPLES) // OVERLAP_SAMPLES + 1
            if num_windows <= 0:
                continue
                
            # Create label array for the whole file
            label_array = np.zeros(total_steps)
            for (start_sec, end_sec) in seizures:
                start_idx = start_sec * fs
                end_idx = end_sec * fs
                label_array[start_idx:end_idx] = 1
                
            for w in range(num_windows):
                start_idx = w * OVERLAP_SAMPLES
                end_idx = start_idx + WINDOW_SAMPLES
                
                window_bands = continuous_bands[:, start_idx:end_idx]  # (4, 512)
                window_labels = label_array[start_idx:end_idx]
                
                # Label is 1 if overlap > 50% (1 second = 256 samples)
                is_seizure = 1 if np.sum(window_labels) >= (fs * 1.0) else 0
                
                all_bands.append(window_bands)
                all_labels.append(is_seizure)
                
            # Clean up
            del raw, data, continuous_bands, label_array
            gc.collect()
            
        except Exception as e:
            print(f"[BandDecomposition] Error processing {filename}: {e}")
            
    if all_bands:
        # Save to disk
        bands_tensor = torch.tensor(np.array(all_bands), dtype=torch.float32)
        labels_tensor = torch.tensor(np.array(all_labels), dtype=torch.long)
        
        save_path = os.path.join(PROCESSED_DATA_DIR, f"{patient_id}_bands.pt")
        torch.save((bands_tensor, labels_tensor), save_path)
        print(f"[BandDecomposition] Saved {patient_id} to {save_path}, shape: {bands_tensor.shape}")
        
        # Print class distribution
        seizure_count = labels_tensor.sum().item()
        normal_count = len(labels_tensor) - seizure_count
        print(f"[BandDecomposition] Class distribution - Normal: {normal_count}, Seizure: {seizure_count}")
        
    del all_bands, all_labels
    gc.collect()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Band decomposition for software baselines")
    parser.add_argument("--patient", type=str, default="chb01", 
                        help="Patient ID to process or 'all'")
    args = parser.parse_args()
    
    if args.patient == "all":
        for pid in ALL_PATIENTS:
            process_patient_bands(pid)
    else:
        process_patient_bands(args.patient)
