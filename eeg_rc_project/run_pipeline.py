"""
此脚本是整个 EEG 储层计算癫痫检测项目的统一启动入口。
提供了命令行参数，可以选择仅处理单名患者进行快速测试，或处理全量数据。
按顺序依次执行：脉冲编码、特征提取、模型训练及可视化。
"""
import argparse
import os
from config import ALL_PATIENTS
from eeg_spike_encoder import process_patient
from rc_feature_extractor import extract_features
from train_classifier import train_model
from visualization import visualize_sample, plot_clustering

def run_pipeline(mode, patient_id):
    print(f"Starting EEG RC Pipeline in '{mode}' mode...")
    
    patients_to_process = []
    if mode == "single":
        patients_to_process = [patient_id]
    elif mode == "all":
        patients_to_process = ALL_PATIENTS
        
    print(f"Patients to process: {patients_to_process}")
    
    # 1. Spike Encoding
    print("\n--- Step 1: Spike Encoding ---")
    for pid in patients_to_process:
        process_patient(pid)
        
    # 2. RC Feature Extraction
    print("\n--- Step 2: RC Feature Extraction ---")
    for pid in patients_to_process:
        extract_features(pid)
        
    # 3. Train Classifier
    print("\n--- Step 3: Train Classifier ---")
    train_model()
    
    # 4. Visualization
    print("\n--- Step 4: Visualization ---")
    # For visualization, we just pick the first patient or the requested one
    vis_pid = patient_id if mode == "single" else ALL_PATIENTS[0]
    visualize_sample(vis_pid)
    
    # Plot clustering using the features we just generated
    plot_clustering(patients_to_process)
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EEG RC Seizure Detection Pipeline")
    parser.add_argument("--mode", type=str, choices=["single", "all"], default="single", 
                        help="Run for a single patient or all patients")
    parser.add_argument("--patient", type=str, default="chb01", 
                        help="Patient ID to process in single mode (e.g., chb01)")
    
    args = parser.parse_args()
    run_pipeline(args.mode, args.patient)
