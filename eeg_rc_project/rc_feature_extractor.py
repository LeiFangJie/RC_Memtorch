"""
此模块定义了基于忆阻器的储层网络 (Memristor Reservoir) 并负责特征提取。
它按时序接收脉冲数据，模拟储层的物理动态过程，并按固定间隔对状态进行采样以生成特征向量。
"""
import os
import torch
import torch.nn as nn
from config import *

class EEGMemristorReservoir(nn.Module):
    """
    针对 EEG 脉冲序列设计的忆阻器储层网络层。
    保留了时间序列的处理过程，避免了数据的展平，同时输出固定间隔的虚拟节点采样状态。
    """
    def __init__(self, g_off=1.0/R_OFF, g_on=1.0/R_ON, v_read=V_READ, decay_rate=DECAY_RATE, sample_interval=RC_SAMPLE_INTERVAL):
        super().__init__()
        self.g_off = g_off
        self.g_on = g_on
        self.v_read = v_read
        self.decay_rate = decay_rate
        self.sample_interval = sample_interval

    def forward(self, spike_input):
        """
        前向传播，处理脉冲序列并生成储层特征。
        
        参数:
            spike_input (Tensor): 脉冲输入张量，形状为 [Batch, 4, TimeSteps] (TimeSteps=512)
            
        返回:
            Tensor: 采样后的电流状态张量，形状为 [Batch, 4, NumSamples]
        """
        batch_size, num_channels, time_steps = spike_input.shape
        
        # Initialize g_state to g_off for each of the 4 channels
        g_state = torch.full((batch_size, num_channels), self.g_off, device=spike_input.device, dtype=torch.float32)
        
        sampled_currents = []
        
        for t in range(time_steps):
            spikes = spike_input[:, :, t]  # [Batch, 4]
            
            # Update logic
            spike_mask = (spikes > 0).float()
            
            # 【核心抗饱和逻辑】
            # Reduce the jump size to prevent quick saturation
            # Instead of jumping fixed step, jump a smaller fraction towards g_on (Exponential approach)
            update_spike = g_state + (self.g_on - g_state) * 0.15
            
            # Make the decay relative to the baseline (g_off) so it returns to baseline, creating "Leaky Integrator" effect
            update_no_spike = self.g_off + (g_state - self.g_off) * self.decay_rate
            
            g_state = spike_mask * update_spike + (1 - spike_mask) * update_no_spike
            
            # Limit range
            g_state = torch.clamp(g_state, self.g_off, self.g_on)
            
            current = g_state * self.v_read
            
            # Sample every sample_interval steps
            if (t + 1) % self.sample_interval == 0:
                sampled_currents.append(current)
                
        # Stack sampled currents: [Batch, 4, NumSamples]
        # For 512 steps and interval 2, NumSamples = 256
        output = torch.stack(sampled_currents, dim=2)
        
        return output

def min_max_normalize(tensor, min_val, max_val):
    """
    使用理论最小值和最大值对特征张量进行 Min-Max 归一化。
    """
    diff = max_val - min_val
    if diff == 0:
        return tensor
    return (tensor - min_val) / diff

def extract_features(patient_id):
    """
    为指定的患者加载脉冲数据，经过 RC 网络提取特征并保存。
    特征输出会被展平为适合 FC 层的 1D 向量。
    
    参数:
        patient_id (str): 患者 ID
    """
    data_path = os.path.join(PROCESSED_DATA_DIR, f"{patient_id}_spikes.pt")
    if not os.path.exists(data_path):
        print(f"Data not found: {data_path}")
        return
        
    print(f"Loading {data_path}...")
    spikes, labels = torch.load(data_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reservoir = EEGMemristorReservoir().to(device)
    
    batch_size = 512
    features_list = []
    
    with torch.no_grad():
        for i in range(0, len(spikes), batch_size):
            batch = spikes[i:i+batch_size].to(device)
            if batch.dtype != torch.float32:
                batch = batch.float()
                
            features = reservoir(batch)
            features_list.append(features.cpu())
            
            if (i + batch_size) % 10240 == 0:
                print(f"Processed {min(i + batch_size, len(spikes))}/{len(spikes)}")
                
    raw_features = torch.cat(features_list, dim=0) # [N, 4, 256]
    
    # Normalize
    theoretical_min = reservoir.g_off * reservoir.v_read
    theoretical_max = reservoir.g_on * reservoir.v_read
    norm_features = min_max_normalize(raw_features, theoretical_min, theoretical_max)
    
    # Flatten spatial and temporal feature dimension for classification
    # Keep Batch dimension
    # Shape: [N, 4 * 256] = [N, 1024]
    flat_features = norm_features.view(norm_features.size(0), -1)
    
    save_path = os.path.join(PROCESSED_DATA_DIR, f"{patient_id}_features.pt")
    torch.save((flat_features, labels), save_path)
    print(f"Saved RC features to {save_path}, shape: {flat_features.shape}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", type=str, default="chb01", help="Patient ID to process or 'all'")
    args = parser.parse_args()
    
    if args.patient == "all":
        for pid in ALL_PATIENTS:
            extract_features(pid)
    else:
        extract_features(args.patient)
