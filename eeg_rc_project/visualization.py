"""
此模块提供可视化工具，用于生成并保存：
1. 提取出的多通道脉冲序列图 (Spike Trains)
2. 储层计算输出的特征热力图 (RC Features)
3. 原始 RC 特征以及 FC 层隐层特征的 t-SNE 降维聚类散点图
"""
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from config import *
from sklearn.manifold import TSNE
import random
from train_classifier import CNN1DClassifier

def plot_multiple_spikes(spikes_list, labels_list, save_path, title_prefix):
    """
    将多个脉冲序列绘制在同一张大图中并保存。
    
    参数:
        spikes_list (list): 包含多个形状为 [4, 512] 的脉冲数组的列表
        labels_list (list): 对应的标签列表
        save_path (str): 图片保存路径
        title_prefix (str): 标题前缀（例如 "Normal" 或 "Seizure"）
    """
    n_samples = len(spikes_list)
    fig, axes = plt.subplots(n_samples, 1, figsize=(10, 3 * n_samples))
    if n_samples == 1:
        axes = [axes]
        
    colors = ['r', 'g', 'b', 'orange']
    channel_labels = ['Delta', 'Theta', 'Alpha', 'Beta']
    
    for i, ax in enumerate(axes):
        spikes = spikes_list[i]
        label = labels_list[i]
        spike_times = []
        for c in range(4):
            times = np.where(spikes[c] == 1)[0]
            spike_times.append(times)
            
        ax.eventplot(spike_times, lineoffsets=[1, 2, 3, 4], linelengths=0.8, colors=colors)
        ax.set_yticks([1, 2, 3, 4])
        ax.set_yticklabels(channel_labels)
        ax.set_xlim(0, 512)
        ax.set_ylim(0.5, 4.5)
        ax.set_title(f"{title_prefix} Sample {i+1} (Label: {'Seizure' if label == 1 else 'Normal'})")
        
        if i == n_samples - 1:
            ax.set_xlabel("Time Step")
        else:
            ax.set_xticks([])
            
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_rc_dynamics_scatter(features_list, labels_list, save_path, title_prefix):
    """
    绘制 RC 储层状态随时间演化的散点图（漏积分器回落效果）。
    每行代表一个频带的输出状态随时间的变化。
    
    参数:
        features_list (list): 包含多个形状为 [1024] 且可重塑为 [4, 256] 的张量列表
        labels_list (list): 对应的标签列表
        save_path (str): 图片保存路径
        title_prefix (str): 标题前缀
    """
    n_samples = len(features_list)
    fig, axes = plt.subplots(n_samples, 1, figsize=(10, 3 * n_samples))
    if n_samples == 1:
        axes = [axes]
        
    colors = ['orange', 'darkblue', 'crimson', 'black']
    channel_labels = ['Delta', 'Theta', 'Alpha', 'Beta']
    
    # We want to separate the 4 channels clearly on the y-axis, similar to eventplot but with scatter values
    # Each channel will have a baseline y-offset, and the current value will be added to it (scaled)
    
    for i, ax in enumerate(axes):
        feat_map = features_list[i].view(4, -1).numpy() # Shape: [4, 256]
        label = labels_list[i]
        
        # Normalize the feat_map locally for this plot to make the scatter variations visible
        # min-max scaling per channel
        for c in range(4):
            c_min, c_max = feat_map[c].min(), feat_map[c].max()
            if c_max > c_min:
                feat_map[c] = (feat_map[c] - c_min) / (c_max - c_min)
            else:
                feat_map[c] = 0
                
        time_steps = np.arange(256)
        
        for c in range(4):
            # y_offset: 4 for Delta, 3 for Theta, 2 for Alpha, 1 for Beta (to match top-down order if desired)
            # Let's do 4,3,2,1 so Delta is at top
            y_offset = 4 - c 
            
            # The y values are the offset + scaled feature
            # We scale it by 0.8 so it doesn't overlap with the next channel
            y_values = y_offset + feat_map[c] * 0.8
            
            # Plot as square markers (s=15, marker='s')
            ax.scatter(time_steps, y_values, c=colors[c], marker='s', s=10, label=channel_labels[c] if i==0 else "")
            
        ax.set_yticks([1.4, 2.4, 3.4, 4.4]) # Approximate centers
        # Reversed order because we plotted Delta at 4, Theta at 3, etc.
        ax.set_yticklabels(['Beta', 'Alpha', 'Theta', 'Delta'])
        ax.set_xlim(0, 256)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0)
        
        ax.set_title(f"{title_prefix} Sample {i+1} RC Dynamics (Label: {'Seizure' if label == 1 else 'Normal'})")
        
        if i == n_samples - 1:
            ax.set_xlabel("Sample Step")
        else:
            ax.set_xticks([])
            
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved RC dynamics scatter plot to {save_path}")
def plot_multiple_rc_features(features_list, labels_list, save_path, title_prefix):
    """
    将多个 RC 储层特征热力图绘制在同一张大图中并保存。
    自动计算全局极值以统一 Colorbar。
    
    参数:
        features_list (list): 包含多个形状为 [1024] 且可重塑为 [4, 256] 的张量列表
        labels_list (list): 对应的标签列表
        save_path (str): 图片保存路径
        title_prefix (str): 标题前缀
    """
    n_samples = len(features_list)
    fig, axes = plt.subplots(n_samples, 1, figsize=(10, 2 * n_samples))
    if n_samples == 1:
        axes = [axes]
        
    # Find global min/max for consistent colorbar
    all_feats = torch.stack(features_list).view(n_samples, 4, -1).numpy()
    vmin, vmax = all_feats.min(), all_feats.max()
    
    for i, ax in enumerate(axes):
        feat_map = all_feats[i]
        label = labels_list[i]
        
        im = ax.imshow(feat_map, aspect='auto', cmap='hot', interpolation='nearest', vmin=vmin, vmax=vmax)
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(['Delta', 'Theta', 'Alpha', 'Beta'])
        ax.set_title(f"{title_prefix} Sample {i+1} (Label: {'Seizure' if label == 1 else 'Normal'})")
        
        if i == n_samples - 1:
            ax.set_xlabel("Sample Step")
        else:
            ax.set_xticks([])
            
    # Add a single colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Current (A)")
    
    plt.savefig(save_path)
    plt.close()

def visualize_sample(patient_id, num_samples=5):
    """
    随机抽取指定数量的正常样本和癫痫发作样本，调用绘图函数生成拼图并保存。
    
    参数:
        patient_id (str): 患者 ID
        num_samples (int): 每类随机抽取的样本数
    """
    spike_path = os.path.join(PROCESSED_DATA_DIR, f"{patient_id}_spikes.pt")
    feat_path = os.path.join(PROCESSED_DATA_DIR, f"{patient_id}_features.pt")
    
    if not os.path.exists(spike_path) or not os.path.exists(feat_path):
        print(f"Data files not found for patient {patient_id}.")
        return
        
    spikes, labels = torch.load(spike_path)
    features, _ = torch.load(feat_path)
    
    # Find normal and seizure samples
    normal_idx = (labels == 0).nonzero(as_tuple=True)[0]
    seizure_idx = (labels == 1).nonzero(as_tuple=True)[0]
    
    # Select random samples
    np.random.seed(42)
    selected_normal = np.random.choice(normal_idx.numpy(), min(num_samples, len(normal_idx)), replace=False)
    selected_seizure = np.random.choice(seizure_idx.numpy(), min(num_samples, len(seizure_idx)), replace=False)
    
    if len(selected_normal) > 0:
        spikes_list = [spikes[i].numpy() for i in selected_normal]
        labels_list = [labels[i].item() for i in selected_normal]
        feat_list = [features[i] for i in selected_normal]
        
        plot_multiple_spikes(spikes_list, labels_list, os.path.join(PLOTS_DIR, f"spikes_normal_top5.png"), "Normal")
        plot_multiple_rc_features(feat_list, labels_list, os.path.join(PLOTS_DIR, f"rc_feat_normal_top5.png"), "Normal")
        plot_rc_dynamics_scatter(feat_list, labels_list, os.path.join(PLOTS_DIR, f"rc_dynamics_scatter_normal_top5.png"), "Normal")
        
    if len(selected_seizure) > 0:
        spikes_list = [spikes[i].numpy() for i in selected_seizure]
        labels_list = [labels[i].item() for i in selected_seizure]
        feat_list = [features[i] for i in selected_seizure]
        
        plot_multiple_spikes(spikes_list, labels_list, os.path.join(PLOTS_DIR, f"spikes_seizure_top5.png"), "Seizure")
        plot_multiple_rc_features(feat_list, labels_list, os.path.join(PLOTS_DIR, f"rc_feat_seizure_top5.png"), "Seizure")
        plot_rc_dynamics_scatter(feat_list, labels_list, os.path.join(PLOTS_DIR, f"rc_dynamics_scatter_seizure_top5.png"), "Seizure")
        
    print(f"Saved multiple sample visualizations to {PLOTS_DIR}")

def plot_clustering(patient_ids, max_samples_per_class=1000):
    """
    使用 t-SNE 分别对原始 RC 储层特征和经过 FC 层提取的隐层嵌入 (Embeddings) 进行降维和聚类可视化。
    
    参数:
        patient_ids (list): 需要参与聚类的患者 ID 列表
        max_samples_per_class (int): 为避免内存和计算开销，每类最多采样的点数
    """
    print("Generating clustering visualization...")
    all_features = []
    all_labels = []
    
    for pid in patient_ids:
        feat_path = os.path.join(PROCESSED_DATA_DIR, f"{pid}_features.pt")
        if os.path.exists(feat_path):
            features, labels = torch.load(feat_path)
            all_features.append(features)
            all_labels.append(labels)
            
    if not all_features:
        print("No features found for clustering.")
        return
        
    all_features = torch.cat(all_features, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    # Subsample to avoid memory/time issues with TSNE
    normal_idx = np.where(all_labels == 0)[0]
    seizure_idx = np.where(all_labels == 1)[0]
    
    np.random.seed(42)
    if len(normal_idx) > max_samples_per_class:
        normal_idx = np.random.choice(normal_idx, max_samples_per_class, replace=False)
    if len(seizure_idx) > max_samples_per_class:
        seizure_idx = np.random.choice(seizure_idx, max_samples_per_class, replace=False)
        
    selected_idx = np.concatenate([normal_idx, seizure_idx])
    sampled_features = all_features[selected_idx]
    sampled_labels = all_labels[selected_idx]
    
    print(f"Running t-SNE on {len(sampled_features)} samples...")
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(sampled_features)
    
    plt.figure(figsize=(8, 6))
    
    # Plot normal
    normal_mask = sampled_labels == 0
    plt.scatter(features_2d[normal_mask, 0], features_2d[normal_mask, 1], 
                c='blue', label='Normal', alpha=0.5, s=10)
                
    # Plot seizure
    seizure_mask = sampled_labels == 1
    plt.scatter(features_2d[seizure_mask, 0], features_2d[seizure_mask, 1], 
                c='red', label='Seizure', alpha=0.5, s=10)
                
    plt.title("t-SNE Clustering of Raw RC Features")
    plt.legend()
    
    save_path = os.path.join(PLOTS_DIR, "clustering_tsne.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved clustering visualization to {save_path}")
    
    # ---------------------------------------------------------
    # Now plot the learned embeddings from the CNN layer
    # ---------------------------------------------------------
    model_path = os.path.join(PROCESSED_DATA_DIR, "fc_model.pth")
    if os.path.exists(model_path):
        print("Generating CNN embeddings clustering visualization...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CNN1DClassifier(input_dim=1024, output_dim=2).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        with torch.no_grad():
            feats_tensor = torch.tensor(sampled_features, dtype=torch.float32).to(device)
            # CNN1DClassifier forward logic
            batch_size = feats_tensor.size(0)
            x = feats_tensor.view(batch_size, 4, 256)
            x = model.features(x)
            x = x.view(batch_size, -1)
            
            # Extract features from the hidden layer of classifier
            # classifier: Dropout(0), Linear(128, 64)(1), ReLU(2), Dropout(3), Linear(64, 2)(4)
            # We run up to ReLU(2) to get 64D embeddings
            hidden = x
            for i in range(3):
                hidden = model.classifier[i](hidden)
            embeddings = hidden.cpu().numpy()
            
        print(f"Running t-SNE on {len(embeddings)} CNN embeddings...")
        tsne_fc = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne_fc.fit_transform(embeddings)
        
        plt.figure(figsize=(8, 6))
        plt.scatter(embeddings_2d[normal_mask, 0], embeddings_2d[normal_mask, 1], 
                    c='blue', label='Normal', alpha=0.5, s=10)
        plt.scatter(embeddings_2d[seizure_mask, 0], embeddings_2d[seizure_mask, 1], 
                    c='red', label='Seizure', alpha=0.5, s=10)
        plt.title("t-SNE Clustering of CNN Learned Embeddings (64D)")
        plt.legend()
        
        fc_save_path = os.path.join(PLOTS_DIR, "clustering_tsne_cnn.png")
        plt.savefig(fc_save_path)
        plt.close()
        print(f"Saved CNN embeddings clustering visualization to {fc_save_path}")

if __name__ == "__main__":
    visualize_sample("chb01")
    plot_clustering(["chb01"])
