"""
科研级脉冲编码对比图绘制脚本
生成单样本对比SVG矢量图：Normal第1样本 与 Seizure第5样本上下拼接

策略：
- 重新加载原始脉冲数据（chb01_spikes.pt）
- 使用与visualize_sample()相同的随机种子(42)复现样本选择
- 提取Normal第1个样本和Seizure第5个样本
- 上下拼接绘制并输出高质量SVG

输出：适合论文插图的矢量图
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from config import PROCESSED_DATA_DIR, PLOTS_DIR

# 科研图样式设置
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
rcParams['axes.linewidth'] = 0.8
rcParams['xtick.major.width'] = 0.8
rcParams['ytick.major.width'] = 0.8
rcParams['xtick.major.size'] = 3
rcParams['ytick.major.size'] = 3


def load_specific_samples(patient_id="chb01"):
    """
    加载特定患者的脉冲数据，并选择指定的样本。
    
    使用与 visualize_sample() 相同的随机种子(42)和选择逻辑，
    确保复现相同的样本索引。
    
    返回:
        normal_sample: Normal类别第1个样本 [4, 512]
        seizure_sample: Seizure类别第5个样本 [4, 512]
    """
    spike_path = os.path.join(PROCESSED_DATA_DIR, f"{patient_id}_spikes.pt")
    
    if not os.path.exists(spike_path):
        raise FileNotFoundError(f"数据文件未找到: {spike_path}")
    
    spikes, labels = torch.load(spike_path)
    
    # 找到正常和癫痫样本的索引
    normal_idx = (labels == 0).nonzero(as_tuple=True)[0]
    seizure_idx = (labels == 1).nonzero(as_tuple=True)[0]
    
    # 使用相同的随机种子确保可复现
    np.random.seed(42)
    
    # 选择5个样本（与visualize_sample一致）
    num_samples = 5
    selected_normal = np.random.choice(normal_idx.numpy(), min(num_samples, len(normal_idx)), replace=False)
    selected_seizure = np.random.choice(seizure_idx.numpy(), min(num_samples, len(seizure_idx)), replace=False)
    
    # 取第1个Normal样本和第5个Seizure样本
    normal_sample = spikes[selected_normal[0]].numpy()  # 第1个
    seizure_sample = spikes[selected_seizure[4]].numpy()  # 第5个
    
    print(f"已加载样本:")
    print(f"  - Normal Sample 1 (索引: {selected_normal[0]})")
    print(f"  - Seizure Sample 5 (索引: {selected_seizure[4]})")
    
    return normal_sample, seizure_sample


def plot_spike_sample(ax, spikes, title, colors=None):
    """
    在指定axes上绘制单个脉冲样本。
    
    参数:
        ax: matplotlib axes对象
        spikes: 脉冲数据 [4, 512]
        title: 图表标题
        colors: 各通道颜色列表
    """
    if colors is None:
        colors = ['#e41a1c', '#4daf4a', '#377eb8', '#ff7f00']  # 红绿蓝橙
    
    channel_labels = ['Delta', 'Theta', 'Alpha', 'Beta']
    
    # 提取每个通道的脉冲时间
    spike_times = []
    for c in range(4):
        times = np.where(spikes[c] == 1)[0]
        spike_times.append(times)
    
    # 使用eventplot绘制脉冲栅格图
    ax.eventplot(spike_times, lineoffsets=[1, 2, 3, 4], linelengths=0.8, 
                 colors=colors, linewidths=1.5)
    
    # 设置Y轴
    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels(channel_labels, fontsize=11)
    ax.set_ylim(0.5, 4.5)
    
    # 设置X轴
    ax.set_xlim(0, 512)
    ax.set_xlabel("Time Step", fontsize=11)
    
    # 设置标题
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    
    # 使用完整方框坐标轴
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    
    # 添加明显的横向网格线
    ax.grid(axis='x', linestyle='--', alpha=0.6, linewidth=0.8)


def create_comparison_plot(normal_sample, seizure_sample, output_path=None):
    """
    创建上下拼接的对比图：Seizure在上，Normal在下。
    
    参数:
        normal_sample: 正常样本脉冲数据 [4, 512]
        seizure_sample: 癫痫样本脉冲数据 [4, 512]
        output_path: SVG输出路径，默认为 plots/spikes_comparison.svg
    """
    if output_path is None:
        output_path = os.path.join(PLOTS_DIR, "spikes_comparison.svg")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 创建图形：2行1列，Seizure在上，Normal在下
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), dpi=300)
    
    # 配色方案（与用户参考图一致）
    colors = ['#e41a1c', '#4daf4a', '#377eb8', '#ff7f00']  # Delta红, Theta绿, Alpha蓝, Beta橙
    
    # 绘制Seizure样本（上半部分）
    plot_spike_sample(axes[0], seizure_sample, "Seizure", colors)
    axes[0].set_xlabel("")  # 移除中间图的x轴标签
    
    # 绘制Normal样本（下半部分）
    plot_spike_sample(axes[1], normal_sample, "Normal", colors)
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    
    # 保存为SVG矢量图
    plt.savefig(output_path, format='svg', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"\nSVG矢量图已保存至: {output_path}")
    print(f"文件格式: SVG (可无损缩放，适合论文插图)")
    
    return output_path


def main():
    """主函数：加载数据并生成对比图。"""
    print("=" * 50)
    print("科研级脉冲编码对比图生成")
    print("=" * 50)
    
    try:
        # 加载特定样本
        normal_sample, seizure_sample = load_specific_samples("chb01")
        
        # 生成对比图
        output_path = create_comparison_plot(normal_sample, seizure_sample)
        
        print("\n生成完成!")
        print(f"输出文件: {output_path}")
        
    except FileNotFoundError as e:
        print(f"\n错误: {e}")
        print("提示: 请先运行 run_pipeline.py 生成脉冲数据文件")
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
