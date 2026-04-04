
import torch
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os

def encode_to_spikes(images, threshold=0.8):
    """
    Convert MNIST images to 4-channel spike trains.
    
    Args:
        images: Tensor of shape (Batch_size, 1, 28, 28)
        threshold: float, threshold for binarization
        
    Returns:
        spikes: Tensor of shape (Batch_size, 4, 196)
    """
    # Ensure input is 4D: (B, C, H, W)
    if images.dim() == 3:
        images = images.unsqueeze(1)
        
    B, _, H, W = images.shape
    
    # 1. Spatial Split
    # We split the 28x28 image into four 14x14 quadrants
    # Quadrants: TL (Top-Left), TR (Top-Right), BL (Bottom-Left), BR (Bottom-Right)
    # Note: image coordinates usually start from top-left (0,0)
    
    # Slicing: [:, :, y_start:y_end, x_start:x_end]
    tl = images[:, :, 0:14, 0:14]   # Top-Left
    tr = images[:, :, 0:14, 14:28]  # Top-Right
    bl = images[:, :, 14:28, 0:14]  # Bottom-Left
    br = images[:, :, 14:28, 14:28] # Bottom-Right
    
    # 2. Temporal Flatten
    # Flatten each 14x14 block into a 1D array of length 196
    # This maps spatial information to temporal domain
    tl_flat = tl.reshape(B, -1) # Shape: (B, 196)
    tr_flat = tr.reshape(B, -1)
    bl_flat = bl.reshape(B, -1)
    br_flat = br.reshape(B, -1)
    
    # 3. Channel Concatenation
    # Stack the four channels along a new dimension (dim=1)
    # Resulting shape: (B, 4, 196)
    # Channel 0: TL, Channel 1: TR, Channel 2: BL, Channel 3: BR
    combined = torch.stack([tl_flat, tr_flat, bl_flat, br_flat], dim=1)
    
    # 4. Binarization (Spike Generation)
    # Values > threshold become 1 (spike), others 0 (no spike)
    spikes = (combined > threshold).float()
    
    return spikes

def plot_spikes(spikes, labels, save_path='plots/spike_trains_visualization.png'):
    """
    Visualize spike trains for 5 samples of each digit (0-9).
    """
    # Select 5 random samples for each digit 0-9
    samples_indices = []
    for digit in range(10):
        # Find indices where label equals digit
        indices = torch.where(labels == digit)[0]
        if len(indices) >= 5:
            # Randomly select 5 indices
            # Use torch.randperm to generate random permutation of indices
            perm = torch.randperm(len(indices))
            random_indices = indices[perm[:5]]
            samples_indices.extend(random_indices.tolist())
        else:
            print(f"Warning: Not enough samples for digit {digit}")
            samples_indices.extend(indices.tolist())
    
    # Create a figure with 10 rows (digits) and 5 columns (samples)
    fig, axes = plt.subplots(10, 5, figsize=(20, 25))
    fig.suptitle('Spike Trains for MNIST Digits (0-9)\n4 Channels (TL, TR, BL, BR) x 196 Time Steps', fontsize=20, y=0.92)
    
    # Define colors for each channel
    # Channel 0 (TL): Red
    # Channel 1 (TR): Green
    # Channel 2 (BL): Blue
    # Channel 3 (BR): Orange
    channel_colors = ['red', 'green', 'blue', 'orange']
    channel_names = ['TL', 'TR', 'BL', 'BR']

    # Adjust layout to prevent overlap
    plt.subplots_adjust(hspace=0.6, wspace=0.3)
    
    for i, idx in enumerate(samples_indices):
        row = i // 5
        col = i % 5
        ax = axes[row, col]
        
        # Get spike data for the current sample: Shape (4, 196)
        spike_data = spikes[idx].numpy()
        
        # Prepare data for eventplot
        # We need a list of lists, where each inner list contains the time steps where a spike occurs
        spike_times = []
        for ch in range(4):
            # Find time indices where value is 1
            times = np.where(spike_data[ch] == 1)[0]
            spike_times.append(times)
        
        # Plot using eventplot
        # lineoffsets determines the y-position of each channel
        # We want Channel 1 at bottom, Channel 4 at top, or vice versa.
        # Let's map Channel 0->1 (TL), 1->2 (TR), 2->3 (BL), 3->4 (BR)
        # Using colors to distinguish channels
        ax.eventplot(spike_times, lineoffsets=[1, 2, 3, 4], linelengths=0.8, colors=channel_colors)
        
        # Formatting
        ax.set_title(f"Digit: {labels[idx].item()} (Idx: {idx})", fontsize=10)
        ax.set_yticks([1, 2, 3, 4])
        ax.set_yticklabels(channel_names, fontsize=8)
        ax.set_xlim(0, 196)
        ax.set_ylim(0.5, 4.5)
        
        # Only show x-axis labels for the bottom row
        if row == 9:
            ax.set_xlabel('Time Step', fontsize=10)
        else:
            ax.set_xticklabels([])

    # Add a single legend to the upper right of the entire figure
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=color, lw=2, label=name) 
                       for color, name in zip(channel_colors, channel_names)]
    fig.legend(handles=legend_elements, loc='upper right', title="Channels", fontsize=12, title_fontsize=14, bbox_to_anchor=(0.95, 0.92))

    print(f"Saving visualization to {save_path}")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    # Setup directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("processed_data", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    print("Step 1: Loading MNIST Dataset...")
    # Define transform to convert images to tensor (values 0-1)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Download and load training and test sets
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    print(f"Loaded {len(train_dataset)} training images and {len(test_dataset)} test images.")
    
    # Create DataLoaders
    # We load the entire dataset into memory for processing since MNIST is small enough
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    
    # Get all data
    train_images, train_labels = next(iter(train_loader))
    test_images, test_labels = next(iter(test_loader))
    
    print("\nStep 2: Encoding Images to Spike Trains...")
    # Process Training Data
    print("Processing Training Set...")
    train_spikes = encode_to_spikes(train_images)
    print(f"Train Spikes Shape: {train_spikes.shape}") # Expected: [60000, 4, 196]
    
    # Process Test Data
    print("Processing Test Set...")
    test_spikes = encode_to_spikes(test_images)
    print(f"Test Spikes Shape: {test_spikes.shape}") # Expected: [10000, 4, 196]
    
    print("\nStep 3: Visualization Validation...")
    plot_spikes(train_spikes, train_labels)
    
    print("\nStep 4: Saving Processed Data...")
    train_save_path = 'processed_data/mnist_spikes_train.pt'
    test_save_path = 'processed_data/mnist_spikes_test.pt'
    
    torch.save((train_spikes, train_labels), train_save_path)
    torch.save((test_spikes, test_labels), test_save_path)
    
    print(f"Training data saved to: {train_save_path}")
    print(f"Test data saved to: {test_save_path}")
    print("\nAll tasks completed successfully!")

if __name__ == "__main__":
    main()
