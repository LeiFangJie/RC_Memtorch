
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

# Physical parameters
r_off = 391623.0
r_on = 42455.0
v_read = 0.1
g_off = 1.0 / r_off
g_on = 1.0 / r_on

print(f"Physical Parameters:")
print(f"r_off = {r_off} Ohm")
print(f"r_on = {r_on} Ohm")
print(f"v_read = {v_read} V")
print(f"Calculated g_off = {g_off:.2e} S")
print(f"Calculated g_on = {g_on:.2e} S")

class MemristorReservoir(nn.Module):
    def __init__(self, g_off, g_on, v_read, decay_rate=0.95, sample_interval=14):
        super().__init__()
        self.g_off = g_off
        self.g_on = g_on
        self.v_read = v_read
        self.decay_rate = decay_rate
        self.sample_interval = sample_interval

    def forward(self, spike_input):
        # spike_input: [Batch, 4, Time]
        batch_size, num_channels, time_steps = spike_input.shape
        
        # Initialize g_state to g_off
        g_state = torch.full((batch_size, num_channels), self.g_off, device=spike_input.device, dtype=torch.float32)
        
        sampled_currents = []
        
        # Time steps 0 to 195 (total 196)
        for t in range(time_steps):
            spikes = spike_input[:, :, t]  # [Batch, 4]
            
            # Dynamics
            # If spike (1): g_state += (g_on - g_off) * 0.05
            # If no spike (0): g_state *= decay_rate
            
            spike_mask = (spikes > 0).float()
            
            # Update logic
            # Note: We apply update for spike and no-spike separately
            update_spike = g_state + (self.g_on - self.g_off) * 0.05
            update_no_spike = g_state * self.decay_rate
            
            g_state = spike_mask * update_spike + (1 - spike_mask) * update_no_spike
            
            # Limit range: ensure g_state is within [g_off, g_on]
            g_state = torch.clamp(g_state, self.g_off, self.g_on)
            
            # Calculate monitor current
            current = g_state * self.v_read
            
            # Virtual node sampling: every 14 time steps
            # t is 0-indexed. If we want 14 samples from 196 steps, 196/14 = 14.
            # We should sample at t=13, 27, ..., 195.
            if (t + 1) % self.sample_interval == 0:
                sampled_currents.append(current)
                
        # Stack sampled currents: [Batch, 4, NumSamples]
        # NumSamples should be 14
        output = torch.stack(sampled_currents, dim=2)
        
        return output

def min_max_normalize(tensor, min_val=None, max_val=None):
    """
    Normalize tensor to [0, 1] based on provided min/max or batch min/max.
    For Memristor RC, we should use the theoretical range [g_off*v, g_on*v] 
    to preserve physical meaning and relative intensity.
    """
    if min_val is None:
        min_val = tensor.min()
    if max_val is None:
        max_val = tensor.max()
        
    # Avoid division by zero
    diff = max_val - min_val
    if diff == 0:
        return tensor # or zeros?
        
    normalized = (tensor - min_val) / diff
    return normalized

def process_data(data_path, save_path, reservoir, is_train=False):
    if not os.path.exists(data_path):
        print(f"Data not found: {data_path}")
        return None, None

    print(f"Loading {data_path}...")
    spikes, labels = torch.load(data_path)
    
    # Ensure spikes are on the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reservoir.to(device)
    
    print(f"Processing data shape: {spikes.shape} on {device}...")
    batch_size = 100
    features_list = []
    
    with torch.no_grad():
        for i in range(0, len(spikes), batch_size):
            batch = spikes[i:i+batch_size].to(device)
            # Ensure float type
            if batch.dtype != torch.float32:
                batch = batch.float()
                
            features = reservoir(batch)
            features_list.append(features.cpu())
            
            if (i + batch_size) % 5000 == 0:
                print(f"Processed {min(i + batch_size, len(spikes))}/{len(spikes)}")
                
    raw_features = torch.cat(features_list, dim=0) # [N, 4, 14]
    
    # Normalize using theoretical limits to preserve physical meaning
    # range: [g_off * v_read, g_on * v_read]
    theoretical_min = reservoir.g_off * reservoir.v_read
    theoretical_max = reservoir.g_on * reservoir.v_read
    print(f"Normalizing with theoretical range: [{theoretical_min:.2e}, {theoretical_max:.2e}]")
    
    norm_features = min_max_normalize(raw_features, theoretical_min, theoretical_max)
    
    # Flatten for saving: [N, 56]
    flat_features = norm_features.view(norm_features.size(0), -1)
    
    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save((flat_features, labels), save_path)
    print(f"Saved features to {save_path}, shape: {flat_features.shape}")
    
    return raw_features, labels

def visualize_features(raw_features, labels, save_path):
    print("Generating visualization...")
    # Find 5 samples for each digit
    unique_labels = sorted(torch.unique(labels).tolist())
    
    fig, axes = plt.subplots(10, 5, figsize=(15, 20)) # Adjusted height
    
    # Set global title
    fig.suptitle('Memristor Reservoir Normalized State', fontsize=16)
    
    # Determine global min/max for consistent colorbar
    vmin = raw_features.min().item()
    vmax = raw_features.max().item()
    
    for digit in unique_labels:
        indices = (labels == digit).nonzero(as_tuple=True)[0]
        selected_indices = indices[:5]
        
        for i, idx in enumerate(selected_indices):
            if digit < 10 and i < 5: # Safety check
                ax = axes[digit, i]
                feature_map = raw_features[idx].numpy() # [4, 14]
                
                # Plot heatmap
                im = ax.imshow(feature_map, aspect='auto', cmap='hot', interpolation='nearest', vmin=vmin, vmax=vmax)
                
                if i == 2: # Title on middle column
                    ax.set_title(f"Digit {digit}", fontsize=10)
                
                if i == 0:
                    ax.set_ylabel("Channel")
                else:
                    ax.set_yticks([])
                    
                if digit == 9:
                    ax.set_xlabel("Time Step")
                else:
                    ax.set_xticks([])

    # Add colorbar
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    cbar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.02])
    sm = plt.cm.ScalarMappable(cmap='hot', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', label='Normalized State')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Saved visualization to {save_path}")

def main():
    reservoir = MemristorReservoir(g_off, g_on, v_read, sample_interval=7)
    
    # Train Data
    train_data_path = 'processed_data/mnist_spikes_train.pt'
    train_save_path = 'processed_data/mnist_rc_features_train.pt'
    raw_train_features, train_labels = process_data(train_data_path, train_save_path, reservoir, is_train=True)
    
    if raw_train_features is not None:
        visualize_features(raw_train_features, train_labels, 'plots/mnist_rc_features_heatmap.png')
        
    # Test Data
    test_data_path = 'processed_data/mnist_spikes_test.pt'
    test_save_path = 'processed_data/mnist_rc_features_test.pt'
    process_data(test_data_path, test_save_path, reservoir, is_train=False)

if __name__ == "__main__":
    main()
