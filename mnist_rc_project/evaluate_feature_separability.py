
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix
import os


# ============ MLP Classifier with BN and ReLU ============

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=10, hidden_dims=[256, 128], dropout=0.3):
        super(MLPClassifier, self).__init__()
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        print(list(nn.Sequential(*layers)))
        self.classifier = nn.Linear(prev_dim, num_classes)

        
        # Kaiming initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x


# ============ Data Loading ============

def load_data():
    try:
        train_features, train_labels = torch.load('processed_data/mnist_rc_features_train.pt')
        test_features, test_labels = torch.load('processed_data/mnist_rc_features_test.pt')
        
        print(f"Loaded Train: {train_features.shape}, Labels: {train_labels.shape}")
        print(f"Loaded Test:  {test_features.shape}, Labels: {test_labels.shape}")
        
        return train_features, train_labels, test_features, test_labels
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None


# ============ Training & Evaluation ============

def train_model(model, train_loader, test_loader, device, epochs=100, lr=1e-3, patience=15):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    best_acc = 0.0
    best_state = None
    patience_counter = 0
    
    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * X_batch.size(0)
        
        avg_loss = total_loss / len(train_loader.dataset)
        
        # ---- Evaluate ----
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y_batch.numpy())
        
        acc = accuracy_score(all_labels, all_preds)
        scheduler.step(acc)
        
        print(f"Epoch [{epoch:03d}/{epochs}] Loss: {avg_loss:.4f} | Test Acc: {acc*100:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # Load best model
    model.load_state_dict(best_state)
    return best_acc, all_labels, all_preds


def plot_tsne(features, labels, save_path, n_samples=2000):
    print(f"Running t-SNE on {n_samples} samples...")
    indices = np.random.choice(len(features), min(n_samples, len(features)), replace=False)
    
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    embedded = tsne.fit_transform(features[indices])
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        embedded[:, 0], embedded[:, 1],
        c=labels[indices], cmap='tab10', alpha=0.6, s=10
    )
    plt.colorbar(scatter, ticks=range(10), label='Digit Class')
    plt.title('t-SNE Visualization of RC Features')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(True, alpha=0.3)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved t-SNE plot to {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path, acc):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.title(f'Confusion Matrix (Accuracy: {acc*100:.2f}%)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved confusion matrix to {save_path}")


# ============ Main ============

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_features, train_labels, test_features, test_labels = load_data()
    if train_features is None:
        return
    
    input_dim = train_features.shape[1]
    print(f"Input feature dimension: {input_dim}")
    
    # DataLoader
    train_dataset = TensorDataset(train_features, train_labels)
    test_dataset = TensorDataset(test_features, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    
    # Model
    model = MLPClassifier(
        input_dim=input_dim,
        num_classes=10,
        hidden_dims=[256, 128],
        dropout=0.3
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")
    
    # Train
    best_acc, y_true, y_pred = train_model(
        model, train_loader, test_loader, device,
        epochs=100, lr=1e-3, patience=15
    )
    
    print(f"\n=== Best Test Accuracy: {best_acc*100:.2f}% ===")
    
    # Visualizations
    plot_tsne(test_features.numpy(), test_labels.numpy(), 'plots/rc_features_tsne.png')
    plot_confusion_matrix(y_true, y_pred, 'plots/rc_features_confusion_matrix.png', best_acc)


if __name__ == "__main__":
    main()
