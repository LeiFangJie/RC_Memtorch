
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import os

def load_data():
    """Load train and test features"""
    try:
        train_features, train_labels = torch.load('processed_data/mnist_rc_features_train.pt')
        test_features, test_labels = torch.load('processed_data/mnist_rc_features_test.pt')
        
        # Convert to numpy for sklearn
        X_train = train_features.numpy()
        y_train = train_labels.numpy()
        X_test = test_features.numpy()
        y_test = test_labels.numpy()
        
        print(f"Loaded Train Data: {X_train.shape}, Labels: {y_train.shape}")
        print(f"Loaded Test Data: {X_test.shape}, Labels: {y_test.shape}")
        
        return X_train, y_train, X_test, y_test
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None

def plot_tsne(X, y, save_path, n_samples=2000):
    """
    Visualize high-dimensional features using t-SNE.
    Since t-SNE is computationally expensive, we use a subset of data.
    """
    print(f"Running t-SNE on {n_samples} random samples...")
    
    # Random sampling
    indices = np.random.choice(len(X), n_samples, replace=False)
    X_subset = X[indices]
    y_subset = y[indices]
    
    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X_subset)
    
    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_subset, cmap='tab10', alpha=0.6, s=10)
    plt.colorbar(scatter, ticks=range(10), label='Digit Class')
    plt.title(f't-SNE Visualization of RC Features (Subset: {n_samples})')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(True, alpha=0.3)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Saved t-SNE plot to {save_path}")
    plt.close()

def evaluate_linear_separability(X_train, y_train, X_test, y_test, save_path):
    """
    Train a simple linear classifier (Logistic Regression) to evaluate linear separability.
    If the accuracy is high, the features are linearly separable, which is ideal for RC.
    """
    print("Training Linear Classifier (Logistic Regression)...")
    
    # Increase max_iter for convergence
    # Using default solver and multi_class for better compatibility
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    
    # Predict
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Linear Classifier Accuracy on Test Set: {acc * 100:.2f}%")
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.title(f'Confusion Matrix (Accuracy: {acc * 100:.2f}%)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Saved Confusion Matrix to {save_path}")
    plt.close()
    
    return acc

def main():
    X_train, y_train, X_test, y_test = load_data()
    
    if X_train is not None:
        # 1. t-SNE Visualization (Qualitative)
        plot_tsne(X_test, y_test, 'plots/rc_features_tsne.png', n_samples=2000)
        
        # 2. Linear Separability Check (Quantitative)
        acc = evaluate_linear_separability(X_train, y_train, X_test, y_test, 'plots/rc_features_confusion_matrix.png')
        
        print("\n=== Evaluation Summary ===")
        print(f"Feature Dimension: {X_train.shape[1]}")
        print(f"Linear Readout Accuracy: {acc * 100:.2f}%")
        if acc > 0.85:
            print("Conclusion: The RC features show GOOD separability!")
        elif acc > 0.70:
            print("Conclusion: The RC features show MODERATE separability.")
        else:
            print("Conclusion: The RC features show POOR separability. Consider tuning reservoir parameters.")

if __name__ == "__main__":
    main()
