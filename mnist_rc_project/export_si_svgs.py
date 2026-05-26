"""Export publication-style SVG figures for the MNIST RC supplementary data.

This script redraws the existing PNG figures from saved tensors so text,
axes, markers, and annotations remain editable vector elements in SVG.
The original PNG files are not modified.
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT / "processed_data"
OUTPUT_DIR = ROOT / "plots" / "si_svg"
SEED = 42

R_OFF = 391623.0
R_ON = 42455.0
V_READ = 0.1
G_OFF = 1.0 / R_OFF
G_ON = 1.0 / R_ON

CHANNEL_COLORS = ["#D62728", "#2CA02C", "#1F77B4", "#FF7F0E"]
CHANNEL_NAMES = ["TL", "TR", "BL", "BR"]


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "svg.fonttype": "none",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.linewidth": 0.6,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "legend.fontsize": 7,
            "legend.title_fontsize": 8,
            "savefig.facecolor": "white",
        }
    )


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_tensor_pair(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.load(path, map_location="cpu")


def selected_indices_by_digit(labels: torch.Tensor, samples_per_digit: int = 5) -> list[int]:
    indices: list[int] = []
    for digit in range(10):
        digit_indices = torch.where(labels == digit)[0]
        if len(digit_indices) < samples_per_digit:
            raise ValueError(f"Digit {digit} has only {len(digit_indices)} samples.")
        indices.extend(digit_indices[:samples_per_digit].tolist())
    return indices


class MemristorReservoir(nn.Module):
    def __init__(
        self,
        g_off: float = G_OFF,
        g_on: float = G_ON,
        v_read: float = V_READ,
        decay_rate: float = 0.95,
        sample_interval: int = 7,
    ) -> None:
        super().__init__()
        self.g_off = g_off
        self.g_on = g_on
        self.v_read = v_read
        self.decay_rate = decay_rate
        self.sample_interval = sample_interval

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, time_steps = spike_input.shape
        g_state = torch.full(
            (batch_size, num_channels),
            self.g_off,
            device=spike_input.device,
            dtype=torch.float32,
        )

        sampled_currents = []
        for t in range(time_steps):
            spikes = spike_input[:, :, t]
            spike_mask = (spikes > 0).float()
            update_spike = g_state + (self.g_on - self.g_off) * 0.05
            update_no_spike = g_state * self.decay_rate
            g_state = spike_mask * update_spike + (1 - spike_mask) * update_no_spike
            g_state = torch.clamp(g_state, self.g_off, self.g_on)
            if (t + 1) % self.sample_interval == 0:
                sampled_currents.append(g_state * self.v_read)

        return torch.stack(sampled_currents, dim=2)


class MLPClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 10,
        hidden_dims: tuple[int, int] = (256, 128),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.feature_extractor(x))


def save_svg(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


def plot_spike_trains(output_dir: Path) -> None:
    spikes, labels = load_tensor_pair(PROCESSED_DIR / "mnist_spikes_train.pt")
    sample_indices = selected_indices_by_digit(labels)

    fig, axes = plt.subplots(10, 5, figsize=(7.2, 9.6), sharex=True, sharey=True)
    for panel_idx, sample_idx in enumerate(sample_indices):
        row, col = divmod(panel_idx, 5)
        ax = axes[row, col]
        spike_data = spikes[sample_idx].numpy()
        spike_times = [np.where(spike_data[channel] == 1)[0] for channel in range(4)]

        ax.eventplot(
            spike_times,
            lineoffsets=[1, 2, 3, 4],
            linelengths=0.75,
            colors=CHANNEL_COLORS,
            linewidths=0.45,
        )
        ax.set_xlim(0, 196)
        ax.set_ylim(0.5, 4.5)
        ax.tick_params(pad=1)

        if col == 0:
            ax.set_yticks([1, 2, 3, 4])
            ax.set_yticklabels(CHANNEL_NAMES)
        else:
            ax.tick_params(labelleft=False)

        if col == 4:
            ax.yaxis.set_label_position("right")
            ax.set_ylabel(f"Digit {int(labels[sample_idx])}", rotation=270, labelpad=11)

        if row == 9:
            ax.set_xlabel("Step", labelpad=2)
            ax.set_xticks([0, 98, 196])
        else:
            ax.tick_params(labelbottom=False)

    legend_handles = [
        Line2D([0], [0], color=color, lw=1.4, label=name)
        for color, name in zip(CHANNEL_COLORS, CHANNEL_NAMES)
    ]
    fig.legend(
        handles=legend_handles,
        title="Channels",
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.995),
        frameon=False,
        handlelength=1.5,
        columnspacing=1.1,
    )
    fig.suptitle("MNIST Spike Trains", y=1.012, fontsize=13)
    fig.tight_layout(pad=0.35, h_pad=0.35, w_pad=0.25, rect=[0, 0, 1, 0.97])
    save_svg(fig, output_dir / "spike_trains_visualization.svg")


def plot_feature_heatmap(output_dir: Path) -> None:
    spikes, labels = load_tensor_pair(PROCESSED_DIR / "mnist_spikes_train.pt")
    sample_indices = selected_indices_by_digit(labels)
    selected_spikes = spikes[sample_indices].float()

    reservoir = MemristorReservoir()
    reservoir.eval()
    with torch.no_grad():
        raw_features = reservoir(selected_spikes)

    # Use all selected panels for a consistent SI color range.
    vmin = raw_features.min().item()
    vmax = raw_features.max().item()

    fig, axes = plt.subplots(10, 5, figsize=(7.0, 9.2), sharex=True, sharey=True)
    image = None
    for panel_idx, sample_idx in enumerate(sample_indices):
        row, col = divmod(panel_idx, 5)
        ax = axes[row, col]
        feature_map = raw_features[panel_idx].numpy()
        image = ax.imshow(
            feature_map,
            aspect="auto",
            cmap="hot",
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )
        ax.tick_params(pad=1)

        if col == 0:
            ax.set_yticks(range(4))
            ax.set_yticklabels(CHANNEL_NAMES, fontsize=8.5)
        else:
            ax.tick_params(labelleft=False)

        if col == 4:
            ax.yaxis.set_label_position("right")
            ax.set_ylabel(
                f"Digit {int(labels[sample_idx])}",
                rotation=270,
                labelpad=13,
                fontsize=10,
            )

        if row == 9:
            ax.set_xlabel("Step", labelpad=2, fontsize=9)
            ax.set_xticks([0, raw_features.shape[2] // 2, raw_features.shape[2] - 1])
        else:
            ax.tick_params(labelbottom=False)
        ax.tick_params(labelsize=8)

    fig.suptitle("Memristor Reservoir Monitor Current", y=0.995, fontsize=12)
    fig.tight_layout(pad=0.35, h_pad=0.35, w_pad=0.25, rect=[0, 0.045, 1, 0.985])

    if image is None:
        raise RuntimeError("No heatmap panels were generated.")
    cbar_ax = fig.add_axes([0.18, 0.018, 0.64, 0.014])
    cbar = fig.colorbar(image, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Monitor current (A)", labelpad=1, fontsize=9)
    cbar.ax.tick_params(labelsize=8, width=0.5, length=2.5, pad=1)

    save_svg(fig, output_dir / "mnist_rc_features_heatmap.svg")


def plot_tsne(output_dir: Path, n_samples: int = 2000) -> None:
    features, labels = load_tensor_pair(PROCESSED_DIR / "mnist_rc_features_test.pt")
    sample_count = min(n_samples, len(features))
    rng = np.random.default_rng(SEED)
    indices = rng.choice(len(features), sample_count, replace=False)

    print(f"Running t-SNE on {sample_count} samples...")
    tsne = TSNE(n_components=2, random_state=SEED, init="pca", learning_rate="auto")
    embedded = tsne.fit_transform(features[indices].numpy())
    sampled_labels = labels[indices].numpy()

    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    scatter = ax.scatter(
        embedded[:, 0],
        embedded[:, 1],
        c=sampled_labels,
        cmap="tab10",
        alpha=0.72,
        s=7,
        linewidths=0,
        rasterized=False,
    )
    ax.set_title("t-SNE of RC Features", pad=3)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, linewidth=0.35, alpha=0.25)
    cbar = fig.colorbar(scatter, ax=ax, ticks=range(10), pad=0.025, fraction=0.06)
    cbar.set_label("Digit class", fontsize=8)
    cbar.ax.tick_params(labelsize=7, width=0.5, length=2.5, pad=1)
    fig.tight_layout(pad=0.5)
    save_svg(fig, output_dir / "rc_features_tsne.svg")


def train_classifier(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    epochs: int = 100,
    patience: int = 15,
) -> tuple[float, np.ndarray, np.ndarray]:
    set_seed(SEED)
    device = torch.device("cpu")
    model = MLPClassifier(input_dim=train_features.shape[1]).to(device)

    generator = torch.Generator()
    generator.manual_seed(SEED)
    train_loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=256,
        shuffle=True,
        generator=generator,
    )
    test_loader = DataLoader(
        TensorDataset(test_features, test_labels),
        batch_size=512,
        shuffle=False,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
    )

    best_acc = 0.0
    best_state = None
    best_true: np.ndarray | None = None
    best_pred: np.ndarray | None = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for features_batch, labels_batch in train_loader:
            features_batch = features_batch.to(device)
            labels_batch = labels_batch.to(device)
            optimizer.zero_grad()
            logits = model(features_batch)
            loss = criterion(logits, labels_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * features_batch.size(0)

        model.eval()
        all_preds: list[int] = []
        all_labels: list[int] = []
        with torch.no_grad():
            for features_batch, labels_batch in test_loader:
                logits = model(features_batch.to(device))
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_labels.extend(labels_batch.numpy().tolist())

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        acc = accuracy_score(y_true, y_pred)
        scheduler.step(acc)
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch:03d}/{epochs} - loss {avg_loss:.4f} - test acc {acc * 100:.2f}%")

        if acc > best_acc:
            best_acc = acc
            best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
            best_true = y_true
            best_pred = y_pred
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    if best_state is None or best_true is None or best_pred is None:
        raise RuntimeError("Classifier training did not produce a valid model.")
    model.load_state_dict(best_state)
    print(f"Best deterministic test accuracy: {best_acc * 100:.2f}%")
    return best_acc, best_true, best_pred


def plot_confusion_matrix(output_dir: Path, epochs: int, patience: int) -> None:
    train_features, train_labels = load_tensor_pair(PROCESSED_DIR / "mnist_rc_features_train.pt")
    test_features, test_labels = load_tensor_pair(PROCESSED_DIR / "mnist_rc_features_test.pt")
    best_acc, y_true, y_pred = train_classifier(
        train_features,
        train_labels,
        test_features,
        test_labels,
        epochs=epochs,
        patience=patience,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))

    fig, ax = plt.subplots(figsize=(3.45, 3.05))
    image = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    ax.set_title(f"Confusion Matrix ({best_acc * 100:.2f}%)", pad=3)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.tick_params(pad=1)

    threshold = cm.max() * 0.55
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(
                col,
                row,
                f"{cm[row, col]}",
                ha="center",
                va="center",
                fontsize=5.6,
                color="white" if cm[row, col] > threshold else "black",
            )

    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, ax=ax, pad=0.025, fraction=0.06)
    cbar.set_label("Count", fontsize=8)
    cbar.ax.tick_params(labelsize=7, width=0.5, length=2.5, pad=1)
    cbar.outline.set_visible(False)
    for spine in cbar.ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0.5)
    save_svg(fig, output_dir / "rc_features_confusion_matrix.svg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export MNIST RC supplementary figures as publication-style SVG files."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for exported SVG files.",
    )
    parser.add_argument(
        "--tsne-samples",
        type=int,
        default=2000,
        help="Number of test samples used for the t-SNE panel.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum epochs for deterministic confusion-matrix retraining.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early-stopping patience for deterministic retraining.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    configure_style()
    output_dir = args.output_dir.resolve()

    print(f"Exporting SVG figures to {output_dir}")
    plot_spike_trains(output_dir)
    plot_feature_heatmap(output_dir)
    plot_tsne(output_dir, n_samples=args.tsne_samples)
    plot_confusion_matrix(output_dir, epochs=args.epochs, patience=args.patience)
    print("All SI SVG exports completed.")


if __name__ == "__main__":
    main()
