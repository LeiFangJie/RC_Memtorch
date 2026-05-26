"""
Generate a four-panel publication-style EEG-RC figure.

Panels:
    a. RC dynamics for one Normal and one Seizure sample
    b. t-SNE clustering of raw RC features
    c. RC feature heatmaps for the same samples used in panel a
    d. Confusion matrix from the saved CNN model on the reproducible test split
"""
import os
import random

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

from config import PLOTS_DIR, PROCESSED_DATA_DIR, TEST_PATIENTS_RATIO
from train_classifier import CNN1DClassifier, smooth_predictions


PATIENT_ID = "chb01"
RANDOM_SEED = 42
MAX_TSNE_SAMPLES_PER_CLASS = 1000
BAND_LABELS = ["Delta", "Theta", "Alpha", "Beta"]
BAND_COLORS = {
    "Delta": "#e45756",
    "Theta": "#59a14f",
    "Alpha": "#4e79a7",
    "Beta": "#f28e2b",
}
CLASS_COLORS = {
    "Normal": "#4c6ef5",
    "Seizure": "#ff6b6b",
}


def set_publication_style():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 8,
            "axes.linewidth": 0.75,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "legend.fontsize": 7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def load_features(patient_id=PATIENT_ID):
    feat_path = os.path.join(PROCESSED_DATA_DIR, f"{patient_id}_features.pt")
    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"Feature file not found: {feat_path}")
    return torch.load(feat_path, map_location="cpu")


def choose_plan_samples(labels):
    normal_idx = (labels == 0).nonzero(as_tuple=True)[0].numpy()
    seizure_idx = (labels == 1).nonzero(as_tuple=True)[0].numpy()

    rng = np.random.RandomState(RANDOM_SEED)
    selected_normal = rng.choice(normal_idx, min(5, len(normal_idx)), replace=False)
    selected_seizure = rng.choice(seizure_idx, min(5, len(seizure_idx)), replace=False)

    if len(selected_normal) < 1 or len(selected_seizure) < 2:
        raise ValueError("Not enough Normal/Seizure samples to select Normal Sample 1 and Seizure Sample 2.")

    return int(selected_normal[0]), int(selected_seizure[1])


def normalize_channels(feat_map):
    feat_map = feat_map.astype(np.float64, copy=True)
    for i in range(feat_map.shape[0]):
        low = np.percentile(feat_map[i], 1)
        high = np.percentile(feat_map[i], 99)
        if high > low:
            feat_map[i] = np.clip((feat_map[i] - low) / (high - low), 0, 1)
        else:
            feat_map[i] = 0
    return feat_map


def style_box(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
        spine.set_color("#4a4a4a")
    ax.tick_params(colors="#2b2b2b", width=0.65, length=2.8)


def add_panel_label(ax, label):
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def plot_rc_dynamics(ax, feature, title, show_xlabel=False):
    feat_map = normalize_channels(feature.view(4, -1).numpy())
    time = np.arange(feat_map.shape[1])
    offsets = np.arange(4)[::-1]

    for band_i, band in enumerate(BAND_LABELS):
        y = offsets[band_i] + feat_map[band_i] * 0.72
        ax.plot(time, y, color=BAND_COLORS[band], lw=1.15, solid_capstyle="round")
        ax.scatter(time[::4], y[::4], color=BAND_COLORS[band], s=3.8, linewidths=0, alpha=0.9)

    ax.set_xlim(0, feat_map.shape[1] - 1)
    ax.set_ylim(-0.25, 3.95)
    ax.set_yticks(offsets + 0.36)
    ax.set_yticklabels(BAND_LABELS)
    ax.set_title(title, pad=4)
    ax.set_xlabel("Time step" if show_xlabel else "", labelpad=3)
    ax.tick_params(axis="x", labelbottom=show_xlabel)
    ax.grid(axis="x", color="#d9d9d9", linestyle="--", linewidth=0.55, alpha=0.75)
    style_box(ax)


def plot_heatmap(ax, feature, title, vmin, vmax, show_xlabel=False):
    feat_map = feature.view(4, -1).numpy()
    im = ax.imshow(
        feat_map,
        aspect="auto",
        cmap="magma",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_yticks(np.arange(4))
    ax.set_yticklabels(BAND_LABELS)
    ax.set_title(title, pad=4)
    ax.set_xlabel("Time step" if show_xlabel else "", labelpad=3)
    ax.tick_params(axis="x", labelbottom=show_xlabel)
    style_box(ax)
    return im


def compute_tsne(features, labels):
    labels_np = labels.numpy()
    normal_idx = np.where(labels_np == 0)[0]
    seizure_idx = np.where(labels_np == 1)[0]
    rng = np.random.RandomState(RANDOM_SEED)

    if len(normal_idx) > MAX_TSNE_SAMPLES_PER_CLASS:
        normal_idx = rng.choice(normal_idx, MAX_TSNE_SAMPLES_PER_CLASS, replace=False)
    if len(seizure_idx) > MAX_TSNE_SAMPLES_PER_CLASS:
        seizure_idx = rng.choice(seizure_idx, MAX_TSNE_SAMPLES_PER_CLASS, replace=False)

    selected = np.concatenate([normal_idx, seizure_idx])
    sampled_features = features[selected].numpy()
    sampled_labels = labels_np[selected]

    tsne = TSNE(
        n_components=2,
        random_state=RANDOM_SEED,
        init="pca",
        learning_rate="auto",
        perplexity=30,
    )
    return tsne.fit_transform(sampled_features), sampled_labels


def plot_tsne(ax, coords, labels):
    normal = labels == 0
    seizure = labels == 1
    ax.scatter(
        coords[normal, 0],
        coords[normal, 1],
        s=9,
        c=CLASS_COLORS["Normal"],
        label="Normal",
        alpha=0.72,
        linewidths=0,
    )
    ax.scatter(
        coords[seizure, 0],
        coords[seizure, 1],
        s=9,
        c=CLASS_COLORS["Seizure"],
        label="Seizure",
        alpha=0.72,
        linewidths=0,
    )
    ax.set_title("t-SNE clustering of raw RC features", pad=5)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(frameon=True, edgecolor="#d0d0d0", facecolor="white", loc="upper right")
    style_box(ax)


def reproducible_test_indices(labels):
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    normal_indices = np.where(labels.numpy() == 0)[0]
    seizure_indices = np.where(labels.numpy() == 1)[0]
    np.random.shuffle(normal_indices)
    np.random.shuffle(seizure_indices)

    normal_split = int(len(normal_indices) * (1 - TEST_PATIENTS_RATIO))
    seizure_split = int(len(seizure_indices) * (1 - TEST_PATIENTS_RATIO))
    test_indices = np.concatenate([normal_indices[normal_split:], seizure_indices[seizure_split:]])
    return np.sort(test_indices)


def compute_confusion_matrix(features, labels):
    model_path = os.path.join(PROCESSED_DATA_DIR, "fc_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    test_indices = reproducible_test_indices(labels)
    x_test = features[test_indices]
    y_test = labels[test_indices].numpy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN1DClassifier(input_dim=1024, output_dim=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    preds = []
    with torch.no_grad():
        for start in range(0, len(x_test), 256):
            batch = x_test[start : start + 256].to(device)
            outputs = model(batch)
            preds.extend(outputs.argmax(dim=1).cpu().numpy())

    smoothed = smooth_predictions(preds, window_size=5)
    return confusion_matrix(y_test, smoothed, labels=[0, 1])


def plot_confusion(ax, cm):
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
    ax.set_title("Confusion matrix", pad=5)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label", labelpad=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Normal", "Seizure"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Normal", "Seizure"])

    threshold = cm.max() * 0.55
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > threshold else "#4a4a4a"
            ax.text(j, i, f"{cm[i, j]:d}", ha="center", va="center", fontsize=8, color=color)

    style_box(ax)
    return im


def build_figure():
    set_publication_style()
    features, labels = load_features()
    normal_sample_idx, seizure_sample_idx = choose_plan_samples(labels)
    normal_feature = features[normal_sample_idx]
    seizure_feature = features[seizure_sample_idx]

    print(f"Using Normal Sample 1 index: {normal_sample_idx}")
    print(f"Using Seizure Sample 2 index: {seizure_sample_idx}")
    print("Computing t-SNE...")
    tsne_coords, tsne_labels = compute_tsne(features, labels)
    print("Computing confusion matrix...")
    cm = compute_confusion_matrix(features, labels)

    fig = plt.figure(figsize=(7.2, 6.6), dpi=300)
    outer = GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[1.12, 1.0],
        height_ratios=[1.0, 1.0],
        wspace=0.55,
        hspace=0.48,
    )

    gs_a = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0, 0], hspace=0.30)
    ax_a_normal = fig.add_subplot(gs_a[0, 0])
    ax_a_seizure = fig.add_subplot(gs_a[1, 0])
    plot_rc_dynamics(ax_a_normal, normal_feature, "Normal")
    plot_rc_dynamics(ax_a_seizure, seizure_feature, "Seizure", show_xlabel=True)
    add_panel_label(ax_a_normal, "a")

    ax_b = fig.add_subplot(outer[0, 1])
    plot_tsne(ax_b, tsne_coords, tsne_labels)
    add_panel_label(ax_b, "b")

    heat_values = torch.stack([normal_feature, seizure_feature]).view(2, 4, -1).numpy()
    vmin, vmax = np.percentile(heat_values, [1, 99])
    gs_c = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1, 0], hspace=0.34)
    ax_c_normal = fig.add_subplot(gs_c[0, 0])
    ax_c_seizure = fig.add_subplot(gs_c[1, 0])
    heat_im = plot_heatmap(ax_c_normal, normal_feature, "Normal", vmin, vmax)
    plot_heatmap(ax_c_seizure, seizure_feature, "Seizure", vmin, vmax, show_xlabel=True)
    add_panel_label(ax_c_normal, "c")
    cbar = fig.colorbar(heat_im, ax=[ax_c_normal, ax_c_seizure], fraction=0.035, pad=0.025)
    cbar.set_label("RC state", fontsize=8, labelpad=2)
    cbar.ax.tick_params(labelsize=7, width=0.65, length=2.8)

    ax_d = fig.add_subplot(outer[1, 1])
    cm_im = plot_confusion(ax_d, cm)
    add_panel_label(ax_d, "d")
    cbar_cm = fig.colorbar(cm_im, ax=ax_d, fraction=0.046, pad=0.035)
    cbar_cm.set_label("Count", fontsize=8)
    cbar_cm.ax.tick_params(labelsize=7, width=0.65, length=2.8)

    return fig


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig = build_figure()
    base = os.path.join(PLOTS_DIR, "paper_figure_abcd")
    for ext in ("svg", "pdf", "png"):
        path = f"{base}.{ext}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
