"""Figures used by the paper: trained-model confusion matrices and the cost/accuracy map."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(r"E:\codexwork\cifar10-local-experiment")
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
DISPLAY = {
    "resnet20": "ResNet-20",
    "resnet32": "ResNet-32",
    "resnet56": "ResNet-56",
    "vgg16_bn": "VGG-16-BN",
    "mobilenetv2_x1_0": "MobileNetV2",
    "se_resnet20": "SE-ResNet-20\n(ours)",
}


def plot_confusions() -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), dpi=200)
    for ax, name, title in zip(
        axes,
        ("resnet20_trained", "se_resnet20_trained"),
        ("ResNet-20 (baseline)", "SE-ResNet-20 (ours)"),
    ):
        matrix = np.loadtxt(RESULTS / f"{name}_confusion.csv", delimiter=",", skiprows=1)
        image = ax.imshow(matrix, cmap="Blues")
        accuracy = 100.0 * np.trace(matrix) / matrix.sum()
        ax.set_title(f"{title}  Top-1 {accuracy:.2f}%", fontsize=11)
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xticklabels(CLASSES, rotation=90, fontsize=8)
        ax.set_yticklabels(CLASSES, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = FIGURES / "confusion_trained.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_tradeoff() -> Path:
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    trained = [
        json.loads((RESULTS / f"{name}_trained_metrics.json").read_text(encoding="utf-8"))
        for name in ("resnet20", "se_resnet20")
    ]

    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=200)
    for row in summary:
        ax.scatter(row["parameters"] / 1e6, row["top1_accuracy_percent"],
                   s=70, color="#2b6cb0", zorder=3)
        ax.annotate(DISPLAY.get(row["model"], row["model"]),
                    (row["parameters"] / 1e6, row["top1_accuracy_percent"]),
                    textcoords="offset points", xytext=(7, 5), fontsize=8)
    for row in trained:
        color = "#c53030" if row["model"] == "se_resnet20" else "#dd6b20"
        ax.scatter(row["parameters"] / 1e6, row["top1_accuracy_percent"],
                   marker="s", s=80, color=color, zorder=4)
        ax.annotate(DISPLAY.get(row["model"], row["model"]),
                    (row["parameters"] / 1e6, row["top1_accuracy_percent"]),
                    textcoords="offset points", xytext=(7, -12), fontsize=8,
                    color=color)

    ax.set_xscale("log")
    ax.set_xlabel("Parameters (millions, log scale)")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title("Accuracy vs. model size (blue: public checkpoints, orange/red: trained here)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIGURES / "tradeoff.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    print(plot_confusions())
    print(plot_tradeoff())


if __name__ == "__main__":
    main()
