"""Generate report figures from evaluate.py outputs.

Produces, under results/figures/:
  accuracy_comparison.png   - top-1 / top-5 bar chart
  confusion_matrices.png    - 5 panels of confusion matrices
  params_vs_accuracy.png    - parameter count vs top-1 accuracy
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)

DISPLAY = {
    "resnet20": "ResNet-20",
    "resnet32": "ResNet-32",
    "resnet56": "ResNet-56",
    "vgg16_bn": "VGG-16-BN",
    "mobilenetv2_x1_0": "MobileNetV2 x1.0",
}


def load_summary() -> list[dict]:
    return json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))


def plot_accuracy(summary: list[dict]) -> Path:
    names = [DISPLAY.get(r["model"], r["model"]) for r in summary]
    top1 = [r["top1_accuracy_percent"] for r in summary]
    top5 = [r["top5_accuracy_percent"] for r in summary]
    x = np.arange(len(names))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), dpi=200)

    ax1.bar(x, top1, 0.55, color="#2b8cbe")
    for xi, v in zip(x, top1):
        ax1.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=20, ha="right")
    ax1.set_ylim(91, 95.5)
    ax1.set_ylabel("Top-1 accuracy (%)")
    ax1.set_title("Top-1 accuracy")
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(x, top5, 0.55, color="#a6bddb")
    for xi, v in zip(x, top5):
        ax2.text(xi, v + 0.005, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=20, ha="right")
    ax2.set_ylim(99.0, 100.0)
    ax2.set_ylabel("Top-5 accuracy (%)")
    ax2.set_title("Top-5 accuracy")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("CIFAR-10 test-set accuracy (n = 10,000)", fontsize=12)
    fig.tight_layout()
    out = FIGURES / "accuracy_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_confusions(summary: list[dict]) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=200)
    axes = axes.ravel()
    for ax, row in zip(axes, summary):
        model = row["model"]
        matrix = np.loadtxt(
            RESULTS / f"confusion_matrix_{model}.csv",
            delimiter=",",
            skiprows=1,
        )
        image = ax.imshow(matrix, cmap="Blues")
        ax.set_title(f"{DISPLAY.get(model, model)}  (top-1 {row['top1_accuracy_percent']:.2f}%)")
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xticklabels(CLASSES, rotation=90, fontsize=7)
        ax.set_yticklabels(CLASSES, fontsize=7)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes[len(summary):]:
        ax.axis("off")
    fig.tight_layout()
    out = FIGURES / "confusion_matrices.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_params_vs_accuracy(summary: list[dict]) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    for row in summary:
        params_m = row["parameters"] / 1e6
        ax.scatter(params_m, row["top1_accuracy_percent"], s=70, color="#2b8cbe")
        ax.annotate(
            DISPLAY.get(row["model"], row["model"]),
            (params_m, row["top1_accuracy_percent"]),
            textcoords="offset points",
            xytext=(7, 5),
            fontsize=8,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (millions, log scale)")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title("Accuracy vs. model size")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIGURES / "params_vs_accuracy.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    summary = load_summary()
    for path in (plot_accuracy(summary), plot_confusions(summary), plot_params_vs_accuracy(summary)):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
