"""Plot training curves from runs/<tag>/metrics.csv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LABELS = {"resnet20": "ResNet-20 (baseline)", "se_resnet20": "SE-ResNet-20 (ours)"}
COLORS = {"resnet20": "#2b6cb0", "se_resnet20": "#c53030"}


def read_metrics(path: Path) -> dict[str, list[float]]:
    columns: dict[str, list[float]] = {}
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for key, value in row.items():
                columns.setdefault(key, []).append(float(value))
    return columns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True,
                        help="tag=path/to/metrics.csv")
    parser.add_argument("--out", type=Path, default=Path("results/figures"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    runs = {}
    for spec in args.run:
        tag, path = spec.split("=", 1)
        runs[tag] = read_metrics(Path(path))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=200)
    for tag, metrics in runs.items():
        epochs = metrics["epoch"]
        axes[0].plot(epochs, metrics["train_loss"], color=COLORS.get(tag, "C0"),
                     linestyle="--", linewidth=1.2, label=f"{LABELS.get(tag, tag)} train")
        axes[0].plot(epochs, metrics["test_loss"], color=COLORS.get(tag, "C0"),
                     linewidth=1.6, label=f"{LABELS.get(tag, tag)} test")
        axes[1].plot(epochs, metrics["train_acc"], color=COLORS.get(tag, "C0"),
                     linestyle="--", linewidth=1.2, label=f"{LABELS.get(tag, tag)} train")
        axes[1].plot(epochs, metrics["test_acc"], color=COLORS.get(tag, "C0"),
                     linewidth=1.6, label=f"{LABELS.get(tag, tag)} test")

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=7, loc="lower right")

    fig.tight_layout()
    out = args.out / "training_curves.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
