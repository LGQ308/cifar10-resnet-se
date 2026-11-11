"""Evaluate trained checkpoints: accuracy, cost and per-class behaviour.

For every checkpoint this script records
  * top-1 / top-5 accuracy on the full 10,000-image test set,
  * parameter count, checkpoint size and multiply-adds (FLOPs) for a 32x32 input,
  * single-image latency and batched throughput on the GPU,
  * the 10x10 confusion matrix and per-class accuracy.

Results are written as JSON/CSV next to the checkpoint so that every number in
the paper can be regenerated from the artefacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode
from torchvision import datasets, transforms

from se_resnet import MODELS

CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)


def load_model(checkpoint_path: Path, device: torch.device):
    blob = torch.load(checkpoint_path, map_location=device, weights_only=False)
    name = blob["model"]
    model = MODELS[name]().to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, name, blob


@torch.no_grad()
def measure_accuracy(model, loader, device):
    confusion = np.zeros((10, 10), dtype=np.int64)
    top1 = top5 = total = 0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        ranked = logits.topk(5, dim=1).indices
        top1 += ranked[:, 0].eq(targets).sum().item()
        top5 += ranked.eq(targets[:, None]).any(dim=1).sum().item()
        total += targets.size(0)
        np.add.at(confusion, (targets.cpu().numpy(), ranked[:, 0].cpu().numpy()), 1)
    return 100.0 * top1 / total, 100.0 * top5 / total, confusion


@torch.no_grad()
def measure_latency(model, device, batch_size: int, warmup: int = 20, iters: int = 100):
    dummy = torch.randn(batch_size, 3, 32, 32, device=device)
    for _ in range(warmup):
        model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / iters * 1000.0  # ms per batch


def measure_flops(model, device) -> float:
    dummy = torch.randn(1, 3, 32, 32, device=device)
    counter = FlopCounterMode(display=False)
    with counter, torch.no_grad():
        model(dummy)
    return float(counter.get_total_flops()) / 1e6  # MFLOPs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, name, blob = load_model(args.checkpoint, device)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    test_set = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)
    loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    top1, top5, confusion = measure_accuracy(model, loader, device)
    parameters = sum(p.numel() for p in model.parameters())
    latency_1 = measure_latency(model, device, 1, warmup=50, iters=200)
    latency_32 = measure_latency(model, device, 32, warmup=20, iters=50)
    flops = measure_flops(model, device)

    record = {
        "model": name,
        "checkpoint": str(args.checkpoint),
        "checkpoint_mb": round(args.checkpoint.stat().st_size / 1e6, 2),
        "parameters": parameters,
        "mflops_32x32": round(flops, 2),
        "top1_accuracy_percent": round(top1, 2),
        "top5_accuracy_percent": round(top5, 2),
        "latency_ms_batch1": round(latency_1, 3),
        "latency_ms_batch32": round(latency_32, 3),
        "throughput_img_per_s": round(32 * 1000.0 / latency_32, 1),
        "train_epochs": blob.get("epoch"),
        "saved_test_acc": round(float(blob.get("test_acc", float("nan"))), 2),
        "device": str(device),
    }

    stem = f"{name}_trained"
    (args.out_dir / f"{stem}_metrics.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savetxt(
        args.out_dir / f"{stem}_confusion.csv",
        confusion,
        delimiter=",",
        fmt="%d",
        header=",".join(CLASSES),
        comments="",
    )
    with (args.out_dir / f"{stem}_per_class.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "correct", "total", "accuracy_percent"])
        for index, class_name in enumerate(CLASSES):
            total = int(confusion[index].sum())
            correct = int(confusion[index, index])
            writer.writerow([class_name, correct, total, round(100.0 * correct / total, 2)])

    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
