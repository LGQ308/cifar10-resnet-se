"""Train a CIFAR-10 residual network from scratch.

Usage
-----
    python train.py --model resnet20    --epochs 200 --out runs/resnet20
    python train.py --model se_resnet20 --epochs 200 --out runs/se_resnet20

Everything that influences the result is written to `<out>/train_config.json`
(seed, hyper-parameters, versions, GPU model) and every epoch is appended to
`<out>/metrics.csv`, so a reported number can always be traced back to a run.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from se_resnet import MODELS

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CIFAR-10 residual networks.")
    parser.add_argument("--model", choices=sorted(MODELS), default="resnet20")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--scheduler",
        choices=("cosine", "multistep"),
        default="cosine",
        help="cosine: single cosine decay; multistep: x0.1 at 50%%/75%% of epochs.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def build_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    train_set = datasets.CIFAR10(
        root=args.data_dir, train=True, download=True, transform=train_transform
    )
    test_set = datasets.CIFAR10(
        root=args.data_dir, train=False, download=True, transform=test_transform
    )
    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin,
        drop_last=True,
        persistent_workers=args.workers > 0,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin,
        persistent_workers=args.workers > 0,
    )
    return train_loader, test_loader


def build_scheduler(optimizer, args: argparse.Namespace):
    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    milestones = [int(args.epochs * 0.5), int(args.epochs * 0.75)]
    return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss_sum += criterion(logits, targets).item() * targets.size(0)
        correct += logits.argmax(1).eq(targets).sum().item()
        total += targets.size(0)
    return loss_sum / total, 100.0 * correct / total


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = build_loaders(args)

    model = MODELS[args.model]().to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = build_scheduler(optimizer, args)

    config = {
        "model": args.model,
        "parameters": parameter_count,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "scheduler": args.scheduler,
        "seed": args.seed,
        "device": str(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python": sys.version,
        "platform": platform.platform(),
        "train_size": len(train_loader.dataset),
        "test_size": len(test_loader.dataset),
        "augmentation": "RandomCrop(32,padding=4) + RandomHorizontalFlip",
    }
    (args.out / "train_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)

    metrics_path = args.out / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["epoch", "lr", "train_loss", "train_acc", "test_loss", "test_acc", "seconds"]
        )

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.perf_counter()
        loss_sum = 0.0
        correct = 0
        total = 0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * targets.size(0)
            correct += logits.argmax(1).eq(targets).sum().item()
            total += targets.size(0)

        train_loss = loss_sum / total
        train_acc = 100.0 * correct / total
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        elapsed = time.perf_counter() - start
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        with metrics_path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(
                [epoch, current_lr, train_loss, train_acc, test_loss, test_acc, elapsed]
            )

        marker = ""
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                {
                    "model": args.model,
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "test_acc": test_acc,
                    "config": config,
                },
                args.out / "best.pt",
            )
            marker = "  *"
        print(
            f"epoch {epoch:3d}/{args.epochs}  lr={current_lr:.4f}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:5.2f}  "
            f"test_loss={test_loss:.4f}  test_acc={test_acc:5.2f}  "
            f"({elapsed:4.1f}s){marker}",
            flush=True,
        )

    torch.save(
        {
            "model": args.model,
            "epoch": args.epochs,
            "state_dict": model.state_dict(),
            "test_acc": test_acc,
            "config": config,
        },
        args.out / "last.pt",
    )
    print(f"\nfinished. best test accuracy = {best_acc:.2f}%", flush=True)


if __name__ == "__main__":
    main()
