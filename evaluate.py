from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate pytorch-cifar-models on the full CIFAR-10 test set."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["resnet20", "resnet32", "resnet56"],
        help="Model names from chenyaofo/pytorch-cifar-models.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    return parser.parse_args()


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def select_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access the GPU.")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def build_loader(args: argparse.Namespace, device: torch.device) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010),
            ),
        ]
    )
    dataset = datasets.CIFAR10(
        root=args.data_dir,
        train=False,
        download=True,
        transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate_model(
    model_name: str,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, object], np.ndarray, list[dict[str, object]]]:
    hub_name = f"cifar10_{model_name}"
    print(f"\nLoading {hub_name} ...", flush=True)
    model = torch.hub.load(
        "chenyaofo/pytorch-cifar-models",
        hub_name,
        pretrained=True,
        trust_repo=True,
    )
    model = model.to(device).eval()

    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    total = 0
    top1_correct = 0
    top5_correct = 0

    synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            predictions = logits.topk(5, dim=1).indices

            total += targets.size(0)
            top1_correct += predictions[:, 0].eq(targets).sum().item()
            top5_correct += predictions.eq(targets[:, None]).any(dim=1).sum().item()

            target_values = targets.cpu().numpy()
            prediction_values = predictions[:, 0].cpu().numpy()
            np.add.at(confusion, (target_values, prediction_values), 1)

    synchronize(device)
    elapsed_seconds = time.perf_counter() - start
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    per_class = []
    for index, class_name in enumerate(CLASS_NAMES):
        class_total = int(confusion[index].sum())
        class_correct = int(confusion[index, index])
        per_class.append(
            {
                "model": model_name,
                "class": class_name,
                "correct": class_correct,
                "total": class_total,
                "accuracy_percent": 100.0 * class_correct / class_total,
            }
        )

    result: dict[str, object] = {
        "model": model_name,
        "samples": total,
        "top1_accuracy_percent": 100.0 * top1_correct / total,
        "top5_accuracy_percent": 100.0 * top5_correct / total,
        "parameters": parameter_count,
        "elapsed_seconds": elapsed_seconds,
        "images_per_second": total / elapsed_seconds,
        "device": str(device),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, confusion, per_class


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_reproducibility(args.seed)
    device = select_device(args.device)

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seed": args.seed,
        "batch_size": args.batch_size,
    }
    print(json.dumps(environment, ensure_ascii=False, indent=2), flush=True)
    (args.output_dir / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    loader = build_loader(args, device)
    results: list[dict[str, object]] = []
    per_class_results: list[dict[str, object]] = []

    for model_name in args.models:
        result, confusion, per_class = evaluate_model(model_name, loader, device)
        results.append(result)
        per_class_results.extend(per_class)
        np.savetxt(
            args.output_dir / f"confusion_matrix_{model_name}.csv",
            confusion,
            delimiter=",",
            fmt="%d",
            header=",".join(CLASS_NAMES),
            comments="",
        )

    write_csv(args.output_dir / "summary.csv", results)
    write_csv(args.output_dir / "per_class_accuracy.csv", per_class_results)
    (args.output_dir / "summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nResults saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
