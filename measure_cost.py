"""Measure parameters, size, latency and FLOPs for all seven models consistently.

Pretrained checkpoints are loaded through torch.hub, the two trained models from
runs/.  Every entry uses the same procedure: warm up, synchronise, average over
repeated single-image and 32-image forward passes.

    python measure_cost.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from se_resnet import MODELS

ROOT = Path(r"E:\codexwork\cifar10-local-experiment")
RESULTS = ROOT / "results"
PRETRAINED = [
    ("ResNet-20", "cifar10_resnet20", "public"),
    ("ResNet-32", "cifar10_resnet32", "public"),
    ("ResNet-56", "cifar10_resnet56", "public"),
    ("VGG-16-BN", "cifar10_vgg16_bn", "public"),
    ("MobileNetV2", "cifar10_mobilenetv2_x1_0", "public"),
]
TRAINED = [
    ("ResNet-20（自训练）", "resnet20", ROOT / "runs" / "resnet20_200" / "best.pt"),
    ("SE-ResNet-20（自训练）", "se_resnet20", ROOT / "runs" / "se_resnet20_200" / "best.pt"),
]


@torch.no_grad()
def latency_ms(model, device, batch_size, warmup=30, iters=100):
    sample = torch.randn(batch_size, 3, 32, 32, device=device)
    for _ in range(warmup):
        model(sample)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        model(sample)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000.0


def flops_m(model, device):
    from torch.utils.flop_counter import FlopCounterMode

    sample = torch.randn(1, 3, 32, 32, device=device)
    counter = FlopCounterMode(display=False)
    with counter, torch.no_grad():
        model(sample)
    return float(counter.get_total_flops()) / 1e6


def describe(name, model, device, size_mb):
    parameters = sum(p.numel() for p in model.parameters())
    lat1 = latency_ms(model, device, 1)
    lat32 = latency_ms(model, device, 32)
    return {
        "name": name,
        "parameters": parameters,
        "size_mb": round(size_mb, 2),
        "latency_ms_batch1": round(lat1, 3),
        "latency_ms_batch32": round(lat32, 3),
        "throughput_img_per_s": round(32000.0 / lat32, 1),
        "mflops": round(flops_m(model, device), 2),
    }


def main() -> None:
    device = torch.device("cuda")
    torch.hub.set_dir(str(Path.home() / ".cache" / "torch" / "hub"))
    records = []

    for label, hub_name, _kind in PRETRAINED:
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", hub_name,
                               pretrained=True, trust_repo=True).to(device).eval()
        cache_file = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / f"{hub_name}-*.pt"
        candidates = list(cache_file.parent.glob(cache_file.name))
        size_mb = candidates[0].stat().st_size / 1e6 if candidates else 0.0
        records.append(describe(label, model, device, size_mb))
        print(records[-1], flush=True)
        del model
        torch.cuda.empty_cache()

    for label, key, checkpoint in TRAINED:
        blob = torch.load(checkpoint, map_location=device, weights_only=False)
        model = MODELS[key]().to(device)
        model.load_state_dict(blob["state_dict"])
        model.eval()
        records.append(describe(label, model, device, checkpoint.stat().st_size / 1e6))
        print(records[-1], flush=True)
        del model
        torch.cuda.empty_cache()

    (RESULTS / "cost.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {RESULTS / 'cost.json'}")


if __name__ == "__main__":
    main()
