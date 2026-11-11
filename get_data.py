"""Download CIFAR-10 with a China-friendly mirror and verify its MD5.

`torchvision.datasets.CIFAR10(download=True)` pulls from cs.toronto.edu, which is
frequently reset on mainland networks (the transfer stalls after a few MB).
This script fetches the same archive from a ModelScope mirror, checks the
official MD5, and drops it into ./data/ so that torchvision only has to extract
it.

Usage
-----
    python get_data.py                 # download to ./data
    python get_data.py --force         # re-download even if present
    python get_data.py --no-verify     # skip the MD5 check (not recommended)
"""

from __future__ import annotations

import argparse
import hashlib
import time
import urllib.request
from pathlib import Path

DEFAULT_MIRROR = (
    "https://www.modelscope.cn/api/v1/datasets/star07/cifar10/repo"
    "?Revision=master&FilePath=cifar-10-python.tar"
)
OFFICIAL_MD5 = "c58f30108f718f92721af3b95e74349a"


def md5_of(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path) -> str:
    digest = hashlib.md5()
    total = 0
    start = time.time()
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response, target.open("wb") as sink:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            sink.write(chunk)
            digest.update(chunk)
            total += len(chunk)
            if total % (20 << 20) < (1 << 20):
                speed = total / 1e6 / max(time.time() - start, 1e-6)
                print(f"  {total / 1e6:6.1f} MB  {speed:4.1f} MB/s", flush=True)
    print(f"downloaded {total} bytes")
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--mirror", default=DEFAULT_MIRROR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)
    target = args.data_dir / "cifar-10-python.tar.gz"

    if target.exists() and not args.force:
        if args.no_verify or md5_of(target) == OFFICIAL_MD5:
            print(f"already present: {target}")
            return
        print("existing archive has a wrong MD5, re-downloading ...")

    if target.exists():
        target.unlink()

    print(f"downloading from {args.mirror}")
    actual = download(args.mirror, target)

    if not args.no_verify and actual != OFFICIAL_MD5:
        target.unlink(missing_ok=True)
        raise SystemExit(f"MD5 mismatch: expected {OFFICIAL_MD5}, got {actual}")

    print(f"OK  {target}  md5={actual}")
    print("torchvision will extract it automatically on first use.")


if __name__ == "__main__":
    main()
