"""CIFAR-10 residual networks, with an optional Squeeze-and-Excitation branch.

The backbone follows He et al., "Deep Residual Learning for Image Recognition"
(CVPR 2016), CIFAR-10 variant: 3 stages of n residual blocks, widths 16/32/64,
global average pooling and a single 10-way linear classifier.  Depth is 6n+2.

The proposed variant (SE-ResNet) inserts a Squeeze-and-Excitation module
(Hu et al., CVPR 2018) after the second convolution of every residual block.

Both variants share every other detail, so a comparison between
`resnet20()` and `se_resnet20()` isolates the effect of the SE branch.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation: channel-wise re-weighting with a bottleneck MLP."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = x.shape
        weights = self.pool(x).view(batch, channels)
        weights = self.fc(weights).view(batch, channels, 1, 1)
        return x * weights


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        use_se: bool = False,
        reduction: int = 16,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SEBlock(planes, reduction) if use_se else nn.Identity()

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = out + self.shortcut(x)
        return torch.relu(out)


class CifarResNet(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        num_classes: int = 10,
        use_se: bool = False,
        reduction: int = 16,
    ) -> None:
        super().__init__()
        self.in_planes = 16
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(16, num_blocks, 1, use_se, reduction)
        self.layer2 = self._make_layer(32, num_blocks, 2, use_se, reduction)
        self.layer3 = self._make_layer(64, num_blocks, 2, use_se, reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _make_layer(
        self,
        planes: int,
        num_blocks: int,
        stride: int,
        use_se: bool,
        reduction: int,
    ) -> nn.Sequential:
        layers = []
        for block_index in range(num_blocks):
            layers.append(
                BasicBlock(
                    self.in_planes,
                    planes,
                    stride if block_index == 0 else 1,
                    use_se=use_se,
                    reduction=reduction,
                )
            )
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pool(out).flatten(1)
        return self.fc(out)


def resnet20(num_classes: int = 10) -> CifarResNet:
    return CifarResNet(3, num_classes=num_classes)


def resnet32(num_classes: int = 10) -> CifarResNet:
    return CifarResNet(5, num_classes=num_classes)


def resnet56(num_classes: int = 10) -> CifarResNet:
    return CifarResNet(9, num_classes=num_classes)


def se_resnet20(num_classes: int = 10, reduction: int = 16) -> CifarResNet:
    return CifarResNet(3, num_classes=num_classes, use_se=True, reduction=reduction)


MODELS = {
    "resnet20": resnet20,
    "resnet32": resnet32,
    "resnet56": resnet56,
    "se_resnet20": se_resnet20,
}
