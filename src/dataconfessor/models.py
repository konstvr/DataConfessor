"""Proxy architectures: resnet18, smallcnn below 64 px, resnet1d18 for signals."""

from __future__ import annotations

import torch
import torch.nn as nn

SMALL_INPUT_MAX = 64
"""Side length at or below which the ImageNet stem is wrong: its stride-2 conv and max-pool
reduce a 28 px input below one pixel before the first residual block."""


def choose_arch(kind: str, input_size: int | None) -> str:
    if kind == "signal":
        return "resnet1d18"
    if input_size is not None and input_size <= SMALL_INPUT_MAX:
        return "smallcnn"
    return "resnet18"


def build(
    arch: str,
    *,
    num_classes: int,
    in_channels: int,
    pretrained: bool = False,
) -> nn.Module:
    if arch == "resnet18":
        return _resnet18(num_classes, in_channels, pretrained)
    if arch == "smallcnn":
        return SmallCNN(num_classes, in_channels, input_type="image")
    if arch == "resnet1d18":
        return ResNet1d(num_classes, in_channels)
    if arch == "smallcnn1d":
        return SmallCNN(num_classes, in_channels, input_type="signal")
    raise ValueError(f"unknown arch {arch!r}; have resnet18, smallcnn, resnet1d18, smallcnn1d")


def _resnet18(num_classes: int, in_channels: int, pretrained: bool) -> nn.Module:
    from torchvision.models import ResNet18_Weights, resnet18

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    if in_channels != 3:
        old = model.conv1
        new = nn.Conv2d(in_channels, old.out_channels, kernel_size=old.kernel_size,
                        stride=old.stride, padding=old.padding, bias=False)
        if pretrained:
            # Sum the pretrained RGB filters rather than averaging them: feeding a grayscale
            # image to the original conv by replicating it across RGB gives the sum of the
            # three responses, so summing reproduces that exactly. Averaging would attenuate
            # every activation threefold and throw away part of what pretraining bought.
            with torch.no_grad():
                summed = old.weight.sum(dim=1, keepdim=True)
                new.weight.copy_(summed.repeat(1, in_channels, 1, 1) / in_channels
                                 if in_channels > 1 else summed)
        model.conv1 = new
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class SmallCNN(nn.Module):
    """Three stride-2 blocks, 16/32/64 channels, ~25k parameters.

    Downsampling is by strided convolution rather than pooling, which is defined at every
    resolution -- pooling fails once the feature map reaches 1x1, which a 28 px input does.
    """

    def __init__(self, num_classes: int, in_channels: int, input_type: str = "image"):
        super().__init__()
        conv = nn.Conv2d if input_type == "image" else nn.Conv1d
        norm = nn.BatchNorm2d if input_type == "image" else nn.BatchNorm1d
        self.gap = (nn.AdaptiveAvgPool2d((1, 1)) if input_type == "image"
                    else nn.AdaptiveAvgPool1d(1))
        layers: list[nn.Module] = []
        c = in_channels
        for w in (16, 32, 64):
            layers += [conv(c, w, 3, stride=2, padding=1, bias=False), norm(w), nn.ReLU()]
            c = w
        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.gap(self.features(x)).flatten(1))


class BasicBlock1d(nn.Module):
    """Kernel 15 at 100 Hz is ~150 ms, roughly one QRS complex."""

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, 15, stride=stride, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, 15, stride=1, padding=7, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm1d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out)) + self.shortcut(x)
        return self.relu(out)


class ResNet1d(nn.Module):
    def __init__(self, num_classes: int, in_channels: int, width: int = 64):
        super().__init__()
        self.in_planes = width
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, width, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(width), nn.ReLU(), nn.MaxPool1d(3, stride=2, padding=1),
        )
        self.layers = nn.Sequential(
            self._stage(width, 2, 1), self._stage(width * 2, 2, 2),
            self._stage(width * 4, 2, 2), self._stage(width * 8, 2, 2),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(width * 8, num_classes)

    def _stage(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        seq = []
        for s in [stride] + [1] * (blocks - 1):
            seq.append(BasicBlock1d(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.gap(self.layers(self.stem(x))).flatten(1))
