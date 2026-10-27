import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out, inplace=True)


class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 1)
        self.layer2 = self._make_layer(128, 2)
        self.layer3 = self._make_layer(256, 2)
        self.layer4 = self._make_layer(512, 2)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(self, planes, stride):
        layers = []
        for step in (stride, 1):
            layers.append(BasicBlock(self.in_planes, planes, step))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward_features(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return F.adaptive_avg_pool2d(out, 1).flatten(1)

    def forward(self, x):
        return self.forward_features(x)


class CosineClassifier(nn.Module):
    def __init__(self, num_classes, scale=30.0):
        super().__init__()
        self.weight = nn.Parameter(F.normalize(torch.randn(num_classes, 512), p=2, dim=1))
        self.log_scale = nn.Parameter(torch.tensor(math.log(scale)))

    def forward(self, features):
        features = F.normalize(features, p=2, dim=-1)
        weights = F.normalize(self.weight, p=2, dim=-1)
        return self.log_scale.exp() * (features @ weights.T)
