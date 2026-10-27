import ast
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision import transforms


class Cutout:
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, image):
        h, w = image.size(1), image.size(2)
        mask = np.ones((h, w), np.float32)
        for _ in range(self.n_holes):
            y, x = np.random.randint(h), np.random.randint(w)
            y1, y2 = np.clip(y - self.length // 2, 0, h), np.clip(y + self.length // 2, 0, h)
            x1, x2 = np.clip(x - self.length // 2, 0, w), np.clip(x + self.length // 2, 0, w)
            mask[y1:y2, x1:x2] = 0.0
        return image * torch.from_numpy(mask).to(image.device).expand_as(image)


class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, root, train=True, transform=None, imbalance_ratio=0.01, download=True):
        super().__init__(root, train=train, transform=transform, download=download)
        if train:
            maximum = len(self.data) / self.cls_num
            self.counts = [
                int(maximum * imbalance_ratio ** (c / (self.cls_num - 1)))
                for c in range(self.cls_num)
            ]
            labels = np.asarray(self.targets)
            indices = np.concatenate(
                [np.flatnonzero(labels == c)[:n] for c, n in enumerate(self.counts)]
            )
            self.data = self.data[indices]
            self.targets = labels[indices].tolist()

    def get_cls_num_list(self):
        return self.counts


class IMBALANCECIFAR100(IMBALANCECIFAR10):
    cls_num = 100
    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [["train", "16019d7e3df5f24257cddd939b257f8d"]]
    test_list = [["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"]]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }


def normalization(dataset):
    if dataset == "cifar10":
        return (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    return (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)


def build_transforms(dataset):
    mean, std = normalization(dataset)
    steps = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    if hasattr(transforms, "AutoAugment") and hasattr(transforms, "AutoAugmentPolicy"):
        steps.append(transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10))
    steps.extend([transforms.ToTensor(), Cutout(), transforms.Normalize(mean, std)])
    test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return transforms.Compose(steps), test


def ood_transform(dataset):
    mean, std = normalization(dataset)
    return transforms.Compose(
        [
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


class SCOODDataset(torch.utils.data.Dataset):
    def __init__(self, root, id_name, ood_name, transform):
        self.root = Path(root)
        self.transform = transform
        path = self.root / "imglist" / f"benchmark_{id_name}" / f"test_{ood_name}.txt"
        self.lines = path.read_text().splitlines()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        path, metadata = self.lines[index].split(" ", 1)
        metadata = ast.literal_eval(metadata)
        label = metadata if isinstance(metadata, int) else metadata["sc_label"]
        with Image.open(self.root / path) as image:
            image = image.convert("RGB")
        return self.transform(image), label
