import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from lt_ood.data import (
    IMBALANCECIFAR10,
    IMBALANCECIFAR100,
    SCOODDataset,
    build_transforms,
    ood_transform,
)
from lt_ood.models import ResNet18, CosineClassifier


def strip_module_prefix(state_dict):
    return {
        k.replace("module.", "", 1) if k.startswith("module.") else k: v
        for (k, v) in state_dict.items()
    }


def load_checkpoint(path: str, device: torch.device):
    return torch.load(path, map_location=device, weights_only=False)


def get_num_classes(dataset: str) -> int:
    return 10 if dataset == "cifar10" else 100


def build_id_loaders(
    dataset: str, data_root: str, imbalance_ratio: float, batch_size: int, num_workers: int
):
    (_, test_tfm) = build_transforms(dataset)
    ds_cls = IMBALANCECIFAR10 if dataset == "cifar10" else IMBALANCECIFAR100
    train_set = ds_cls(
        root=data_root, train=True, transform=test_tfm, imbalance_ratio=imbalance_ratio
    )
    test_set = ds_cls(root=data_root, train=False, transform=test_tfm, imbalance_ratio=1.0)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return (train_set, train_loader, test_loader, test_tfm)


def build_scood_loaders(dataset: str, scood_root: str, batch_size: int, num_workers: int):
    names = (
        ["cifar100", "svhn", "texture", "tin", "lsun"]
        if dataset == "cifar10"
        else ["cifar10", "svhn", "texture", "tin", "lsun"]
    )
    ood_eval_tfm = ood_transform(dataset)
    out = {}
    for name in names:
        ds = SCOODDataset(root=scood_root, id_name=dataset, ood_name=name, transform=ood_eval_tfm)
        out[name] = DataLoader(
            ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
    return out


def load_pcerm(dataset: str, ckpt_path: str, device: torch.device):
    state = load_checkpoint(ckpt_path, device)
    cfg = state.get("args", {})
    num_classes = get_num_classes(dataset)
    if (
        cfg.get("arch", "resnet18") != "resnet18"
        or cfg.get("classifier", "cosine") != "cosine"
        or int(cfg.get("feat_dim", 512)) != 512
    ):
        raise ValueError("Expected a ResNet-18 cosine checkpoint")
    backbone = ResNet18().to(device)
    classifier = CosineClassifier(num_classes, scale=float(cfg.get("scale", 30.0))).to(device)
    backbone.load_state_dict(strip_module_prefix(state["backbone"]), strict=True)
    classifier.load_state_dict(strip_module_prefix(state["classifier"]), strict=True)
    backbone.eval()
    classifier.eval()

    def forward_fn(x):
        features = backbone(x)
        logits = classifier(features)
        return (features, logits)

    weight = classifier.weight
    return {
        "forward_fn": forward_fn,
        "classifier_weight": weight.detach().cpu(),
        "meta": {"feat_dim": 512, "num_classes": num_classes, "family": "PCERM"},
    }


def collect_loader_outputs(loader: DataLoader, forward_fn, device: torch.device):
    (feats, logits, labels) = ([], [], [])
    with torch.no_grad():
        for images, target in loader:
            images = images.to(device, non_blocking=True)
            (features, logit) = forward_fn(images)
            feats.append(features.detach().cpu())
            logits.append(logit.detach().cpu())
            labels.append(target.detach().cpu())
    return {
        "features": torch.cat(feats, dim=0),
        "logits": torch.cat(logits, dim=0),
        "labels": torch.cat(labels, dim=0),
    }


def main():
    parser = argparse.ArgumentParser(description="PC-ERM feature extraction")
    parser.add_argument("--dataset", required=True, choices=["cifar10", "cifar100"])
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--scood_root", required=True)
    parser.add_argument("--imbalance_ratio", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    model_bundle = load_pcerm(args.dataset, args.ckpt, device)
    (_, train_loader, test_loader, _) = build_id_loaders(
        args.dataset, args.data_root, args.imbalance_ratio, args.batch_size, args.num_workers
    )
    scood_loaders = build_scood_loaders(
        args.dataset, args.scood_root, args.batch_size, args.num_workers
    )
    bank = {
        "config": {
            "model_family": "PCERM",
            "dataset": args.dataset,
            "ckpt": args.ckpt,
            "imbalance_ratio": args.imbalance_ratio,
        },
        "meta": model_bundle["meta"],
        "classifier_weight": model_bundle["classifier_weight"],
        "train": collect_loader_outputs(train_loader, model_bundle["forward_fn"], device),
        "test": collect_loader_outputs(test_loader, model_bundle["forward_fn"], device),
        "scood": {},
    }
    for name, loader in scood_loaders.items():
        bank["scood"][name] = collect_loader_outputs(loader, model_bundle["forward_fn"], device)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank, out_path)
    print(out_path)


if __name__ == "__main__":
    main()
