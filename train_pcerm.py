import argparse
import json
import platform
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from lt_ood.data import IMBALANCECIFAR10, IMBALANCECIFAR100, build_transforms
from lt_ood.models import ResNet18, CosineClassifier


def parse_args():
    parser = argparse.ArgumentParser("PC-ERM training")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--imbalance_ratio", type=float, default=0.01)
    parser.add_argument("--scale", type=float, default=30.0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--wd", type=float, default=0.0005)
    parser.add_argument("--lr_schedule", choices=["cosine", "step", "none"], default="cosine")
    parser.add_argument("--step_milestones", default="160,180")
    parser.add_argument("--logit_adjust_tau", type=float, default=1.0)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def build_model(args, num_classes, device):
    backbone = ResNet18().to(device)
    classifier = CosineClassifier(num_classes, scale=args.scale).to(device)
    return (backbone, classifier)


def build_optimizer(args, backbone, classifier):
    params = list(backbone.parameters()) + list(classifier.parameters())
    if args.optimizer == "sgd":
        return torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.wd)
    return torch.optim.Adam(params, lr=args.lr, weight_decay=args.wd)


def build_scheduler(args, optimizer):
    if args.lr_schedule == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    if args.lr_schedule == "step":
        milestones = [int(x) for x in args.step_milestones.split(",") if x.strip()]
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)


@torch.no_grad()
def evaluate(backbone, classifier, loader, device):
    backbone.eval()
    classifier.eval()
    correct = 0
    total = 0
    energy_sum = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = classifier(backbone(x))
        correct += int((logits.argmax(1) == y).sum().item())
        total += int(y.numel())
        energy_sum += float((-torch.logsumexp(logits, dim=1)).sum().item())
    return {"overall": correct / max(total, 1), "mean_energy_id": energy_sum / max(total, 1)}


def write_training_time(path, device, started_at, epochs, start_epoch):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started_at
    cpu = platform.processor() or platform.machine()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        cpu = next(
            (
                line.split(":", 1)[1].strip()
                for line in cpuinfo.read_text().splitlines()
                if line.startswith("model name")
            ),
            cpu,
        )
    timing = {
        "total_seconds": elapsed,
        "start_epoch": start_epoch,
        "completed_epochs": epochs - start_epoch,
        "scope": "training, validation, logs, checkpoint saves",
        "device": str(device),
        "cpu": cpu,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "gpu_memory_bytes": (
            torch.cuda.get_device_properties(device).total_memory
            if device.type == "cuda"
            else None
        ),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
    }
    if start_epoch == 0 and epochs == 200:
        timing["cost_seconds_per_epoch"] = elapsed / 200
    path.write_text(json.dumps(timing, indent=2))
    return timing


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.out)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.jsonl"
    (train_tf, test_tf) = build_transforms(args.dataset)
    dataset_cls = IMBALANCECIFAR10 if args.dataset == "cifar10" else IMBALANCECIFAR100
    train_set = dataset_cls(
        root=args.data_root, train=True, transform=train_tf, imbalance_ratio=args.imbalance_ratio
    )
    test_set = dataset_cls(
        root=args.data_root, train=False, transform=test_tf, imbalance_ratio=args.imbalance_ratio
    )
    cls_num_list = train_set.get_cls_num_list()
    num_classes = train_set.cls_num
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=256, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    (backbone, classifier) = build_model(args, num_classes, device)
    optimizer = build_optimizer(args, backbone, classifier)
    scheduler = build_scheduler(args, optimizer)
    scaler = GradScaler(enabled=args.amp)
    counts = torch.tensor(cls_num_list, dtype=torch.float32, device=device)
    log_prior = torch.log((counts / counts.sum()).clamp_min(1e-12))
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    best_acc = 0.0
    last_path = run_dir / "last_model.pth"
    start_epoch = 0
    if last_path.exists():
        payload = torch.load(last_path, map_location=device, weights_only=False)
        backbone.load_state_dict(payload["backbone"])
        classifier.load_state_dict(payload["classifier"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        start_epoch = int(payload.get("epoch", -1)) + 1
        best_acc = float(payload.get("best_acc", 0.0))
        print(json.dumps({"resume_from_epoch": start_epoch - 1, "best_acc": best_acc}), flush=True)
    if start_epoch >= args.epochs:
        return
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started_at = time.perf_counter()
    for epoch in range(start_epoch, args.epochs):
        backbone.train()
        classifier.train()
        loss_sum = 0.0
        correct = 0
        total = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=args.amp):
                logits = classifier(backbone(x))
                adjusted = logits + args.logit_adjust_tau * log_prior.view(1, -1)
                loss = F.cross_entropy(adjusted, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(backbone.parameters()) + list(classifier.parameters()), 5.0
            )
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach().item()) * y.numel()
            correct += int((logits.detach().argmax(1) == y).sum().item())
            total += int(y.numel())
        scheduler.step()
        val = evaluate(backbone, classifier, test_loader, device)
        best_acc = max(best_acc, val["overall"])
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(total, 1),
            "train_acc1": correct / max(total, 1),
            "overall": val["overall"],
            "mean_energy_id": val["mean_energy_id"],
            "best_acc": best_acc,
            "lr": scheduler.get_last_lr()[0],
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        ckpt = {
            "epoch": epoch,
            "args": vars(args),
            "backbone": backbone.state_dict(),
            "classifier": classifier.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "cls_num_list": cls_num_list,
            "metrics": row,
            "best_acc": best_acc,
        }
        torch.save(ckpt, run_dir / "last_model.pth")
        if args.save_every > 0 and ((epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs):
            torch.save(ckpt, run_dir / f"epoch_{epoch + 1:03d}.pth")
        if val["overall"] >= best_acc:
            torch.save(ckpt, run_dir / "best_model.pth")

    write_training_time(run_dir / "timing.json", device, started_at, args.epochs, start_epoch)


if __name__ == "__main__":
    main()
