import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


def _finite_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    return scores[np.isfinite(scores)]


def compute_auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])
    return float(roc_auc_score(labels, scores))


def compute_fpr95(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    threshold = np.percentile(ood_scores, 5)
    return float(np.mean(id_scores >= threshold))


def summarize_scores(id_scores: np.ndarray, ood_scores: np.ndarray):
    id_scores = _finite_scores(id_scores)
    ood_scores = _finite_scores(ood_scores)
    if len(id_scores) == 0 or len(ood_scores) == 0:
        return {"AUROC": float("nan"), "FPR95": float("nan")}
    return {
        "AUROC": compute_auroc(id_scores, ood_scores),
        "FPR95": compute_fpr95(id_scores, ood_scores),
    }


def split_id_ood(id_scores: np.ndarray, scood_scores: np.ndarray, sc_labels: np.ndarray):
    fake_ood = scood_scores[sc_labels >= 0]
    real_ood = scood_scores[sc_labels < 0]
    id_pool = np.concatenate([id_scores, fake_ood], axis=0)
    return (id_pool, real_ood)


def energy_scores(logits: torch.Tensor, temperature: float = 1.0) -> np.ndarray:
    return (-(temperature * torch.logsumexp(logits.float() / temperature, dim=1))).cpu().numpy()


def msp_scores(logits: torch.Tensor) -> np.ndarray:
    return (-F.softmax(logits.float(), dim=1).max(dim=1).values).cpu().numpy()


def maybe_l2(features: torch.Tensor, enable: bool) -> torch.Tensor:
    return F.normalize(features.float(), dim=1) if enable else features.float()


def pooled_covariance_config(num_samples: int, num_classes: int, reg: float):
    if num_classes < 1 or num_samples <= num_classes:
        raise ValueError("Pooled covariance requires N > K >= 1")
    if not np.isfinite(reg) or reg < 0:
        raise ValueError("The reference ridge must be finite and nonnegative")
    scale = num_samples / (num_samples - num_classes)
    return {
        "denominator": "N-K",
        "num_samples": num_samples,
        "num_classes": num_classes,
        "degrees_of_freedom": num_samples - num_classes,
        "reference_reg": reg,
        "effective_reg": scale * reg,
        "scale_from_sample_average": scale,
    }


def fit_shared_maha(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    num_classes: int,
    reg: float = 1e-05,
    fit_device: torch.device = None,
):
    if fit_device is not None and fit_device.type != "cpu":
        train_features = train_features.to(fit_device)
        train_labels = train_labels.to(fit_device)
    feat_dim = train_features.shape[1]
    means = []
    centered_chunks = []
    for cls in range(num_classes):
        cls_feat = train_features[train_labels == cls]
        means.append(cls_feat.mean(dim=0))
        centered_chunks.append(cls_feat - cls_feat.mean(dim=0, keepdim=True))
    means = torch.stack(means, dim=0)
    centered = torch.cat(centered_chunks, dim=0)
    convention = pooled_covariance_config(centered.shape[0], num_classes, reg)
    cov = centered.t().mm(centered) / convention["degrees_of_freedom"]
    cov = cov + convention["effective_reg"] * torch.eye(feat_dim, device=cov.device)
    precision = torch.linalg.pinv(cov)
    return (means.cpu(), precision.cpu())


def fit_per_class_maha(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    num_classes: int,
    reg: float = 1e-05,
    fit_device: torch.device = None,
):
    if fit_device is not None and fit_device.type != "cpu":
        train_features = train_features.to(fit_device)
        train_labels = train_labels.to(fit_device)
    (means, precisions) = ([], [])
    feat_dim = train_features.shape[1]
    eye = reg * torch.eye(feat_dim, device=train_features.device)
    for cls in range(num_classes):
        cls_feat = train_features[train_labels == cls]
        mean = cls_feat.mean(dim=0)
        diff = cls_feat - mean.unsqueeze(0)
        cov = diff.t().mm(diff) / max(cls_feat.shape[0], 1)
        cov = cov + eye
        means.append(mean)
        precisions.append(torch.linalg.pinv(cov))
    return (torch.stack(means, dim=0).cpu(), torch.stack(precisions, dim=0).cpu())


def score_shared_maha_chunked(
    features: torch.Tensor,
    means: torch.Tensor,
    precision: torch.Tensor,
    device: torch.device,
    chunk_size: int,
) -> np.ndarray:
    outs = []
    means = means.to(device)
    precision = precision.to(device)
    for start in range(0, features.shape[0], chunk_size):
        chunk = features[start : start + chunk_size].to(device)
        diff = chunk.unsqueeze(1) - means.unsqueeze(0)
        proj = torch.matmul(diff, precision)
        dists = (proj * diff).sum(dim=2)
        outs.append(dists.min(dim=1).values.cpu())
    return torch.cat(outs, dim=0).numpy()


def score_per_class_maha_chunked(
    features: torch.Tensor,
    means: torch.Tensor,
    precisions: torch.Tensor,
    device: torch.device,
    chunk_size: int,
) -> np.ndarray:
    outs = []
    means = means.to(device)
    precisions = precisions.to(device)
    for start in range(0, features.shape[0], chunk_size):
        chunk = features[start : start + chunk_size].to(device)
        diff = chunk.unsqueeze(1) - means.unsqueeze(0)
        proj = torch.einsum("qcd,cde->qce", diff, precisions)
        dists = (proj * diff).sum(dim=2)
        outs.append(dists.min(dim=1).values.cpu())
    return torch.cat(outs, dim=0).numpy()


def evaluate_from_bank(bank, temperature, reg, device, chunk_size):
    num_classes = int(bank["meta"]["num_classes"])
    train_features = bank["train"]["features"].float()
    train_labels = bank["train"]["labels"].long()
    test_features = bank["test"]["features"].float()
    test_logits = bank["test"]["logits"].float()
    fit_device = device if device.type != "cpu" else None
    specifications = (
        ("maha_tied_raw", False, True),
        ("maha_tied_l2", True, True),
        ("maha_pc_raw", False, False),
        ("maha_pc_l2", True, False),
    )
    fitted = {}
    id_scores = {"energy": energy_scores(test_logits, temperature), "msp": msp_scores(test_logits)}
    for name, normalized, shared in specifications:
        fit = fit_shared_maha if shared else fit_per_class_maha
        score = score_shared_maha_chunked if shared else score_per_class_maha_chunked
        fitted[name] = fit(
            maybe_l2(train_features, normalized),
            train_labels,
            num_classes,
            reg=reg,
            fit_device=fit_device,
        )
        id_scores[name] = score(
            maybe_l2(test_features, normalized), *fitted[name], device=device, chunk_size=chunk_size
        )
    results = {name: {"per_dataset": {}, "avg": {}} for name in id_scores}
    for source, entry in bank["scood"].items():
        features = entry["features"].float()
        logits = entry["logits"].float()
        labels = entry["labels"].numpy()
        source_scores = {"energy": energy_scores(logits, temperature), "msp": msp_scores(logits)}
        for name, normalized, shared in specifications:
            score = score_shared_maha_chunked if shared else score_per_class_maha_chunked
            source_scores[name] = score(
                maybe_l2(features, normalized), *fitted[name], device=device, chunk_size=chunk_size
            )
        for name, scores in source_scores.items():
            (ids, ood) = split_id_ood(id_scores[name], scores, labels)
            results[name]["per_dataset"][source] = summarize_scores(ids, ood)
    for entry in results.values():
        entry["avg"] = {
            metric: float(np.mean([row[metric] for row in entry["per_dataset"].values()]))
            for metric in ("AUROC", "FPR95")
        }
    return results


def main():
    parser = argparse.ArgumentParser(description="CIFAR-LT OOD evaluation")
    parser.add_argument("--bank", required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--reg", type=float, default=1e-05)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk_size", type=int, default=1024)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    bank = torch.load(args.bank, map_location="cpu", weights_only=False)
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    results = evaluate_from_bank(bank, args.temperature, args.reg, device, args.chunk_size)
    results["config"] = {
        "bank": args.bank,
        "temperature": args.temperature,
        "reg": args.reg,
        "device": str(device),
        "chunk_size": args.chunk_size,
        "pooled_covariance": pooled_covariance_config(
            len(bank["train"]["features"]), int(bank["meta"]["num_classes"]), args.reg
        ),
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))
    print(path)


if __name__ == "__main__":
    main()
