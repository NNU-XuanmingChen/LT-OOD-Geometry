# LT-OOD Geometry

CIFAR-LT PC-ERM training, feature extraction, and six post-hoc OOD scores.

Python 3.10+.

```bash
pip install -r requirements.txt
```

Download [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html) and [SCOOD](https://jingkang50.github.io/projects/scood/).
Use the CIFAR parent directory for `--data_root` and the SCOOD root containing `imglist/` for `--scood_root`.

```bash
python train_pcerm.py --dataset cifar10 --data_root data --out checkpoints/cifar10
python extract_features.py --dataset cifar10 --ckpt checkpoints/cifar10/epoch_200.pth --data_root data --scood_root data/scood --out features/cifar10.pt
python evaluate_ood.py --bank features/cifar10.pt --out outputs/cifar10.json
```

For CIFAR-100-LT, replace `cifar10` with `cifar100` in all commands.

Reported PC-ERM metrics are in `results/`.

CS-MD: `maha_pc_raw`; RP-MD: `maha_tied_raw`; HC-MD: `maha_pc_l2`; HPM: `maha_tied_l2`.
`energy` and `msp` denote Energy and MSP. AUROC and FPR95 are fractions; FPR95 uses 95% OOD recall.
`--reg` is the reference ridge λ₀ (default 1e-5). Pooled fits use S/(N−K) + [N/(N−K)]λ₀I.
