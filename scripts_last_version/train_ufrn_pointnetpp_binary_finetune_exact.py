#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_ufrn_pointnetpp_binary_finetune.py

Fine-tuning binario UFRN d21/resto usando PointNet++ preentrenado en Teeth3DS.

Diseñado para calzar con checkpoints que tienen claves:
  sa1, sa2, sa3, sa4
  fp4, fp3, fp2, fp1
  classifier.0 ... classifier.4

La capa final multiclase:
  classifier.4.weight: (15, 128, 1)
  classifier.4.bias:   (15,)

se reemplaza por salida binaria:
  classifier.4.weight: (2, 128, 1)
  classifier.4.bias:   (2,)

Corrección importante:
  - Las convoluciones seguidas por BatchNorm usan bias=False,
    tal como su checkpoint original.
  - La única capa no cargada debe ser classifier.4 por cambio 15 → 2 clases.
"""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Reproducibilidad
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# ============================================================
# Dataset
# ============================================================

class CloudDataset(Dataset):
    def __init__(self, data_dir, split):
        data_dir = Path(data_dir)

        self.X = np.load(
            data_dir / f"X_{split}.npz",
            allow_pickle=True
        )["X"].astype(np.float32)

        self.Y = np.load(
            data_dir / f"Y_{split}.npz",
            allow_pickle=True
        )["Y"].astype(np.int64)

        if self.X.shape[0] != self.Y.shape[0]:
            raise RuntimeError(f"X_{split} e Y_{split} tienen distinto número de muestras.")

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, idx):
        x = np.asarray(self.X[idx], dtype=np.float32)
        y = np.asarray(self.Y[idx], dtype=np.int64)

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        )


# ============================================================
# Geometría: kNN, gather, FP interpolation
# ============================================================

def knn_indices(query, ref, k):
    """
    query: (B, M, 3)
    ref:   (B, N, 3)
    out:   (B, M, k)
    """
    d = torch.cdist(query, ref)
    k = min(int(k), ref.size(1))
    idx = torch.topk(d, k=k, dim=-1, largest=False).indices
    return idx


def batched_gather(points, idx):
    """
    points: (B, N, C)
    idx:    (B, M, K)
    out:    (B, M, K, C)
    """
    B, N, C = points.shape
    _, M, K = idx.shape
    batch = torch.arange(B, device=points.device)[:, None, None].expand(B, M, K)
    return points[batch, idx, :]


# ============================================================
# Bloques PointNet++
# ============================================================

class MLP2d(nn.Module):
    """
    MLP para grupos locales con Conv2d.
    Nombre interno self.net para calzar con:
      saX.mlp.net.0, saX.mlp.net.1, ...
    """
    def __init__(self, in_ch, mlp):
        super().__init__()

        layers = []
        c = int(in_ch)

        for oc in mlp:
            layers.append(nn.Conv2d(c, int(oc), 1, bias=False))
            layers.append(nn.BatchNorm2d(int(oc)))
            layers.append(nn.ReLU(inplace=True))
            c = int(oc)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLP1d(nn.Module):
    """
    MLP para feature propagation con Conv1d.
    Nombre interno self.net para calzar con:
      fpX.mlp.net.0, fpX.mlp.net.1, ...
    """
    def __init__(self, in_ch, mlp):
        super().__init__()

        layers = []
        c = int(in_ch)

        for oc in mlp:
            layers.append(nn.Conv1d(c, int(oc), 1, bias=False))
            layers.append(nn.BatchNorm1d(int(oc)))
            layers.append(nn.ReLU(inplace=True))
            c = int(oc)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SA_Layer(nn.Module):
    """
    Set Abstraction:
      1) selecciona centros por submuestreo uniforme determinista
      2) agrupa vecinos con kNN
      3) concatena coordenadas locales + features
      4) MLP2d + max-pool local

    Estructura diseñada para que los nombres de pesos calcen con:
      sa1.mlp.net.*
      sa2.mlp.net.*
      sa3.mlp.net.*
      sa4.mlp.net.*
    """
    def __init__(self, nsample, in_ch, mlp):
        super().__init__()

        self.nsample = int(nsample)
        self.in_ch = int(in_ch)
        self.mlp = MLP2d(self.in_ch + 3, mlp)
        self.out_ch = int(mlp[-1])

    def forward(self, xyz, feats):
        """
        xyz:   (B, P, 3)
        feats: None o (B, C, P)

        return:
          centers: (B, M, 3)
          out:     (B, C_out, M)
        """
        B, P, _ = xyz.shape

        # M = P/4 como en versión jerárquica liviana usada en scripts previos.
        M = max(1, P // 4)

        idx_center = torch.linspace(
            0,
            P - 1,
            M,
            device=xyz.device
        ).long()

        centers = xyz[:, idx_center, :]  # (B, M, 3)

        idx_knn = knn_indices(
            centers,
            xyz,
            k=min(self.nsample, P)
        )  # (B, M, K)

        neigh_xyz = batched_gather(
            xyz,
            idx_knn
        )  # (B, M, K, 3)

        local_xyz = neigh_xyz - centers[:, :, None, :]  # (B, M, K, 3)
        local_xyz = local_xyz.permute(0, 3, 1, 2).contiguous()  # (B,3,M,K)

        if feats is not None:
            feats_perm = feats.transpose(1, 2).contiguous()  # (B,P,C)
            neigh_f = batched_gather(
                feats_perm,
                idx_knn
            )  # (B,M,K,C)

            neigh_f = neigh_f.permute(0, 3, 1, 2).contiguous()  # (B,C,M,K)
            cat = torch.cat([local_xyz, neigh_f], dim=1)
        else:
            cat = local_xyz

        out = self.mlp(cat)          # (B,Cout,M,K)
        out = out.max(dim=-1)[0]     # (B,Cout,M)

        return centers, out


class FP_Layer(nn.Module):
    """
    Feature Propagation coarse -> fine con interpolación 3-NN.
    Nombres de pesos:
      fp4.mlp.net.*
      fp3.mlp.net.*
      fp2.mlp.net.*
      fp1.mlp.net.*
    """
    def __init__(self, in_ch, mlp):
        super().__init__()
        self.mlp = MLP1d(in_ch, mlp)
        self.out_ch = int(mlp[-1])

    def forward(self, xyz1, xyz2, feats1, feats2):
        """
        xyz1:   puntos finos destino (B,N1,3)
        xyz2:   puntos gruesos fuente (B,N2,3)
        feats1: skip fino, None o (B,C1,N1)
        feats2: features gruesas (B,C2,N2)
        """
        B, N1, _ = xyz1.shape
        _, C2, N2 = feats2.shape

        k = min(3, N2)

        idx = knn_indices(
            xyz1,
            xyz2,
            k=k
        )  # (B,N1,k)

        d = torch.cdist(xyz1, xyz2)
        knn_d = torch.gather(d, 2, idx).clamp_min(1e-8)

        w = 1.0 / knn_d
        w = w / w.sum(dim=-1, keepdim=True)

        feats2_perm = feats2.transpose(1, 2).contiguous()  # (B,N2,C2)
        neigh = batched_gather(
            feats2_perm,
            idx
        )  # (B,N1,k,C2)

        out = (w[..., None] * neigh).sum(dim=2)  # (B,N1,C2)
        out = out.transpose(1, 2).contiguous()   # (B,C2,N1)

        if feats1 is not None:
            out = torch.cat([out, feats1], dim=1)

        return self.mlp(out)


class PointNetPPBinary(nn.Module):
    """
    PointNet++ de segmentación con arquitectura compatible con checkpoint:
      sa1: in=3      -> [64,64,128]
      sa2: in=128+3  -> [128,128,256]
      sa3: in=256+3  -> [256,256,512]
      sa4: in=512+3  -> [512,512,1024]

      fp4: 1024+512 -> [256,256]
      fp3: 256+256  -> [256,256]
      fp2: 256+128  -> [256,128]
      fp1: 128      -> [128,128]

      classifier:
        Conv1d(128,128), BN, ReLU, Dropout, Conv1d(128,num_classes)
    """
    def __init__(self, num_classes=2, nsample=32, dropout=0.5):
        super().__init__()

        self.sa1 = SA_Layer(nsample=nsample, in_ch=0,   mlp=[64, 64, 128])
        self.sa2 = SA_Layer(nsample=nsample, in_ch=128, mlp=[128, 128, 256])
        self.sa3 = SA_Layer(nsample=nsample, in_ch=256, mlp=[256, 256, 512])
        self.sa4 = SA_Layer(nsample=nsample, in_ch=512, mlp=[512, 512, 1024])

        self.fp4 = FP_Layer(in_ch=1024 + 512, mlp=[256, 256])
        self.fp3 = FP_Layer(in_ch=256 + 256,  mlp=[256, 256])
        self.fp2 = FP_Layer(in_ch=256 + 128,  mlp=[256, 128])
        self.fp1 = FP_Layer(in_ch=128,        mlp=[128, 128])

        self.classifier = nn.Sequential(
            nn.Conv1d(128, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Conv1d(128, int(num_classes), 1),
        )

    def forward(self, xyz):
        """
        xyz: (B,P,3)
        out: (B,P,2)
        """
        feats = None

        xyz1, f1 = self.sa1(xyz, feats)      # P -> P/4,   128
        xyz2, f2 = self.sa2(xyz1, f1)        # -> P/16,    256
        xyz3, f3 = self.sa3(xyz2, f2)        # -> P/64,    512
        xyz4, f4 = self.sa4(xyz3, f3)        # -> P/256,   1024

        f = self.fp4(xyz3, xyz4, f3, f4)     # 256 at xyz3
        f = self.fp3(xyz2, xyz3, f2, f)      # 256 at xyz2
        f = self.fp2(xyz1, xyz2, f1, f)      # 128 at xyz1
        f = self.fp1(xyz,  xyz1, None, f)    # 128 at original xyz

        logits = self.classifier(f).transpose(2, 1).contiguous()

        return logits


# ============================================================
# Transfer learning
# ============================================================

def get_state_from_ckpt(ckpt):
    if isinstance(ckpt, dict):
        for key in ["model_state", "model_state_dict", "model", "state_dict"]:
            if key in ckpt:
                return ckpt[key]
    return ckpt


def load_pretrained(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = get_state_from_ckpt(ckpt)

    state = {
        str(k).replace("module.", ""): v
        for k, v in state.items()
    }

    model_state = model.state_dict()

    filtered = {}
    skipped = []

    for k, v in state.items():
        if k in model_state and tuple(model_state[k].shape) == tuple(v.shape):
            filtered[k] = v
        else:
            skipped.append(k)

    missing, unexpected = model.load_state_dict(filtered, strict=False)

    return missing, unexpected, skipped


# ============================================================
# Métricas binarias
# ============================================================

def counts_from_logits(logits, y):
    pred = logits.argmax(dim=-1)

    p = pred.reshape(-1)
    t = y.reshape(-1)

    tp = ((p == 1) & (t == 1)).sum().item()
    fp = ((p == 1) & (t == 0)).sum().item()
    fn = ((p == 0) & (t == 1)).sum().item()
    tn = ((p == 0) & (t == 0)).sum().item()

    return int(tp), int(fp), int(fn), int(tn)


def metrics_from_counts(tp, fp, fn, tn):
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2.0 * prec * rec / max(prec + rec, 1e-8)
    iou = tp / max(tp + fp + fn, 1)
    specificity = tn / max(tn + fp, 1)

    return {
        "acc": float(acc),
        "prec": float(prec),
        "rec": float(rec),
        "f1": float(f1),
        "iou": float(iou),
        "specificity": float(specificity),
    }


def run_epoch(model, loader, criterion, optimizer, device, train=True, grad_clip=1.0):
    model.train(bool(train))

    losses = []
    total_tp = total_fp = total_fn = total_tn = 0

    for X, Y in loader:
        X = X.to(device, non_blocking=True)
        Y = Y.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(X)
        loss = criterion(
            logits.reshape(-1, 2),
            Y.reshape(-1)
        )

        if train:
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()

        losses.append(float(loss.item()))

        with torch.no_grad():
            tp, fp, fn, tn = counts_from_logits(logits, Y)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

    m = metrics_from_counts(total_tp, total_fp, total_fn, total_tn)
    m.update({
        "loss": float(np.mean(losses)) if losses else 0.0,
        "tp": int(total_tp),
        "fp": int(total_fp),
        "fn": int(total_fn),
        "tn": int(total_tn),
    })

    return m


# ============================================================
# Plots
# ============================================================

def plot_curves(history, out_dir):
    plot_dir = Path(out_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        "loss",
        "f1",
        "iou",
        "prec",
        "rec",
        "acc",
        "specificity",
        "lr",
    ]

    for metric in metrics:
        xs = [r["epoch"] for r in history]

        plt.figure(figsize=(8, 5))

        if metric == "lr":
            ys = [r["lr"] for r in history]
            plt.plot(xs, ys, label="learning_rate")
            plt.ylabel("lr")
        else:
            plt.plot(xs, [r[f"train_{metric}"] for r in history], label=f"train_{metric}")
            plt.plot(xs, [r[f"val_{metric}"] for r in history], label=f"val_{metric}")
            plt.ylabel(metric)

            if metric == "loss":
                best = min(history, key=lambda r: r[f"val_{metric}"])
                plt.axvline(
                    best["epoch"],
                    linestyle="--",
                    alpha=0.6,
                    label=f"best val loss: ep {best['epoch']}"
                )
            else:
                best = max(history, key=lambda r: r[f"val_{metric}"])
                plt.axvline(
                    best["epoch"],
                    linestyle="--",
                    alpha=0.6,
                    label=f"best val {metric}: ep {best['epoch']}"
                )

        plt.xlabel("Época")
        plt.title(f"PointNet++ UFRN binario d21 - {metric}")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(plot_dir / f"curve_{metric}.png", dpi=300)
        plt.close()


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--from_scratch", action="store_true")

    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eta_min", type=float, default=1e-6)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--pos_weight", type=float, default=20.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--min_delta", type=float, default=1e-5)

    ap.add_argument("--nsample", type=int, default=32)

    args = ap.parse_args()

    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    loaders = {}
    for split in ["train", "val", "test"]:
        ds = CloudDataset(args.data_dir, split)
        loaders[split] = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    model = PointNetPPBinary(
        num_classes=2,
        nsample=args.nsample,
        dropout=args.dropout
    ).to(device)

    if args.from_scratch or args.ckpt is None:
        missing, unexpected, skipped = [], [], []
        print("[INFO] Entrenando PointNet++ binario desde cero.")
    else:
        missing, unexpected, skipped = load_pretrained(model, args.ckpt)
        print("[INFO] missing:", missing)
        print("[INFO] unexpected:", unexpected)
        print("[INFO] skipped:", skipped)

    weights = torch.tensor(
        [1.0, float(args.pos_weight)],
        dtype=torch.float32,
        device=device
    )

    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.eta_min
    )

    best_val_f1 = -1.0
    best_epoch = -1
    bad_epochs = 0
    history = []

    fieldnames = [
        "epoch", "lr",
        "train_loss", "train_acc", "train_prec", "train_rec", "train_f1", "train_iou", "train_specificity",
        "train_tp", "train_fp", "train_fn", "train_tn",
        "val_loss", "val_acc", "val_prec", "val_rec", "val_f1", "val_iou", "val_specificity",
        "val_tp", "val_fp", "val_fn", "val_tn",
    ]

    csv_path = out_dir / "metrics_epoch.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            lr_now = float(optimizer.param_groups[0]["lr"])

            tr = run_epoch(
                model,
                loaders["train"],
                criterion,
                optimizer,
                device,
                train=True,
                grad_clip=args.grad_clip
            )

            va = run_epoch(
                model,
                loaders["val"],
                criterion,
                optimizer,
                device,
                train=False,
                grad_clip=args.grad_clip
            )

            row = {
                "epoch": int(epoch),
                "lr": lr_now,
            }

            for k in ["loss", "acc", "prec", "rec", "f1", "iou", "specificity", "tp", "fp", "fn", "tn"]:
                row[f"train_{k}"] = tr[k]
                row[f"val_{k}"] = va[k]

            writer.writerow(row)
            f.flush()

            history.append(row)

            improved = va["f1"] > best_val_f1 + args.min_delta

            if improved:
                best_val_f1 = float(va["f1"])
                best_epoch = int(epoch)
                bad_epochs = 0

                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "args": vars(args),
                    "best_val_f1": best_val_f1,
                    "val_metrics": va,
                    "architecture": "PointNetPPBinary_4SA_4FP_biasFalseBN_exact",
                }, out_dir / "best.pt")
            else:
                bad_epochs += 1

            print(
                f"[{epoch:03d}/{args.epochs}] "
                f"lr={lr_now:.2e} | "
                f"train loss={tr['loss']:.4f} f1={tr['f1']:.4f} iou={tr['iou']:.4f} "
                f"P={tr['prec']:.4f} R={tr['rec']:.4f} | "
                f"val loss={va['loss']:.4f} f1={va['f1']:.4f} iou={va['iou']:.4f} "
                f"P={va['prec']:.4f} R={va['rec']:.4f} | "
                f"best_ep={best_epoch} bad={bad_epochs}/{args.patience}"
            )

            scheduler.step()

            if bad_epochs >= args.patience:
                print(f"[EARLY STOP] Sin mejora en val_f1 por {args.patience} épocas.")
                break

    torch.save({
        "model_state": model.state_dict(),
        "args": vars(args),
        "last_epoch": int(history[-1]["epoch"] if history else 0),
        "architecture": "PointNetPPBinary_4SA_4FP_biasFalseBN_exact",
    }, out_dir / "last.pt")

    save_json(history, out_dir / "history.json")
    plot_curves(history, out_dir)

    best = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(best["model_state"])

    test = run_epoch(
        model,
        loaders["test"],
        criterion,
        optimizer,
        device,
        train=False,
        grad_clip=args.grad_clip
    )

    save_json(test, out_dir / "test_metrics.json")

    run_meta = {
        "script_name": Path(__file__).name,
        "data_dir": args.data_dir,
        "ckpt": args.ckpt,
        "from_scratch": bool(args.from_scratch),
        "out_dir": str(out_dir),
        "device": str(device),
        "epochs_requested": int(args.epochs),
        "epochs_ran": int(history[-1]["epoch"] if history else 0),
        "best_epoch": int(best_epoch),
        "best_val_f1": float(best_val_f1),
        "test": test,
        "missing": [str(x) for x in missing],
        "unexpected": [str(x) for x in unexpected],
        "skipped": [str(x) for x in skipped],
        "scheduler": "CosineAnnealingLR",
        "eta_min": float(args.eta_min),
        "pos_weight": float(args.pos_weight),
        "nsample": int(args.nsample),
        "architecture": "PointNetPPBinary_4SA_4FP_biasFalseBN_exact",
    }

    save_json(run_meta, out_dir / "run_meta.json")

    print("[DONE]")
    print(json.dumps(run_meta, indent=2))


if __name__ == "__main__":
    main()
