#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_ufrn_dgcnn_binary_finetune.py

Fine-tuning binario UFRN d21/resto usando DGCNN preentrenado en Teeth3DS.

Checkpoint esperado:
  ec1.net.0.weight (64, 6, 1, 1)
  ec2.net.0.weight (64, 128, 1, 1)
  ec3.net.0.weight (128, 128, 1, 1)
  ec4.net.0.weight (256, 256, 1, 1)
  fuse.0.weight    (emb_dims, 512, 1), usualmente emb_dims=768
  head.0.weight    (512, emb_dims + 512, 1)
  head.4.weight    (256, 512, 1)
  head.8.weight    (15, 256, 1)  -> se reemplaza por (2, 256, 1)
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


class CloudDataset(Dataset):
    def __init__(self, data_dir, split):
        data_dir = Path(data_dir)
        self.X = np.load(data_dir / f"X_{split}.npz", allow_pickle=True)["X"].astype(np.float32)
        self.Y = np.load(data_dir / f"Y_{split}.npz", allow_pickle=True)["Y"].astype(np.int64)
        if self.X.shape[0] != self.Y.shape[0]:
            raise RuntimeError(f"X_{split} e Y_{split} tienen distinto número de muestras.")

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, idx):
        x = np.asarray(self.X[idx], dtype=np.float32)
        y = np.asarray(self.Y[idx], dtype=np.int64)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


# ============================================================
# DGCNN helpers
# ============================================================

def knn(x, k):
    """x: (B,C,N) -> idx: (B,N,k)."""
    inner = -2.0 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    return pairwise_distance.topk(k=k, dim=-1)[1]


def get_graph_feature(x, k=20, idx=None):
    """x: (B,C,N) -> (B,2C,N,k), concat(neighbor-center, center)."""
    B, C, N = x.size()
    if idx is None:
        idx = knn(x, k=k)
    device = x.device
    idx_base = torch.arange(0, B, device=device).view(-1, 1, 1) * N
    idx = (idx + idx_base).view(-1)
    x_t = x.transpose(2, 1).contiguous()
    feature = x_t.view(B * N, C)[idx, :]
    feature = feature.view(B, N, k, C)
    x_center = x_t.view(B, N, 1, C).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x_center, x_center), dim=3)
    return feature.permute(0, 3, 1, 2).contiguous()


class EdgeConvBlock(nn.Module):
    """Conv2d + BN2d + LeakyReLU. Nombres: ecX.net.0 y ecX.net.1."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(int(in_ch), int(out_ch), kernel_size=1, bias=False),
            nn.BatchNorm2d(int(out_ch)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class DGCNNBinary(nn.Module):
    """
    Compatible con checkpoint DGCNN:
      ec1: 6 -> 64
      ec2: 128 -> 64
      ec3: 128 -> 128
      ec4: 256 -> 256
      fuse: 512 -> emb_dims
      head: emb_dims+512 -> 512 -> 256 -> num_classes
    """
    def __init__(self, num_classes=2, k=20, emb_dims=768, dropout=0.5):
        super().__init__()
        self.k = int(k)
        self.emb_dims = int(emb_dims)

        self.ec1 = EdgeConvBlock(6, 64)
        self.ec2 = EdgeConvBlock(128, 64)
        self.ec3 = EdgeConvBlock(128, 128)
        self.ec4 = EdgeConvBlock(256, 256)

        self.fuse = nn.Sequential(
            nn.Conv1d(64 + 64 + 128 + 256, self.emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.emb_dims),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.head = nn.Sequential(
            nn.Conv1d(self.emb_dims + 64 + 64 + 128 + 256, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(float(dropout)),
            nn.Conv1d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(float(dropout)),
            nn.Conv1d(256, int(num_classes), kernel_size=1, bias=True),
        )

    def forward(self, xyz):
        # xyz: (B,N,3)
        x = xyz.transpose(2, 1).contiguous()

        x = get_graph_feature(x, k=self.k)
        x = self.ec1(x)
        x1 = x.max(dim=-1)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.ec2(x)
        x2 = x.max(dim=-1)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.ec3(x)
        x3 = x.max(dim=-1)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.ec4(x)
        x4 = x.max(dim=-1)[0]

        x_cat = torch.cat((x1, x2, x3, x4), dim=1)
        x_fuse = self.fuse(x_cat)
        x_global = F.adaptive_max_pool1d(x_fuse, 1).repeat(1, 1, xyz.size(1))
        x_head = torch.cat((x_global, x_cat), dim=1)
        return self.head(x_head).transpose(2, 1).contiguous()


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
    state = {str(k).replace("module.", ""): v for k, v in state.items()}
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
# Métricas
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
        loss = criterion(logits.reshape(-1, 2), Y.reshape(-1))

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
    metrics = ["loss", "f1", "iou", "prec", "rec", "acc", "specificity", "lr"]

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
                plt.axvline(best["epoch"], linestyle="--", alpha=0.6, label=f"best val loss: ep {best['epoch']}")
            else:
                best = max(history, key=lambda r: r[f"val_{metric}"])
                plt.axvline(best["epoch"], linestyle="--", alpha=0.6, label=f"best val {metric}: ep {best['epoch']}")

        plt.xlabel("Época")
        plt.title(f"DGCNN UFRN binario d21 - {metric}")
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
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--emb_dims", type=int, default=768)
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

    model = DGCNNBinary(num_classes=2, k=args.k, emb_dims=args.emb_dims, dropout=args.dropout).to(device)

    if args.from_scratch or args.ckpt is None:
        missing, unexpected, skipped = [], [], []
        print("[INFO] Entrenando DGCNN binario desde cero.")
    else:
        missing, unexpected, skipped = load_pretrained(model, args.ckpt)
        print("[INFO] missing:", missing)
        print("[INFO] unexpected:", unexpected)
        print("[INFO] skipped:", skipped)

    weights = torch.tensor([1.0, float(args.pos_weight)], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)

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
            tr = run_epoch(model, loaders["train"], criterion, optimizer, device, train=True, grad_clip=args.grad_clip)
            va = run_epoch(model, loaders["val"], criterion, optimizer, device, train=False, grad_clip=args.grad_clip)

            row = {"epoch": int(epoch), "lr": lr_now}
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
                    "architecture": "DGCNNBinary_EdgeConv_k20_emb768",
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
        "architecture": "DGCNNBinary_EdgeConv_k20_emb768",
    }, out_dir / "last.pt")

    save_json(history, out_dir / "history.json")
    plot_curves(history, out_dir)

    best = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(best["model_state"])

    test = run_epoch(model, loaders["test"], criterion, optimizer, device, train=False, grad_clip=args.grad_clip)
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
        "k": int(args.k),
        "emb_dims": int(args.emb_dims),
        "architecture": "DGCNNBinary_EdgeConv_k20_emb768",
    }
    save_json(run_meta, out_dir / "run_meta.json")
    print("[DONE]")
    print(json.dumps(run_meta, indent=2))


if __name__ == "__main__":
    main()
