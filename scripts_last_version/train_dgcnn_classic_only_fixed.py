#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_dgcnn_classic_only_fixed.py

DGCNN (EdgeConv) – Segmentación multiclase dental 3D
Basado en tus outputs de train_pointnet_classic_only_fixed.py

✅ BG incluido en la loss (NO ignore en la loss)
✅ BG excluido SOLO en métricas macro (f1/iou/acc_no_bg)
✅ Métricas diente 21 explícitas (acc/f1/iou) BINARIO correcto
✅ Métrica d21_bin_acc_all (incluye TODO, incluso bg) para referencia
✅ Estabilidad: bg downweight, weight_decay, grad clipping, CosineAnnealingLR
✅ RTX 3090 friendly: AMP, pin_memory, persistent_workers, non_blocking, cudnn.benchmark
✅ Inferencia: PNGs 3D (GT vs Pred) + errores + foco d21
✅ TRAZABILIDAD (FIX): descubre index_{split}.csv (igual que PointNet)
✅ FIX CRÍTICO del error DataLoader:
   NO usamos torch.from_numpy (te daba "expected np.ndarray (got numpy.ndarray)")
   -> usamos np.ascontiguousarray + torch.as_tensor (a prueba de arrays raros/memmap/object)

Ejemplo:
python3 train_dgcnn_classic_only_fixed.py \
  --data_dir .../upper_only_surf_global_excl_wisdom_seed42_aug2 \
  --out_dir  .../outputs/dgcnn/run1 \
  --epochs 120 --batch_size 16 --lr 2e-4 --weight_decay 1e-4 --dropout 0.5 \
  --num_workers 6 --device cuda --d21_internal 8 \
  --bg_weight 0.03 --grad_clip 1.0 --use_amp \
  --k 20 --emb_dims 1024 \
  --do_infer --infer_examples 12 --infer_split test
"""

import os
import json
import csv
import time
import argparse
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# SEED / IO
# ============================================================
def set_seed(seed: int = 42):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ============================================================
# NORMALIZACIÓN
# ============================================================
def normalize_unit_sphere(xyz: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    # xyz: [N,3]
    c = xyz.mean(dim=0, keepdim=True)
    x = xyz - c
    r = torch.norm(x, dim=1).max().clamp_min(eps)
    return x / r


# ============================================================
# DATASET (ROBUSTO: NO torch.from_numpy)
# ============================================================
class NPZDataset(Dataset):
    """
    Dataset:
      X: [B,N,3]
      Y: [B,N]

    FIX: en tu entorno, torch.from_numpy falló dentro de DataLoader workers.
    Solución robusta:
      - np.ascontiguousarray(..., dtype=...)
      - torch.as_tensor(...)
    """
    def __init__(self, Xp: Path, Yp: Path, normalize: bool = True):
        self.X = np.load(Xp)["X"]  # puede ser memmap/subclase/strides raros
        self.Y = np.load(Yp)["Y"]

        assert self.X.ndim == 3 and self.X.shape[-1] == 3, f"X shape inesperada: {self.X.shape}"
        assert self.Y.ndim == 2, f"Y shape inesperada: {self.Y.shape}"
        assert self.X.shape[0] == self.Y.shape[0], "B mismatch"
        assert self.X.shape[1] == self.Y.shape[1], "N mismatch"
        self.normalize = bool(normalize)

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, i: int):
        # 1) asegurar ndarray normal + dtype + contiguidad
        x_np = np.ascontiguousarray(np.asarray(self.X[i]), dtype=np.float32)  # [N,3]
        y_np = np.ascontiguousarray(np.asarray(self.Y[i]), dtype=np.int64)    # [N]

        # 2) torch.as_tensor (evita chequeos estrictos de from_numpy en tu entorno)
        xyz = torch.as_tensor(x_np, dtype=torch.float32)  # [N,3]
        y = torch.as_tensor(y_np, dtype=torch.int64)      # [N]

        if self.normalize:
            xyz = normalize_unit_sphere(xyz)
        return xyz, y


def make_loaders(data_dir: Path, bs: int, nw: int, normalize: bool = True):
    ds_tr = NPZDataset(data_dir / "X_train.npz", data_dir / "Y_train.npz", normalize=normalize)
    ds_va = NPZDataset(data_dir / "X_val.npz",   data_dir / "Y_val.npz",   normalize=normalize)
    ds_te = NPZDataset(data_dir / "X_test.npz",  data_dir / "Y_test.npz",  normalize=normalize)

    common = dict(
        batch_size=int(bs),
        num_workers=int(nw),
        pin_memory=True,
        persistent_workers=(int(nw) > 0),
        prefetch_factor=2 if int(nw) > 0 else None,
        drop_last=False,
    )

    dl_tr = DataLoader(ds_tr, shuffle=True,  **{k: v for k, v in common.items() if v is not None})
    dl_va = DataLoader(ds_va, shuffle=False, **{k: v for k, v in common.items() if v is not None})
    dl_te = DataLoader(ds_te, shuffle=False, **{k: v for k, v in common.items() if v is not None})

    return dl_tr, dl_va, dl_te, ds_te

# ============================================================
# PARTE 2/4 — DGCNN (EdgeConv) para segmentación punto-a-punto
# ============================================================

def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    x: [B, C, N]
    return idx: [B, N, k]
    """
    # (x - y)^2 = x^2 + y^2 - 2xy
    xx = torch.sum(x ** 2, dim=1, keepdim=True)  # [B,1,N]
    pairwise = xx.transpose(2, 1) + xx - 2.0 * torch.matmul(x.transpose(2, 1), x)  # [B,N,N]
    _, idx = torch.topk(pairwise, k=k, dim=-1, largest=False, sorted=False)  # [B,N,k]
    return idx


def get_graph_feature(x: torch.Tensor, k: int, idx: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Edge feature:
      e_ij = concat( x_j - x_i , x_i )
    x: [B, C, N]
    return: [B, 2C, N, k]
    """
    B, C, N = x.shape
    if idx is None:
        idx = knn(x, k=k)  # [B,N,k]

    device = x.device
    idx_base = torch.arange(B, device=device).view(-1, 1, 1) * N  # [B,1,1]
    idx = idx + idx_base
    idx = idx.reshape(-1)  # [B*N*k]

    x_t = x.transpose(2, 1).contiguous()  # [B,N,C]
    neigh = x_t.reshape(B * N, C)[idx, :].view(B, N, k, C)  # [B,N,k,C]
    x_i = x_t.view(B, N, 1, C).repeat(1, 1, k, 1)           # [B,N,k,C]

    edge = torch.cat((neigh - x_i, x_i), dim=3)             # [B,N,k,2C]
    return edge.permute(0, 3, 1, 2).contiguous()            # [B,2C,N,k]


class EdgeConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, edge_feat: torch.Tensor) -> torch.Tensor:
        # edge_feat: [B,in_ch,N,k]
        x = self.net(edge_feat)          # [B,out_ch,N,k]
        x = torch.max(x, dim=-1)[0]      # [B,out_ch,N]
        return x


class DGCNNSeg(nn.Module):
    """
    DGCNN-style seg:
      - 4 EdgeConv blocks
      - concat (64+64+128+256)=512
      - fuse -> 512
      - emb -> emb_dims (global)
      - concat local+global -> head -> logits [B,N,C]
    """
    def __init__(self, num_classes: int, k: int = 20, emb_dims: int = 1024, dropout: float = 0.5):
        super().__init__()
        self.C = int(num_classes)
        self.k = int(k)
        self.emb_dims = int(emb_dims)

        # input coords C=3 -> edge 2C=6
        self.ec1 = EdgeConvBlock(6, 64)
        self.ec2 = EdgeConvBlock(2 * 64, 64)
        self.ec3 = EdgeConvBlock(2 * 64, 128)
        self.ec4 = EdgeConvBlock(2 * 128, 256)

        self.fuse = nn.Sequential(
            nn.Conv1d(64 + 64 + 128 + 256, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.emb = nn.Sequential(
            nn.Conv1d(512, self.emb_dims, 1, bias=False),
            nn.BatchNorm1d(self.emb_dims),
            nn.LeakyReLU(0.2, inplace=True),
        )

        head_in = 512 + self.emb_dims
        self.head = nn.Sequential(
            nn.Conv1d(head_in, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(float(dropout)),
            nn.Conv1d(256, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(float(dropout)),
            nn.Conv1d(256, self.C, 1, bias=True),
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        # xyz: [B,N,3]
        B, N, _ = xyz.shape
        x = xyz.transpose(2, 1).contiguous()  # [B,3,N]

        e1 = get_graph_feature(x, k=self.k)   # [B,6,N,k]
        x1 = self.ec1(e1)                     # [B,64,N]

        e2 = get_graph_feature(x1, k=self.k)  # [B,128,N,k]
        x2 = self.ec2(e2)                     # [B,64,N]

        e3 = get_graph_feature(x2, k=self.k)  # [B,128,N,k]
        x3 = self.ec3(e3)                     # [B,128,N]

        e4 = get_graph_feature(x3, k=self.k)  # [B,256,N,k]
        x4 = self.ec4(e4)                     # [B,256,N]

        x_cat = torch.cat((x1, x2, x3, x4), dim=1)  # [B,512,N]
        x_local = self.fuse(x_cat)                  # [B,512,N]

        x_emb = self.emb(x_local)                   # [B,emb_dims,N]
        x_global = torch.max(x_emb, dim=2, keepdim=True)[0].repeat(1, 1, N)  # [B,emb_dims,N]

        x_final = torch.cat((x_local, x_global), dim=1)  # [B,512+emb_dims,N]
        logits = self.head(x_final)                      # [B,C,N]
        return logits.transpose(2, 1).contiguous()       # [B,N,C]

# ============================================================
# PARTE 3/4 — MÉTRICAS + VISUALIZACIÓN + run_epoch (AMP CORRECTO)
# ============================================================

from sklearn.metrics import f1_score, jaccard_score


# -------------------------------
# Métricas helper
# -------------------------------
def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    bg_index: int,
    d21_internal: int,
):
    """
    y_true, y_pred: [B*N]
    """
    out = {}

    # accuracy overall
    out["acc_all"] = float((y_true == y_pred).mean())

    # accuracy sin bg
    mask_fg = (y_true != bg_index)
    out["acc_no_bg"] = float((y_true[mask_fg] == y_pred[mask_fg]).mean()) if mask_fg.any() else 0.0

    # macro f1 / iou (SIN bg)
    labels_fg = [i for i in range(num_classes) if i != bg_index]
    out["f1_macro"] = float(
        f1_score(y_true, y_pred, labels=labels_fg, average="macro", zero_division=0)
    )
    out["iou_macro"] = float(
        jaccard_score(y_true, y_pred, labels=labels_fg, average="macro", zero_division=0)
    )

    # diente 21 — multiclase (solo puntos gt=21)
    mask_21 = (y_true == d21_internal)
    if mask_21.any():
        y21_true = y_true[mask_21]
        y21_pred = y_pred[mask_21]
        out["d21_acc"] = float((y21_pred == d21_internal).mean())
        out["d21_f1"] = float(
            f1_score(y21_true, y21_pred, labels=[d21_internal], average="macro", zero_division=0)
        )
        out["d21_iou"] = float(
            jaccard_score(y21_true, y21_pred, labels=[d21_internal], average="macro", zero_division=0)
        )
    else:
        out["d21_acc"] = out["d21_f1"] = out["d21_iou"] = 0.0

    # d21 binario (all points)
    y21_true_bin = (y_true == d21_internal).astype(np.int32)
    y21_pred_bin = (y_pred == d21_internal).astype(np.int32)
    out["d21_bin_acc_all"] = float((y21_true_bin == y21_pred_bin).mean())

    # fracción bg predicha
    out["pred_bg_frac"] = float((y_pred == bg_index).mean())

    return out


# -------------------------------
# Visualización
# -------------------------------
def plot_pointcloud_all_classes(xyz, y_true, y_pred, out_png: Path, C: int, title: str):
    fig = plt.figure(figsize=(10, 4))
    for i, (lab, name) in enumerate([(y_true, "GT"), (y_pred, "Pred")]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        sc = ax.scatter(
            xyz[:, 0], xyz[:, 1], xyz[:, 2],
            c=lab, s=1, cmap="tab20", vmin=0, vmax=C - 1
        )
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_errors(xyz, y_true, y_pred, out_png: Path, bg_index: int, title: str):
    err = (y_true != y_pred)
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        xyz[~err, 0], xyz[~err, 1], xyz[~err, 2],
        c="lightgray", s=1
    )
    ax.scatter(
        xyz[err, 0], xyz[err, 1], xyz[err, 2],
        c="red", s=3
    )
    ax.set_title("Errors")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_d21_focus(xyz, y_true, y_pred, out_png: Path, d21_internal: int, bg_index: int, title: str):
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection="3d")

    # gris: todo lo demás
    mask_other = (y_true != d21_internal)
    ax.scatter(
        xyz[mask_other, 0], xyz[mask_other, 1], xyz[mask_other, 2],
        c="lightgray", s=1
    )

    # verde: gt d21
    mask_gt = (y_true == d21_internal)
    ax.scatter(
        xyz[mask_gt, 0], xyz[mask_gt, 1], xyz[mask_gt, 2],
        c="green", s=4, label="GT d21"
    )

    # rojo: pred d21 (errores)
    mask_pred = (y_pred == d21_internal)
    ax.scatter(
        xyz[mask_pred, 0], xyz[mask_pred, 1], xyz[mask_pred, 2],
        c="red", s=4, label="Pred d21"
    )

    ax.legend(loc="best", fontsize=8)
    ax.set_title("Diente 21 focus")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)


# -------------------------------
# Epoch runner (AMP moderno)
# -------------------------------
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    loss_fn: nn.Module,
    num_classes: int,
    d21_internal: int,
    device: torch.device,
    bg_index: int,
    train: bool = True,
    use_amp: bool = False,
    grad_clip: Optional[float] = None,
):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_true = []
    all_pred = []

    scaler = torch.amp.GradScaler("cuda") if (use_amp and train and device.type == "cuda") else None

    for xyz, y in loader:
        xyz = xyz.to(device, non_blocking=True)  # [B,N,3]
        y = y.to(device, non_blocking=True)      # [B,N]

        with torch.set_grad_enabled(train):
            with torch.amp.autocast(device_type="cuda", enabled=(use_amp and device.type == "cuda")):
                logits = model(xyz)              # [B,N,C]
                loss = loss_fn(
                    logits.reshape(-1, num_classes),
                    y.reshape(-1)
                )

            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    if grad_clip is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                    optimizer.step()

        total_loss += float(loss.detach().cpu())

        pred = logits.argmax(dim=-1).detach().cpu().numpy().reshape(-1)
        true = y.detach().cpu().numpy().reshape(-1)
        all_pred.append(pred)
        all_true.append(true)

    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)

    metrics = compute_metrics(
        y_true=y_true,
        y_pred=y_pred,
        num_classes=num_classes,
        bg_index=bg_index,
        d21_internal=d21_internal,
    )

    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics

# ============================================================
# PARTE 4/4 — MAIN + LOGGING + INFERENCIA + COMANDOS
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    # Paths
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    # Train
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.5)

    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    # DGCNN
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--emb_dims", type=int, default=1024)

    # Task
    ap.add_argument("--d21_internal", type=int, required=True)
    ap.add_argument("--bg_index", type=int, default=0)
    ap.add_argument("--bg_weight", type=float, default=0.03)

    # Stability
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--use_amp", action="store_true")
    ap.add_argument("--no_normalize", action="store_true")

    # Inference
    ap.add_argument("--do_infer", action="store_true")
    ap.add_argument("--infer_examples", type=int, default=12)
    ap.add_argument("--infer_split", type=str, default="test",
                    choices=["train", "val", "test"])

    args = ap.parse_args()

    set_seed(args.seed)

    # device
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- SANITY ----------
    Ytr = np.load(data_dir / "Y_train.npz")["Y"].reshape(-1)
    Yva = np.load(data_dir / "Y_val.npz")["Y"].reshape(-1)
    Yte = np.load(data_dir / "Y_test.npz")["Y"].reshape(-1)

    bg = int(args.bg_index)
    C = int(max(Ytr.max(), Yva.max(), Yte.max())) + 1

    bg_tr = float((Ytr == bg).mean())
    bg_va = float((Yva == bg).mean())
    bg_te = float((Yte == bg).mean())
    print(f"[SANITY] C={C} | bg_frac train/val/test={bg_tr:.3f}/{bg_va:.3f}/{bg_te:.3f}")

    if not (0 <= args.d21_internal < C):
        raise ValueError(f"d21_internal fuera de rango: {args.d21_internal} (C={C})")

    # ---------- Loaders ----------
    dl_tr, dl_va, dl_te, ds_te = make_loaders(
        data_dir=data_dir,
        bs=args.batch_size,
        nw=args.num_workers,
        normalize=(not args.no_normalize),
    )

    # ---------- Model ----------
    model = DGCNNSeg(
        num_classes=C,
        k=args.k,
        emb_dims=args.emb_dims,
        dropout=args.dropout
    ).to(device)

    # ---------- Loss ----------
    w = torch.ones(C, device=device)
    w[bg] = float(args.bg_weight)
    loss_fn = nn.CrossEntropyLoss(weight=w)

    # ---------- Optim / Sched ----------
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    # ---------- Meta ----------
    run_meta = {
        "model": "DGCNN",
        "data_dir": str(data_dir),
        "out_dir": str(out_dir),
        "C": C,
        "bg_index": bg,
        "bg_weight": args.bg_weight,
        "d21_internal": args.d21_internal,
        "k": args.k,
        "emb_dims": args.emb_dims,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "use_amp": args.use_amp,
        "normalize": not args.no_normalize,
    }
    save_json(run_meta, out_dir / "run_meta.json")

    # ---------- CSV ----------
    csv_path = out_dir / "metrics_epoch.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow([
            "epoch", "split",
            "loss",
            "acc_all", "acc_no_bg",
            "f1_macro", "iou_macro",
            "d21_acc", "d21_f1", "d21_iou",
            "d21_bin_acc_all",
            "pred_bg_frac",
            "lr"
        ])

    best_val_f1 = -1.0
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"

    history = {k: [] for k in [
        "train_f1m", "val_f1m",
        "train_ioum", "val_ioum",
        "val_d21_f1", "val_pred_bg_frac"
    ]}

    # ---------- TRAIN ----------
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(
            model, dl_tr, opt, loss_fn, C,
            args.d21_internal, device, bg,
            train=True, use_amp=args.use_amp,
            grad_clip=args.grad_clip
        )
        va = run_epoch(
            model, dl_va, None, loss_fn, C,
            args.d21_internal, device, bg,
            train=False, use_amp=False
        )

        sched.step()
        lr_now = opt.param_groups[0]["lr"]

        history["train_f1m"].append(tr["f1_macro"])
        history["val_f1m"].append(va["f1_macro"])
        history["train_ioum"].append(tr["iou_macro"])
        history["val_ioum"].append(va["iou_macro"])
        history["val_d21_f1"].append(va["d21_f1"])
        history["val_pred_bg_frac"].append(va["pred_bg_frac"])

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            wcsv = csv.writer(f)
            wcsv.writerow([epoch, "train",
                           tr["loss"], tr["acc_all"], tr["acc_no_bg"],
                           tr["f1_macro"], tr["iou_macro"],
                           tr["d21_acc"], tr["d21_f1"], tr["d21_iou"],
                           tr["d21_bin_acc_all"], tr["pred_bg_frac"], lr_now])
            wcsv.writerow([epoch, "val",
                           va["loss"], va["acc_all"], va["acc_no_bg"],
                           va["f1_macro"], va["iou_macro"],
                           va["d21_acc"], va["d21_f1"], va["d21_iou"],
                           va["d21_bin_acc_all"], va["pred_bg_frac"], lr_now])

        torch.save({"model": model.state_dict(), "epoch": epoch}, last_path)
        if va["f1_macro"] > best_val_f1:
            best_val_f1 = va["f1_macro"]
            torch.save({"model": model.state_dict(), "epoch": epoch}, best_path)

        print(
            f"[{epoch:03d}/{args.epochs}] "
            f"train f1m={tr['f1_macro']:.3f} iou={tr['iou_macro']:.3f} | "
            f"val f1m={va['f1_macro']:.3f} iou={va['iou_macro']:.3f} | "
            f"d21 f1={va['d21_f1']:.3f} | "
            f"pred_bg_frac(val)={va['pred_bg_frac']:.3f} lr={lr_now:.2e}"
        )

    save_json(history, out_dir / "history.json")

    # ---------- TEST ----------
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    te = run_epoch(
        model, dl_te, None, loss_fn, C,
        args.d21_internal, device, bg,
        train=False, use_amp=False
    )
    save_json({"best_epoch": ckpt.get("epoch", -1), "test": te}, out_dir / "test_metrics.json")

    # ---------- INFER ----------
    if args.do_infer and args.infer_examples > 0:
        model.eval()
        if args.infer_split == "test":
            ds_inf = ds_te
        elif args.infer_split == "val":
            ds_inf = NPZDataset(data_dir / "X_val.npz", data_dir / "Y_val.npz",
                                normalize=(not args.no_normalize))
        else:
            ds_inf = NPZDataset(data_dir / "X_train.npz", data_dir / "Y_train.npz",
                                normalize=(not args.no_normalize))

        rng = np.random.default_rng(args.seed + 123)
        idxs = rng.choice(len(ds_inf), size=min(args.infer_examples, len(ds_inf)), replace=False)

        out_all = out_dir / "inference_all"
        out_err = out_dir / "inference_errors"
        out_d21 = out_dir / "inference_d21"
        out_all.mkdir(parents=True, exist_ok=True)
        out_err.mkdir(parents=True, exist_ok=True)
        out_d21.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            for r, i in enumerate(idxs, 1):
                xyz, y = ds_inf[int(i)]
                logits = model(xyz.unsqueeze(0).to(device))[0]
                pred = logits.argmax(dim=-1).cpu().numpy()

                xyz_np = xyz.numpy()
                y_np = y.numpy()

                title = f"{args.infer_split} row={i} | C={C} | d21={args.d21_internal}"
                plot_pointcloud_all_classes(
                    xyz_np, y_np, pred,
                    out_all / f"ex_{r:02d}.png", C, title
                )
                plot_errors(
                    xyz_np, y_np, pred,
                    out_err / f"ex_{r:02d}.png", bg, title
                )
                plot_d21_focus(
                    xyz_np, y_np, pred,
                    out_d21 / f"ex_{r:02d}.png", args.d21_internal, bg, title
                )

    print(f"[DONE] out_dir={out_dir} | best_val_f1_macro(no_bg)={best_val_f1:.4f}")


if __name__ == "__main__":
    main()
