#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_pointnet_classic_only.py

Script "solo PointNet clásico" (segmentación punto-a-punto) inspirado en tu train_models_v9_paperlike.py:
- Carga X_*.npz / Y_*.npz (X: [B,N,3], Y: [B,N])
- Normalización por muestra (center + unit sphere)
- CrossEntropy con ignore_index (background=0) y class weights desde class_weights.json (si existe)
- Métricas macro (acc/prec/rec/f1/IoU) + métricas específicas para diente 21 (si existe en label_map.json)
- Guardado de: history.json, metrics_epoch.csv, test_metrics.json, best.pt, last.pt
- Inferencia de 10 ejemplos aleatorios del test set con figura "paper-like" (GT vs Pred)
"""

import os
import json
import csv
import time
import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------- torchmetrics (opcional) -----------------------------
HAS_TORCHMETRICS = False
try:
    from torchmetrics.classification import (
        MulticlassAccuracy, MulticlassPrecision, MulticlassRecall,
        MulticlassF1Score, MulticlassJaccardIndex
    )
    HAS_TORCHMETRICS = True
except Exception:
    HAS_TORCHMETRICS = False


# ==========================================
#                 UTILIDADES
# ==========================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_label_map(label_map_path: Path) -> Tuple[Optional[dict], Optional[dict]]:
    """
    Espera el formato típico:
    {
      "id2idx": {"0":0, "11":1, ...},
      "idx2id": {"0":0, "1":11, ...}
    }
    """
    if not label_map_path.exists():
        return None, None
    try:
        data = json.load(open(label_map_path, "r", encoding="utf-8"))
        id2idx = data.get("id2idx", None)
        idx2id = data.get("idx2id", None)
        # Asegurar tipos
        if isinstance(id2idx, dict):
            id2idx = {str(k): int(v) for k, v in id2idx.items()}
        if isinstance(idx2id, dict):
            idx2id = {str(k): int(v) for k, v in idx2id.items()}
        return id2idx, idx2id
    except Exception:
        return None, None


def infer_num_classes(data_dir: Path) -> int:
    lm = data_dir / "label_map.json"
    id2idx, idx2id = load_label_map(lm)
    if id2idx:
        return int(max(id2idx.values())) + 1
    # fallback: max label en Y_train
    y = np.load(data_dir / "Y_train.npz")["Y"]
    return int(y.max()) + 1


def compute_class_weights_from_json(artifacts_dir: Path, num_classes: int) -> Optional[np.ndarray]:
    cw_file = artifacts_dir / "class_weights.json"
    if not cw_file.exists():
        return None
    try:
        data = json.load(open(cw_file, "r", encoding="utf-8"))
        cw = data.get("class_weights", None)
        if not isinstance(cw, dict):
            return None
        w = np.ones((num_classes,), dtype=np.float32)
        for k, v in cw.items():
            try:
                w[int(k)] = float(v)
            except Exception:
                continue
        return w
    except Exception:
        return None


def normalize_unit_sphere(xyz: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    xyz: [N,3] o [B,N,3]
    - centra en media
    - escala por max norma
    """
    if xyz.dim() == 2:
        c = xyz.mean(dim=0, keepdim=True)
        x = xyz - c
        r = torch.norm(x, dim=1).max().clamp_min(eps)
        return x / r
    elif xyz.dim() == 3:
        c = xyz.mean(dim=1, keepdim=True)
        x = xyz - c
        r = torch.norm(x, dim=2).max(dim=1, keepdim=True).values.clamp_min(eps)  # [B,1]
        r = r.unsqueeze(-1)  # [B,1,1]
        return x / r
    else:
        raise ValueError("xyz debe ser [N,3] o [B,N,3]")


def plot_curves(history: Dict[str, List[float]], out_dir: Path, model_name: str):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for m in ["loss","acc","prec","rec","f1","iou","d21_acc","d21_f1","d21_iou"]:
        plt.figure(figsize=(7,4))
        for split in ["train","val"]:
            key = f"{split}_{m}"
            if key in history and len(history[key]) > 0:
                plt.plot(history[key], label=split)
        plt.xlabel("Época"); plt.ylabel(m.upper())
        plt.title(f"{model_name} – {m.upper()}"); plt.legend(); plt.tight_layout()
        plt.savefig(out_dir/f"{model_name}_{m}.png", dpi=300); plt.close()


# ==========================================
#                 DATASET
# ==========================================
class NPZPointSegDataset(Dataset):
    def __init__(self, x_path: Path, y_path: Path, normalize: bool = True):
        self.X = np.load(x_path)["X"].astype(np.float32)  # [B,N,3]
        self.Y = np.load(y_path)["Y"].astype(np.int64)    # [B,N]
        assert self.X.shape[0] == self.Y.shape[0], "X e Y deben tener mismo #muestras"
        self.normalize = normalize

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, i: int):
        xyz = torch.from_numpy(self.X[i])  # [N,3]
        y   = torch.from_numpy(self.Y[i])  # [N]
        if self.normalize:
            xyz = normalize_unit_sphere(xyz)
        return xyz, y


def make_loaders(data_dir: Path, batch_size: int, num_workers: int, normalize: bool = True):
    data_dir = Path(data_dir)
    ds_train = NPZPointSegDataset(data_dir/"X_train.npz", data_dir/"Y_train.npz", normalize=normalize)
    ds_val   = NPZPointSegDataset(data_dir/"X_val.npz",   data_dir/"Y_val.npz",   normalize=normalize)
    ds_test  = NPZPointSegDataset(data_dir/"X_test.npz",  data_dir/"Y_test.npz",  normalize=normalize)

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True, drop_last=False)
    dl_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)
    dl_test  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)
    return dl_train, dl_val, dl_test, ds_test


# ==========================================
#                 MÉTRICAS
# ==========================================
class MetricsBundle:
    def __init__(self, num_classes: int, device: torch.device, ignore_index: int = 0):
        self.num_classes = num_classes
        self.device = device
        self.ignore = ignore_index
        self.has_tm = HAS_TORCHMETRICS
        if self.has_tm:
            self._acc  = MulticlassAccuracy(num_classes=num_classes, average="macro", ignore_index=self.ignore).to(device)
            self._prec = MulticlassPrecision(num_classes=num_classes, average="macro", ignore_index=self.ignore).to(device)
            self._rec  = MulticlassRecall(num_classes=num_classes, average="macro", ignore_index=self.ignore).to(device)
            self._f1   = MulticlassF1Score(num_classes=num_classes, average="macro", ignore_index=self.ignore).to(device)
            self._iou  = MulticlassJaccardIndex(num_classes=num_classes, average="macro", ignore_index=self.ignore).to(device)
        self.reset_cm()

    def reset_cm(self):
        self.cm = torch.zeros((self.num_classes, self.num_classes), device=self.device, dtype=torch.long)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, y_true: torch.Tensor):
        # logits: [B,N,C]
        preds = logits.argmax(dim=-1)
        t = y_true.view(-1); p = preds.view(-1)
        valid = (t >= 0) & (t < self.num_classes)
        if self.ignore is not None:
            valid = valid & (t != self.ignore)
        t = t[valid]; p = p[valid]
        if t.numel() == 0:
            return
        idx = t * self.num_classes + p
        binc = torch.bincount(idx, minlength=self.num_classes**2).reshape(self.num_classes, self.num_classes)
        self.cm += binc.long()
        if self.has_tm:
            # torchmetrics espera (preds, target) como índices
            self._acc.update(p, t)
            self._prec.update(p, t)
            self._rec.update(p, t)
            self._f1.update(p, t)
            self._iou.update(p, t)

    def compute_macro(self) -> Dict[str, float]:
        if self.has_tm:
            return {
                "acc": float(self._acc.compute()),
                "prec": float(self._prec.compute()),
                "rec": float(self._rec.compute()),
                "f1": float(self._f1.compute()),
                "iou": float(self._iou.compute()),
            }
        # fallback por CM
        cm = self.cm.float()
        tp = torch.diag(cm)
        gt = cm.sum(1)
        pd = cm.sum(0)
        acc = torch.nan_to_num(tp.sum() / (cm.sum() + 1e-8)).item()
        prec = torch.nan_to_num(tp / (pd + 1e-8)).mean().item()
        rec = torch.nan_to_num(tp / (gt + 1e-8)).mean().item()
        f1 = torch.nan_to_num(2 * (tp / (pd + 1e-8)) * (tp / (gt + 1e-8)) / ((tp / (pd + 1e-8)) + (tp / (gt + 1e-8)) + 1e-8)).mean().item()
        iou = torch.nan_to_num(tp / (gt + pd - tp + 1e-8)).mean().item()
        return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "iou": iou}


def binary_metrics_for_class(pred: torch.Tensor, gt: torch.Tensor, cls: int, ignore_index: Optional[int] = 0) -> Dict[str, float]:
    """
    pred, gt: [B,N] (índices de clase)
    """
    t = gt.view(-1)
    p = pred.view(-1)
    valid = (t >= 0)
    if ignore_index is not None:
        valid = valid & (t != ignore_index)
    t = t[valid]; p = p[valid]
    if t.numel() == 0:
        return {"acc": 0.0, "f1": 0.0, "iou": 0.0}

    t_pos = (t == cls)
    p_pos = (p == cls)
    tp = (t_pos & p_pos).sum().item()
    fp = ((~t_pos) & p_pos).sum().item()
    fn = (t_pos & (~p_pos)).sum().item()
    tn = ((~t_pos) & (~p_pos)).sum().item()

    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    return {"acc": float(acc), "f1": float(f1), "iou": float(iou)}


# ==========================================
#              POINTNET CLÁSICO
# ==========================================
class STN3d(nn.Module):
    """T-Net clásico 3×3 (sin regularizador explícito)."""
    def __init__(self, k: int = 3):
        super().__init__()
        self.k = k
        self.conv1, self.bn1 = nn.Conv1d(k, 64, 1), nn.BatchNorm1d(64)
        self.conv2, self.bn2 = nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128)
        self.conv3, self.bn3 = nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024)
        self.fc1, self.bn4 = nn.Linear(1024, 512), nn.BatchNorm1d(512)
        self.fc2, self.bn5 = nn.Linear(512, 256), nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, k * k)

    def forward(self, x):
        # x: [B,k,N]
        B = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2)[0]
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x).view(B, self.k, self.k)
        iden = torch.eye(self.k, device=x.device).unsqueeze(0).repeat(B, 1, 1)
        return x + iden


class PointNetSeg(nn.Module):
    """PointNet de segmentación (Qi et al., 2017)."""
    def __init__(self, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.input_tnet = STN3d(k=3)
        self.conv1, self.bn1 = nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64)
        self.conv2, self.bn2 = nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128)
        self.conv3, self.bn3 = nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024)
        self.fconv1, self.bn4 = nn.Conv1d(1152, 512, 1), nn.BatchNorm1d(512)
        self.fconv2, self.bn5 = nn.Conv1d(512, 256, 1), nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(dropout)
        self.fconv3 = nn.Conv1d(256, num_classes, 1)

    def forward(self, xyz):
        # xyz: [B,N,3]
        B, P, _ = xyz.shape
        x = xyz.transpose(2, 1).contiguous()   # [B,3,N]
        T = self.input_tnet(x)                 # [B,3,3]
        x = torch.bmm(T, x)                    # [B,3,N]

        x1 = F.relu(self.bn1(self.conv1(x)))   # [B,64,N]
        x2 = F.relu(self.bn2(self.conv2(x1)))  # [B,128,N]
        x3 = F.relu(self.bn3(self.conv3(x2)))  # [B,1024,N]

        xg = torch.max(x3, 2, keepdim=True)[0].repeat(1, 1, P)  # [B,1024,N]
        x_cat = torch.cat([xg, x2], 1)                           # [B,1152,N]

        x = F.relu(self.bn4(self.fconv1(x_cat)))
        x = F.relu(self.bn5(self.fconv2(x)))
        x = self.dropout(x)
        return self.fconv3(x).transpose(2, 1).contiguous()       # [B,N,C]


# ==========================================
#          VISUALIZACIÓN INFERENCIA
# ==========================================
# Colores para IDs originales (si label_map permite revertir idx -> id).
LABEL_COLORS = {
    0: "#D62728", 11: "#1F77B4", 12: "#2CA02C", 13: "#FF7F0E", 14: "#9467BD", 15: "#17BECF",
    16: "#E377C2", 17: "#BCBD22", 18: "#8C564B", 21: "#98DF8A", 22: "#1F3A93", 23: "#008080",
    24: "#7F3C8D", 25: "#FA8072", 26: "#FFD700", 27: "#87CEFA", 28: "#FF7F50", 31: "#808000",
    32: "#C49C94", 33: "#AEC7E8", 34: "#FFBB78", 35: "#C5B0D5", 36: "#9EDAE5",
    37: "#F7B6D2", 38: "#DBDB8D", 41: "#393B79", 42: "#637939", 43: "#8C6D31",
    44: "#843C39", 45: "#7B4173", 46: "#5254A3", 47: "#6B6ECF", 48: "#9C9EDE",
}

def colors_for_labels(lbl_internal: np.ndarray, idx2id: Optional[dict], num_classes: int) -> np.ndarray:
    """
    Devuelve array [N,4] RGBA en [0,1] para cada punto.
    Si idx2id existe, mapea label interno -> ID original y usa LABEL_COLORS si está.
    """
    N = lbl_internal.shape[0]
    rgba = np.zeros((N, 4), dtype=np.float32)

    # fallback: colormap por índice interno
    cmap = plt.get_cmap("tab20", num_classes)

    for i in range(N):
        li = int(lbl_internal[i])
        if idx2id is not None and str(li) in idx2id:
            orig = int(idx2id[str(li)])
            hexcol = LABEL_COLORS.get(orig, None)
            if hexcol is not None:
                rgba[i, :3] = matplotlib.colors.to_rgb(hexcol)
                rgba[i, 3] = 1.0
                continue
        rgba[i, :] = cmap(li)
    return rgba


def plot_pointcloud_gt_pred(
    xyz: np.ndarray,
    y_gt: np.ndarray,
    y_pr: np.ndarray,
    out_png: Path,
    idx2id: Optional[dict],
    num_classes: int,
    title: str = ""
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    # colores
    c_gt = colors_for_labels(y_gt, idx2id, num_classes)
    c_pr = colors_for_labels(y_pr, idx2id, num_classes)

    fig = plt.figure(figsize=(10, 5), dpi=200)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    for ax, c, t in [(ax1, c_gt, "GT"), (ax2, c_pr, "Pred")]:
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=c, s=1.2, linewidths=0, depthshade=False)
        ax.set_title(t, fontsize=10)
        ax.set_axis_off()
        # vista consistente
        ax.view_init(elev=20, azim=45)

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# ==========================================
#                 TRAIN / EVAL
# ==========================================
def run_epoch(model, loader, criterion, optimizer, device, num_classes, ignore_index, d21_idx: Optional[int], train: bool):
    if train:
        model.train()
    else:
        model.eval()

    mb = MetricsBundle(num_classes=num_classes, device=device, ignore_index=ignore_index)
    loss_meter = 0.0
    n_batches = 0

    # acumuladores d21
    d21_acc = d21_f1 = d21_iou = None
    d21_count = 0

    for xyz, y in loader:
        xyz = xyz.to(device, non_blocking=True)  # [B,N,3]
        y = y.to(device, non_blocking=True)      # [B,N]

        if train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(xyz)  # [B,N,C]
        loss = criterion(logits.view(-1, num_classes), y.view(-1))

        if train:
            loss.backward()
            optimizer.step()

        loss_meter += float(loss.item())
        n_batches += 1

        mb.update(logits, y)

        if d21_idx is not None:
            pred = logits.argmax(dim=-1)
            dm = binary_metrics_for_class(pred, y, cls=d21_idx, ignore_index=ignore_index)
            # promedio simple por batch
            if d21_acc is None:
                d21_acc, d21_f1, d21_iou = dm["acc"], dm["f1"], dm["iou"]
            else:
                d21_acc += dm["acc"]; d21_f1 += dm["f1"]; d21_iou += dm["iou"]
            d21_count += 1

    macro = mb.compute_macro()
    out = {
        "loss": loss_meter / max(1, n_batches),
        **macro
    }
    if d21_idx is not None and d21_count > 0:
        out["d21_acc"] = float(d21_acc / d21_count)
        out["d21_f1"]  = float(d21_f1 / d21_count)
        out["d21_iou"] = float(d21_iou / d21_count)
    else:
        out["d21_acc"] = 0.0
        out["d21_f1"]  = 0.0
        out["d21_iou"] = 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True,
                    help="Carpeta con X_train.npz/Y_train.npz/X_val.npz/Y_val.npz/X_test.npz/Y_test.npz (y ojalá label_map.json).")
    ap.add_argument("--artifacts_dir", type=str, default=None,
                    help="Carpeta donde está class_weights.json (por defecto usa data_dir).")
    ap.add_argument("--out_dir", type=str, required=True, help="Salida de la corrida (runs/...)")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--ignore_index", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda", help="cuda | cpu")
    ap.add_argument("--infer_examples", type=int, default=10)
    ap.add_argument("--no_normalize", action="store_true", help="Desactiva normalización unit sphere por muestra.")
    args = ap.parse_args()

    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts_dir = Path(args.artifacts_dir) if args.artifacts_dir else data_dir

    # device
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    num_classes = infer_num_classes(data_dir)

    # label_map (para colorear y para encontrar diente 21 si existe)
    id2idx, idx2id = load_label_map(data_dir / "label_map.json")

    d21_idx = None
    if id2idx is not None:
        # intenta con "21" (FDI incisor sup izq)
        if "21" in id2idx:
            d21_idx = int(id2idx["21"])

    # loaders
    dl_train, dl_val, dl_test, ds_test = make_loaders(
        data_dir=data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalize=(not args.no_normalize),
    )

    # model
    model = PointNetSeg(num_classes=num_classes, dropout=args.dropout).to(device)

    # weights
    cw = compute_class_weights_from_json(artifacts_dir, num_classes)
    weight_tensor = None
    if cw is not None:
        weight_tensor = torch.from_numpy(cw).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor, ignore_index=args.ignore_index)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # logging
    run_meta = {
        "start_time": datetime.now().isoformat(timespec="seconds"),
        "data_dir": str(data_dir),
        "artifacts_dir": str(artifacts_dir),
        "out_dir": str(out_dir),
        "num_classes": num_classes,
        "device": str(device),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "ignore_index": args.ignore_index,
        "torchmetrics": HAS_TORCHMETRICS,
        "d21_internal_idx": d21_idx,
    }
    save_json(run_meta, out_dir / "run_meta.json")

    history: Dict[str, List[float]] = {}
    for k in ["train_loss","val_loss","train_acc","val_acc","train_prec","val_prec","train_rec","val_rec",
              "train_f1","val_f1","train_iou","val_iou","train_d21_acc","val_d21_acc","train_d21_f1","val_d21_f1","train_d21_iou","val_d21_iou"]:
        history[k] = []

    # CSV por época
    csv_path = out_dir / "metrics_epoch.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch","split","loss","acc","prec","rec","f1","iou","d21_acc","d21_f1","d21_iou","sec"])

    best_val_f1 = -1.0
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        e0 = time.time()

        tr = run_epoch(model, dl_train, criterion, optimizer, device, num_classes, args.ignore_index, d21_idx, train=True)
        va = run_epoch(model, dl_val,   criterion, optimizer, device, num_classes, args.ignore_index, d21_idx, train=False)

        # history
        history["train_loss"].append(tr["loss"]); history["val_loss"].append(va["loss"])
        history["train_acc"].append(tr["acc"]);  history["val_acc"].append(va["acc"])
        history["train_prec"].append(tr["prec"]);history["val_prec"].append(va["prec"])
        history["train_rec"].append(tr["rec"]);  history["val_rec"].append(va["rec"])
        history["train_f1"].append(tr["f1"]);    history["val_f1"].append(va["f1"])
        history["train_iou"].append(tr["iou"]);  history["val_iou"].append(va["iou"])
        history["train_d21_acc"].append(tr["d21_acc"]); history["val_d21_acc"].append(va["d21_acc"])
        history["train_d21_f1"].append(tr["d21_f1"]);   history["val_d21_f1"].append(va["d21_f1"])
        history["train_d21_iou"].append(tr["d21_iou"]); history["val_d21_iou"].append(va["d21_iou"])

        sec = time.time() - e0
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([epoch,"train",tr["loss"],tr["acc"],tr["prec"],tr["rec"],tr["f1"],tr["iou"],tr["d21_acc"],tr["d21_f1"],tr["d21_iou"],sec])
            w.writerow([epoch,"val",  va["loss"],va["acc"],va["prec"],va["rec"],va["f1"],va["iou"],va["d21_acc"],va["d21_f1"],va["d21_iou"],sec])

        # checkpoints
        torch.save({"model": model.state_dict(), "epoch": epoch, "val_f1": float(va["f1"])}, last_path)

        if float(va["f1"]) > best_val_f1:
            best_val_f1 = float(va["f1"])
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_f1": best_val_f1}, best_path)

        print(f"[{epoch:03d}/{args.epochs}] "
              f"train loss={tr['loss']:.4f} f1={tr['f1']:.4f} iou={tr['iou']:.4f} | "
              f"val loss={va['loss']:.4f} f1={va['f1']:.4f} iou={va['iou']:.4f} | "
              f"d21_f1={va['d21_f1']:.4f}")

    # guardar history + curvas
    save_json(history, out_dir / "history.json")
    plot_curves(history, out_dir / "plots", model_name="PointNetClassic")

    # cargar best y test
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    te = run_epoch(model, dl_test, criterion, optimizer, device, num_classes, args.ignore_index, d21_idx, train=False)
    save_json({"best_epoch": int(ckpt.get("epoch", -1)), "test": te}, out_dir / "test_metrics.json")

    # inferencia 10 ejemplos
    model.eval()
    inf_dir = out_dir / "inference"
    inf_dir.mkdir(parents=True, exist_ok=True)

    n = len(ds_test)
    k = min(args.infer_examples, n)
    indices = np.random.choice(n, size=k, replace=False)

    with torch.no_grad():
        for rank, i in enumerate(indices, start=1):
            xyz, y = ds_test[i]  # ya normalizado si corresponde
            xyz_b = xyz.unsqueeze(0).to(device)  # [1,N,3]
            logits = model(xyz_b)[0]             # [N,C]
            pred = logits.argmax(dim=-1).cpu().numpy().astype(np.int32)

            xyz_np = xyz.cpu().numpy()
            y_np = y.numpy().astype(np.int32)

            out_png = inf_dir / f"ex_{rank:02d}_idx_{i:05d}.png"
            plot_pointcloud_gt_pred(
                xyz=xyz_np,
                y_gt=y_np,
                y_pr=pred,
                out_png=out_png,
                idx2id=idx2id,
                num_classes=num_classes,
                title=f"PointNetClassic | test idx={i} | best_epoch={int(ckpt.get('epoch',-1))}"
            )

    total = time.time() - t0
    print(f"[DONE] out_dir={out_dir} | total_sec={total:.1f} | best_val_f1={best_val_f1:.4f}")


if __name__ == "__main__":
    main()
