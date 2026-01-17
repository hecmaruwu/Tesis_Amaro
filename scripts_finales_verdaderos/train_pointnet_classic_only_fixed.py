#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_pointnet_classic_only_fixed.py

PointNet clásico (segmentación punto-a-punto) + visualización enfocada en diente 21 (paper-quality).

- Carga X_*.npz / Y_*.npz (X: [B,N,3], Y: [B,N])
- Normalización por muestra (center + unit sphere)
- CrossEntropy con ignore_index (background=0) y class weights desde class_weights.json (si existe)
- Métricas macro (acc/prec/rec/f1/IoU) + métricas específicas para diente 21
- Guardado: history.json, metrics_epoch.csv, test_metrics.json, best.pt, last.pt
- Inferencia: K ejemplos aleatorios del test set:
    * d21 focus (binario + errores)  [RECOMENDADO]
    * multiclass (contexto)          [OPCIONAL]
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
import matplotlib.colors as mcolors


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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_label_map(label_map_path: Path) -> Tuple[Optional[dict], Optional[dict]]:
    """
    Espera:
    {
      "id2idx": {"0":0, "11":1, ...},
      "idx2id": {"0":0, "1":11, ...}
    }
    """
    if not label_map_path.exists():
        return None, None
    try:
        with open(label_map_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        id2idx = data.get("id2idx", None)
        idx2id = data.get("idx2id", None)

        if isinstance(id2idx, dict):
            id2idx = {str(k): int(v) for k, v in id2idx.items()}
        else:
            id2idx = None

        if isinstance(idx2id, dict):
            idx2id = {str(k): int(v) for k, v in idx2id.items()}
        else:
            idx2id = None

        return id2idx, idx2id
    except Exception:
        return None, None


def infer_num_classes(data_dir: Path) -> int:
    lm = data_dir / "label_map.json"
    id2idx, _ = load_label_map(lm)
    if id2idx:
        return int(max(id2idx.values())) + 1
    y = np.load(data_dir / "Y_train.npz")["Y"]
    return int(np.max(y)) + 1


def compute_class_weights_from_json(artifacts_dir: Path, num_classes: int) -> Optional[np.ndarray]:
    """
    Lee artifacts/class_weights.json con formato:
    {"class_weights": {"0": 1.0, "1": 2.3, ...}}
    """
    cw_file = artifacts_dir / "class_weights.json"
    if not cw_file.exists():
        return None

    try:
        with open(cw_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        cw = data.get("class_weights", None)
        if not isinstance(cw, dict):
            return None

        w = np.ones((num_classes,), dtype=np.float32)
        for k, v in cw.items():
            try:
                kk = int(k)
                if 0 <= kk < num_classes:
                    w[kk] = float(v)
            except Exception:
                continue

        # ndarray plain
        return np.asarray(w, dtype=np.float32).copy()
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
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for m in ["loss", "acc", "prec", "rec", "f1", "iou", "d21_acc", "d21_f1", "d21_iou"]:
        plt.figure(figsize=(7, 4))
        for split in ["train", "val"]:
            key = f"{split}_{m}"
            if key in history and len(history[key]) > 0:
                plt.plot(history[key], label=split)
        plt.xlabel("Época")
        plt.ylabel(m.upper())
        plt.title(f"{model_name} – {m.upper()}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{model_name}_{m}.png", dpi=300)
        plt.close()


# ==========================================
#                 DATASET
# ==========================================
class NPZPointSegDataset(Dataset):
    def __init__(self, x_path: Path, y_path: Path, normalize: bool = True):
        x_obj = np.load(x_path)
        y_obj = np.load(y_path)

        self.X = np.asarray(x_obj["X"], dtype=np.float32)   # [B,N,3]
        self.Y = np.asarray(y_obj["Y"], dtype=np.int64)     # [B,N]

        assert self.X.shape[0] == self.Y.shape[0], "X e Y deben tener mismo #muestras"
        assert self.X.ndim == 3 and self.X.shape[-1] == 3, f"X debe ser [B,N,3], got {self.X.shape}"
        assert self.Y.ndim == 2, f"Y debe ser [B,N], got {self.Y.shape}"
        assert self.X.shape[1] == self.Y.shape[1], f"N puntos no coincide: X {self.X.shape} vs Y {self.Y.shape}"

        self.normalize = normalize

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, i: int):
        xyz = torch.as_tensor(self.X[i], dtype=torch.float32)  # [N,3]
        y = torch.as_tensor(self.Y[i], dtype=torch.int64)      # [N]
        if self.normalize:
            xyz = normalize_unit_sphere(xyz)
        return xyz, y


def make_loaders(data_dir: Path, batch_size: int, num_workers: int, normalize: bool = True):
    data_dir = Path(data_dir)
    ds_train = NPZPointSegDataset(data_dir / "X_train.npz", data_dir / "Y_train.npz", normalize=normalize)
    ds_val   = NPZPointSegDataset(data_dir / "X_val.npz",   data_dir / "Y_val.npz",   normalize=normalize)
    ds_test  = NPZPointSegDataset(data_dir / "X_test.npz",  data_dir / "Y_test.npz",  normalize=normalize)

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True, drop_last=False)
    dl_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)
    dl_test  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)
    return dl_train, dl_val, dl_test, ds_test

# ==========================================
#                 MÉTRICAS
# ==========================================
class MetricsBundle:
    def __init__(self, num_classes: int, device: torch.device, ignore_index: Optional[int] = 0):
        self.num_classes = int(num_classes)
        self.device = device
        self.ignore = ignore_index
        self.has_tm = HAS_TORCHMETRICS

        if self.has_tm:
            self._acc  = MulticlassAccuracy(num_classes=self.num_classes, average="macro", ignore_index=self.ignore).to(device)
            self._prec = MulticlassPrecision(num_classes=self.num_classes, average="macro", ignore_index=self.ignore).to(device)
            self._rec  = MulticlassRecall(num_classes=self.num_classes, average="macro", ignore_index=self.ignore).to(device)
            self._f1   = MulticlassF1Score(num_classes=self.num_classes, average="macro", ignore_index=self.ignore).to(device)
            self._iou  = MulticlassJaccardIndex(num_classes=self.num_classes, average="macro", ignore_index=self.ignore).to(device)

        self.cm = torch.zeros((self.num_classes, self.num_classes), device=self.device, dtype=torch.long)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, y_true: torch.Tensor):
        preds = logits.argmax(dim=-1)
        t = y_true.reshape(-1)
        p = preds.reshape(-1)

        valid = (t >= 0) & (t < self.num_classes)
        if self.ignore is not None:
            valid = valid & (t != self.ignore)

        t = t[valid]
        p = p[valid]
        if t.numel() == 0:
            return

        idx = t * self.num_classes + p
        binc = torch.bincount(idx, minlength=self.num_classes * self.num_classes).reshape(self.num_classes, self.num_classes)
        self.cm += binc.long()

        if self.has_tm:
            self._acc.update(p, t)
            self._prec.update(p, t)
            self._rec.update(p, t)
            self._f1.update(p, t)
            self._iou.update(p, t)

    def compute_macro(self) -> Dict[str, float]:
        if self.has_tm:
            out = {
                "acc": float(self._acc.compute().item()),
                "prec": float(self._prec.compute().item()),
                "rec": float(self._rec.compute().item()),
                "f1": float(self._f1.compute().item()),
                "iou": float(self._iou.compute().item()),
            }
            self._acc.reset(); self._prec.reset(); self._rec.reset(); self._f1.reset(); self._iou.reset()
            return out

        cm = self.cm.float()
        tp = torch.diag(cm)
        gt = cm.sum(1)
        pd = cm.sum(0)

        acc = torch.nan_to_num(tp.sum() / (cm.sum() + 1e-8)).item()

        prec_c = torch.nan_to_num(tp / (pd + 1e-8))
        rec_c  = torch.nan_to_num(tp / (gt + 1e-8))
        f1_c   = torch.nan_to_num(2 * prec_c * rec_c / (prec_c + rec_c + 1e-8))
        iou_c  = torch.nan_to_num(tp / (gt + pd - tp + 1e-8))

        return {
            "acc": float(acc),
            "prec": float(prec_c.mean().item()),
            "rec": float(rec_c.mean().item()),
            "f1": float(f1_c.mean().item()),
            "iou": float(iou_c.mean().item()),
        }


def binary_metrics_for_class(pred: torch.Tensor, gt: torch.Tensor, cls: int, ignore_index: Optional[int] = 0) -> Dict[str, float]:
    t = gt.reshape(-1)
    p = pred.reshape(-1)

    valid = (t >= 0)
    if ignore_index is not None:
        valid = valid & (t != ignore_index)

    t = t[valid]
    p = p[valid]
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
# Colores por ID original (FDI) si idx2id existe.
# IMPORTANTE: aquí NO pintamos la encía de rojo; la dejamos gris claro para que no opaque.
LABEL_COLORS = {
    0: "#D9D9D9",  # encía/fondo -> gris claro
    11: "#1F77B4", 12: "#2CA02C", 13: "#FF7F0E", 14: "#9467BD", 15: "#17BECF",
    16: "#E377C2", 17: "#BCBD22", 18: "#8C564B",
    21: "#00FF00",  # diente 21 -> verde fuerte
    22: "#1F3A93", 23: "#008080", 24: "#7F3C8D", 25: "#FA8072", 26: "#FFD700",
    27: "#87CEFA", 28: "#FF7F50", 31: "#808000", 32: "#C49C94", 33: "#AEC7E8",
    34: "#FFBB78", 35: "#C5B0D5", 36: "#9EDAE5", 37: "#F7B6D2", 38: "#DBDB8D",
    41: "#393B79", 42: "#637939", 43: "#8C6D31", 44: "#843C39", 45: "#7B4173",
    46: "#5254A3", 47: "#6B6ECF", 48: "#9C9EDE",
}


def colors_for_labels_multiclass(lbl_internal: np.ndarray, idx2id: Optional[dict], num_classes: int) -> np.ndarray:
    """
    Multiclase “contexto”. Encía (0) en gris claro.
    """
    lbl_internal = np.asarray(lbl_internal, dtype=np.int32)
    N = int(lbl_internal.shape[0])
    rgba = np.zeros((N, 4), dtype=np.float32)

    cmap = plt.get_cmap("tab20", num_classes)

    if idx2id is None:
        for i in range(N):
            li = int(lbl_internal[i])
            rgba[i, :] = cmap(li)
            if li == 0:
                rgba[i, :3] = mcolors.to_rgb(LABEL_COLORS[0])
                rgba[i, 3] = 1.0
        return rgba

    for i in range(N):
        li = int(lbl_internal[i])
        orig = None
        if str(li) in idx2id:
            try:
                orig = int(idx2id[str(li)])
            except Exception:
                orig = None

        if orig is not None and orig in LABEL_COLORS:
            rgba[i, :3] = mcolors.to_rgb(LABEL_COLORS[orig])
            rgba[i, 3] = 1.0
        else:
            rgba[i, :] = cmap(li)

        # refuerzo encía si quedó como 0 interno
        if li == 0:
            rgba[i, :3] = mcolors.to_rgb(LABEL_COLORS[0])
            rgba[i, 3] = 1.0

    return rgba


def plot_pointcloud_gt_pred_multiclass(
    xyz: np.ndarray,
    y_gt: np.ndarray,
    y_pr: np.ndarray,
    out_png: Path,
    idx2id: Optional[dict],
    num_classes: int,
    title: str = ""
) -> None:
    """
    Multiclase (contexto). Útil, pero puede ser poco interpretable.
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)

    xyz = np.asarray(xyz, dtype=np.float32).copy()
    y_gt = np.asarray(y_gt, dtype=np.int32).copy()
    y_pr = np.asarray(y_pr, dtype=np.int32).copy()

    c_gt = np.asarray(colors_for_labels_multiclass(y_gt, idx2id, num_classes), dtype=np.float32).copy()
    c_pr = np.asarray(colors_for_labels_multiclass(y_pr, idx2id, num_classes), dtype=np.float32).copy()

    fig = plt.figure(figsize=(10, 5), dpi=200)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    for ax, c, t in [(ax1, c_gt, "GT"), (ax2, c_pr, "Pred")]:
        ax.scatter(xyz[:, 0].copy(), xyz[:, 1].copy(), xyz[:, 2].copy(), c=c, s=1.2, linewidths=0, depthshade=False)
        ax.set_title(t, fontsize=10)
        ax.set_axis_off()
        ax.view_init(elev=20, azim=45)

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_pointcloud_d21_focus(
    xyz: np.ndarray,
    y_gt: np.ndarray,
    y_pr: np.ndarray,
    out_png: Path,
    d21_idx: int,
    ignore_index: Optional[int] = 0,
    title: str = ""
) -> None:
    """
    Visualización recomendada para tu tesis:
    - Gris: no-21 (y no ignore)
    - Verde: TP (GT=21 y Pred=21)
    - Rojo: errores (FP o FN)
    - (Opcional): ignore_index en azul muy claro para contexto (si ignore_index != None)
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)

    xyz = np.asarray(xyz, dtype=np.float32).copy()
    y_gt = np.asarray(y_gt, dtype=np.int32).copy()
    y_pr = np.asarray(y_pr, dtype=np.int32).copy()

    N = xyz.shape[0]
    rgba = np.zeros((N, 4), dtype=np.float32)

    # Colores
    col_bg = mcolors.to_rgb("#BFBFBF")   # gris
    col_tp = mcolors.to_rgb("#00C853")   # verde fuerte
    col_err = mcolors.to_rgb("#D50000")  # rojo fuerte
    col_ign = mcolors.to_rgb("#9EC9FF")  # celeste suave

    # Máscaras
    is_ign = (y_gt == ignore_index) if (ignore_index is not None) else np.zeros(N, dtype=bool)
    gt21 = (y_gt == d21_idx) & (~is_ign)
    pr21 = (y_pr == d21_idx) & (~is_ign)

    tp = gt21 & pr21
    fp = (~gt21) & pr21
    fn = gt21 & (~pr21)
    err = fp | fn

    # Default: gris (no-21)
    rgba[:, :3] = col_bg
    rgba[:, 3] = 1.0

    # ignore
    if ignore_index is not None:
        rgba[is_ign, :3] = col_ign
        rgba[is_ign, 3] = 0.8

    # TP en verde
    rgba[tp, :3] = col_tp
    rgba[tp, 3] = 1.0

    # errores en rojo
    rgba[err, :3] = col_err
    rgba[err, 3] = 1.0

    # Plot lado a lado: GT21 vs Pred21 (binario) + overlay error
    fig = plt.figure(figsize=(10, 5), dpi=200)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    # Panel 1: GT highlight (solo GT21 en verde, resto gris)
    rgba_gt = rgba.copy()
    # En GT panel: marcamos FN también como verde (porque pertenecen al GT21)
    rgba_gt[gt21, :3] = col_tp
    rgba_gt[gt21, 3] = 1.0
    # errores no se muestran rojos aquí; el rojo lo dejamos para el panel Pred/Errors

    # Panel 2: Pred+Errors (TP verde, FP/FN rojo)
    rgba_pr = rgba

    for ax, c, t in [(ax1, rgba_gt, "GT (d21)"), (ax2, rgba_pr, "Pred + errors")]:
        ax.scatter(xyz[:, 0].copy(), xyz[:, 1].copy(), xyz[:, 2].copy(), c=c, s=1.2, linewidths=0, depthshade=False)
        ax.set_title(t, fontsize=10)
        ax.set_axis_off()
        ax.view_init(elev=20, azim=45)

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

# ==========================================
#                 TRAIN / EVAL
# ==========================================
def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    num_classes: int,
    ignore_index: Optional[int],
    d21_idx: Optional[int],
    train: bool
):
    model.train() if train else model.eval()

    mb = MetricsBundle(num_classes=num_classes, device=device, ignore_index=ignore_index)
    loss_meter = 0.0
    n_batches = 0

    d21_acc = d21_f1 = d21_iou = 0.0
    d21_count = 0

    for xyz, y in loader:
        xyz = xyz.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # Defensa extra: labels en rango (evita CUDA assert raro)
        if torch.any((y < 0) | (y >= num_classes)):
            bad_min = int(y.min().item())
            bad_max = int(y.max().item())
            raise ValueError(f"[LABEL ERROR] y fuera de rango: min={bad_min}, max={bad_max}, num_classes={num_classes}")

        if train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(xyz)  # [B,N,C]
        loss = criterion(logits.reshape(-1, num_classes), y.reshape(-1))

        if train:
            loss.backward()
            optimizer.step()

        loss_meter += float(loss.item())
        n_batches += 1

        mb.update(logits, y)

        if d21_idx is not None:
            pred = logits.argmax(dim=-1)
            dm = binary_metrics_for_class(pred, y, cls=d21_idx, ignore_index=ignore_index)
            d21_acc += dm["acc"]
            d21_f1 += dm["f1"]
            d21_iou += dm["iou"]
            d21_count += 1

    macro = mb.compute_macro()
    out = {"loss": loss_meter / max(1, n_batches), **macro}

    if d21_idx is not None and d21_count > 0:
        out["d21_acc"] = float(d21_acc / d21_count)
        out["d21_f1"] = float(d21_f1 / d21_count)
        out["d21_iou"] = float(d21_iou / d21_count)
    else:
        out["d21_acc"] = 0.0
        out["d21_f1"] = 0.0
        out["d21_iou"] = 0.0

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--artifacts_dir", type=str, default=None)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.5)

    ap.add_argument("--ignore_index", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--infer_examples", type=int, default=10)
    ap.add_argument("--infer_mode", type=str, default="d21", choices=["d21", "multiclass", "both"],
                    help="d21: solo visualización enfocada en diente 21 (recomendado). "
                         "multiclass: solo GT vs Pred multiclase. both: genera ambos.")
    ap.add_argument("--no_normalize", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = Path(args.artifacts_dir) if args.artifacts_dir else data_dir

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    num_classes = infer_num_classes(data_dir)
    id2idx, idx2id = load_label_map(data_dir / "label_map.json")

    d21_idx = None
    if id2idx is not None and "21" in id2idx:
        d21_idx = int(id2idx["21"])

    dl_train, dl_val, dl_test, ds_test = make_loaders(
        data_dir=data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalize=(not args.no_normalize),
    )

    model = PointNetSeg(num_classes=num_classes, dropout=args.dropout).to(device)

    cw = compute_class_weights_from_json(artifacts_dir, num_classes)
    weight_tensor = torch.as_tensor(cw, dtype=torch.float32, device=device) if cw is not None else None

    criterion = nn.CrossEntropyLoss(weight=weight_tensor, ignore_index=args.ignore_index)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_meta = {
        "start_time": datetime.now().isoformat(timespec="seconds"),
        "data_dir": str(data_dir),
        "artifacts_dir": str(artifacts_dir),
        "out_dir": str(out_dir),
        "num_classes": int(num_classes),
        "device": str(device),
        "seed": int(args.seed),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "dropout": float(args.dropout),
        "ignore_index": int(args.ignore_index),
        "torchmetrics": bool(HAS_TORCHMETRICS),
        "d21_internal_idx": (int(d21_idx) if d21_idx is not None else None),
        "normalize_unit_sphere": bool(not args.no_normalize),
        "infer_mode": args.infer_mode,
        "infer_examples": int(args.infer_examples),
    }
    save_json(run_meta, out_dir / "run_meta.json")

    history: Dict[str, List[float]] = {k: [] for k in [
        "train_loss","val_loss",
        "train_acc","val_acc",
        "train_prec","val_prec",
        "train_rec","val_rec",
        "train_f1","val_f1",
        "train_iou","val_iou",
        "train_d21_acc","val_d21_acc",
        "train_d21_f1","val_d21_f1",
        "train_d21_iou","val_d21_iou"
    ]}

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

        torch.save({"model": model.state_dict(), "epoch": epoch, "val_f1": float(va["f1"])}, last_path)
        if float(va["f1"]) > best_val_f1:
            best_val_f1 = float(va["f1"])
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_f1": best_val_f1}, best_path)

        print(f"[{epoch:03d}/{args.epochs}] "
              f"train loss={tr['loss']:.4f} f1={tr['f1']:.4f} iou={tr['iou']:.4f} | "
              f"val loss={va['loss']:.4f} f1={va['f1']:.4f} iou={va['iou']:.4f} | "
              f"d21_f1={va['d21_f1']:.4f}")

    save_json(history, out_dir / "history.json")
    plot_curves(history, out_dir / "plots", model_name="PointNetClassic")

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    te = run_epoch(model, dl_test, criterion, optimizer, device, num_classes, args.ignore_index, d21_idx, train=False)
    save_json({"best_epoch": int(ckpt.get("epoch", -1)), "test": te}, out_dir / "test_metrics.json")

    # Inferencia
    if args.infer_examples > 0:
        model.eval()
        n = len(ds_test)
        k = min(int(args.infer_examples), int(n))
        indices = np.random.choice(n, size=k, replace=False)

        if args.infer_mode in ["multiclass", "both"]:
            out_mc = out_dir / "inference_multiclass"
            out_mc.mkdir(parents=True, exist_ok=True)

        if args.infer_mode in ["d21", "both"]:
            out_d21 = out_dir / "inference_d21"
            out_d21.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            for rank, i in enumerate(indices, start=1):
                xyz, y = ds_test[int(i)]
                xyz_b = xyz.unsqueeze(0).to(device)
                logits = model(xyz_b)[0]
                pred = logits.argmax(dim=-1).detach().cpu().numpy().astype(np.int32)

                xyz_np = xyz.detach().cpu().numpy().astype(np.float32, copy=True)
                y_np = y.detach().cpu().numpy().astype(np.int32, copy=True)

                best_epoch = int(ckpt.get("epoch", -1))
                base_name = f"ex_{rank:02d}_idx_{int(i):05d}.png"

                if args.infer_mode in ["multiclass", "both"]:
                    plot_pointcloud_gt_pred_multiclass(
                        xyz=xyz_np,
                        y_gt=y_np,
                        y_pr=pred,
                        out_png=(out_mc / base_name),
                        idx2id=idx2id,
                        num_classes=num_classes,
                        title=f"PointNetClassic | test idx={int(i)} | best_epoch={best_epoch}"
                    )

                if args.infer_mode in ["d21", "both"] and d21_idx is not None:
                    plot_pointcloud_d21_focus(
                        xyz=xyz_np,
                        y_gt=y_np,
                        y_pr=pred,
                        out_png=(out_d21 / base_name),
                        d21_idx=int(d21_idx),
                        ignore_index=args.ignore_index,
                        title=f"PointNetClassic D21 | test idx={int(i)} | best_epoch={best_epoch}"
                    )

    total = time.time() - t0
    print(f"[DONE] out_dir={out_dir} | total_sec={total:.1f} | best_val_f1={best_val_f1:.4f}")


if __name__ == "__main__":
    main()

