#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_pointnet_classic.py

PointNet clásico para segmentación punto-a-punto (NPZ) + métricas robustas + inferencia paper-like.

SUPUESTOS DE DATASET (tu pipeline actual):
- data_dir/
    X_train.npz, Y_train.npz
    X_val.npz,   Y_val.npz
    X_test.npz,  Y_test.npz
    artifacts/label_map.json          (opcional)
    artifacts/class_weights.json      (opcional)
- X: float32 [B,N,3]
- Y: int64   [B,N] con clases internas 0..C-1
- ignore_index=0 = encía/fondo (bg)

CAMBIOS CLAVE vs bug anterior:
- acc_all REAL incluye bg (no ignora 0)
- acc_no_bg excluye ignore_index
- macro metrics (prec/rec/f1/iou/acc_macro) excluyen ignore_index
- métricas d21 por clase (IoU/F1/Acc) visibles DURANTE entrenamiento
- inferencia:
    (A) d21 focus (GT vs Pred + errores)
    (B) multiclass (GT vs Pred con colores para TODAS las clases del entrenamiento)
- sanity checks fuertes: rangos, num_classes global, distribuciones, label_map coherente, d21 presente, baseline bg

Uso típico:
python3 train_pointnet_classic.py \
  --data_dir /home/htaucare/Tesis_Amaro/data/Teeth_3ds/fixed_split/8192/upper_only_surf_global_excl_wisdom_seed42_aug2 \
  --out_dir  /home/htaucare/Tesis_Amaro/outputs/pointnet_classic/upper_only_seed42_aug2 \
  --epochs 120 --batch_size 8 --lr 1e-3 --weight_decay 1e-4 --dropout 0.5 \
  --ignore_index 0 --num_workers 4 --device cuda \
  --d21_internal 8 \
  --do_infer --infer_examples 12
"""

import os
import json
import csv
import time
import argparse
import random
from dataclasses import dataclass
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
        data = json.loads(label_map_path.read_text(encoding="utf-8"))
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


def find_label_map(data_dir: Path) -> Path:
    p1 = data_dir / "artifacts" / "label_map.json"
    if p1.exists():
        return p1
    return data_dir / "label_map.json"


def infer_num_classes_from_splits(data_dir: Path) -> int:
    """
    num_classes global = max(Y_train, Y_val, Y_test)+1
    (esto evita el bug típico de inferir solo con Y_train)
    """
    mx = -1
    for s in ["train", "val", "test"]:
        y = np.load(data_dir / f"Y_{s}.npz")["Y"]
        mx = max(mx, int(y.max()))
    return int(mx) + 1


def compute_class_weights_from_json(artifacts_dir: Path, num_classes: int) -> Optional[np.ndarray]:
    candidates = [
        artifacts_dir / "class_weights.json",
        artifacts_dir / "artifacts" / "class_weights.json",
    ]
    cw_file = None
    for p in candidates:
        if p.exists():
            cw_file = p
            break
    if cw_file is None:
        return None

    try:
        data = json.loads(cw_file.read_text(encoding="utf-8"))
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

        return w.astype(np.float32, copy=True)
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
    raise ValueError("xyz debe ser [N,3] o [B,N,3]")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _safe_div(a: float, b: float) -> float:
    return float(a / (b + 1e-8))


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
#          MÉTRICAS (macro + sanity)
# ==========================================
class MetricsBundle:
    """
    - acc_all   : overall accuracy incluyendo bg (NO ignora ignore_index)
    - acc_no_bg : accuracy excluyendo ignore_index
    - macro: acc_macro/prec_macro/rec_macro/f1_macro/iou_macro excluyendo ignore_index
    - confusion matrix excluyendo ignore_index (para per-class y fallback)
    """
    def __init__(self, num_classes: int, device: torch.device, ignore_index: Optional[int] = 0):
        self.num_classes = int(num_classes)
        self.device = device
        self.ignore = ignore_index
        self.has_tm = HAS_TORCHMETRICS

        self.correct_all = 0
        self.total_all = 0
        self.correct_no_bg = 0
        self.total_no_bg = 0

        self.cm = torch.zeros((self.num_classes, self.num_classes), device=self.device, dtype=torch.long)

        if self.has_tm:
            self._acc_macro  = MulticlassAccuracy(num_classes=self.num_classes, average="macro", ignore_index=self.ignore).to(device)
            self._prec_macro = MulticlassPrecision(num_classes=self.num_classes, average="macro", ignore_index=self.ignore).to(device)
            self._rec_macro  = MulticlassRecall(num_classes=self.num_classes, average="macro", ignore_index=self.ignore).to(device)
            self._f1_macro   = MulticlassF1Score(num_classes=self.num_classes, average="macro", ignore_index=self.ignore).to(device)
            self._iou_macro  = MulticlassJaccardIndex(num_classes=self.num_classes, average="macro", ignore_index=self.ignore).to(device)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, y_true: torch.Tensor):
        preds = logits.argmax(dim=-1)  # [B,N]

        t_all = y_true.reshape(-1)
        p_all = preds.reshape(-1)

        self.correct_all += int((p_all == t_all).sum().item())
        self.total_all += int(t_all.numel())

        valid = (t_all >= 0) & (t_all < self.num_classes)
        if self.ignore is not None:
            valid_no_bg = valid & (t_all != self.ignore)
        else:
            valid_no_bg = valid

        t = t_all[valid_no_bg]
        p = p_all[valid_no_bg]

        if t.numel() > 0:
            self.correct_no_bg += int((p == t).sum().item())
            self.total_no_bg += int(t.numel())

            idx = t * self.num_classes + p
            binc = torch.bincount(idx, minlength=self.num_classes * self.num_classes).reshape(self.num_classes, self.num_classes)
            self.cm += binc.long()

            if self.has_tm:
                self._acc_macro.update(p, t)
                self._prec_macro.update(p, t)
                self._rec_macro.update(p, t)
                self._f1_macro.update(p, t)
                self._iou_macro.update(p, t)

    def compute_macro(self) -> Dict[str, float]:
        acc_all = self.correct_all / max(1, self.total_all)
        acc_no_bg = self.correct_no_bg / max(1, self.total_no_bg) if self.total_no_bg > 0 else 0.0

        if self.has_tm:
            out = {
                "acc_all": float(acc_all),
                "acc_no_bg": float(acc_no_bg),
                "acc_macro": float(self._acc_macro.compute().item()),
                "prec_macro": float(self._prec_macro.compute().item()),
                "rec_macro": float(self._rec_macro.compute().item()),
                "f1_macro": float(self._f1_macro.compute().item()),
                "iou_macro": float(self._iou_macro.compute().item()),
            }
            self._acc_macro.reset()
            self._prec_macro.reset()
            self._rec_macro.reset()
            self._f1_macro.reset()
            self._iou_macro.reset()
            return out

        cm = self.cm.float()
        tp = torch.diag(cm)
        gt = cm.sum(1)
        pd = cm.sum(0)

        prec_c = torch.nan_to_num(tp / (pd + 1e-8))
        rec_c  = torch.nan_to_num(tp / (gt + 1e-8))
        f1_c   = torch.nan_to_num(2 * prec_c * rec_c / (prec_c + rec_c + 1e-8))
        iou_c  = torch.nan_to_num(tp / (gt + pd - tp + 1e-8))

        # balanced accuracy ~ mean recall
        acc_macro = float(rec_c.mean().item())

        return {
            "acc_all": float(acc_all),
            "acc_no_bg": float(acc_no_bg),
            "acc_macro": float(acc_macro),
            "prec_macro": float(prec_c.mean().item()),
            "rec_macro": float(rec_c.mean().item()),
            "f1_macro": float(f1_c.mean().item()),
            "iou_macro": float(iou_c.mean().item()),
        }

    def per_class_from_cm(self, cls: int) -> Dict[str, float]:
        """
        Per-class metrics (sobre cm no-bg):
        precision/recall/f1/iou + "cls_acc" (recall)
        """
        cls = int(cls)
        if not (0 <= cls < self.num_classes):
            return {"prec": 0.0, "rec": 0.0, "f1": 0.0, "iou": 0.0, "cls_acc": 0.0}

        cm = self.cm.float()
        tp = float(cm[cls, cls].item())
        gt = float(cm[cls, :].sum().item())
        pd = float(cm[:, cls].sum().item())

        prec = _safe_div(tp, pd)
        rec  = _safe_div(tp, gt)  # = acc de esa clase (recall)
        f1 = _safe_div(2 * prec * rec, (prec + rec))
        iou = _safe_div(tp, (gt + pd - tp))
        return {"prec": prec, "rec": rec, "f1": f1, "iou": iou, "cls_acc": rec}


def binary_metrics_for_class(
    pred: torch.Tensor,
    gt: torch.Tensor,
    cls: int,
    ignore_index: Optional[int],
    exclude_ignore: bool
) -> Dict[str, float]:
    """
    Métricas binarias d21 vs no-d21
    - exclude_ignore=True  -> excluye bg
    - exclude_ignore=False -> incluye todo (incluye bg)
    """
    t = gt.reshape(-1)
    p = pred.reshape(-1)
    valid = (t >= 0)
    if exclude_ignore and (ignore_index is not None):
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

    acc = _safe_div(tp + tn, (tp + tn + fp + fn))
    prec = _safe_div(tp, (tp + fp))
    rec = _safe_div(tp, (tp + fn))
    f1 = _safe_div(2 * prec * rec, (prec + rec))
    iou = _safe_div(tp, (tp + fp + fn))
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
#                 PLOTS
# ==========================================
def plot_curves(history: Dict[str, List[float]], out_dir: Path, model_name: str):
    ensure_dir(out_dir)
    metrics = [
        "loss",
        "acc_all", "acc_no_bg",
        "acc_macro", "prec_macro", "rec_macro", "f1_macro", "iou_macro",
        "d21_cls_iou", "d21_cls_f1", "d21_cls_acc",
        "d21_bin_acc_all", "d21_bin_f1", "d21_bin_iou",
    ]
    for m in metrics:
        plt.figure(figsize=(7, 4))
        for split in ["train", "val"]:
            k = f"{split}_{m}"
            if k in history and len(history[k]) > 0:
                plt.plot(history[k], label=split)
        plt.xlabel("Época")
        plt.ylabel(m)
        plt.title(f"{model_name} – {m}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{model_name}_{m}.png", dpi=300)
        plt.close()


def _distinct_colors_hex(n: int) -> List[str]:
    # colores distintivos usando HSV (determinístico)
    cols = []
    for i in range(n):
        h = (i / max(1, n)) % 1.0
        s = 0.75
        v = 0.95
        rgb = mcolors.hsv_to_rgb((h, s, v))
        cols.append(mcolors.to_hex(rgb))
    return cols


def build_class_color_map(num_classes: int, ignore_index: int = 0) -> Dict[int, str]:
    """
    0 (bg) gris, resto colores vivos.
    """
    cmap = {int(ignore_index): "#BFBFBF"}
    cols = _distinct_colors_hex(num_classes)
    for c in range(num_classes):
        if c == ignore_index:
            continue
        cmap[c] = cols[c]
    return cmap


def plot_pointcloud_multiclass_gt_pred(
    xyz: np.ndarray,
    y_gt: np.ndarray,
    y_pr: np.ndarray,
    out_png: Path,
    class_colors: Dict[int, str],
    ignore_index: Optional[int],
    title: str
) -> None:
    """
    Panel 2x: GT (multiclase) vs Pred (multiclase).
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)

    xyz = np.asarray(xyz, dtype=np.float32)
    y_gt = np.asarray(y_gt, dtype=np.int32)
    y_pr = np.asarray(y_pr, dtype=np.int32)

    def colors_for_labels(y: np.ndarray) -> np.ndarray:
        rgba = np.zeros((y.shape[0], 4), dtype=np.float32)
        for c in np.unique(y):
            hexc = class_colors.get(int(c), "#000000")
            rgb = np.array(mcolors.to_rgb(hexc), dtype=np.float32)
            mask = (y == c)
            rgba[mask, :3] = rgb
            rgba[mask, 3] = 1.0
        # opcional: ignorados azul clarito
        if ignore_index is not None:
            ign = (y == ignore_index)
            rgba[ign, :3] = np.array(mcolors.to_rgb("#9EC9FF"), dtype=np.float32)
            rgba[ign, 3] = 0.85
        return rgba

    c_gt = colors_for_labels(y_gt)
    c_pr = colors_for_labels(y_pr)

    fig = plt.figure(figsize=(10, 5), dpi=200)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    for ax, c, t in [(ax1, c_gt, "GT (multiclass)"), (ax2, c_pr, "Pred (multiclass)")]:
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=c, s=1.1, linewidths=0, depthshade=False)
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
    ignore_index: Optional[int],
    title: str
) -> None:
    """
    Panel 2x: GT(d21) vs Pred + errores(d21)
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)

    xyz = np.asarray(xyz, dtype=np.float32)
    y_gt = np.asarray(y_gt, dtype=np.int32)
    y_pr = np.asarray(y_pr, dtype=np.int32)

    N = xyz.shape[0]
    rgba = np.zeros((N, 4), dtype=np.float32)

    col_bg  = mcolors.to_rgb("#BFBFBF")
    col_tp  = mcolors.to_rgb("#00C853")
    col_err = mcolors.to_rgb("#D50000")
    col_ign = mcolors.to_rgb("#9EC9FF")

    is_ign = (y_gt == ignore_index) if (ignore_index is not None) else np.zeros(N, dtype=bool)
    gt21 = (y_gt == d21_idx) & (~is_ign)
    pr21 = (y_pr == d21_idx) & (~is_ign)

    tp = gt21 & pr21
    fp = (~gt21) & pr21
    fn = gt21 & (~pr21)
    err = fp | fn

    rgba[:, :3] = col_bg
    rgba[:, 3] = 1.0

    if ignore_index is not None:
        rgba[is_ign, :3] = col_ign
        rgba[is_ign, 3] = 0.85

    rgba[tp, :3] = col_tp
    rgba[tp, 3] = 1.0
    rgba[err, :3] = col_err
    rgba[err, 3] = 1.0

    fig = plt.figure(figsize=(10, 5), dpi=200)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    rgba_gt = np.zeros((N, 4), dtype=np.float32)
    rgba_gt[:, :3] = col_bg
    rgba_gt[:, 3] = 1.0
    if ignore_index is not None:
        rgba_gt[is_ign, :3] = col_ign
        rgba_gt[is_ign, 3] = 0.85
    rgba_gt[gt21, :3] = col_tp
    rgba_gt[gt21, 3] = 1.0

    for ax, c, t in [(ax1, rgba_gt, "GT (d21)"), (ax2, rgba, "Pred + errors (d21)")]:
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=c, s=1.1, linewidths=0, depthshade=False)
        ax.set_title(t, fontsize=10)
        ax.set_axis_off()
        ax.view_init(elev=20, azim=45)

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# ==========================================
#           d21 RESOLUCIÓN + CHECKS
# ==========================================
def resolve_d21_internal_idx(
    id2idx: Optional[dict],
    idx2id: Optional[dict],
    num_classes: int,
    forced: Optional[int]
) -> Optional[int]:
    if forced is not None:
        if 0 <= forced < num_classes:
            return int(forced)
        raise ValueError(f"--d21_internal fuera de rango: {forced} (num_classes={num_classes})")

    if id2idx is not None and "21" in id2idx:
        v = int(id2idx["21"])
        if 0 <= v < num_classes:
            return v

    if idx2id is not None:
        for k, v in idx2id.items():
            try:
                if int(v) == 21:
                    kk = int(k)
                    if 0 <= kk < num_classes:
                        return kk
            except Exception:
                continue

    return None


def dataset_sanity_checks(
    data_dir: Path,
    num_classes: int,
    ignore_index: int,
    d21_idx: Optional[int],
    out_dir: Path
) -> None:
    """
    Checks que te dicen si “realmente está bien”:
    - rangos de etiquetas por split
    - num_classes global coherente
    - distribución de clases + bg%
    - d21 presente por split (si d21_idx != None)
    - baseline trivial: always predict bg => acc_all ~= bg%
    """
    report = {"splits": {}, "num_classes": int(num_classes), "ignore_index": int(ignore_index), "d21_idx": d21_idx}

    for s in ["train", "val", "test"]:
        y = np.load(data_dir / f"Y_{s}.npz")["Y"].reshape(-1)
        mn = int(y.min())
        mx = int(y.max())
        if mn < 0 or mx >= num_classes:
            raise ValueError(f"[SANITY] Split {s}: labels fuera de rango: min={mn}, max={mx}, num_classes={num_classes}")

        counts = np.bincount(y.astype(np.int64), minlength=num_classes)
        tot = int(y.size)
        bg = int(counts[ignore_index])
        bg_frac = float(bg / max(1, tot))
        uniq = int((counts > 0).sum())

        entry = {
            "min": mn, "max": mx, "total_points": tot,
            "unique_classes_present": uniq,
            "bg_points": bg,
            "bg_frac": bg_frac,
            "baseline_bg_acc_all": bg_frac,  # siempre predecir bg
        }

        if d21_idx is not None:
            d21_pts = int(counts[d21_idx])
            entry["d21_points"] = d21_pts
            entry["d21_frac"] = float(d21_pts / max(1, tot))

        report["splits"][s] = entry

    save_json(report, out_dir / "sanity_checks.json")
    print(f"[SANITY] Guardado: {out_dir/'sanity_checks.json'}")


# ==========================================
#                 TRAIN / EVAL
# ==========================================
@dataclass
class EpochOut:
    loss: float
    acc_all: float
    acc_no_bg: float
    acc_macro: float
    prec_macro: float
    rec_macro: float
    f1_macro: float
    iou_macro: float
    # d21 por clase (multiclase, desde CM no-bg)
    d21_cls_acc: float
    d21_cls_f1: float
    d21_cls_iou: float
    # d21 binario
    d21_bin_acc_all: float
    d21_bin_f1: float
    d21_bin_iou: float


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
) -> EpochOut:
    model.train() if train else model.eval()

    mb = MetricsBundle(num_classes=num_classes, device=device, ignore_index=ignore_index)
    loss_meter = 0.0
    n_batches = 0

    # binario d21 promediado por batch
    d21_bin_acc_all = 0.0
    d21_bin_f1 = 0.0
    d21_bin_iou = 0.0
    d21_bin_count = 0

    for xyz, y in loader:
        xyz = xyz.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if torch.any((y < 0) | (y >= num_classes)):
            raise ValueError(f"[LABEL ERROR] y fuera de rango: min={int(y.min())}, max={int(y.max())}, num_classes={num_classes}")

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
            dm_all = binary_metrics_for_class(pred, y, cls=d21_idx, ignore_index=ignore_index, exclude_ignore=False)
            d21_bin_acc_all += dm_all["acc"]
            d21_bin_f1 += dm_all["f1"]
            d21_bin_iou += dm_all["iou"]
            d21_bin_count += 1

    macro = mb.compute_macro()

    # d21 “por clase” (multiclase): desde la CM (no-bg)
    if d21_idx is not None:
        pc = mb.per_class_from_cm(d21_idx)
        d21_cls_acc = float(pc["cls_acc"])
        d21_cls_f1  = float(pc["f1"])
        d21_cls_iou = float(pc["iou"])
    else:
        d21_cls_acc = d21_cls_f1 = d21_cls_iou = 0.0

    if d21_bin_count > 0:
        d21_bin_acc_all /= d21_bin_count
        d21_bin_f1 /= d21_bin_count
        d21_bin_iou /= d21_bin_count
    else:
        d21_bin_acc_all = d21_bin_f1 = d21_bin_iou = 0.0

    return EpochOut(
        loss=loss_meter / max(1, n_batches),
        acc_all=float(macro["acc_all"]),
        acc_no_bg=float(macro["acc_no_bg"]),
        acc_macro=float(macro["acc_macro"]),
        prec_macro=float(macro["prec_macro"]),
        rec_macro=float(macro["rec_macro"]),
        f1_macro=float(macro["f1_macro"]),
        iou_macro=float(macro["iou_macro"]),
        d21_cls_acc=d21_cls_acc,
        d21_cls_f1=d21_cls_f1,
        d21_cls_iou=d21_cls_iou,
        d21_bin_acc_all=float(d21_bin_acc_all),
        d21_bin_f1=float(d21_bin_f1),
        d21_bin_iou=float(d21_bin_iou),
    )


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

    ap.add_argument("--no_normalize", action="store_true")

    # d21
    ap.add_argument("--d21_internal", type=int, default=None,
                    help="Indice interno (0..C-1) del diente 21. Ej: 8 para upper-only sin wisdom en tu setup.")

    # inferencia
    ap.add_argument("--do_infer", action="store_true")
    ap.add_argument("--infer_examples", type=int, default=12)

    args = ap.parse_args()

    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    artifacts_dir = Path(args.artifacts_dir) if args.artifacts_dir else data_dir

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    # num_classes GLOBAL (robusto)
    num_classes = infer_num_classes_from_splits(data_dir)

    lm_path = find_label_map(data_dir)
    id2idx, idx2id = load_label_map(lm_path)
    d21_idx = resolve_d21_internal_idx(id2idx, idx2id, num_classes=num_classes, forced=args.d21_internal)

    # sanity checks antes de entrenar
    dataset_sanity_checks(
        data_dir=data_dir,
        num_classes=num_classes,
        ignore_index=args.ignore_index,
        d21_idx=d21_idx,
        out_dir=out_dir
    )

    dl_train, dl_val, dl_test, ds_test = make_loaders(
        data_dir=data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalize=(not args.no_normalize),
    )

    model = PointNetSeg(num_classes=num_classes, dropout=args.dropout).to(device)

    cw = compute_class_weights_from_json(artifacts_dir, num_classes)
    if cw is not None and len(cw) != num_classes:
        raise ValueError(f"class_weights len mismatch: {len(cw)} vs num_classes={num_classes}")
    weight_tensor = torch.as_tensor(cw, dtype=torch.float32, device=device) if cw is not None else None

    criterion = nn.CrossEntropyLoss(weight=weight_tensor, ignore_index=args.ignore_index)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_meta = {
        "start_time": datetime.now().isoformat(timespec="seconds"),
        "data_dir": str(data_dir),
        "artifacts_dir": str(artifacts_dir),
        "out_dir": str(out_dir),
        "num_classes": int(num_classes),
        "label_map_path": str(lm_path),
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
        "do_infer": bool(args.do_infer),
        "infer_examples": int(args.infer_examples),
    }
    save_json(run_meta, out_dir / "run_meta.json")

    # history
    keys = [
        "loss",
        "acc_all", "acc_no_bg",
        "acc_macro", "prec_macro", "rec_macro", "f1_macro", "iou_macro",
        "d21_cls_acc", "d21_cls_f1", "d21_cls_iou",
        "d21_bin_acc_all", "d21_bin_f1", "d21_bin_iou",
    ]
    history: Dict[str, List[float]] = {}
    for split in ["train", "val"]:
        for k in keys:
            history[f"{split}_{k}"] = []

    csv_path = out_dir / "metrics_epoch.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "split"] + keys + ["sec"])

    best_val_f1 = -1.0
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        e0 = time.time()

        tr = run_epoch(model, dl_train, criterion, optimizer, device, num_classes, args.ignore_index, d21_idx, train=True)
        va = run_epoch(model, dl_val,   criterion, optimizer, device, num_classes, args.ignore_index, d21_idx, train=False)

        for k in keys:
            history[f"train_{k}"].append(float(getattr(tr, k)))
            history[f"val_{k}"].append(float(getattr(va, k)))

        sec = time.time() - e0
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([epoch, "train"] + [float(getattr(tr, k)) for k in keys] + [sec])
            w.writerow([epoch, "val"]   + [float(getattr(va, k)) for k in keys] + [sec])

        torch.save({"model": model.state_dict(), "epoch": epoch, "val_f1_macro": float(va.f1_macro)}, last_path)
        if float(va.f1_macro) > best_val_f1:
            best_val_f1 = float(va.f1_macro)
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_f1_macro": best_val_f1}, best_path)

        # print con lo que pediste: IoU/Acc/F1 de clase 21 (multiclase) + macro general
        print(
            f"[{epoch:03d}/{args.epochs}] "
            f"train loss={tr.loss:.4f} f1m={tr.f1_macro:.4f} ioum={tr.iou_macro:.4f} acc_no_bg={tr.acc_no_bg:.4f} acc_all={tr.acc_all:.4f} | "
            f"val loss={va.loss:.4f} f1m={va.f1_macro:.4f} ioum={va.iou_macro:.4f} acc_no_bg={va.acc_no_bg:.4f} acc_all={va.acc_all:.4f} | "
            f"d21(cls) acc={va.d21_cls_acc:.4f} f1={va.d21_cls_f1:.4f} iou={va.d21_cls_iou:.4f} | "
            f"d21(bin all) acc={va.d21_bin_acc_all:.4f}"
        )

    save_json(history, out_dir / "history.json")
    plot_curves(history, out_dir / "plots", model_name="PointNetClassic")

    # Test con best
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    te = run_epoch(model, dl_test, criterion, optimizer, device, num_classes, args.ignore_index, d21_idx, train=False)
    save_json({"best_epoch": int(ckpt.get("epoch", -1)), "test": te.__dict__}, out_dir / "test_metrics.json")

    # Inferencia (multiclase + d21)
    if args.do_infer and args.infer_examples > 0:
        model.eval()
        n = len(ds_test)
        k = min(int(args.infer_examples), int(n))
        indices = np.random.choice(n, size=k, replace=False)

        class_colors = build_class_color_map(num_classes=num_classes, ignore_index=args.ignore_index)

        out_mc = out_dir / "inference_multiclass"
        out_d21 = out_dir / "inference_d21"
        ensure_dir(out_mc)
        ensure_dir(out_d21)

        with torch.no_grad():
            for rank, i in enumerate(indices, start=1):
                xyz, y = ds_test[int(i)]
                xyz_b = xyz.unsqueeze(0).to(device)
                logits = model(xyz_b)[0]  # [N,C]
                pred = logits.argmax(dim=-1).detach().cpu().numpy().astype(np.int32)

                xyz_np = xyz.detach().cpu().numpy().astype(np.float32, copy=True)
                y_np = y.detach().cpu().numpy().astype(np.int32, copy=True)

                best_epoch = int(ckpt.get("epoch", -1))

                # multiclass
                plot_pointcloud_multiclass_gt_pred(
                    xyz=xyz_np,
                    y_gt=y_np,
                    y_pr=pred,
                    out_png=(out_mc / f"ex_{rank:02d}_idx_{int(i):05d}.png"),
                    class_colors=class_colors,
                    ignore_index=args.ignore_index,
                    title=f"PointNetClassic multiclass | test idx={int(i)} | best_epoch={best_epoch}"
                )

                # d21 focus (si existe)
                if d21_idx is not None:
                    plot_pointcloud_d21_focus(
                        xyz=xyz_np,
                        y_gt=y_np,
                        y_pr=pred,
                        out_png=(out_d21 / f"ex_{rank:02d}_idx_{int(i):05d}.png"),
                        d21_idx=int(d21_idx),
                        ignore_index=args.ignore_index,
                        title=f"PointNetClassic d21_idx={int(d21_idx)} | test idx={int(i)} | best_epoch={best_epoch}"
                    )

    total = time.time() - t0
    print(f"[DONE] out_dir={out_dir} | total_sec={total:.1f} | best_val_f1_macro={best_val_f1:.4f} | d21_idx={d21_idx}")


if __name__ == "__main__":
    main()
