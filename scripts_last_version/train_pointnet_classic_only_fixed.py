#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_pointnet_classic_only_fixed.py

PointNet clásico – Segmentación multiclase dental 3D (OPCIÓN A – PAPER CORRECTA)

✅ BG incluido en la loss (NO ignore en la loss)
✅ BG excluido SOLO en métricas macro (f1/iou/prec/rec/acc_no_bg)
✅ Métricas diente 21 explícitas (acc/f1/iou) de forma BINARIA correcta
✅ Métrica "d21_bin_acc_all" (incluye TODO, incluso bg) para referencia
✅ Estabilidad: bg downweight, weight_decay, grad clipping, CosineAnnealingLR
✅ RTX 3090 friendly: AMP, pin_memory, persistent_workers, non_blocking, cudnn.benchmark
✅ Inferencia: PNGs 3D (GT vs Pred) + errores + foco d21
✅ TRAZABILIDAD (FIX): busca index_{split}.csv en:
   1) data_dir/index_{split}.csv
   2) ancestros de data_dir (hasta Teeth_3ds)
   3) Teeth_3ds/merged_*/index_{split}.csv (elige el más reciente por mtime)
   y si lo encuentra, añade sample_name/jaw/path a título y nombre de archivo,
   y guarda inference_manifest.csv con el mapeo row_i -> paciente.

Dataset esperado:
  data_dir/X_train.npz, Y_train.npz, X_val.npz, Y_val.npz, X_test.npz, Y_test.npz
  X: [B,N,3], Y: [B,N] con clases internas 0..C-1 (0=bg)

Ejemplo:
python3 train_pointnet_classic_only_fixed.py \
  --data_dir .../upper_only_surf_global_excl_wisdom_seed42_aug2 \
  --out_dir  .../outputs/pointnet_classic/run1 \
  --epochs 120 --batch_size 16 --lr 2e-4 --weight_decay 1e-4 --dropout 0.5 \
  --num_workers 6 --device cuda --d21_internal 8 \
  --bg_weight 0.03 --grad_clip 1.0 --use_amp \
  --do_infer --infer_examples 12
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
# DATASET
# ============================================================
class NPZDataset(Dataset):
    def __init__(self, Xp: Path, Yp: Path, normalize: bool = True):
        self.X = np.load(Xp)["X"].astype(np.float32)  # [B,N,3]
        self.Y = np.load(Yp)["Y"].astype(np.int64)    # [B,N]
        assert self.X.ndim == 3 and self.X.shape[-1] == 3, f"X shape inesperada: {self.X.shape}"
        assert self.Y.ndim == 2, f"Y shape inesperada: {self.Y.shape}"
        assert self.X.shape[0] == self.Y.shape[0], "B mismatch"
        assert self.X.shape[1] == self.Y.shape[1], "N mismatch"
        self.normalize = bool(normalize)

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, i):
        xyz = torch.from_numpy(self.X[i])  # [N,3]
        y = torch.from_numpy(self.Y[i])    # [N]
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
# POINTNET (paper-like)
# ============================================================
class STN3d(nn.Module):
    def __init__(self, k: int = 3):
        super().__init__()
        self.k = int(k)
        self.conv1, self.bn1 = nn.Conv1d(self.k, 64, 1), nn.BatchNorm1d(64)
        self.conv2, self.bn2 = nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128)
        self.conv3, self.bn3 = nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024)
        self.fc1, self.bn4 = nn.Linear(1024, 512), nn.BatchNorm1d(512)
        self.fc2, self.bn5 = nn.Linear(512, 256), nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, self.k * self.k)

    def forward(self, x):
        # x: [B,k,N]
        B = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2)[0]  # [B,1024]
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x).view(B, self.k, self.k)
        iden = torch.eye(self.k, device=x.device).unsqueeze(0).repeat(B, 1, 1)
        return x + iden


class PointNetSeg(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.stn = STN3d(k=3)

        self.conv1, self.bn1 = nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64)
        self.conv2, self.bn2 = nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128)
        self.conv3, self.bn3 = nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024)

        # concat global(1024) + local(128) = 1152
        self.fconv1, self.fbn1 = nn.Conv1d(1152, 512, 1), nn.BatchNorm1d(512)
        self.fconv2, self.fbn2 = nn.Conv1d(512, 256, 1), nn.BatchNorm1d(256)
        self.drop = nn.Dropout(float(dropout))
        self.fconv3 = nn.Conv1d(256, int(num_classes), 1)

    def forward(self, xyz):
        # xyz: [B,N,3]
        B, N, _ = xyz.shape
        x = xyz.transpose(2, 1).contiguous()    # [B,3,N]
        T = self.stn(x)                         # [B,3,3]
        x = torch.bmm(T, x)                     # [B,3,N]

        x1 = F.relu(self.bn1(self.conv1(x)))    # [B,64,N]
        x2 = F.relu(self.bn2(self.conv2(x1)))   # [B,128,N]
        x3 = F.relu(self.bn3(self.conv3(x2)))   # [B,1024,N]

        g = torch.max(x3, 2, keepdim=True)[0].repeat(1, 1, N)  # [B,1024,N]
        cat = torch.cat([g, x2], dim=1)                        # [B,1152,N]

        x = F.relu(self.fbn1(self.fconv1(cat)))
        x = F.relu(self.fbn2(self.fconv2(x)))
        x = self.drop(x)
        logits = self.fconv3(x).transpose(2, 1).contiguous()   # [B,N,C]
        return logits


# ============================================================
# MÉTRICAS (macro sin bg)
# ============================================================
@torch.no_grad()
def macro_metrics_no_bg(pred: torch.Tensor, gt: torch.Tensor, C: int, bg: int = 0) -> Tuple[float, float]:
    pred = pred.reshape(-1)
    gt = gt.reshape(-1)
    mask = (gt != bg)
    pred = pred[mask]
    gt = gt[mask]
    if gt.numel() == 0:
        return 0.0, 0.0

    f1s = []
    ious = []
    for c in range(1, int(C)):
        tp = ((pred == c) & (gt == c)).sum().item()
        fp = ((pred == c) & (gt != c)).sum().item()
        fn = ((pred != c) & (gt == c)).sum().item()
        denom = (tp + fp + fn)
        if denom == 0:
            continue
        f1 = (2 * tp) / (2 * tp + fp + fn + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        f1s.append(f1)
        ious.append(iou)

    if len(f1s) == 0:
        return 0.0, 0.0
    return float(np.mean(f1s)), float(np.mean(ious))


@torch.no_grad()
def d21_metrics_binary(pred: torch.Tensor, gt: torch.Tensor, d21_idx: int, bg: int = 0, include_bg: bool = False):
    pred = pred.reshape(-1)
    gt = gt.reshape(-1)

    if not include_bg:
        mask = (gt != bg)
        pred = pred[mask]
        gt = gt[mask]
        if gt.numel() == 0:
            return 0.0, 0.0, 0.0

    t_pos = (gt == d21_idx)
    p_pos = (pred == d21_idx)

    tp = (p_pos & t_pos).sum().item()
    fp = (p_pos & (~t_pos)).sum().item()
    fn = ((~p_pos) & t_pos).sum().item()
    tn = ((~p_pos) & (~t_pos)).sum().item()

    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    f1 = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    return float(acc), float(f1), float(iou)


# ============================================================
# VISUALIZACIÓN
# ============================================================
def _class_colors(C: int):
    cmap = plt.colormaps.get_cmap("tab20")
    C = max(int(C), 2)
    cols = [cmap(i / max(C - 1, 1)) for i in range(C)]
    return cols


def plot_pointcloud_all_classes(
    xyz: np.ndarray,
    y_gt: np.ndarray,
    y_pr: np.ndarray,
    out_png: Path,
    C: int,
    title: str = "",
    s: float = 1.0
):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cols = _class_colors(C)

    fig = plt.figure(figsize=(12, 5), dpi=200)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    c_gt = np.array([cols[int(k)] for k in y_gt], dtype=np.float32)
    c_pr = np.array([cols[int(k)] for k in y_pr], dtype=np.float32)

    ax1.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=c_gt, s=s, linewidths=0, depthshade=False)
    ax2.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=c_pr, s=s, linewidths=0, depthshade=False)

    ax1.set_title("GT (todas las clases)", fontsize=10)
    ax2.set_title("Pred (todas las clases)", fontsize=10)
    for ax in (ax1, ax2):
        ax.set_axis_off()
        ax.view_init(elev=20, azim=45)

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_errors(
    xyz: np.ndarray,
    y_gt: np.ndarray,
    y_pr: np.ndarray,
    out_png: Path,
    bg: int = 0,
    title: str = "",
    s: float = 1.0
):
    out_png.parent.mkdir(parents=True, exist_ok=True)

    ok = (y_gt == y_pr)
    c = np.zeros((xyz.shape[0], 4), dtype=np.float32)
    c[:, :] = (0.75, 0.75, 0.75, 1.0)
    c[~ok, :] = (0.85, 0.10, 0.10, 1.0)
    c[y_gt == bg, :] = (0.85, 0.85, 0.85, 0.6)

    fig = plt.figure(figsize=(6, 5), dpi=200)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=c, s=s, linewidths=0, depthshade=False)
    ax.set_axis_off()
    ax.view_init(elev=20, azim=45)
    ax.set_title("Errores (rojo) | Correcto (gris)", fontsize=10)
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_d21_focus(
    xyz: np.ndarray,
    y_gt: np.ndarray,
    y_pr: np.ndarray,
    out_png: Path,
    d21_idx: int,
    bg: int = 0,
    title: str = "",
    s: float = 1.2
):
    out_png.parent.mkdir(parents=True, exist_ok=True)

    gt21 = (y_gt == d21_idx)
    pr21 = (y_pr == d21_idx)
    tp = gt21 & pr21
    fp = (~gt21) & pr21
    fn = gt21 & (~pr21)
    err = fp | fn

    c = np.zeros((xyz.shape[0], 4), dtype=np.float32)
    c[:, :] = (0.75, 0.75, 0.75, 1.0)
    c[y_gt == bg, :] = (0.85, 0.85, 0.85, 0.6)

    c[tp, :] = (0.10, 0.75, 0.25, 1.0)   # verde
    c[err, :] = (0.85, 0.10, 0.10, 1.0)  # rojo

    fig = plt.figure(figsize=(6, 5), dpi=200)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=c, s=s, linewidths=0, depthshade=False)
    ax.set_axis_off()
    ax.view_init(elev=20, azim=45)
    ax.set_title("Foco d21: TP (verde) | FP/FN (rojo)", fontsize=10)
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# TRAIN / EVAL
# ============================================================
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    loss_fn: nn.Module,
    C: int,
    d21_idx: int,
    device: torch.device,
    bg: int,
    train: bool,
    use_amp: bool,
    grad_clip: Optional[float] = None,
) -> Dict[str, float]:
    model.train(train)
    scaler = run_epoch.scaler  # type: ignore
    if use_amp and scaler is None:
        scaler = torch.cuda.amp.GradScaler()
        run_epoch.scaler = scaler  # type: ignore

    loss_sum = 0.0
    acc_all_sum = 0.0
    acc_no_bg_sum = 0.0
    f1m_sum = 0.0
    ioum_sum = 0.0

    d21_acc_sum = 0.0
    d21_f1_sum = 0.0
    d21_iou_sum = 0.0
    d21_bin_all_sum = 0.0

    pred_bg_frac_sum = 0.0
    n_batches = 0

    for xyz, y in loader:
        xyz = xyz.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(xyz)  # [B,N,C]
            loss = loss_fn(logits.reshape(-1, C), y.reshape(-1))

        if train:
            if use_amp:
                scaler.scale(loss).backward()
                if grad_clip is not None and grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                optimizer.step()

        pred = logits.argmax(dim=-1)  # [B,N]

        acc_all = (pred == y).float().mean().item()

        mask = (y != bg)
        if mask.any():
            acc_no_bg = (pred[mask] == y[mask]).float().mean().item()
        else:
            acc_no_bg = 0.0

        f1m, ioum = macro_metrics_no_bg(pred, y, C=C, bg=bg)

        d21_acc, d21_f1, d21_iou = d21_metrics_binary(pred, y, d21_idx=d21_idx, bg=bg, include_bg=False)
        d21_bin_acc_all, _, _ = d21_metrics_binary(pred, y, d21_idx=d21_idx, bg=bg, include_bg=True)

        pred_bg_frac = (pred.reshape(-1) == bg).float().mean().item()

        loss_sum += float(loss.item())
        acc_all_sum += acc_all
        acc_no_bg_sum += acc_no_bg
        f1m_sum += f1m
        ioum_sum += ioum

        d21_acc_sum += d21_acc
        d21_f1_sum += d21_f1
        d21_iou_sum += d21_iou
        d21_bin_all_sum += d21_bin_acc_all

        pred_bg_frac_sum += pred_bg_frac
        n_batches += 1

    n = max(1, n_batches)
    return {
        "loss": loss_sum / n,
        "acc_all": acc_all_sum / n,
        "acc_no_bg": acc_no_bg_sum / n,
        "f1_macro": f1m_sum / n,
        "iou_macro": ioum_sum / n,
        "d21_acc": d21_acc_sum / n,
        "d21_f1": d21_f1_sum / n,
        "d21_iou": d21_iou_sum / n,
        "d21_bin_acc_all": d21_bin_all_sum / n,
        "pred_bg_frac": pred_bg_frac_sum / n,
    }


run_epoch.scaler = None  # type: ignore


# ============================================================
# TRAZABILIDAD: index_*.csv discovery + parse
# ============================================================
def _read_index_csv(path: Path) -> Optional[Dict[int, Dict[str, str]]]:
    """
    Devuelve dict: row_i -> {"sample_name":..., "jaw":..., "path":..., "idx_global":..., "has_labels":...}
    row_i = posición (fila) en el split, o sea df.iloc[row_i].
    """
    if not path.exists():
        return None
    try:
        import pandas as pd
        df = pd.read_csv(path)
    except Exception:
        return None

    out: Dict[int, Dict[str, str]] = {}
    for i in range(len(df)):
        row = df.iloc[i]
        out[int(i)] = {
            "idx_global": str(row.get("idx", "")),
            "sample_name": str(row.get("sample_name", "")),
            "jaw": str(row.get("jaw", "")),
            "path": str(row.get("path", "")),
            "has_labels": str(row.get("has_labels", "")),
        }
    return out


def _find_teeth3ds_root(start: Path) -> Optional[Path]:
    for p in [start] + list(start.parents):
        if p.name == "Teeth_3ds":
            return p
    return None


def _discover_index_csv(data_dir: Path, split: str) -> Optional[Path]:
    """
    Busca index_{split}.csv en:
      1) data_dir/index_{split}.csv
      2) ancestros de data_dir (hasta Teeth_3ds)
      3) Teeth_3ds/merged_*/index_{split}.csv (elige más reciente por mtime)
    """
    fname = f"index_{split}.csv"

    p1 = data_dir / fname
    if p1.exists():
        return p1

    # 2) ancestros (útil si data_dir está dentro de algo que sí tiene index_*.csv)
    for p in data_dir.parents:
        cand = p / fname
        if cand.exists():
            return cand
        if p.name == "Teeth_3ds":
            break

    # 3) buscar en merged_* dentro del root Teeth_3ds
    root = _find_teeth3ds_root(data_dir)
    if root is None:
        return None

    cands = []
    for d in root.glob("merged_*"):
        cand = d / fname
        if cand.exists():
            cands.append(cand)

    if not cands:
        return None

    # elegir el más reciente (mtime)
    cands.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return cands[0]


def _sanitize_tag(s: str) -> str:
    s = str(s).strip()
    if not s:
        return ""
    ok = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_"):
            ok.append(ch)
        else:
            ok.append("_")
    return "".join(ok)


# ============================================================
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.5)

    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    ap.add_argument("--d21_internal", type=int, required=True)

    # Opción A (bg incluido en loss) -> controlamos desbalance con pesos
    ap.add_argument("--bg_index", type=int, default=0)
    ap.add_argument("--bg_weight", type=float, default=0.10,
                    help="Peso de la clase bg en CE (menor = penaliza menos bg).")

    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--use_amp", action="store_true")

    ap.add_argument("--no_normalize", action="store_true")

    # Inferencia
    ap.add_argument("--do_infer", action="store_true")
    ap.add_argument("--infer_examples", type=int, default=12)
    ap.add_argument("--infer_split", type=str, default="test", choices=["test", "val", "train"],
                    help="Split usado para inferencia (para trazabilidad con index_*.csv).")

    args = ap.parse_args()

    set_seed(args.seed)

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------- Sanity dataset ----------
    Ytr = np.load(data_dir / "Y_train.npz")["Y"].reshape(-1)
    Yva = np.load(data_dir / "Y_val.npz")["Y"].reshape(-1)
    Yte = np.load(data_dir / "Y_test.npz")["Y"].reshape(-1)

    bg = int(args.bg_index)
    bg_tr = float((Ytr == bg).mean())
    bg_va = float((Yva == bg).mean())
    bg_te = float((Yte == bg).mean())

    C = int(max(Ytr.max(), Yva.max(), Yte.max())) + 1

    print(f"[SANITY] num_classes C = {C}")
    print(f"[SANITY] bg_frac train/val/test = {bg_tr:.4f} {bg_va:.4f} {bg_te:.4f}")
    print(f"[SANITY] baseline acc_all (always-bg) train/val/test = {bg_tr:.4f} {bg_va:.4f} {bg_te:.4f}")

    if not (0 <= args.d21_internal < C):
        raise ValueError(f"d21_internal fuera de rango: {args.d21_internal} (C={C})")

    # loaders
    dl_tr, dl_va, dl_te, ds_te = make_loaders(
        data_dir=data_dir,
        bs=args.batch_size,
        nw=args.num_workers,
        normalize=(not args.no_normalize),
    )

    # model
    model = PointNetSeg(num_classes=C, dropout=float(args.dropout)).to(device)

    # loss weights (Opción A: bg incluido)
    w = torch.ones(C, device=device, dtype=torch.float32)
    w[bg] = float(args.bg_weight)
    loss_fn = nn.CrossEntropyLoss(weight=w)

    # optimizer + scheduler
    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(args.epochs), eta_min=1e-6)

    # logging meta
    run_meta = {
        "start_time": datetime.now().isoformat(timespec="seconds"),
        "data_dir": str(data_dir),
        "out_dir": str(out_dir),
        "device": str(device),
        "seed": int(args.seed),
        "num_classes": int(C),
        "bg_index": int(bg),
        "bg_weight": float(args.bg_weight),
        "d21_internal": int(args.d21_internal),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "dropout": float(args.dropout),
        "grad_clip": float(args.grad_clip),
        "use_amp": bool(args.use_amp),
        "normalize_unit_sphere": bool(not args.no_normalize),
        "infer_examples": int(args.infer_examples),
        "do_infer": bool(args.do_infer),
        "infer_split": str(args.infer_split),
    }
    save_json(run_meta, out_dir / "run_meta.json")

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
            "lr",
            "sec"
        ])

    best_val_f1 = -1.0
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"

    history: Dict[str, List[float]] = {k: [] for k in [
        "train_loss", "val_loss",
        "train_acc_all", "val_acc_all",
        "train_acc_no_bg", "val_acc_no_bg",
        "train_f1m", "val_f1m",
        "train_ioum", "val_ioum",
        "val_d21_acc", "val_d21_f1", "val_d21_iou",
        "val_d21_bin_acc_all",
        "train_pred_bg_frac", "val_pred_bg_frac",
    ]}

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        e0 = time.time()

        tr = run_epoch(
            model=model, loader=dl_tr, optimizer=opt, loss_fn=loss_fn, C=C,
            d21_idx=args.d21_internal, device=device, bg=bg, train=True,
            use_amp=bool(args.use_amp), grad_clip=float(args.grad_clip),
        )
        va = run_epoch(
            model=model, loader=dl_va, optimizer=None, loss_fn=loss_fn, C=C,
            d21_idx=args.d21_internal, device=device, bg=bg, train=False,
            use_amp=False, grad_clip=None,
        )

        sched.step()
        lr_now = float(opt.param_groups[0]["lr"])
        sec = time.time() - e0

        history["train_loss"].append(tr["loss"]); history["val_loss"].append(va["loss"])
        history["train_acc_all"].append(tr["acc_all"]); history["val_acc_all"].append(va["acc_all"])
        history["train_acc_no_bg"].append(tr["acc_no_bg"]); history["val_acc_no_bg"].append(va["acc_no_bg"])
        history["train_f1m"].append(tr["f1_macro"]); history["val_f1m"].append(va["f1_macro"])
        history["train_ioum"].append(tr["iou_macro"]); history["val_ioum"].append(va["iou_macro"])
        history["val_d21_acc"].append(va["d21_acc"]); history["val_d21_f1"].append(va["d21_f1"]); history["val_d21_iou"].append(va["d21_iou"])
        history["val_d21_bin_acc_all"].append(va["d21_bin_acc_all"])
        history["train_pred_bg_frac"].append(tr["pred_bg_frac"]); history["val_pred_bg_frac"].append(va["pred_bg_frac"])

        # warning colapso bg
        if va["pred_bg_frac"] > max(0.95, bg_va + 0.12):
            print(f"[WARN] posible colapso a BG: val pred_bg_frac={va['pred_bg_frac']:.3f} (bg_gt≈{bg_va:.3f})")

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            wcsv = csv.writer(f)
            wcsv.writerow([epoch, "train",
                           tr["loss"], tr["acc_all"], tr["acc_no_bg"], tr["f1_macro"], tr["iou_macro"],
                           tr["d21_acc"], tr["d21_f1"], tr["d21_iou"], tr["d21_bin_acc_all"],
                           tr["pred_bg_frac"], lr_now, sec])
            wcsv.writerow([epoch, "val",
                           va["loss"], va["acc_all"], va["acc_no_bg"], va["f1_macro"], va["iou_macro"],
                           va["d21_acc"], va["d21_f1"], va["d21_iou"], va["d21_bin_acc_all"],
                           va["pred_bg_frac"], lr_now, sec])

        torch.save({"model": model.state_dict(), "epoch": epoch, "val_f1_macro": float(va["f1_macro"])}, last_path)
        if float(va["f1_macro"]) > best_val_f1:
            best_val_f1 = float(va["f1_macro"])
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_f1_macro": best_val_f1}, best_path)

        print(
            f"[{epoch:03d}/{args.epochs}] "
            f"train loss={tr['loss']:.4f} f1m={tr['f1_macro']:.3f} ioum={tr['iou_macro']:.3f} "
            f"acc_all={tr['acc_all']:.3f} acc_no_bg={tr['acc_no_bg']:.3f} | "
            f"val loss={va['loss']:.4f} f1m={va['f1_macro']:.3f} ioum={va['iou_macro']:.3f} "
            f"acc_all={va['acc_all']:.3f} acc_no_bg={va['acc_no_bg']:.3f} | "
            f"d21(cls) acc={va['d21_acc']:.3f} f1={va['d21_f1']:.3f} iou={va['d21_iou']:.3f} | "
            f"d21(bin all) acc={va['d21_bin_acc_all']:.3f} | "
            f"pred_bg_frac(val)={va['pred_bg_frac']:.3f} lr={lr_now:.2e}"
        )

    save_json(history, out_dir / "history.json")

    # plots
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    def _plot_curve(keys, name):
        plt.figure(figsize=(7, 4))
        for k in keys:
            plt.plot(history[k], label=k)
        plt.xlabel("epoch")
        plt.ylabel(name)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / f"{name}.png", dpi=250)
        plt.close()

    _plot_curve(["train_f1m", "val_f1m"], "f1_macro_no_bg")
    _plot_curve(["train_ioum", "val_ioum"], "iou_macro_no_bg")
    _plot_curve(["train_acc_no_bg", "val_acc_no_bg"], "acc_no_bg")
    _plot_curve(["train_acc_all", "val_acc_all"], "acc_all")
    _plot_curve(["val_d21_f1"], "d21_f1")
    _plot_curve(["val_pred_bg_frac"], "val_pred_bg_frac")

    # Test con best
    ckpt = torch.load(best_path, map_location=device)
    ckpt_epoch = int(ckpt.get("epoch", -1))
    model.load_state_dict(ckpt["model"])
    te = run_epoch(
        model=model, loader=dl_te, optimizer=None, loss_fn=loss_fn, C=C,
        d21_idx=args.d21_internal, device=device, bg=bg, train=False,
        use_amp=False, grad_clip=None,
    )
    save_json({"best_epoch": ckpt_epoch, "test": te}, out_dir / "test_metrics.json")

    # Inferencia / visualización + trazabilidad (FIX integrado)
    if args.do_infer and args.infer_examples > 0:
        model.eval()

        split = str(args.infer_split)
        if split == "test":
            ds_inf = ds_te
        elif split == "val":
            ds_inf = NPZDataset(data_dir / "X_val.npz", data_dir / "Y_val.npz", normalize=(not args.no_normalize))
        else:
            ds_inf = NPZDataset(data_dir / "X_train.npz", data_dir / "Y_train.npz", normalize=(not args.no_normalize))

        # 🔥 FIX: descubrir index_{split}.csv aunque no esté dentro de data_dir
        index_path = _discover_index_csv(data_dir, split)
        index_map = _read_index_csv(index_path) if index_path is not None else None
        if index_path is not None and index_map is not None:
            print(f"[TRACE] usando index CSV: {index_path}")
        else:
            print(f"[TRACE] sin index CSV (no encontrado para split={split}) -> nombres 'unknown'")

        k = min(int(args.infer_examples), len(ds_inf))
        rng = np.random.default_rng(int(args.seed) + 12345)
        idxs = rng.choice(len(ds_inf), size=k, replace=False)

        out_all = out_dir / "inference_all"
        out_err = out_dir / "inference_errors"
        out_d21 = out_dir / "inference_d21"
        out_all.mkdir(parents=True, exist_ok=True)
        out_err.mkdir(parents=True, exist_ok=True)
        out_d21.mkdir(parents=True, exist_ok=True)

        manifest_path = out_dir / "inference_manifest.csv"
        with open(manifest_path, "w", newline="", encoding="utf-8") as fman:
            wman = csv.writer(fman)
            wman.writerow([
                "split", "row_i",
                "idx_global", "sample_name", "jaw", "source_path", "has_labels",
                "png_all", "png_err", "png_d21"
            ])

            with torch.no_grad():
                for r, i in enumerate(idxs, start=1):
                    i = int(i)

                    xyz, y = ds_inf[i]
                    xyz_b = xyz.unsqueeze(0).to(device, non_blocking=True)
                    logits = model(xyz_b)[0]  # [N,C]
                    pred = logits.argmax(dim=-1).detach().cpu().numpy().astype(np.int32)

                    xyz_np = xyz.cpu().numpy().astype(np.float32, copy=True)
                    y_np = y.cpu().numpy().astype(np.int32, copy=True)

                    # trazabilidad (row_i = i)
                    meta = {"idx_global": "", "sample_name": "", "jaw": "", "path": "", "has_labels": ""}
                    if index_map is not None and i in index_map:
                        meta = index_map[i]

                    sample = meta.get("sample_name", "") or ""
                    jaw = meta.get("jaw", "") or ""
                    idx_global = meta.get("idx_global", "") or ""
                    src_path = meta.get("path", "") or ""
                    has_labels = meta.get("has_labels", "") or ""

                    sample_tag = _sanitize_tag(sample)
                    jaw_tag = _sanitize_tag(jaw)

                    tag = f"{sample_tag}_{jaw_tag}".strip("_") if (sample_tag or jaw_tag) else "unknown"

                    png_all = out_all / f"ex_{r:02d}_row_{i:05d}_{tag}.png"
                    png_err = out_err / f"ex_{r:02d}_row_{i:05d}_{tag}.png"
                    png_d21 = out_d21 / f"ex_{r:02d}_row_{i:05d}_{tag}.png"

                    title = (
                        f"{split} row={i} | sample={sample} | jaw={jaw} | idx_global={idx_global} | "
                        f"best_epoch={ckpt_epoch} | C={C} | d21={args.d21_internal}"
                    )

                    plot_pointcloud_all_classes(xyz_np, y_np, pred, png_all, C=C, title=title, s=1.0)
                    plot_errors(xyz_np, y_np, pred, png_err, bg=bg, title=title, s=1.0)
                    plot_d21_focus(xyz_np, y_np, pred, png_d21, d21_idx=int(args.d21_internal), bg=bg, title=title, s=1.2)

                    wman.writerow([
                        split, i,
                        idx_global, sample, jaw, src_path, has_labels,
                        str(png_all), str(png_err), str(png_d21)
                    ])

        print(f"[INFER] manifest guardado en: {manifest_path}")

    total = time.time() - t0
    print(f"[DONE] out_dir={out_dir} | total_sec={total:.1f} | best_val_f1_macro(no_bg)={best_val_f1:.4f} | C={C} | d21={args.d21_internal}")


if __name__ == "__main__":
    main()
