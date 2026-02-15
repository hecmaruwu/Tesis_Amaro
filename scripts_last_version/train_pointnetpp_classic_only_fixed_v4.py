#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_pointnetpp_classic_only_fixed_v4.py

PointNet++ (SSG) – segmentación multiclase dental 3D
✅ Alineado al estilo/logging de tu PointNet Classic
✅ Métricas epoch-level “paper-correctas” (macro F1/IoU sin BG)
✅ acc_all / acc_no_bg por conteo global de puntos (no promedio por batch)
✅ d21 binario acumulado (sin BG) + d21(bin all) incluyendo BG
✅ pred_bg_frac y gt_bg_frac (baseline) en val
✅ Infer trazable con idx_local REAL (dataset retorna índice) + discovery index_*.csv robusto
✅ Geometría (FPS/ball-query/interp) SIEMPRE FP32 aunque uses AMP
✅ FIX pandas: lee index csv como string (evita "Cannot convert numpy.ndarray to numpy.ndarray")
✅ FIX unpack: make_loaders SIEMPRE retorna 3 loaders (train/val/test). Infer usa make_infer_loader aparte.

Uso típico:
  python train_pointnetpp_classic_only_fixed_v4.py \
    --data_dir .../fixed_split/8192/... \
    --out_dir  .../outputs/pointnetpp_v4 \
    --epochs 200 --batch_size 8 --num_workers 4 \
    --lr 1e-3 --weight_decay 1e-4 --dropout 0.5 --grad_clip 1.0 \
    --use_amp \
    --d21_internal 8 \
    --infer_split test --infer_max 12
"""

# ============================================================
# PARTE 1/4
# - imports
# - seed / io
# - dataset + loaders (train/val/test + infer con idx_local real)
# ============================================================

import os
import json
import csv
import time
import argparse
import random
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


# ----------------------------
# SEED / IO
# ----------------------------
def set_seed(seed: int = 42):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _get_lr(opt: torch.optim.Optimizer) -> float:
    try:
        return float(opt.param_groups[0].get("lr", 0.0))
    except Exception:
        return float("nan")


# ----------------------------
# NORMALIZACIÓN
# ----------------------------
def normalize_unit_sphere(xyz: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    xyz: [N,3] -> centra y escala a radio 1 (unit sphere).
    """
    c = xyz.mean(dim=0, keepdim=True)
    x = xyz - c
    r = torch.norm(x, dim=1).max().clamp_min(eps)
    return x / r


# ----------------------------
# DATASET
# ----------------------------
class NPZDataset(Dataset):
    """
    Carga X_*.npz / Y_*.npz:
      X: [B,N,3] float32
      Y: [B,N]   int64

    return_index:
      - False: (xyz, y)   -> train/val/test normal
      - True : (xyz, y, idx_local) -> infer trazable (row_i real del split)
    """
    def __init__(self, Xp: Path, Yp: Path, normalize: bool = True, return_index: bool = False):
        self.X = np.load(Xp)["X"].astype(np.float32)
        self.Y = np.load(Yp)["Y"].astype(np.int64)

        assert self.X.ndim == 3 and self.X.shape[-1] == 3, f"X shape inesperada: {self.X.shape}"
        assert self.Y.ndim == 2, f"Y shape inesperada: {self.Y.shape}"
        assert self.X.shape[0] == self.Y.shape[0], "B mismatch"
        assert self.X.shape[1] == self.Y.shape[1], "N mismatch"

        self.normalize = bool(normalize)
        self.return_index = bool(return_index)

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, i: int):
        i = int(i)
        xi = np.ascontiguousarray(self.X[i])
        yi = np.ascontiguousarray(self.Y[i])

        xyz = torch.as_tensor(xi, dtype=torch.float32)  # [N,3]
        y   = torch.as_tensor(yi, dtype=torch.int64)    # [N]

        if self.normalize:
            xyz = normalize_unit_sphere(xyz)

        if self.return_index:
            return xyz, y, torch.tensor(i, dtype=torch.int64)
        return xyz, y


def make_loaders(data_dir: Path, bs: int, nw: int, normalize: bool = True):
    """
    ✅ SIEMPRE retorna 3 loaders: (dl_tr, dl_va, dl_te)
    """
    ds_tr = NPZDataset(data_dir / "X_train.npz", data_dir / "Y_train.npz", normalize=normalize, return_index=False)
    ds_va = NPZDataset(data_dir / "X_val.npz",   data_dir / "Y_val.npz",   normalize=normalize, return_index=False)
    ds_te = NPZDataset(data_dir / "X_test.npz",  data_dir / "Y_test.npz",  normalize=normalize, return_index=False)

    common = dict(
        batch_size=int(bs),
        num_workers=int(nw),
        pin_memory=True,
        persistent_workers=(int(nw) > 0),
        prefetch_factor=2 if int(nw) > 0 else None,
        drop_last=False,
    )
    common = {k: v for k, v in common.items() if v is not None}

    dl_tr = DataLoader(ds_tr, shuffle=True,  **common)
    dl_va = DataLoader(ds_va, shuffle=False, **common)
    dl_te = DataLoader(ds_te, shuffle=False, **common)
    return dl_tr, dl_va, dl_te


def make_infer_loader(data_dir: Path, split: str, bs: int, nw: int, normalize: bool = True) -> DataLoader:
    """
    Loader para infer trazable:
    - shuffle=False
    - dataset return_index=True -> idx_local real del split (row_i)
    Retorna batches: (xyz, y, idx_local)
    """
    split = str(split).lower()
    if split == "train":
        ds = NPZDataset(data_dir / "X_train.npz", data_dir / "Y_train.npz", normalize=normalize, return_index=True)
    elif split == "val":
        ds = NPZDataset(data_dir / "X_val.npz", data_dir / "Y_val.npz", normalize=normalize, return_index=True)
    elif split == "test":
        ds = NPZDataset(data_dir / "X_test.npz", data_dir / "Y_test.npz", normalize=normalize, return_index=True)
    else:
        raise ValueError(f"split inválido: {split}")

    common = dict(
        batch_size=int(bs),
        num_workers=int(nw),
        pin_memory=True,
        persistent_workers=(int(nw) > 0),
        prefetch_factor=2 if int(nw) > 0 else None,
        drop_last=False,
    )
    common = {k: v for k, v in common.items() if v is not None}

    return DataLoader(ds, shuffle=False, **common)


# --- fin parte 1/4 ---

# ============================================================
# PARTE 2/4
# - PointNet++ utilidades (geometría SIEMPRE FP32)
# - Set Abstraction / Feature Propagation
# - Modelo PointNet2Seg (SSG) para segmentación
# ============================================================

def _as_fp32(x: torch.Tensor) -> torch.Tensor:
    return x.float() if x.dtype != torch.float32 else x


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    src: [B,N,3], dst: [B,M,3] -> dist^2: [B,N,M]
    """
    src = _as_fp32(src)
    dst = _as_fp32(dst)
    dist = -2.0 * torch.matmul(src, dst.transpose(2, 1))
    dist += torch.sum(src ** 2, dim=-1, keepdim=True)
    dist += torch.sum(dst ** 2, dim=-1, keepdim=True).transpose(2, 1)
    return torch.clamp(dist, min=0.0)


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    points: [B,N,C]
    idx: [B,S] o [B,S,K]
    return: [B,S,C] o [B,S,K,C]
    """
    B, N, _ = points.shape
    idx = torch.clamp(idx, 0, N - 1)

    device = points.device
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1

    batch_indices = torch.arange(B, device=device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    xyz: [B,N,3] -> idx [B,npoint]
    """
    xyz = _as_fp32(xyz)
    device = xyz.device
    B, N, _ = xyz.shape

    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, dtype=torch.float32, device=device)

    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=1)[1]

    return centroids


def query_ball_point(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """
    xyz: [B,N,3], new_xyz: [B,S,3] -> group_idx: [B,S,nsample]
    robusto: si no hay vecinos, replica el primer válido (o 0).
    """
    xyz = _as_fp32(xyz)
    new_xyz = _as_fp32(new_xyz)

    device = xyz.device
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape

    sqrdists = square_distance(new_xyz, xyz)  # [B,S,N]
    group_idx = torch.arange(N, device=device).view(1, 1, N).repeat(B, S, 1)

    invalid = sqrdists > (radius * radius)
    group_idx[invalid] = N

    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]  # [B,S,nsample]

    # si el primer es N, lo reemplazamos por 0
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat(1, 1, nsample)
    group_first = torch.where(group_first == N, torch.zeros_like(group_first), group_first)

    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    group_idx = torch.clamp(group_idx, 0, N - 1)
    return group_idx


def sample_and_group(npoint: int, radius: float, nsample: int,
                     xyz: torch.Tensor, points: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    xyz: [B,N,3]
    points: [B,N,D] o None
    return:
      new_xyz: [B,S,3]
      new_points: [B,S,K,3(+D)]
    """
    xyz = _as_fp32(xyz)
    if points is not None:
        points = _as_fp32(points)

    fps_idx = farthest_point_sample(xyz, npoint)          # [B,S]
    new_xyz = index_points(xyz, fps_idx)                  # [B,S,3]
    idx = query_ball_point(radius, nsample, xyz, new_xyz) # [B,S,K]

    grouped_xyz = index_points(xyz, idx)                  # [B,S,K,3]
    grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2) # [B,S,K,3]

    if points is not None:
        grouped_points = index_points(points, idx)        # [B,S,K,D]
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B,S,K,3+D]
    else:
        new_points = grouped_xyz_norm  # [B,S,K,3]

    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint: int, radius: float, nsample: int, in_channel: int, mlp: List[int]):
        super().__init__()
        self.npoint = int(npoint)
        self.radius = float(radius)
        self.nsample = int(nsample)

        layers = []
        last_ch = int(in_channel)
        for out_ch in mlp:
            layers += [
                nn.Conv2d(last_ch, int(out_ch), 1),
                nn.BatchNorm2d(int(out_ch)),
                nn.ReLU(inplace=True),
            ]
            last_ch = int(out_ch)
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz: torch.Tensor, points: Optional[torch.Tensor]):
        # geometría fp32 SIEMPRE
        with torch.cuda.amp.autocast(enabled=False):
            xyz_f = _as_fp32(xyz)
            pts_f = _as_fp32(points) if points is not None else None
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz_f, pts_f)
            # [B,S,K,C] -> [B,C,K,S]
            new_points = new_points.permute(0, 3, 2, 1).contiguous().float()

        new_points = self.mlp(new_points)            # [B,out,K,S]
        new_points = torch.max(new_points, dim=2)[0] # [B,out,S]
        new_points = new_points.transpose(2, 1).contiguous()  # [B,S,out]
        return new_xyz, new_points


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel: int, mlp: List[int]):
        super().__init__()
        layers = []
        last_ch = int(in_channel)
        for out_ch in mlp:
            layers += [
                nn.Conv1d(last_ch, int(out_ch), 1),
                nn.BatchNorm1d(int(out_ch)),
                nn.ReLU(inplace=True),
            ]
            last_ch = int(out_ch)
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz1: torch.Tensor, xyz2: torch.Tensor,
                points1: Optional[torch.Tensor], points2: torch.Tensor) -> torch.Tensor:
        """
        xyz1: [B,N,3] (denso)
        xyz2: [B,S,3] (subsample)
        points1: [B,N,D1] o None
        points2: [B,S,D2]
        """
        with torch.cuda.amp.autocast(enabled=False):
            xyz1 = _as_fp32(xyz1)
            xyz2 = _as_fp32(xyz2)
            points2 = _as_fp32(points2)
            if points1 is not None:
                points1 = _as_fp32(points1)

            B, N, _ = xyz1.shape
            _, S, _ = xyz2.shape

            if S == 1:
                interpolated = points2.repeat(1, N, 1)  # [B,N,D2]
            else:
                dists = square_distance(xyz1, xyz2)  # [B,N,S]
                dists, idx = dists.sort(dim=-1)
                dists = dists[:, :, :3]
                idx = idx[:, :, :3]

                dist_recip = 1.0 / (dists + 1e-8)
                norm = torch.sum(dist_recip, dim=2, keepdim=True)
                weight = dist_recip / norm

                grouped = index_points(points2, idx)  # [B,N,3,D2]
                interpolated = torch.sum(grouped * weight.unsqueeze(-1), dim=2)  # [B,N,D2]

            if points1 is not None:
                new_points = torch.cat([points1, interpolated], dim=-1)  # [B,N,D1+D2]
            else:
                new_points = interpolated  # [B,N,D2]

            new_points = new_points.transpose(2, 1).contiguous().float()  # [B,D,N]

        new_points = self.mlp(new_points)  # [B,out,N]
        return new_points.transpose(2, 1).contiguous()  # [B,N,out]


class PointNet2Seg(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.5):
        super().__init__()
        C = int(num_classes)

        # radios pensados para unit sphere (~1)
        self.sa1 = PointNetSetAbstraction(1024, 0.10, 32, 3,       [32, 32, 64])
        self.sa2 = PointNetSetAbstraction(256,  0.20, 32, 3 + 64,  [64, 64, 128])
        self.sa3 = PointNetSetAbstraction(64,   0.40, 32, 3 + 128, [128, 128, 256])
        self.sa4 = PointNetSetAbstraction(16,   0.80, 32, 3 + 256, [256, 256, 512])

        self.fp4 = PointNetFeaturePropagation(512 + 256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(256 + 128, [256, 256])
        self.fp2 = PointNetFeaturePropagation(256 + 64,  [256, 128])
        self.fp1 = PointNetFeaturePropagation(128,       [128, 128, 128])

        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(float(dropout))
        self.conv2 = nn.Conv1d(128, C, 1)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        xyz: [B,N,3]
        return logits: [B,N,C]
        """
        xyz = xyz.float()

        l0_xyz = xyz
        l0_points = None

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3p = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2p = self.fp3(l2_xyz, l3_xyz, l2_points, l3p)
        l1p = self.fp2(l1_xyz, l2_xyz, l1_points, l2p)
        l0p = self.fp1(l0_xyz, l1_xyz, None, l1p)

        x = l0p.transpose(2, 1).contiguous()  # [B,128,N]
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.drop(x)
        x = self.conv2(x)                     # [B,C,N]
        return x.transpose(2, 1).contiguous()  # [B,N,C]


# --- fin parte 2/4 ---

# ============================================================
# PARTE 3/4
# - Métricas epoch-level (correctas, estilo PointNet Classic)
# - Confusion matrix global
# - d21 binario (sin BG y con BG)
# - run_epoch (train / val / test)
# ============================================================

def _fast_confusion_matrix(pred: torch.Tensor, gt: torch.Tensor, C: int) -> torch.Tensor:
    """
    pred, gt: [*] int64 en [0, C-1]
    return: [C,C] filas=gt, columnas=pred
    """
    pred = pred.reshape(-1).to(torch.int64)
    gt   = gt.reshape(-1).to(torch.int64)

    valid = (gt >= 0) & (gt < C) & (pred >= 0) & (pred < C)
    if not valid.any():
        return torch.zeros((C, C), dtype=torch.int64, device=pred.device)

    gt = gt[valid]
    pred = pred[valid]

    idx = gt * C + pred
    cm = torch.bincount(idx, minlength=C * C).reshape(C, C)
    return cm


def macro_f1_iou_from_cm(cm: torch.Tensor, bg: int = 0):
    """
    Macro F1 / IoU SIN background.
    Ignora clases sin soporte.
    """
    C = int(cm.shape[0])
    f1s, ious = [], []

    for c in range(C):
        if c == bg:
            continue
        tp = cm[c, c].item()
        fp = (cm[:, c].sum() - cm[c, c]).item()
        fn = (cm[c, :].sum() - cm[c, c]).item()

        denom = tp + fp + fn
        if denom == 0:
            continue

        f1  = (2 * tp) / (2 * tp + fp + fn + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        f1s.append(f1)
        ious.append(iou)

    if len(f1s) == 0:
        return 0.0, 0.0
    return float(np.mean(f1s)), float(np.mean(ious))


def d21_binary_counts(pred: torch.Tensor, gt: torch.Tensor,
                      d21_idx: int, bg: int = 0, include_bg: bool = False):
    """
    Binario: d21 vs resto
    include_bg=False  -> ignora puntos GT==BG
    include_bg=True   -> incluye todo (baseline)
    """
    pred = pred.reshape(-1)
    gt   = gt.reshape(-1)

    if d21_idx < 0:
        return 0, 0, 0, 0

    if not include_bg:
        m = (gt != bg)
        if not m.any():
            return 0, 0, 0, 0
        pred = pred[m]
        gt = gt[m]

    t_pos = (gt == d21_idx)
    p_pos = (pred == d21_idx)

    tp = int((p_pos & t_pos).sum())
    fp = int((p_pos & (~t_pos)).sum())
    fn = int(((~p_pos) & t_pos).sum())
    tn = int(((~p_pos) & (~t_pos)).sum())
    return tp, fp, fn, tn


def binary_metrics(tp: int, fp: int, fn: int, tn: int):
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    f1  = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    return float(acc), float(f1), float(iou)


def _check_finite(t: torch.Tensor, name: str):
    if not torch.isfinite(t).all():
        raise FloatingPointError(f"[NaN/Inf] detectado en {name}")


# ------------------------------------------------------------
# RUN EPOCH
# ------------------------------------------------------------
def run_epoch(model: nn.Module,
              loader: DataLoader,
              optimizer: Optional[torch.optim.Optimizer],
              loss_fn: nn.Module,
              C: int,
              d21_idx: int,
              device: torch.device,
              bg: int,
              train: bool,
              use_amp: bool,
              grad_clip: Optional[float] = None) -> Dict[str, float]:

    model.train(train)

    if use_amp and run_epoch.scaler is None:
        run_epoch.scaler = torch.cuda.amp.GradScaler()
    scaler = run_epoch.scaler

    loss_sum = 0.0
    n_batches = 0

    correct_all = total_all = 0
    correct_no_bg = total_no_bg = 0

    cm = torch.zeros((C, C), dtype=torch.int64, device=device)

    d21_tp = d21_fp = d21_fn = d21_tn = 0
    d21a_tp = d21a_fp = d21a_fn = d21a_tn = 0

    pred_bg = pred_tot = 0
    gt_bg = gt_tot = 0

    for batch in loader:
        if len(batch) == 3:
            xyz, y, _ = batch
        else:
            xyz, y = batch

        xyz = xyz.to(device, non_blocking=True)
        y   = y.to(device, non_blocking=True)

        _check_finite(xyz, "xyz")
        _check_finite(y.float(), "labels")

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(xyz)           # [B,N,C]
            _check_finite(logits, "logits")
            loss = loss_fn(logits.reshape(-1, C), y.reshape(-1))
            _check_finite(loss, "loss")

        if train:
            if use_amp:
                scaler.scale(loss).backward()
                if grad_clip and grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        pred = logits.argmax(dim=-1)

        loss_sum += float(loss.item())
        n_batches += 1

        # accuracies
        correct_all += int((pred == y).sum())
        total_all += int(y.numel())

        m = (y != bg)
        if m.any():
            correct_no_bg += int((pred[m] == y[m]).sum())
            total_no_bg += int(m.sum())

        # confusion
        cm += _fast_confusion_matrix(pred, y, C)

        # d21
        tp, fp, fn, tn = d21_binary_counts(pred, y, d21_idx, bg, include_bg=False)
        d21_tp += tp; d21_fp += fp; d21_fn += fn; d21_tn += tn

        tp, fp, fn, tn = d21_binary_counts(pred, y, d21_idx, bg, include_bg=True)
        d21a_tp += tp; d21a_fp += fp; d21a_fn += fn; d21a_tn += tn

        # bg fractions
        pred_bg += int((pred == bg).sum())
        pred_tot += int(pred.numel())
        gt_bg += int((y == bg).sum())
        gt_tot += int(y.numel())

    n = max(1, n_batches)

    acc_all = correct_all / (total_all + 1e-8)
    acc_no_bg = correct_no_bg / (total_no_bg + 1e-8) if total_no_bg > 0 else 0.0

    f1m, ioum = macro_f1_iou_from_cm(cm, bg)

    d21_acc, d21_f1, d21_iou = binary_metrics(d21_tp, d21_fp, d21_fn, d21_tn)
    d21_bin_acc_all, _, _ = binary_metrics(d21a_tp, d21a_fp, d21a_fn, d21a_tn)

    return {
        "loss": float(loss_sum / n),
        "acc_all": float(acc_all),
        "acc_no_bg": float(acc_no_bg),
        "f1_macro": float(f1m),
        "iou_macro": float(ioum),
        "d21_acc": float(d21_acc),
        "d21_f1": float(d21_f1),
        "d21_iou": float(d21_iou),
        "d21_bin_acc_all": float(d21_bin_acc_all),
        "pred_bg_frac": float(pred_bg / (pred_tot + 1e-8)),
        "gt_bg_frac": float(gt_bg / (gt_tot + 1e-8)),
    }


run_epoch.scaler = None

# --- fin parte 3/4 ---

# ============================================================
# PARTE 4/4
# - index_*.csv discovery + lectura robusta (dtype=str)
# - plots + helpers
# - main(): train/val/test + best.pt + history.json + metrics_epoch.csv
# - inferencia trazable con idx_local real
# ============================================================

def _read_index_csv(path: Optional[Path]) -> Optional[Dict[int, Dict[str, str]]]:
    """
    row_i (fila dentro del split) -> meta dict con strings.
    Lee dtype=str para evitar el crash pandas: "Cannot convert numpy.ndarray to numpy.ndarray".
    """
    if path is None or (not path.exists()):
        return None
    try:
        import pandas as pd
        df = pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            low_memory=False,
        )
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
      3) Teeth_3ds/merged_*/index_{split}.csv (más reciente por mtime)
    """
    fname = f"index_{split}.csv"

    p1 = data_dir / fname
    if p1.exists():
        return p1

    for p in data_dir.parents:
        cand = p / fname
        if cand.exists():
            return cand
        if p.name == "Teeth_3ds":
            break

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
# VISUALIZACIÓN
# ============================================================
def _class_colors(C: int):
    cmap = plt.colormaps.get_cmap("tab20")
    C = max(int(C), 2)
    return [cmap(i / max(C - 1, 1)) for i in range(C)]


def _safe_np(a: np.ndarray, dtype=None) -> np.ndarray:
    out = np.asarray(a if dtype is None else np.asarray(a, dtype=dtype))
    if not out.flags["C_CONTIGUOUS"]:
        out = np.ascontiguousarray(out)
    if type(out) is not np.ndarray:
        out = np.array(out, copy=True)
    return out


def plot_pointcloud_all_classes(xyz: np.ndarray, y_gt: np.ndarray, y_pr: np.ndarray,
                                out_png: Path, C: int, title: str = "", s: float = 1.0):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cols = _class_colors(C)

    xyz = _safe_np(xyz, dtype=np.float32)
    y_gt = _safe_np(y_gt, dtype=np.int32)
    y_pr = _safe_np(y_pr, dtype=np.int32)

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


def plot_errors(xyz: np.ndarray, y_gt: np.ndarray, y_pr: np.ndarray,
                out_png: Path, bg: int = 0, title: str = "", s: float = 1.0):
    out_png.parent.mkdir(parents=True, exist_ok=True)

    xyz = _safe_np(xyz, dtype=np.float32)
    y_gt = _safe_np(y_gt, dtype=np.int32)
    y_pr = _safe_np(y_pr, dtype=np.int32)

    ok = (y_gt == y_pr)
    c = np.zeros((xyz.shape[0], 4), dtype=np.float32)
    c[:, :] = (0.75, 0.75, 0.75, 1.0)
    c[~ok, :] = (0.85, 0.10, 0.10, 1.0)
    c[y_gt == int(bg), :] = (0.85, 0.85, 0.85, 0.6)

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


def plot_d21_focus(xyz: np.ndarray, y_gt: np.ndarray, y_pr: np.ndarray,
                   out_png: Path, d21_idx: int, bg: int = 0, title: str = "", s: float = 1.2):
    out_png.parent.mkdir(parents=True, exist_ok=True)

    xyz = _safe_np(xyz, dtype=np.float32)
    y_gt = _safe_np(y_gt, dtype=np.int32)
    y_pr = _safe_np(y_pr, dtype=np.int32)

    gt21 = (y_gt == int(d21_idx))
    pr21 = (y_pr == int(d21_idx))
    tp = gt21 & pr21
    fp = (~gt21) & pr21
    fn = gt21 & (~pr21)
    err = fp | fn

    c = np.zeros((xyz.shape[0], 4), dtype=np.float32)
    c[:, :] = (0.75, 0.75, 0.75, 1.0)
    c[y_gt == int(bg), :] = (0.85, 0.85, 0.85, 0.6)

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
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--use_amp", action="store_true")

    ap.add_argument("--bg", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)

    # ✅ igual que tu PointNet classic: pasas el índice interno directamente
    ap.add_argument("--d21_internal", type=int, default=-1)

    # infer opcional
    ap.add_argument("--infer_split", type=str, default=None, choices=["train", "val", "test"])
    ap.add_argument("--infer_max", type=int, default=12)

    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------- num_classes robusto (label_map si existe) --------
    label_map_path = data_dir / "label_map.json"
    if label_map_path.exists():
        lm = json.load(open(label_map_path, "r"))
        idx2id = {int(k): int(v) for k, v in lm.get("idx2id", {}).items()}
        num_classes = int(lm.get("num_classes", (max(idx2id.keys()) + 1) if idx2id else 0))
        if num_classes <= 0:
            Ytr = np.load(data_dir / "Y_train.npz")["Y"]
            num_classes = int(Ytr.max()) + 1
    else:
        Ytr = np.load(data_dir / "Y_train.npz")["Y"]
        num_classes = int(Ytr.max()) + 1

    d21_idx = int(args.d21_internal)

    print(f"[INFO] device={device} | C={num_classes} | bg={int(args.bg)} | d21_idx={d21_idx}")

    # -------- loaders --------
    dl_tr, dl_va, dl_te = make_loaders(
        data_dir=data_dir,
        bs=int(args.batch_size),
        nw=int(args.num_workers),
        normalize=True,
    )

    # -------- modelo --------
    model = PointNet2Seg(num_classes=num_classes, dropout=float(args.dropout)).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(args.epochs),
        eta_min=1e-6,
    )

    # ✅ BG incluido en la loss (como tu PointNet normal)
    loss_fn = nn.CrossEntropyLoss()

    history: List[Dict[str, float]] = []

    # ✅ BEST por val_f1_macro(no_bg)
    best = {"epoch": -1, "val_f1_macro": -1.0}

    t0 = time.time()

    for epoch in range(1, int(args.epochs) + 1):
        tr = run_epoch(
            model, dl_tr, optimizer, loss_fn,
            C=num_classes, d21_idx=d21_idx, device=device,
            bg=int(args.bg), train=True, use_amp=bool(args.use_amp),
            grad_clip=float(args.grad_clip),
        )
        va = run_epoch(
            model, dl_va, None, loss_fn,
            C=num_classes, d21_idx=d21_idx, device=device,
            bg=int(args.bg), train=False, use_amp=bool(args.use_amp),
        )

        lr_now = _get_lr(optimizer)

        print(
            f"[{epoch}/{int(args.epochs)}] "
            f"train loss={tr['loss']:.4f} f1m={tr['f1_macro']:.3f} ioum={tr['iou_macro']:.3f} "
            f"acc_all={tr['acc_all']:.3f} acc_no_bg={tr['acc_no_bg']:.3f} | "
            f"val loss={va['loss']:.4f} f1m={va['f1_macro']:.3f} ioum={va['iou_macro']:.3f} "
            f"acc_all={va['acc_all']:.3f} acc_no_bg={va['acc_no_bg']:.3f} | "
            f"d21(cls) acc={va['d21_acc']:.3f} f1={va['d21_f1']:.3f} iou={va['d21_iou']:.3f} | "
            f"d21(bin all) acc={va['d21_bin_acc_all']:.3f} | "
            f"pred_bg_frac(val)={va['pred_bg_frac']:.3f} gt_bg_frac(val)={va['gt_bg_frac']:.3f} "
            f"lr={lr_now:.2e}"
        )

        history.append({
            "epoch": float(epoch),
            **{f"train_{k}": float(v) for k, v in tr.items()},
            **{f"val_{k}": float(v) for k, v in va.items()},
            "lr": float(lr_now),
        })

        if float(va["f1_macro"]) > float(best["val_f1_macro"]):
            best["val_f1_macro"] = float(va["f1_macro"])
            best["epoch"] = int(epoch)
            torch.save(model.state_dict(), out_dir / "best.pt")

        torch.save(model.state_dict(), out_dir / "last.pt")
        scheduler.step()

    # -------- guardar history --------
    save_json(
        {
            "best_epoch": int(best["epoch"]),
            "best_val_f1_macro": float(best["val_f1_macro"]),
            "history": history,
        },
        out_dir / "history.json"
    )

    # -------- csv epoch --------
    if history:
        csv_path = out_dir / "metrics_epoch.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=history[0].keys())
            w.writeheader()
            for r in history:
                w.writerow(r)

    # -------- TEST con best --------
    if (out_dir / "best.pt").exists():
        model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))

    te = run_epoch(
        model, dl_te, None, loss_fn,
        C=num_classes, d21_idx=d21_idx, device=device,
        bg=int(args.bg), train=False, use_amp=bool(args.use_amp),
    )

    save_json(
        {
            "best_epoch": int(best["epoch"]),
            "test": te,
            "elapsed_sec": float(time.time() - t0),
            "num_classes": int(num_classes),
            "d21_idx": int(d21_idx),
        },
        out_dir / "test_metrics.json"
    )

    print("[TEST]", te)

    # ============================================================
    # INFERENCIA (opcional) – trazable con idx_local real
    # ============================================================
    if args.infer_split is not None:
        split = str(args.infer_split).lower()

        infer_loader = make_infer_loader(
            data_dir=data_dir,
            split=split,
            bs=int(args.batch_size),
            nw=int(args.num_workers),
            normalize=True,
        )

        idx_csv = _discover_index_csv(data_dir, split)
        idx_map = _read_index_csv(idx_csv) if idx_csv else None

        inf_dir = out_dir / "inference" / split
        inf_dir.mkdir(parents=True, exist_ok=True)

        manifest = []
        model.eval()
        seen = 0

        with torch.no_grad():
            for batch in infer_loader:
                xyz, y, idx_local = batch
                xyz = xyz.to(device, non_blocking=True)
                y   = y.to(device, non_blocking=True)

                logits = model(xyz)
                pred = logits.argmax(dim=-1)

                B = xyz.shape[0]
                for i in range(B):
                    if seen >= int(args.infer_max):
                        break

                    row_i = int(idx_local[i].item())  # ✅ row real del split

                    tag = jaw = idx_global = ""
                    if idx_map and row_i in idx_map:
                        tag = _sanitize_tag(idx_map[row_i].get("sample_name", ""))
                        jaw = _sanitize_tag(idx_map[row_i].get("jaw", ""))
                        idx_global = _sanitize_tag(idx_map[row_i].get("idx_global", ""))

                    name = f"ex_{seen:02d}_row_{row_i:05d}"
                    if tag:
                        name += f"_{tag}"
                    if jaw:
                        name += f"_{jaw}"

                    out_all = inf_dir / f"{name}_all.png"
                    out_err = inf_dir / f"{name}_err.png"
                    out_d21 = inf_dir / f"{name}_d21.png"

                    xyz_np = xyz[i].detach().cpu().numpy()
                    y_np   = y[i].detach().cpu().numpy()
                    p_np   = pred[i].detach().cpu().numpy()

                    plot_pointcloud_all_classes(xyz_np, y_np, p_np, out_all, C=num_classes, title=name, s=1.0)
                    plot_errors(xyz_np, y_np, p_np, out_err, bg=int(args.bg), title=name, s=1.0)
                    if d21_idx >= 0:
                        plot_d21_focus(xyz_np, y_np, p_np, out_d21, d21_idx=d21_idx, bg=int(args.bg), title=name, s=1.2)

                    manifest.append({
                        "row_i": int(row_i),
                        "name": name,
                        "idx_global": idx_global,
                        "sample_name": tag,
                        "jaw": jaw,
                        "split": split,
                        "index_csv_used": str(idx_csv) if idx_csv else "",
                    })
                    seen += 1

                if seen >= int(args.infer_max):
                    break

        if manifest:
            with open(inf_dir / "inference_manifest.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=manifest[0].keys())
                w.writeheader()
                for r in manifest:
                    w.writerow(r)

        print(f"[INFER] guardado en {inf_dir}")
        if idx_csv:
            print(f"[INFER] index_{split}.csv usado: {idx_csv}")

    print(
        f"[DONE] out_dir={out_dir} | total_sec={time.time()-t0:.1f} | "
        f"best_epoch={best['epoch']} | best_val_f1_macro(no_bg)={best['val_f1_macro']:.4f} | "
        f"C={num_classes} | d21={d21_idx}"
    )


if __name__ == "__main__":
    main()
