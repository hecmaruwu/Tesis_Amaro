#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_pointnetpp_classic_only_fixed_v3.py

PointNet++ (SSG) – Segmentación multiclase dental 3D (OPCIÓN A – BG incluido en loss)

CORRECCIONES CLAVE (v3):
✅ FIX DataLoader/NumPy raro: torch.as_tensor + np.ascontiguousarray en __getitem__
✅ FIX Matplotlib/numpy.may_share_memory crash: evitar __array_function__ y forzar arrays "base" (np.asarray + copy)
✅ Geometría SIEMPRE en FP32 (square_distance, FPS, ball query, interp) incluso con AMP
✅ BallQuery robusto: nunca deja idx=N (invalid), clampa índices, evita NaNs silenciosos
✅ Check NaN/Inf en logits y loss
✅ Mantiene misma estructura del script v2 (solo fixes + pequeñas defensas)

NOTA:
- PARTE 1/2: imports + utils + dataset + modelo + métricas + run_epoch + trazabilidad + visualización (FIXED)
- PARTE 2/2: main() completo (train loop + logging + plots + test + infer)

Guárdalo como:
  /home/htaucare/Tesis_Amaro/scripts_last_version/train_pointnetpp_classic_only_fixed_v3.py
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
# DATASET  (FIX: as_tensor + contig)
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
        # ✅ FIX: forzar arrays contiguos y tensores sin el bug extraño del worker
        xi = np.ascontiguousarray(self.X[i])
        yi = np.ascontiguousarray(self.Y[i])

        xyz = torch.as_tensor(xi, dtype=torch.float32)  # [N,3]
        y   = torch.as_tensor(yi, dtype=torch.int64)    # [N]

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
# POINTNET++ UTILIDADES (ROBUSTAS + FP32)
# ============================================================
def _as_fp32(x: torch.Tensor) -> torch.Tensor:
    return x.float() if x.dtype != torch.float32 else x


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    src: [B, N, 3], dst: [B, M, 3]  ->  dist^2 [B, N, M]
    """
    src = _as_fp32(src)
    dst = _as_fp32(dst)

    dist = -2.0 * torch.matmul(src, dst.transpose(2, 1))              # [B,N,M]
    dist += torch.sum(src ** 2, dim=-1, keepdim=True)                 # [B,N,1]
    dist += torch.sum(dst ** 2, dim=-1, keepdim=True).transpose(2, 1) # [B,1,M]
    return torch.clamp(dist, min=0.0)


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    points: [B, N, C]
    idx: [B, S] o [B, S, K]
    return: [B, S, C] o [B, S, K, C]
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
    xyz: [B, N, 3] -> idx: [B, npoint]
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
    ROBUSTO: sin vecinos -> replica el primer índice válido (o 0).
    """
    xyz = _as_fp32(xyz)
    new_xyz = _as_fp32(new_xyz)

    device = xyz.device
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape

    sqrdists = square_distance(new_xyz, xyz)  # [B,S,N]
    group_idx = torch.arange(N, device=device).view(1, 1, N).repeat(B, S, 1)

    invalid = sqrdists > (radius * radius)
    group_idx[invalid] = N  # marcador inválido

    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]  # [B,S,nsample]

    group_first = group_idx[:, :, 0].view(B, S, 1).repeat(1, 1, nsample)
    group_first = torch.where(group_first == N, torch.zeros_like(group_first), group_first)

    mask = group_idx == N
    group_idx[mask] = group_first[mask]

    group_idx = torch.clamp(group_idx, 0, N - 1)
    return group_idx


def sample_and_group(npoint: int, radius: float, nsample: int,
                     xyz: torch.Tensor, points: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    xyz = _as_fp32(xyz)
    if points is not None:
        points = _as_fp32(points)

    fps_idx = farthest_point_sample(xyz, npoint)          # [B,npoint]
    new_xyz = index_points(xyz, fps_idx)                  # [B,npoint,3]
    idx = query_ball_point(radius, nsample, xyz, new_xyz) # [B,npoint,nsample]

    grouped_xyz = index_points(xyz, idx)                  # [B,npoint,nsample,3]
    grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2) # relativo

    if points is not None:
        grouped_points = index_points(points, idx)        # [B,npoint,nsample,D]
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm

    return new_xyz, new_points


# ============================================================
# POINTNET++ BLOQUES (FORZAR FP32 EN GEOMETRÍA)
# ============================================================
class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint: int, radius: float, nsample: int,
                 in_channel: int, mlp: List[int]):
        super().__init__()
        self.npoint = int(npoint)
        self.radius = float(radius)
        self.nsample = int(nsample)

        layers = []
        last_ch = int(in_channel)
        for out_ch in mlp:
            layers.append(nn.Conv2d(last_ch, out_ch, 1))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            last_ch = out_ch
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz: torch.Tensor, points: Optional[torch.Tensor]):
        # xyz [B,N,3], points [B,N,D] or None
        with torch.cuda.amp.autocast(enabled=False):
            xyz_f = _as_fp32(xyz)
            pts_f = _as_fp32(points) if points is not None else None
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz_f, pts_f)
            # [B,S,K,3(+D)] -> [B,C,K,S]
            new_points = new_points.permute(0, 3, 2, 1).contiguous().float()

        # MLP ya puede ir con AMP afuera
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
            layers.append(nn.Conv1d(last_ch, out_ch, 1))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            last_ch = out_ch
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz1: torch.Tensor, xyz2: torch.Tensor,
                points1: Optional[torch.Tensor], points2: torch.Tensor) -> torch.Tensor:
        # xyz1 [B,N,3], xyz2 [B,S,3], points1 [B,N,D1] or None, points2 [B,S,D2]
        with torch.cuda.amp.autocast(enabled=False):
            xyz1 = _as_fp32(xyz1)
            xyz2 = _as_fp32(xyz2)
            points2 = _as_fp32(points2)
            if points1 is not None:
                points1 = _as_fp32(points1)

            B, N, _ = xyz1.shape
            _, S, _ = xyz2.shape

            if S == 1:
                interpolated = points2.repeat(1, N, 1)
            else:
                dists = square_distance(xyz1, xyz2)  # [B,N,S]
                dists, idx = dists.sort(dim=-1)
                dists = dists[:, :, :3]
                idx = idx[:, :, :3]

                dist_recip = 1.0 / (dists + 1e-8)
                norm = torch.sum(dist_recip, dim=2, keepdim=True)
                weight = dist_recip / norm

                grouped = index_points(points2, idx)  # [B,N,3,D2]
                interpolated = torch.sum(grouped * weight.unsqueeze(-1), dim=2)

            if points1 is not None:
                new_points = torch.cat([points1, interpolated], dim=-1)
            else:
                new_points = interpolated

            new_points = new_points.transpose(2, 1).contiguous().float()  # [B,D,N]

        new_points = self.mlp(new_points)                     # [B,out,N]
        return new_points.transpose(2, 1).contiguous()        # [B,N,out]


class PointNet2Seg(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.5):
        super().__init__()
        C = int(num_classes)

        # Radios pensados para unit sphere (≈1)
        self.sa1 = PointNetSetAbstraction(npoint=1024, radius=0.10, nsample=32, in_channel=3,       mlp=[32, 32, 64])
        self.sa2 = PointNetSetAbstraction(npoint=256,  radius=0.20, nsample=32, in_channel=3 + 64,  mlp=[64, 64, 128])
        self.sa3 = PointNetSetAbstraction(npoint=64,   radius=0.40, nsample=32, in_channel=3 + 128, mlp=[128, 128, 256])
        self.sa4 = PointNetSetAbstraction(npoint=16,   radius=0.80, nsample=32, in_channel=3 + 256, mlp=[256, 256, 512])

        self.fp4 = PointNetFeaturePropagation(in_channel=512 + 256, mlp=[256, 256])
        self.fp3 = PointNetFeaturePropagation(in_channel=256 + 128, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=256 + 64,  mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128,       mlp=[128, 128, 128])

        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(float(dropout))
        self.conv2 = nn.Conv1d(128, C, 1)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        # xyz: [B,N,3]
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
        l0p = self.fp1(l0_xyz, l1_xyz, None, l1p)  # [B,N,128]

        x = l0p.transpose(2, 1).contiguous()       # [B,128,N]
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.drop(x)
        x = self.conv2(x)                          # [B,C,N]
        return x.transpose(2, 1).contiguous()       # [B,N,C]


# ============================================================
# MÉTRICAS (macro sin bg + d21 binario)
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

    f1s, ious = [], []
    for c in range(1, int(C)):
        tp = ((pred == c) & (gt == c)).sum().item()
        fp = ((pred == c) & (gt != c)).sum().item()
        fn = ((pred != c) & (gt == c)).sum().item()
        if (tp + fp + fn) == 0:
            continue
        f1 = (2 * tp) / (2 * tp + fp + fn + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        f1s.append(f1)
        ious.append(iou)

    if len(f1s) == 0:
        return 0.0, 0.0
    return float(np.mean(f1s)), float(np.mean(ious))


@torch.no_grad()
def d21_metrics_binary(pred: torch.Tensor, gt: torch.Tensor, d21_idx: int,
                       bg: int = 0, include_bg: bool = False) -> Tuple[float, float, float]:
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
    f1  = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    return float(acc), float(f1), float(iou)


# ============================================================
# TRAIN / EVAL helpers
# ============================================================
def _check_finite(t: torch.Tensor, name: str):
    if not torch.isfinite(t).all():
        raise FloatingPointError(f"[NAN/INF] detectado en {name}")


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
    scaler = run_epoch.scaler  # type: ignore
    if use_amp and scaler is None:
        scaler = torch.cuda.amp.GradScaler()
        run_epoch.scaler = scaler  # type: ignore

    loss_sum = acc_all_sum = acc_no_bg_sum = 0.0
    f1m_sum = ioum_sum = 0.0
    d21_acc_sum = d21_f1_sum = d21_iou_sum = 0.0
    d21_bin_all_sum = pred_bg_frac_sum = 0.0
    n_batches = 0

    for xyz, y in loader:
        xyz = xyz.to(device, non_blocking=True)
        y   = y.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(xyz)  # [B,N,C]
            _check_finite(logits, "logits")
            loss = loss_fn(logits.reshape(-1, C), y.reshape(-1))
            _check_finite(loss, "loss")

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
        acc_no_bg = (pred[mask] == y[mask]).float().mean().item() if mask.any() else 0.0

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
# VISUALIZACIÓN (FIX Matplotlib crash)
# ============================================================
def _class_colors(C: int):
    cmap = plt.colormaps.get_cmap("tab20")
    C = max(int(C), 2)
    cols = [cmap(i / max(C - 1, 1)) for i in range(C)]
    return cols


def _safe_np(a: np.ndarray, dtype=None) -> np.ndarray:
    """
    FIX para el crash:
      TypeError: no implementation found for 'numpy.may_share_memory' on types that implement __array_function__
    En algunos setups, Matplotlib termina llamando np.may_share_memory sobre objetos raros.
    Esto fuerza a arrays "plain" y contiguos, con base NumPy estándar.
    """
    if dtype is None:
        out = np.asarray(a)
    else:
        out = np.asarray(a, dtype=dtype)
    if not out.flags["C_CONTIGUOUS"]:
        out = np.ascontiguousarray(out)
    # defensivo: copy para asegurar que no es un "array subclass"
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
# PARTE 2/2
# main(): train loop + logging + plots + test + infer (con FIX Matplotlib)
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.5)

    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    ap.add_argument("--d21_internal", type=int, required=True)

    # Opción A: BG incluido en loss, controlamos con pesos
    ap.add_argument("--bg_index", type=int, default=0)
    ap.add_argument("--bg_weight", type=float, default=0.10)

    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--use_amp", action="store_true")

    ap.add_argument("--no_normalize", action="store_true")

    # warmup LR opcional
    ap.add_argument("--warmup_epochs", type=int, default=3)

    # Inferencia
    ap.add_argument("--do_infer", action="store_true")
    ap.add_argument("--infer_examples", type=int, default=12)

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

    if not (0 <= int(args.d21_internal) < C):
        raise ValueError(f"d21_internal fuera de rango: {args.d21_internal} (C={C})")

    # loaders
    dl_tr, dl_va, dl_te, ds_te = make_loaders(
        data_dir=data_dir,
        bs=args.batch_size,
        nw=args.num_workers,
        normalize=(not args.no_normalize),
    )

    # model
    model = PointNet2Seg(num_classes=C, dropout=float(args.dropout)).to(device)

    # loss weights (Opción A: bg incluido pero downweighted)
    w = torch.ones(C, device=device, dtype=torch.float32)
    w[bg] = float(args.bg_weight)
    loss_fn = nn.CrossEntropyLoss(weight=w)

    # optimizer + scheduler
    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(args.epochs), eta_min=1e-6)

    # ---- metadata run ----
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
        "warmup_epochs": int(args.warmup_epochs),
        "do_infer": bool(args.do_infer),
        "infer_examples": int(args.infer_examples),
    }
    save_json(run_meta, out_dir / "run_meta.json")

    # ---- CSV logging ----
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
        "lr"
    ]}

    t0 = time.time()
    for epoch in range(1, int(args.epochs) + 1):
        e0 = time.time()

        # warmup LR
        if int(args.warmup_epochs) > 0 and epoch <= int(args.warmup_epochs):
            lr_w = float(args.lr) * (epoch / float(args.warmup_epochs))
            for pg in opt.param_groups:
                pg["lr"] = lr_w

        tr = run_epoch(
            model=model, loader=dl_tr, optimizer=opt, loss_fn=loss_fn, C=C,
            d21_idx=int(args.d21_internal), device=device, bg=bg, train=True,
            use_amp=bool(args.use_amp), grad_clip=float(args.grad_clip),
        )
        va = run_epoch(
            model=model, loader=dl_va, optimizer=None, loss_fn=loss_fn, C=C,
            d21_idx=int(args.d21_internal), device=device, bg=bg, train=False,
            use_amp=False, grad_clip=None,
        )

        # cosine solo después del warmup
        if not (int(args.warmup_epochs) > 0 and epoch <= int(args.warmup_epochs)):
            sched.step()

        lr_now = float(opt.param_groups[0]["lr"])
        sec = time.time() - e0

        # historial
        history["train_loss"].append(tr["loss"]); history["val_loss"].append(va["loss"])
        history["train_acc_all"].append(tr["acc_all"]); history["val_acc_all"].append(va["acc_all"])
        history["train_acc_no_bg"].append(tr["acc_no_bg"]); history["val_acc_no_bg"].append(va["acc_no_bg"])
        history["train_f1m"].append(tr["f1_macro"]); history["val_f1m"].append(va["f1_macro"])
        history["train_ioum"].append(tr["iou_macro"]); history["val_ioum"].append(va["iou_macro"])
        history["val_d21_acc"].append(va["d21_acc"]); history["val_d21_f1"].append(va["d21_f1"]); history["val_d21_iou"].append(va["d21_iou"])
        history["val_d21_bin_acc_all"].append(va["d21_bin_acc_all"])
        history["train_pred_bg_frac"].append(tr["pred_bg_frac"]); history["val_pred_bg_frac"].append(va["pred_bg_frac"])
        history["lr"].append(lr_now)

        # warning colapso a BG
        if va["pred_bg_frac"] > max(0.95, bg_va + 0.12):
            print(f"[WARN] posible colapso a BG: val pred_bg_frac={va['pred_bg_frac']:.3f} (bg_gt≈{bg_va:.3f})")

        # CSV
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

        # checkpoints
        torch.save({"model": model.state_dict(), "epoch": epoch, "val_f1_macro": float(va["f1_macro"])}, last_path)
        if float(va["f1_macro"]) > best_val_f1:
            best_val_f1 = float(va["f1_macro"])
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_f1_macro": best_val_f1}, best_path)

        print(
            f"[{epoch:03d}/{int(args.epochs)}] "
            f"train loss={tr['loss']:.4f} f1m={tr['f1_macro']:.3f} ioum={tr['iou_macro']:.3f} "
            f"acc_all={tr['acc_all']:.3f} acc_no_bg={tr['acc_no_bg']:.3f} | "
            f"val loss={va['loss']:.4f} f1m={va['f1_macro']:.3f} ioum={va['iou_macro']:.3f} "
            f"acc_all={va['acc_all']:.3f} acc_no_bg={va['acc_no_bg']:.3f} | "
            f"d21(cls) acc={va['d21_acc']:.3f} f1={va['d21_f1']:.3f} iou={va['d21_iou']:.3f} | "
            f"d21(bin all) acc={va['d21_bin_acc_all']:.3f} | "
            f"pred_bg_frac(val)={va['pred_bg_frac']:.3f} lr={lr_now:.2e}"
        )

    # ---- guardar history ----
    save_json(history, out_dir / "history.json")

    # ---- plots simples ----
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
    _plot_curve(["lr"], "lr")

    # ---- test con best ----
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    te = run_epoch(
        model=model, loader=dl_te, optimizer=None, loss_fn=loss_fn, C=C,
        d21_idx=int(args.d21_internal), device=device, bg=bg, train=False,
        use_amp=False, grad_clip=None,
    )
    save_json({"best_epoch": int(ckpt.get("epoch", -1)), "test": te}, out_dir / "test_metrics.json")

    # ============================================================
    # INFERENCIA + TRAZABILIDAD (index_test.csv -> manifest)
    # ============================================================
    if bool(args.do_infer) and int(args.infer_examples) > 0:
        model.eval()
        k = min(int(args.infer_examples), len(ds_te))
        idxs = np.random.choice(len(ds_te), size=k, replace=False)

        out_all = out_dir / "inference_all"
        out_err = out_dir / "inference_errors"
        out_d21 = out_dir / "inference_d21"
        out_all.mkdir(parents=True, exist_ok=True)
        out_err.mkdir(parents=True, exist_ok=True)
        out_d21.mkdir(parents=True, exist_ok=True)

        # intenta descubrir index_test.csv en data_dir o arriba
        idx_csv_path = _discover_index_csv(data_dir, "test")
        idx_map = _read_index_csv(idx_csv_path) if idx_csv_path is not None else None

        manifest_path = out_dir / "inference_manifest.csv"
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            wcsv = csv.writer(f)
            wcsv.writerow([
                "row_id", "ds_index",
                "idx_global", "sample_name", "jaw", "path", "has_labels",
                "png_all", "png_errors", "png_d21"
            ])

            with torch.no_grad():
                for r, i in enumerate(idxs, start=1):
                    xyz, y = ds_te[int(i)]

                    xyz_b = xyz.unsqueeze(0).to(device, non_blocking=True)
                    logits = model(xyz_b)[0]  # [N,C]
                    pred = logits.argmax(dim=-1).detach().cpu().numpy().astype(np.int32)

                    xyz_np = xyz.detach().cpu().numpy().astype(np.float32, copy=True)
                    y_np = y.detach().cpu().numpy().astype(np.int32, copy=True)

                    # trazabilidad (si existe)
                    meta = {"idx_global": "", "sample_name": "", "jaw": "", "path": "", "has_labels": ""}
                    if idx_map is not None and int(i) in idx_map:
                        meta = idx_map[int(i)]

                    tag = f"ex_{r:02d}_i_{int(i):05d}"
                    if meta.get("sample_name"):
                        tag += f"_{_sanitize_tag(meta.get('sample_name', ''))}"

                    png_all = out_all / f"{tag}.png"
                    png_err = out_err / f"{tag}.png"
                    png_d21 = out_d21 / f"{tag}.png"

                    title = (
                        f"test ds_idx={int(i)} | best_epoch={int(ckpt.get('epoch', -1))} | "
                        f"C={C} | d21={int(args.d21_internal)} | "
                        f"sample={meta.get('sample_name', '')}"
                    )

                    # ✅ FIX: estas funciones ya fuerzan arrays safe, así que no debería crashear
                    plot_pointcloud_all_classes(xyz_np, y_np, pred, png_all, C=C, title=title, s=1.0)
                    plot_errors(xyz_np, y_np, pred, png_err, bg=bg, title=title, s=1.0)
                    plot_d21_focus(xyz_np, y_np, pred, png_d21, d21_idx=int(args.d21_internal), bg=bg, title=title, s=1.2)

                    wcsv.writerow([
                        r, int(i),
                        meta.get("idx_global", ""), meta.get("sample_name", ""), meta.get("jaw", ""),
                        meta.get("path", ""), meta.get("has_labels", ""),
                        str(png_all), str(png_err), str(png_d21)
                    ])

        print(f"[INFER] wrote manifest: {manifest_path}")
        if idx_csv_path is not None:
            print(f"[INFER] index_test.csv used: {idx_csv_path}")
        else:
            print("[INFER] index_test.csv not found (solo ds_index en manifest).")

    total = time.time() - t0
    print(f"[DONE] out_dir={out_dir} | total_sec={total:.1f} | best_val_f1_macro(no_bg)={best_val_f1:.4f} | C={C} | d21={int(args.d21_internal)}")


if __name__ == "__main__":
    main()
