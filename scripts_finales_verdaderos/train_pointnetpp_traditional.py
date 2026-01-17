# =========================
# PART 1/4 — Imports + Utils + Dataset
# Archivo sugerido: train_pointnetpp_traditional.py
# PointNet++ tradicional (SSG) para segmentación punto-a-punto sobre NPZ (X_[split].npz, Y_[split].npz)
# =========================

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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


# -----------------------------
# Repro + IO
# -----------------------------
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
    {"id2idx": {"0":0, "11":1, ...}, "idx2id": {"0":0, "1":11, ...}}
    """
    if not label_map_path.exists():
        return None, None
    try:
        data = json.load(open(label_map_path, "r", encoding="utf-8"))
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
                kk = int(k)
                if 0 <= kk < num_classes:
                    w[kk] = float(v)
            except Exception:
                continue
        return np.asarray(w, dtype=np.float32).copy()
    except Exception:
        return None


def normalize_unit_sphere(xyz: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    xyz: [N,3] o [B,N,3]
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


# -----------------------------
# Dataset NPZ
# -----------------------------
class NPZPointSegDataset(Dataset):
    def __init__(self, x_path: Path, y_path: Path, normalize: bool = True):
        x_obj = np.load(x_path)
        y_obj = np.load(y_path)

        self.X = np.asarray(x_obj["X"], dtype=np.float32)   # [B,N,3]
        self.Y = np.asarray(y_obj["Y"], dtype=np.int64)     # [B,N]

        assert self.X.shape[0] == self.Y.shape[0]
        assert self.X.ndim == 3 and self.X.shape[-1] == 3
        assert self.Y.ndim == 2
        assert self.X.shape[1] == self.Y.shape[1]

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

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True, drop_last=False)
    dl_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True, drop_last=False)
    dl_test  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True, drop_last=False)
    return dl_train, dl_val, dl_test, ds_test


# -----------------------------
# Métricas (macro desde CM) + binario d21
# -----------------------------
class MetricsBundle:
    def __init__(self, num_classes: int, device: torch.device, ignore_index: Optional[int] = 0):
        self.num_classes = int(num_classes)
        self.device = device
        self.ignore = ignore_index
        self.reset_cm()

    def reset_cm(self):
        self.cm = torch.zeros((self.num_classes, self.num_classes), device=self.device, dtype=torch.long)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, y_true: torch.Tensor):
        # logits: [B,N,C], y_true: [B,N]
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

    def compute_macro(self) -> Dict[str, float]:
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
    """
    pred, gt: [B,N] indices
    """
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

# =========================
# PART 2/4 — PointNet++ Ops (FPS, grouping) + Bloques SA/FP
# =========================

def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    src: [B, N, 3], dst: [B, M, 3]
    return: [B, N, M] dist^2
    """
    # (x - y)^2 = x^2 + y^2 - 2xy
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.transpose(2, 1))  # [B,N,M]
    dist += torch.sum(src ** 2, dim=-1, keepdim=True)   # [B,N,1]
    dist += torch.sum(dst ** 2, dim=-1).unsqueeze(1)    # [B,1,M]
    return dist


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    points: [B, N, C]
    idx: [B, S] o [B, S, K]
    return: [B, S, C] o [B, S, K, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, device=device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


@torch.no_grad()
def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    xyz: [B, N, 3]
    return idx: [B, npoint]
    FPS clásico O(B*N*npoint)
    """
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, device=device, dtype=torch.long)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)  # [B,N]
        distance = torch.minimum(distance, dist)
        farthest = torch.max(distance, dim=-1).indices
    return centroids


def query_ball_point(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """
    Agrupa vecinos en una esfera.
    xyz: [B, N, 3] puntos originales
    new_xyz: [B, S, 3] centros (sampled)
    return idx: [B, S, nsample]
    """
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape

    dist2 = square_distance(new_xyz, xyz)  # [B,S,N]
    # candidatos dentro de radio
    group_idx = torch.argsort(dist2, dim=-1)[:, :, :nsample]  # [B,S,nsample] (más cercanos)
    # máscara por radio
    group_dist2 = torch.gather(dist2, dim=-1, index=group_idx)
    mask = group_dist2 > (radius * radius)

    # si se salen del radio, reemplaza por el primer vecino (para evitar huecos)
    first = group_idx[:, :, 0:1].repeat(1, 1, nsample)
    group_idx = torch.where(mask, first, group_idx)
    return group_idx


def sample_and_group(npoint: int, radius: float, nsample: int,
                     xyz: torch.Tensor, points: Optional[torch.Tensor]):
    """
    xyz: [B, N, 3]
    points: [B, N, D] o None
    return:
      new_xyz: [B, npoint, 3]
      new_points: [B, npoint, nsample, 3 + D] (incluye coords normalizadas localmente)
    """
    B, N, C = xyz.shape
    fps_idx = farthest_point_sample(xyz, npoint)              # [B,npoint]
    new_xyz = index_points(xyz, fps_idx)                      # [B,npoint,3]
    idx = query_ball_point(radius, nsample, xyz, new_xyz)     # [B,npoint,nsample]
    grouped_xyz = index_points(xyz, idx)                      # [B,npoint,nsample,3]
    grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)     # local coords

    if points is not None:
        grouped_points = index_points(points, idx)            # [B,npoint,nsample,D]
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B,npoint,nsample,3+D]
    else:
        new_points = grouped_xyz_norm

    return new_xyz, new_points


def sample_and_group_all(xyz: torch.Tensor, points: Optional[torch.Tensor]):
    """
    Agrupa todo a un solo "centro".
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, 3, device=device)
    grouped_xyz = xyz.view(B, 1, N, 3)
    grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)

    if points is not None:
        new_points = torch.cat([grouped_xyz_norm, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz_norm
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    """
    SA layer PointNet++ (SSG)
    """
    def __init__(self, npoint: Optional[int], radius: Optional[float], nsample: Optional[int],
                 in_channel: int, mlp: List[int], group_all: bool):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        last_channel = in_channel
        convs = []
        bns = []
        for out_channel in mlp:
            convs.append(nn.Conv2d(last_channel, out_channel, 1))
            bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

    def forward(self, xyz: torch.Tensor, points: Optional[torch.Tensor]):
        """
        xyz: [B, N, 3]
        points: [B, N, D] o None
        return:
          new_xyz: [B, S, 3]
          new_points: [B, S, mlp[-1]]
        """
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)   # [B,1,N,3+D]
        else:
            assert self.npoint is not None and self.radius is not None and self.nsample is not None
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)

        # new_points: [B,S,nsample,C] -> [B,C,S,nsample]
        new_points = new_points.permute(0, 3, 1, 2).contiguous()

        for conv, bn in zip(self.convs, self.bns):
            new_points = F.relu(bn(conv(new_points)))

        # max pool sobre nsample
        new_points = torch.max(new_points, dim=-1).values  # [B,mlp[-1],S]
        new_points = new_points.permute(0, 2, 1).contiguous()  # [B,S,mlp[-1]]
        return new_xyz, new_points


class PointNetFeaturePropagation(nn.Module):
    """
    FP layer PointNet++ (upsampling + MLP 1x1)
    """
    def __init__(self, in_channel: int, mlp: List[int]):
        super().__init__()
        last_channel = in_channel
        convs = []
        bns = []
        for out_channel in mlp:
            convs.append(nn.Conv1d(last_channel, out_channel, 1))
            bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

    def forward(self, xyz1: torch.Tensor, xyz2: torch.Tensor,
                points1: Optional[torch.Tensor], points2: torch.Tensor):
        """
        Propaga de xyz2 (sparser) -> xyz1 (denser)

        xyz1: [B, N, 3] (denso)
        xyz2: [B, S, 3] (subsampled)
        points1: [B, N, D1] o None
        points2: [B, S, D2]

        return new_points: [B, N, mlp[-1]]
        """
        B, N, _ = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated = points2.repeat(1, N, 1)  # [B,N,D2]
        else:
            dist2 = square_distance(xyz1, xyz2)  # [B,N,S]
            # 3-NN
            dists, idx = torch.topk(dist2, k=3, dim=-1, largest=False, sorted=False)  # [B,N,3]
            dists = torch.clamp(dists, min=1e-10)
            inv = 1.0 / dists
            norm = torch.sum(inv, dim=-1, keepdim=True)
            weight = inv / norm  # [B,N,3]

            grouped_points = index_points(points2, idx)  # [B,N,3,D2]
            interpolated = torch.sum(grouped_points * weight.unsqueeze(-1), dim=2)  # [B,N,D2]

        if points1 is not None:
            new_points = torch.cat([points1, interpolated], dim=-1)  # [B,N,D1+D2]
        else:
            new_points = interpolated

        # [B,N,C] -> [B,C,N]
        new_points = new_points.permute(0, 2, 1).contiguous()
        for conv, bn in zip(self.convs, self.bns):
            new_points = F.relu(bn(conv(new_points)))
        new_points = new_points.permute(0, 2, 1).contiguous()  # [B,N,mlp[-1]]
        return new_points

# =========================
# PART 3/4 — Modelo PointNet++ tradicional (SSG) + Visualización
# =========================

# Paleta opcional por ID original (si tienes idx2id)
LABEL_COLORS = {
    0: "#B0B0B0",  # background/encía: gris
    11: "#1F77B4", 12: "#2CA02C", 13: "#FF7F0E", 14: "#9467BD", 15: "#17BECF",
    16: "#E377C2", 17: "#BCBD22", 18: "#8C564B",
    21: "#00C853",  # diente 21: verde bien visible
    22: "#1F3A93", 23: "#008080", 24: "#7F3C8D", 25: "#FA8072", 26: "#FFD700",
    27: "#87CEFA", 28: "#FF7F50", 31: "#808000", 32: "#C49C94", 33: "#AEC7E8",
    34: "#FFBB78", 35: "#C5B0D5", 36: "#9EDAE5", 37: "#F7B6D2", 38: "#DBDB8D",
    41: "#393B79", 42: "#637939", 43: "#8C6D31", 44: "#843C39", 45: "#7B4173",
    46: "#5254A3", 47: "#6B6ECF", 48: "#9C9EDE",
}


def colors_for_labels(lbl_internal: np.ndarray, idx2id: Optional[dict], num_classes: int) -> np.ndarray:
    lbl_internal = np.asarray(lbl_internal, dtype=np.int32)
    N = int(lbl_internal.shape[0])

    cmap = plt.get_cmap("tab20", num_classes)
    rgba = np.zeros((N, 4), dtype=np.float32)

    if idx2id is None:
        for i in range(N):
            rgba[i, :] = cmap(int(lbl_internal[i]))
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

    xyz = np.asarray(xyz, dtype=np.float32).copy()
    y_gt = np.asarray(y_gt, dtype=np.int32).copy()
    y_pr = np.asarray(y_pr, dtype=np.int32).copy()

    c_gt = np.asarray(colors_for_labels(y_gt, idx2id, num_classes), dtype=np.float32).copy()
    c_pr = np.asarray(colors_for_labels(y_pr, idx2id, num_classes), dtype=np.float32).copy()

    fig = plt.figure(figsize=(10, 5), dpi=200)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    for ax, c, t in [(ax1, c_gt, "GT"), (ax2, c_pr, "Pred")]:
        ax.scatter(xyz[:, 0].copy(), xyz[:, 1].copy(), xyz[:, 2].copy(),
                   c=c, s=1.2, linewidths=0, depthshade=False)
        ax.set_title(t, fontsize=10)
        ax.set_axis_off()
        ax.view_init(elev=20, azim=45)

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# PointNet++ Traditional (SSG) Segmentation
# -----------------------------
class PointNet2SegSSG(nn.Module):
    """
    Arquitectura clásica tipo PointNet++ SSG para segmentación:
    SA1 (N -> 1024) -> SA2 (1024 -> 256) -> SA3 (256 -> 64) -> SA4 (global)
    luego FP hacia atrás hasta N y 1x1 conv para logits
    """
    def __init__(self, num_classes: int, dropout: float = 0.5):
        super().__init__()
        # Input: xyz only. points=None en SA1, así que in_channel = 3 (coords local)
        # SA1: npoint=1024, radius=0.1, nsample=32
        self.sa1 = PointNetSetAbstraction(
            npoint=1024, radius=0.10, nsample=32,
            in_channel=3, mlp=[32, 32, 64], group_all=False
        )
        # SA2: input points=64, concat coords local(3) => in_channel=64+3
        self.sa2 = PointNetSetAbstraction(
            npoint=256, radius=0.20, nsample=32,
            in_channel=64+3, mlp=[64, 64, 128], group_all=False
        )
        # SA3
        self.sa3 = PointNetSetAbstraction(
            npoint=64, radius=0.40, nsample=32,
            in_channel=128+3, mlp=[128, 128, 256], group_all=False
        )
        # SA4 global
        self.sa4 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None,
            in_channel=256+3, mlp=[256, 512, 1024], group_all=True
        )

        # Feature Propagation
        self.fp4 = PointNetFeaturePropagation(in_channel=1024 + 256, mlp=[256, 256])
        self.fp3 = PointNetFeaturePropagation(in_channel=256 + 128,  mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=256 + 64,   mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128 + 0,    mlp=[128, 128, 128])

        self.drop = nn.Dropout(dropout)
        self.cls1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.cls2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        xyz: [B,N,3]
        return logits: [B,N,C]
        """
        B, N, _ = xyz.shape

        l0_xyz = xyz
        l0_points = None

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)  # [B,1024,3], [B,1024,64]
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # [B,256,3],  [B,256,128]
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # [B,64,3],   [B,64,256]
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)  # [B,1,3],    [B,1,1024]

        # FP: (xyz1=denso, xyz2=menos denso)
        l3_points_new = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)  # [B,64,256]
        l2_points_new = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points_new)  # [B,256,256]
        l1_points_new = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points_new)  # [B,1024,128]
        l0_points_new = self.fp1(l0_xyz, l1_xyz, None, l1_points_new)       # [B,N,128]

        # Clasificación punto a punto
        x = l0_points_new.permute(0, 2, 1).contiguous()  # [B,128,N]
        x = F.relu(self.bn1(self.cls1(x)))
        x = self.drop(x)
        x = self.cls2(x)                                 # [B,C,N]
        x = x.permute(0, 2, 1).contiguous()              # [B,N,C]
        return x

# =========================
# PART 4/4 — Train/Eval + CLI + Inference
# =========================

def run_epoch(model, loader, criterion, optimizer, device,
              num_classes: int, ignore_index: Optional[int], d21_idx: Optional[int], train: bool):
    if train:
        model.train()
    else:
        model.eval()

    mb = MetricsBundle(num_classes=num_classes, device=device, ignore_index=ignore_index)
    loss_meter = 0.0
    n_batches = 0

    d21_acc = d21_f1 = d21_iou = 0.0
    d21_count = 0

    for xyz, y in loader:
        xyz = xyz.to(device, non_blocking=True)  # [B,N,3]
        y = y.to(device, non_blocking=True)      # [B,N]

        if torch.any((y < 0) | (y >= num_classes)):
            raise ValueError(f"[LABEL ERROR] y fuera de rango: min={int(y.min())}, max={int(y.max())}, C={num_classes}")

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
            d21_f1  += dm["f1"]
            d21_iou += dm["iou"]
            d21_count += 1

    macro = mb.compute_macro()
    out = {"loss": loss_meter / max(1, n_batches), **macro}

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
                    help="Carpeta con X_train.npz/Y_train.npz/X_val.npz/Y_val.npz/X_test.npz/Y_test.npz y ojalá label_map.json")
    ap.add_argument("--artifacts_dir", type=str, default=None,
                    help="Carpeta con class_weights.json (default=data_dir)")
    ap.add_argument("--out_dir", type=str, required=True, help="Salida (runs/...)")

    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--batch_size", type=int, default=4)  # PointNet++ suele ser más pesado
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.5)

    ap.add_argument("--ignore_index", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda", help="cuda|cpu")

    ap.add_argument("--infer_examples", type=int, default=10)
    ap.add_argument("--no_normalize", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts_dir = Path(args.artifacts_dir) if args.artifacts_dir else data_dir

    device = torch.device("cuda") if (args.device == "cuda" and torch.cuda.is_available()) else torch.device("cpu")

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

    # Modelo PointNet++
    model = PointNet2SegSSG(num_classes=num_classes, dropout=args.dropout).to(device)

    # Pesos por clase (opcional)
    cw = compute_class_weights_from_json(artifacts_dir, num_classes)
    weight_tensor = None
    if cw is not None:
        weight_tensor = torch.as_tensor(cw, dtype=torch.float32, device=device)

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
        "d21_internal_idx": (int(d21_idx) if d21_idx is not None else None),
        "normalize_unit_sphere": bool(not args.no_normalize),
        "model": "PointNet++ Traditional SSG",
        "notes": "SA radii/nsample: (0.10,32)->(0.20,32)->(0.40,32)->global; npoint: 1024,256,64"
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

    save_json(history, out_dir / "history.json")
    plot_curves(history, out_dir / "plots", model_name="PointNetPP_Traditional")

    # Test con best
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    te = run_epoch(model, dl_test, criterion, optimizer, device, num_classes, args.ignore_index, d21_idx, train=False)
    save_json({"best_epoch": int(ckpt.get("epoch", -1)), "test": te}, out_dir / "test_metrics.json")

    # Inference K ejemplos
    if args.infer_examples > 0:
        model.eval()
        inf_dir = out_dir / "inference"
        inf_dir.mkdir(parents=True, exist_ok=True)

        n = len(ds_test)
        k = min(int(args.infer_examples), int(n))
        indices = np.random.choice(n, size=k, replace=False)

        with torch.no_grad():
            for rank, i in enumerate(indices, start=1):
                xyz, y = ds_test[int(i)]
                xyz_b = xyz.unsqueeze(0).to(device)  # [1,N,3]
                logits = model(xyz_b)[0]            # [N,C]
                pred = logits.argmax(dim=-1).detach().cpu().numpy().astype(np.int32)

                xyz_np = xyz.detach().cpu().numpy().astype(np.float32, copy=True)
                y_np = y.detach().cpu().numpy().astype(np.int32, copy=True)

                out_png = inf_dir / f"ex_{rank:02d}_idx_{int(i):05d}.png"
                plot_pointcloud_gt_pred(
                    xyz=xyz_np,
                    y_gt=y_np,
                    y_pr=pred,
                    out_png=out_png,
                    idx2id=idx2id,
                    num_classes=num_classes,
                    title=f"PointNet++ Traditional | test idx={int(i)} | best_epoch={int(ckpt.get('epoch',-1))}"
                )

    total = time.time() - t0
    print(f"[DONE] out_dir={out_dir} | total_sec={total:.1f} | best_val_f1={best_val_f1:.4f}")


if __name__ == "__main__":
    main()
