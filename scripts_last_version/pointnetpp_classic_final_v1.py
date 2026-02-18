#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pointnetpp_classic_final_v1.py

PointNet++ clásico (SSG) – Segmentación multiclase dental 3D
(Adaptación directa desde pointnet_classic_final_v5.py manteniendo TODA la trazabilidad/salidas)

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

FIXES IMPORTANTES (2026-01):
✅ DataLoader crash (torch.from_numpy): reemplazado por torch.as_tensor + np.ascontiguousarray
✅ Matplotlib/may_share_memory crash: convierte colores a float32 y xyz a float32 antes de scatter

(NEW v3):
✅ NO graficar TEST como línea horizontal (solo Train vs Val + best_epoch vertical)
✅ Inferencia arreglada (helpers _discover_index_csv/_read_index_csv/_sanitize_tag definidos)

(NEW v4):
✅ Train vs Val comparable: métricas de TRAIN se calculan con forward en eval() (dropout OFF),
   mientras el update/backprop se hace en train() (dropout ON). Evita el gap "train bajo / val alto"
   que era solo artefacto de medir train con dropout activo.
✅ Trazabilidad: añade --index_csv (opcional) para forzar el index_{split}.csv correcto si el auto-discovery
   no calza con el dataset actual.

(NEW v5):
✅ Soporte REAL de “neighbor teeth metrics”:
   - Flag --neighbor_teeth "d11:1,d22:9,..." (lista arbitraria nombre:idx)
   - Loggea métricas binarias (acc/f1/iou + bin_acc_all) para cada vecino
   - Integra en CSV por epoch, history.json, test_metrics.json

(NEW v1 PointNet++):
✅ Se reemplaza SOLO el backbone (PointNet clásico -> PointNet++ SSG con SA/FP),
   manteniendo dataset, métricas, logging, outputs, inferencia y trazabilidad idénticos.

Dataset esperado:
  data_dir/X_train.npz, Y_train.npz, X_val.npz, Y_val.npz, X_test.npz, Y_test.npz
  X: [B,N,3], Y: [B,N] con clases internas 0..C-1 (0=bg)
"""

import os
import re
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


def _fmt_hms(seconds: float) -> str:
    seconds = float(seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:d}h {m:02d}m {s:05.2f}s"


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
    """
    Carga X_*.npz / Y_*.npz donde:
      X: [B,N,3] float32
      Y: [B,N]   int64
    """
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

    def __getitem__(self, i: int):
        # FIX robusto: evita torch.from_numpy en workers (crash raro "expected np.ndarray")
        x = np.ascontiguousarray(self.X[int(i)], dtype=np.float32)  # [N,3]
        y = np.ascontiguousarray(self.Y[int(i)], dtype=np.int64)    # [N]
        xyz = torch.as_tensor(x, dtype=torch.float32)
        lab = torch.as_tensor(y, dtype=torch.int64)
        if self.normalize:
            xyz = normalize_unit_sphere(xyz)
        return xyz, lab


def make_loaders(data_dir: Path, bs: int, nw: int, normalize: bool = True):
    ds_tr = NPZDataset(data_dir / "X_train.npz", data_dir / "Y_train.npz", normalize=normalize)
    ds_va = NPZDataset(data_dir / "X_val.npz",   data_dir / "Y_val.npz",   normalize=normalize)
    ds_te = NPZDataset(data_dir / "X_test.npz",  data_dir / "Y_test.npz",  normalize=normalize)

    common = dict(
        batch_size=int(bs),
        num_workers=int(nw),
        pin_memory=True,
        persistent_workers=(int(nw) > 0),
        drop_last=False,
    )
    if int(nw) > 0:
        common["prefetch_factor"] = 2

    dl_tr = DataLoader(ds_tr, shuffle=True,  **common)
    dl_va = DataLoader(ds_va, shuffle=False, **common)
    dl_te = DataLoader(ds_te, shuffle=False, **common)
    return dl_tr, dl_va, dl_te, ds_te


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# PARTE 2/5:
#   - Implementación PointNet++ SSG (FPS + grouping + SA + FP)
#   - Modelo PointNet2Seg que devuelve logits [B,N,C]
# Manteniendo exactamente el contrato del backbone anterior.
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# ============================================================
# POINTNET++ (SSG) – IMPLEMENTACIÓN SELF-CONTAINED (sin libs externas)
# ============================================================
# Nota: Esto es un PointNet++ "clásico" estilo Qi et al. (2017):
# - Set Abstraction (SA): FPS + ball query (radius) + PointNet local + maxpool
# - Feature Propagation (FP): interpolación 3-NN + MLP 1D
# Contracto final: forward(xyz[B,N,3]) -> logits[B,N,C]
#
# Mantiene dropout configurable en el head (igual filosofía que el PointNet v5),
# y NO toca nada del pipeline de métricas/logging/inferencia.

def _square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    src: [B, N, 3], dst: [B, M, 3]
    return: [B, N, M] dist^2
    """
    # (x-y)^2 = x^2 + y^2 - 2xy
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2.0 * torch.matmul(src, dst.transpose(2, 1))  # [B,N,M]
    dist += torch.sum(src ** 2, dim=-1, keepdim=True)      # [B,N,1]
    dist += torch.sum(dst ** 2, dim=-1).unsqueeze(1)       # [B,1,M]
    return dist


def _index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    points: [B, N, C]
    idx:    [B, S] o [B, S, K]
    return: [B, S, C] o [B, S, K, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    xyz: [B, N, 3]
    return: idx [B, npoint]
    """
    device = xyz.device
    B, N, _ = xyz.shape
    npoint = int(npoint)
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device, dtype=xyz.dtype)
    farthest = torch.randint(0, N, (B,), device=device, dtype=torch.long)
    batch_indices = torch.arange(B, device=device, dtype=torch.long)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)   # [B,1,3]
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)            # [B,N]
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=1)[1]
    return centroids


def query_ball_point(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """
    radius: float
    nsample: int
    xyz: [B, N, 3] (all points)
    new_xyz: [B, S, 3] (centroids)
    return: group_idx [B, S, nsample]
    """
    radius = float(radius)
    nsample = int(nsample)
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape
    device = xyz.device

    sqrdists = _square_distance(new_xyz, xyz)  # [B,S,N]
    group_idx = torch.arange(N, device=device, dtype=torch.long).view(1, 1, N).repeat(B, S, 1)

    # mask puntos fuera de la esfera
    group_idx[sqrdists > (radius * radius)] = N  # N = dummy

    # ordenar por idx (los N quedarán al final), tomar primeros nsample
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]  # [B,S,nsample]

    # si algún centroid no tiene suficientes vecinos (todo N), rellena con el primero válido
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat(1, 1, nsample)
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint: int, radius: float, nsample: int, xyz: torch.Tensor, points: Optional[torch.Tensor]):
    """
    npoint: #centroids
    radius: ball query radius
    nsample: #neighbors
    xyz: [B,N,3]
    points: [B,N,D] o None
    return:
      new_xyz: [B,S,3]
      new_points: [B,S,nsample,3+D] (incluye xyz relativo + features si existen)
    """
    B, N, _ = xyz.shape
    S = int(npoint)

    fps_idx = farthest_point_sample(xyz, S)              # [B,S]
    new_xyz = _index_points(xyz, fps_idx)                # [B,S,3]

    idx = query_ball_point(radius, nsample, xyz, new_xyz)  # [B,S,nsample]
    grouped_xyz = _index_points(xyz, idx)                   # [B,S,nsample,3]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, 3)

    if points is not None:
        grouped_points = _index_points(points, idx)         # [B,S,nsample,D]
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B,S,nsample,3+D]
    else:
        new_points = grouped_xyz_norm  # [B,S,nsample,3]

    return new_xyz, new_points


def sample_and_group_all(xyz: torch.Tensor, points: Optional[torch.Tensor]):
    """
    Agrupa todo en un solo "centroid" (global SA)
    return:
      new_xyz: [B,1,3] (dummy: mean)
      new_points: [B,1,N,3+D]
    """
    device = xyz.device
    B, N, _ = xyz.shape
    new_xyz = xyz.mean(dim=1, keepdim=True)  # [B,1,3]
    grouped_xyz = xyz.view(B, 1, N, 3)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, 1, 1, 3)

    if points is not None:
        grouped_points = points.view(B, 1, N, -1)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    """
    SA layer:
      - si group_all=True: global SA (sin FPS/ball query)
      - si no: FPS + ball query
    mlp: lista de canales (salida final = último)
    """
    def __init__(self, npoint: int, radius: float, nsample: int,
                 in_channel: int, mlp: List[int], group_all: bool):
        super().__init__()
        self.npoint = int(npoint)
        self.radius = float(radius)
        self.nsample = int(nsample)
        self.group_all = bool(group_all)

        last = int(in_channel)
        convs = []
        bns = []
        for out_ch in mlp:
            convs.append(nn.Conv2d(last, int(out_ch), kernel_size=1))
            bns.append(nn.BatchNorm2d(int(out_ch)))
            last = int(out_ch)
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

    def forward(self, xyz: torch.Tensor, points: Optional[torch.Tensor]):
        """
        xyz: [B,N,3]
        points: [B,N,D] o None
        return:
          new_xyz: [B,S,3]
          new_points: [B,S,D']  (pooled)
        """
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)  # [B,1,3], [B,1,N,3+D]
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
            # new_xyz: [B,S,3], new_points: [B,S,nsample,3+D]

        # para PointNet local: [B, in_channel, nsample, S]
        new_points = new_points.permute(0, 3, 2, 1).contiguous()

        for conv, bn in zip(self.convs, self.bns):
            new_points = F.relu(bn(conv(new_points)))

        # maxpool sobre nsample -> [B, D', 1, S]
        new_points = torch.max(new_points, 2, keepdim=True)[0]
        new_points = new_points.squeeze(2).permute(0, 2, 1).contiguous()  # [B,S,D']
        return new_xyz, new_points


class PointNetFeaturePropagation(nn.Module):
    """
    FP layer:
      - Interpola features de xyz2 -> xyz1 (3-NN)
      - Concat con puntos skip (si existe)
      - MLP 1D (Conv1d)
    """
    def __init__(self, in_channel: int, mlp: List[int]):
        super().__init__()
        last = int(in_channel)
        convs = []
        bns = []
        for out_ch in mlp:
            convs.append(nn.Conv1d(last, int(out_ch), kernel_size=1))
            bns.append(nn.BatchNorm1d(int(out_ch)))
            last = int(out_ch)
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

    def forward(self, xyz1: torch.Tensor, xyz2: torch.Tensor,
                points1: Optional[torch.Tensor], points2: torch.Tensor):
        """
        xyz1: [B,N,3]  (target)
        xyz2: [B,S,3]  (source, S<=N)
        points1: [B,N,D1] o None (skip)
        points2: [B,S,D2]        (source feats)
        return:
          new_points: [B,N,D']
        """
        B, N, _ = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated = points2.repeat(1, N, 1)  # [B,N,D2]
        else:
            dists = _square_distance(xyz1, xyz2)  # [B,N,S]
            dists, idx = dists.sort(dim=-1)
            dists = dists[:, :, :3]  # [B,N,3]
            idx = idx[:, :, :3]      # [B,N,3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm  # [B,N,3]

            grouped_points = _index_points(points2, idx)  # [B,N,3,D2]
            interpolated = torch.sum(grouped_points * weight.unsqueeze(-1), dim=2)  # [B,N,D2]

        if points1 is not None:
            new_points = torch.cat([points1, interpolated], dim=-1)  # [B,N,D1+D2]
        else:
            new_points = interpolated

        # MLP 1D: [B, D, N]
        new_points = new_points.permute(0, 2, 1).contiguous()
        for conv, bn in zip(self.convs, self.bns):
            new_points = F.relu(bn(conv(new_points)))
        new_points = new_points.permute(0, 2, 1).contiguous()  # [B,N,D']
        return new_points


class PointNet2Seg(nn.Module):
    """
    PointNet++ SSG para segmentación.
    Entrada: xyz [B,N,3]
    Salida: logits [B,N,C]
    """
    def __init__(
        self,
        num_classes: int,
        dropout: float = 0.5,
        # hiperparams backbone (exponibles por argparse en main, pero default paper-like)
        npoints: Tuple[int, int, int] = (1024, 256, 64),
        radii: Tuple[float, float, float] = (0.10, 0.20, 0.40),
        nsamples: Tuple[int, int, int] = (32, 32, 32),
        # MLPs SA (in_channel incluye xyz_rel(3) + feats_prev)
        sa_mlps: Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]] = (
            (64, 64, 128),     # SA1
            (128, 128, 256),   # SA2
            (256, 256, 512),   # SA3
            (512, 1024),       # SA4 global
        ),
        fp_mlps: Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]] = (
            (256, 256),        # FP4 (l3<-l4)
            (256, 256),        # FP3 (l2<-l3)
            (256, 128),        # FP2 (l1<-l2)
            (128, 128, 128),   # FP1 (l0<-l1)
        ),
    ):
        super().__init__()
        C = int(num_classes)
        self.C = C
        self.dropout = float(dropout)

        n1, n2, n3 = [int(x) for x in npoints]
        r1, r2, r3 = [float(x) for x in radii]
        k1, k2, k3 = [int(x) for x in nsamples]

        # SA layers
        # Nivel 0: xyz solamente => points=None => in_channel para SA1 = 3 (xyz_rel)
        self.sa1 = PointNetSetAbstraction(
            npoint=n1, radius=r1, nsample=k1,
            in_channel=3, mlp=list(sa_mlps[0]), group_all=False
        )
        # SA2: concat xyz_rel(3) + feats(128) => 3+128
        self.sa2 = PointNetSetAbstraction(
            npoint=n2, radius=r2, nsample=k2,
            in_channel=3 + int(sa_mlps[0][-1]), mlp=list(sa_mlps[1]), group_all=False
        )
        # SA3: 3+256
        self.sa3 = PointNetSetAbstraction(
            npoint=n3, radius=r3, nsample=k3,
            in_channel=3 + int(sa_mlps[1][-1]), mlp=list(sa_mlps[2]), group_all=False
        )
        # SA4 global: 3+512
        self.sa4 = PointNetSetAbstraction(
            npoint=1, radius=0.0, nsample=0,
            in_channel=3 + int(sa_mlps[2][-1]), mlp=list(sa_mlps[3]), group_all=True
        )

        # FP layers (orden inverso)
        # FP4: l3 (512) + up(l4=1024) => 512+1024
        self.fp4 = PointNetFeaturePropagation(
            in_channel=int(sa_mlps[2][-1]) + int(sa_mlps[3][-1]),
            mlp=list(fp_mlps[0])
        )
        # FP3: l2 (256) + up(fp4=256) => 256+256
        self.fp3 = PointNetFeaturePropagation(
            in_channel=int(sa_mlps[1][-1]) + int(fp_mlps[0][-1]),
            mlp=list(fp_mlps[1])
        )
        # FP2: l1 (128) + up(fp3=256) => 128+256
        self.fp2 = PointNetFeaturePropagation(
            in_channel=int(sa_mlps[0][-1]) + int(fp_mlps[1][-1]),
            mlp=list(fp_mlps[2])
        )
        # FP1: l0 (no feats) + up(fp2=128) => 128
        self.fp1 = PointNetFeaturePropagation(
            in_channel=int(fp_mlps[2][-1]),
            mlp=list(fp_mlps[3])
        )

        # Head de segmentación (paper-like)
        self.conv1 = nn.Conv1d(int(fp_mlps[3][-1]), 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(self.dropout)
        self.conv2 = nn.Conv1d(128, C, 1)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        xyz: [B,N,3]
        return logits: [B,N,C]
        """
        B, N, _ = xyz.shape
        l0_xyz = xyz
        l0_points = None

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)  # [B,n1,3], [B,n1,128]
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # [B,n2,3], [B,n2,256]
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # [B,n3,3], [B,n3,512]
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)  # [B,1,3],  [B,1,1024]

        l3_points_fp = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)        # [B,n3,256]
        l2_points_fp = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points_fp)     # [B,n2,256]
        l1_points_fp = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points_fp)     # [B,n1,128]
        l0_points_fp = self.fp1(l0_xyz, l1_xyz, None, l1_points_fp)          # [B,N,128]

        x = l0_points_fp.permute(0, 2, 1).contiguous()  # [B,128,N]
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.drop1(x)
        x = self.conv2(x)                               # [B,C,N]
        logits = x.permute(0, 2, 1).contiguous()         # [B,N,C]
        return logits


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# PARTE 3/5:
#   - Métricas (macro sin bg, d21 binario, neighbors)
#   - Visualización (all/errors/d21) + helpers trazabilidad
# (Esta parte será prácticamente idéntica a tu v5, solo pegada tal cual)
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# ============================================================
# MÉTRICAS (macro sin bg) + d21 binario
# ============================================================
@torch.no_grad()
def macro_metrics_no_bg(pred: torch.Tensor, gt: torch.Tensor, C: int, bg: int = 0) -> Tuple[float, float]:
    """
    Macro-F1 e IoU macro calculados EXCLUYENDO BG (gt!=bg), promediando sobre clases 1..C-1.
    """
    pred = pred.reshape(-1)
    gt = gt.reshape(-1)

    mask = (gt != int(bg))
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
            # clase ausente -> la omitimos del macro (estilo robusto)
            continue

        f1 = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        f1s.append(f1)
        ious.append(iou)

    if len(f1s) == 0:
        return 0.0, 0.0
    return float(np.mean(f1s)), float(np.mean(ious))


@torch.no_grad()
def d21_metrics_binary(
    pred: torch.Tensor,
    gt: torch.Tensor,
    d21_idx: int,
    bg: int = 0,
    include_bg: bool = False
) -> Tuple[float, float, float]:
    """
    d21 como binario: positivo = clase d21_idx, negativo = resto.
    include_bg=False => excluye puntos bg del cálculo (métrica principal)
    include_bg=True  => incluye bg también (referencia)
    """
    pred = pred.reshape(-1)
    gt = gt.reshape(-1)

    if not bool(include_bg):
        mask = (gt != int(bg))
        pred = pred[mask]
        gt = gt[mask]
        if gt.numel() == 0:
            return 0.0, 0.0, 0.0

    t_pos = (gt == int(d21_idx))
    p_pos = (pred == int(d21_idx))

    tp = (p_pos & t_pos).sum().item()
    fp = (p_pos & (~t_pos)).sum().item()
    fn = ((~p_pos) & t_pos).sum().item()
    tn = ((~p_pos) & (~t_pos)).sum().item()

    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    f1 = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    return float(acc), float(f1), float(iou)


def _mean_std(xs: List[float]) -> Tuple[float, float]:
    if xs is None or len(xs) == 0:
        return 0.0, 0.0
    a = np.asarray(xs, dtype=np.float64)
    return float(a.mean()), float(a.std(ddof=0))


@torch.no_grad()
def _acc_all(pred: torch.Tensor, gt: torch.Tensor) -> float:
    return float((pred == gt).float().mean().item())


# ============================================================
# VISUALIZACIÓN (robusta para numpy/torch)
# ============================================================
def _class_colors(C: int):
    cmap = plt.colormaps.get_cmap("tab20")
    C = max(int(C), 2)
    cols = [cmap(i / max(C - 1, 1)) for i in range(C)]
    return cols


def _to_np(a) -> np.ndarray:
    """
    Matplotlib 3D a veces se pone mañoso si llega un objeto raro (memmap/subclase/etc).
    Esto fuerza un np.ndarray "normal" y contiguo.
    """
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    a = np.asarray(a)
    if not a.flags["C_CONTIGUOUS"]:
        a = np.ascontiguousarray(a)
    return a


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

    xyz = _to_np(xyz).astype(np.float32, copy=False)
    y_gt = _to_np(y_gt).astype(np.int32, copy=False).reshape(-1)
    y_pr = _to_np(y_pr).astype(np.int32, copy=False).reshape(-1)

    cols = _class_colors(C)
    c_gt = np.array([cols[int(k)] for k in y_gt], dtype=np.float32)
    c_pr = np.array([cols[int(k)] for k in y_pr], dtype=np.float32)

    fig = plt.figure(figsize=(12, 5), dpi=200)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

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

    xyz = _to_np(xyz).astype(np.float32, copy=False)
    y_gt = _to_np(y_gt).astype(np.int32, copy=False).reshape(-1)
    y_pr = _to_np(y_pr).astype(np.int32, copy=False).reshape(-1)

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

    xyz = _to_np(xyz).astype(np.float32, copy=False)
    y_gt = _to_np(y_gt).astype(np.int32, copy=False).reshape(-1)
    y_pr = _to_np(y_pr).astype(np.int32, copy=False).reshape(-1)

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
# HELPERS (vecinos, plots Train/Val SIN testline, trazabilidad)
# ============================================================
@torch.no_grad()
def _tooth_metrics_binary(pred: torch.Tensor, gt: torch.Tensor, tooth_idx: int, bg: int = 0) -> Dict[str, float]:
    acc, f1, iou = d21_metrics_binary(pred, gt, d21_idx=int(tooth_idx), bg=int(bg), include_bg=False)
    acc_all, f1_all, iou_all = d21_metrics_binary(pred, gt, d21_idx=int(tooth_idx), bg=int(bg), include_bg=True)
    return {
        "acc": float(acc), "f1": float(f1), "iou": float(iou),
        "bin_acc_all": float(acc_all), "bin_f1_all": float(f1_all), "bin_iou_all": float(iou_all),
    }


def parse_neighbor_teeth(spec: Optional[str]) -> List[Tuple[str, int]]:
    """
    Parse de --neighbor_teeth con formato:
      "d11:1,d22:9,foo:3"

    Devuelve lista [(name, idx), ...] (orden estable).
    Reglas:
      - tokens separados por coma
      - cada token: name:idx (idx int)
      - ignora tokens vacíos y espacios
      - si hay duplicados por nombre, se queda el último (estilo override)
    """
    if spec is None:
        return []
    s = str(spec).strip()
    if not s:
        return []
    out: List[Tuple[str, int]] = []
    seen = {}
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" not in tok:
            # si viene mal formateado, lo ignoramos silenciosamente (robusto)
            continue
        name, idxs = tok.split(":", 1)
        name = name.strip()
        idxs = idxs.strip()
        if not name:
            continue
        try:
            idx = int(idxs)
        except Exception:
            continue
        seen[name] = idx
    # preserva orden de aparición: recorre original y arma con último idx
    used = set()
    for tok in s.split(","):
        tok = tok.strip()
        if ":" not in tok:
            continue
        name = tok.split(":", 1)[0].strip()
        if name in seen and name not in used:
            out.append((name, int(seen[name])))
            used.add(name)
    return out


def eval_neighbors_on_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    neighbor_list: List[Tuple[str, int]],
    bg: int
) -> Dict[str, float]:
    if not neighbor_list:
        return {}
    model.eval()
    sums = {f"{name}_{k}": 0.0 for name, _ in neighbor_list for k in ("acc", "f1", "iou", "bin_acc_all")}
    nb = 0
    with torch.no_grad():
        for xyz, y in loader:
            xyz = xyz.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(xyz)
            pred = logits.argmax(dim=-1)
            for name, idx in neighbor_list:
                m = _tooth_metrics_binary(pred, y, tooth_idx=idx, bg=bg)
                sums[f"{name}_acc"] += m["acc"]
                sums[f"{name}_f1"] += m["f1"]
                sums[f"{name}_iou"] += m["iou"]
                sums[f"{name}_bin_acc_all"] += m["bin_acc_all"]
            nb += 1
    nb = max(1, nb)
    return {k: v / nb for k, v in sums.items()}


def plot_train_val(name: str, y_tr: List[float], y_va: List[float], out_png: Path, best_epoch: Optional[int] = None):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(y_tr, label="train")
    plt.plot(y_va, label="val")
    if best_epoch is not None and int(best_epoch) > 0:
        plt.axvline(int(best_epoch) - 1, linestyle=":", label=f"best_epoch={int(best_epoch)}")
    plt.xlabel("epoch")
    plt.ylabel(name)
    plt.title(f"{name} (Train vs Val)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=250)
    plt.close()


def _sanitize_tag(s: str, maxlen: int = 80) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    s = s.replace(" ", "_")
    s = re.sub(r"[^a-zA-Z0-9_\-\.]+", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if len(s) > int(maxlen):
        s = s[: int(maxlen)]
    return s


def _read_index_csv(index_path: Path) -> Optional[Dict[int, Dict[str, str]]]:
    """
    Devuelve un mapa: row_i (int) -> dict con keys:
      idx_global, sample_name, jaw, path, has_labels
    Acepta encabezados flexibles (solo requiere row_i).
    """
    if index_path is None or not Path(index_path).exists():
        return None

    with open(index_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return None

        fields = {h.strip(): h for h in reader.fieldnames}

        row_key = None
        for k in ("row_i", "row", "i", "idx", "index"):
            if k in fields:
                row_key = fields[k]
                break
        if row_key is None:
            for h in reader.fieldnames:
                if h.strip().lower() in ("row_i", "row", "index", "idx"):
                    row_key = h
                    break
        if row_key is None:
            return None

        def _pick(*cands):
            for c in cands:
                if c in fields:
                    return fields[c]
            for h in reader.fieldnames:
                if h.strip().lower() in [cc.lower() for cc in cands]:
                    return h
            return None

        k_idxg = _pick("idx_global", "global_idx", "global_id", "patient_global_idx")
        k_name = _pick("sample_name", "sample", "name", "patient", "patient_id", "scan_id")
        k_jaw  = _pick("jaw", "arch", "upperlower", "part")
        k_path = _pick("path", "source_path", "src_path", "mesh_path", "file_path")
        k_lab  = _pick("has_labels", "labels", "has_gt", "has_y")

        mp: Dict[int, Dict[str, str]] = {}
        for row in reader:
            try:
                ri = int(str(row.get(row_key, "")).strip())
            except Exception:
                continue
            mp[ri] = {
                "idx_global": str(row.get(k_idxg, "")).strip() if k_idxg else "",
                "sample_name": str(row.get(k_name, "")).strip() if k_name else "",
                "jaw": str(row.get(k_jaw, "")).strip() if k_jaw else "",
                "path": str(row.get(k_path, "")).strip() if k_path else "",
                "has_labels": str(row.get(k_lab, "")).strip() if k_lab else "",
            }
        return mp if len(mp) > 0 else None


def _discover_index_csv(data_dir: Path, split: str) -> Optional[Path]:
    """
    Busca index_{split}.csv en:
      1) data_dir/index_{split}.csv
      2) ancestros de data_dir (hasta 'Teeth_3ds')
      3) Teeth_3ds/merged_*/index_{split}.csv (más reciente por mtime)
    """
    split = str(split).strip().lower()
    fname = f"index_{split}.csv"

    p1 = data_dir / fname
    if p1.exists():
        return p1

    cur = data_dir.resolve()
    for _ in range(12):
        cand = cur / fname
        if cand.exists():
            return cand
        if cur.name.lower() == "teeth_3ds":
            break
        if cur.parent == cur:
            break
        cur = cur.parent

    cur = data_dir.resolve()
    teeth_root = None
    for _ in range(20):
        if cur.name.lower() == "teeth_3ds":
            teeth_root = cur
            break
        if cur.parent == cur:
            break
        cur = cur.parent

    if teeth_root is None:
        return None

    merged = list(teeth_root.glob("merged_*"))
    cands = []
    for m in merged:
        if m.is_dir():
            p = m / fname
            if p.exists():
                cands.append(p)

    if not cands:
        return None

    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def _resolve_tooth_internal_from_label_map(data_dir: Path, target_fdi: int) -> Optional[int]:
    """
    Intenta mapear FDI -> clase interna usando label_map.json si existe.
    Acepta formatos comunes:
      - {"internal_to_fdi": {"1":11, "2":12, ...}}
      - {"idx2id": {"1":"11", ...}} o {"idx2fdi": {...}}
      - {"label_map": {"0":"bg","1":"11",...}}
    """
    p = data_dir / "label_map.json"
    if not p.exists():
        return None
    try:
        lm = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

    if isinstance(lm, dict) and "internal_to_fdi" in lm and isinstance(lm["internal_to_fdi"], dict):
        for k, v in lm["internal_to_fdi"].items():
            try:
                if int(v) == int(target_fdi):
                    return int(k)
            except Exception:
                continue

    for key in ("idx2id", "idx2fdi"):
        if isinstance(lm, dict) and key in lm and isinstance(lm[key], dict):
            for k, v in lm[key].items():
                try:
                    if int(str(v)) == int(target_fdi):
                        return int(k)
                except Exception:
                    continue

    if isinstance(lm, dict) and "label_map" in lm and isinstance(lm["label_map"], dict):
        for k, v in lm["label_map"].items():
            try:
                if int(str(v)) == int(target_fdi):
                    return int(k)
            except Exception:
                continue

    return None


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# PARTE 4/5:
#   - run_epoch (idéntico: update en train(), métricas en eval() opcional)
#   - main(): argparse + sanity + loaders + model=PointNet2Seg + logging/csv/history
#   - train loop + prints + best/last checkpoints
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# ============================================================
# TRAIN / EVAL (v4: update en train(), métricas en eval() para TRAIN)
# ============================================================
@torch.no_grad()
def _compute_metrics_from_logits(
    logits: torch.Tensor,
    y: torch.Tensor,
    C: int,
    d21_idx: int,
    bg: int
) -> Dict[str, float]:
    pred = logits.argmax(dim=-1)

    acc_all = _acc_all(pred, y)

    mask = (y != int(bg))
    if mask.any():
        acc_no_bg = float((pred[mask] == y[mask]).float().mean().item())
    else:
        acc_no_bg = 0.0

    f1m, ioum = macro_metrics_no_bg(pred, y, C=C, bg=bg)

    d21_acc, d21_f1, d21_iou = d21_metrics_binary(
        pred, y, d21_idx=d21_idx, bg=bg, include_bg=False
    )
    d21_bin_acc_all, _, _ = d21_metrics_binary(
        pred, y, d21_idx=d21_idx, bg=bg, include_bg=True
    )

    pred_bg_frac = float((pred.reshape(-1) == int(bg)).float().mean().item())

    return {
        "acc_all": float(acc_all),
        "acc_no_bg": float(acc_no_bg),
        "f1_macro": float(f1m),
        "iou_macro": float(ioum),
        "d21_acc": float(d21_acc),
        "d21_f1": float(d21_f1),
        "d21_iou": float(d21_iou),
        "d21_bin_acc_all": float(d21_bin_acc_all),
        "pred_bg_frac": float(pred_bg_frac),
    }


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
    collect_batch_stats: bool = False
) -> Dict[str, float]:
    """
    v4:
      - Si train=True: hace update con model.train() (dropout ON),
        PERO calcula métricas con segundo forward en model.eval() (dropout OFF),
        para que train vs val sea comparable.
      - Si train=False: eval normal (model.eval()).
    """
    scaler = run_epoch.scaler  # type: ignore
    if use_amp and (scaler is None):
        scaler = torch.amp.GradScaler("cuda")
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

    batch_stats: Dict[str, List[float]] = {
        "loss": [],
        "acc_all": [],
        "acc_no_bg": [],
        "f1_macro": [],
        "iou_macro": [],
        "d21_acc": [],
        "d21_f1": [],
        "d21_iou": [],
        "d21_bin_acc_all": [],
        "pred_bg_frac": [],
    }

    if not bool(train):
        model.eval()

    for xyz, y in loader:
        xyz = xyz.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if train:
            assert optimizer is not None
            model.train(True)
            optimizer.zero_grad(set_to_none=True)

            if use_amp and device.type == "cuda":
                ctx = torch.amp.autocast("cuda", enabled=True)
            else:
                ctx = torch.amp.autocast("cpu", enabled=False)

            # ---- forward (train) para UPDATE ----
            with ctx:
                logits_train = model(xyz)
                loss = loss_fn(logits_train.reshape(-1, C), y.reshape(-1))

            # ---- backward/step ----
            if use_amp and device.type == "cuda":
                assert scaler is not None
                scaler.scale(loss).backward()

                if grad_clip is not None and float(grad_clip) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))

                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip is not None and float(grad_clip) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                optimizer.step()

            # ---- métricas: segundo forward en eval() (dropout OFF) ----
            model.eval()
            with torch.no_grad():
                logits_eval = model(xyz)
            metrics = _compute_metrics_from_logits(logits_eval, y, C=C, d21_idx=d21_idx, bg=bg)

        else:
            # ---- eval normal (sin update) ----
            if use_amp and device.type == "cuda":
                ctx = torch.amp.autocast("cuda", enabled=True)
            else:
                ctx = torch.amp.autocast("cpu", enabled=False)

            with torch.no_grad():
                with ctx:
                    logits = model(xyz)
                    loss = loss_fn(logits.reshape(-1, C), y.reshape(-1))
                metrics = _compute_metrics_from_logits(logits, y, C=C, d21_idx=d21_idx, bg=bg)

        loss_sum += float(loss.item())
        acc_all_sum += float(metrics["acc_all"])
        acc_no_bg_sum += float(metrics["acc_no_bg"])
        f1m_sum += float(metrics["f1_macro"])
        ioum_sum += float(metrics["iou_macro"])
        d21_acc_sum += float(metrics["d21_acc"])
        d21_f1_sum += float(metrics["d21_f1"])
        d21_iou_sum += float(metrics["d21_iou"])
        d21_bin_all_sum += float(metrics["d21_bin_acc_all"])
        pred_bg_frac_sum += float(metrics["pred_bg_frac"])
        n_batches += 1

        if collect_batch_stats:
            batch_stats["loss"].append(float(loss.item()))
            batch_stats["acc_all"].append(float(metrics["acc_all"]))
            batch_stats["acc_no_bg"].append(float(metrics["acc_no_bg"]))
            batch_stats["f1_macro"].append(float(metrics["f1_macro"]))
            batch_stats["iou_macro"].append(float(metrics["iou_macro"]))
            batch_stats["d21_acc"].append(float(metrics["d21_acc"]))
            batch_stats["d21_f1"].append(float(metrics["d21_f1"]))
            batch_stats["d21_iou"].append(float(metrics["d21_iou"]))
            batch_stats["d21_bin_acc_all"].append(float(metrics["d21_bin_acc_all"]))
            batch_stats["pred_bg_frac"].append(float(metrics["pred_bg_frac"]))

    n = max(1, n_batches)

    out = {
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

    if collect_batch_stats:
        out_std = {}
        for k, v in batch_stats.items():
            _, std = _mean_std(v)
            out_std[f"{k}_std"] = std
        out.update(out_std)

    return out


run_epoch.scaler = None  # type: ignore


# ============================================================
# MAIN (PointNet++ clásico)
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.5)

    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    ap.add_argument("--d21_internal", type=int, required=True)

    ap.add_argument("--bg_index", type=int, default=0)
    ap.add_argument("--bg_weight", type=float, default=0.10)

    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--use_amp", action="store_true")

    # normalización: se controla con flag inverso (igual que v3)
    ap.add_argument("--no_normalize", action="store_true")

    # trazabilidad: opcional forzar index CSV
    ap.add_argument("--index_csv", type=str, default=None)

    ap.add_argument("--do_infer", action="store_true")
    ap.add_argument("--infer_examples", type=int, default=12)
    ap.add_argument("--infer_split", type=str, default="test", choices=["test", "val", "train"])

    # (NEW v4) Métricas train comparables: forward de métricas en eval()
    ap.add_argument("--train_metrics_eval", action="store_true",
                    help="Si está activo, calcula métricas de TRAIN con model.eval() (sin dropout/BN train), "
                         "pero mantiene el forward train para backprop.")

    # (NEW v5) Vecinos del d21 (métricas binarias por diente, estilo DGCNN)
    ap.add_argument("--neighbor_teeth", type=str, default="",
                    help='Lista "name:idx" separada por coma. Ej: "d11:1,d22:9" (idx = clase interna).')
    ap.add_argument("--neighbor_eval_split", type=str, default="val", choices=["val", "test", "both", "none"],
                    help="Dónde calcular métricas de vecinos durante entrenamiento. default=val")
    ap.add_argument("--neighbor_every", type=int, default=1,
                    help="Cada cuántos epochs calcular vecinos (1=siempre). Si es >1, reduce costo.")

    # ---------------- PointNet++ hyperparams (exponibles) ----------------
    ap.add_argument("--sa_npoints", type=str, default="1024,256,64",
                    help="npoints por SA (3 niveles), ej: 1024,256,64")
    ap.add_argument("--sa_radii", type=str, default="0.10,0.20,0.40",
                    help="radii por SA, ej: 0.10,0.20,0.40")
    ap.add_argument("--sa_nsamples", type=str, default="32,32,32",
                    help="nsamples (k) por SA, ej: 32,32,32")
    ap.add_argument("--sa_mlps", type=str, default="64-64-128|128-128-256|256-256-512|512-1024",
                    help="MLPs SA por nivel separados por |. Ej: 64-64-128|128-128-256|256-256-512|512-1024")
    ap.add_argument("--fp_mlps", type=str, default="256-256|256-256|256-128|128-128-128",
                    help="MLPs FP por nivel separados por |. Ej: 256-256|256-256|256-128|128-128-128")

    args = ap.parse_args()
    set_seed(args.seed)

    # ---------------- device ----------------
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================
    # SANITY DATA
    # =========================================================
    Ytr = np.asarray(np.load(data_dir / "Y_train.npz")["Y"]).reshape(-1)
    Yva = np.asarray(np.load(data_dir / "Y_val.npz")["Y"]).reshape(-1)
    Yte = np.asarray(np.load(data_dir / "Y_test.npz")["Y"]).reshape(-1)

    bg = int(args.bg_index)

    bg_tr = float((Ytr == bg).mean())
    bg_va = float((Yva == bg).mean())
    bg_te = float((Yte == bg).mean())

    C = int(max(int(Ytr.max()), int(Yva.max()), int(Yte.max()))) + 1

    print(f"[SANITY] num_classes C = {C}")
    print(f"[SANITY] bg_frac train/val/test = {bg_tr:.4f} {bg_va:.4f} {bg_te:.4f}")
    print(f"[SANITY] baseline acc_all (always-bg) train/val/test = {bg_tr:.4f} {bg_va:.4f} {bg_te:.4f}")

    if not (0 <= int(args.d21_internal) < C):
        raise ValueError(f"d21_internal fuera de rango: {args.d21_internal} (C={C})")

    # =========================================================
    # LOADERS
    # =========================================================
    dl_tr, dl_va, dl_te, ds_te = make_loaders(
        data_dir=data_dir,
        bs=args.batch_size,
        nw=args.num_workers,
        normalize=(not args.no_normalize),
    )

    # =========================================================
    # PARSE BACKBONE HPARAMS
    # =========================================================
    def _parse_ints_csv(s: str, n: int) -> Tuple[int, ...]:
        parts = [p.strip() for p in str(s).split(",") if p.strip()]
        if len(parts) != int(n):
            raise ValueError(f"esperaba {n} ints en '{s}'")
        return tuple(int(p) for p in parts)

    def _parse_floats_csv(s: str, n: int) -> Tuple[float, ...]:
        parts = [p.strip() for p in str(s).split(",") if p.strip()]
        if len(parts) != int(n):
            raise ValueError(f"esperaba {n} floats en '{s}'")
        return tuple(float(p) for p in parts)

    def _parse_mlps_pipe(s: str, expected_blocks: int) -> Tuple[Tuple[int, ...], ...]:
        blocks = [b.strip() for b in str(s).split("|") if b.strip()]
        if len(blocks) != int(expected_blocks):
            raise ValueError(f"esperaba {expected_blocks} bloques en '{s}' (separados por '|')")
        out = []
        for b in blocks:
            nums = [x.strip() for x in b.split("-") if x.strip()]
            if len(nums) == 0:
                raise ValueError(f"bloque vacío en '{s}'")
            out.append(tuple(int(x) for x in nums))
        return tuple(out)

    sa_npoints = _parse_ints_csv(args.sa_npoints, 3)
    sa_radii = _parse_floats_csv(args.sa_radii, 3)
    sa_nsamples = _parse_ints_csv(args.sa_nsamples, 3)
    sa_mlps = _parse_mlps_pipe(args.sa_mlps, 4)
    fp_mlps = _parse_mlps_pipe(args.fp_mlps, 4)

    # =========================================================
    # MODEL / OPT / LOSS
    # =========================================================
    model = PointNet2Seg(
        num_classes=C,
        dropout=float(args.dropout),
        npoints=sa_npoints,
        radii=sa_radii,
        nsamples=sa_nsamples,
        sa_mlps=sa_mlps,
        fp_mlps=fp_mlps,
    ).to(device)

    w = torch.ones(C, device=device, dtype=torch.float32)
    w[bg] = float(args.bg_weight)
    loss_fn = nn.CrossEntropyLoss(weight=w)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=int(args.epochs),
        eta_min=1e-6,
    )

    d21_int = int(args.d21_internal)
    print(f"[INFO] d21_internal={d21_int}")

    # =========================================================
    # NEIGHBORS (v5)
    # =========================================================
    neighbor_list = parse_neighbor_teeth(args.neighbor_teeth)
    if neighbor_list:
        bad = [(n, i) for (n, i) in neighbor_list if not (0 <= int(i) < C)]
        if bad:
            raise ValueError(f"neighbor_teeth fuera de rango (C={C}): {bad}")
        print(f"[NEIGHBORS] parsed: {neighbor_list}")
    else:
        print(f"[NEIGHBORS] none")

    neighbor_cols = []
    for name, _ in neighbor_list:
        neighbor_cols += [f"{name}_acc", f"{name}_f1", f"{name}_iou", f"{name}_bin_acc_all"]

    # =========================================================
    # LOGGING
    # =========================================================
    run_meta = {
        "script_name": "pointnetpp_classic_final_v1.py",
        "start_time": datetime.now().isoformat(timespec="seconds"),
        "data_dir": str(data_dir),
        "out_dir": str(out_dir),
        "device": str(device),
        "seed": int(args.seed),
        "num_classes": int(C),
        "bg_index": int(bg),
        "bg_weight": float(args.bg_weight),
        "d21_internal": int(d21_int),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "dropout": float(args.dropout),
        "grad_clip": float(args.grad_clip),
        "use_amp": bool(args.use_amp),
        "normalize_unit_sphere": bool(not args.no_normalize),
        "train_metrics_eval": bool(args.train_metrics_eval),
        "infer_examples": int(args.infer_examples),
        "do_infer": bool(args.do_infer),
        "infer_split": str(args.infer_split),
        "index_csv": str(args.index_csv) if args.index_csv else "",
        # PointNet++ hyperparams
        "sa_npoints": list(sa_npoints),
        "sa_radii": list(sa_radii),
        "sa_nsamples": list(sa_nsamples),
        "sa_mlps": [list(b) for b in sa_mlps],
        "fp_mlps": [list(b) for b in fp_mlps],
        # neighbors
        "neighbor_teeth": str(args.neighbor_teeth),
        "neighbor_eval_split": str(args.neighbor_eval_split),
        "neighbor_every": int(args.neighbor_every),
        "neighbor_parsed": neighbor_list,
    }
    save_json(run_meta, out_dir / "run_meta.json")

    # =========================================================
    # CSV por epoch (base + neighbors al final)
    # =========================================================
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
            "sec",
        ] + neighbor_cols)

    # =========================================================
    # HISTORY (plots)
    # =========================================================
    history: Dict[str, List[float]] = {}
    def _mk(k: str):
        history[k] = []

    for k in (
        "loss", "acc_all", "acc_no_bg",
        "f1_macro", "iou_macro",
        "d21_acc", "d21_f1", "d21_iou",
        "d21_bin_acc_all",
        "pred_bg_frac",
    ):
        _mk(f"train_{k}")
        _mk(f"val_{k}")

    for name, _ in neighbor_list:
        for met in ("acc", "f1", "iou", "bin_acc_all"):
            _mk(f"val_{name}_{met}")

    best_val_f1 = -1.0
    best_epoch = -1
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"

    t0 = time.time()

    # =========================================================
    # TRAIN LOOP
    # =========================================================
    for epoch in range(1, int(args.epochs) + 1):
        e0 = time.time()

        # 1) Paso de entrenamiento (backprop)
        tr_backprop = run_epoch(
            model=model,
            loader=dl_tr,
            optimizer=opt,
            loss_fn=loss_fn,
            C=C,
            d21_idx=d21_int,
            device=device,
            bg=bg,
            train=True,
            use_amp=bool(args.use_amp),
            grad_clip=float(args.grad_clip),
        )

        # 2) Métricas train comparables (opcional)
        if bool(args.train_metrics_eval):
            tr = run_epoch(
                model=model,
                loader=dl_tr,
                optimizer=None,
                loss_fn=loss_fn,
                C=C,
                d21_idx=d21_int,
                device=device,
                bg=bg,
                train=False,
                use_amp=False,
                grad_clip=None,
            )
        else:
            tr = tr_backprop

        # 3) Validación
        va = run_epoch(
            model=model,
            loader=dl_va,
            optimizer=None,
            loss_fn=loss_fn,
            C=C,
            d21_idx=d21_int,
            device=device,
            bg=bg,
            train=False,
            use_amp=False,
            grad_clip=None,
        )

        # 4) Neighbors (val por epoch según neighbor_every)
        do_neighbors = (len(neighbor_list) > 0) and (str(args.neighbor_eval_split).lower() != "none")
        do_neighbors = do_neighbors and (int(args.neighbor_every) > 0) and ((epoch % int(args.neighbor_every)) == 0)

        nb_vals: Dict[str, float] = {}
        if do_neighbors:
            split_mode = str(args.neighbor_eval_split).lower()
            if split_mode in ("val", "both"):
                nb_vals = eval_neighbors_on_loader(
                    model=model,
                    loader=dl_va,
                    device=device,
                    neighbor_list=neighbor_list,
                    bg=bg
                )

        sched.step()
        lr_now = float(opt.param_groups[0]["lr"])
        sec = float(time.time() - e0)

        # history
        for k in (
            "loss", "acc_all", "acc_no_bg",
            "f1_macro", "iou_macro",
            "d21_acc", "d21_f1", "d21_iou",
            "d21_bin_acc_all",
            "pred_bg_frac",
        ):
            history[f"train_{k}"].append(float(tr[k]))
            history[f"val_{k}"].append(float(va[k]))

        if neighbor_list:
            for name, _ in neighbor_list:
                for met in ("acc", "f1", "iou", "bin_acc_all"):
                    key_csv = f"{name}_{met}" if met != "bin_acc_all" else f"{name}_bin_acc_all"
                    hk = f"val_{name}_{met}"
                    history[hk].append(float(nb_vals.get(key_csv, 0.0)))

        # CSV
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            wcsv = csv.writer(f)

            row_train = [
                epoch, "train",
                tr["loss"], tr["acc_all"], tr["acc_no_bg"],
                tr["f1_macro"], tr["iou_macro"],
                tr["d21_acc"], tr["d21_f1"], tr["d21_iou"],
                tr["d21_bin_acc_all"],
                tr["pred_bg_frac"],
                lr_now,
                sec,
            ]

            row_val = [
                epoch, "val",
                va["loss"], va["acc_all"], va["acc_no_bg"],
                va["f1_macro"], va["iou_macro"],
                va["d21_acc"], va["d21_f1"], va["d21_iou"],
                va["d21_bin_acc_all"],
                va["pred_bg_frac"],
                lr_now,
                sec,
            ]

            if neighbor_cols:
                row_train += ["" for _ in neighbor_cols]
                row_val += [float(nb_vals.get(col, 0.0)) for col in neighbor_cols]

            wcsv.writerow(row_train)
            wcsv.writerow(row_val)

        # checkpoints
        torch.save({"model": model.state_dict(), "epoch": epoch}, last_path)

        if float(va["f1_macro"]) > best_val_f1:
            best_val_f1 = float(va["f1_macro"])
            best_epoch = int(epoch)
            torch.save({"model": model.state_dict(), "epoch": epoch}, best_path)

        # warning colapso a bg
        if float(va["pred_bg_frac"]) > max(0.95, bg_va + 0.12):
            print(
                f"[WARN] posible colapso a BG: "
                f"val pred_bg_frac={va['pred_bg_frac']:.3f} (bg_gt≈{bg_va:.3f})"
            )

        nb_str = ""
        if neighbor_list and do_neighbors:
            parts = []
            for name, _ in neighbor_list:
                parts.append(f"{name}_f1={nb_vals.get(f'{name}_f1', 0.0):.3f}")
            nb_str = " | nb(" + ",".join(parts) + ")"

        print(
            f"[{epoch:03d}/{int(args.epochs)}] "
            f"train loss={tr['loss']:.4f} f1m={tr['f1_macro']:.3f} ioum={tr['iou_macro']:.3f} "
            f"acc_all={tr['acc_all']:.3f} acc_no_bg={tr['acc_no_bg']:.3f} | "
            f"val loss={va['loss']:.4f} f1m={va['f1_macro']:.3f} ioum={va['iou_macro']:.3f} "
            f"acc_all={va['acc_all']:.3f} acc_no_bg={va['acc_no_bg']:.3f} | "
            f"d21(cls) acc={va['d21_acc']:.3f} f1={va['d21_f1']:.3f} iou={va['d21_iou']:.3f} | "
            f"d21(bin all) acc={va['d21_bin_acc_all']:.3f} | "
            f"pred_bg_frac(train)={tr['pred_bg_frac']:.3f} pred_bg_frac(val)={va['pred_bg_frac']:.3f} "
            f"lr={lr_now:.2e} sec={sec:.1f}"
            f"{nb_str}"
        )

        if neighbor_list and do_neighbors:
            parts_full = []
            for name, _ in neighbor_list:
                parts_full.append(
                    f"{name}_acc={nb_vals.get(f'{name}_acc', 0.0):.3f} "
                    f"{name}_f1={nb_vals.get(f'{name}_f1', 0.0):.3f} "
                    f"{name}_iou={nb_vals.get(f'{name}_iou', 0.0):.3f} "
                    f"{name}_bin_acc_all={nb_vals.get(f'{name}_bin_acc_all', 0.0):.3f}"
                )
            print("[neighbors val] " + " | ".join(parts_full))

    save_json(history, out_dir / "history.json")

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # PARTE 5/5:
    #   - Cargar best, test eval (print completo), neighbors test
    #   - guardar test_metrics.json
    #   - plots (Train vs Val) + plots neighbors
    #   - inferencia (3 PNGs por sample) + manifest
    #   - footer __main__
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # =========================================================
    # TEST (best checkpoint) + PLOTS (SIN testline) + INFERENCIA
    # =========================================================

    # =========================================================
    # CARGAR BEST
    # =========================================================
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"[test] Cargado best.pt (epoch={ckpt.get('epoch','?')})")
    else:
        print("[test] best.pt no encontrado, usando modelo final (last.pt si existe)")
        if last_path.exists():
            ckpt = torch.load(last_path, map_location=device)
            model.load_state_dict(ckpt["model"])

    # =========================================================
    # TEST EVAL
    # =========================================================
    test_metrics = run_epoch(
        model=model,
        loader=dl_te,
        optimizer=None,
        loss_fn=loss_fn,
        C=C,
        d21_idx=d21_int,
        device=device,
        bg=bg,
        train=False,
        use_amp=False,
        grad_clip=None,
    )

    print(
        f"[test] "
        f"loss={test_metrics['loss']:.4f} "
        f"f1m={test_metrics['f1_macro']:.3f} "
        f"ioum={test_metrics['iou_macro']:.3f} "
        f"acc_all={test_metrics['acc_all']:.3f} "
        f"acc_no_bg={test_metrics['acc_no_bg']:.3f} | "
        f"d21(cls) acc={test_metrics['d21_acc']:.3f} "
        f"f1={test_metrics['d21_f1']:.3f} "
        f"iou={test_metrics['d21_iou']:.3f} | "
        f"d21(bin all) acc={test_metrics['d21_bin_acc_all']:.3f} | "
        f"pred_bg_frac(test)={test_metrics['pred_bg_frac']:.3f}"
    )

    # =========================================================
    # v5 NEIGHBORS EN TEST
    # =========================================================
    test_neighbors = {}
    if neighbor_list:
        split_mode = str(args.neighbor_eval_split).lower()
        if split_mode in ("test", "both"):
            test_neighbors = eval_neighbors_on_loader(
                model=model,
                loader=dl_te,
                device=device,
                neighbor_list=neighbor_list,
                bg=bg
            )

            parts = []
            for name, _ in neighbor_list:
                parts.append(
                    f"{name}_acc={test_neighbors.get(f'{name}_acc', 0.0):.3f} "
                    f"{name}_f1={test_neighbors.get(f'{name}_f1', 0.0):.3f} "
                    f"{name}_iou={test_neighbors.get(f'{name}_iou', 0.0):.3f} "
                    f"{name}_bin_acc_all={test_neighbors.get(f'{name}_bin_acc_all', 0.0):.3f}"
                )
            print("[test neighbors] " + " | ".join(parts))

            print("[test neighbors]")
            for k, v in test_neighbors.items():
                print(f"  {k} = {v:.4f}")

    # =========================================================
    # GUARDAR TEST_METRICS.JSON
    # =========================================================
    out_test = dict(test_metrics)
    out_test["best_epoch"] = int(best_epoch)
    out_test["neighbor_metrics_test"] = test_neighbors
    save_json(out_test, out_dir / "test_metrics.json")

    # =========================================================
    # PLOTS (solo Train vs Val)
    # =========================================================
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    for k in (
        "loss", "acc_all", "acc_no_bg",
        "f1_macro", "iou_macro",
        "d21_acc", "d21_f1", "d21_iou",
        "d21_bin_acc_all",
        "pred_bg_frac",
    ):
        plot_train_val(
            name=k,
            y_tr=history[f"train_{k}"],
            y_va=history[f"val_{k}"],
            out_png=plot_dir / f"{k}.png",
            best_epoch=best_epoch,
        )

    for name, _ in neighbor_list:
        for met in ("acc", "f1", "iou", "bin_acc_all"):
            key = f"val_{name}_{met}"
            if key in history:
                plot_train_val(
                    name=f"{name}_{met}",
                    y_tr=[0.0] * len(history[key]),
                    y_va=history[key],
                    out_png=plot_dir / f"{name}_{met}.png",
                    best_epoch=best_epoch,
                )

    # =========================================================
    # INFERENCIA (visual)
    # =========================================================
    if bool(args.do_infer):
        infer_split = str(args.infer_split).lower()
        if infer_split == "test":
            loader = dl_te
        elif infer_split == "val":
            loader = dl_va
        else:
            loader = dl_tr

        model.eval()
        infer_dir = out_dir / "inference"
        infer_dir.mkdir(exist_ok=True)

        index_map = None
        if args.index_csv:
            index_map = _read_index_csv(Path(args.index_csv))
        else:
            auto_index = _discover_index_csv(data_dir, infer_split)
            if auto_index:
                index_map = _read_index_csv(auto_index)

        manifest_rows = []

        done = 0
        with torch.no_grad():
            for batch_idx, (xyz, y) in enumerate(loader):
                xyz = xyz.to(device)
                logits = model(xyz)
                pred = logits.argmax(dim=-1)

                for b in range(xyz.shape[0]):
                    if done >= int(args.infer_examples):
                        break

                    xyz_np = xyz[b].detach().cpu().numpy()
                    y_np = y[b].detach().cpu().numpy()
                    pr_np = pred[b].detach().cpu().numpy()

                    tag = f"{infer_split}_sample_{done}"
                    info = {}
                    if index_map and done in index_map:
                        info = index_map[done]
                        safe = _sanitize_tag(info.get("sample_name", ""))
                        if safe:
                            tag += "_" + safe

                    plot_pointcloud_all_classes(
                        xyz_np, y_np, pr_np,
                        infer_dir / f"{tag}_all.png",
                        C=C,
                        title=tag,
                    )

                    plot_errors(
                        xyz_np, y_np, pr_np,
                        infer_dir / f"{tag}_errors.png",
                        bg=bg,
                        title=tag,
                    )

                    plot_d21_focus(
                        xyz_np, y_np, pr_np,
                        infer_dir / f"{tag}_d21.png",
                        d21_idx=d21_int,
                        bg=bg,
                        title=tag,
                    )

                    manifest_rows.append({
                        "row_i": done,
                        "tag": tag,
                        "split": infer_split,
                        "sample_name": info.get("sample_name", ""),
                        "jaw": info.get("jaw", ""),
                        "path": info.get("path", ""),
                    })

                    done += 1

                if done >= int(args.infer_examples):
                    break

        if manifest_rows:
            manifest_path = infer_dir / "inference_manifest.csv"
            with open(manifest_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["row_i", "tag", "split", "sample_name", "jaw", "path"]
                )
                writer.writeheader()
                for row in manifest_rows:
                    writer.writerow(row)

    total_sec = time.time() - t0
    print(
        f"[done] Entrenamiento terminado en {_fmt_hms(total_sec)}. "
        f"best_epoch={best_epoch} best_val={best_val_f1:.4f}"
    )


if __name__ == "__main__":
    main()
