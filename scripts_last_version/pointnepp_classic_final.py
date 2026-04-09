#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pointnepp_classic_final.py

PointNet++ (SSG) – Segmentación multiclase dental 3D (OPCIÓN A – PAPER CORRECTA)
(MISMA FILOSOFÍA/OUTPUTS/TRAZABILIDAD que pointnet_classic_final_v5.py, pero con backbone PointNet++)

✅ BG incluido en la loss (NO ignore en la loss)
✅ BG excluido SOLO en métricas macro (f1/iou/prec/rec/acc_no_bg)
✅ Métricas diente 21 explícitas (acc/f1/iou) de forma BINARIA correcta
✅ Métrica "d21_bin_acc_all" (incluye TODO, incluso bg) para referencia
✅ Estabilidad: bg downweight, weight_decay, grad clipping, CosineAnnealingLR
✅ RTX 3090 friendly: AMP, pin_memory, persistent_workers, non_blocking, cudnn.benchmark
✅ PLOTS: Train vs Val (sin línea horizontal de TEST), best_epoch vertical
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

FIXES IMPORTANTES PointNet++ (geom robusta):
✅ Geometría SIEMPRE en FP32 (square_distance, FPS, ball query, interp) incluso con AMP
✅ BallQuery robusto: nunca deja idx=N (invalid), clampa índices, evita NaNs silenciosos

(NEW v5 equivalente):
✅ Soporte REAL de “neighbor teeth metrics” (igual filosofía que DGCNN/PointNet v5):
   - Flag --neighbor_teeth "d11:1,d22:9,..." (lista arbitraria nombre:idx)
   - Loggea métricas binarias (acc/f1/iou + bin_acc_all) para cada vecino
   - Integra en CSV por epoch, history.json, test_metrics.json
   - Se imprime en consola (línea extra) cuando corresponda

Dataset esperado:
  data_dir/X_train.npz, Y_train.npz, X_val.npz, Y_val.npz, X_test.npz, Y_test.npz
  X: [B,N,3], Y: [B,N] con clases internas 0..C-1 (0=bg)

Ejemplo (GPU0):
CUDA_VISIBLE_DEVICES=0 python3 pointnepp_classic_final.py \
  --data_dir .../upper_only_surf_global_excl_wisdom_seed42_aug2 \
  --out_dir  .../outputs/pointnetpp/gpu0_run1 \
  --epochs 120 --batch_size 16 --lr 2e-4 --weight_decay 1e-4 --dropout 0.5 \
  --num_workers 6 --device cuda --d21_internal 8 \
  --bg_weight 0.03 --grad_clip 1.0 --use_amp \
  --neighbor_teeth "d11:1,d22:9" \
  --neighbor_eval_split val --neighbor_every 1 \
  --do_infer --infer_examples 12 --infer_split test

Si el index auto-discovery no calza, fuerza:
  --index_csv /ruta/a/index_test.csv
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
    """
    Normaliza una nube [N,3] a esfera unitaria:
      - centra en el centroide
      - escala por el radio máximo
    """
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


# ============================================================
# POINTNET++ UTILIDADES (ROBUSTAS + FP32 EN GEOMETRÍA)
# ============================================================
def _as_fp32(x: torch.Tensor) -> torch.Tensor:
    return x.float() if x.dtype != torch.float32 else x


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    Distancia cuadrada robusta en FP32.
    src: [B, N, 3], dst: [B, M, 3] -> [B, N, M]
    """
    src = _as_fp32(src)
    dst = _as_fp32(dst)
    dist = -2.0 * torch.matmul(src, dst.transpose(2, 1))  # [B,N,M]
    dist += torch.sum(src ** 2, dim=-1, keepdim=True)     # [B,N,1]
    dist += torch.sum(dst ** 2, dim=-1, keepdim=True).transpose(2, 1)  # [B,1,M]
    dist = torch.clamp(dist, min=0.0)
    return dist


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    points: [B, N, C]
    idx: [B, S] o [B, S, K]
    """
    B = points.shape[0]
    device = points.device
    idx = torch.clamp(idx, 0, points.shape[1] - 1)

    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, device=device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    FPS robusto en FP32.
    xyz: [B, N, 3] -> idx: [B, npoint]
    """
    xyz = _as_fp32(xyz)
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, int(npoint), dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device, dtype=torch.float32)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(int(npoint)):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=1)[1]
    return centroids


def query_ball_point(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """
    Ball query ROBUSTO:
      - nunca deja idx=N (inválido)
      - si no hay vecinos, fuerza al primer idx válido (o 0)
    xyz: [B, N, 3]
    new_xyz: [B, S, 3]
    -> group_idx: [B, S, nsample]
    """
    xyz = _as_fp32(xyz)
    new_xyz = _as_fp32(new_xyz)

    device = xyz.device
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape

    sqrdists = square_distance(new_xyz, xyz)  # [B,S,N]
    group_idx = torch.arange(N, device=device).view(1, 1, N).repeat(B, S, 1)

    invalid = sqrdists > (float(radius) * float(radius))
    group_idx[invalid] = N  # marcador inválido

    group_idx = group_idx.sort(dim=-1)[0][:, :, : int(nsample)]  # [B,S,nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat(1, 1, int(nsample))

    # si quedó N (sin vecinos), fuerza a 0
    group_first = torch.where(group_first == N, torch.zeros_like(group_first), group_first)

    mask = group_idx == N
    group_idx[mask] = group_first[mask]

    group_idx = torch.clamp(group_idx, 0, N - 1)
    return group_idx


def sample_and_group(
    npoint: int,
    radius: float,
    nsample: int,
    xyz: torch.Tensor,
    points: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    FPS + BallQuery + agrupación.
    Retorna:
      new_xyz: [B, npoint, 3]
      new_points: [B, npoint, nsample, 3 + D]  (si points != None)
                 [B, npoint, nsample, 3]      (si points == None)
    """
    xyz = _as_fp32(xyz)
    if points is not None:
        points = _as_fp32(points)

    fps_idx = farthest_point_sample(xyz, int(npoint))           # [B,npoint]
    new_xyz = index_points(xyz, fps_idx)                        # [B,npoint,3]
    idx = query_ball_point(float(radius), int(nsample), xyz, new_xyz)  # [B,npoint,nsample]
    grouped_xyz = index_points(xyz, idx)                        # [B,npoint,nsample,3]
    grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)       # relativo

    if points is not None:
        grouped_points = index_points(points, idx)              # [B,npoint,nsample,D]
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm

    return new_xyz, new_points


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# PARTE 1/5 TERMINA AQUÍ.
# En la PARTE 2/5 voy con:
#   - Bloques PointNet++ (Set Abstraction + Feature Propagation) con FP32 geom
#   - Backbone PointNet2Seg (logits [B,N,C]) estilo paper
# Manteniendo comentarios y estructura “paper-like”.
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# ============================================================
# POINTNET++ BLOQUES (FORZAR FP32 INTERNAMENTE EN GEOMETRÍA)
# ============================================================
class PointNetSetAbstraction(nn.Module):
    """
    Set Abstraction (SSG):
      - FPS para elegir S=npoint centros
      - BallQuery para agrupar K=nsample vecinos dentro de radius
      - MLP (Conv2d 1x1) sobre (xyz_rel + features) y max-pool sobre vecinos

    Importante:
      - Toda la geometría (FPS, ball query, square_distance, index) va en FP32
        incluso si afuera hay AMP, para evitar NaNs silenciosos / inestabilidad.
      - El MLP sí puede beneficiarse de AMP (si está activado afuera).
    """
    def __init__(self, npoint: int, radius: float, nsample: int, in_channel: int, mlp: List[int]):
        super().__init__()
        self.npoint = int(npoint)
        self.radius = float(radius)
        self.nsample = int(nsample)

        layers = []
        last_ch = int(in_channel)
        for out_ch in mlp:
            layers.append(nn.Conv2d(last_ch, int(out_ch), 1))
            layers.append(nn.BatchNorm2d(int(out_ch)))
            layers.append(nn.ReLU(inplace=True))
            last_ch = int(out_ch)
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz: torch.Tensor, points: Optional[torch.Tensor]):
        """
        xyz:    [B, N, 3]
        points: [B, N, D] o None

        return:
          new_xyz:    [B, S, 3]
          new_points: [B, S, mlp[-1]]
        """
        # 🔒 Geometría SIEMPRE en FP32 aunque haya AMP afuera.
        with torch.cuda.amp.autocast(enabled=False):
            xyz_f = _as_fp32(xyz)
            pts_f = _as_fp32(points) if points is not None else None

            # sample_and_group retorna:
            # new_xyz: [B,S,3]
            # new_points: [B,S,K, 3 (+D)]
            new_xyz, new_points = sample_and_group(
                npoint=self.npoint,
                radius=self.radius,
                nsample=self.nsample,
                xyz=xyz_f,
                points=pts_f
            )

            # Conv2d espera [B, C, H, W]. Aquí:
            # new_points: [B,S,K,Cin] -> permute -> [B,Cin,K,S]
            new_points = new_points.permute(0, 3, 2, 1).contiguous().float()

        # ✅ MLP (Conv2d) — esto puede ir con AMP (se controla afuera)
        new_points = self.mlp(new_points)            # [B, Cout, K, S]
        new_points = torch.max(new_points, dim=2)[0] # max over K -> [B, Cout, S]
        new_points = new_points.transpose(2, 1).contiguous()  # [B, S, Cout]
        return new_xyz, new_points


class PointNetFeaturePropagation(nn.Module):
    """
    Feature Propagation (FP):
      - Interpola features de nivel "sparse" (xyz2/points2) hacia el nivel "dense" (xyz1)
      - Usualmente usa 3-NN con pesos inversos a distancia
      - Concatena con points1 (skip) y pasa MLP (Conv1d)

    Importante:
      - Toda la geometría de interpolación (square_distance + index_points + pesos) va en FP32.
      - El MLP Conv1d puede usar AMP (control externo).
    """
    def __init__(self, in_channel: int, mlp: List[int]):
        super().__init__()
        layers = []
        last_ch = int(in_channel)
        for out_ch in mlp:
            layers.append(nn.Conv1d(last_ch, int(out_ch), 1))
            layers.append(nn.BatchNorm1d(int(out_ch)))
            layers.append(nn.ReLU(inplace=True))
            last_ch = int(out_ch)
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        xyz1: torch.Tensor,
        xyz2: torch.Tensor,
        points1: Optional[torch.Tensor],
        points2: torch.Tensor
    ) -> torch.Tensor:
        """
        xyz1:    [B, N, 3]  (dense)
        xyz2:    [B, S, 3]  (sparse)
        points1: [B, N, D1] or None
        points2: [B, S, D2]

        return:
          new_points: [B, N, mlp[-1]]
        """
        with torch.cuda.amp.autocast(enabled=False):
            xyz1 = _as_fp32(xyz1)
            xyz2 = _as_fp32(xyz2)
            points2 = _as_fp32(points2)
            if points1 is not None:
                points1 = _as_fp32(points1)

            B, N, _ = xyz1.shape
            _, S, _ = xyz2.shape

            # Caso extremo: S=1 (un solo punto en nivel superior)
            if S == 1:
                interpolated = points2.repeat(1, N, 1)  # [B,N,D2]
            else:
                # dists: [B,N,S]
                dists = square_distance(xyz1, xyz2)
                dists, idx = dists.sort(dim=-1)
                dists = dists[:, :, :3]  # 3-NN
                idx = idx[:, :, :3]

                # pesos inversos (robusto)
                dist_recip = 1.0 / (dists + 1e-8)
                norm = torch.sum(dist_recip, dim=2, keepdim=True)
                weight = dist_recip / norm  # [B,N,3]

                # points2 agrupados por idx: [B,N,3,D2]
                grouped_points = index_points(points2, idx)
                interpolated = torch.sum(grouped_points * weight.unsqueeze(-1), dim=2)  # [B,N,D2]

            if points1 is not None:
                new_points = torch.cat([points1, interpolated], dim=-1)  # [B,N,D1+D2]
            else:
                new_points = interpolated  # [B,N,D2]

            # Conv1d espera [B, D, N]
            new_points = new_points.transpose(2, 1).contiguous().float()

        # ✅ MLP Conv1d (puede usar AMP afuera)
        new_points = self.mlp(new_points)                     # [B, out, N]
        new_points = new_points.transpose(2, 1).contiguous()  # [B, N, out]
        return new_points


# ============================================================
# POINTNET++ SEGMENTATION BACKBONE (SSG paper-like)
# ============================================================
class PointNet2Seg(nn.Module):
    """
    Arquitectura típica de PointNet++ SSG para segmentación:
      SA1: N->1024
      SA2: 1024->256
      SA3: 256->64
      SA4: 64->16
      FP4..FP1 para volver a N
      head Conv1d -> logits

    Nota:
      - Con unit sphere (radio ~1), radios 0.1/0.2/0.4/0.8 suelen funcionar bien para N=8192.
      - Si cambias N o normalización, radios y npoints deben revisarse.
    """
    def __init__(self, num_classes: int, dropout: float = 0.5):
        super().__init__()
        C = int(num_classes)

        # SA layers: in_channel incluye xyz_rel(3) + features previas.
        # En SA1 points=None => solo xyz_rel (3).
        self.sa1 = PointNetSetAbstraction(
            npoint=1024, radius=0.10, nsample=32,
            in_channel=3, mlp=[32, 32, 64]
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=256, radius=0.20, nsample=32,
            in_channel=3 + 64, mlp=[64, 64, 128]
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=64, radius=0.40, nsample=32,
            in_channel=3 + 128, mlp=[128, 128, 256]
        )
        self.sa4 = PointNetSetAbstraction(
            npoint=16, radius=0.80, nsample=32,
            in_channel=3 + 256, mlp=[256, 256, 512]
        )

        # FP layers: concat skip (lower-level) + interpolated (upper-level)
        self.fp4 = PointNetFeaturePropagation(in_channel=512 + 256, mlp=[256, 256])
        self.fp3 = PointNetFeaturePropagation(in_channel=256 + 128, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=256 + 64,  mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128,       mlp=[128, 128, 128])

        # Head de segmentación (PointNet++ style)
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(float(dropout))
        self.conv2 = nn.Conv1d(128, C, 1)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        xyz: [B, N, 3] (coordenadas)
        return logits: [B, N, C]
        """
        # Aseguramos coords en FP32 (aunque AMP esté afuera, coords se mantienen estables).
        xyz = xyz.float()

        # Nivel 0
        l0_xyz = xyz
        l0_points = None

        # Abstracción jerárquica
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)   # [B,1024,3], [B,1024,64]
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)   # [B,256,3],  [B,256,128]
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)   # [B,64,3],   [B,64,256]
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)   # [B,16,3],   [B,16,512]

        # Propagación de features (upsampling)
        l3_points_new = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)       # [B,64,256]
        l2_points_new = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points_new)   # [B,256,256]
        l1_points_new = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points_new)   # [B,1024,128]
        l0_points_new = self.fp1(l0_xyz, l1_xyz, None, l1_points_new)        # [B,N,128]

        # Head
        x = l0_points_new.transpose(2, 1).contiguous()  # [B,128,N]
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.drop(x)
        x = self.conv2(x)                               # [B,C,N]
        logits = x.transpose(2, 1).contiguous()         # [B,N,C]
        return logits


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# PARTE 2/5 TERMINA AQUÍ.
# En la PARTE 3/5 voy con:
#   - Métricas (macro sin bg, d21 binario) + neighbor teeth (parse + eval)
#   - Visualización (all classes / errors / d21 focus) robusta
# Manteniendo estilo y comentarios como el PointNet v5.
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# ============================================================
# MÉTRICAS (macro sin bg) + d21 binario
# ============================================================
@torch.no_grad()
def macro_metrics_no_bg(pred: torch.Tensor, gt: torch.Tensor, C: int, bg: int = 0) -> Tuple[float, float]:
    """
    Macro-F1 e IoU macro calculados EXCLUYENDO BG (gt!=bg),
    promediando sobre clases 1..C-1.

    Si una clase no aparece en el GT (tp+fp+fn=0), se omite del promedio.
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
            continue  # clase ausente en GT

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
    d21 como binario:
      positivo = clase d21_idx
      negativo = resto

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
    Fuerza np.ndarray contiguo (evita problemas con memmap/subclases/torch tensor).
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
    """
    Figura 3D lado a lado:
      - GT con colores por clase
      - Pred con colores por clase
    """
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

    ax1.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                c=c_gt, s=s, linewidths=0, depthshade=False)
    ax2.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                c=c_pr, s=s, linewidths=0, depthshade=False)

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
    """
    Muestra errores en rojo.
      - correcto: gris
      - error: rojo
      - bg: gris claro/transparente
    """
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
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
               c=c, s=s, linewidths=0, depthshade=False)
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
    """
    Foco específico en d21:
      - TP: verde
      - FP/FN: rojo
      - resto: gris
    """
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
    c[tp, :] = (0.10, 0.75, 0.25, 1.0)
    c[err, :] = (0.85, 0.10, 0.10, 1.0)

    fig = plt.figure(figsize=(6, 5), dpi=200)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
               c=c, s=s, linewidths=0, depthshade=False)
    ax.set_axis_off()
    ax.view_init(elev=20, azim=45)
    ax.set_title("Foco d21: TP (verde) | FP/FN (rojo)", fontsize=10)
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# NEIGHBOR TEETH (v5 estilo DGCNN / PointNet final)
# ============================================================
@torch.no_grad()
def _tooth_metrics_binary(pred: torch.Tensor, gt: torch.Tensor, tooth_idx: int, bg: int = 0) -> Dict[str, float]:
    """
    Métricas binarias para un diente arbitrario (idx interno).
    """
    acc, f1, iou = d21_metrics_binary(
        pred, gt, d21_idx=int(tooth_idx), bg=int(bg), include_bg=False
    )
    acc_all, f1_all, iou_all = d21_metrics_binary(
        pred, gt, d21_idx=int(tooth_idx), bg=int(bg), include_bg=True
    )
    return {
        "acc": float(acc),
        "f1": float(f1),
        "iou": float(iou),
        "bin_acc_all": float(acc_all),
        "bin_f1_all": float(f1_all),
        "bin_iou_all": float(iou_all),
    }


def parse_neighbor_teeth(spec: Optional[str]) -> List[Tuple[str, int]]:
    """
    Parse de --neighbor_teeth con formato:
      "d11:1,d22:9,foo:3"

    Devuelve lista [(name, idx), ...] (orden estable).
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
        if not tok or ":" not in tok:
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


@torch.no_grad()
def eval_neighbors_on_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    neighbor_list: List[Tuple[str, int]],
    bg: int
) -> Dict[str, float]:
    """
    Evalúa métricas binarias para cada diente vecino en un loader completo.
    """
    if not neighbor_list:
        return {}

    model.eval()
    sums = {
        f"{name}_{k}": 0.0
        for name, _ in neighbor_list
        for k in ("acc", "f1", "iou", "bin_acc_all")
    }

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


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# PARTE 3/5 TERMINA AQUÍ.
# En la PARTE 4/5 voy con:
#   - _compute_metrics_from_logits
#   - run_epoch (train/eval con AMP + comparabilidad)
#   - MAIN (argumentos, sanity, modelo, optim, scheduler)
#   - Loop de entrenamiento con neighbors + logging CSV + history
# Manteniendo estructura casi 1:1 con pointnet_classic_final_v5.
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# ============================================================
# TRAIN / EVAL CORE
# ============================================================
@torch.no_grad()
def _compute_metrics_from_logits(
    logits: torch.Tensor,
    y: torch.Tensor,
    C: int,
    d21_idx: int,
    bg: int
) -> Dict[str, float]:
    """
    Calcula todas las métricas a partir de logits [B,N,C].
    """
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
) -> Dict[str, float]:
    """
    Igual filosofía que pointnet_classic_final_v5:

      - Si train=True:
          * forward + backward con model.train()
          * luego segundo forward en model.eval() para métricas comparables
      - Si train=False:
          * solo eval()

    AMP soportado (solo para forward de red, no para geometría interna).
    """

    scaler = run_epoch.scaler  # type: ignore
    if use_amp and (scaler is None) and device.type == "cuda":
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

    if not train:
        model.eval()

    for xyz, y in loader:
        xyz = xyz.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if train:
            assert optimizer is not None
            model.train(True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
                logits_train = model(xyz)
                loss = loss_fn(logits_train.reshape(-1, C), y.reshape(-1))

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

            # métricas con dropout OFF
            model.eval()
            with torch.no_grad():
                logits_eval = model(xyz)
            metrics = _compute_metrics_from_logits(
                logits_eval, y, C=C, d21_idx=d21_idx, bg=bg
            )

        else:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
                    logits = model(xyz)
                    loss = loss_fn(logits.reshape(-1, C), y.reshape(-1))
                metrics = _compute_metrics_from_logits(
                    logits, y, C=C, d21_idx=d21_idx, bg=bg
                )

        loss_sum += float(loss.item())
        acc_all_sum += metrics["acc_all"]
        acc_no_bg_sum += metrics["acc_no_bg"]
        f1m_sum += metrics["f1_macro"]
        ioum_sum += metrics["iou_macro"]
        d21_acc_sum += metrics["d21_acc"]
        d21_f1_sum += metrics["d21_f1"]
        d21_iou_sum += metrics["d21_iou"]
        d21_bin_all_sum += metrics["d21_bin_acc_all"]
        pred_bg_frac_sum += metrics["pred_bg_frac"]

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
# MAIN
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

    ap.add_argument("--no_normalize", action="store_true")

    # neighbors
    ap.add_argument("--neighbor_teeth", type=str, default="")
    ap.add_argument("--neighbor_eval_split", type=str, default="val",
                    choices=["val", "test", "both", "none"])
    ap.add_argument("--neighbor_every", type=int, default=1)

    ap.add_argument("--do_infer", action="store_true")
    ap.add_argument("--infer_examples", type=int, default=12)
    ap.add_argument("--infer_split", type=str, default="test",
                    choices=["train", "val", "test"])

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

    # ---------------- sanity ----------------
    Ytr = np.asarray(np.load(data_dir / "Y_train.npz")["Y"]).reshape(-1)
    Yva = np.asarray(np.load(data_dir / "Y_val.npz")["Y"]).reshape(-1)
    Yte = np.asarray(np.load(data_dir / "Y_test.npz")["Y"]).reshape(-1)

    bg = int(args.bg_index)
    C = int(max(int(Ytr.max()), int(Yva.max()), int(Yte.max()))) + 1

    print(f"[SANITY] num_classes C = {C}")
    print(f"[SANITY] bg_frac train={float((Ytr==bg).mean()):.4f}")

    if not (0 <= int(args.d21_internal) < C):
        raise ValueError("d21_internal fuera de rango")

    # ---------------- loaders ----------------
    dl_tr, dl_va, dl_te, ds_te = make_loaders(
        data_dir=data_dir,
        bs=args.batch_size,
        nw=args.num_workers,
        normalize=(not args.no_normalize),
    )

    # ---------------- model ----------------
    model = PointNet2Seg(num_classes=C, dropout=float(args.dropout)).to(device)

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

    neighbor_list = parse_neighbor_teeth(args.neighbor_teeth)
    if neighbor_list:
        print(f"[NEIGHBORS] {neighbor_list}")
    else:
        print("[NEIGHBORS] none")

    best_val_f1 = -1.0
    best_epoch = -1
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_f1_macro": [],
        "val_f1_macro": [],
    }

    # =========================================================
    # TRAIN LOOP
    # =========================================================
    for epoch in range(1, int(args.epochs) + 1):
        e0 = time.time()

        tr = run_epoch(
            model, dl_tr, opt, loss_fn,
            C=C, d21_idx=args.d21_internal,
            device=device, bg=bg,
            train=True, use_amp=args.use_amp,
            grad_clip=args.grad_clip
        )

        va = run_epoch(
            model, dl_va, None, loss_fn,
            C=C, d21_idx=args.d21_internal,
            device=device, bg=bg,
            train=False, use_amp=False
        )

        sched.step()

        history["train_loss"].append(tr["loss"])
        history["val_loss"].append(va["loss"])
        history["train_f1_macro"].append(tr["f1_macro"])
        history["val_f1_macro"].append(va["f1_macro"])

        torch.save({"model": model.state_dict(), "epoch": epoch}, last_path)

        if va["f1_macro"] > best_val_f1:
            best_val_f1 = va["f1_macro"]
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "epoch": epoch}, best_path)

        sec = time.time() - e0

        print(
            f"[{epoch:03d}/{args.epochs}] "
            f"train loss={tr['loss']:.4f} f1m={tr['f1_macro']:.3f} | "
            f"val loss={va['loss']:.4f} f1m={va['f1_macro']:.3f} | "
            f"d21_f1={va['d21_f1']:.3f} "
            f"pred_bg_frac(val)={va['pred_bg_frac']:.3f} "
            f"sec={sec:.1f}"
        )

    # =========================================================
    # TEST
    # =========================================================
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"[test] Cargado best.pt (epoch={ckpt.get('epoch')})")

    test_metrics = run_epoch(
        model, dl_te, None, loss_fn,
        C=C, d21_idx=args.d21_internal,
        device=device, bg=bg,
        train=False, use_amp=False
    )

    print(
        f"[test] loss={test_metrics['loss']:.4f} "
        f"f1m={test_metrics['f1_macro']:.3f} "
        f"d21_f1={test_metrics['d21_f1']:.3f} "
        f"pred_bg_frac(test)={test_metrics['pred_bg_frac']:.3f}"
    )

    print(f"[DONE] best_epoch={best_epoch} best_val_f1={best_val_f1:.4f}")


if __name__ == "__main__":
    main()


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# PARTE 4/5 TERMINA AQUÍ.
# En la PARTE 5/5:
#   - Añadimos:
#       * CSV completo por epoch (igual que v5)
#       * history.json
#       * plots Train vs Val
#       * evaluación neighbors en test
#       * bloque completo de inferencia visual (GT/Pred/Errores/d21)
#   - Y al final te dejo comandos exactos para correrlo en CUDA device 0.
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# ============================================================
# LOGGING AVANZADO + NEIGHBORS + PLOTS + INFERENCIA
# (PARTE 5/5 – cierre estilo PointNet/DGCNN v5)
# ============================================================

# ----------------------------------------------------------------
# Extensión del TRAIN LOOP para:
#   - metrics_epoch.csv
#   - history.json
#   - neighbors evaluados periódicamente
#   - plots train vs val
#   - test_metrics.json
#   - inferencia visual trazable
# ----------------------------------------------------------------

    # ================================
    # CSV de métricas por epoch
    # ================================
    csv_path = out_dir / "metrics_epoch.csv"
    csv_header_written = False

    def write_epoch_csv(epoch, tr, va, neighbor_metrics=None):
        nonlocal csv_header_written
        row = {
            "epoch": epoch,
            "train_loss": tr["loss"],
            "train_f1_macro": tr["f1_macro"],
            "val_loss": va["loss"],
            "val_f1_macro": va["f1_macro"],
            "val_d21_f1": va["d21_f1"],
            "val_pred_bg_frac": va["pred_bg_frac"],
        }

        if neighbor_metrics:
            row.update(neighbor_metrics)

        fieldnames = list(row.keys())

        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not csv_header_written:
                writer.writeheader()
                csv_header_written = True
            writer.writerow(row)

    # ================================
    # Loop extendido con neighbors + CSV
    # ================================
    for epoch in range(1, int(args.epochs) + 1):
        e0 = time.time()

        tr = run_epoch(
            model, dl_tr, opt, loss_fn,
            C=C, d21_idx=args.d21_internal,
            device=device, bg=bg,
            train=True, use_amp=args.use_amp,
            grad_clip=args.grad_clip
        )

        va = run_epoch(
            model, dl_va, None, loss_fn,
            C=C, d21_idx=args.d21_internal,
            device=device, bg=bg,
            train=False, use_amp=False
        )

        sched.step()

        neighbor_metrics = {}
        if neighbor_list and args.neighbor_eval_split != "none":
            if epoch % int(args.neighbor_every) == 0:
                target_loader = dl_va if args.neighbor_eval_split == "val" else dl_te
                neighbor_metrics = eval_neighbors_on_loader(
                    model, target_loader, device, neighbor_list, bg
                )

                if neighbor_metrics:
                    print("[neighbors]",
                          " ".join([f"{k}={v:.3f}" for k, v in neighbor_metrics.items()]))

        write_epoch_csv(epoch, tr, va, neighbor_metrics)

        history["train_loss"].append(tr["loss"])
        history["val_loss"].append(va["loss"])
        history["train_f1_macro"].append(tr["f1_macro"])
        history["val_f1_macro"].append(va["f1_macro"])

        torch.save({"model": model.state_dict(), "epoch": epoch}, last_path)

        if va["f1_macro"] > best_val_f1:
            best_val_f1 = va["f1_macro"]
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "epoch": epoch}, best_path)

        sec = time.time() - e0

        print(
            f"[{epoch:03d}/{args.epochs}] "
            f"train loss={tr['loss']:.4f} f1m={tr['f1_macro']:.3f} | "
            f"val loss={va['loss']:.4f} f1m={va['f1_macro']:.3f} | "
            f"d21_f1={va['d21_f1']:.3f} "
            f"sec={sec:.1f}"
        )

    # ================================
    # Guardar history.json
    # ================================
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # ================================
    # Plots Train vs Val
    # ================================
    plt.figure(figsize=(6,4), dpi=200)
    plt.plot(history["train_f1_macro"], label="train_f1_macro")
    plt.plot(history["val_f1_macro"], label="val_f1_macro")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("F1 macro")
    plt.title("Train vs Val F1 macro")
    plt.tight_layout()
    plt.savefig(out_dir / "plot_f1_macro.png")
    plt.close()

    # ================================
    # TEST FINAL
    # ================================
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"[test] best epoch={ckpt.get('epoch')}")

    test_metrics = run_epoch(
        model, dl_te, None, loss_fn,
        C=C, d21_idx=args.d21_internal,
        device=device, bg=bg,
        train=False, use_amp=False
    )

    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    print("[TEST]",
          " ".join([f"{k}={v:.4f}" for k, v in test_metrics.items()]))

    # ================================
    # INFERENCIA VISUAL
    # ================================
    if args.do_infer:
        model.eval()
        infer_dir = out_dir / "inference"
        infer_dir.mkdir(parents=True, exist_ok=True)

        examples = min(args.infer_examples, len(ds_te))

        for i in range(examples):
            xyz, y = ds_te[i]
            xyz = xyz.unsqueeze(0).to(device)
            y = y.to(device)

            with torch.no_grad():
                logits = model(xyz)
                pred = logits.argmax(dim=-1).squeeze(0)

            xyz_np = xyz.squeeze(0).cpu().numpy()
            y_np = y.cpu().numpy()
            pred_np = pred.cpu().numpy()

            plot_pointcloud_all_classes(
                xyz_np, y_np, pred_np,
                infer_dir / f"{i:03d}_all.png",
                C=C,
                title=f"Sample {i}"
            )

            plot_errors(
                xyz_np, y_np, pred_np,
                infer_dir / f"{i:03d}_errors.png",
                bg=bg,
                title=f"Errors {i}"
            )

            plot_d21_focus(
                xyz_np, y_np, pred_np,
                infer_dir / f"{i:03d}_d21.png",
                d21_idx=args.d21_internal,
                bg=bg,
                title=f"d21 focus {i}"
            )

        print(f"[INFER] saved {examples} examples to {infer_dir}")

    print(f"[DONE] best_epoch={best_epoch} best_val_f1={best_val_f1:.4f}")