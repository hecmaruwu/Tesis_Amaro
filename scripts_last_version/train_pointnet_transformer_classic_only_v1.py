#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_pointnet_transformer_classic_only_v1.py

Point Transformer (PointNet-Transformer style) – segmentación multiclase dental 3D
✅ Mantiene el mismo estilo de logging/outputs que tu train_pointnetpp_classic_only_fixed_v4.py
✅ history.json, metrics_epoch.csv, best.pt, last.pt, test_metrics.json
✅ Métricas epoch-level: macro F1/IoU sin BG, acc_all/acc_no_bg por conteo global
✅ d21 binario acumulado (sin BG) + d21(bin all) incluyendo BG
✅ pred_bg_frac y gt_bg_frac (baseline) en val
✅ Infer trazable con idx_local REAL + discovery index_*.csv robusto
✅ Geometría/KNN SIEMPRE FP32 aunque uses AMP

Uso típico (igual que antes, con extras del transformer):
  python train_pointnet_transformer_classic_only_v1.py \
    --data_dir .../fixed_split/8192/... \
    --out_dir  .../outputs/point_transformer_v1 \
    --epochs 200 --batch_size 8 --num_workers 4 \
    --lr 1e-3 --weight_decay 1e-4 --dropout 0.1 --grad_clip 1.0 \
    --use_amp \
    --d21_internal 8 \
    --infer_split test --infer_max 12 \
    --pt_dim 96 --pt_depth 6 --pt_k 16 --pt_heads 4 --pt_ffn_mult 4 --pt_attn_drop 0.1
"""

# ============================================================
# PARTE 1/4
# - imports
# - seed / io
# - dataset + loaders (train/val/test + infer con idx_local real)
# - args (incluye hiperparámetros extra del Point Transformer)
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


# ----------------------------
# ARGS
# ----------------------------
def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)     # (en transformer: drop MLP/head)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--use_amp", action="store_true")

    ap.add_argument("--bg", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)

    # Igual que tu clásico: índice interno d21 directo
    ap.add_argument("--d21_internal", type=int, default=-1)

    # Infer opcional (igual)
    ap.add_argument("--infer_split", type=str, default=None, choices=["train", "val", "test"])
    ap.add_argument("--infer_max", type=int, default=12)

    # -------- extras ad hoc Point Transformer --------
    ap.add_argument("--pt_dim", type=int, default=96, help="ancho embedding")
    ap.add_argument("--pt_depth", type=int, default=6, help="#bloques transformer")
    ap.add_argument("--pt_k", type=int, default=16, help="vecinos kNN por punto")
    ap.add_argument("--pt_heads", type=int, default=4, help="#heads (si aplica en el bloque)")
    ap.add_argument("--pt_ffn_mult", type=int, default=4, help="expansión FFN: dim*mult")
    ap.add_argument("--pt_attn_drop", type=float, default=0.1, help="dropout de atención")
    ap.add_argument("--pt_pe_dim", type=int, default=16, help="dim del MLP pos-enc (relativo)")
    ap.add_argument("--pt_stem", type=str, default="mlp", choices=["mlp", "linear"],
                    help="stem: mlp (más no-lineal) vs linear (más liviano)")
    ap.add_argument("--pt_norm", type=str, default="ln", choices=["ln", "bn"],
                    help="normalización interna (LN recomendado en transformer)")

    return ap


# --- fin parte 1/4 ---

# ============================================================
# PARTE 2/4
# - Utilidades geométricas (kNN) SIEMPRE FP32 (aunque uses AMP)
# - Bloques Point Transformer (atención local + PE relativa)
# - Modelo PointTransformerSeg -> logits [B,N,C]
# ============================================================

def _as_fp32(x: torch.Tensor) -> torch.Tensor:
    return x.float() if x.dtype != torch.float32 else x


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    src: [B,N,3], dst: [B,M,3] -> dist^2: [B,N,M]
    (FP32 siempre)
    """
    src = _as_fp32(src)
    dst = _as_fp32(dst)
    dist = -2.0 * torch.matmul(src, dst.transpose(2, 1))
    dist += torch.sum(src ** 2, dim=-1, keepdim=True)
    dist += torch.sum(dst ** 2, dim=-1, keepdim=True).transpose(2, 1)
    return torch.clamp(dist, min=0.0)


def knn_indices(xyz: torch.Tensor, k: int) -> torch.Tensor:
    """
    xyz: [B,N,3] -> idx: [B,N,k] (excluye self si es posible)
    Implementación simple O(N^2) con cdist (FP32).
    Para N=8192 y batch grande puede ser pesado, pero es estable y sin deps externas.
    """
    k = int(k)
    with torch.cuda.amp.autocast(enabled=False):
        xyz = _as_fp32(xyz)
        # dist: [B,N,N]
        dist = torch.cdist(xyz, xyz, p=2.0)  # FP32
        # evita self como vecino #0 (si k < N)
        if xyz.shape[1] > 1:
            eye = torch.eye(xyz.shape[1], device=xyz.device, dtype=dist.dtype).unsqueeze(0)  # [1,N,N]
            dist = dist + eye * 1e6
        idx = dist.topk(k=min(k, xyz.shape[1]-1 if xyz.shape[1] > 1 else 1), dim=-1, largest=False).indices
        # si N==1, idx será [B,1,1] con 0
        if idx.shape[-1] < k:
            # pad repitiendo el primer vecino
            pad = k - idx.shape[-1]
            idx = torch.cat([idx, idx[..., :1].repeat(1, 1, pad)], dim=-1)
        return idx


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    points: [B,N,C]
    idx: [B,N,k] -> out: [B,N,k,C]
    """
    B, N, C = points.shape
    idx = torch.clamp(idx, 0, N - 1)
    device = points.device
    batch = torch.arange(B, device=device).view(B, 1, 1).expand(B, idx.shape[1], idx.shape[2])
    return points[batch, idx, :]  # [B,N,k,C]


class MLP(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int, out_ch: int, drop: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(in_ch), int(hidden_ch)),
            nn.GELU(),
            nn.Dropout(float(drop)),
            nn.Linear(int(hidden_ch), int(out_ch)),
        )

    def forward(self, x):
        return self.net(x)


class Norm1D(nn.Module):
    """
    Norm sobre el canal C para tensores [B,N,C].
    - ln: LayerNorm(C)
    - bn: BatchNorm1d(C) aplicado como [B,C,N] y vuelve
    """
    def __init__(self, C: int, kind: str = "ln"):
        super().__init__()
        kind = str(kind).lower()
        self.kind = kind
        if kind == "ln":
            self.n = nn.LayerNorm(int(C))
        elif kind == "bn":
            self.n = nn.BatchNorm1d(int(C))
        else:
            raise ValueError(f"norm inválida: {kind}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "ln":
            return self.n(x)
        # bn
        x = x.transpose(2, 1).contiguous()  # [B,C,N]
        x = self.n(x)
        return x.transpose(2, 1).contiguous()


class PointTransformerBlock(nn.Module):
    """
    Atención local tipo Point Transformer:
      - vecinos por kNN en xyz
      - PE relativa basada en (pi - pj)
      - agregación softmax sobre vecinos
    """
    def __init__(self, dim: int, k: int, heads: int, ffn_mult: int,
                 attn_drop: float, drop: float, pe_dim: int, norm: str = "ln"):
        super().__init__()
        self.dim = int(dim)
        self.k = int(k)
        self.heads = max(1, int(heads))
        assert self.dim % self.heads == 0, "pt_dim debe ser divisible por pt_heads"
        self.dh = self.dim // self.heads

        self.norm1 = Norm1D(self.dim, kind=norm)
        self.norm2 = Norm1D(self.dim, kind=norm)

        # proyecciones QKV
        self.to_q = nn.Linear(self.dim, self.dim, bias=False)
        self.to_k = nn.Linear(self.dim, self.dim, bias=False)
        self.to_v = nn.Linear(self.dim, self.dim, bias=False)

        # PE relativa: (Δxyz -> emb -> modula scores y/o valores)
        # estilo "MLP(pos) -> dim"
        hid_pe = max(int(pe_dim), 8)
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, hid_pe),
            nn.GELU(),
            nn.Linear(hid_pe, self.dim),
        )

        self.attn_drop = nn.Dropout(float(attn_drop))
        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(float(drop))

        # FFN
        hid = int(self.dim * max(1, int(ffn_mult)))
        self.ffn = nn.Sequential(
            nn.Linear(self.dim, hid),
            nn.GELU(),
            nn.Dropout(float(drop)),
            nn.Linear(hid, self.dim),
            nn.Dropout(float(drop)),
        )

    def forward(self, x: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        """
        x:   [B,N,C]
        xyz: [B,N,3]
        """
        B, N, C = x.shape

        # ---- atención local (kNN en FP32 SIEMPRE) ----
        with torch.cuda.amp.autocast(enabled=False):
            xyz_f = _as_fp32(xyz)
            idx = knn_indices(xyz_f, self.k)              # [B,N,k]
            # Δxyz: [B,N,k,3]
            nbr_xyz = index_points(xyz_f, idx)            # [B,N,k,3]
            rel = (xyz_f.unsqueeze(2) - nbr_xyz).float()  # FP32

        # ---- atención (puede ir en AMP, pero rel/idx ya quedaron bien) ----
        h = self.norm1(x)

        q = self.to_q(h).view(B, N, self.heads, self.dh)  # [B,N,H,D]
        k = self.to_k(h).view(B, N, self.heads, self.dh)
        v = self.to_v(h).view(B, N, self.heads, self.dh)

        # vecinos en feature-space: [B,N,k,H,D]
        k_n = index_points(k.view(B, N, -1), idx).view(B, N, self.k, self.heads, self.dh)
        v_n = index_points(v.view(B, N, -1), idx).view(B, N, self.k, self.heads, self.dh)

        # PE relativa: [B,N,k,C] -> [B,N,k,H,D]
        pe = self.pos_mlp(rel.to(h.dtype))  # mantiene dtype de h
        pe = pe.view(B, N, self.k, self.heads, self.dh)

        # score: (q · (k+pe)) / sqrt(D)
        q_e = q.unsqueeze(2)                                # [B,N,1,H,D]
        attn_logits = (q_e * (k_n + pe)).sum(dim=-1)        # [B,N,k,H]
        attn_logits = attn_logits / (self.dh ** 0.5)

        attn = F.softmax(attn_logits, dim=2)                # [B,N,k,H]
        attn = self.attn_drop(attn)

        # value: sum attn * (v + pe)
        out = (attn.unsqueeze(-1) * (v_n + pe)).sum(dim=2)  # [B,N,H,D]
        out = out.reshape(B, N, C)                          # [B,N,C]

        out = self.proj(out)
        out = self.proj_drop(out)
        x = x + out

        # ---- FFN ----
        x = x + self.ffn(self.norm2(x))
        return x


class PointTransformerSeg(nn.Module):
    """
    Segmentación punto-a-punto:
      input xyz [B,N,3]
      output logits [B,N,num_classes]
    """
    def __init__(self,
                 num_classes: int,
                 pt_dim: int = 96,
                 pt_depth: int = 6,
                 pt_k: int = 16,
                 pt_heads: int = 4,
                 pt_ffn_mult: int = 4,
                 pt_attn_drop: float = 0.1,
                 dropout: float = 0.1,
                 pt_pe_dim: int = 16,
                 pt_stem: str = "mlp",
                 pt_norm: str = "ln"):
        super().__init__()
        C = int(num_classes)
        D = int(pt_dim)

        stem = str(pt_stem).lower()
        if stem == "linear":
            self.stem = nn.Linear(3, D)
        elif stem == "mlp":
            self.stem = nn.Sequential(
                nn.Linear(3, D),
                nn.GELU(),
                nn.Linear(D, D),
            )
        else:
            raise ValueError(f"pt_stem inválido: {pt_stem}")

        self.drop_in = nn.Dropout(float(dropout))
        self.blocks = nn.ModuleList([
            PointTransformerBlock(
                dim=D,
                k=int(pt_k),
                heads=int(pt_heads),
                ffn_mult=int(pt_ffn_mult),
                attn_drop=float(pt_attn_drop),
                drop=float(dropout),
                pe_dim=int(pt_pe_dim),
                norm=str(pt_norm),
            )
            for _ in range(int(pt_depth))
        ])

        self.norm_out = Norm1D(D, kind=str(pt_norm))

        # head de segmentación (similar idea a PointNet head)
        self.head = nn.Sequential(
            nn.Linear(D, D),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(D, C),
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        xyz: [B,N,3] -> logits: [B,N,C]
        """
        xyz = xyz.float()
        x = self.stem(xyz)        # [B,N,D]
        x = self.drop_in(x)

        for blk in self.blocks:
            x = blk(x, xyz)

        x = self.norm_out(x)
        logits = self.head(x)     # [B,N,C]
        return logits


# --- fin parte 2/4 ---

# ============================================================
# PARTE 3/4
# - Métricas epoch-level (idénticas a tu PointNet/PointNet++)
# - Confusion matrix global
# - Macro F1 / IoU sin background
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
# RUN EPOCH (train / val / test)
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

        # d21 binario
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
# - discovery + lectura robusta index_*.csv (dtype=str)
# - plots (GT vs Pred, errores, foco d21)
# - main(): train/val/test + best.pt + history.json + metrics_epoch.csv
# - inferencia trazable con idx_local real
# ============================================================

def _read_index_csv(path: Optional[Path]) -> Optional[Dict[int, Dict[str, str]]]:
    """
    row_i (fila dentro del split) -> meta dict con strings.
    Lee dtype=str para evitar crashes de pandas.
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
      2) ancestros hasta Teeth_3ds
      3) Teeth_3ds/merged_*/index_{split}.csv (más reciente)
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


def plot_pointcloud_all_classes(xyz, y_gt, y_pr, out_png, C, title="", s=1.0):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cols = _class_colors(C)

    xyz = _safe_np(xyz, np.float32)
    y_gt = _safe_np(y_gt, np.int32)
    y_pr = _safe_np(y_pr, np.int32)

    fig = plt.figure(figsize=(12, 5), dpi=200)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    c_gt = np.array([cols[int(k)] for k in y_gt])
    c_pr = np.array([cols[int(k)] for k in y_pr])

    ax1.scatter(xyz[:,0], xyz[:,1], xyz[:,2], c=c_gt, s=s, linewidths=0)
    ax2.scatter(xyz[:,0], xyz[:,1], xyz[:,2], c=c_pr, s=s, linewidths=0)

    ax1.set_title("GT")
    ax2.set_title("Pred")
    for ax in (ax1, ax2):
        ax.set_axis_off()
        ax.view_init(20, 45)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_errors(xyz, y_gt, y_pr, out_png, bg=0, title="", s=1.0):
    out_png.parent.mkdir(parents=True, exist_ok=True)

    xyz = _safe_np(xyz, np.float32)
    y_gt = _safe_np(y_gt, np.int32)
    y_pr = _safe_np(y_pr, np.int32)

    ok = (y_gt == y_pr)
    c = np.zeros((xyz.shape[0], 4), np.float32)
    c[:] = (0.75, 0.75, 0.75, 1.0)
    c[~ok] = (0.85, 0.10, 0.10, 1.0)
    c[y_gt == bg] = (0.85, 0.85, 0.85, 0.6)

    fig = plt.figure(figsize=(6, 5), dpi=200)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], c=c, s=s, linewidths=0)
    ax.set_axis_off()
    ax.view_init(20, 45)
    ax.set_title("Errores (rojo)")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_d21_focus(xyz, y_gt, y_pr, out_png, d21_idx, bg=0, title="", s=1.2):
    out_png.parent.mkdir(parents=True, exist_ok=True)

    xyz = _safe_np(xyz, np.float32)
    y_gt = _safe_np(y_gt, np.int32)
    y_pr = _safe_np(y_pr, np.int32)

    gt21 = (y_gt == d21_idx)
    pr21 = (y_pr == d21_idx)
    tp = gt21 & pr21
    err = gt21 ^ pr21

    c = np.zeros((xyz.shape[0], 4), np.float32)
    c[:] = (0.75, 0.75, 0.75, 1.0)
    c[y_gt == bg] = (0.85, 0.85, 0.85, 0.6)
    c[tp] = (0.10, 0.75, 0.25, 1.0)
    c[err] = (0.85, 0.10, 0.10, 1.0)

    fig = plt.figure(figsize=(6, 5), dpi=200)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], c=c, s=s, linewidths=0)
    ax.set_axis_off()
    ax.view_init(20, 45)
    ax.set_title("Foco d21")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================
def main():
    args = build_argparser().parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # num_classes robusto
    label_map_path = data_dir / "label_map.json"
    if label_map_path.exists():
        lm = json.load(open(label_map_path))
        num_classes = int(lm.get("num_classes", 0))
    else:
        Ytr = np.load(data_dir / "Y_train.npz")["Y"]
        num_classes = int(Ytr.max()) + 1

    d21_idx = int(args.d21_internal)

    print(f"[INFO] device={device} | C={num_classes} | bg={args.bg} | d21_idx={d21_idx}")

    dl_tr, dl_va, dl_te = make_loaders(
        data_dir, args.batch_size, args.num_workers, normalize=True
    )

    model = PointTransformerSeg(
        num_classes=num_classes,
        pt_dim=args.pt_dim,
        pt_depth=args.pt_depth,
        pt_k=args.pt_k,
        pt_heads=args.pt_heads,
        pt_ffn_mult=args.pt_ffn_mult,
        pt_attn_drop=args.pt_attn_drop,
        dropout=args.dropout,
        pt_pe_dim=args.pt_pe_dim,
        pt_stem=args.pt_stem,
        pt_norm=args.pt_norm,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    loss_fn = nn.CrossEntropyLoss()

    history = []
    best = {"epoch": -1, "val_f1_macro": -1.0}

    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(
            model, dl_tr, optimizer, loss_fn,
            num_classes, d21_idx, device,
            args.bg, True, args.use_amp, args.grad_clip
        )
        va = run_epoch(
            model, dl_va, None, loss_fn,
            num_classes, d21_idx, device,
            args.bg, False, args.use_amp
        )

        lr_now = _get_lr(optimizer)

        print(
            f"[{epoch}/{args.epochs}] "
            f"train loss={tr['loss']:.4f} f1m={tr['f1_macro']:.3f} ioum={tr['iou_macro']:.3f} "
            f"acc_all={tr['acc_all']:.3f} acc_no_bg={tr['acc_no_bg']:.3f} | "
            f"val loss={va['loss']:.4f} f1m={va['f1_macro']:.3f} ioum={va['iou_macro']:.3f} "
            f"acc_all={va['acc_all']:.3f} acc_no_bg={va['acc_no_bg']:.3f} | "
            f"d21 f1={va['d21_f1']:.3f} iou={va['d21_iou']:.3f} | "
            f"lr={lr_now:.2e}"
        )

        history.append({
            "epoch": epoch,
            **{f"train_{k}": v for k, v in tr.items()},
            **{f"val_{k}": v for k, v in va.items()},
            "lr": lr_now,
        })

        if va["f1_macro"] > best["val_f1_macro"]:
            best = {"epoch": epoch, "val_f1_macro": va["f1_macro"]}
            torch.save(model.state_dict(), out_dir / "best.pt")

        torch.save(model.state_dict(), out_dir / "last.pt")
        scheduler.step()

    save_json(
        {"best_epoch": best["epoch"], "best_val_f1_macro": best["val_f1_macro"], "history": history},
        out_dir / "history.json",
    )

    with open(out_dir / "metrics_epoch.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=history[0].keys())
        w.writeheader()
        for r in history:
            w.writerow(r)

    if (out_dir / "best.pt").exists():
        model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))

    te = run_epoch(
        model, dl_te, None, loss_fn,
        num_classes, d21_idx, device,
        args.bg, False, args.use_amp
    )

    save_json(
        {
            "best_epoch": best["epoch"],
            "test": te,
            "elapsed_sec": time.time() - t0,
            "num_classes": num_classes,
            "d21_idx": d21_idx,
        },
        out_dir / "test_metrics.json",
    )

    print("[TEST]", te)


if __name__ == "__main__":
    main()
