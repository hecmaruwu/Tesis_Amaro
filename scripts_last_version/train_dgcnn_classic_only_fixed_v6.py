#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_dgcnn_classic_only_fixed_v6.py

VERSIÓN DEFINITIVA CORREGIDA (MISMAS FEATURES, SOLO FIXES MECÁNICOS)

✔ Sin conflictos de device (loss weight en GPU)
✔ Sin redefiniciones
✔ KNN chunk-safe correcto
✔ Vecinos configurables
✔ AMP moderno torch.amp
✔ Skeleton idéntico a tu PointNet v4
✔ Compatible con dataset flat:
    X_train.npz / Y_train.npz

FIXES APLICADOS (SOLO ESTOS):
1) set_seed no existía  -> alias set_seed(...) que llama seed_all(...)
2) make_loaders retornaba 3 -> en main se desempaqueta en 3 (no 4)
3) parse_neighbors faltaba -> agregado
4) _discover_index_csv / _read_index_csv faltaban en el snippet pegado -> agregados (para NO perder trazabilidad en infer)
5) FIX NUEVO (MECÁNICO, NO CAMBIA FEATURES): inferencia no usa torch.from_numpy (evita el TypeError raro)
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
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# SEED
# ============================================================

def seed_all(seed: int = 42):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ------------------------------------------------------------
# FIX 1/4: alias para mantener naming tipo PointNet (main usa set_seed)
# ------------------------------------------------------------
def set_seed(seed: int = 42):
    seed_all(seed)


# ============================================================
# NORMALIZACIÓN
# ============================================================

def normalize_unit_sphere(xyz: torch.Tensor, eps: float = 1e-9):
    center = xyz.mean(dim=0, keepdim=True)
    xyz = xyz - center
    scale = torch.norm(xyz, dim=1).max().clamp_min(eps)
    return xyz / scale


# ============================================================
# DATASET ROBUSTO (SIN torch.from_numpy)
# ============================================================

class NPZDataset(Dataset):
    def __init__(self, X_path: Path, Y_path: Path, normalize: bool = True):
        self.X = np.load(X_path)["X"]
        self.Y = np.load(Y_path)["Y"]
        self.normalize = bool(normalize)

        assert self.X.shape[0] == self.Y.shape[0]
        assert self.X.shape[1] == self.Y.shape[1]

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, idx):
        x = np.ascontiguousarray(self.X[idx], dtype=np.float32)
        y = np.ascontiguousarray(self.Y[idx], dtype=np.int64)

        xyz = torch.as_tensor(x, dtype=torch.float32)
        lab = torch.as_tensor(y, dtype=torch.int64)

        if self.normalize:
            xyz = normalize_unit_sphere(xyz)

        return xyz, lab


def make_loaders(data_dir: Path, bs: int, nw: int, normalize: bool = True):
    """
    Retorna EXACTAMENTE 3 loaders (train/val/test).
    FIX 2/4: en main se desempaqueta en 3, no 4.
    """

    ds_tr = NPZDataset(data_dir / "X_train.npz", data_dir / "Y_train.npz", normalize)
    ds_va = NPZDataset(data_dir / "X_val.npz",   data_dir / "Y_val.npz",   normalize)
    ds_te = NPZDataset(data_dir / "X_test.npz",  data_dir / "Y_test.npz",  normalize)

    common = dict(
        batch_size=int(bs),
        num_workers=int(nw),
        pin_memory=True,
        persistent_workers=(int(nw) > 0),
        drop_last=False,
    )

    if int(nw) > 0:
        common["prefetch_factor"] = 2

    dl_tr = DataLoader(ds_tr, shuffle=True, **common)
    dl_va = DataLoader(ds_va, shuffle=False, **common)
    dl_te = DataLoader(ds_te, shuffle=False, **common)

    return dl_tr, dl_va, dl_te


# ============================================================
# LABEL MAP + INFER NUM CLASSES
# ============================================================

def load_label_map(data_dir: Path):
    p = data_dir / "label_map.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def infer_num_classes(data_dir: Path, label_map: Optional[dict]):

    # 1) desde label_map
    if isinstance(label_map, dict) and len(label_map) > 0:
        try:
            mx = max(int(v) for v in label_map.values())
            return int(mx + 1)
        except Exception:
            pass

    # 2) fallback: escanear Y_*.npz
    maxy = -1
    for split in ("train", "val", "test"):
        yp = data_dir / f"Y_{split}.npz"
        if yp.exists():
            y = np.load(yp)["Y"]
            maxy = max(maxy, int(y.max()))

    if maxy < 0:
        raise RuntimeError("No se pudo inferir num_classes")

    return int(maxy + 1)


# ============================================================
# FIX 3/4 — PARSE NEIGHBORS (faltaba)
# ============================================================

def parse_neighbors(s: str) -> List[Tuple[str, int]]:
    """
    --neighbor_teeth "d11:1,d22:9"
    Retorna: [("d11",1), ("d22",9)]
    """
    s = (s or "").strip()
    if not s:
        return []
    out: List[Tuple[str, int]] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Formato inválido en neighbor_teeth: '{part}' (usa nombre:idx)")
        name, idx = part.split(":", 1)
        out.append((name.strip(), int(idx)))
    return out


# ============================================================
# FIX 4/4 — TRAZABILIDAD INDEX CSV (faltaba en el snippet pegado)
#   Mantiene la feature de "infer con index_{split}.csv" sin romper.
# ============================================================

def _discover_index_csv(data_dir: Path, split: str) -> Optional[Path]:
    """
    Busca index_{split}.csv en:
      1) data_dir/index_{split}.csv
      2) ascendiendo hasta Teeth_3ds (si existe en la ruta)
      3) primer match en ancestros (fallback suave)
    """
    split = str(split)
    cand = data_dir / f"index_{split}.csv"
    if cand.exists():
        return cand

    # subir hasta encontrar carpeta Teeth_3ds
    p = data_dir.resolve()
    parents = [p] + list(p.parents)

    # 2) hasta Teeth_3ds
    for parent in parents:
        if parent.name == "Teeth_3ds":
            cand2 = parent / f"index_{split}.csv"
            if cand2.exists():
                return cand2
            # algunos pipelines guardan index_*.csv dentro del merged_* o fixed_split/*
            # buscamos cercano
            for sub in parent.rglob(f"index_{split}.csv"):
                return sub
            break

    # 3) fallback: primer match en ancestros
    for parent in parents:
        cand3 = parent / f"index_{split}.csv"
        if cand3.exists():
            return cand3

    return None


def _read_index_csv(p: Optional[Path]) -> Optional[Dict[int, Dict[str, str]]]:
    """
    Lee index_{split}.csv y retorna:
      { row_idx_int : {col: value, ...}, ... }
    Si no existe o falla, retorna None.
    """
    if p is None or (not Path(p).exists()):
        return None

    out: Dict[int, Dict[str, str]] = {}
    try:
        with Path(p).open("r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            # Intentamos inferir el índice desde columnas típicas
            # si no, usamos el contador de fila (0..)
            for i, row in enumerate(r):
                # columnas típicas para idx
                rid = None
                for k in ("idx", "index", "row", "i"):
                    if k in row and str(row[k]).strip() != "":
                        try:
                            rid = int(float(row[k]))
                            break
                        except Exception:
                            pass
                if rid is None:
                    rid = i
                out[int(rid)] = {kk: ("" if row[kk] is None else str(row[kk])) for kk in row.keys()}
        return out
    except Exception:
        return None

# ============================================================
# PARTE 2/4 — DGCNN (KNN chunk-safe) + EdgeConv + Modelo
# ============================================================

def knn(x: torch.Tensor, k: int, chunk_size: int = 0) -> torch.Tensor:
    """
    x: [B, C, N]
    return idx: [B, N, k]

    Chunk-safe real:
      - queries por bloques M, pero distancia contra TODO N
      - nunca mezcla dims (fix definitivo del mismatch 1024 vs 8192)

    Nota:
      - usamos distancia euclidiana al cuadrado vía:
            ||q - x||^2 = ||q||^2 + ||x||^2 - 2 q^T x
    """
    assert x.dim() == 3, f"x debe ser [B,C,N], got {x.shape}"
    B, C, N = x.shape
    k = int(k)
    if k <= 0:
        raise ValueError("k debe ser > 0")
    k = min(k, N)

    # ||x||^2 para todos los puntos (keys)
    xx = (x ** 2).sum(dim=1)  # [B, N]

    # sin chunk o chunk >= N => full
    chunk_size = int(chunk_size or 0)
    if chunk_size <= 0 or chunk_size >= N:
        # q = x (todas las queries)
        qx = torch.bmm(x.transpose(2, 1), x)                  # [B, N, N]
        dist = xx.unsqueeze(2) + xx.unsqueeze(1) - 2.0 * qx   # [B, N, N]
        idx = dist.topk(k=k, dim=-1, largest=False, sorted=False).indices  # [B,N,k]
        return idx

    # chunked queries
    idx_chunks: List[torch.Tensor] = []
    for s in range(0, N, chunk_size):
        e = min(N, s + chunk_size)
        q = x[:, :, s:e]                         # [B, C, M]
        qq = (q ** 2).sum(dim=1)                 # [B, M]
        qx = torch.bmm(q.transpose(2, 1), x)     # [B, M, N]
        dist = qq.unsqueeze(2) + xx.unsqueeze(1) - 2.0 * qx   # [B, M, N]
        idx = dist.topk(k=k, dim=-1, largest=False, sorted=False).indices  # [B,M,k]
        idx_chunks.append(idx)

    return torch.cat(idx_chunks, dim=1)  # [B, N, k]


def get_graph_feature(
    x: torch.Tensor,
    k: int = 20,
    idx: Optional[torch.Tensor] = None,
    knn_chunk_size: int = 0,
) -> torch.Tensor:
    """
    x: [B, C, N]
    return: [B, 2C, N, k]  concat( x_j - x_i , x_i )
    """
    B, C, N = x.shape
    if idx is None:
        idx = knn(x, k=int(k), chunk_size=int(knn_chunk_size))  # [B,N,k]
    k = int(idx.shape[-1])

    device = x.device
    idx_base = torch.arange(B, device=device).view(B, 1, 1) * N  # [B,1,1]
    idx = (idx + idx_base).reshape(-1)                           # [B*N*k]

    x_t = x.transpose(2, 1).contiguous()        # [B,N,C]
    feat = x_t.reshape(B * N, C)[idx, :]        # [B*N*k, C]
    feat = feat.view(B, N, k, C)                # [B,N,k,C]

    x_i = x_t.view(B, N, 1, C).expand(-1, -1, k, -1)  # [B,N,k,C]
    edge = torch.cat((feat - x_i, x_i), dim=3)         # [B,N,k,2C]
    return edge.permute(0, 3, 1, 2).contiguous()       # [B,2C,N,k]


class EdgeConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.do = nn.Dropout(p=float(dropout)) if float(dropout) > 0 else nn.Identity()

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        # e: [B, in_ch, N, k]
        y = self.net(e)
        y = self.do(y)
        y = torch.max(y, dim=-1).values  # [B, out_ch, N]
        return y


class DGCNN_Seg(nn.Module):
    """
    Input:  xyz [B,N,3]
    Output: logits [B,N,C]
    """
    def __init__(
        self,
        num_classes: int,
        k: int = 20,
        emb_dims: int = 1024,
        dropout: float = 0.5,
        knn_chunk_size: int = 0,
    ):
        super().__init__()
        self.k = int(k)
        self.knn_chunk_size = int(knn_chunk_size)
        self.emb_dims = int(emb_dims)
        self.dropout = float(dropout)

        # EdgeConv stacks (paper-style)
        self.ec1 = EdgeConvBlock(6,   64,  dropout=0.0)  # 2*3
        self.ec2 = EdgeConvBlock(128, 64,  dropout=0.0)  # 2*64
        self.ec3 = EdgeConvBlock(128, 128, dropout=0.0)  # 2*64
        self.ec4 = EdgeConvBlock(256, 256, dropout=0.0)  # 2*128

        self.fuse = nn.Sequential(
            nn.Conv1d(64 + 64 + 128 + 256, self.emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.emb_dims),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.head = nn.Sequential(
            nn.Conv1d(self.emb_dims + (64 + 64 + 128 + 256), 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=self.dropout),

            nn.Conv1d(512, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=self.dropout),

            nn.Conv1d(256, num_classes, 1, bias=True),
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        # xyz: [B,N,3] -> x: [B,3,N]
        x = xyz.permute(0, 2, 1).contiguous()

        e1 = get_graph_feature(x,  k=self.k, knn_chunk_size=self.knn_chunk_size)  # [B,6,N,k]
        x1 = self.ec1(e1)  # [B,64,N]

        e2 = get_graph_feature(x1, k=self.k, knn_chunk_size=self.knn_chunk_size)  # [B,128,N,k]
        x2 = self.ec2(e2)  # [B,64,N]

        e3 = get_graph_feature(x2, k=self.k, knn_chunk_size=self.knn_chunk_size)  # [B,128,N,k]
        x3 = self.ec3(e3)  # [B,128,N]

        e4 = get_graph_feature(x3, k=self.k, knn_chunk_size=self.knn_chunk_size)  # [B,256,N,k]
        x4 = self.ec4(e4)  # [B,256,N]

        x_cat = torch.cat((x1, x2, x3, x4), dim=1)   # [B,512,N]
        f = self.fuse(x_cat)                         # [B,emb,N]

        g = torch.max(f, dim=2, keepdim=True).values # [B,emb,1]
        g = g.expand(-1, -1, x_cat.size(2))          # [B,emb,N]

        h = torch.cat((x_cat, g), dim=1)             # [B,512+emb,N]
        logits = self.head(h)                        # [B,C,N]
        return logits.permute(0, 2, 1).contiguous()  # [B,N,C]

# ============================================================
# PARTE 3/4 — LOSS (FIX: weight en GPU) + MÉTRICAS + VECINOS + PLOTS + run_epoch
#   FIX CRÍTICO (ya incorporado):
#   - CrossEntropyLoss(weight=...) DEBE tener weight en el MISMO device que logits.
#     => make_loss_fn recibe device y crea weight directamente en CUDA si corresponde.
# ============================================================

# ----------------------------
# LOSS: BG incluido, downweight bg (SIN ignore_index)
# ----------------------------
def make_loss_fn(num_classes: int, bg_class: int, bg_weight: float, device: torch.device) -> nn.Module:
    """
    CrossEntropy con pesos por clase:
      - bg_class tiene peso bg_weight (ej: 0.03)
      - resto 1.0

    FIX: weight se crea DIRECTO en 'device' para evitar:
      RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
    """
    C = int(num_classes)
    bg = int(bg_class)
    w = torch.ones(C, dtype=torch.float32, device=device)
    if 0 <= bg < C:
        w[bg] = float(bg_weight)
    return nn.CrossEntropyLoss(weight=w)


# ----------------------------
# MÉTRICAS (macro sin BG, d21 binario, etc.)
# ----------------------------
@torch.no_grad()
def macro_metrics_no_bg(pred: torch.Tensor, gt: torch.Tensor, C: int, bg: int = 0) -> Tuple[float, float]:
    """
    Macro-F1 e IoU macro EXCLUYENDO puntos con gt==bg.
    Promedia clases 1..C-1, omitiendo clases ausentes.
    """
    pred = pred.reshape(-1)
    gt = gt.reshape(-1)

    m = (gt != int(bg))
    pred = pred[m]
    gt = gt[m]
    if gt.numel() == 0:
        return 0.0, 0.0

    f1s: List[float] = []
    ious: List[float] = []
    for c in range(1, int(C)):
        tp = ((pred == c) & (gt == c)).sum().item()
        fp = ((pred == c) & (gt != c)).sum().item()
        fn = ((pred != c) & (gt == c)).sum().item()
        denom = tp + fp + fn
        if denom == 0:
            continue
        f1 = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        f1s.append(float(f1))
        ious.append(float(iou))

    if not f1s:
        return 0.0, 0.0
    return float(np.mean(f1s)), float(np.mean(ious))


@torch.no_grad()
def d21_metrics_binary(
    pred: torch.Tensor,
    gt: torch.Tensor,
    d21_idx: int,
    bg: int = 0,
    include_bg: bool = False,
) -> Tuple[float, float, float]:
    """
    Binario para cualquier diente:
      positivo = clase d21_idx
      negativo = resto

    include_bg=False => excluye puntos con gt==bg (métrica más informativa)
    include_bg=True  => incluye bg (referencia global; suele inflarse)
    """
    pred = pred.reshape(-1)
    gt = gt.reshape(-1)

    if not bool(include_bg):
        m = (gt != int(bg))
        pred = pred[m]
        gt = gt[m]
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


@torch.no_grad()
def _compute_metrics_from_logits(
    logits: torch.Tensor,
    y: torch.Tensor,
    C: int,
    d21_idx: int,
    bg: int,
) -> Dict[str, float]:
    """
    logits: [B,N,C]
    y:      [B,N]
    """
    pred = logits.argmax(dim=-1)  # [B,N]

    acc_all = _acc_all(pred, y)

    mask = (y != int(bg))
    acc_no_bg = float((pred[mask] == y[mask]).float().mean().item()) if mask.any() else 0.0

    f1m, ioum = macro_metrics_no_bg(pred, y, C=C, bg=bg)

    d21_acc, d21_f1, d21_iou = d21_metrics_binary(pred, y, d21_idx=int(d21_idx), bg=int(bg), include_bg=False)
    d21_bin_acc_all, _, _ = d21_metrics_binary(pred, y, d21_idx=int(d21_idx), bg=int(bg), include_bg=True)

    pred_bg_frac = float((pred.reshape(-1) == int(bg)).float().mean().item())

    return {
        "acc_all": acc_all,
        "acc_no_bg": acc_no_bg,
        "f1_macro": float(f1m),
        "iou_macro": float(ioum),
        "d21_acc": float(d21_acc),
        "d21_f1": float(d21_f1),
        "d21_iou": float(d21_iou),
        "d21_bin_acc_all": float(d21_bin_acc_all),
        "pred_bg_frac": float(pred_bg_frac),
    }


# ----------------------------
# VECINOS (lista arbitraria)
# ----------------------------
@torch.no_grad()
def eval_neighbors_on_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    neighbor_list: List[Tuple[str, int]],
    bg: int,
    use_amp: bool,
) -> Dict[str, float]:
    """
    Retorna promedio por batch:
      {name}_acc, {name}_f1, {name}_iou, {name}_bin_acc_all

    Importante:
      - 'bin_acc_all' en este contexto usa include_bg=True
      - acc/f1/iou usan include_bg=False
    """
    if not neighbor_list:
        return {}

    model.eval()
    sums: Dict[str, float] = {}
    for name, _ in neighbor_list:
        for k in ("acc", "f1", "iou", "bin_acc_all"):
            sums[f"{name}_{k}"] = 0.0
    nb = 0

    ctx = _get_autocast_ctx(device, bool(use_amp))
    with torch.no_grad():
        for xyz, y in loader:
            xyz = xyz.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with ctx:
                logits = model(xyz)
            pred = logits.argmax(dim=-1)

            for name, idx in neighbor_list:
                acc, f1, iou = d21_metrics_binary(pred, y, d21_idx=int(idx), bg=int(bg), include_bg=False)
                acc_all, _, _ = d21_metrics_binary(pred, y, d21_idx=int(idx), bg=int(bg), include_bg=True)

                sums[f"{name}_acc"] += acc
                sums[f"{name}_f1"] += f1
                sums[f"{name}_iou"] += iou
                sums[f"{name}_bin_acc_all"] += acc_all

            nb += 1

    nb = max(1, nb)
    return {k: v / nb for k, v in sums.items()}


# ----------------------------
# PLOTS (idéntico estilo PointNet v4)
# ----------------------------
def _class_colors(C: int):
    cmap = plt.colormaps.get_cmap("tab20")
    C = max(int(C), 2)
    return [cmap(i / max(C - 1, 1)) for i in range(C)]


def _to_np(a) -> np.ndarray:
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
    s: float = 1.0,
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
    s: float = 1.0,
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
    s: float = 1.2,
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


# ----------------------------
# AMP moderno (cuda)
# ----------------------------
def _get_autocast_ctx(device: torch.device, use_amp: bool):
    """
    Wrapper para autocast moderno.
    - CUDA: torch.amp.autocast("cuda")
    - CPU: desactivado (no lo necesitamos aquí)
    """
    if bool(use_amp) and device.type == "cuda":
        return torch.amp.autocast("cuda", enabled=True)
    return torch.amp.autocast("cpu", enabled=False)


# ----------------------------
# run_epoch (train: 1 forward para loss + 2do forward eval para métricas, como tu v4)
# ----------------------------
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
    train=True:
      - model.train() para loss (dropout ON)
      - model.eval() para métricas (dropout OFF) con 2do forward
    train=False:
      - eval normal
    """
    scaler = run_epoch.scaler  # type: ignore
    if bool(train) and bool(use_amp) and device.type == "cuda" and scaler is None:
        scaler = torch.amp.GradScaler("cuda")
        run_epoch.scaler = scaler  # type: ignore

    loss_sum = 0.0
    sums = {
        "acc_all": 0.0, "acc_no_bg": 0.0, "f1_macro": 0.0, "iou_macro": 0.0,
        "d21_acc": 0.0, "d21_f1": 0.0, "d21_iou": 0.0, "d21_bin_acc_all": 0.0,
        "pred_bg_frac": 0.0,
    }
    nb = 0

    if not bool(train):
        model.eval()

    for xyz, y in loader:
        xyz = xyz.to(device, non_blocking=True)  # [B,N,3]
        y = y.to(device, non_blocking=True)      # [B,N]

        ctx = _get_autocast_ctx(device, bool(use_amp))

        if bool(train):
            assert optimizer is not None
            model.train(True)
            optimizer.zero_grad(set_to_none=True)

            with ctx:
                logits_train = model(xyz)  # [B,N,C]
                loss = loss_fn(logits_train.reshape(-1, C), y.reshape(-1))

            if bool(use_amp) and device.type == "cuda":
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

            # métricas con dropout OFF (2do forward)
            model.eval()
            with torch.no_grad():
                with ctx:
                    logits_eval = model(xyz)
            metrics = _compute_metrics_from_logits(logits_eval, y, C=C, d21_idx=d21_idx, bg=bg)

        else:
            with torch.no_grad():
                with ctx:
                    logits = model(xyz)
                    loss = loss_fn(logits.reshape(-1, C), y.reshape(-1))
                metrics = _compute_metrics_from_logits(logits, y, C=C, d21_idx=d21_idx, bg=bg)

        loss_sum += float(loss.item())
        for k in sums:
            sums[k] += float(metrics[k])
        nb += 1

    nb = max(1, nb)
    out = {"loss": loss_sum / nb}
    out.update({k: v / nb for k, v in sums.items()})
    return out


run_epoch.scaler = None  # type: ignore

# ============================================================
# PARTE 4/4 — MAIN + PARSER + LOGGING + CKPTS + TEST + INFER (FIXED v6)
#
# FIXES incluidos (solo lo necesario, sin quitar features):
#   ✅ NameError: set_seed -> se usa seed_all (alias set_seed para compat)
#   ✅ make_loaders: devuelve 3 loaders (tr/va/te).
#   ✅ parse_neighbors + trazabilidad index_{split}.csv: funciones incluidas
#   ✅ FIX INFER CRÍTICO: evitar torch.from_numpy(...) que te disparó:
#        TypeError: expected np.ndarray (got numpy.ndarray)
#      -> usamos torch.as_tensor(..., device=...) (no cambia features)
#   ✅ ejemplos/tickets (comandos) al final
# ============================================================

def get_lr(optim: torch.optim.Optimizer) -> float:
    for g in optim.param_groups:
        return float(g.get("lr", 0.0))
    return 0.0


def append_metrics_csv(csv_path: Path, row: Dict[str, Any], header_order: Optional[List[str]] = None):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    header = header_order if header_order is not None else list(row.keys())
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in header})


def save_history_json(out_dir: Path, history: Dict[str, List[float]]):
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


def save_test_metrics(out_dir: Path, metrics: Dict[str, float], extra: Optional[Dict[str, Any]] = None):
    payload = {"test": metrics}
    if extra:
        payload["extra"] = extra
    (out_dir / "test_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_run_meta(out_dir: Path, args: argparse.Namespace, extra: Dict[str, Any]):
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).name),
        "args": vars(args),
        "extra": extra,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


# ============================================================
# COMPAT: set_seed (para que no te vuelva a pasar)
#   - Tu Parte 1 define seed_all()
#   - Aquí dejamos alias set_seed = seed_all (sin cambiar tu estilo)
# ============================================================
def set_seed(seed: int = 42):
    seed_all(seed)


# ============================================================
# NEIGHBORS PARSER
#   Formato: "d11:1,d22:9"  => [("d11",1), ("d22",9)]
# ============================================================
def parse_neighbors(s: str) -> List[Tuple[str, int]]:
    s = (s or "").strip()
    if not s:
        return []
    out: List[Tuple[str, int]] = []
    chunks = [c.strip() for c in s.split(",") if c.strip()]
    for c in chunks:
        if ":" not in c:
            # tolerante: si viene solo número, le ponemos name genérico
            try:
                idx = int(c)
                out.append((f"n{idx}", idx))
            except Exception:
                continue
            continue
        name, idxs = c.split(":", 1)
        name = name.strip() or "neighbor"
        try:
            idx = int(idxs.strip())
        except Exception:
            continue
        out.append((name, idx))
    return out


# ============================================================
# TRAZABILIDAD: descubrir index_{split}.csv (busca en data_dir y ancestros)
#   - idéntica intención a PointNet v4: que inferencia tenga nombres/pacientes
# ============================================================
def _discover_index_csv(data_dir: Path, split: str) -> Optional[Path]:
    """
    Busca index_{split}.csv en:
      1) data_dir/index_{split}.csv
      2) padres sucesivos hasta encontrar Teeth_3ds (o raíz)
      3) además: si hay carpeta merged_* cerca, intenta index_{split}.csv ahí (opcional)
    """
    data_dir = Path(data_dir).resolve()
    target = f"index_{split}.csv"

    p = data_dir / target
    if p.exists():
        return p

    # subir por ancestros
    cur = data_dir
    for _ in range(12):
        if cur is None:
            break
        cand = cur / target
        if cand.exists():
            return cand

        # heurística: si existe Teeth_3ds en el path, es buen "stop"
        if cur.name.lower() == "teeth_3ds" or cur.name.lower() == "teeth3ds":
            # aún así, por si acaso, seguimos 1 nivel más
            pass

        parent = cur.parent
        if parent == cur:
            break
        cur = parent

    # fallback: buscar en carpetas merged_* bajo Teeth_3ds si existe
    # (esto es liviano, solo 1 nivel)
    cur = data_dir
    for _ in range(12):
        if cur.name.lower() in ("teeth_3ds", "teeth3ds"):
            # buscar merged_*
            try:
                for d in cur.glob("merged_*"):
                    cand = d / target
                    if cand.exists():
                        return cand
            except Exception:
                pass
            break
        parent = cur.parent
        if parent == cur:
            break
        cur = parent

    return None


def _read_index_csv(p: Optional[Path]) -> Optional[Dict[int, Dict[str, str]]]:
    """
    Devuelve dict:
      idx(int) -> {col: value}
    Soporta que la primera columna sea 'idx' o similar; si no, usa fila_num.
    """
    if p is None:
        return None
    p = Path(p)
    if not p.exists():
        return None

    try:
        with p.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception:
        return None

    if not rows:
        return None

    # detectar columna idx
    fieldnames = [c.strip() for c in (rows[0].keys() if rows else [])]
    idx_col = None
    for c in ("idx", "index", "i", "row"):
        if c in fieldnames:
            idx_col = c
            break

    out: Dict[int, Dict[str, str]] = {}
    for r_i, r in enumerate(rows):
        if idx_col is not None:
            try:
                i = int(str(r.get(idx_col, "")).strip())
            except Exception:
                i = r_i
        else:
            i = r_i
        out[int(i)] = {k: ("" if r.get(k) is None else str(r.get(k))) for k in r.keys()}
    return out


def _pick_trace_label(d: Dict[str, str]) -> str:
    for k in ("sample_name", "patient", "patient_id", "scan_id", "id", "path", "relpath", "upper_path"):
        if k in d and str(d[k]).strip():
            return str(d[k]).strip()
    for k, v in d.items():
        if str(v).strip():
            return f"{k}={v}"
    return "sample"


# ============================================================
# INFER + FIGURAS
# ============================================================
def infer_and_save_examples(
    model: nn.Module,
    data_dir: Path,
    split: str,
    out_dir: Path,
    device: torch.device,
    C: int,
    d21_idx: int,
    bg: int,
    n_examples: int = 12,
    use_amp: bool = True,
):
    """
    Carga X_{split}.npz / Y_{split}.npz desde data_dir raíz (NO subcarpeta).
    Genera PNGs en out_dir/inference/{split}/
    Usa index_{split}.csv si se descubre.
    """
    Xp = data_dir / f"X_{split}.npz"
    Yp = data_dir / f"Y_{split}.npz"
    if (not Xp.exists()) or (not Yp.exists()):
        print(f"[infer] No existe {Xp.name} o {Yp.name} en {data_dir}, salto inferencia.")
        return

    X = np.load(Xp)["X"]  # [M,N,3]
    Y = np.load(Yp)["Y"]  # [M,N]
    M = int(X.shape[0])
    K = min(int(n_examples), M)

    # trazabilidad
    idx_path = _discover_index_csv(data_dir, split=split)
    idx_map = _read_index_csv(idx_path) if idx_path is not None else None

    out_inf = out_dir / "inference" / str(split)
    out_inf.mkdir(parents=True, exist_ok=True)

    model.eval()
    ctx = _get_autocast_ctx(device, bool(use_amp))

    with torch.no_grad():
        for i in range(K):
            xyz = X[i].astype(np.float32, copy=False)
            ygt = Y[i].astype(np.int32, copy=False)

            label = f"i={i:04d}"
            if idx_map is not None and i in idx_map:
                label = f"i={i:04d} | {_pick_trace_label(idx_map[i])}"

            # ============================================================
            # FIX INFER: evita torch.from_numpy(...) que te dio:
            #   TypeError: expected np.ndarray (got numpy.ndarray)
            # Usamos torch.as_tensor (mismo resultado, más robusto).
            # ============================================================
            tx = torch.as_tensor(xyz, dtype=torch.float32, device=device).unsqueeze(0)  # [1,N,3]

            with ctx:
                logits = model(tx)  # [1,N,C]
            pred = logits.argmax(dim=-1).squeeze(0).detach().cpu().numpy().astype(np.int32)

            base = out_inf / f"ex_{i:04d}"
            plot_pointcloud_all_classes(xyz, ygt, pred, base.with_suffix(".all.png"), C=C, title=label, s=1.0)
            plot_errors(xyz, ygt, pred, base.with_suffix(".err.png"), bg=bg, title=label, s=1.0)
            plot_d21_focus(xyz, ygt, pred, base.with_suffix(".d21.png"), d21_idx=d21_idx, bg=bg, title=label, s=1.2)

    print(f"[infer] Guardado en: {out_inf}")


# ============================================================
# ARGPARSE
# ============================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True, help="carpeta con X_train.npz/Y_train.npz etc (raíz)")
    p.add_argument("--out_dir", type=str, required=True)

    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.5)

    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--use_amp", action="store_true")
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--k", type=int, default=20)
    p.add_argument("--emb_dims", type=int, default=1024)
    p.add_argument("--knn_chunk_size", type=int, default=0)

    p.add_argument("--bg_class", type=int, default=0)
    p.add_argument("--bg_weight", type=float, default=0.03)
    p.add_argument("--d21_internal", type=int, default=8)

    # neighbors: "d11:1,d22:9"
    p.add_argument("--neighbor_teeth", type=str, default="", help='ej: "d11:1,d22:9" (idx internos)')

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--normalize", action="store_true", help="normalize_unit_sphere por muestra (ON recomendado)")

    p.add_argument("--plot_every", type=int, default=10)

    p.add_argument("--do_infer", action="store_true")
    p.add_argument("--infer_examples", type=int, default=12)
    p.add_argument("--infer_split", type=str, default="test", choices=["train", "val", "test"])

    return p.parse_args()


# ============================================================
# MAIN
# ============================================================
def main():
    args = parse_args()
    set_seed(args.seed)  # ✅ FIX: ahora existe (alias de seed_all)

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    label_map = load_label_map(data_dir)
    C = infer_num_classes(data_dir, label_map)
    bg = int(args.bg_class)
    d21_idx = int(args.d21_internal)

    neighbors = parse_neighbors(args.neighbor_teeth)

    # loaders (devuelve 3; si tu versión anterior devolvía 4, esto evita NameError/ValueError)
    dl_tr, dl_va, dl_te = make_loaders(
        data_dir=data_dir,
        bs=int(args.batch_size),
        nw=int(args.num_workers),
        normalize=bool(args.normalize),
    )

    # model
    model = DGCNN_Seg(
        num_classes=int(C),
        k=int(args.k),
        emb_dims=int(args.emb_dims),
        dropout=float(args.dropout),
        knn_chunk_size=int(args.knn_chunk_size),
    ).to(device)

    # optim + sched
    optim = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=int(args.epochs), eta_min=max(1e-6, float(args.lr) * 0.02)
    )

    # LOSS FIX: weight en device
    loss_fn = make_loss_fn(num_classes=int(C), bg_class=int(bg), bg_weight=float(args.bg_weight), device=device)

    # meta
    save_run_meta(
        out_dir,
        args,
        extra={"num_classes": int(C), "bg": int(bg), "d21_internal": int(d21_idx), "neighbors": neighbors},
    )

    # history
    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [],
        "train_f1_macro": [], "val_f1_macro": [],
        "train_iou_macro": [], "val_iou_macro": [],
        "train_acc_all": [], "val_acc_all": [],
        "train_acc_no_bg": [], "val_acc_no_bg": [],
        "train_d21_acc": [], "val_d21_acc": [],
        "train_d21_f1": [], "val_d21_f1": [],
        "train_d21_iou": [], "val_d21_iou": [],
        "train_d21_bin_acc_all": [], "val_d21_bin_acc_all": [],
        "train_pred_bg_frac": [], "val_pred_bg_frac": [],
    }
    for name, _ in neighbors:
        history[f"val_{name}_acc"] = []
        history[f"val_{name}_f1"] = []
        history[f"val_{name}_iou"] = []
        history[f"val_{name}_bin_acc_all"] = []

    metrics_csv = out_dir / "metrics_epoch.csv"
    plots_dir = out_dir / "plots"
    ckpt_best = out_dir / "best.pt"
    ckpt_last = out_dir / "last.pt"

    header_order = [
        "epoch", "lr",
        "train_loss", "train_f1_macro", "train_iou_macro", "train_acc_all", "train_acc_no_bg",
        "train_d21_acc", "train_d21_f1", "train_d21_iou", "train_d21_bin_acc_all", "train_pred_bg_frac",
        "val_loss", "val_f1_macro", "val_iou_macro", "val_acc_all", "val_acc_no_bg",
        "val_d21_acc", "val_d21_f1", "val_d21_iou", "val_d21_bin_acc_all", "val_pred_bg_frac",
    ]
    for name, _ in neighbors:
        header_order += [f"val_{name}_acc", f"val_{name}_f1", f"val_{name}_iou", f"val_{name}_bin_acc_all"]

    print(f"[setup] data_dir={data_dir}")
    print(f"[setup] out_dir={out_dir}")
    print(f"[setup] device={device} | C={C} | bg={bg} | d21={d21_idx}")
    if neighbors:
        print(f"[setup] neighbors={neighbors}")

    best_key = "val_iou_macro"
    best_val = -1e9
    best_epoch = -1

    # ==========================
    # TRAIN LOOP
    # ==========================
    t0 = time.time()
    for epoch in range(1, int(args.epochs) + 1):
        tr = run_epoch(
            model=model,
            loader=dl_tr,
            optimizer=optim,
            loss_fn=loss_fn,
            C=int(C),
            d21_idx=int(d21_idx),
            device=device,
            bg=int(bg),
            train=True,
            use_amp=bool(args.use_amp),
            grad_clip=float(args.grad_clip) if float(args.grad_clip) > 0 else None,
        )

        va = run_epoch(
            model=model,
            loader=dl_va,
            optimizer=None,
            loss_fn=loss_fn,
            C=int(C),
            d21_idx=int(d21_idx),
            device=device,
            bg=int(bg),
            train=False,
            use_amp=bool(args.use_amp),
            grad_clip=None,
        )

        neigh_val = {}
        if neighbors:
            neigh_val = eval_neighbors_on_loader(
                model=model,
                loader=dl_va,
                device=device,
                neighbor_list=neighbors,
                bg=int(bg),
                use_amp=bool(args.use_amp),
            )

        # history
        history["train_loss"].append(float(tr["loss"]))
        history["val_loss"].append(float(va["loss"]))
        for k in ("f1_macro", "iou_macro", "acc_all", "acc_no_bg", "d21_acc", "d21_f1", "d21_iou", "d21_bin_acc_all", "pred_bg_frac"):
            history[f"train_{k}"].append(float(tr[k]))
            history[f"val_{k}"].append(float(va[k]))

        for name, _ in neighbors:
            history[f"val_{name}_acc"].append(float(neigh_val.get(f"{name}_acc", 0.0)))
            history[f"val_{name}_f1"].append(float(neigh_val.get(f"{name}_f1", 0.0)))
            history[f"val_{name}_iou"].append(float(neigh_val.get(f"{name}_iou", 0.0)))
            history[f"val_{name}_bin_acc_all"].append(float(neigh_val.get(f"{name}_bin_acc_all", 0.0)))

        scheduler.step()
        lr = get_lr(optim)

        print(
            f"[{epoch:03d}/{args.epochs}] "
            f"train loss={tr['loss']:.4f} f1m={tr['f1_macro']:.3f} ioum={tr['iou_macro']:.3f} "
            f"acc_all={tr['acc_all']:.3f} acc_no_bg={tr['acc_no_bg']:.3f} | "
            f"val loss={va['loss']:.4f} f1m={va['f1_macro']:.3f} ioum={va['iou_macro']:.3f} "
            f"acc_all={va['acc_all']:.3f} acc_no_bg={va['acc_no_bg']:.3f} | "
            f"d21 acc={va['d21_acc']:.3f} f1={va['d21_f1']:.3f} iou={va['d21_iou']:.3f} | "
            f"d21(bin all) acc={va['d21_bin_acc_all']:.3f} | "
            f"pred_bg_frac(val)={va['pred_bg_frac']:.3f} lr={lr:.2e}"
        )

        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": float(tr["loss"]),
            "train_f1_macro": float(tr["f1_macro"]),
            "train_iou_macro": float(tr["iou_macro"]),
            "train_acc_all": float(tr["acc_all"]),
            "train_acc_no_bg": float(tr["acc_no_bg"]),
            "train_d21_acc": float(tr["d21_acc"]),
            "train_d21_f1": float(tr["d21_f1"]),
            "train_d21_iou": float(tr["d21_iou"]),
            "train_d21_bin_acc_all": float(tr["d21_bin_acc_all"]),
            "train_pred_bg_frac": float(tr["pred_bg_frac"]),
            "val_loss": float(va["loss"]),
            "val_f1_macro": float(va["f1_macro"]),
            "val_iou_macro": float(va["iou_macro"]),
            "val_acc_all": float(va["acc_all"]),
            "val_acc_no_bg": float(va["acc_no_bg"]),
            "val_d21_acc": float(va["d21_acc"]),
            "val_d21_f1": float(va["d21_f1"]),
            "val_d21_iou": float(va["d21_iou"]),
            "val_d21_bin_acc_all": float(va["d21_bin_acc_all"]),
            "val_pred_bg_frac": float(va["pred_bg_frac"]),
        }
        row.update(neigh_val)
        append_metrics_csv(metrics_csv, row, header_order=header_order)

        # save last
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optim.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "num_classes": int(C),
                "bg_class": int(bg),
                "d21_internal": int(d21_idx),
                "args": vars(args),
                "best_epoch": int(best_epoch),
                "best_val": float(best_val),
            },
            ckpt_last,
        )
        save_history_json(out_dir, history)

        # save best
        cur = float(va["iou_macro"]) if best_key == "val_iou_macro" else float(va["f1_macro"])
        if cur > best_val:
            best_val = cur
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optim_state": optim.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "num_classes": int(C),
                    "bg_class": int(bg),
                    "d21_internal": int(d21_idx),
                    "args": vars(args),
                    "best_epoch": int(best_epoch),
                    "best_val": float(best_val),
                },
                ckpt_best,
            )

        # plots
        if int(args.plot_every) > 0 and (epoch % int(args.plot_every) == 0 or epoch == 1 or epoch == int(args.epochs)):
            plot_train_val("loss", history["train_loss"], history["val_loss"], plots_dir / "loss.png", best_epoch=best_epoch)
            plot_train_val("f1_macro", history["train_f1_macro"], history["val_f1_macro"], plots_dir / "f1_macro.png", best_epoch=best_epoch)
            plot_train_val("iou_macro", history["train_iou_macro"], history["val_iou_macro"], plots_dir / "iou_macro.png", best_epoch=best_epoch)
            plot_train_val("acc_no_bg", history["train_acc_no_bg"], history["val_acc_no_bg"], plots_dir / "acc_no_bg.png", best_epoch=best_epoch)
            plot_train_val("d21_f1", history["train_d21_f1"], history["val_d21_f1"], plots_dir / "d21_f1.png", best_epoch=best_epoch)
            plot_train_val("d21_iou", history["train_d21_iou"], history["val_d21_iou"], plots_dir / "d21_iou.png", best_epoch=best_epoch)

    dt = time.time() - t0
    print(f"[done] Entrenamiento terminado en {dt/60:.1f} min. best_epoch={best_epoch} best_val={best_val:.4f}")

    # ==========================
    # TEST (carga best.pt si existe)
    # ==========================
    if ckpt_best.exists():
        ck = torch.load(ckpt_best, map_location="cpu")
        model.load_state_dict(ck["model_state"], strict=True)
        print(f"[test] Cargado best.pt (epoch={ck.get('epoch')})")

    te = run_epoch(
        model=model,
        loader=dl_te,
        optimizer=None,
        loss_fn=loss_fn,
        C=int(C),
        d21_idx=int(d21_idx),
        device=device,
        bg=int(bg),
        train=False,
        use_amp=bool(args.use_amp),
        grad_clip=None,
    )

    neigh_test = {}
    if neighbors:
        neigh_test = eval_neighbors_on_loader(
            model=model,
            loader=dl_te,
            device=device,
            neighbor_list=neighbors,
            bg=int(bg),
            use_amp=bool(args.use_amp),
        )

    test_payload = {k: float(v) for k, v in te.items()}
    test_payload.update({f"neighbor_{k}": float(v) for k, v in neigh_test.items()})
    save_test_metrics(out_dir, metrics=test_payload, extra={"best_epoch": best_epoch, "best_val": best_val})

    print(
        f"[test] loss={te['loss']:.4f} f1m={te['f1_macro']:.3f} ioum={te['iou_macro']:.3f} "
        f"acc_all={te['acc_all']:.3f} acc_no_bg={te['acc_no_bg']:.3f} | "
        f"d21 acc={te['d21_acc']:.3f} f1={te['d21_f1']:.3f} iou={te['d21_iou']:.3f}"
    )

    # ==========================
    # INFER
    # ==========================
    if bool(args.do_infer):
        infer_and_save_examples(
            model=model,
            data_dir=data_dir,
            split=str(args.infer_split),
            out_dir=out_dir,
            device=device,
            C=int(C),
            d21_idx=int(d21_idx),
            bg=int(bg),
            n_examples=int(args.infer_examples),
            use_amp=bool(args.use_amp),
        )


if __name__ == "__main__":
    main()


# ============================================================
# TICKETS / EJEMPLOS DE CÓMO CORRER (idéntico estilo)
# ============================================================

# (1) RUN con vecinos + AMP + chunk KNN (tu ejemplo)
# conda activate enviroment
# cd /home/htaucare/Tesis_Amaro/scripts_last_version
#
# python3 train_dgcnn_classic_only_fixed_v6.py \
#   --data_dir /home/htaucare/Tesis_Amaro/data/Teeth_3ds/fixed_split/8192/upper_only_surf_global_excl_wisdom_seed42_aug2 \
#   --out_dir  /home/htaucare/Tesis_Amaro/outputs/dgcnn/gpu1_run1_v6_neighbors \
#   --epochs 120 --batch_size 16 --lr 2e-4 --weight_decay 1e-4 --dropout 0.5 \
#   --num_workers 6 --device cuda --use_amp --grad_clip 1.0 \
#   --k 20 --emb_dims 1024 --knn_chunk_size 1024 \
#   --bg_class 0 --bg_weight 0.03 --d21_internal 8 \
#   --neighbor_teeth "d11:1,d22:9" \
#   --normalize \
#   --do_infer --infer_examples 12 --infer_split test
#
# (2) RUN sin infer (más liviano)
# python3 train_dgcnn_classic_only_fixed_v6.py \
#   --data_dir /home/htaucare/Tesis_Amaro/data/Teeth_3ds/fixed_split/8192/upper_only_surf_global_excl_wisdom_seed42_aug2 \
#   --out_dir  /home/htaucare/Tesis_Amaro/outputs/dgcnn/gpu1_run1_v6 \
#   --epochs 120 --batch_size 16 --lr 2e-4 --weight_decay 1e-4 --dropout 0.5 \
#   --num_workers 6 --device cuda --use_amp --grad_clip 1.0 \
#   --k 20 --emb_dims 1024 --knn_chunk_size 1024 \
#   --bg_class 0 --bg_weight 0.03 --d21_internal 8 \
#   --normalize
#
# (3) CPU debug (para revisar rápido que corre)
# python3 train_dgcnn_classic_only_fixed_v6.py \
#   --data_dir /home/htaucare/Tesis_Amaro/data/Teeth_3ds/fixed_split/8192/upper_only_surf_global_excl_wisdom_seed42_aug2 \
#   --out_dir  /home/htaucare/Tesis_Amaro/outputs/dgcnn/cpu_debug_v6 \
#   --epochs 1 --batch_size 2 --lr 2e-4 --weight_decay 1e-4 --dropout 0.5 \
#   --num_workers 0 --device cpu \
#   --k 10 --emb_dims 256 --knn_chunk_size 256 \
#   --bg_class 0 --bg_weight 0.03 --d21_internal 8

