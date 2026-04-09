#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_dgcnn_only_v1.py

DGCNN (EdgeConv) – Segmentación multiclase dental 3D
VERSIÓN SOLO ENTRENAMIENTO / VALIDACIÓN / TEST / CHECKPOINTS

Objetivo de este script:
- Entrenar modelo DGCNN para segmentación dental 3D
- Guardar:
    - run_meta.json
    - run.log / errors.log
    - metrics_epoch.csv
    - history.json
    - history_epoch.jsonl
    - best.pt / last.pt
    - test_metrics.json
    - plots/*.png
- NO hace inferencia visual aquí
- La inferencia se hará en un script aparte

Supuestos del dataset:
- data_dir contiene:
    X_train.npz, Y_train.npz
    X_val.npz,   Y_val.npz
    X_test.npz,  Y_test.npz
- X: [B, N, 3]
- Y: [B, N]
- BG = clase 0 por defecto
"""

import os
import csv
import json
import time
import math
import argparse
import random
import logging
from logging import Logger
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
# UTILIDADES GENERALES
# ============================================================

def seed_all(seed: int = 42):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_seed(seed: int = 42):
    seed_all(seed)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def fmt_hms(seconds: float) -> str:
    seconds = float(seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:d}h {m:02d}m {s:05.2f}s"


def save_json(obj: Any, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def safe_write_json(path: Path, obj: Any):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def append_jsonl(path: Path, row: Dict[str, Any]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_metrics_csv(csv_path: Path, row: Dict[str, Any], header_order: Optional[List[str]] = None):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    exists = csv_path.exists()
    header = header_order if header_order is not None else list(row.keys())

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in header})


# ============================================================
# LOGGING
# ============================================================

def setup_logging(out_dir: Path, name: str = "dgcnn_train_only_v1") -> Logger:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_log = out_dir / "run.log"
    err_log = out_dir / "errors.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(run_log, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    eh = logging.FileHandler(err_log, encoding="utf-8")
    eh.setLevel(logging.ERROR)
    eh.setFormatter(fmt)
    logger.addHandler(eh)

    logger.info(f"[log] run.log={run_log}")
    logger.info(f"[log] errors.log={err_log}")
    return logger


# ============================================================
# NORMALIZACIÓN
# ============================================================

def normalize_unit_sphere(xyz: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    xyz: [N,3]
    - centra por media
    - escala por radio máximo
    """
    center = xyz.mean(dim=0, keepdim=True)
    xyz = xyz - center
    scale = torch.norm(xyz, dim=1).max().clamp_min(eps)
    return xyz / scale


# ============================================================
# DATASET NPZ
# ============================================================

class DentalNPZDataset(Dataset):
    """
    Espera:
      X_*.npz con key "X": [M,N,3]
      Y_*.npz con key "Y": [M,N]
    """
    def __init__(self, X_path: Path, Y_path: Path, normalize: bool = True):
        self.X = np.load(X_path)["X"]
        self.Y = np.load(Y_path)["Y"]
        self.normalize = bool(normalize)

        assert self.X.shape[0] == self.Y.shape[0], f"M mismatch: {self.X.shape} vs {self.Y.shape}"
        assert self.X.shape[1] == self.Y.shape[1], f"N mismatch: {self.X.shape} vs {self.Y.shape}"
        assert self.X.shape[2] == 3, f"X debe ser [M,N,3], llegó {self.X.shape}"

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, idx):
        x = np.ascontiguousarray(self.X[int(idx)], dtype=np.float32)
        y = np.ascontiguousarray(self.Y[int(idx)], dtype=np.int64)

        xyz = torch.as_tensor(x, dtype=torch.float32)
        lab = torch.as_tensor(y, dtype=torch.int64)

        if self.normalize:
            xyz = normalize_unit_sphere(xyz)

        return xyz, lab


def make_loaders(
    data_dir: Path,
    batch_size: int,
    num_workers: int,
    normalize: bool = True,
):
    """
    Devuelve:
      dl_tr, dl_va, dl_te, ds_tr, ds_va, ds_te
    """
    data_dir = Path(data_dir)

    ds_tr = DentalNPZDataset(data_dir / "X_train.npz", data_dir / "Y_train.npz", normalize=normalize)
    ds_va = DentalNPZDataset(data_dir / "X_val.npz",   data_dir / "Y_val.npz",   normalize=normalize)
    ds_te = DentalNPZDataset(data_dir / "X_test.npz",  data_dir / "Y_test.npz",  normalize=normalize)

    common = dict(
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        pin_memory=True,
        persistent_workers=(int(num_workers) > 0),
        drop_last=False,
    )
    if int(num_workers) > 0:
        common["prefetch_factor"] = 2

    dl_tr = DataLoader(ds_tr, shuffle=True,  **common)
    dl_va = DataLoader(ds_va, shuffle=False, **common)
    dl_te = DataLoader(ds_te, shuffle=False, **common)

    return dl_tr, dl_va, dl_te, ds_tr, ds_va, ds_te


# ============================================================
# LABEL MAP / NUM CLASSES
# ============================================================

def load_label_map(data_dir: Path) -> Optional[dict]:
    p = Path(data_dir) / "label_map.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def infer_num_classes(data_dir: Path, label_map: Optional[dict]) -> int:
    # 1) desde label_map
    if isinstance(label_map, dict) and len(label_map) > 0:
        try:
            mx = max(int(v) for v in label_map.values())
            return int(mx + 1)
        except Exception:
            pass

    # 2) fallback desde Y_*.npz
    maxy = -1
    for split in ("train", "val", "test"):
        yp = Path(data_dir) / f"Y_{split}.npz"
        if yp.exists():
            y = np.load(yp)["Y"]
            maxy = max(maxy, int(y.max()))

    if maxy < 0:
        raise RuntimeError("No se pudo inferir num_classes")
    return int(maxy + 1)


# ============================================================
# PARSE DE VECINOS
# ============================================================

def parse_neighbors(spec: str) -> List[Tuple[str, int]]:
    """
    --neighbor_teeth "d11:1,d22:9"
    -> [("d11",1), ("d22",9)]
    """
    spec = (spec or "").strip()
    if not spec:
        return []

    out: List[Tuple[str, int]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Formato inválido en neighbor_teeth: '{part}' (usa nombre:idx)")
        name, idx = part.split(":", 1)
        out.append((name.strip(), int(idx)))
    return out


# ============================================================
# AUTocast helper
# ============================================================

def get_autocast_ctx(device: torch.device, use_amp: bool):
    if bool(use_amp) and device.type == "cuda":
        return torch.amp.autocast("cuda", enabled=True)
    return torch.amp.autocast("cpu", enabled=False)

# ============================================================
# KNN CHUNK-SAFE + GRAPH FEATURE
# ============================================================

def pairwise_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Distancia cuadrada entre puntos:
      x: [B, N, D]
      y: [B, M, D]
    retorna:
      dist: [B, N, M]
    """
    xx = (x ** 2).sum(dim=-1, keepdim=True)                 # [B,N,1]
    yy = (y ** 2).sum(dim=-1, keepdim=True).transpose(1, 2) # [B,1,M]
    xy = torch.bmm(x, y.transpose(1, 2))                    # [B,N,M]
    dist = xx + yy - 2.0 * xy
    return dist


@torch.no_grad()
def knn_indices_chunked(
    x: torch.Tensor,
    k: int,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """
    KNN robusto por chunks para evitar picos de memoria.

    x:
      [B, N, D]

    retorna:
      idx: [B, N, k]

    Importante:
    - excluye self-neighbor poniendo +inf en la diagonal de cada chunk
    - mantiene forma consistente [B,N,k]
    """
    B, N, D = x.shape
    k = int(k)
    chunk_size = int(chunk_size)

    if k <= 0:
        raise ValueError("k debe ser > 0")
    if chunk_size <= 0:
        chunk_size = N

    device = x.device
    idx_out = torch.empty((B, N, k), device=device, dtype=torch.long)

    for s in range(0, N, chunk_size):
        e = min(N, s + chunk_size)
        q = x[:, s:e, :]                     # [B,Q,D]
        dist = pairwise_dist(q, x)           # [B,Q,N]

        # excluir self-neighbor dentro del rango del chunk
        qn = e - s
        ar = torch.arange(qn, device=device)
        dist[:, ar, s + ar] = float("inf")

        _, idx = dist.topk(k=k, dim=-1, largest=False, sorted=False)  # [B,Q,k]
        idx_out[:, s:e, :] = idx

    return idx_out


def get_graph_feature(
    x: torch.Tensor,
    k: int,
    idx: Optional[torch.Tensor] = None,
    knn_chunk_size: int = 1024,
) -> torch.Tensor:
    """
    Construye feature para EdgeConv.

    x:
      [B, C, N]

    idx:
      [B, N, k] opcional

    retorna:
      feature: [B, 2C, N, k]
      con concat( x_j - x_i , x_i )
    """
    B, C, N = x.shape
    k = int(k)

    xt = x.transpose(1, 2).contiguous()  # [B,N,C]

    if idx is None:
        idx = knn_indices_chunked(xt, k=k, chunk_size=int(knn_chunk_size))  # [B,N,k]

    device = x.device
    idx_base = torch.arange(0, B, device=device).view(-1, 1, 1) * N
    idx = idx + idx_base
    idx = idx.reshape(-1)

    neigh = xt.reshape(B * N, C)[idx, :].view(B, N, k, C)         # [B,N,k,C]
    center = xt.view(B, N, 1, C).expand(-1, -1, k, -1)            # [B,N,k,C]

    edge = torch.cat((neigh - center, center), dim=3)             # [B,N,k,2C]
    edge = edge.permute(0, 3, 1, 2).contiguous()                  # [B,2C,N,k]
    return edge


# ============================================================
# BLOQUES EDGE CONV
# ============================================================

class EdgeConvBlock(nn.Module):
    """
    EdgeConv block:
      - graph feature [B,2Cin,N,k]
      - conv2d 1x1
      - BN
      - ReLU
      - max over neighbors
    """
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2 * int(c_in), int(c_out), kernel_size=1, bias=False),
            nn.BatchNorm2d(int(c_out)),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, k: int, knn_chunk_size: int) -> torch.Tensor:
        feat = get_graph_feature(
            x=x,
            k=int(k),
            idx=None,
            knn_chunk_size=int(knn_chunk_size),
        )                                   # [B,2Cin,N,k]
        feat = self.conv(feat)              # [B,Cout,N,k]
        x = feat.max(dim=-1)[0]             # [B,Cout,N]
        return x


# ============================================================
# MODELO DGCNN SEGMENTACIÓN
# ============================================================

class DGCNNSeg(nn.Module):
    """
    DGCNN para segmentación por punto.

    Flujo:
      xyz [B,N,3]
        -> transpose [B,3,N]
        -> EdgeConv x4
        -> concat local features
        -> embedding global
        -> concat local + global
        -> head per-point
        -> logits [B,N,C]
    """
    def __init__(
        self,
        num_classes: int,
        k: int = 20,
        emb_dims: int = 1024,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.k = int(k)
        self.emb_dims = int(emb_dims)

        # Input: [B,3,N]
        self.ec1 = EdgeConvBlock(c_in=3,   c_out=64)
        self.ec2 = EdgeConvBlock(c_in=64,  c_out=64)
        self.ec3 = EdgeConvBlock(c_in=64,  c_out=128)
        self.ec4 = EdgeConvBlock(c_in=128, c_out=256)

        # concat local: 64 + 64 + 128 + 256 = 512
        self.conv_global = nn.Sequential(
            nn.Conv1d(512, self.emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.emb_dims),
            nn.ReLU(inplace=True),
        )

        # head de segmentación
        self.conv1 = nn.Sequential(
            nn.Conv1d(512 + self.emb_dims, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.dp1 = nn.Dropout(float(dropout))

        self.conv2 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.dp2 = nn.Dropout(float(dropout))

        self.classifier = nn.Conv1d(256, int(num_classes), kernel_size=1, bias=True)

    def forward(self, xyz: torch.Tensor, knn_chunk_size: int = 1024) -> torch.Tensor:
        """
        xyz:
          [B,N,3]

        retorna:
          logits [B,N,C]
        """
        B, N, _ = xyz.shape
        x = xyz.transpose(1, 2).contiguous()   # [B,3,N]

        x1 = self.ec1(x, k=self.k, knn_chunk_size=int(knn_chunk_size))   # [B,64,N]
        x2 = self.ec2(x1, k=self.k, knn_chunk_size=int(knn_chunk_size))  # [B,64,N]
        x3 = self.ec3(x2, k=self.k, knn_chunk_size=int(knn_chunk_size))  # [B,128,N]
        x4 = self.ec4(x3, k=self.k, knn_chunk_size=int(knn_chunk_size))  # [B,256,N]

        x_cat = torch.cat((x1, x2, x3, x4), dim=1)                       # [B,512,N]

        g = self.conv_global(x_cat)                                      # [B,emb,N]
        g = torch.max(g, dim=2, keepdim=True)[0]                         # [B,emb,1]
        g = g.repeat(1, 1, N)                                            # [B,emb,N]

        feat = torch.cat((x_cat, g), dim=1)                              # [B,512+emb,N]
        feat = self.conv1(feat)
        feat = self.dp1(feat)
        feat = self.conv2(feat)
        feat = self.dp2(feat)

        logits = self.classifier(feat).transpose(1, 2).contiguous()      # [B,N,C]
        return logits


# ============================================================
# HELPERS DE LOSS / LR / CHECKPOINTS
# ============================================================

def make_weighted_ce(
    num_classes: int,
    bg_class: int,
    bg_weight: float,
    device: torch.device,
) -> nn.Module:
    """
    CrossEntropy con BG incluido en la loss,
    pero con menor peso si se quiere evitar colapso al background.
    """
    w = torch.ones(int(num_classes), dtype=torch.float32, device=device)
    if 0 <= int(bg_class) < int(num_classes):
        w[int(bg_class)] = float(bg_weight)
    return nn.CrossEntropyLoss(weight=w)


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    if optimizer is None:
        return 0.0
    if len(optimizer.param_groups) == 0:
        return 0.0
    return float(optimizer.param_groups[0].get("lr", 0.0))


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    args,
    best_epoch: int,
    best_val: float,
    num_classes: int,
    bg_class: int,
    d21_idx: int,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "args": vars(args) if hasattr(args, "__dict__") else args,
        "best_epoch": int(best_epoch),
        "best_val": float(best_val),
        "num_classes": int(num_classes),
        "bg_class": int(bg_class),
        "d21_internal": int(d21_idx),
    }, path)


def load_checkpoint(path: Path, model: nn.Module, map_location="cpu"):
    path = Path(path)
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state"], strict=True)
    return ckpt

# ============================================================
# MÉTRICAS MULTICLASE (MACRO SIN BG)
# ============================================================

@torch.no_grad()
def compute_multiclass_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    C: int,
    bg: int,
):
    """
    logits:
      [B,N,C]

    target:
      [B,N]

    retorna:
      acc_all,
      acc_no_bg,
      prec_macro,
      rec_macro,
      f1_macro,
      iou_macro,
      pred_bg_frac
    """
    pred = logits.argmax(dim=-1)      # [B,N]

    pred_f = pred.reshape(-1)
    tgt_f  = target.reshape(-1)

    total = tgt_f.numel()
    correct = (pred_f == tgt_f).sum().item()
    acc_all = float(correct) / float(total + 1e-9)

    mask_no_bg = (tgt_f != int(bg))
    if mask_no_bg.any():
        correct_no_bg = (pred_f[mask_no_bg] == tgt_f[mask_no_bg]).sum().item()
        acc_no_bg = float(correct_no_bg) / float(mask_no_bg.sum().item() + 1e-9)
    else:
        acc_no_bg = 0.0

    eps = 1e-9
    prec_list: List[float] = []
    rec_list: List[float] = []
    f1_list: List[float] = []
    iou_list: List[float] = []

    for c in range(int(C)):
        if c == int(bg):
            continue

        pred_c = (pred_f == c)
        tgt_c  = (tgt_f  == c)

        tp = (pred_c & tgt_c).sum().item()
        fp = (pred_c & (~tgt_c)).sum().item()
        fn = ((~pred_c) & tgt_c).sum().item()

        prec = tp / (tp + fp + eps)
        rec  = tp / (tp + fn + eps)
        f1   = 2.0 * prec * rec / (prec + rec + eps)
        iou  = tp / (tp + fp + fn + eps)

        prec_list.append(float(prec))
        rec_list.append(float(rec))
        f1_list.append(float(f1))
        iou_list.append(float(iou))

    if len(f1_list) > 0:
        prec_macro = float(np.mean(prec_list))
        rec_macro  = float(np.mean(rec_list))
        f1_macro   = float(np.mean(f1_list))
        iou_macro  = float(np.mean(iou_list))
    else:
        prec_macro = 0.0
        rec_macro  = 0.0
        f1_macro   = 0.0
        iou_macro  = 0.0

    pred_bg_frac = float((pred_f == int(bg)).sum().item()) / float(total + 1e-9)

    return (
        float(acc_all),
        float(acc_no_bg),
        float(prec_macro),
        float(rec_macro),
        float(f1_macro),
        float(iou_macro),
        float(pred_bg_frac),
    )


# ============================================================
# MÉTRICA BINARIA DEL DIENTE 21
# ============================================================

@torch.no_grad()
def compute_d21_binary_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    d21_idx: int,
):
    """
    Trata la clase d21 como positiva y todo lo demás como negativo.

    retorna:
      d21_acc        -> accuracy sobre positivos reales
      d21_f1
      d21_iou
      d21_bin_acc_all -> accuracy binaria sobre todo el espacio
    """
    pred = logits.argmax(dim=-1)

    pred_bin = (pred == int(d21_idx))
    tgt_bin  = (target == int(d21_idx))

    eps = 1e-9

    tp = (pred_bin & tgt_bin).sum().item()
    fp = (pred_bin & (~tgt_bin)).sum().item()
    fn = ((~pred_bin) & tgt_bin).sum().item()
    tn = ((~pred_bin) & (~tgt_bin)).sum().item()

    pos = tgt_bin.sum().item()
    if pos > 0:
        d21_acc = float(tp) / float(pos + eps)
    else:
        d21_acc = 0.0

    prec = tp / (tp + fp + eps)
    rec  = tp / (tp + fn + eps)
    d21_f1  = 2.0 * prec * rec / (prec + rec + eps)
    d21_iou = tp / (tp + fp + fn + eps)

    d21_bin_acc_all = float(tp + tn) / float(tp + tn + fp + fn + eps)

    return (
        float(d21_acc),
        float(d21_f1),
        float(d21_iou),
        float(d21_bin_acc_all),
    )


# ============================================================
# MÉTRICAS BINARIAS PARA VECINOS ARBITRARIOS
# ============================================================

@torch.no_grad()
def eval_neighbors_on_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    neighbors: List[Tuple[str, int]],
    C: int,
    bg: int,
    knn_chunk_size: int,
    use_amp: bool,
):
    """
    Evalúa una lista arbitraria de dientes vecinos.

    neighbors:
      [("d11",1), ("d22",9), ...]

    retorna:
      {
        "d11": {"acc":..., "f1":..., "iou":..., "bin_acc_all":...},
        "d22": {...},
      }
    """
    if not neighbors:
        return {}

    model.eval()
    eps = 1e-9

    stats = {}
    for name, _ in neighbors:
        stats[name] = dict(tp=0, fp=0, fn=0, tn=0)

    for xyz, y in loader:
        xyz = xyz.to(device, non_blocking=True)
        y   = y.to(device, non_blocking=True)

        with get_autocast_ctx(device, bool(use_amp)):
            logits = model(xyz, knn_chunk_size=int(knn_chunk_size))

        pred = logits.argmax(dim=-1)

        for name, idx in neighbors:
            pred_bin = (pred == int(idx))
            tgt_bin  = (y    == int(idx))

            tp = (pred_bin & tgt_bin).sum().item()
            fp = (pred_bin & (~tgt_bin)).sum().item()
            fn = ((~pred_bin) & tgt_bin).sum().item()
            tn = ((~pred_bin) & (~tgt_bin)).sum().item()

            stats[name]["tp"] += tp
            stats[name]["fp"] += fp
            stats[name]["fn"] += fn
            stats[name]["tn"] += tn

    out = {}
    for name, _ in neighbors:
        tp = stats[name]["tp"]
        fp = stats[name]["fp"]
        fn = stats[name]["fn"]
        tn = stats[name]["tn"]

        prec = tp / (tp + fp + eps)
        rec  = tp / (tp + fn + eps)
        f1   = 2.0 * prec * rec / (prec + rec + eps)
        iou  = tp / (tp + fp + fn + eps)
        bin_acc_all = (tp + tn) / (tp + tn + fp + fn + eps)

        pos = tp + fn
        acc_cls = tp / (pos + eps) if pos > 0 else 0.0

        out[name] = dict(
            acc=float(acc_cls),
            f1=float(f1),
            iou=float(iou),
            bin_acc_all=float(bin_acc_all),
        )

    return out


# ============================================================
# FORWARD AUXILIAR DE LOSS
# ============================================================

def _loss_forward(
    model: nn.Module,
    xyz: torch.Tensor,
    y: torch.Tensor,
    loss_fn: nn.Module,
    C: int,
    knn_chunk_size: int,
    device: torch.device,
    use_amp: bool,
):
    with get_autocast_ctx(device, bool(use_amp)):
        logits = model(xyz, knn_chunk_size=int(knn_chunk_size))   # [B,N,C]
        loss = loss_fn(logits.reshape(-1, int(C)), y.reshape(-1))
    return logits, loss


# ============================================================
# EVAL ONE EPOCH
# ============================================================

@torch.no_grad()
def eval_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    *,
    device: torch.device,
    C: int,
    bg: int,
    d21_idx: int,
    knn_chunk_size: int,
) -> Dict[str, float]:
    """
    Eval estándar sobre un loader completo.
    """
    model.eval()

    loss_sum = 0.0

    acc_all_s = 0.0
    acc_no_bg_s = 0.0
    prec_s = 0.0
    rec_s = 0.0
    f1_s = 0.0
    iou_s = 0.0
    pred_bg_frac_s = 0.0

    d21_acc_s = 0.0
    d21_f1_s = 0.0
    d21_iou_s = 0.0
    d21_bin_acc_all_s = 0.0

    nb = 0

    for xyz, y in loader:
        xyz = xyz.to(device, non_blocking=True)
        y   = y.to(device, non_blocking=True)

        logits, loss = _loss_forward(
            model=model,
            xyz=xyz,
            y=y,
            loss_fn=loss_fn,
            C=int(C),
            knn_chunk_size=int(knn_chunk_size),
            device=device,
            use_amp=False,
        )

        acc_all, acc_no_bg, prec_macro, rec_macro, f1_macro, iou_macro, pred_bg_frac = \
            compute_multiclass_metrics(
                logits=logits,
                target=y,
                C=int(C),
                bg=int(bg),
            )

        d21_acc, d21_f1, d21_iou, d21_bin_acc_all = compute_d21_binary_metrics(
            logits=logits,
            target=y,
            d21_idx=int(d21_idx),
        )

        loss_sum += float(loss.item())

        acc_all_s += float(acc_all)
        acc_no_bg_s += float(acc_no_bg)
        prec_s += float(prec_macro)
        rec_s  += float(rec_macro)
        f1_s   += float(f1_macro)
        iou_s  += float(iou_macro)
        pred_bg_frac_s += float(pred_bg_frac)

        d21_acc_s += float(d21_acc)
        d21_f1_s  += float(d21_f1)
        d21_iou_s += float(d21_iou)
        d21_bin_acc_all_s += float(d21_bin_acc_all)

        nb += 1

    nb = max(1, nb)
    return dict(
        loss=float(loss_sum / nb),
        acc_all=float(acc_all_s / nb),
        acc_no_bg=float(acc_no_bg_s / nb),
        prec_macro=float(prec_s / nb),
        rec_macro=float(rec_s / nb),
        f1_macro=float(f1_s / nb),
        iou_macro=float(iou_s / nb),
        pred_bg_frac=float(pred_bg_frac_s / nb),
        d21_acc=float(d21_acc_s / nb),
        d21_f1=float(d21_f1_s / nb),
        d21_iou=float(d21_iou_s / nb),
        d21_bin_acc_all=float(d21_bin_acc_all_s / nb),
    )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    *,
    device: torch.device,
    C: int,
    bg: int,
    d21_idx: int,
    knn_chunk_size: int,
    use_amp: bool,
    grad_clip: float,
) -> Dict[str, float]:
    """
    Train de 1 epoch.

    Filosofía:
    - forward/backward/step con model.train()
    - métricas con segundo forward en model.eval()
      para que train vs val sea comparable
    """
    model.train(True)

    scaler = train_one_epoch.scaler  # type: ignore
    if bool(use_amp) and device.type == "cuda" and scaler is None:
        scaler = torch.amp.GradScaler("cuda")
        train_one_epoch.scaler = scaler  # type: ignore

    loss_sum = 0.0

    acc_all_s = 0.0
    acc_no_bg_s = 0.0
    prec_s = 0.0
    rec_s = 0.0
    f1_s = 0.0
    iou_s = 0.0
    pred_bg_frac_s = 0.0

    d21_acc_s = 0.0
    d21_f1_s = 0.0
    d21_iou_s = 0.0
    d21_bin_acc_all_s = 0.0

    nb = 0

    for xyz, y in loader:
        xyz = xyz.to(device, non_blocking=True)
        y   = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # ---------------------------
        # forward/loss para update
        # ---------------------------
        with get_autocast_ctx(device, bool(use_amp)):
            logits_train = model(xyz, knn_chunk_size=int(knn_chunk_size))
            loss = loss_fn(logits_train.reshape(-1, int(C)), y.reshape(-1))

        # ---------------------------
        # backward / step
        # ---------------------------
        if bool(use_amp) and device.type == "cuda":
            assert scaler is not None
            scaler.scale(loss).backward()

            if float(grad_clip) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if float(grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()

        # ---------------------------
        # métricas comparables
        # segundo forward en eval()
        # ---------------------------
        model.eval()
        with torch.no_grad():
            logits_eval = model(xyz, knn_chunk_size=int(knn_chunk_size))

        acc_all, acc_no_bg, prec_macro, rec_macro, f1_macro, iou_macro, pred_bg_frac = \
            compute_multiclass_metrics(
                logits=logits_eval,
                target=y,
                C=int(C),
                bg=int(bg),
            )

        d21_acc, d21_f1, d21_iou, d21_bin_acc_all = compute_d21_binary_metrics(
            logits=logits_eval,
            target=y,
            d21_idx=int(d21_idx),
        )

        loss_sum += float(loss.item())

        acc_all_s += float(acc_all)
        acc_no_bg_s += float(acc_no_bg)
        prec_s += float(prec_macro)
        rec_s  += float(rec_macro)
        f1_s   += float(f1_macro)
        iou_s  += float(iou_macro)
        pred_bg_frac_s += float(pred_bg_frac)

        d21_acc_s += float(d21_acc)
        d21_f1_s  += float(d21_f1)
        d21_iou_s += float(d21_iou)
        d21_bin_acc_all_s += float(d21_bin_acc_all)

        nb += 1
        model.train(True)

    nb = max(1, nb)
    return dict(
        loss=float(loss_sum / nb),
        acc_all=float(acc_all_s / nb),
        acc_no_bg=float(acc_no_bg_s / nb),
        prec_macro=float(prec_s / nb),
        rec_macro=float(rec_s / nb),
        f1_macro=float(f1_s / nb),
        iou_macro=float(iou_s / nb),
        pred_bg_frac=float(pred_bg_frac_s / nb),
        d21_acc=float(d21_acc_s / nb),
        d21_f1=float(d21_f1_s / nb),
        d21_iou=float(d21_iou_s / nb),
        d21_bin_acc_all=float(d21_bin_acc_all_s / nb),
    )


train_one_epoch.scaler = None  # type: ignore


# ============================================================
# PLOTS TRAIN VS VAL
# ============================================================

def plot_train_val_curves(history: dict, out_dir: Path):
    """
    Genera plots Train vs Val sin línea de test.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def plot_one(key: str, ylabel: str):
        tr = history["train"].get(key, [])
        va = history["val"].get(key, [])

        if not tr or not va:
            return

        plt.figure()
        plt.plot(tr, label="train")
        plt.plot(va, label="val")
        plt.xlabel("epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{key}.png", dpi=200)
        plt.close()

    # principales
    plot_one("loss", "loss")
    plot_one("f1_macro", "f1_macro")
    plot_one("iou_macro", "iou_macro")
    plot_one("acc_all", "acc_all")
    plot_one("acc_no_bg", "acc_no_bg")
    plot_one("pred_bg_frac", "pred_bg_frac")

    # d21
    plot_one("d21_acc", "d21_acc")
    plot_one("d21_f1", "d21_f1")
    plot_one("d21_iou", "d21_iou")
    plot_one("d21_bin_acc_all", "d21_bin_acc_all")

    # vecinos
    neighbors_hist = history.get("neighbors_val", {})
    for name, sub in neighbors_hist.items():
        for met, vals in sub.items():
            if not vals:
                continue
            plt.figure()
            plt.plot(vals, label=f"val_{name}_{met}")
            plt.xlabel("epoch")
            plt.ylabel(f"{name}_{met}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / f"{name}_{met}.png", dpi=200)
            plt.close()

# ============================================================
# TRAIN LOOP
# ============================================================

def train_loop(
    model: nn.Module,
    dl_tr: DataLoader,
    dl_va: DataLoader,
    dl_te: DataLoader,
    *,
    device: torch.device,
    out_dir: Path,
    C: int,
    bg: int,
    d21_idx: int,
    epochs: int,
    optimizer,
    scheduler,
    loss_fn,
    use_amp: bool,
    grad_clip: float,
    knn_chunk_size: int,
    plot_every: int,
    neighbors: List[Tuple[str, int]],
):

    out_dir = Path(out_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"

    history = {
        "train": {},
        "val": {},
        "neighbors_val": {},
    }

    metrics_keys = [
        "loss",
        "acc_all",
        "acc_no_bg",
        "prec_macro",
        "rec_macro",
        "f1_macro",
        "iou_macro",
        "pred_bg_frac",
        "d21_acc",
        "d21_f1",
        "d21_iou",
        "d21_bin_acc_all",
    ]

    for split in ["train", "val"]:
        history[split] = {k: [] for k in metrics_keys}

    for name, _ in neighbors:
        history["neighbors_val"][name] = {
            "acc": [],
            "f1": [],
            "iou": [],
            "bin_acc_all": [],
        }

    csv_path = out_dir / "metrics_epoch.csv"
    hist_json_path = out_dir / "history.json"
    hist_jsonl_path = out_dir / "history_epoch.jsonl"

    best_val = -1.0
    best_epoch = -1

    for epoch in range(1, int(epochs) + 1):

        t0 = time.time()

        # -------------------------
        # TRAIN
        # -------------------------
        tr = train_one_epoch(
            model=model,
            loader=dl_tr,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            C=C,
            bg=bg,
            d21_idx=d21_idx,
            knn_chunk_size=knn_chunk_size,
            use_amp=use_amp,
            grad_clip=grad_clip,
        )

        # -------------------------
        # VAL
        # -------------------------
        va = eval_one_epoch(
            model=model,
            loader=dl_va,
            loss_fn=loss_fn,
            device=device,
            C=C,
            bg=bg,
            d21_idx=d21_idx,
            knn_chunk_size=knn_chunk_size,
        )

        # -------------------------
        # neighbors val
        # -------------------------
        nb = eval_neighbors_on_loader(
            model=model,
            loader=dl_va,
            device=device,
            neighbors=neighbors,
            C=C,
            bg=bg,
            knn_chunk_size=knn_chunk_size,
            use_amp=False,
        )

        for name in nb:
            for k in nb[name]:
                history["neighbors_val"][name][k].append(nb[name][k])

        # -------------------------
        # history
        # -------------------------
        for k in metrics_keys:
            history["train"][k].append(tr[k])
            history["val"][k].append(va[k])

        lr_now = get_lr(optimizer)
        sec = time.time() - t0

        append_jsonl(hist_jsonl_path, {
            "epoch": epoch,
            "train": tr,
            "val": va,
            "neighbors": nb,
        })

        append_metrics_csv(csv_path, {
            "epoch": epoch,
            "train_loss": tr["loss"],
            "val_loss": va["loss"],
            "train_f1": tr["f1_macro"],
            "val_f1": va["f1_macro"],
            "train_iou": tr["iou_macro"],
            "val_iou": va["iou_macro"],
            "lr": lr_now,
            "sec": sec,
        })

        if scheduler is not None:
            scheduler.step()

        # -------------------------
        # checkpoint
        # -------------------------
        save_checkpoint(
            last_path,
            epoch,
            model,
            optimizer,
            scheduler,
            args=None,
            best_epoch=best_epoch,
            best_val=best_val,
            num_classes=C,
            bg_class=bg,
            d21_idx=d21_idx,
        )

        if va["f1_macro"] > best_val:
            best_val = float(va["f1_macro"])
            best_epoch = epoch

            save_checkpoint(
                best_path,
                epoch,
                model,
                optimizer,
                scheduler,
                args=None,
                best_epoch=best_epoch,
                best_val=best_val,
                num_classes=C,
                bg_class=bg,
                d21_idx=d21_idx,
            )

        # -------------------------
        # print
        # -------------------------
        print(
            f"[{epoch:03d}/{epochs}] "
            f"train loss={tr['loss']:.4f} f1m={tr['f1_macro']:.3f} "
            f"| val loss={va['loss']:.4f} f1m={va['f1_macro']:.3f} "
            f"| d21 f1={va['d21_f1']:.3f} "
            f"| lr={lr_now:.2e}"
        )

        if plot_every > 0 and epoch % plot_every == 0:
            plot_train_val_curves(history, plots_dir)

    save_json(history, hist_json_path)

    print(f"[done] best_epoch={best_epoch} best_val={best_val:.4f}")

    return best_epoch, best_val


# ============================================================
# TEST LOOP
# ============================================================

def test_loop(
    model,
    dl_te,
    device,
    out_dir,
    C,
    bg,
    d21_idx,
    loss_fn,
    knn_chunk_size,
):

    ckpt = load_checkpoint(out_dir / "best.pt", model, map_location=device)

    te = eval_one_epoch(
        model=model,
        loader=dl_te,
        loss_fn=loss_fn,
        device=device,
        C=C,
        bg=bg,
        d21_idx=d21_idx,
        knn_chunk_size=knn_chunk_size,
    )

    print(
        f"[test] loss={te['loss']:.4f} "
        f"f1m={te['f1_macro']:.3f} "
        f"iou={te['iou_macro']:.3f}"
    )

    save_json(te, out_dir / "test_metrics.json")


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)

    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch_size", type=int, default=16)

    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)

    p.add_argument("--dropout", type=float, default=0.5)

    p.add_argument("--num_workers", type=int, default=6)

    p.add_argument("--device", default="cuda")

    p.add_argument("--use_amp", action="store_true")

    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--k", type=int, default=20)
    p.add_argument("--emb_dims", type=int, default=1024)

    p.add_argument("--knn_chunk_size", type=int, default=1024)

    p.add_argument("--bg_class", type=int, default=0)
    p.add_argument("--bg_weight", type=float, default=0.03)

    p.add_argument("--d21_internal", type=int, default=8)

    p.add_argument("--neighbor_teeth", default="")

    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--normalize", action="store_true")

    p.add_argument("--plot_every", type=int, default=10)

    return p.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_json(vars(args), out_dir / "run_meta.json")

    logger = setup_logging(out_dir)

    dl_tr, dl_va, dl_te, ds_tr, ds_va, ds_te = make_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalize=args.normalize,
    )

    label_map = load_label_map(args.data_dir)
    C = infer_num_classes(args.data_dir, label_map)

    bg = int(args.bg_class)
    d21_idx = int(args.d21_internal)

    neighbors = parse_neighbors(args.neighbor_teeth)

    logger.info(f"[setup] device={device} | C={C} | bg={bg} | d21={d21_idx}")

    model = DGCNNSeg(
        num_classes=C,
        k=args.k,
        emb_dims=args.emb_dims,
        dropout=args.dropout,
    ).to(device)

    loss_fn = make_weighted_ce(
        num_classes=C,
        bg_class=bg,
        bg_weight=args.bg_weight,
        device=device,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    best_epoch, best_val = train_loop(
        model=model,
        dl_tr=dl_tr,
        dl_va=dl_va,
        dl_te=dl_te,
        device=device,
        out_dir=out_dir,
        C=C,
        bg=bg,
        d21_idx=d21_idx,
        epochs=args.epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        use_amp=args.use_amp,
        grad_clip=args.grad_clip,
        knn_chunk_size=args.knn_chunk_size,
        plot_every=args.plot_every,
        neighbors=neighbors,
    )

    test_loop(
        model,
        dl_te,
        device,
        out_dir,
        C,
        bg,
        d21_idx,
        loss_fn,
        args.knn_chunk_size,
    )


if __name__ == "__main__":
    main()