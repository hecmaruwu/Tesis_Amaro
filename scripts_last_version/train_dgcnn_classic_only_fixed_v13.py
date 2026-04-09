#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_dgcnn_classic_only_fixed_v13.py

DGCNN (EdgeConv) – Segmentación multiclase dental 3D

✅ MISMA TRAZABILIDAD/OUTPUTS "paper-like" que tu stack final:
   - run_meta.json
   - run.log / errors.log
   - metrics_epoch.csv
   - history.json + history_epoch.jsonl
   - best.pt / last.pt
   - test_metrics.json
   - plots/*.png  (Train vs Val; SIN línea de test)  ✅ (v13: incluye d21 + pred_bg_frac + neighbors)
   - inference/:
       - inference_manifest.csv
       - inference_all/*.png
       - inference_errors/*.png
       - inference_d21/*.png

✅ Vecinos configurables (lista arbitraria) y se imprimen en consola en cada epoch.
✅ Inferencia trazable usando index_{split}.csv (busca en data_dir y ancestros hasta Teeth_3ds).
✅ Dataset flat NPZ: X_train.npz/Y_train.npz (y val/test).
✅ Normalización opcional (unit sphere).
✅ Estabilidad: bg_weight, weight_decay, grad_clip, CosineAnnealingLR, AMP torch.amp
✅ KNN chunk-safe (evita mismatch 1024 vs 8192)

NEW v13 (lo que faltaba vs v12):
✅ PLOTS igual estilo PointNet (train vs val + línea vertical best_epoch):
   - loss, f1m, ioum, acc_all, acc_no_bg
   - d21_acc, d21_f1, d21_iou
   - pred_bg_frac (train/val)  (útil para colapso a bg)
   - neighbors: {name}_acc/{name}_f1/{name}_iou (val-only) con mismo estilo de plots
✅ Inferencia: visual estilo PointNet (axis_off, view_init, depthshade=False, colores robustos float32)

NOTA IMPORTANTE:
- NO se eliminó ninguna funcionalidad de v12; solo se agregan los plots y pequeños helpers
  para que el estilo/outputs queden idénticos a PointNet.

Ejemplo (GPU 1):
CUDA_VISIBLE_DEVICES=1 python3 train_dgcnn_classic_only_fixed_v13.py \
  --data_dir /home/htaucare/Tesis_Amaro/data/Teeth_3ds/fixed_split/8192/upper_only_surf_global_excl_wisdom_seed42_aug2 \
  --out_dir  /home/htaucare/Tesis_Amaro/outputs/dgcnn/gpu1_run1_v13_neighbors \
  --epochs 120 --batch_size 16 --lr 2e-4 --weight_decay 1e-4 --dropout 0.5 \
  --num_workers 6 --device cuda --use_amp --grad_clip 1.0 \
  --k 20 --emb_dims 1024 --knn_chunk_size 1024 \
  --bg_class 0 --bg_weight 0.03 --d21_internal 8 \
  --neighbor_teeth "d11:1,d22:9" \
  --seed 42 --normalize --plot_every 10 \
  --do_infer --infer_examples 12 --infer_split test
"""

import os
import io
import re
import csv
import json
import time
import math
import argparse
import random
import traceback
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

import logging
from logging import Logger

def save_json(obj, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ============================================================
# Utils: seed / time / json / jsonl / csv
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

def safe_write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)

def append_jsonl(path: Path, row: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

def append_metrics_csv(csv_path: Path, row: Dict[str, Any], header_order: Optional[List[str]] = None):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    header = header_order if header_order is not None else list(row.keys())
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in header})

def _fmt_hms(seconds: float) -> str:
    seconds = float(seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:d}h {m:02d}m {s:05.2f}s"


# ============================================================
# Logging robusto: run.log + errors.log + console
# ============================================================

def setup_logging(out_dir: Path, name: str = "dgcnn_v13") -> Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_run = out_dir / "run.log"
    log_err = out_dir / "errors.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_run, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    eh = logging.FileHandler(log_err, encoding="utf-8")
    eh.setLevel(logging.ERROR)
    eh.setFormatter(fmt)
    logger.addHandler(eh)

    logger.info(f"[log] run.log={log_run}")
    logger.info(f"[log] errors.log={log_err}")
    return logger


# ============================================================
# Normalización (unit sphere)
# ============================================================

def normalize_unit_sphere(xyz: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    xyz: [N,3]
    center = mean
    scale = max ||x|| (después de centrar)
    """
    center = xyz.mean(dim=0, keepdim=True)
    xyz = xyz - center
    scale = torch.norm(xyz, dim=1).max().clamp_min(eps)
    return xyz / scale


# ============================================================
# Dataset NPZ (flat)
# ============================================================

class NPZDataset(Dataset):
    """
    Espera:
      X_*.npz con key "X": [M,N,3]
      Y_*.npz con key "Y": [M,N]
    """
    def __init__(self, X_path: Path, Y_path: Path, normalize: bool = True):
        self.X = np.load(X_path)["X"]
        self.Y = np.load(Y_path)["Y"]
        self.normalize = bool(normalize)

        assert self.X.shape[0] == self.Y.shape[0], "M mismatch X vs Y"
        assert self.X.shape[1] == self.Y.shape[1], "N mismatch X vs Y"
        assert self.X.shape[2] == 3, "X debe ser [M,N,3]"

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, idx):
        x = np.ascontiguousarray(self.X[int(idx)], dtype=np.float32)
        y = np.ascontiguousarray(self.Y[int(idx)], dtype=np.int64)

        xyz = torch.as_tensor(x, dtype=torch.float32)  # [N,3]
        lab = torch.as_tensor(y, dtype=torch.int64)    # [N]

        if self.normalize:
            xyz = normalize_unit_sphere(xyz)

        return xyz, lab


def make_loaders(data_dir: Path, bs: int, nw: int, normalize: bool = True):
    """
    Devuelve EXACTAMENTE 3 loaders: train/val/test.
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

    dl_tr = DataLoader(ds_tr, shuffle=True,  **common)
    dl_va = DataLoader(ds_va, shuffle=False, **common)
    dl_te = DataLoader(ds_te, shuffle=False, **common)
    return dl_tr, dl_va, dl_te


# ============================================================
# Label map / num classes
# ============================================================

def load_label_map(data_dir: Path) -> Optional[dict]:
    p = data_dir / "label_map.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def infer_num_classes(data_dir: Path, label_map: Optional[dict]) -> int:
    # 1) label_map
    if isinstance(label_map, dict) and len(label_map) > 0:
        try:
            mx = max(int(v) for v in label_map.values())
            return int(mx + 1)
        except Exception:
            pass

    # 2) fallback: max over Y_*.npz
    maxy = -1
    for split in ("train", "val", "test"):
        yp = data_dir / f"Y_{split}.npz"
        if yp.exists():
            y = np.load(yp)["Y"]
            maxy = max(maxy, int(y.max()))
    if maxy < 0:
        raise RuntimeError("No se pudo inferir num_classes (no Y_*.npz?)")
    return int(maxy + 1)


# ============================================================
# Neighbors parsing
# ============================================================

def parse_neighbors(s: str) -> List[Tuple[str, int]]:
    """
    --neighbor_teeth "d11:1,d22:9"
    -> [("d11",1), ("d22",9)]
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
            raise ValueError(f"Formato inválido neighbor_teeth: '{part}' (usa nombre:idx)")
        name, idx = part.split(":", 1)
        out.append((name.strip(), int(idx)))
    return out


# ============================================================
# Trazabilidad: index_{split}.csv discovery + read
# ============================================================

def discover_index_csv(data_dir: Path, split: str) -> Optional[Path]:
    """
    Busca index_{split}.csv en:
      1) data_dir/index_{split}.csv
      2) ancestros (hasta 12 niveles)
      3) si encuentra Teeth_3ds/Teeth3DS en el camino, rglob (primer match)
    """
    data_dir = Path(data_dir).resolve()
    target = f"index_{split}.csv"

    p = data_dir / target
    if p.exists():
        return p

    parents = [data_dir] + list(data_dir.parents)

    for par in parents[:13]:
        cand = par / target
        if cand.exists():
            return cand

    for par in parents:
        if par.name.lower() in ("teeth_3ds", "teeth3ds"):
            try:
                for found in par.rglob(target):
                    return found
            except Exception:
                return None
            return None

    return None


def read_index_csv(p: Optional[Path]) -> Optional[Dict[int, Dict[str, str]]]:
    """
    Retorna:
      { row_idx : {col: value, ...} }
    Si hay columna idx/index/row/i la usa; si no, usa contador.
    """
    if p is None:
        return None
    p = Path(p)
    if not p.exists():
        return None

    try:
        with p.open("r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            rows = list(r)
        if not rows:
            return None

        fieldnames = list(rows[0].keys())
        idx_col = None
        for c in ("idx", "index", "row", "i", "row_i"):
            if c in fieldnames:
                idx_col = c
                break

        out: Dict[int, Dict[str, str]] = {}
        for i, row in enumerate(rows):
            rid = i
            if idx_col is not None:
                try:
                    rid = int(float(str(row.get(idx_col, "")).strip()))
                except Exception:
                    rid = i
            out[int(rid)] = {k: ("" if row.get(k) is None else str(row.get(k))) for k in row.keys()}
        return out
    except Exception:
        return None


def pick_trace_label(d: Dict[str, str]) -> str:
    for k in ("sample_name", "patient", "patient_id", "scan_id", "id", "path", "relpath", "upper_path"):
        if k in d and str(d[k]).strip():
            return str(d[k]).strip()
    for k, v in d.items():
        if str(v).strip():
            return f"{k}={v}"
    return "sample"


# ============================================================
# AMP ctx (single source of truth)
# ============================================================

def get_autocast_ctx(device: torch.device, use_amp: bool):
    if bool(use_amp) and device.type == "cuda":
        return torch.amp.autocast("cuda", enabled=True)
    return torch.amp.autocast("cpu", enabled=False)


# ============================================================
# Plot helpers (mismo estilo PointNet)
# ============================================================

def class_colors(C: int):
    cmap = plt.colormaps.get_cmap("tab20")
    C = max(int(C), 2)
    return [cmap(i / max(C - 1, 1)) for i in range(C)]

def to_np(a) -> np.ndarray:
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    a = np.asarray(a)
    if not a.flags["C_CONTIGUOUS"]:
        a = np.ascontiguousarray(a)
    return a

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


# ============================================================
# LR helper
# ============================================================

def get_lr(optim: torch.optim.Optimizer) -> float:
    if optim is None:
        return 0.0
    if len(optim.param_groups) == 0:
        return 0.0
    return float(optim.param_groups[0].get("lr", 0.0))


# ============================================================
# Checkpoints
# ============================================================

def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optim: torch.optim.Optimizer,
    scheduler,
    args,
    best_epoch: int,
    best_val: float,
    C: int,
    bg: int,
    d21_idx: int,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optim_state": optim.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "args": vars(args) if hasattr(args, "__dict__") else args,
        "best_epoch": int(best_epoch),
        "best_val": float(best_val),
        "num_classes": int(C),
        "bg": int(bg),
        "d21_idx": int(d21_idx),
    }, path)


def load_checkpoint(path: Path, model: nn.Module, map_location="cpu"):
    path = Path(path)
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state"], strict=True)
    return ckpt

# ============================================================
# KNN (chunk-safe) + EdgeConv + DGCNN Segmentation
# ============================================================

def pairwise_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Distancia cuadrada entre puntos:
      x: [B, N, D]
      y: [B, M, D]
    retorna: [B, N, M]
    """
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    xx = (x ** 2).sum(dim=-1, keepdim=True)          # [B,N,1]
    yy = (y ** 2).sum(dim=-1, keepdim=True).transpose(1, 2)  # [B,1,M]
    xy = torch.bmm(x, y.transpose(1, 2))             # [B,N,M]
    dist = xx + yy - 2.0 * xy
    return dist


@torch.no_grad()
def knn_indices_chunked(x: torch.Tensor, k: int, chunk_size: int = 1024) -> torch.Tensor:
    """
    KNN robusto para N=8192:
      x: [B, N, D]
    Retorna idx: [B, N, k] con vecinos (excluye self por estabilidad).
    """
    B, N, D = x.shape
    k = int(k)
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        chunk_size = N

    # Para excluir self: ponemos dist self = +inf (solo dentro del chunk correcto)
    device = x.device

    idx_out = torch.empty((B, N, k), device=device, dtype=torch.long)
    # procesamos queries en chunks para no explotar VRAM
    for s in range(0, N, chunk_size):
        e = min(N, s + chunk_size)
        xq = x[:, s:e, :]                # [B, Q, D]
        dist = pairwise_dist(xq, x)      # [B, Q, N]

        # excluir self cuando aplica (solo si Q corresponde a ese rango)
        # dist[b, q, (s+q)] = +inf
        q = e - s
        ar = torch.arange(q, device=device)
        dist[:, ar, s + ar] = float("inf")

        # topk menores (largest=False)
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
    Construye features EdgeConv:
      x: [B, C, N]
      idx (opcional): [B, N, k]
    retorna:
      feature: [B, 2C, N, k] con concat( (x_j - x_i), x_i )
    """
    B, C, N = x.shape
    k = int(k)

    xt = x.transpose(1, 2).contiguous()  # [B, N, C]
    if idx is None:
        # KNN en espacio de características (xt)
        idx = knn_indices_chunked(xt, k=k, chunk_size=int(knn_chunk_size))  # [B,N,k]

    # idx -> índices lineales por batch
    device = x.device
    idx_base = torch.arange(0, B, device=device).view(-1, 1, 1) * N  # [B,1,1]
    idx = idx + idx_base                                            # [B,N,k]
    idx = idx.reshape(-1)                                           # [B*N*k]

    # neighbors: [B, N, k, C]
    neigh = xt.reshape(B * N, C)[idx, :].view(B, N, k, C)
    # center: [B, N, 1, C] -> expand a k
    center = xt.view(B, N, 1, C).expand(-1, -1, k, -1)

    # EdgeConv: (x_j - x_i, x_i)
    edge = torch.cat((neigh - center, center), dim=3)  # [B,N,k,2C]
    edge = edge.permute(0, 3, 1, 2).contiguous()       # [B,2C,N,k]
    return edge


class EdgeConvBlock(nn.Module):
    """
    EdgeConv block (Wang et al. 2019):
      - construye graph feature: [B,2C_in,N,k]
      - 1x1 conv2d + BN + ReLU
      - max over neighbors -> [B,C_out,N]
    """
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2 * int(c_in), int(c_out), kernel_size=1, bias=False),
            nn.BatchNorm2d(int(c_out)),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, k: int, knn_chunk_size: int) -> torch.Tensor:
        # x: [B,C,N]
        feat = get_graph_feature(x, k=k, idx=None, knn_chunk_size=knn_chunk_size)  # [B,2C,N,k]
        feat = self.conv(feat)                                                     # [B,Cout,N,k]
        x = feat.max(dim=-1)[0]                                                    # [B,Cout,N]
        return x


class DGCNNSeg(nn.Module):
    """
    DGCNN para segmentación (paper-like):
      - 4 EdgeConv layers (64, 64, 128, 256)
      - concat local features
      - global pooling
      - head per-point
    """
    def __init__(self, num_classes: int, k: int = 20, emb_dims: int = 1024, dropout: float = 0.5):
        super().__init__()
        self.k = int(k)
        self.emb_dims = int(emb_dims)

        # Input x: [B,3,N]
        self.ec1 = EdgeConvBlock(c_in=3,   c_out=64)
        self.ec2 = EdgeConvBlock(c_in=64,  c_out=64)
        self.ec3 = EdgeConvBlock(c_in=64,  c_out=128)
        self.ec4 = EdgeConvBlock(c_in=128, c_out=256)

        # after concat: 64+64+128+256 = 512
        self.conv_global = nn.Sequential(
            nn.Conv1d(512, self.emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.emb_dims),
            nn.ReLU(inplace=True),
        )

        # seg head: concat local(512) + global(emb_dims) -> 512+emb_dims
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
        xyz: [B, N, 3]
        logits: [B, N, C]
        """
        B, N, _ = xyz.shape
        x = xyz.transpose(1, 2).contiguous()  # [B,3,N]

        x1 = self.ec1(x, k=self.k, knn_chunk_size=knn_chunk_size)    # [B,64,N]
        x2 = self.ec2(x1, k=self.k, knn_chunk_size=knn_chunk_size)   # [B,64,N]
        x3 = self.ec3(x2, k=self.k, knn_chunk_size=knn_chunk_size)   # [B,128,N]
        x4 = self.ec4(x3, k=self.k, knn_chunk_size=knn_chunk_size)   # [B,256,N]

        x_cat = torch.cat((x1, x2, x3, x4), dim=1)                   # [B,512,N]

        g = self.conv_global(x_cat)                                  # [B,emb_dims,N]
        g = torch.max(g, dim=2, keepdim=True)[0]                     # [B,emb_dims,1]
        g = g.repeat(1, 1, N)                                        # [B,emb_dims,N]

        feat = torch.cat((x_cat, g), dim=1)                          # [B,512+emb_dims,N]
        feat = self.conv1(feat)
        feat = self.dp1(feat)
        feat = self.conv2(feat)
        feat = self.dp2(feat)

        logits = self.classifier(feat).transpose(1, 2).contiguous()  # [B,N,C]
        return logits
    
# ============================================================
# MÉTRICAS (macro sin bg) + d21 binario + neighbors (binario)
# + PLOTS 3D (all/errors/d21) estilo PointNet v5
# ============================================================

@torch.no_grad()
def _acc_all(pred: torch.Tensor, gt: torch.Tensor) -> float:
    return float((pred == gt).float().mean().item())


@torch.no_grad()
def macro_metrics_no_bg(pred: torch.Tensor, gt: torch.Tensor, C: int, bg: int = 0) -> Tuple[float, float]:
    """
    Macro-F1 e IoU macro EXCLUYENDO BG (gt!=bg), promedio sobre clases 1..C-1.
    (Robusto: si una clase no aparece, se omite del macro.)
    """
    pred = pred.reshape(-1)
    gt = gt.reshape(-1)

    mask = (gt != int(bg))
    pred = pred[mask]
    gt = gt[mask]
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
    f1  = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    return float(acc), float(f1), float(iou)


@torch.no_grad()
def _tooth_metrics_binary(pred: torch.Tensor, gt: torch.Tensor, tooth_idx: int, bg: int = 0) -> Dict[str, float]:
    """
    Métricas binarias para un diente cualquiera (vecino):
      - acc/f1/iou excluyendo bg (principal)
      - bin_acc_all (incluye bg) (referencia)
    """
    acc, f1, iou = d21_metrics_binary(pred, gt, d21_idx=int(tooth_idx), bg=int(bg), include_bg=False)
    acc_all, _, _ = d21_metrics_binary(pred, gt, d21_idx=int(tooth_idx), bg=int(bg), include_bg=True)
    return {
        "acc": float(acc),
        "f1": float(f1),
        "iou": float(iou),
        "bin_acc_all": float(acc_all),
    }


def parse_neighbor_teeth(spec: Optional[str]) -> List[Tuple[str, int]]:
    """
    Parse de --neighbor_teeth con formato:
      "d11:1,d22:9,foo:3"
    Devuelve [(name, idx), ...] (orden estable, override por nombre).
    """
    if spec is None:
        return []
    s = str(spec).strip()
    if not s:
        return []
    seen: Dict[str, int] = {}
    tokens = [t.strip() for t in s.split(",") if t.strip()]
    for tok in tokens:
        if ":" not in tok:
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

    out: List[Tuple[str, int]] = []
    used = set()
    for tok in tokens:
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
    bg: int,
    knn_chunk_size: int,
) -> Dict[str, float]:
    """
    Eval promedio por batches: {d11_acc, d11_f1, d11_iou, d11_bin_acc_all, ...}
    OJO: esto es "promedio de batches" (igual que tu filosofía actual).
    """
    if not neighbor_list:
        return {}

    model.eval()
    sums: Dict[str, float] = {}
    for name, _ in neighbor_list:
        for k in ("acc", "f1", "iou", "bin_acc_all"):
            sums[f"{name}_{k}"] = 0.0

    nb = 0
    for xyz, y in loader:
        xyz = xyz.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(xyz, knn_chunk_size=knn_chunk_size)
        pred = logits.argmax(dim=-1)

        for name, idx in neighbor_list:
            m = _tooth_metrics_binary(pred, y, tooth_idx=int(idx), bg=int(bg))
            sums[f"{name}_acc"] += float(m["acc"])
            sums[f"{name}_f1"] += float(m["f1"])
            sums[f"{name}_iou"] += float(m["iou"])
            sums[f"{name}_bin_acc_all"] += float(m["bin_acc_all"])
        nb += 1

    nb = max(1, nb)
    return {k: v / nb for k, v in sums.items()}


@torch.no_grad()
def compute_metrics_from_logits(
    logits: torch.Tensor,
    y: torch.Tensor,
    C: int,
    d21_idx: int,
    bg: int
) -> Dict[str, float]:
    """
    Métricas base (idénticas a PointNet v5):
      acc_all, acc_no_bg, f1_macro, iou_macro,
      d21_acc, d21_f1, d21_iou,
      d21_bin_acc_all,
      pred_bg_frac
    """
    pred = logits.argmax(dim=-1)

    acc_all = _acc_all(pred, y)

    mask = (y != int(bg))
    if mask.any():
        acc_no_bg = float((pred[mask] == y[mask]).float().mean().item())
    else:
        acc_no_bg = 0.0

    f1m, ioum = macro_metrics_no_bg(pred, y, C=int(C), bg=int(bg))

    d21_acc, d21_f1, d21_iou = d21_metrics_binary(
        pred, y, d21_idx=int(d21_idx), bg=int(bg), include_bg=False
    )
    d21_bin_acc_all, _, _ = d21_metrics_binary(
        pred, y, d21_idx=int(d21_idx), bg=int(bg), include_bg=True
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


# ============================================================
# VISUALIZACIÓN 3D (idéntica filosofía PointNet v5)
# ============================================================

def _class_colors(C: int):
    cmap = plt.colormaps.get_cmap("tab20")
    C = max(int(C), 2)
    cols = [cmap(i / max(C - 1, 1)) for i in range(C)]
    return cols


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
    s: float = 1.0
):
    out_png.parent.mkdir(parents=True, exist_ok=True)

    xyz = _to_np(xyz).astype(np.float32, copy=False)
    y_gt = _to_np(y_gt).astype(np.int32, copy=False).reshape(-1)
    y_pr = _to_np(y_pr).astype(np.int32, copy=False).reshape(-1)

    cols = _class_colors(int(C))
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
# PLOTS Train vs Val (SIN línea de test; best_epoch vertical)
# ============================================================

def plot_train_val(
    name: str,
    y_tr: List[float],
    y_va: List[float],
    out_png: Path,
    best_epoch: Optional[int] = None
):
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

# ============================================================
# TRAIN / EVAL (v13)
# - AMP + grad_clip + CosineAnnealingLR
# - (IMPORTANT) Train metrics comparables: forward para métricas en eval()
#   (igual filosofía PointNet v5)
# - Neighbors: logging + print en consola + plots estilo PointNet v5
# ============================================================

@torch.no_grad()
def _mean_std(xs: List[float]) -> Tuple[float, float]:
    if xs is None or len(xs) == 0:
        return 0.0, 0.0
    a = np.asarray(xs, dtype=np.float64)
    return float(a.mean()), float(a.std(ddof=0))


def _maybe_autocast(device: torch.device, use_amp: bool):
    if bool(use_amp) and device.type == "cuda":
        return torch.amp.autocast("cuda", enabled=True)
    return torch.amp.autocast("cpu", enabled=False)


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
    grad_clip: Optional[float],
    knn_chunk_size: int,
    # v13: para guardar std por batch (opcional)
    collect_batch_stats: bool = False,
) -> Dict[str, float]:
    """
    Devuelve dict con:
      loss, acc_all, acc_no_bg, f1_macro, iou_macro,
      d21_acc, d21_f1, d21_iou, d21_bin_acc_all,
      pred_bg_frac
    + opcional *_std si collect_batch_stats=True
    """
    scaler = run_epoch.scaler  # type: ignore
    if bool(use_amp) and device.type == "cuda" and scaler is None:
        scaler = torch.amp.GradScaler("cuda")
        run_epoch.scaler = scaler  # type: ignore

    loss_sum = 0.0
    sums = {
        "acc_all": 0.0,
        "acc_no_bg": 0.0,
        "f1_macro": 0.0,
        "iou_macro": 0.0,
        "d21_acc": 0.0,
        "d21_f1": 0.0,
        "d21_iou": 0.0,
        "d21_bin_acc_all": 0.0,
        "pred_bg_frac": 0.0,
    }
    n_batches = 0

    batch_stats: Dict[str, List[float]] = {k: [] for k in (["loss"] + list(sums.keys()))}

    if not bool(train):
        model.eval()

    for xyz, y in loader:
        xyz = xyz.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if bool(train):
            assert optimizer is not None
            model.train(True)
            optimizer.zero_grad(set_to_none=True)

            # ---- forward (train) para UPDATE ----
            with _maybe_autocast(device, use_amp):
                logits_train = model(xyz, knn_chunk_size=int(knn_chunk_size))  # [B,N,C]
                loss = loss_fn(logits_train.reshape(-1, int(C)), y.reshape(-1))

            # ---- backward/step ----
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

            # ---- métricas: segundo forward en eval() (dropout OFF) ----
            model.eval()
            with torch.no_grad():
                logits_eval = model(xyz, knn_chunk_size=int(knn_chunk_size))
            mets = compute_metrics_from_logits(logits_eval, y, C=int(C), d21_idx=int(d21_idx), bg=int(bg))

        else:
            # ---- eval normal ----
            with torch.no_grad():
                with _maybe_autocast(device, use_amp=False):
                    logits = model(xyz, knn_chunk_size=int(knn_chunk_size))
                    loss = loss_fn(logits.reshape(-1, int(C)), y.reshape(-1))
                mets = compute_metrics_from_logits(logits, y, C=int(C), d21_idx=int(d21_idx), bg=int(bg))

        loss_sum += float(loss.item())
        for k in sums.keys():
            sums[k] += float(mets[k])
        n_batches += 1

        if bool(collect_batch_stats):
            batch_stats["loss"].append(float(loss.item()))
            for k in sums.keys():
                batch_stats[k].append(float(mets[k]))

    n = max(1, n_batches)
    out = {"loss": loss_sum / n}
    for k in sums.keys():
        out[k] = sums[k] / n

    if bool(collect_batch_stats):
        for k, arr in batch_stats.items():
            _, sd = _mean_std(arr)
            out[f"{k}_std"] = float(sd)

    return out


run_epoch.scaler = None  # type: ignore


# ============================================================
# TRAIN LOOP (v13): logging idéntico estilo PointNet v5
# - metrics_epoch.csv (train/val) + columnas neighbors al final
# - history.json para plots
# - prints consola con neighbors
# ============================================================

def train_loop_v13(
    model: nn.Module,
    dl_tr: DataLoader,
    dl_va: DataLoader,
    dl_te: DataLoader,
    *,
    device: torch.device,
    out_dir: Path,
    C: int,
    bg: int,
    bg_va: float,
    d21_int: int,
    epochs: int,
    opt: torch.optim.Optimizer,
    sched: torch.optim.lr_scheduler._LRScheduler,
    loss_fn: nn.Module,
    use_amp: bool,
    grad_clip: float,
    knn_chunk_size: int,
    # v13 neighbors
    neighbor_list: List[Tuple[str, int]],
    neighbor_eval_split: str,
    neighbor_every: int,
    # v13 plotting control
    plot_every: int,
) -> Tuple[int, float]:
    """
    Retorna (best_epoch, best_val_f1_macro).
    Produce:
      - best.pt / last.pt
      - metrics_epoch.csv
      - history.json
      - plots/*.png (solo Train vs Val + best_epoch vertical)
      - (neighbors) plots/{name}_{met}.png
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"

    # columnas dinámicas neighbors (al final)
    neighbor_cols: List[str] = []
    for name, _ in neighbor_list:
        neighbor_cols += [f"{name}_acc", f"{name}_f1", f"{name}_iou", f"{name}_bin_acc_all"]

    # CSV por epoch
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

    # HISTORY (para plots)
    history: Dict[str, List[float]] = {}
    def _mk(k: str): history[k] = []

    for k in (
        "loss", "acc_all", "acc_no_bg",
        "f1_macro", "iou_macro",
        "d21_acc", "d21_f1", "d21_iou",
        "d21_bin_acc_all",
        "pred_bg_frac",
    ):
        _mk(f"train_{k}")
        _mk(f"val_{k}")

    # neighbors history (val por defecto, igual que PointNet v5)
    for name, _ in neighbor_list:
        for met in ("acc", "f1", "iou", "bin_acc_all"):
            _mk(f"val_{name}_{met}")

    best_val_f1 = -1.0
    best_epoch = -1

    t0 = time.time()

    # loop
    for epoch in range(1, int(epochs) + 1):
        e0 = time.time()

        # ---- TRAIN (update + métricas comparables ya incluidas dentro de run_epoch) ----
        tr = run_epoch(
            model=model,
            loader=dl_tr,
            optimizer=opt,
            loss_fn=loss_fn,
            C=int(C),
            d21_idx=int(d21_int),
            device=device,
            bg=int(bg),
            train=True,
            use_amp=bool(use_amp),
            grad_clip=float(grad_clip),
            knn_chunk_size=int(knn_chunk_size),
            collect_batch_stats=False,
        )

        # ---- VAL ----
        va = run_epoch(
            model=model,
            loader=dl_va,
            optimizer=None,
            loss_fn=loss_fn,
            C=int(C),
            d21_idx=int(d21_int),
            device=device,
            bg=int(bg),
            train=False,
            use_amp=False,
            grad_clip=None,
            knn_chunk_size=int(knn_chunk_size),
            collect_batch_stats=False,
        )

        # ---- Neighbors (opcional, configurable) ----
        nb_vals: Dict[str, float] = {}
        do_neighbors = (len(neighbor_list) > 0) and (str(neighbor_eval_split).lower() != "none")
        do_neighbors = do_neighbors and (int(neighbor_every) > 0) and ((epoch % int(neighbor_every)) == 0)

        if do_neighbors:
            mode = str(neighbor_eval_split).lower()
            # durante training: por default val (test por epoch es caro; lo hacemos al final)
            if mode in ("val", "both"):
                nb_vals = eval_neighbors_on_loader(
                    model=model,
                    loader=dl_va,
                    device=device,
                    neighbor_list=neighbor_list,
                    bg=int(bg),
                    knn_chunk_size=int(knn_chunk_size),
                )

        # ---- scheduler ----
        sched.step()
        lr_now = float(opt.param_groups[0]["lr"])
        sec = float(time.time() - e0)

        # ---- history ----
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
                    hk = f"val_{name}_{met}"
                    key_csv = f"{name}_{met}"
                    history[hk].append(float(nb_vals.get(key_csv, 0.0)))

        # ---- CSV append ----
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            wcsv = csv.writer(f)

            row_train = [
                epoch, "train",
                tr["loss"],
                tr["acc_all"], tr["acc_no_bg"],
                tr["f1_macro"], tr["iou_macro"],
                tr["d21_acc"], tr["d21_f1"], tr["d21_iou"],
                tr["d21_bin_acc_all"],
                tr["pred_bg_frac"],
                lr_now,
                sec,
            ]
            row_val = [
                epoch, "val",
                va["loss"],
                va["acc_all"], va["acc_no_bg"],
                va["f1_macro"], va["iou_macro"],
                va["d21_acc"], va["d21_f1"], va["d21_iou"],
                va["d21_bin_acc_all"],
                va["pred_bg_frac"],
                lr_now,
                sec,
            ]

            if neighbor_cols:
                row_train += ["" for _ in neighbor_cols]  # no calculamos neighbors en train (costo)
                row_val += [float(nb_vals.get(col, 0.0)) for col in neighbor_cols]

            wcsv.writerow(row_train)
            wcsv.writerow(row_val)

        # ---- checkpoints ----
        torch.save({"model": model.state_dict(), "epoch": int(epoch)}, last_path)

        if float(va["f1_macro"]) > float(best_val_f1):
            best_val_f1 = float(va["f1_macro"])
            best_epoch = int(epoch)
            torch.save({"model": model.state_dict(), "epoch": int(epoch)}, best_path)

        # ---- warn colapso a BG ----
        if float(va["pred_bg_frac"]) > max(0.95, float(bg_va) + 0.12):
            print(f"[WARN] posible colapso a BG: val pred_bg_frac={va['pred_bg_frac']:.3f} (bg_gt≈{bg_va:.3f})")

        # ---- print consola (idéntico estilo PointNet v5) ----
        nb_str = ""
        if neighbor_list and do_neighbors:
            parts = []
            for name, _ in neighbor_list:
                parts.append(f"{name}_f1={nb_vals.get(f'{name}_f1', 0.0):.3f}")
            nb_str = " | nb(" + ",".join(parts) + ")"

        print(
            f"[{epoch:03d}/{int(epochs)}] "
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

        # ---- plots durante entrenamiento (opcional, cada plot_every) ----
        if int(plot_every) > 0 and (epoch % int(plot_every) == 0):
            plot_dir = out_dir / "plots"
            plot_dir.mkdir(exist_ok=True)

            # base plots
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

            # neighbors plots (val)
            for name, _ in neighbor_list:
                for met in ("acc", "f1", "iou", "bin_acc_all"):
                    key = f"val_{name}_{met}"
                    if key in history:
                        plot_train_val(
                            name=f"{name}_{met}",
                            y_tr=[0.0] * len(history[key]),  # no train para neighbors
                            y_va=history[key],
                            out_png=plot_dir / f"{name}_{met}.png",
                            best_epoch=best_epoch,
                        )

    # fin train
    save_json(history, out_dir / "history.json")

    total_sec = float(time.time() - t0)
    print(f"[done] Entrenamiento terminado en {_fmt_hms(total_sec)}. best_epoch={best_epoch} best_val={best_val_f1:.4f}")
    return int(best_epoch), float(best_val_f1)


# ============================================================
# TEST + PLOTS FINAL (solo Train vs Val, sin testline)
# (La inferencia visual + index discovery va en PARTE 5/5)
# ============================================================

def test_and_final_plots_v13(
    model: nn.Module,
    dl_te: DataLoader,
    *,
    device: torch.device,
    out_dir: Path,
    C: int,
    bg: int,
    d21_int: int,
    loss_fn: nn.Module,
    knn_chunk_size: int,
    best_epoch: int,
    neighbor_list: List[Tuple[str, int]],
    neighbor_eval_split: str,
):
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"

    # cargar best
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"[test] Cargado best.pt (epoch={ckpt.get('epoch','?')})")
    else:
        print("[test] best.pt no encontrado, usando last.pt si existe")
        if last_path.exists():
            ckpt = torch.load(last_path, map_location=device)
            model.load_state_dict(ckpt["model"])

    # eval test
    te = run_epoch(
        model=model,
        loader=dl_te,
        optimizer=None,
        loss_fn=loss_fn,
        C=int(C),
        d21_idx=int(d21_int),
        device=device,
        bg=int(bg),
        train=False,
        use_amp=False,
        grad_clip=None,
        knn_chunk_size=int(knn_chunk_size),
        collect_batch_stats=False,
    )

    print(
        f"[test] "
        f"loss={te['loss']:.4f} "
        f"f1m={te['f1_macro']:.3f} "
        f"ioum={te['iou_macro']:.3f} "
        f"acc_all={te['acc_all']:.3f} "
        f"acc_no_bg={te['acc_no_bg']:.3f} | "
        f"d21(cls) acc={te['d21_acc']:.3f} "
        f"f1={te['d21_f1']:.3f} "
        f"iou={te['d21_iou']:.3f} | "
        f"d21(bin all) acc={te['d21_bin_acc_all']:.3f} | "
        f"pred_bg_frac(test)={te['pred_bg_frac']:.3f}"
    )

    # neighbors en test (si corresponde)
    test_neighbors: Dict[str, float] = {}
    if neighbor_list:
        mode = str(neighbor_eval_split).lower()
        if mode in ("test", "both"):
            test_neighbors = eval_neighbors_on_loader(
                model=model,
                loader=dl_te,
                device=device,
                neighbor_list=neighbor_list,
                bg=int(bg),
                knn_chunk_size=int(knn_chunk_size),
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

    # guardar test_metrics.json
    out = dict(te)
    out["best_epoch"] = int(best_epoch)
    out["neighbor_metrics_test"] = test_neighbors
    save_json(out, out_dir / "test_metrics.json")

# ============================================================
# PARTE 5/5
# train_dgcnn_classic_only_fixed_v13.py
# - main() completo v13
# - run_meta.json
# - index_csv auto-discovery + inference_manifest.csv
# - inference/ con subcarpetas:
#     inference_all / inference_errors / inference_d21
# - PNGs 3D (all/errors/d21) MISMO estilo/libs que PointNet v5
# ============================================================

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
    Devuelve mapa: row_i (int) -> dict con keys:
      idx_global, sample_name, jaw, path, has_labels
    Header flexible (solo requiere row_i).
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
      3) Teeth_3ds/merged_*/index_{split}.csv (elige el más reciente por mtime)
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


def _ensure_infer_dirs(base: Path) -> Dict[str, Path]:
    d_all = base / "inference_all"
    d_err = base / "inference_errors"
    d_d21 = base / "inference_d21"
    d_all.mkdir(parents=True, exist_ok=True)
    d_err.mkdir(parents=True, exist_ok=True)
    d_d21.mkdir(parents=True, exist_ok=True)
    return {"all": d_all, "errors": d_err, "d21": d_d21}


@torch.no_grad()
def do_inference_v13(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    out_dir: Path,
    split: str,
    C: int,
    bg: int,
    d21_int: int,
    infer_examples: int,
    knn_chunk_size: int,
    data_dir: Path,
    index_csv: Optional[str],
):
    """
    Guarda:
      inference/inference_manifest.csv
      inference/inference_all/*.png
      inference/inference_errors/*.png
      inference/inference_d21/*.png
    (1 sample -> 3 png)
    """
    infer_dir = out_dir / "inference"
    infer_dir.mkdir(parents=True, exist_ok=True)
    dirs = _ensure_infer_dirs(infer_dir)

    # index map (trazabilidad)
    index_map = None
    index_path_used = ""
    if index_csv:
        p = Path(index_csv)
        index_map = _read_index_csv(p)
        index_path_used = str(p) if index_map else ""
    else:
        auto = _discover_index_csv(Path(data_dir), str(split))
        if auto:
            index_map = _read_index_csv(auto)
            index_path_used = str(auto) if index_map else ""

    # manifest
    manifest_path = infer_dir / "inference_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "infer_i", "split_row_i", "sample_name", "idx_global", "jaw", "path",
            "png_all", "png_errors", "png_d21",
            "index_csv_used",
        ])

        model.eval()
        done = 0
        split = str(split).strip().lower()

        for batch_idx, (xyz, y) in enumerate(loader):
            xyz = xyz.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(xyz, knn_chunk_size=int(knn_chunk_size))
            pred = logits.argmax(dim=-1)

            for b in range(int(xyz.shape[0])):
                if done >= int(infer_examples):
                    break

                xyz_np = xyz[b].detach().cpu().numpy()
                y_np = y[b].detach().cpu().numpy()
                pr_np = pred[b].detach().cpu().numpy()

                # row_i: en NPZDataset el orden es fijo (0..len-1)
                row_i = int(done)

                info = {"sample_name": "", "idx_global": "", "jaw": "", "path": ""}
                if index_map is not None and row_i in index_map:
                    info = index_map[row_i]

                tag = f"{split}_sample_{done:03d}"
                safe_name = _sanitize_tag(info.get("sample_name", ""))
                if safe_name:
                    tag = f"{tag}_{safe_name}"

                # nombres de archivos (misma idea PointNet v5)
                png_all_rel = f"inference_all/{tag}_all.png"
                png_err_rel = f"inference_errors/{tag}_errors.png"
                png_d21_rel = f"inference_d21/{tag}_d21.png"

                plot_pointcloud_all_classes(
                    xyz_np, y_np, pr_np,
                    dirs["all"] / f"{tag}_all.png",
                    C=int(C),
                    title=tag,
                    s=1.0
                )
                plot_errors(
                    xyz_np, y_np, pr_np,
                    dirs["errors"] / f"{tag}_errors.png",
                    bg=int(bg),
                    title=tag,
                    s=1.0
                )
                plot_d21_focus(
                    xyz_np, y_np, pr_np,
                    dirs["d21"] / f"{tag}_d21.png",
                    d21_idx=int(d21_int),
                    bg=int(bg),
                    title=tag,
                    s=1.2
                )

                w.writerow([
                    done,
                    row_i,
                    info.get("sample_name", ""),
                    info.get("idx_global", ""),
                    info.get("jaw", ""),
                    info.get("path", ""),
                    png_all_rel,
                    png_err_rel,
                    png_d21_rel,
                    index_path_used,
                ])

                done += 1

            if done >= int(infer_examples):
                break

    print(f"[infer] Guardado: {manifest_path}")
    print(f"[infer] index_csv_used: {index_path_used if index_path_used else '(none)'}")


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

    ap.add_argument("--use_amp", action="store_true")
    ap.add_argument("--grad_clip", type=float, default=1.0)

    # DGCNN
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--emb_dims", type=int, default=1024)
    ap.add_argument("--knn_chunk_size", type=int, default=1024)

    # classes
    ap.add_argument("--bg_class", type=int, default=0)
    ap.add_argument("--bg_weight", type=float, default=0.03)
    ap.add_argument("--d21_internal", type=int, required=True)

    # normalize
    ap.add_argument("--normalize", action="store_true")

    # neighbors
    ap.add_argument("--neighbor_teeth", type=str, default="",
                    help='Lista "name:idx" separada por coma. Ej: "d11:1,d22:9"')
    ap.add_argument("--neighbor_eval_split", type=str, default="val", choices=["val", "test", "both", "none"])
    ap.add_argument("--neighbor_every", type=int, default=1)

    # plots
    ap.add_argument("--plot_every", type=int, default=10,
                    help="Cada cuántos epochs re-genera plots (0 desactiva).")

    # inference
    ap.add_argument("--do_infer", action="store_true")
    ap.add_argument("--infer_examples", type=int, default=12)
    ap.add_argument("--infer_split", type=str, default="test", choices=["test", "val", "train"])
    ap.add_argument("--index_csv", type=str, default=None,
                    help="Opcional: fuerza index_{split}.csv específico para trazabilidad.")

    args = ap.parse_args()
    set_seed(int(args.seed))

    # device
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # sanity data + C
    Ytr = np.asarray(np.load(data_dir / "Y_train.npz")["Y"]).reshape(-1)
    Yva = np.asarray(np.load(data_dir / "Y_val.npz")["Y"]).reshape(-1)
    Yte = np.asarray(np.load(data_dir / "Y_test.npz")["Y"]).reshape(-1)

    bg = int(args.bg_class)
    bg_tr = float((Ytr == bg).mean())
    bg_va = float((Yva == bg).mean())
    bg_te = float((Yte == bg).mean())

    C = int(max(int(Ytr.max()), int(Yva.max()), int(Yte.max()))) + 1

    print(f"[SANITY] num_classes C = {C}")
    print(f"[SANITY] bg_frac train/val/test = {bg_tr:.4f} {bg_va:.4f} {bg_te:.4f}")
    print(f"[SANITY] baseline acc_all (always-bg) train/val/test = {bg_tr:.4f} {bg_va:.4f} {bg_te:.4f}")

    d21_int = int(args.d21_internal)
    if not (0 <= d21_int < C):
        raise ValueError(f"d21_internal fuera de rango: {d21_int} (C={C})")

    # loaders
    dl_tr, dl_va, dl_te, ds_te = make_loaders(
        data_dir=data_dir,
        bs=int(args.batch_size),
        nw=int(args.num_workers),
        normalize=bool(args.normalize),
    )

    # model
    model = DGCNNSeg(
        num_classes=int(C),
        k=int(args.k),
        emb_dims=int(args.emb_dims),
        dropout=float(args.dropout),
    ).to(device)

    # loss weights
    w = torch.ones(int(C), device=device, dtype=torch.float32)
    w[int(bg)] = float(args.bg_weight)
    loss_fn = nn.CrossEntropyLoss(weight=w)

    # opt + sched
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

    # neighbors parse
    neighbor_list = parse_neighbor_teeth(args.neighbor_teeth)
    if neighbor_list:
        bad = [(n, i) for (n, i) in neighbor_list if not (0 <= int(i) < C)]
        if bad:
            raise ValueError(f"neighbor_teeth fuera de rango (C={C}): {bad}")
        print(f"[NEIGHBORS] parsed: {neighbor_list}")
    else:
        print("[NEIGHBORS] none")

    # run meta
    run_meta = {
        "script_name": "train_dgcnn_classic_only_fixed_v13.py",
        "start_time": datetime.now().isoformat(timespec="seconds"),
        "data_dir": str(data_dir),
        "out_dir": str(out_dir),
        "device": str(device),
        "seed": int(args.seed),
        "num_classes": int(C),
        "bg_class": int(bg),
        "bg_weight": float(args.bg_weight),
        "d21_internal": int(d21_int),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "dropout": float(args.dropout),
        "grad_clip": float(args.grad_clip),
        "use_amp": bool(args.use_amp),
        "normalize_unit_sphere": bool(args.normalize),
        "k": int(args.k),
        "emb_dims": int(args.emb_dims),
        "knn_chunk_size": int(args.knn_chunk_size),
        "plot_every": int(args.plot_every),
        "do_infer": bool(args.do_infer),
        "infer_examples": int(args.infer_examples),
        "infer_split": str(args.infer_split),
        "index_csv": str(args.index_csv) if args.index_csv else "",
        "neighbor_teeth": str(args.neighbor_teeth),
        "neighbor_eval_split": str(args.neighbor_eval_split),
        "neighbor_every": int(args.neighbor_every),
        "neighbor_parsed": neighbor_list,
    }
    save_json(run_meta, out_dir / "run_meta.json")

    # TRAIN
    best_epoch, best_val = train_loop_v13(
        model=model,
        dl_tr=dl_tr,
        dl_va=dl_va,
        dl_te=dl_te,
        device=device,
        out_dir=out_dir,
        C=int(C),
        bg=int(bg),
        bg_va=float(bg_va),
        d21_int=int(d21_int),
        epochs=int(args.epochs),
        opt=opt,
        sched=sched,
        loss_fn=loss_fn,
        use_amp=bool(args.use_amp),
        grad_clip=float(args.grad_clip),
        knn_chunk_size=int(args.knn_chunk_size),
        neighbor_list=neighbor_list,
        neighbor_eval_split=str(args.neighbor_eval_split),
        neighbor_every=int(args.neighbor_every),
        plot_every=int(args.plot_every),
    )

    # TEST + final plots (usa history.json ya guardado por train_loop_v13)
    test_and_final_plots_v13(
        model=model,
        dl_te=dl_te,
        device=device,
        out_dir=out_dir,
        C=int(C),
        bg=int(bg),
        d21_int=int(d21_int),
        loss_fn=loss_fn,
        knn_chunk_size=int(args.knn_chunk_size),
        best_epoch=int(best_epoch),
        neighbor_list=neighbor_list,
        neighbor_eval_split=str(args.neighbor_eval_split),
    )

    # (FINAL) plots completos asegurados (por si plot_every=0 durante train)
    hist_path = out_dir / "history.json"
    if hist_path.exists():
        history = json.loads(hist_path.read_text(encoding="utf-8"))
        plot_dir = out_dir / "plots"
        plot_dir.mkdir(exist_ok=True)

        for k in (
            "loss", "acc_all", "acc_no_bg",
            "f1_macro", "iou_macro",
            "d21_acc", "d21_f1", "d21_iou",
            "d21_bin_acc_all",
            "pred_bg_frac",
        ):
            if f"train_{k}" in history and f"val_{k}" in history:
                plot_train_val(
                    name=k,
                    y_tr=history[f"train_{k}"],
                    y_va=history[f"val_{k}"],
                    out_png=plot_dir / f"{k}.png",
                    best_epoch=int(best_epoch),
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
                        best_epoch=int(best_epoch),
                    )

    # INFERENCE
    if bool(args.do_infer):
        infer_split = str(args.infer_split).lower()
        if infer_split == "test":
            loader = dl_te
        elif infer_split == "val":
            loader = dl_va
        else:
            loader = dl_tr

        do_inference_v13(
            model=model,
            loader=loader,
            device=device,
            out_dir=out_dir,
            split=infer_split,
            C=int(C),
            bg=int(bg),
            d21_int=int(d21_int),
            infer_examples=int(args.infer_examples),
            knn_chunk_size=int(args.knn_chunk_size),
            data_dir=data_dir,
            index_csv=args.index_csv,
        )


if __name__ == "__main__":
    main()

