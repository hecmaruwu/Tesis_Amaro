#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_dgcnn_classic_only_fixed_v10.py

DGCNN (EdgeConv) – Segmentación multiclase dental 3D
✅ MISMA TRAZABILIDAD/OUTPUTS "paper-like" que tu stack final:
   - run_meta.json
   - run.log / errors.log
   - metrics_epoch.csv
   - history.json + history_epoch.jsonl
   - best.pt / last.pt
   - test_metrics.json
   - plots/*.png  (Train vs Val; SIN línea de test)
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

Notas:
- bg incluido en la loss (NO ignore_index)
- bg excluido SOLO en métricas macro y métricas binarias de diente (por defecto)
"""

import os
import re
import io
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


# ============================================================
# LOGGING (run.log + errors.log) — estilo PointNet/DGCNN final
# ============================================================

class Tee:
    """Duplica stdout/stderr a archivo."""
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            try:
                f.write(data)
                f.flush()
            except Exception:
                pass

    def flush(self):
        for f in self.files:
            try:
                f.flush()
            except Exception:
                pass


def setup_logging(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    run_log = (out_dir / "run.log").open("a", encoding="utf-8")
    err_log = (out_dir / "errors.log").open("a", encoding="utf-8")

    # Redirigir stdout/stderr
    import sys
    sys.stdout = Tee(sys.__stdout__, run_log)     # type: ignore
    sys.stderr = Tee(sys.__stderr__, err_log)     # type: ignore

    print(f"[log] stdout -> {out_dir/'run.log'}")
    print(f"[log] stderr -> {out_dir/'errors.log'}")


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def save_json(p: Path, obj: Any):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(p: Path, obj: Any):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def append_metrics_csv(csv_path: Path, row: Dict[str, Any], header_order: Optional[List[str]] = None):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    header = header_order if header_order is not None else list(row.keys())
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in header})


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


def set_seed(seed: int = 42):
    # alias compat con tus scripts
    seed_all(seed)


# ============================================================
# NORMALIZACIÓN
# ============================================================

def normalize_unit_sphere(xyz: torch.Tensor, eps: float = 1e-9):
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
# DATASET NPZ (ROBUSTO)
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
        x = np.ascontiguousarray(self.X[idx], dtype=np.float32)
        y = np.ascontiguousarray(self.Y[idx], dtype=np.int64)

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
# LABEL MAP + NUM CLASSES
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
# NEIGHBORS
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
# TRAZABILIDAD: index_{split}.csv discovery + read
# ============================================================

def _discover_index_csv(data_dir: Path, split: str) -> Optional[Path]:
    """
    Busca index_{split}.csv en:
      1) data_dir/index_{split}.csv
      2) ancestros (hasta 12 niveles)
      3) si encuentra Teeth_3ds/Teeth3DS en el camino, hace rglob del archivo (primer match)
    """
    data_dir = Path(data_dir).resolve()
    target = f"index_{split}.csv"

    p = data_dir / target
    if p.exists():
        return p

    cur = data_dir
    parents = [cur] + list(cur.parents)
    # 2) ancestros directos
    for par in parents[:13]:
        cand = par / target
        if cand.exists():
            return cand

    # 3) si existe Teeth_3ds en el path, rglob
    for par in parents:
        if par.name.lower() in ("teeth_3ds", "teeth3ds"):
            try:
                for found in par.rglob(target):
                    return found
            except Exception:
                return None
            return None

    return None


def _read_index_csv(p: Optional[Path]) -> Optional[Dict[int, Dict[str, str]]]:
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
        for c in ("idx", "index", "row", "i"):
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


def _pick_trace_label(d: Dict[str, str]) -> str:
    for k in ("sample_name", "patient", "patient_id", "scan_id", "id", "path", "relpath", "upper_path"):
        if k in d and str(d[k]).strip():
            return str(d[k]).strip()
    # fallback: algo compacto
    for k, v in d.items():
        if str(v).strip():
            return f"{k}={v}"
    return "sample"


# ============================================================
# AMP ctx
# ============================================================

def _get_autocast_ctx(device: torch.device, use_amp: bool):
    if bool(use_amp) and device.type == "cuda":
        return torch.amp.autocast("cuda", enabled=True)
    return torch.amp.autocast("cpu", enabled=False)


# ============================================================
# Plot helpers (colores por clase)
# ============================================================

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


# ============================================================
# ARGPARSE (solo flags; el resto viene en Parte 5)
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
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
    p.add_argument("--knn_chunk_size", type=int, default=1024)

    p.add_argument("--bg_class", type=int, default=0)
    p.add_argument("--bg_weight", type=float, default=0.03)
    p.add_argument("--d21_internal", type=int, default=8)

    p.add_argument("--neighbor_teeth", type=str, default="")  # "d11:1,d22:9"

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--normalize", action="store_true")

    p.add_argument("--plot_every", type=int, default=10)

    p.add_argument("--do_infer", action="store_true")
    p.add_argument("--infer_examples", type=int, default=12)
    p.add_argument("--infer_split", type=str, default="test", choices=["train", "val", "test"])

    # robustez: infer workers (por defecto limitado, como acordamos)
    p.add_argument("--infer_num_workers", type=int, default=-1,
                  help="si -1 => min(num_workers,2); si 0 => single process")

    return p.parse_args()

# ============================================================
# PARTE 2/5 — DGCNN (KNN chunk-safe) + EdgeConv + Modelo
# ============================================================

def knn(x: torch.Tensor, k: int, chunk_size: int = 0) -> torch.Tensor:
    """
    x: [B, C, N]
    return idx: [B, N, k]

    Chunk-safe REAL:
      - queries por bloques M, pero distancia contra TODO N
      - evita mismatch tipo 1024 vs 8192

    dist(q, x) = ||q||^2 + ||x||^2 - 2 q^T x
    """
    assert x.dim() == 3, f"x debe ser [B,C,N], got {x.shape}"
    B, C, N = x.shape
    k = int(k)
    if k <= 0:
        raise ValueError("k debe ser > 0")
    k = min(k, N)

    # ||x||^2 para todos los puntos (keys)
    xx = (x ** 2).sum(dim=1)  # [B, N]

    chunk_size = int(chunk_size or 0)
    if chunk_size <= 0 or chunk_size >= N:
        # full
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
        knn_chunk_size: int = 1024,
    ):
        super().__init__()
        self.k = int(k)
        self.knn_chunk_size = int(knn_chunk_size)
        self.emb_dims = int(emb_dims)
        self.dropout = float(dropout)

        # EdgeConv stacks
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

        e1 = get_graph_feature(x,  k=self.k, knn_chunk_size=self.knn_chunk_size)
        x1 = self.ec1(e1)  # [B,64,N]

        e2 = get_graph_feature(x1, k=self.k, knn_chunk_size=self.knn_chunk_size)
        x2 = self.ec2(e2)  # [B,64,N]

        e3 = get_graph_feature(x2, k=self.k, knn_chunk_size=self.knn_chunk_size)
        x3 = self.ec3(e3)  # [B,128,N]

        e4 = get_graph_feature(x3, k=self.k, knn_chunk_size=self.knn_chunk_size)
        x4 = self.ec4(e4)  # [B,256,N]

        x_cat = torch.cat((x1, x2, x3, x4), dim=1)   # [B,512,N]
        f = self.fuse(x_cat)                         # [B,emb,N]

        g = torch.max(f, dim=2, keepdim=True).values # [B,emb,1]
        g = g.expand(-1, -1, x_cat.size(2))          # [B,emb,N]

        h = torch.cat((x_cat, g), dim=1)             # [B,512+emb,N]
        logits = self.head(h)                        # [B,C,N]
        return logits.permute(0, 2, 1).contiguous()  # [B,N,C]
    
    # ============================================================
# PARTE 3/5 — LOSS + MÉTRICAS + VECINOS + PLOTS + run_epoch
# ============================================================

# ----------------------------
# LOSS: BG incluido, downweight bg (SIN ignore_index)
# ----------------------------
def make_loss_fn(num_classes: int, bg_class: int, bg_weight: float, device: torch.device) -> nn.Module:
    """
    CrossEntropy con pesos por clase:
      - bg_class tiene peso bg_weight (ej: 0.03)
      - resto 1.0

    IMPORTANTE: weight se crea en el MISMO device que el modelo/logits.
    """
    C = int(num_classes)
    bg = int(bg_class)
    w = torch.ones(C, dtype=torch.float32, device=device)
    if 0 <= bg < C:
        w[bg] = float(bg_weight)
    return nn.CrossEntropyLoss(weight=w)


# ----------------------------
# MÉTRICAS (macro sin BG, diente binario, etc.)
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
    Métricas binarias para un diente (sirve para d21 y para vecinos):
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

    f1m, ioum = macro_metrics_no_bg(pred, y, C=int(C), bg=int(bg))

    d21_acc, d21_f1, d21_iou = d21_metrics_binary(pred, y, d21_idx=int(d21_idx), bg=int(bg), include_bg=False)
    d21_bin_acc_all, _, _ = d21_metrics_binary(pred, y, d21_idx=int(d21_idx), bg=int(bg), include_bg=True)

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


# ----------------------------
# VECINOS (eval por loader; lista arbitraria)
# ----------------------------
@torch.no_grad()
def eval_neighbors_on_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    neighbor_list: List[Tuple[str, int]],
    bg: int,
    use_amp: bool,
    C: int,
) -> Dict[str, float]:
    """
    Retorna promedios:
      {name}_acc, {name}_f1, {name}_iou, {name}_bin_acc_all

    - acc/f1/iou: include_bg=False
    - bin_acc_all: include_bg=True
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
    for xyz, y in loader:
        xyz = xyz.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with ctx:
            logits = model(xyz)  # [B,N,C]
        pred = logits.argmax(dim=-1)

        for name, idx in neighbor_list:
            acc, f1, iou = d21_metrics_binary(pred, y, d21_idx=int(idx), bg=int(bg), include_bg=False)
            acc_all, _, _ = d21_metrics_binary(pred, y, d21_idx=int(idx), bg=int(bg), include_bg=True)

            sums[f"{name}_acc"] += float(acc)
            sums[f"{name}_f1"] += float(f1)
            sums[f"{name}_iou"] += float(iou)
            sums[f"{name}_bin_acc_all"] += float(acc_all)

        nb += 1

    nb = max(1, nb)
    return {k: v / nb for k, v in sums.items()}


# ----------------------------
# PLOTS (mismo estilo: Train vs Val, + 3D GT/Pred/Err/d21)
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


# ----------------------------
# AMP moderno
# ----------------------------
def _get_autocast_ctx(device: torch.device, use_amp: bool):
    if bool(use_amp) and device.type == "cuda":
        return torch.amp.autocast("cuda", enabled=True)
    return torch.amp.autocast("cpu", enabled=False)


# ----------------------------
# run_epoch (train: 1 forward para loss + 2do forward eval para métricas)
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
                loss = loss_fn(logits_train.reshape(-1, int(C)), y.reshape(-1))

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
            metrics = _compute_metrics_from_logits(logits_eval, y, C=int(C), d21_idx=int(d21_idx), bg=int(bg))

        else:
            with torch.no_grad():
                with ctx:
                    logits = model(xyz)
                    loss = loss_fn(logits.reshape(-1, int(C)), y.reshape(-1))
                metrics = _compute_metrics_from_logits(logits, y, C=int(C), d21_idx=int(d21_idx), bg=int(bg))

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
# PARTE 4/5 — OUTPUTS "PAPER-LIKE": logging, json/csv/jsonl, ckpts, test, infer manifest
# ============================================================

import logging
from logging import Logger


# ----------------------------
# Logging robusto (run.log + errors.log + console)
# ----------------------------
def setup_logging(out_dir: Path) -> Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_run = out_dir / "run.log"
    log_err = out_dir / "errors.log"

    logger = logging.getLogger("dgcnn_v10")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # run.log
    fh = logging.FileHandler(log_run, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # errors.log
    eh = logging.FileHandler(log_err, encoding="utf-8")
    eh.setLevel(logging.ERROR)
    eh.setFormatter(fmt)
    logger.addHandler(eh)

    logger.info(f"[log] run.log={log_run}")
    logger.info(f"[log] errors.log={log_err}")
    return logger


# ----------------------------
# JSON helpers
# ----------------------------
def safe_write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, row: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ----------------------------
# CSV helpers
# ----------------------------
def append_metrics_csv(csv_path: Path, row: Dict[str, Any], header_order: Optional[List[str]] = None):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    header = header_order if header_order is not None else list(row.keys())
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in header})


# ----------------------------
# Run meta / history / test metrics
# ----------------------------
def save_run_meta(out_dir: Path, args: argparse.Namespace, extra: Dict[str, Any]):
    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).name),
        "args": vars(args),
        "extra": extra,
    }
    safe_write_json(out_dir / "run_meta.json", meta)


def save_history(out_dir: Path, history: Dict[str, List[float]]):
    safe_write_json(out_dir / "history.json", history)


def save_test_metrics(out_dir: Path, metrics: Dict[str, float], extra: Optional[Dict[str, Any]] = None):
    payload = {"test": metrics}
    if extra:
        payload["extra"] = extra
    safe_write_json(out_dir / "test_metrics.json", payload)


# ----------------------------
# Checkpoints
# ----------------------------
def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optim: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    args: argparse.Namespace,
    best_epoch: int,
    best_val: float,
    C: int,
    bg: int,
    d21_idx: int,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model_state": model.state_dict(),
            "optim_state": optim.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "args": vars(args),
            "best_epoch": int(best_epoch),
            "best_val": float(best_val),
            "num_classes": int(C),
            "bg_class": int(bg),
            "d21_internal": int(d21_idx),
        },
        path,
    )


def load_checkpoint(path: Path, model: nn.Module, map_location: str = "cpu") -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    ck = torch.load(path, map_location=map_location)
    model.load_state_dict(ck["model_state"], strict=True)
    return ck


# ----------------------------
# LR helper
# ----------------------------
def get_lr(optim: torch.optim.Optimizer) -> float:
    for g in optim.param_groups:
        return float(g.get("lr", 0.0))
    return 0.0


# ============================================================
# Inferencia "paper-like": carpetas + manifest.csv
#   inference/
#     inference_manifest.csv
#     inference_all/
#     inference_errors/
#     inference_d21/
# ============================================================

def _discover_index_csv(data_dir: Path, split: str) -> Optional[Path]:
    """
    Busca index_{split}.csv de forma robusta:
      1) data_dir/index_{split}.csv
      2) en ancestros hasta Teeth_3ds (o raíz)
      3) fallback: primer match en ancestros
    """
    data_dir = Path(data_dir).resolve()
    target = f"index_{split}.csv"

    p = data_dir / target
    if p.exists():
        return p

    parents = [data_dir] + list(data_dir.parents)

    # 2) hasta Teeth_3ds
    for parent in parents:
        cand = parent / target
        if cand.exists():
            return cand
        if parent.name.lower() in ("teeth_3ds", "teeth3ds"):
            # buscar bajo Teeth_3ds (merged_* / fixed_split/*)
            try:
                for sub in parent.rglob(target):
                    return sub
            except Exception:
                pass
            break

    # 3) fallback: primer match
    for parent in parents:
        cand = parent / target
        if cand.exists():
            return cand

    return None


def _read_index_csv(p: Optional[Path]) -> Optional[Dict[int, Dict[str, str]]]:
    if p is None:
        return None
    p = Path(p)
    if not p.exists():
        return None

    out: Dict[int, Dict[str, str]] = {}
    try:
        with p.open("r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for i, row in enumerate(r):
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
                out[int(rid)] = {kk: ("" if row.get(kk) is None else str(row.get(kk))) for kk in row.keys()}
        return out
    except Exception:
        return None


def _pick_trace_label(d: Dict[str, str]) -> str:
    for k in ("sample_name", "patient", "patient_id", "scan_id", "id", "path", "relpath", "upper_path"):
        if k in d and str(d[k]).strip():
            return str(d[k]).strip()
    for k, v in d.items():
        if str(v).strip():
            return f"{k}={v}"
    return "sample"


def _infer_dirs(out_dir: Path) -> Dict[str, Path]:
    base = out_dir / "inference"
    d_all = base / "inference_all"
    d_err = base / "inference_errors"
    d_d21 = base / "inference_d21"
    d_all.mkdir(parents=True, exist_ok=True)
    d_err.mkdir(parents=True, exist_ok=True)
    d_d21.mkdir(parents=True, exist_ok=True)
    return {"base": base, "all": d_all, "err": d_err, "d21": d_d21}


def _open_manifest(path: Path, header: List[str]) -> Tuple[csv.DictWriter, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    f = path.open("a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=header)
    if not exists:
        w.writeheader()
    return w, f


@torch.no_grad()
def infer_and_save_examples_paperlike(
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
    logger: Optional[Logger] = None,
):
    """
    Inferencia trazable y organizada:
      - usa X_{split}.npz / Y_{split}.npz
      - busca index_{split}.csv (ancestros) para label humano
      - guarda PNGs en inference/inference_all|errors|d21
      - escribe inference/inference_manifest.csv

    Nombres PNG:
      ex_{i:04d}.all.png
      ex_{i:04d}.err.png
      ex_{i:04d}.d21.png
    """
    data_dir = Path(data_dir).resolve()
    Xp = data_dir / f"X_{split}.npz"
    Yp = data_dir / f"Y_{split}.npz"
    if (not Xp.exists()) or (not Yp.exists()):
        if logger:
            logger.info(f"[infer] No existe {Xp.name} o {Yp.name} en {data_dir}. Skip infer.")
        return

    X = np.load(Xp)["X"]  # [M,N,3]
    Y = np.load(Yp)["Y"]  # [M,N]
    M = int(X.shape[0])
    K = min(int(n_examples), M)

    idx_path = _discover_index_csv(data_dir, split=split)
    idx_map = _read_index_csv(idx_path) if idx_path is not None else None

    dirs = _infer_dirs(out_dir)
    manifest = dirs["base"] / "inference_manifest.csv"
    header = ["split", "i", "trace", "png_all", "png_err", "png_d21"]
    w, f = _open_manifest(manifest, header)

    model.eval()
    ctx = _get_autocast_ctx(device, bool(use_amp))

    if logger:
        logger.info(f"[infer] split={split} M={M} examples={K} index_csv={idx_path if idx_path else 'None'}")
        logger.info(f"[infer] out={dirs['base']}")

    for i in range(K):
        xyz = X[i].astype(np.float32, copy=False)
        ygt = Y[i].astype(np.int32, copy=False)

        trace = f"i={i:04d}"
        if idx_map is not None and i in idx_map:
            trace = f"i={i:04d} | {_pick_trace_label(idx_map[i])}"

        tx = torch.as_tensor(xyz, dtype=torch.float32, device=device).unsqueeze(0)  # [1,N,3]
        with ctx:
            logits = model(tx)  # [1,N,C]
        pred = logits.argmax(dim=-1).squeeze(0).detach().cpu().numpy().astype(np.int32)

        # paths
        p_all = dirs["all"] / f"ex_{i:04d}.all.png"
        p_err = dirs["err"] / f"ex_{i:04d}.err.png"
        p_d21 = dirs["d21"] / f"ex_{i:04d}.d21.png"

        plot_pointcloud_all_classes(xyz, ygt, pred, p_all, C=int(C), title=trace, s=1.0)
        plot_errors(xyz, ygt, pred, p_err, bg=int(bg), title=trace, s=1.0)
        plot_d21_focus(xyz, ygt, pred, p_d21, d21_idx=int(d21_idx), bg=int(bg), title=trace, s=1.2)

        w.writerow(
            {
                "split": str(split),
                "i": int(i),
                "trace": trace,
                "png_all": str(p_all.relative_to(out_dir)),
                "png_err": str(p_err.relative_to(out_dir)),
                "png_d21": str(p_d21.relative_to(out_dir)),
            }
        )

    f.close()
    if logger:
        logger.info(f"[infer] manifest={manifest}")


# ============================================================
# Plots "todas las métricas" (Train vs Val) incluyendo vecinos
# ============================================================

def plot_all_metrics_paperlike(
    plots_dir: Path,
    history: Dict[str, List[float]],
    best_epoch: int,
    neighbors: List[Tuple[str, int]],
):
    plots_dir.mkdir(parents=True, exist_ok=True)

    # core
    plot_train_val("loss", history["train_loss"], history["val_loss"], plots_dir / "loss.png", best_epoch=best_epoch)
    plot_train_val("f1_macro", history["train_f1_macro"], history["val_f1_macro"], plots_dir / "f1_macro.png", best_epoch=best_epoch)
    plot_train_val("iou_macro", history["train_iou_macro"], history["val_iou_macro"], plots_dir / "iou_macro.png", best_epoch=best_epoch)
    plot_train_val("acc_all", history["train_acc_all"], history["val_acc_all"], plots_dir / "acc_all.png", best_epoch=best_epoch)
    plot_train_val("acc_no_bg", history["train_acc_no_bg"], history["val_acc_no_bg"], plots_dir / "acc_no_bg.png", best_epoch=best_epoch)
    plot_train_val("pred_bg_frac", history["train_pred_bg_frac"], history["val_pred_bg_frac"], plots_dir / "pred_bg_frac.png", best_epoch=best_epoch)

    # d21
    plot_train_val("d21_acc", history["train_d21_acc"], history["val_d21_acc"], plots_dir / "d21_acc.png", best_epoch=best_epoch)
    plot_train_val("d21_f1", history["train_d21_f1"], history["val_d21_f1"], plots_dir / "d21_f1.png", best_epoch=best_epoch)
    plot_train_val("d21_iou", history["train_d21_iou"], history["val_d21_iou"], plots_dir / "d21_iou.png", best_epoch=best_epoch)
    plot_train_val("d21_bin_acc_all", history["train_d21_bin_acc_all"], history["val_d21_bin_acc_all"], plots_dir / "d21_bin_acc_all.png", best_epoch=best_epoch)

    # vecinos (solo val, pero lo graficamos igual como serie)
    for name, _ in neighbors:
        if f"val_{name}_acc" in history:
            plot_train_val(f"{name}_acc (val)", history[f"val_{name}_acc"], history[f"val_{name}_acc"], plots_dir / f"neighbor_{name}_acc.png", best_epoch=best_epoch)
        if f"val_{name}_f1" in history:
            plot_train_val(f"{name}_f1 (val)", history[f"val_{name}_f1"], history[f"val_{name}_f1"], plots_dir / f"neighbor_{name}_f1.png", best_epoch=best_epoch)
        if f"val_{name}_iou" in history:
            plot_train_val(f"{name}_iou (val)", history[f"val_{name}_iou"], history[f"val_{name}_iou"], plots_dir / f"neighbor_{name}_iou.png", best_epoch=best_epoch)
        if f"val_{name}_bin_acc_all" in history:
            plot_train_val(f"{name}_bin_acc_all (val)", history[f"val_{name}_bin_acc_all"], history[f"val_{name}_bin_acc_all"], plots_dir / f"neighbor_{name}_bin_acc_all.png", best_epoch=best_epoch)


# ============================================================
# Infer num_workers robusto (por defecto limit)
# ============================================================
def cap_infer_num_workers(num_workers: int, hard_cap: int = 2) -> int:
    try:
        nw = int(num_workers)
    except Exception:
        nw = 0
    return int(min(max(nw, 0), int(hard_cap)))

# ============================================================
# PARTE 5/5 — MAIN v10 (todo conectado)
#   - imprime métricas core + d21 + vecinos
#   - genera plots de TODAS las métricas (incluye vecinos)
#   - guarda: run_meta.json, run.log/errors.log, metrics_epoch.csv,
#             history.json + history_epoch.jsonl, best.pt/last.pt, test_metrics.json
#   - inferencia: inference/ con manifest + inference_all/errors/d21
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
    p.add_argument("--infer_num_workers_cap", type=int, default=2, help="cap para workers SOLO en infer (robustez)")

    return p.parse_args()


def main():
    args = parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(out_dir)

    # seed
    set_seed(int(args.seed))

    # device
    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # label map / num classes
    label_map = load_label_map(data_dir)
    C = infer_num_classes(data_dir, label_map)
    bg = int(args.bg_class)
    d21_idx = int(args.d21_internal)

    # neighbors
    neighbors = parse_neighbors(args.neighbor_teeth)

    # loaders
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

    # loss (BG incluido, downweight bg) — weight en device
    loss_fn = make_loss_fn(num_classes=int(C), bg_class=int(bg), bg_weight=float(args.bg_weight), device=device)

    # outputs
    metrics_csv = out_dir / "metrics_epoch.csv"
    plots_dir = out_dir / "plots"
    hist_jsonl = out_dir / "history_epoch.jsonl"
    ckpt_best = out_dir / "best.pt"
    ckpt_last = out_dir / "last.pt"

    # run_meta
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

    # CSV header order (idéntico espíritu PointNet/DGCNN paper-like)
    header_order = [
        "epoch", "lr",
        "train_loss", "train_f1_macro", "train_iou_macro", "train_acc_all", "train_acc_no_bg",
        "train_d21_acc", "train_d21_f1", "train_d21_iou", "train_d21_bin_acc_all", "train_pred_bg_frac",
        "val_loss", "val_f1_macro", "val_iou_macro", "val_acc_all", "val_acc_no_bg",
        "val_d21_acc", "val_d21_f1", "val_d21_iou", "val_d21_bin_acc_all", "val_pred_bg_frac",
    ]
    for name, _ in neighbors:
        header_order += [f"val_{name}_acc", f"val_{name}_f1", f"val_{name}_iou", f"val_{name}_bin_acc_all"]

    # prints setup
    logger.info(f"[setup] data_dir={data_dir}")
    logger.info(f"[setup] out_dir={out_dir}")
    logger.info(f"[setup] device={device} | C={C} | bg={bg} | d21={d21_idx}")
    logger.info(f"[setup] args={vars(args)}")
    if neighbors:
        logger.info(f"[setup] neighbors={neighbors}")

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

        scheduler.step()
        lr = get_lr(optim)

        # history (core + d21)
        history["train_loss"].append(float(tr["loss"]))
        history["val_loss"].append(float(va["loss"]))
        for k in ("f1_macro", "iou_macro", "acc_all", "acc_no_bg",
                  "d21_acc", "d21_f1", "d21_iou", "d21_bin_acc_all", "pred_bg_frac"):
            history[f"train_{k}"].append(float(tr[k]))
            history[f"val_{k}"].append(float(va[k]))

        # neighbors history (val)
        for name, _ in neighbors:
            history[f"val_{name}_acc"].append(float(neigh_val.get(f"{name}_acc", 0.0)))
            history[f"val_{name}_f1"].append(float(neigh_val.get(f"{name}_f1", 0.0)))
            history[f"val_{name}_iou"].append(float(neigh_val.get(f"{name}_iou", 0.0)))
            history[f"val_{name}_bin_acc_all"].append(float(neigh_val.get(f"{name}_bin_acc_all", 0.0)))

        # console line (incluye vecinos)
        neigh_str = ""
        if neighbors:
            parts = []
            for name, _ in neighbors:
                parts.append(
                    f"{name}_acc={neigh_val.get(f'{name}_acc', 0.0):.3f} "
                    f"{name}_f1={neigh_val.get(f'{name}_f1', 0.0):.3f} "
                    f"{name}_iou={neigh_val.get(f'{name}_iou', 0.0):.3f} "
                    f"{name}_bin_acc_all={neigh_val.get(f'{name}_bin_acc_all', 0.0):.3f}"
                )
            neigh_str = " | [neighbors val] " + " | ".join(parts)

        logger.info(
            f"[{epoch:03d}/{args.epochs}] "
            f"train loss={tr['loss']:.4f} f1m={tr['f1_macro']:.3f} ioum={tr['iou_macro']:.3f} "
            f"acc_all={tr['acc_all']:.3f} acc_no_bg={tr['acc_no_bg']:.3f} | "
            f"val loss={va['loss']:.4f} f1m={va['f1_macro']:.3f} ioum={va['iou_macro']:.3f} "
            f"acc_all={va['acc_all']:.3f} acc_no_bg={va['acc_no_bg']:.3f} | "
            f"d21 acc={va['d21_acc']:.3f} f1={va['d21_f1']:.3f} iou={va['d21_iou']:.3f} | "
            f"d21(bin all) acc={va['d21_bin_acc_all']:.3f} | "
            f"pred_bg_frac(val)={va['pred_bg_frac']:.3f} lr={lr:.2e}"
            f"{neigh_str}"
        )

        # CSV row
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

        # history json + epoch jsonl (extra trazabilidad)
        save_history(out_dir, history)
        append_jsonl(hist_jsonl, {"epoch": epoch, "lr": lr, "train": tr, "val": va, "neighbors_val": neigh_val})

        # save last
        save_checkpoint(
            ckpt_last,
            epoch=epoch,
            model=model,
            optim=optim,
            scheduler=scheduler,
            args=args,
            best_epoch=best_epoch,
            best_val=best_val,
            C=int(C),
            bg=int(bg),
            d21_idx=int(d21_idx),
        )

        # save best
        cur = float(va["iou_macro"]) if best_key == "val_iou_macro" else float(va["f1_macro"])
        if cur > best_val:
            best_val = cur
            best_epoch = epoch
            save_checkpoint(
                ckpt_best,
                epoch=epoch,
                model=model,
                optim=optim,
                scheduler=scheduler,
                args=args,
                best_epoch=best_epoch,
                best_val=best_val,
                C=int(C),
                bg=int(bg),
                d21_idx=int(d21_idx),
            )

        # plots: TODAS las métricas (incluye vecinos)
        if int(args.plot_every) > 0 and (epoch % int(args.plot_every) == 0 or epoch == 1 or epoch == int(args.epochs)):
            plot_all_metrics_paperlike(plots_dir, history, best_epoch=best_epoch, neighbors=neighbors)

    dt = time.time() - t0
    logger.info(f"[done] Entrenamiento terminado en {dt/60:.1f} min. best_epoch={best_epoch} best_val={best_val:.4f}")

    # ==========================
    # TEST (cargar best.pt si existe)
    # ==========================
    if ckpt_best.exists():
        ck = load_checkpoint(ckpt_best, model=model, map_location="cpu")
        if ck is not None:
            logger.info(f"[test] Cargado best.pt (epoch={ck.get('epoch')})")

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

    # guardar test_metrics.json (incluye vecinos como neighbor_{name}_*)
    test_payload = {k: float(v) for k, v in te.items()}
    test_payload.update({f"neighbor_{k}": float(v) for k, v in neigh_test.items()})
    save_test_metrics(out_dir, metrics=test_payload, extra={"best_epoch": best_epoch, "best_val": best_val})

    # imprimir test + vecinos
    neigh_t_str = ""
    if neighbors:
        parts = []
        for name, _ in neighbors:
            parts.append(
                f"{name}_acc={neigh_test.get(f'{name}_acc', 0.0):.3f} "
                f"{name}_f1={neigh_test.get(f'{name}_f1', 0.0):.3f} "
                f"{name}_iou={neigh_test.get(f'{name}_iou', 0.0):.3f} "
                f"{name}_bin_acc_all={neigh_test.get(f'{name}_bin_acc_all', 0.0):.3f}"
            )
        neigh_t_str = " | [neighbors test] " + " | ".join(parts)

    logger.info(
        f"[test] loss={te['loss']:.4f} f1m={te['f1_macro']:.3f} ioum={te['iou_macro']:.3f} "
        f"acc_all={te['acc_all']:.3f} acc_no_bg={te['acc_no_bg']:.3f} | "
        f"d21 acc={te['d21_acc']:.3f} f1={te['d21_f1']:.3f} iou={te['d21_iou']:.3f}"
        f"{neigh_t_str}"
    )

    # ==========================
    # INFER (paper-like)
    # ==========================
    if bool(args.do_infer):
        # cap workers solo para infer (robustez)
        _ = cap_infer_num_workers(int(args.num_workers), hard_cap=int(args.infer_num_workers_cap))

        infer_and_save_examples_paperlike(
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
            logger=logger,
        )


if __name__ == "__main__":
    main()


# ============================================================
# CÓMO CORRER (v10) — mismo estilo que tú
# ============================================================

# conda activate enviroment
# cd /home/htaucare/Tesis_Amaro/scripts_last_version
#
# CUDA_VISIBLE_DEVICES=1 python3 train_dgcnn_classic_only_fixed_v10.py \
#   --data_dir /home/htaucare/Tesis_Amaro/data/Teeth_3ds/fixed_split/8192/upper_only_surf_global_excl_wisdom_seed42_aug2 \
#   --out_dir  /home/htaucare/Tesis_Amaro/outputs/dgcnn/gpu1_run1_v10_neighbors \
#   --epochs 120 --batch_size 16 --lr 2e-4 --weight_decay 1e-4 --dropout 0.5 \
#   --num_workers 6 --device cuda --use_amp --grad_clip 1.0 \
#   --k 20 --emb_dims 1024 --knn_chunk_size 1024 \
#   --bg_class 0 --bg_weight 0.03 --d21_internal 8 \
#   --neighbor_teeth "d11:1,d22:9" \
#   --normalize \
#   --plot_every 10 \
#   --do_infer --infer_examples 12 --infer_split test
#
# GPU0 (ejemplo):
# CUDA_VISIBLE_DEVICES=0 python3 train_dgcnn_classic_only_fixed_v10.py \
#   --data_dir /home/htaucare/Tesis_Amaro/data/Teeth_3ds/fixed_split/8192/upper_only_surf_global_excl_wisdom_seed42_aug2 \
#   --out_dir  /home/htaucare/Tesis_Amaro/outputs/dgcnn/gpu0_run1_v10_neighbors \
#   --epochs 120 --batch_size 16 --lr 2e-4 --weight_decay 1e-4 --dropout 0.5 \
#   --num_workers 6 --device cuda --use_amp --grad_clip 1.0 \
#   --k 20 --emb_dims 1024 --knn_chunk_size 1024 \
#   --bg_class 0 --bg_weight 0.03 --d21_internal 8 \
#   --neighbor_teeth "d11:1,d22:9" \
#   --normalize \
#   --plot_every 10 \
#   --do_infer --infer_examples 12 --infer_split test