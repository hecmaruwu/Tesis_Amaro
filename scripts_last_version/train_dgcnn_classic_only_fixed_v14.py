#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_dgcnn_classic_only_fixed_v14.py

DGCNN (EdgeConv) – Segmentación multiclase dental 3D

✅ MISMA TRAZABILIDAD/OUTPUTS "paper-like" que tu stack final:
   - run_meta.json
   - run.log / errors.log
   - metrics_epoch.csv
   - history.json + history_epoch.jsonl
   - best.pt / last.pt
   - test_metrics.json
   - plots/*.png  (Train vs Val; SIN línea de test)  ✅ (v14: incluye d21 + pred_bg_frac + neighbors)
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

FIX v14:
✅ make_loaders() devuelve 4 valores (dl_tr, dl_va, dl_te, ds_te) para compatibilidad con main()
✅ Mantiene TODO lo demás: outputs, forma de dibujar, métricas, plots, inferencia.

Ejemplo (GPU 1):
CUDA_VISIBLE_DEVICES=1 python3 train_dgcnn_classic_only_fixed_v14.py \
  --data_dir /home/htaucare/Tesis_Amaro/data/Teeth_3ds/fixed_split/8192/upper_only_surf_global_excl_wisdom_seed42_aug2 \
  --out_dir  /home/htaucare/Tesis_Amaro/outputs/dgcnn/gpu1_run1_v14_neighbors \
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

def setup_logging(out_dir: Path, name: str = "dgcnn_v14") -> Logger:
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
    v14 FIX:
    Devuelve EXACTAMENTE 4 objetos:
      dl_tr, dl_va, dl_te, ds_te
    (Para compatibilidad con main() estilo PointNet v5.)
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

    return dl_tr, dl_va, dl_te, ds_te

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
# Neighbors parsing (v14 mantiene ambas funciones por compat)
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


def parse_neighbor_teeth(spec: Optional[str]) -> List[Tuple[str, int]]:
    """
    Parse de --neighbor_teeth con formato:
      "d11:1,d22:9,foo:3"
    Devuelve [(name, idx), ...] (orden estable, override por nombre).
    (Se mantiene por compat con versiones previas.)
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
# Checkpoints (mantener helpers completos)
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
    xx = (x ** 2).sum(dim=-1, keepdim=True)                      # [B,N,1]
    yy = (y ** 2).sum(dim=-1, keepdim=True).transpose(1, 2)      # [B,1,M]
    xy = torch.bmm(x, y.transpose(1, 2))                         # [B,N,M]
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

    device = x.device
    idx_out = torch.empty((B, N, k), device=device, dtype=torch.long)

    for s in range(0, N, chunk_size):
        e = min(N, s + chunk_size)
        xq = x[:, s:e, :]                # [B, Q, D]
        dist = pairwise_dist(xq, x)      # [B, Q, N]

        # excluir self: dist[b, q, (s+q)] = +inf
        q = e - s
        ar = torch.arange(q, device=device)
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
        idx = knn_indices_chunked(xt, k=k, chunk_size=int(knn_chunk_size))  # [B,N,k]

    device = x.device
    idx_base = torch.arange(0, B, device=device).view(-1, 1, 1) * N         # [B,1,1]
    idx = idx + idx_base                                                    # [B,N,k]
    idx = idx.reshape(-1)                                                   # [B*N*k]

    neigh = xt.reshape(B * N, C)[idx, :].view(B, N, k, C)                   # [B,N,k,C]
    center = xt.view(B, N, 1, C).expand(-1, -1, k, -1)                      # [B,N,k,C]

    edge = torch.cat((neigh - center, center), dim=3)                       # [B,N,k,2C]
    edge = edge.permute(0, 3, 1, 2).contiguous()                            # [B,2C,N,k]
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
# Métricas multiclase (macro SIN BG)
# ============================================================

@torch.no_grad()
def compute_multiclass_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    C: int,
    bg: int,
):
    """
    logits: [B,N,C]
    target: [B,N]
    Retorna:
      acc_all, acc_no_bg, prec_m, rec_m, f1_m, iou_m, pred_bg_frac
    """
    pred = logits.argmax(dim=-1)              # [B,N]

    # flatten
    pred_f = pred.reshape(-1)
    tgt_f  = target.reshape(-1)

    total = tgt_f.numel()
    correct = (pred_f == tgt_f).sum().item()
    acc_all = float(correct) / float(total + 1e-9)

    # mask no-bg
    mask_no_bg = (tgt_f != int(bg))
    if mask_no_bg.any():
        correct_no_bg = (pred_f[mask_no_bg] == tgt_f[mask_no_bg]).sum().item()
        acc_no_bg = float(correct_no_bg) / float(mask_no_bg.sum().item() + 1e-9)
    else:
        acc_no_bg = 0.0

    # per-class
    eps = 1e-9
    prec_list, rec_list, f1_list, iou_list = [], [], [], []

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

        prec_list.append(prec)
        rec_list.append(rec)
        f1_list.append(f1)
        iou_list.append(iou)

    if len(f1_list) > 0:
        prec_m = float(np.mean(prec_list))
        rec_m  = float(np.mean(rec_list))
        f1_m   = float(np.mean(f1_list))
        iou_m  = float(np.mean(iou_list))
    else:
        prec_m = rec_m = f1_m = iou_m = 0.0

    pred_bg_frac = float((pred_f == int(bg)).sum().item()) / float(total + 1e-9)

    return acc_all, acc_no_bg, prec_m, rec_m, f1_m, iou_m, pred_bg_frac


# ============================================================
# Métrica diente 21 (binaria correcta)
# ============================================================

@torch.no_grad()
def compute_d21_binary_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    d21_idx: int,
):
    """
    Trata clase d21 como positiva (1), resto como negativa (0).
    Retorna:
      acc_cls (solo donde GT=d21),
      f1_cls,
      iou_cls,
      bin_acc_all (incluye todo, incluso bg)
    """
    pred = logits.argmax(dim=-1)

    pred_bin = (pred == int(d21_idx))
    tgt_bin  = (target == int(d21_idx))

    eps = 1e-9

    tp = (pred_bin & tgt_bin).sum().item()
    fp = (pred_bin & (~tgt_bin)).sum().item()
    fn = ((~pred_bin) & tgt_bin).sum().item()
    tn = ((~pred_bin) & (~tgt_bin)).sum().item()

    # acc solo sobre positivos reales
    pos = tgt_bin.sum().item()
    if pos > 0:
        acc_cls = float(tp) / float(pos + eps)
    else:
        acc_cls = 0.0

    prec = tp / (tp + fp + eps)
    rec  = tp / (tp + fn + eps)
    f1   = 2.0 * prec * rec / (prec + rec + eps)
    iou  = tp / (tp + fp + fn + eps)

    bin_acc_all = float(tp + tn) / float(tp + tn + fp + fn + eps)

    return float(acc_cls), float(f1), float(iou), float(bin_acc_all)


# ============================================================
# Métricas vecinos arbitrarios
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
    Retorna dict:
      {
        "d11": {"acc":..., "f1":..., "iou":..., "bin_acc_all":...},
        ...
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

        with get_autocast_ctx(device, use_amp):
            logits = model(xyz, knn_chunk_size=knn_chunk_size)

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

        # acc sobre positivos reales
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
# Visualización 3D (GT vs Pred) estilo Paper
# ============================================================

def plot_3d_seg(
    xyz,
    gt,
    pred,
    C: int,
    save_path: Path,
    title: str = "",
):
    """
    xyz: [N,3]
    gt/pred: [N]
    """
    xyz = to_np(xyz)
    gt  = to_np(gt)
    pred= to_np(pred)

    colors = class_colors(C)

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    for c in range(int(C)):
        mask = (gt == c)
        if mask.any():
            ax1.scatter(
                xyz[mask, 0],
                xyz[mask, 1],
                xyz[mask, 2],
                s=2,
                color=colors[c],
            )

    for c in range(int(C)):
        mask = (pred == c)
        if mask.any():
            ax2.scatter(
                xyz[mask, 0],
                xyz[mask, 1],
                xyz[mask, 2],
                s=2,
                color=colors[c],
            )

    ax1.set_title("GT")
    ax2.set_title("Pred")
    fig.suptitle(title)

    for ax in (ax1, ax2):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


# ============================================================
# Plot Train vs Val (todas las métricas relevantes)
# ============================================================

def plot_train_val_curves(
    history: dict,
    out_dir: Path,
):
    """
    history:
      history["train"]["f1_macro"]
      history["val"]["f1_macro"]
      etc.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def plot_one(key, ylabel):
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
    plot_one("d21_f1", "d21_f1")
    plot_one("d21_iou", "d21_iou")
    plot_one("d21_bin_acc_all", "d21_bin_acc_all")


# ============================================================
# Train/Eval helpers (1 epoch) + Train loop (v14)
# - FIX CRÍTICO v14: make_loaders devuelve 3 valores (dl_tr, dl_va, dl_te)
#   => en main() NO se debe desempacar ds_te.
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
    with get_autocast_ctx(device, use_amp):
        logits = model(xyz, knn_chunk_size=int(knn_chunk_size))  # [B,N,C]
        loss = loss_fn(logits.reshape(-1, int(C)), y.reshape(-1))
    return logits, loss


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
    Eval estándar (sin AMP).
    Retorna dict con:
      loss, acc_all, acc_no_bg, prec_macro, rec_macro, f1_macro, iou_macro, pred_bg_frac,
      d21_acc, d21_f1, d21_iou, d21_bin_acc_all
    """
    model.eval()
    loss_sum = 0.0

    # promedios por batch (consistente con tus otros scripts)
    acc_all_s = acc_no_bg_s = 0.0
    prec_s = rec_s = f1_s = iou_s = 0.0
    pred_bg_frac_s = 0.0
    d21_acc_s = d21_f1_s = d21_iou_s = d21_bin_acc_all_s = 0.0
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

        acc_all, acc_no_bg, prec_m, rec_m, f1_m, iou_m, pred_bg_frac = compute_multiclass_metrics(
            logits=logits, target=y, C=int(C), bg=int(bg)
        )
        d21_acc, d21_f1, d21_iou, d21_bin_acc_all = compute_d21_binary_metrics(
            logits=logits, target=y, d21_idx=int(d21_idx)
        )

        loss_sum += float(loss.item())
        acc_all_s += float(acc_all)
        acc_no_bg_s += float(acc_no_bg)
        prec_s += float(prec_m)
        rec_s  += float(rec_m)
        f1_s   += float(f1_m)
        iou_s  += float(iou_m)
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
    Train:
      - forward/backward/step con AMP opcional
      - para métricas: segundo forward en eval() (dropout OFF) para comparabilidad (como PointNet)
    Retorna mismas métricas que eval_one_epoch.
    """
    model.train(True)

    scaler = train_one_epoch.scaler  # type: ignore
    if bool(use_amp) and device.type == "cuda" and scaler is None:
        scaler = torch.amp.GradScaler("cuda")
        train_one_epoch.scaler = scaler  # type: ignore

    loss_sum = 0.0
    acc_all_s = acc_no_bg_s = 0.0
    prec_s = rec_s = f1_s = iou_s = 0.0
    pred_bg_frac_s = 0.0
    d21_acc_s = d21_f1_s = d21_iou_s = d21_bin_acc_all_s = 0.0
    nb = 0

    for xyz, y in loader:
        xyz = xyz.to(device, non_blocking=True)
        y   = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # ---- forward/loss para update ----
        with get_autocast_ctx(device, use_amp):
            logits_train = model(xyz, knn_chunk_size=int(knn_chunk_size))
            loss = loss_fn(logits_train.reshape(-1, int(C)), y.reshape(-1))

        # ---- backward/step ----
        if bool(use_amp) and device.type == "cuda":
            assert scaler is not None
            scaler.scale(loss).backward()

            if float(grad_clip) and float(grad_clip) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if float(grad_clip) and float(grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()

        # ---- métricas comparables (dropout OFF): segundo forward en eval ----
        model.eval()
        with torch.no_grad():
            logits_eval = model(xyz, knn_chunk_size=int(knn_chunk_size))

        acc_all, acc_no_bg, prec_m, rec_m, f1_m, iou_m, pred_bg_frac = compute_multiclass_metrics(
            logits=logits_eval, target=y, C=int(C), bg=int(bg)
        )
        d21_acc, d21_f1, d21_iou, d21_bin_acc_all = compute_d21_binary_metrics(
            logits=logits_eval, target=y, d21_idx=int(d21_idx)
        )

        loss_sum += float(loss.item())
        acc_all_s += float(acc_all)
        acc_no_bg_s += float(acc_no_bg)
        prec_s += float(prec_m)
        rec_s  += float(rec_m)
        f1_s   += float(f1_m)
        iou_s  += float(iou_m)
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


def train_loop_v14(
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
    d21_idx: int,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    scheduler,
    loss_fn: nn.Module,
    use_amp: bool,
    grad_clip: float,
    knn_chunk_size: int,
    plot_every: int,
    neighbors: List[Tuple[str, int]],
    neighbor_eval_split: str,
    neighbor_every: int,
) -> Tuple[int, float]:
    """
    Produce:
      - metrics_epoch.csv (train+val)
      - history.json + history_epoch.jsonl
      - best.pt / last.pt (con state dict + args básicos)
      - plots/*.png (Train vs Val; sin línea test)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)

    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"

    # CSV header (dinámico neighbors)
    neighbor_cols = []
    for name, _ in neighbors:
        neighbor_cols += [f"{name}_acc", f"{name}_f1", f"{name}_iou", f"{name}_bin_acc_all"]

    csv_path = out_dir / "metrics_epoch.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "epoch", "split",
            "loss",
            "acc_all", "acc_no_bg",
            "prec_macro", "rec_macro", "f1_macro", "iou_macro",
            "d21_acc", "d21_f1", "d21_iou",
            "d21_bin_acc_all",
            "pred_bg_frac",
            "lr", "sec",
        ] + neighbor_cols)

    history = {
        "train": {k: [] for k in [
            "loss",
            "acc_all", "acc_no_bg",
            "prec_macro", "rec_macro", "f1_macro", "iou_macro",
            "d21_acc", "d21_f1", "d21_iou",
            "d21_bin_acc_all",
            "pred_bg_frac",
        ]},
        "val": {k: [] for k in [
            "loss",
            "acc_all", "acc_no_bg",
            "prec_macro", "rec_macro", "f1_macro", "iou_macro",
            "d21_acc", "d21_f1", "d21_iou",
            "d21_bin_acc_all",
            "pred_bg_frac",
        ]},
        "neighbors_val": {},     # name -> met -> series
        "neighbors_test": {},    # llenado en test (parte 5)
    }

    for name, _ in neighbors:
        history["neighbors_val"][name] = {m: [] for m in ["acc", "f1", "iou", "bin_acc_all"]}

    hist_json_path = out_dir / "history.json"
    hist_jsonl_path = out_dir / "history_epoch.jsonl"

    best_epoch = -1
    best_val_f1 = -1.0

    t0 = time.time()

    for epoch in range(1, int(epochs) + 1):
        e0 = time.time()

        tr = train_one_epoch(
            model=model,
            loader=dl_tr,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            C=int(C),
            bg=int(bg),
            d21_idx=int(d21_idx),
            knn_chunk_size=int(knn_chunk_size),
            use_amp=bool(use_amp),
            grad_clip=float(grad_clip),
        )

        va = eval_one_epoch(
            model=model,
            loader=dl_va,
            loss_fn=loss_fn,
            device=device,
            C=int(C),
            bg=int(bg),
            d21_idx=int(d21_idx),
            knn_chunk_size=int(knn_chunk_size),
        )

        # neighbors (val por epoch o cada neighbor_every)
        nb_vals_flat = {}
        do_neighbors = (len(neighbors) > 0) and (str(neighbor_eval_split).lower() in ("val", "both")) \
                       and (int(neighbor_every) > 0) and ((epoch % int(neighbor_every)) == 0)
        if do_neighbors:
            nb = eval_neighbors_on_loader(
                model=model,
                loader=dl_va,
                device=device,
                neighbors=neighbors,
                C=int(C),
                bg=int(bg),
                knn_chunk_size=int(knn_chunk_size),
                use_amp=False,
            )
            for name, _ in neighbors:
                for m in ["acc", "f1", "iou", "bin_acc_all"]:
                    history["neighbors_val"][name][m].append(float(nb.get(name, {}).get(m, 0.0)))
                    nb_vals_flat[f"{name}_{m}"] = float(nb.get(name, {}).get(m, 0.0))
        else:
            for name, _ in neighbors:
                for m in ["acc", "f1", "iou", "bin_acc_all"]:
                    history["neighbors_val"][name][m].append(0.0)
                    nb_vals_flat[f"{name}_{m}"] = 0.0

        # scheduler step
        if scheduler is not None:
            scheduler.step()

        lr_now = get_lr(optimizer)
        sec = float(time.time() - e0)

        # history append
        for k in history["train"].keys():
            history["train"][k].append(float(tr[k]))
            history["val"][k].append(float(va[k]))

        # JSONL epoch snapshot (paper-like)
        append_jsonl(hist_jsonl_path, dict(
            epoch=int(epoch),
            sec=float(sec),
            lr=float(lr_now),
            train=tr,
            val=va,
            neighbors_val=nb_vals_flat,
        ))

        # write CSV rows (train/val)
        row_tr = [
            epoch, "train",
            tr["loss"],
            tr["acc_all"], tr["acc_no_bg"],
            tr["prec_macro"], tr["rec_macro"], tr["f1_macro"], tr["iou_macro"],
            tr["d21_acc"], tr["d21_f1"], tr["d21_iou"],
            tr["d21_bin_acc_all"],
            tr["pred_bg_frac"],
            lr_now, sec
        ]
        row_va = [
            epoch, "val",
            va["loss"],
            va["acc_all"], va["acc_no_bg"],
            va["prec_macro"], va["rec_macro"], va["f1_macro"], va["iou_macro"],
            va["d21_acc"], va["d21_f1"], va["d21_iou"],
            va["d21_bin_acc_all"],
            va["pred_bg_frac"],
            lr_now, sec
        ]
        if neighbor_cols:
            row_tr += ["" for _ in neighbor_cols]  # no neighbors en train (caro)
            row_va += [nb_vals_flat.get(col, 0.0) for col in neighbor_cols]

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(row_tr)
            w.writerow(row_va)

        # checkpoints
        ckpt_obj = dict(
            epoch=int(epoch),
            model_state=model.state_dict(),
            optim_state=optimizer.state_dict(),
            sched_state=(scheduler.state_dict() if scheduler is not None else None),
            best_epoch=int(best_epoch),
            best_val_f1=float(best_val_f1),
            num_classes=int(C),
            bg_class=int(bg),
            d21_internal=int(d21_idx),
        )
        torch.save(ckpt_obj, last_path)

        if float(va["f1_macro"]) > float(best_val_f1):
            best_val_f1 = float(va["f1_macro"])
            best_epoch = int(epoch)
            ckpt_obj["best_epoch"] = int(best_epoch)
            ckpt_obj["best_val_f1"] = float(best_val_f1)
            torch.save(ckpt_obj, best_path)

        # warn colapso
        if float(va["pred_bg_frac"]) > max(0.95, float(bg_va) + 0.12):
            print(f"[WARN] posible colapso a BG: val pred_bg_frac={va['pred_bg_frac']:.3f} (bg_gt≈{bg_va:.3f})")

        # print consola estilo tuyo
        nb_str = ""
        if neighbors and do_neighbors:
            parts = []
            for name, _ in neighbors:
                parts.append(f"{name}_f1={nb_vals_flat.get(f'{name}_f1', 0.0):.3f}")
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

        if neighbors and do_neighbors:
            parts_full = []
            for name, _ in neighbors:
                parts_full.append(
                    f"{name}_acc={nb_vals_flat.get(f'{name}_acc', 0.0):.3f} "
                    f"{name}_f1={nb_vals_flat.get(f'{name}_f1', 0.0):.3f} "
                    f"{name}_iou={nb_vals_flat.get(f'{name}_iou', 0.0):.3f} "
                    f"{name}_bin_acc_all={nb_vals_flat.get(f'{name}_bin_acc_all', 0.0):.3f}"
                )
            print("[neighbors val] " + " | ".join(parts_full))

        # plots (cada plot_every)
        if int(plot_every) > 0 and (epoch % int(plot_every) == 0):
            plot_train_val_curves(history, out_dir / "plots")

    # fin train: history.json
    save_json(history, hist_json_path)

    total_sec = float(time.time() - t0)
    print(f"[done] Entrenamiento terminado en {_fmt_hms(total_sec)}. best_epoch={best_epoch} best_val={best_val_f1:.4f}")

    return int(best_epoch), float(best_val_f1)


def test_loop_v14(
    model: nn.Module,
    dl_te: DataLoader,
    *,
    device: torch.device,
    out_dir: Path,
    C: int,
    bg: int,
    d21_idx: int,
    loss_fn: nn.Module,
    knn_chunk_size: int,
    neighbors: List[Tuple[str, int]],
    neighbor_eval_split: str,
):
    """
    Carga best.pt si existe, evalúa test y guarda test_metrics.json.
    Neighbors en test si neighbor_eval_split en ("test","both").
    """
    out_dir = Path(out_dir)
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"

    if best_path.exists():
        ck = torch.load(best_path, map_location=device)
        model.load_state_dict(ck["model_state"])
        print(f"[test] Cargado best.pt (epoch={ck.get('epoch','?')})")
    elif last_path.exists():
        ck = torch.load(last_path, map_location=device)
        model.load_state_dict(ck["model_state"])
        print(f"[test] best.pt no existe; cargado last.pt (epoch={ck.get('epoch','?')})")
    else:
        print("[test] No hay checkpoints; usando modelo actual.")

    te = eval_one_epoch(
        model=model,
        loader=dl_te,
        loss_fn=loss_fn,
        device=device,
        C=int(C),
        bg=int(bg),
        d21_idx=int(d21_idx),
        knn_chunk_size=int(knn_chunk_size),
    )

    print(
        f"[test] loss={te['loss']:.4f} "
        f"f1m={te['f1_macro']:.3f} ioum={te['iou_macro']:.3f} "
        f"acc_all={te['acc_all']:.3f} acc_no_bg={te['acc_no_bg']:.3f} | "
        f"d21(cls) acc={te['d21_acc']:.3f} f1={te['d21_f1']:.3f} iou={te['d21_iou']:.3f} | "
        f"d21(bin all) acc={te['d21_bin_acc_all']:.3f} | "
        f"pred_bg_frac(test)={te['pred_bg_frac']:.3f}"
    )

    test_neighbors = {}
    mode = str(neighbor_eval_split).lower()
    if neighbors and mode in ("test", "both"):
        nb = eval_neighbors_on_loader(
            model=model,
            loader=dl_te,
            device=device,
            neighbors=neighbors,
            C=int(C),
            bg=int(bg),
            knn_chunk_size=int(knn_chunk_size),
            use_amp=False,
        )
        test_neighbors = nb

        parts_full = []
        for name, _ in neighbors:
            parts_full.append(
                f"{name}_acc={nb.get(name, {}).get('acc', 0.0):.3f} "
                f"{name}_f1={nb.get(name, {}).get('f1', 0.0):.3f} "
                f"{name}_iou={nb.get(name, {}).get('iou', 0.0):.3f} "
                f"{name}_bin_acc_all={nb.get(name, {}).get('bin_acc_all', 0.0):.3f}"
            )
        print("[test neighbors] " + " | ".join(parts_full))

    out = dict(te)
    out["neighbor_metrics_test"] = test_neighbors
    save_json(out, out_dir / "test_metrics.json")


# ============================================================
# Inference (paper-like, trazable)
# ============================================================

@torch.no_grad()
def run_inference_v14(
    model: nn.Module,
    data_dir: Path,
    split: str,
    device: torch.device,
    out_dir: Path,
    C: int,
    d21_idx: int,
    infer_examples: int,
    knn_chunk_size: int,
    normalize: bool,
    infer_num_workers_cap: int,
):
    out_dir = Path(out_dir)
    inf_root = out_dir / "inference"
    dir_all = inf_root / "inference_all"
    dir_err = inf_root / "inference_errors"
    dir_d21 = inf_root / "inference_d21"

    for d in (dir_all, dir_err, dir_d21):
        d.mkdir(parents=True, exist_ok=True)

    # dataset
    Xp = data_dir / f"X_{split}.npz"
    Yp = data_dir / f"Y_{split}.npz"
    ds = NPZDataset(Xp, Yp, normalize=bool(normalize))

    # index csv trazable
    idx_csv = discover_index_csv(data_dir, split)
    idx_map = read_index_csv(idx_csv)

    # limitar num_workers
    nw = 0
    if int(infer_num_workers_cap) > 0:
        nw = int(infer_num_workers_cap)

    dl = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
    )

    # cargar best.pt si existe
    best_path = out_dir / "best.pt"
    if best_path.exists():
        ck = torch.load(best_path, map_location=device)
        model.load_state_dict(ck["model_state"])
        print(f"[infer] Cargado best.pt (epoch={ck.get('epoch','?')})")

    model.eval()

    manifest_path = inf_root / "inference_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "sample_index",
            "trace_label",
            "png_all",
            "png_error",
            "png_d21",
        ])

        count = 0
        for i, (xyz, y) in enumerate(dl):
            if int(infer_examples) > 0 and count >= int(infer_examples):
                break

            xyz = xyz.to(device, non_blocking=True)
            y   = y.to(device, non_blocking=True)

            logits = model(xyz, knn_chunk_size=int(knn_chunk_size))
            pred = logits.argmax(dim=-1)

            xyz_np = xyz[0].cpu()
            y_np   = y[0].cpu()
            pred_np= pred[0].cpu()

            # trazabilidad
            if idx_map is not None and i in idx_map:
                trace_label = pick_trace_label(idx_map[i])
            else:
                trace_label = f"{split}_{i:04d}"

            tag = _sanitize_tag(trace_label)
            base = f"{split}_{i:04d}_{tag}" if tag else f"{split}_{i:04d}"

            png_all = dir_all / f"{base}.png"
            png_err = dir_err / f"{base}.png"
            png_d21 = dir_d21 / f"{base}.png"

            # ALL (GT vs Pred)
            plot_3d_seg(
                xyz=xyz_np,
                gt=y_np,
                pred=pred_np,
                C=int(C),
                save_path=png_all,
                title=f"{split} idx={i} ALL",
            )

            # ERROR (pred!=gt)
            err_mask = (pred_np != y_np).long()
            plot_3d_seg(
                xyz=xyz_np,
                gt=y_np,
                pred=err_mask,
                C=2,
                save_path=png_err,
                title=f"{split} idx={i} ERROR",
            )

            # D21 binario
            pred_d21 = (pred_np == int(d21_idx)).long()
            gt_d21   = (y_np   == int(d21_idx)).long()
            plot_3d_seg(
                xyz=xyz_np,
                gt=gt_d21,
                pred=pred_d21,
                C=2,
                save_path=png_d21,
                title=f"{split} idx={i} D21",
            )

            w.writerow([
                i,
                trace_label,
                str(png_all.relative_to(out_dir)),
                str(png_err.relative_to(out_dir)),
                str(png_d21.relative_to(out_dir)),
            ])

            count += 1

    print(f"[infer] Generadas {count} inferencias en {inf_root}")


# ============================================================
# MAIN v14 (FIX unpack 3 valores)
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--emb_dims", type=int, default=1024)
    parser.add_argument("--knn_chunk_size", type=int, default=1024)

    parser.add_argument("--bg_class", type=int, default=0)
    parser.add_argument("--bg_weight", type=float, default=0.03)
    parser.add_argument("--d21_internal", type=int, default=8)

    parser.add_argument("--neighbor_teeth", type=str, default="")
    parser.add_argument("--neighbor_eval_split", type=str, default="val")
    parser.add_argument("--neighbor_every", type=int, default=1)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--plot_every", type=int, default=10)

    parser.add_argument("--do_infer", action="store_true")
    parser.add_argument("--infer_examples", type=int, default=12)
    parser.add_argument("--infer_split", type=str, default="test")
    parser.add_argument("--infer_num_workers_cap", type=int, default=2)

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # logging básico en consola
    print(f"[setup] data_dir={data_dir}")
    print(f"[setup] out_dir={out_dir}")
    print(f"[setup] device={device}")

    label_map = load_label_map(data_dir)
    C = infer_num_classes(data_dir, label_map)

    bg = int(args.bg_class)
    d21_idx = int(args.d21_internal)

    neighbors = parse_neighbors(args.neighbor_teeth)

    print(f"[setup] C={C} | bg={bg} | d21={d21_idx}")
    print(f"[setup] neighbors={neighbors}")

    # loaders (FIX v14: solo 3 valores)
    dl_tr, dl_va, dl_te = make_loaders(
        data_dir=data_dir,
        bs=args.batch_size,
        nw=args.num_workers,
        normalize=args.normalize,
    )

    # SANITY bg fraction
    def bg_frac(loader):
        tot = 0
        bgc = 0
        for _, y in loader:
            y = y.reshape(-1)
            tot += y.numel()
            bgc += (y == bg).sum().item()
        return bgc / max(1, tot)

    bg_tr = bg_frac(dl_tr)
    bg_va = bg_frac(dl_va)
    bg_te = bg_frac(dl_te)

    print(f"[SANITY] bg_frac train/val/test = {bg_tr:.4f} {bg_va:.4f} {bg_te:.4f}")
    print(f"[SANITY] baseline acc_all (always-bg) train/val/test = {bg_tr:.4f} {bg_va:.4f} {bg_te:.4f}")

    # model
    model = DGCNNSeg(
        num_classes=C,
        k=args.k,
        emb_dims=args.emb_dims,
        dropout=args.dropout,
    ).to(device)

    # loss con bg_weight
    weight = torch.ones(C, dtype=torch.float32)
    weight[bg] = float(args.bg_weight)
    weight = weight.to(device)

    loss_fn = nn.CrossEntropyLoss(weight=weight)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(args.epochs),
    )

    # guardar run_meta
    save_json({
        "args": vars(args),
        "C": C,
        "bg_class": bg,
        "d21_internal": d21_idx,
        "neighbors": neighbors,
        "bg_frac": {
            "train": bg_tr,
            "val": bg_va,
            "test": bg_te,
        },
    }, out_dir / "run_meta.json")

    # ---- TRAIN ----
    best_epoch, best_val = train_loop_v14(
        model=model,
        dl_tr=dl_tr,
        dl_va=dl_va,
        dl_te=dl_te,
        device=device,
        out_dir=out_dir,
        C=C,
        bg=bg,
        bg_va=bg_va,
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
        neighbor_eval_split=args.neighbor_eval_split,
        neighbor_every=args.neighbor_every,
    )

    # ---- TEST ----
    test_loop_v14(
        model=model,
        dl_te=dl_te,
        device=device,
        out_dir=out_dir,
        C=C,
        bg=bg,
        d21_idx=d21_idx,
        loss_fn=loss_fn,
        knn_chunk_size=args.knn_chunk_size,
        neighbors=neighbors,
        neighbor_eval_split=args.neighbor_eval_split,
    )

    # ---- INFER ----
    if args.do_infer:
        run_inference_v14(
            model=model,
            data_dir=data_dir,
            split=args.infer_split,
            device=device,
            out_dir=out_dir,
            C=C,
            d21_idx=d21_idx,
            infer_examples=args.infer_examples,
            knn_chunk_size=args.knn_chunk_size,
            normalize=args.normalize,
            infer_num_workers_cap=args.infer_num_workers_cap,
        )


if __name__ == "__main__":
    main()