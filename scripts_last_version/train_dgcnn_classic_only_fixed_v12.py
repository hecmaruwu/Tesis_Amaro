#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_dgcnn_classic_only_fixed_v12.py

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

NEW v12 (FIX CRÍTICO trazabilidad):
- El Dataset ahora retorna (xyz, lab, ds_idx) para poder mapear EXACTO contra index_{split}.csv.
- La inferencia usa ds_idx real (NO bidx/contador) tanto para naming como para inference_manifest.csv.
- Se mantiene la misma estructura de outputs; solo cambia la forma de indexar/etiquetar para que sea correcta.
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


# ============================================================
# Utils: seed / json / jsonl / csv
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


# ============================================================
# Logging robusto: run.log + errors.log + console
# ============================================================

def setup_logging(out_dir: Path, name: str = "dgcnn_v12") -> Logger:
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
# Dataset NPZ (flat)  (v12: retorna ds_idx)
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
        idx = int(idx)
        x = np.ascontiguousarray(self.X[idx], dtype=np.float32)
        y = np.ascontiguousarray(self.Y[idx], dtype=np.int64)

        xyz = torch.as_tensor(x, dtype=torch.float32)  # [N,3]
        lab = torch.as_tensor(y, dtype=torch.int64)    # [N]

        if self.normalize:
            xyz = normalize_unit_sphere(xyz)

        # v12: retornar índice REAL del dataset para trazabilidad exacta
        return xyz, lab, idx


def make_loaders(data_dir: Path, bs: int, nw: int, normalize: bool = True):
    """
    Devuelve EXACTAMENTE 3 loaders: train/val/test.
    (Se mantiene igual la interfaz, pero cada batch ahora trae (xyz, lab, idxs).)
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
    split = str(split).strip().lower()
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
    Si hay columna idx/index/row/row_i/i la usa; si no, usa contador.

    Nota: En v12, el dataset entrega ds_idx real. Ese ds_idx debe mapear al row_idx
    de este CSV. Por eso buscamos varias alternativas de columna.
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

        # Detecta columna índice (flexible)
        fieldnames = list(rows[0].keys())
        idx_col = None
        for c in ("row_i", "row", "index", "idx", "i"):
            if c in fieldnames:
                idx_col = c
                break

        # Fallback case-insensitive
        if idx_col is None:
            lowers = {str(c).strip().lower(): c for c in fieldnames}
            for c in ("row_i", "row", "index", "idx", "i"):
                if c in lowers:
                    idx_col = lowers[c]
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
    """
    Elige un label “humano” para naming (sample_name/patient_id/path/etc).
    """
    for k in ("sample_name", "patient", "patient_id", "scan_id", "id", "path", "relpath", "upper_path"):
        if k in d and str(d[k]).strip():
            return str(d[k]).strip()
    for k, v in d.items():
        if str(v).strip():
            return f"{k}={v}"
    return "sample"


def sanitize_filename(s: str, maxlen: int = 100) -> str:
    s = (s or "").strip()
    if not s:
        return "sample"
    s = s.replace(" ", "_")
    s = re.sub(r"[^a-zA-Z0-9_\-\.]+", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if len(s) > int(maxlen):
        s = s[: int(maxlen)]
    return s or "sample"


# ============================================================
# AMP ctx (single source of truth)
# ============================================================

def get_autocast_ctx(device: torch.device, use_amp: bool):
    if bool(use_amp) and device.type == "cuda":
        return torch.amp.autocast("cuda", enabled=True)
    return torch.amp.autocast("cpu", enabled=False)


# ============================================================
# Plot helpers
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
# KNN (chunk-safe) para DGCNN
# ============================================================

def knn_chunked(x: torch.Tensor, k: int, chunk_size: int = 1024):
    """
    x: [B, C, N]
    Retorna idx: [B, N, k]
    Implementación chunk-safe sobre N (queries), sin romper forma final.
    """
    B, C, N = x.shape
    k = int(k)
    assert k > 0

    x_t = x.transpose(1, 2).contiguous()  # [B, N, C]
    idx_out = torch.empty(B, N, k, dtype=torch.long, device=x.device)

    for start in range(0, N, int(chunk_size)):
        end = min(start + int(chunk_size), N)
        q = x_t[:, start:end, :]  # [B, Nc, C]

        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
        qq = torch.sum(q ** 2, dim=-1, keepdim=True)             # [B, Nc, 1]
        xx = torch.sum(x_t ** 2, dim=-1).unsqueeze(1)            # [B, 1, N]
        dist = qq + xx - 2 * torch.bmm(q, x_t.transpose(1, 2))   # [B, Nc, N]

        _, idx_chunk = torch.topk(dist, k=k, dim=-1, largest=False, sorted=False)
        idx_out[:, start:end, :] = idx_chunk

    return idx_out  # [B, N, k]


def get_graph_feature(
    x: torch.Tensor,
    k: int = 20,
    idx: Optional[torch.Tensor] = None,
    chunk_size: int = 1024
):
    """
    x: [B, C, N]
    Devuelve feature: [B, 2C, N, k]
    """
    B, C, N = x.shape

    if idx is None:
        idx = knn_chunked(x, k=int(k), chunk_size=int(chunk_size))  # [B,N,k]

    device = x.device
    idx_base = torch.arange(0, B, device=device).view(-1, 1, 1) * N
    idx = idx + idx_base
    idx = idx.view(-1)

    x_t = x.transpose(1, 2).contiguous()  # [B,N,C]
    feature = x_t.view(B * N, C)[idx, :]  # [B*N*k, C]
    feature = feature.view(B, N, int(k), C)

    x_c = x_t.view(B, N, 1, C).repeat(1, 1, int(k), 1)
    feature = torch.cat((feature - x_c, x_c), dim=3)  # [B,N,k,2C]
    return feature.permute(0, 3, 1, 2).contiguous()   # [B,2C,N,k]

# ============================================================
# EdgeConv Block
# ============================================================

class EdgeConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x: torch.Tensor, k: int, chunk_size: int):
        """
        x: [B,C,N]
        """
        feat = get_graph_feature(x, k=int(k), chunk_size=int(chunk_size))  # [B,2C,N,k]
        feat = self.conv(feat)
        feat = torch.max(feat, dim=-1)[0]  # max over k
        return feat  # [B,out,N]


# ============================================================
# Modelo DGCNN Segmentación
# ============================================================

class DGCNN_Seg(nn.Module):
    def __init__(
        self,
        num_classes: int,
        k: int = 20,
        emb_dims: int = 1024,
        dropout: float = 0.5,
        knn_chunk_size: int = 1024
    ):
        super().__init__()

        self.k = int(k)
        self.emb_dims = int(emb_dims)
        self.knn_chunk_size = int(knn_chunk_size)

        # EdgeConv layers
        self.ec1 = EdgeConvBlock(3, 64)
        self.ec2 = EdgeConvBlock(64, 64)
        self.ec3 = EdgeConvBlock(64, 128)
        self.ec4 = EdgeConvBlock(128, 256)

        # Embedding global
        self.conv_global = nn.Sequential(
            nn.Conv1d(64 + 64 + 128 + 256, self.emb_dims, 1, bias=False),
            nn.BatchNorm1d(self.emb_dims),
            nn.LeakyReLU(0.2)
        )

        # Segment head
        self.conv_seg = nn.Sequential(
            nn.Conv1d(self.emb_dims + 64 + 64 + 128 + 256, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=float(dropout)),

            nn.Conv1d(512, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=float(dropout)),

            nn.Conv1d(256, int(num_classes), 1)
        )

    def forward(self, x: torch.Tensor):
        """
        x: [B,N,3]
        Retorna logits: [B,N,C]
        """
        x = x.permute(0, 2, 1).contiguous()  # [B,3,N]

        x1 = self.ec1(x, self.k, self.knn_chunk_size)           # [B,64,N]
        x2 = self.ec2(x1, self.k, self.knn_chunk_size)          # [B,64,N]
        x3 = self.ec3(x2, self.k, self.knn_chunk_size)          # [B,128,N]
        x4 = self.ec4(x3, self.k, self.knn_chunk_size)          # [B,256,N]

        x_cat = torch.cat((x1, x2, x3, x4), dim=1)              # [B,512,N]
        x_global = self.conv_global(x_cat)                      # [B,emb,N]
        x_global = torch.max(x_global, dim=2, keepdim=True)[0]  # [B,emb,1]
        x_global = x_global.repeat(1, 1, x_cat.size(2))         # [B,emb,N]

        x_feat = torch.cat((x_cat, x_global), dim=1)            # [B,emb+512,N]
        logits = self.conv_seg(x_feat)                          # [B,C,N]

        return logits.permute(0, 2, 1).contiguous()             # [B,N,C]


# ============================================================
# Métricas base (Confusion Matrix + Macro sin BG real)
# ============================================================

@torch.no_grad()
def confusion_matrix(pred: torch.Tensor, target: torch.Tensor, C: int):
    """
    pred, target: [M] (flatten)
    Retorna cm: [C,C]  (fila=gt, col=pred)
    """
    pred = pred.view(-1).long()
    target = target.view(-1).long()

    mask = (target >= 0) & (target < int(C))
    pred = pred[mask]
    target = target[mask]

    idx = target * int(C) + pred
    cm = torch.bincount(idx, minlength=int(C) * int(C))
    cm = cm.view(int(C), int(C))
    return cm


def metrics_from_confmat(cm: torch.Tensor, bg: int):
    """
    Macro métricas excluyendo bg SOLO en macro.
    """
    C = cm.size(0)
    eps = 1e-9

    tp = torch.diag(cm)
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp

    prec = tp / (tp + fp + eps)
    rec  = tp / (tp + fn + eps)
    f1   = 2 * prec * rec / (prec + rec + eps)
    iou  = tp / (tp + fp + fn + eps)

    acc_all = tp.sum() / (cm.sum() + eps)

    mask = torch.ones(C, dtype=torch.bool, device=cm.device)
    if 0 <= int(bg) < C:
        mask[int(bg)] = False

    f1m   = f1[mask].mean()
    ioum  = iou[mask].mean()
    precm = prec[mask].mean()
    recm  = rec[mask].mean()

    correct_no_bg = tp[mask].sum()
    total_no_bg = cm.sum(dim=1)[mask].sum()
    acc_no_bg = correct_no_bg / (total_no_bg + eps)

    return {
        "acc_all": float(acc_all),
        "acc_no_bg": float(acc_no_bg),
        "f1m": float(f1m),
        "ioum": float(ioum),
        "precm": float(precm),
        "recm": float(recm),
    }


# ============================================================
# Métrica binaria (para d21 o cualquier diente)
# ============================================================

def binary_metrics(pred: torch.Tensor, target: torch.Tensor, pos_class: int):
    """
    pred,target: [M]
    """
    eps = 1e-9
    pred = pred.view(-1)
    target = target.view(-1)

    pred_pos = (pred == int(pos_class))
    gt_pos   = (target == int(pos_class))

    tp = (pred_pos & gt_pos).sum().float()
    fp = (pred_pos & ~gt_pos).sum().float()
    fn = (~pred_pos & gt_pos).sum().float()
    tn = (~pred_pos & ~gt_pos).sum().float()

    prec = tp / (tp + fp + eps)
    rec  = tp / (tp + fn + eps)
    f1   = 2 * prec * rec / (prec + rec + eps)
    iou  = tp / (tp + fp + fn + eps)
    acc  = (tp + tn) / (tp + tn + fp + fn + eps)

    return {
        "acc": float(acc),
        "f1": float(f1),
        "iou": float(iou),
    }


# ============================================================
# Evaluación loader completo (VAL / TEST)  (v12: ignora ds_idx)
# ============================================================

@torch.no_grad()
def eval_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    C: int,
    bg: int,
    d21_internal: int,
    use_amp: bool
):
    model.eval()
    cm_total = torch.zeros(int(C), int(C), device=device)

    loss_total = 0.0
    n_batches = 0

    criterion = nn.CrossEntropyLoss()

    pred_all = []
    gt_all = []

    for xyz, lab, _ in loader:   # <-- v12: ignoramos idx aquí
        xyz = xyz.to(device, non_blocking=True)
        lab = lab.to(device, non_blocking=True)

        with get_autocast_ctx(device, use_amp):
            logits = model(xyz)
            loss = criterion(logits.view(-1, int(C)), lab.view(-1))

        pred = torch.argmax(logits, dim=-1)

        cm_total += confusion_matrix(pred.view(-1), lab.view(-1), int(C))

        pred_all.append(pred.detach().cpu())
        gt_all.append(lab.detach().cpu())

        loss_total += float(loss.item())
        n_batches += 1

    base = metrics_from_confmat(cm_total, bg=int(bg))

    pred_all = torch.cat(pred_all).view(-1)
    gt_all = torch.cat(gt_all).view(-1)

    d21m = binary_metrics(pred_all, gt_all, pos_class=int(d21_internal))

    base.update({
        "loss": loss_total / max(n_batches, 1),
        "d21_acc": d21m["acc"],
        "d21_f1": d21m["f1"],
        "d21_iou": d21m["iou"],
    })

    return base


# ============================================================
# Métricas vecinos (lista arbitraria)  (v12: ignora ds_idx)
# ============================================================

@torch.no_grad()
def eval_neighbors_on_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    neighbor_list: List[Tuple[str, int]],
    bg: int,
    use_amp: bool,
    C: int
):
    if not neighbor_list:
        return {}

    model.eval()

    pred_all = []
    gt_all = []

    for xyz, lab, _ in loader:  # <-- v12: ignoramos idx
        xyz = xyz.to(device, non_blocking=True)
        with get_autocast_ctx(device, use_amp):
            logits = model(xyz)
        pred = torch.argmax(logits, dim=-1).detach().cpu()
        pred_all.append(pred)
        gt_all.append(lab)

    pred_all = torch.cat(pred_all).view(-1)
    gt_all = torch.cat(gt_all).view(-1)

    out = {}
    for name, idx in neighbor_list:
        m = binary_metrics(pred_all, gt_all, pos_class=int(idx))
        out[name] = {
            "acc": m["acc"],
            "f1": m["f1"],
            "iou": m["iou"],
        }

    return out

# ============================================================
# TRAIN LOOP + CHECKPOINTS + HISTORY/CSV + PLOTS  (v12)
# ============================================================

def make_weighted_ce(C: int, bg: int, bg_weight: float, device: torch.device) -> nn.Module:
    """
    BG incluido en la loss (NO ignore_index), pero downweight al bg.
    """
    w = torch.ones(int(C), device=device, dtype=torch.float32)
    if 0 <= int(bg) < int(C):
        w[int(bg)] = float(bg_weight)
    return nn.CrossEntropyLoss(weight=w)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    C: int,
    bg: int,
    use_amp: bool,
    grad_clip: float = 0.0,
):
    """
    Entrena 1 epoch y retorna:
      dict(loss, acc_all, acc_no_bg, f1m, ioum, precm, recm)
    """
    model.train()
    scaler = train_one_epoch.scaler  # type: ignore
    if bool(use_amp) and device.type == "cuda" and scaler is None:
        scaler = torch.amp.GradScaler("cuda")
        train_one_epoch.scaler = scaler  # type: ignore

    cm_total = torch.zeros(int(C), int(C), device=device)
    loss_sum = 0.0
    nb = 0

    for xyz, lab, _ in loader:  # v12: ignoramos idx aquí
        xyz = xyz.to(device, non_blocking=True)
        lab = lab.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with get_autocast_ctx(device, use_amp):
            logits = model(xyz)  # [B,N,C]
            loss = criterion(logits.view(-1, int(C)), lab.view(-1))

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

        with torch.no_grad():
            pred = torch.argmax(logits, dim=-1)
            cm_total += confusion_matrix(pred.view(-1), lab.view(-1), int(C))

        loss_sum += float(loss.item())
        nb += 1

    base = metrics_from_confmat(cm_total, bg=int(bg))
    base["loss"] = loss_sum / max(nb, 1)
    return base


train_one_epoch.scaler = None  # type: ignore


# ============================================================
# Plot helpers (Train vs Val; SIN línea test)
# ============================================================

def plot_train_val_curve(
    out_path: Path,
    epochs: int,
    train_vals: List[float],
    val_vals: List[float],
    ylabel: str,
    best_epoch: Optional[int] = None
):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    xs = list(range(1, epochs + 1))
    plt.figure(figsize=(6, 4))
    plt.plot(xs, train_vals, label="train")
    plt.plot(xs, val_vals, label="val")
    if best_epoch is not None and best_epoch > 0:
        plt.axvline(best_epoch, linestyle=":", label=f"best={best_epoch}")
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_all_metrics_paperlike(
    history: Dict[str, List[float]],
    out_dir: Path,
    best_epoch: int
):
    plots_dir = Path(out_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    epochs = len(history["train_loss"])

    plot_train_val_curve(
        plots_dir / "loss.png",
        epochs,
        history["train_loss"],
        history["val_loss"],
        ylabel="loss",
        best_epoch=best_epoch
    )

    plot_train_val_curve(
        plots_dir / "f1m.png",
        epochs,
        history["train_f1m"],
        history["val_f1m"],
        ylabel="f1_macro",
        best_epoch=best_epoch
    )

    plot_train_val_curve(
        plots_dir / "ioum.png",
        epochs,
        history["train_ioum"],
        history["val_ioum"],
        ylabel="iou_macro",
        best_epoch=best_epoch
    )

    plot_train_val_curve(
        plots_dir / "acc_all.png",
        epochs,
        history["train_acc_all"],
        history["val_acc_all"],
        ylabel="acc_all",
        best_epoch=best_epoch
    )

    plot_train_val_curve(
        plots_dir / "acc_no_bg.png",
        epochs,
        history["train_acc_no_bg"],
        history["val_acc_no_bg"],
        ylabel="acc_no_bg",
        best_epoch=best_epoch
    )

    # d21 solo val
    plt.figure(figsize=(6,4))
    xs = list(range(1, epochs + 1))
    plt.plot(xs, history["val_d21_f1"], label="val_d21_f1")
    if best_epoch > 0:
        plt.axvline(best_epoch, linestyle=":", label=f"best={best_epoch}")
    plt.xlabel("epoch")
    plt.ylabel("d21_f1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "d21_f1.png", dpi=200)
    plt.close()


# ============================================================
# Train loop completo paper-like
# ============================================================

def train_loop_paperlike(
    model: nn.Module,
    dl_tr: DataLoader,
    dl_va: DataLoader,
    dl_te: DataLoader,
    out_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
    C: int,
    bg: int,
    d21_internal: int,
    neighbors: List[Tuple[str, int]],
    logger,
):

    out_dir = Path(out_dir)
    metrics_csv = out_dir / "metrics_epoch.csv"
    hist_json = out_dir / "history.json"
    hist_jsonl = out_dir / "history_epoch.jsonl"
    ckpt_best = out_dir / "best.pt"
    ckpt_last = out_dir / "last.pt"

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay)
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim,
        T_max=int(args.epochs),
        eta_min=max(1e-6, float(args.lr) * 0.02)
    )

    criterion = make_weighted_ce(int(C), int(bg), float(args.bg_weight), device=device)

    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [],
        "train_f1m": [], "val_f1m": [],
        "train_ioum": [], "val_ioum": [],
        "train_acc_all": [], "val_acc_all": [],
        "train_acc_no_bg": [], "val_acc_no_bg": [],
        "val_d21_acc": [], "val_d21_f1": [], "val_d21_iou": [],
    }

    for name, _ in neighbors:
        history[f"val_{name}_acc"] = []
        history[f"val_{name}_f1"] = []
        history[f"val_{name}_iou"] = []

    best_val = -1e9
    best_epoch = -1

    for epoch in range(1, int(args.epochs) + 1):

        # -------- TRAIN --------
        tr = train_one_epoch(
            model=model,
            loader=dl_tr,
            optimizer=optim,
            criterion=criterion,
            device=device,
            C=int(C),
            bg=int(bg),
            use_amp=bool(args.use_amp),
            grad_clip=float(args.grad_clip),
        )

        # -------- VAL --------
        va = eval_loader(
            model=model,
            loader=dl_va,
            device=device,
            C=int(C),
            bg=int(bg),
            d21_internal=int(d21_internal),
            use_amp=bool(args.use_amp),
        )

        # -------- NEIGHBORS --------
        neigh_val_flat = {}
        if neighbors:
            neigh_dict = eval_neighbors_on_loader(
                model=model,
                loader=dl_va,
                device=device,
                neighbor_list=neighbors,
                bg=int(bg),
                use_amp=bool(args.use_amp),
                C=int(C),
            )
            for name, _ in neighbors:
                d = neigh_dict.get(name, {})
                neigh_val_flat[f"val_{name}_acc"] = float(d.get("acc", 0.0))
                neigh_val_flat[f"val_{name}_f1"]  = float(d.get("f1", 0.0))
                neigh_val_flat[f"val_{name}_iou"] = float(d.get("iou", 0.0))

        scheduler.step()
        lr = get_lr(optim)

        # -------- HISTORY --------
        history["train_loss"].append(float(tr["loss"]))
        history["train_f1m"].append(float(tr["f1m"]))
        history["train_ioum"].append(float(tr["ioum"]))
        history["train_acc_all"].append(float(tr["acc_all"]))
        history["train_acc_no_bg"].append(float(tr["acc_no_bg"]))

        history["val_loss"].append(float(va["loss"]))
        history["val_f1m"].append(float(va["f1m"]))
        history["val_ioum"].append(float(va["ioum"]))
        history["val_acc_all"].append(float(va["acc_all"]))
        history["val_acc_no_bg"].append(float(va["acc_no_bg"]))
        history["val_d21_acc"].append(float(va["d21_acc"]))
        history["val_d21_f1"].append(float(va["d21_f1"]))
        history["val_d21_iou"].append(float(va["d21_iou"]))

        for name, _ in neighbors:
            history[f"val_{name}_acc"].append(float(neigh_val_flat.get(f"val_{name}_acc", 0.0)))
            history[f"val_{name}_f1"].append(float(neigh_val_flat.get(f"val_{name}_f1", 0.0)))
            history[f"val_{name}_iou"].append(float(neigh_val_flat.get(f"val_{name}_iou", 0.0)))

        # -------- PRINT --------
        neigh_str = ""
        if neighbors:
            parts = []
            for name, _ in neighbors:
                parts.append(
                    f"{name}: acc={neigh_val_flat.get(f'val_{name}_acc',0.0):.3f} "
                    f"f1={neigh_val_flat.get(f'val_{name}_f1',0.0):.3f} "
                    f"iou={neigh_val_flat.get(f'val_{name}_iou',0.0):.3f}"
                )
            neigh_str = " | [neighbors val] " + " | ".join(parts)

        logger.info(
            f"[{epoch:03d}/{args.epochs}] "
            f"train loss={tr['loss']:.4f} f1m={tr['f1m']:.3f} ioum={tr['ioum']:.3f} "
            f"| val loss={va['loss']:.4f} f1m={va['f1m']:.3f} ioum={va['ioum']:.3f} "
            f"| d21 f1={va['d21_f1']:.3f} iou={va['d21_iou']:.3f} "
            f"| lr={lr:.2e}"
            f"{neigh_str}"
        )

        # -------- SAVE CKPTS --------
        save_checkpoint(
            ckpt_last,
            epoch=int(epoch),
            model=model,
            optim=optim,
            scheduler=scheduler,
            args=args,
            best_epoch=int(best_epoch),
            best_val=float(best_val),
            C=int(C),
            bg=int(bg),
            d21_idx=int(d21_internal),
        )

        if float(va["ioum"]) > best_val:
            best_val = float(va["ioum"])
            best_epoch = int(epoch)
            save_checkpoint(
                ckpt_best,
                epoch=int(epoch),
                model=model,
                optim=optim,
                scheduler=scheduler,
                args=args,
                best_epoch=int(best_epoch),
                best_val=float(best_val),
                C=int(C),
                bg=int(bg),
                d21_idx=int(d21_internal),
            )

        # -------- SAVE HISTORY FILES --------
        safe_write_json(hist_json, history)
        append_jsonl(hist_jsonl, {
            "epoch": epoch,
            "train": tr,
            "val": va,
            "neighbors_val": neigh_val_flat
        })

        append_metrics_csv(metrics_csv, {
            "epoch": epoch,
            "lr": lr,
            **tr,
            **va,
            **neigh_val_flat
        })

    logger.info(f"[done] best_epoch={best_epoch} best_val(ioum)={best_val:.4f}")

    # -------- TEST --------
    if ckpt_best.exists():
        ck = load_checkpoint(ckpt_best, model=model, map_location="cpu")
        if ck is not None:
            logger.info(f"[test] loaded best.pt epoch={ck.get('epoch')}")

    te = eval_loader(
        model=model,
        loader=dl_te,
        device=device,
        C=int(C),
        bg=int(bg),
        d21_internal=int(d21_internal),
        use_amp=bool(args.use_amp),
    )

    # -------- PLOTS --------
    plot_all_metrics_paperlike(history, out_dir, best_epoch)

    return {
        "best_epoch": int(best_epoch),
        "best_val": float(best_val),
        "test": te,
        "history": history,
    }

# ============================================================
# PARTE 5/5 — MAIN + META + TEST + INFERENCIA PAPER-LIKE (v12)
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--num_workers", type=int, default=6)

    ap.add_argument("--device", default="cuda")
    ap.add_argument("--use_amp", action="store_true")
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--emb_dims", type=int, default=1024)
    ap.add_argument("--knn_chunk_size", type=int, default=1024)

    ap.add_argument("--bg_class", type=int, default=0)
    ap.add_argument("--bg_weight", type=float, default=0.03)
    ap.add_argument("--d21_internal", type=int, default=8)
    ap.add_argument("--neighbor_teeth", default="")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--normalize", action="store_true")

    ap.add_argument("--do_infer", action="store_true")
    ap.add_argument("--infer_examples", type=int, default=12)
    ap.add_argument("--infer_split", default="test", choices=["train","val","test"])

    return ap.parse_args()


# ============================================================
# INFERENCIA PAPER-LIKE (v12: USA ds_idx REAL)
# ============================================================

@torch.no_grad()
def infer_and_save_examples_paperlike(
    model,
    loader,
    device,
    out_dir,
    split,
    C,
    d21_idx,
    max_examples,
    index_map=None
):
    model.eval()

    base = Path(out_dir) / "inference"
    all_dir = base / "inference_all"
    err_dir = base / "inference_errors"
    d21_dir = base / "inference_d21"

    for d in [all_dir, err_dir, d21_dir]:
        d.mkdir(parents=True, exist_ok=True)

    manifest_csv = base / "inference_manifest.csv"
    manifest_exists = manifest_csv.exists()

    colors = class_colors(C)
    saved = 0

    with manifest_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["name","split","index","png_all","png_errors","png_d21"]
        )
        if not manifest_exists:
            writer.writeheader()

        for xyz, lab, idxs in loader:
            if saved >= int(max_examples):
                break

            xyz = xyz.to(device)
            logits = model(xyz)
            pred = torch.argmax(logits, dim=-1).cpu()

            B = xyz.shape[0]

            for b in range(B):
                if saved >= int(max_examples):
                    break

                ds_idx = int(idxs[b])  # <<< FIX CLAVE v12

                xyz_np = xyz[b].cpu().numpy()
                gt_np = lab[b].numpy()
                pr_np = pred[b].numpy()

                name = f"{split}_{ds_idx:05d}"

                if index_map and ds_idx in index_map:
                    trace_label = sanitize_filename(pick_trace_label(index_map[ds_idx]))
                    if trace_label:
                        name = trace_label

                # ---------- ALL ----------
                fig = plt.figure(figsize=(10,5))
                ax1 = fig.add_subplot(121, projection="3d")
                ax2 = fig.add_subplot(122, projection="3d")

                for c in range(int(C)):
                    m = (gt_np == c)
                    if m.any():
                        ax1.scatter(xyz_np[m,0], xyz_np[m,1], xyz_np[m,2], s=1, color=colors[c])
                ax1.set_title("GT")

                for c in range(int(C)):
                    m = (pr_np == c)
                    if m.any():
                        ax2.scatter(xyz_np[m,0], xyz_np[m,1], xyz_np[m,2], s=1, color=colors[c])
                ax2.set_title("PRED")

                plt.tight_layout()
                png_all = f"{name}.png"
                fig.savefig(all_dir / png_all, dpi=200)
                plt.close(fig)

                # ---------- ERRORS ----------
                fig = plt.figure(figsize=(5,5))
                ax = fig.add_subplot(111, projection="3d")
                correct = (gt_np == pr_np)
                ax.scatter(xyz_np[correct,0], xyz_np[correct,1], xyz_np[correct,2], s=1, color="gray")
                ax.scatter(xyz_np[~correct,0], xyz_np[~correct,1], xyz_np[~correct,2], s=3, color="red")
                ax.set_title("Errors (red)")
                plt.tight_layout()
                png_err = f"{name}.png"
                fig.savefig(err_dir / png_err, dpi=200)
                plt.close(fig)

                # ---------- D21 ----------
                fig = plt.figure(figsize=(5,5))
                ax = fig.add_subplot(111, projection="3d")
                tp = (pr_np == d21_idx) & (gt_np == d21_idx)
                fp = (pr_np == d21_idx) & (gt_np != d21_idx)
                fn = (pr_np != d21_idx) & (gt_np == d21_idx)

                ax.scatter(xyz_np[tp,0], xyz_np[tp,1], xyz_np[tp,2], s=2, color="green")
                ax.scatter(xyz_np[fp|fn,0], xyz_np[fp|fn,1], xyz_np[fp|fn,2], s=3, color="red")
                ax.set_title("D21 focus")
                plt.tight_layout()
                png_d21 = f"{name}.png"
                fig.savefig(d21_dir / png_d21, dpi=200)
                plt.close(fig)

                writer.writerow({
                    "name": name,
                    "split": split,
                    "index": ds_idx,  # <<< índice REAL del dataset
                    "png_all": png_all,
                    "png_errors": png_err,
                    "png_d21": png_d21,
                })

                saved += 1


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    seed_all(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(out_dir, name="dgcnn_v12")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    label_map = load_label_map(data_dir)
    C = infer_num_classes(data_dir, label_map)
    bg = int(args.bg_class)

    neighbors = parse_neighbors(args.neighbor_teeth)

    logger.info(f"[setup] data_dir={data_dir}")
    logger.info(f"[setup] out_dir={out_dir}")
    logger.info(f"[setup] device={device} | C={C} | bg={bg} | d21={args.d21_internal}")
    logger.info(f"[setup] neighbors={neighbors}")

    dl_tr, dl_va, dl_te = make_loaders(
        data_dir,
        bs=args.batch_size,
        nw=args.num_workers,
        normalize=args.normalize
    )

    model = DGCNN_Seg(
        num_classes=C,
        k=args.k,
        emb_dims=args.emb_dims,
        dropout=args.dropout,
        knn_chunk_size=args.knn_chunk_size
    ).to(device)

    # ---------- run_meta ----------
    run_meta = {
        "timestamp": now_iso(),
        "script": "train_dgcnn_classic_only_fixed_v12.py",
        "args": vars(args),
        "extra": {
            "num_classes": C,
            "bg": bg,
            "d21_internal": args.d21_internal,
            "neighbors": neighbors,
        }
    }
    safe_write_json(out_dir / "run_meta.json", run_meta)

    # ---------- TRAIN LOOP ----------
    results = train_loop_paperlike(
        model=model,
        dl_tr=dl_tr,
        dl_va=dl_va,
        dl_te=dl_te,
        out_dir=out_dir,
        args=args,
        device=device,
        C=C,
        bg=bg,
        d21_internal=args.d21_internal,
        neighbors=neighbors,
        logger=logger,
    )

    # ---------- test_metrics.json ----------
    test_metrics = {
        "best_epoch": results["best_epoch"],
        "best_val_ioum": results["best_val"],
        "test": results["test"],
    }
    safe_write_json(out_dir / "test_metrics.json", test_metrics)

    # ---------- INFERENCIA ----------
    if args.do_infer:
        split = args.infer_split.lower()
        loader = {"train": dl_tr, "val": dl_va, "test": dl_te}[split]

        index_csv = discover_index_csv(data_dir, split)
        index_map = read_index_csv(index_csv)

        infer_and_save_examples_paperlike(
            model=model,
            loader=loader,
            device=device,
            out_dir=out_dir,
            split=split,
            C=C,
            d21_idx=args.d21_internal,
            max_examples=args.infer_examples,
            index_map=index_map
        )

    logger.info("[done] everything finished successfully.")


if __name__ == "__main__":
    main()