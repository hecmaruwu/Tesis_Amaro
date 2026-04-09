#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pointnet_classic_final_v7.py

PointNet clásico – Segmentación multiclase dental 3D (OPCIÓN A – PAPER CORRECTA)

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
✅ Añade soporte REAL de “neighbor teeth metrics” (igual filosofía que DGCNN v6):
   - Flag --neighbor_teeth "d11:1,d22:9,..." (lista arbitraria nombre:idx)
   - Loggea métricas binarias (acc/f1/iou + bin_acc_all) para cada vecino
   - Integra en CSV por epoch, history.json, test_metrics.json
   - NO se elimina ningún contenido existente; solo se agrega esta capacidad

(NEW v6):
❌ Se había dejado un bloque DEBUG con sys.exit(0) dentro del loop de entrenamiento.
✅ Eliminado completamente.

(NEW v7):
✅ Misma capa de outputs/trazabilidad que pointnettransformer_classic_final_v5.py
✅ Añade history_epoch.jsonl
✅ Inferencia reestructurada en:
   - inference/inference_all
   - inference/inference_errors
   - inference/inference_d21
✅ Guarda inference_manifest.csv con rutas relativas de PNGs
✅ Añade --infer_num_workers para inferencia robusta
✅ Mantiene PointNet backbone, métricas paper-correct y filosofía original del script

Dataset esperado:
  data_dir/X_train.npz, Y_train.npz, X_val.npz, Y_val.npz, X_test.npz, Y_test.npz
  X: [B,N,3], Y: [B,N] con clases internas 0..C-1 (0=bg)

Ejemplo:
python3 pointnet_classic_final_v7.py \
  --data_dir .../upper_only_surf_global_excl_wisdom_seed42_aug2 \
  --out_dir  .../outputs/pointnet_classic/run1 \
  --epochs 120 --batch_size 16 --lr 2e-4 --weight_decay 1e-4 --dropout 0.5 \
  --num_workers 6 --device cuda --d21_internal 8 \
  --bg_weight 0.03 --grad_clip 1.0 --use_amp \
  --neighbor_teeth "d11:1,d22:9" \
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
from collections import defaultdict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# SEED / IO / LOG
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


def append_jsonl(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def ensure_dir(p: Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_line(msg: str, log_path: Optional[Path] = None, also_print: bool = True):
    line = f"[{_now_str()}] {msg}"
    if also_print:
        print(line, flush=True)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


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

    return_index:
      - False: (xyz, y)
      - True : (xyz, y, idx_local) -> para infer trazable (row dentro del split)
    """
    def __init__(self, Xp: Path, Yp: Path, normalize: bool = True, return_index: bool = False):
        self.X = np.load(Xp)["X"].astype(np.float32)  # [B,N,3]
        self.Y = np.load(Yp)["Y"].astype(np.int64)    # [B,N]
        assert self.X.ndim == 3 and self.X.shape[-1] == 3, f"X shape inesperada: {self.X.shape}"
        assert self.Y.ndim == 2, f"Y shape inesperada: {self.Y.shape}"
        assert self.X.shape[0] == self.Y.shape[0], "B mismatch"
        assert self.X.shape[1] == self.Y.shape[1], "N mismatch"
        self.normalize = bool(normalize)
        self.return_index = bool(return_index)

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, i: int):
        # FIX robusto: evita torch.from_numpy en workers (crash raro "expected np.ndarray")
        i = int(i)
        x = np.ascontiguousarray(self.X[i], dtype=np.float32)  # [N,3]
        y = np.ascontiguousarray(self.Y[i], dtype=np.int64)    # [N]
        xyz = torch.as_tensor(x, dtype=torch.float32)
        lab = torch.as_tensor(y, dtype=torch.int64)
        if self.normalize:
            xyz = normalize_unit_sphere(xyz)
        if self.return_index:
            return xyz, lab, torch.tensor(i, dtype=torch.int64)
        return xyz, lab


def make_loaders(data_dir: Path, bs: int, nw: int, normalize: bool = True):
    ds_tr = NPZDataset(data_dir / "X_train.npz", data_dir / "Y_train.npz", normalize=normalize, return_index=False)
    ds_va = NPZDataset(data_dir / "X_val.npz",   data_dir / "Y_val.npz",   normalize=normalize, return_index=False)
    ds_te = NPZDataset(data_dir / "X_test.npz",  data_dir / "Y_test.npz",  normalize=normalize, return_index=False)

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


def make_infer_loader(data_dir: Path, split: str, bs: int, nw: int, normalize: bool = True) -> DataLoader:
    """
    Loader para infer trazable:
    - shuffle=False
    - dataset return_index=True -> idx_local real del split (row_i)

    Nota:
    - Para robustez en inferencia/plots en algunos clusters,
      conviene usar menos workers que en train.
    """
    split = str(split).lower().strip()
    if split == "train":
        ds = NPZDataset(data_dir / "X_train.npz", data_dir / "Y_train.npz", normalize=normalize, return_index=True)
    elif split == "val":
        ds = NPZDataset(data_dir / "X_val.npz",   data_dir / "Y_val.npz",   normalize=normalize, return_index=True)
    elif split == "test":
        ds = NPZDataset(data_dir / "X_test.npz",  data_dir / "Y_test.npz",  normalize=normalize, return_index=True)
    else:
        raise ValueError(f"split inválido: {split}")

    nw = int(nw)
    common = dict(
        batch_size=int(bs),
        num_workers=nw,
        pin_memory=True,
        persistent_workers=(nw > 0),
        drop_last=False,
    )
    if nw > 0:
        common["prefetch_factor"] = 2

    return DataLoader(ds, shuffle=False, **common)


# ============================================================
# POINTNET (paper-like)
# ============================================================
class STN3d(nn.Module):
    def __init__(self, k: int = 3):
        super().__init__()
        self.k = int(k)
        self.conv1, self.bn1 = nn.Conv1d(self.k, 64, 1), nn.BatchNorm1d(64)
        self.conv2, self.bn2 = nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128)
        self.conv3, self.bn3 = nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024)
        self.fc1, self.bn4 = nn.Linear(1024, 512), nn.BatchNorm1d(512)
        self.fc2, self.bn5 = nn.Linear(512, 256), nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, self.k * self.k)

    def forward(self, x):
        # x: [B,k,N]
        B = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2)[0]  # [B,1024]
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x).view(B, self.k, self.k)
        iden = torch.eye(self.k, device=x.device, dtype=x.dtype).unsqueeze(0).repeat(B, 1, 1)
        return x + iden


class PointNetSeg(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.stn = STN3d(k=3)

        self.conv1, self.bn1 = nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64)
        self.conv2, self.bn2 = nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128)
        self.conv3, self.bn3 = nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024)

        # concat global(1024) + local(128) = 1152
        self.fconv1, self.fbn1 = nn.Conv1d(1152, 512, 1), nn.BatchNorm1d(512)
        self.fconv2, self.fbn2 = nn.Conv1d(512, 256, 1), nn.BatchNorm1d(256)
        self.drop = nn.Dropout(float(dropout))
        self.fconv3 = nn.Conv1d(256, int(num_classes), 1)

    def forward(self, xyz):
        # xyz: [B,N,3]
        B, N, _ = xyz.shape
        x = xyz.transpose(2, 1).contiguous()    # [B,3,N]
        T = self.stn(x)                         # [B,3,3]
        x = torch.bmm(T, x)                     # [B,3,N]

        x1 = F.relu(self.bn1(self.conv1(x)))    # [B,64,N]
        x2 = F.relu(self.bn2(self.conv2(x1)))   # [B,128,N]
        x3 = F.relu(self.bn3(self.conv3(x2)))   # [B,1024,N]

        g = torch.max(x3, 2, keepdim=True)[0].repeat(1, 1, N)  # [B,1024,N]
        cat = torch.cat([g, x2], dim=1)                        # [B,1152,N]

        x = F.relu(self.fbn1(self.fconv1(cat)))
        x = F.relu(self.fbn2(self.fconv2(x)))
        x = self.drop(x)
        logits = self.fconv3(x).transpose(2, 1).contiguous()   # [B,N,C]
        return logits


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


@torch.no_grad()
def _tooth_metrics_binary(pred: torch.Tensor, gt: torch.Tensor, tooth_idx: int, bg: int = 0) -> Dict[str, float]:
    acc, f1, iou = d21_metrics_binary(pred, gt, d21_idx=int(tooth_idx), bg=int(bg), include_bg=False)
    acc_all, f1_all, iou_all = d21_metrics_binary(pred, gt, d21_idx=int(tooth_idx), bg=int(bg), include_bg=True)
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


def _safe_savefig(path: Path, errors_log: Optional[Path] = None):
    """
    Guardado seguro: si matplotlib falla, lo registramos y seguimos.
    """
    try:
        plt.savefig(path, bbox_inches="tight")
    except Exception as e:
        if errors_log is not None:
            log_line(f"[WARN] matplotlib savefig falló en {path}: {repr(e)}", errors_log, also_print=True)
    finally:
        try:
            plt.close()
        except Exception:
            pass


def plot_pointcloud_all_classes(
    xyz: np.ndarray,
    y_gt: np.ndarray,
    y_pr: np.ndarray,
    out_png: Path,
    C: int,
    title: str = "",
    s: float = 1.0,
    errors_log: Optional[Path] = None,
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
    _safe_savefig(out_png, errors_log=errors_log)


def plot_errors(
    xyz: np.ndarray,
    y_gt: np.ndarray,
    y_pr: np.ndarray,
    out_png: Path,
    bg: int = 0,
    title: str = "",
    s: float = 1.0,
    errors_log: Optional[Path] = None,
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
    _safe_savefig(out_png, errors_log=errors_log)


def plot_d21_focus(
    xyz: np.ndarray,
    y_gt: np.ndarray,
    y_pr: np.ndarray,
    out_png: Path,
    d21_idx: int,
    bg: int = 0,
    title: str = "",
    s: float = 1.2,
    errors_log: Optional[Path] = None,
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
    _safe_savefig(out_png, errors_log=errors_log)


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


# ============================================================
# HELPERS DE TRAZABILIDAD (index_*.csv)
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


# ============================================================
# TRAIN / EVAL (v4 + vecinos integrados)
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


@torch.no_grad()
def _neighbors_metrics_from_logits(
    logits: torch.Tensor,
    y: torch.Tensor,
    neighbor_list: List[Tuple[str, int]],
    bg: int
) -> Dict[str, float]:
    if not neighbor_list:
        return {}

    pred = logits.argmax(dim=-1)
    out: Dict[str, float] = {}

    for name, idx in neighbor_list:
        m = _tooth_metrics_binary(pred, y, tooth_idx=int(idx), bg=int(bg))
        out[f"{name}_acc"] = float(m["acc"])
        out[f"{name}_f1"] = float(m["f1"])
        out[f"{name}_iou"] = float(m["iou"])
        out[f"{name}_bin_acc_all"] = float(m["bin_acc_all"])
        out[f"{name}_bin_f1_all"] = float(m["bin_f1_all"])
        out[f"{name}_bin_iou_all"] = float(m["bin_iou_all"])

    return out


def _format_neighbor_console(metrics: Dict[str, float], neighbor_list: List[Tuple[str, int]]) -> str:
    """
    Formato estilo PointNet/DGCNN para imprimir vecinos:
      nb[d11] acc=... f1=... iou=...
    """
    if not neighbor_list:
        return ""
    chunks = []
    for name, _ in neighbor_list:
        acc = metrics.get(f"{name}_acc", None)
        f1  = metrics.get(f"{name}_f1", None)
        iou = metrics.get(f"{name}_iou", None)
        if acc is None and f1 is None and iou is None:
            continue
        s = f"nb[{name}] acc={acc:.3f} f1={f1:.3f} iou={iou:.3f}"
        chunks.append(s)
    return " | " + " ".join(chunks) if chunks else ""


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
    neighbor_list: Optional[List[Tuple[str, int]]] = None,
    train_metrics_eval: bool = True,
) -> Dict[str, float]:
    """
    Si train=True:
      - hace update con model.train()
      - y calcula métricas comparables con un segundo forward en eval() si train_metrics_eval=True
    Si train=False:
      - eval normal
    """
    neighbor_list = neighbor_list or []

    scaler = run_epoch.scaler
    if use_amp and device.type == "cuda" and scaler is None:
        scaler = torch.amp.GradScaler("cuda")
        run_epoch.scaler = scaler

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

    nb_sums: Dict[str, float] = {}
    if neighbor_list:
        for name, _ in neighbor_list:
            for k in ("acc", "f1", "iou", "bin_acc_all"):
                nb_sums[f"{name}_{k}"] = 0.0

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

            with torch.amp.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
                logits_train = model(xyz)
                loss = loss_fn(logits_train.reshape(-1, C), y.reshape(-1))

            if use_amp and device.type == "cuda":
                scaler.scale(loss).backward()
                if grad_clip and float(grad_clip) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip and float(grad_clip) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                optimizer.step()

            if train_metrics_eval:
                model.eval()
                with torch.no_grad():
                    with torch.amp.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
                        logits_eval = model(xyz)
                logits_for_metrics = logits_eval
            else:
                logits_for_metrics = logits_train.detach()

            metrics = _compute_metrics_from_logits(
                logits_for_metrics, y, C=C, d21_idx=d21_idx, bg=bg
            )
            nb_metrics = _neighbors_metrics_from_logits(
                logits_for_metrics, y, neighbor_list=neighbor_list, bg=bg
            )

        else:
            with torch.no_grad():
                with torch.amp.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
                    logits = model(xyz)
                    loss = loss_fn(logits.reshape(-1, C), y.reshape(-1))

            metrics = _compute_metrics_from_logits(
                logits, y, C=C, d21_idx=d21_idx, bg=bg
            )
            nb_metrics = _neighbors_metrics_from_logits(
                logits, y, neighbor_list=neighbor_list, bg=bg
            )

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

        if neighbor_list:
            for name, _ in neighbor_list:
                nb_sums[f"{name}_acc"] += float(nb_metrics.get(f"{name}_acc", 0.0))
                nb_sums[f"{name}_f1"] += float(nb_metrics.get(f"{name}_f1", 0.0))
                nb_sums[f"{name}_iou"] += float(nb_metrics.get(f"{name}_iou", 0.0))
                nb_sums[f"{name}_bin_acc_all"] += float(nb_metrics.get(f"{name}_bin_acc_all", 0.0))

        n_batches += 1

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

    if neighbor_list:
        for k, v in nb_sums.items():
            out[k] = float(v) / n

    return out


run_epoch.scaler = None  # type: ignore

# ============================================================
# ARGPARSE
# ============================================================
def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.5)

    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--infer_num_workers", type=int, default=None)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    ap.add_argument("--d21_internal", type=int, required=True)

    ap.add_argument("--bg_index", type=int, default=0)
    ap.add_argument("--bg_weight", type=float, default=0.10)

    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--use_amp", action="store_true")

    # normalización: se controla con flag inverso
    ap.add_argument("--no_normalize", action="store_true")

    # trazabilidad: opcional forzar index CSV
    ap.add_argument("--index_csv", type=str, default=None)

    ap.add_argument("--do_infer", action="store_true")
    ap.add_argument("--infer_examples", type=int, default=12)
    ap.add_argument("--infer_split", type=str, default="test", choices=["test", "val", "train"])

    # v4
    ap.add_argument("--train_metrics_eval", action="store_true",
                    help="Si está activo, calcula métricas de TRAIN con model.eval() (sin dropout/BN train), "
                         "pero mantiene el forward train para backprop.")

    # v5
    ap.add_argument("--neighbor_teeth", type=str, default="",
                    help='Lista "name:idx" separada por coma. Ej: "d11:1,d22:9" (idx = clase interna).')
    ap.add_argument("--neighbor_eval_split", type=str, default="val", choices=["val", "test", "both", "none"],
                    help="Dónde calcular métricas de vecinos durante entrenamiento. default=val")
    ap.add_argument("--neighbor_every", type=int, default=1,
                    help="Cada cuántos epochs calcular vecinos (1=siempre). Si es >1, reduce costo.")

    return ap.parse_args()


# ============================================================
# MAIN
# ============================================================
def main():
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "plots")
    ensure_dir(out_dir / "inference")

    run_log = out_dir / "run.log"
    err_log = out_dir / "errors.log"

    # ---------------- device ----------------
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

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

    log_line(f"[SANITY] num_classes C = {C}", run_log)
    log_line(f"[SANITY] bg_frac train/val/test = {bg_tr:.4f} {bg_va:.4f} {bg_te:.4f}", run_log)
    log_line(f"[SANITY] baseline acc_all (always-bg) train/val/test = {bg_tr:.4f} {bg_va:.4f} {bg_te:.4f}", run_log)

    if not (0 <= int(args.d21_internal) < C):
        raise ValueError(f"d21_internal fuera de rango: {args.d21_internal} (C={C})")

    # =========================================================
    # LOADERS
    # =========================================================
    dl_tr, dl_va, dl_te = make_loaders(
        data_dir=data_dir,
        bs=args.batch_size,
        nw=args.num_workers,
        normalize=(not args.no_normalize),
    )

    # =========================================================
    # MODEL / OPT / LOSS
    # =========================================================
    model = PointNetSeg(num_classes=C, dropout=float(args.dropout)).to(device)

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
    log_line(f"[INFO] d21_internal={d21_int}", run_log)

    # =========================================================
    # NEIGHBORS
    # =========================================================
    neighbor_list = parse_neighbor_teeth(args.neighbor_teeth)
    if neighbor_list:
        bad = [(n, i) for (n, i) in neighbor_list if not (0 <= int(i) < C)]
        if bad:
            raise ValueError(f"neighbor_teeth fuera de rango (C={C}): {bad}")
        log_line(f"[NEIGHBORS] parsed: {neighbor_list}", run_log)
    else:
        log_line("[NEIGHBORS] none", run_log)

    # columnas dinámicas (aparecen al final para NO romper el orden base)
    neighbor_cols = []
    for name, _ in neighbor_list:
        neighbor_cols += [
            f"{name}_acc", f"{name}_f1", f"{name}_iou", f"{name}_bin_acc_all"
        ]

    # =========================================================
    # RUN META
    # =========================================================
    run_meta = {
        "script_name": "pointnet_classic_final_v7.py",
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
        "infer_num_workers": int(args.infer_num_workers) if args.infer_num_workers is not None else None,
        "index_csv": str(args.index_csv) if args.index_csv else "",
        "neighbor_teeth": str(args.neighbor_teeth),
        "neighbor_eval_split": str(args.neighbor_eval_split),
        "neighbor_every": int(args.neighbor_every),
        "neighbor_parsed": neighbor_list,
    }
    save_json(run_meta, out_dir / "run_meta.json")

    # =========================================================
    # CSV por epoch
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
    # HISTORY
    # =========================================================
    history: Dict[str, List[float]] = defaultdict(list)

    for k in (
        "loss", "acc_all", "acc_no_bg",
        "f1_macro", "iou_macro",
        "d21_acc", "d21_f1", "d21_iou",
        "d21_bin_acc_all",
        "pred_bg_frac",
    ):
        history[f"train_{k}"] = []
        history[f"val_{k}"] = []

    for name, _ in neighbor_list:
        for met in ("acc", "f1", "iou", "bin_acc_all"):
            history[f"train_{name}_{met}"] = []
            history[f"val_{name}_{met}"] = []

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

        tr = run_epoch(
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
            neighbor_list=neighbor_list,
            train_metrics_eval=bool(args.train_metrics_eval),
        )

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
            neighbor_list=neighbor_list,
            train_metrics_eval=False,
        )

        sched.step()
        lr_now = float(opt.param_groups[0]["lr"])
        sec = float(time.time() - e0)

        # history base
        for k in (
            "loss", "acc_all", "acc_no_bg",
            "f1_macro", "iou_macro",
            "d21_acc", "d21_f1", "d21_iou",
            "d21_bin_acc_all",
            "pred_bg_frac",
        ):
            history[f"train_{k}"].append(float(tr[k]))
            history[f"val_{k}"].append(float(va[k]))

        # history neighbors
        if neighbor_list:
            for name, _ in neighbor_list:
                for met in ("acc", "f1", "iou", "bin_acc_all"):
                    history[f"train_{name}_{met}"].append(float(tr.get(f"{name}_{met}", 0.0)))
                    history[f"val_{name}_{met}"].append(float(va.get(f"{name}_{met}", 0.0)))

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
                row_train += [float(tr.get(col, 0.0)) for col in neighbor_cols]
                row_val += [float(va.get(col, 0.0)) for col in neighbor_cols]

            wcsv.writerow(row_train)
            wcsv.writerow(row_val)

        # JSONL por epoch
        epoch_row = {
            "epoch": epoch,
            **{f"train_{k}": float(v) for k, v in tr.items()},
            **{f"val_{k}": float(v) for k, v in va.items()},
            "lr": float(lr_now),
            "sec": float(sec),
        }
        append_jsonl(epoch_row, out_dir / "history_epoch.jsonl")

        # checkpoints
        torch.save({"model": model.state_dict(), "epoch": epoch}, last_path)

        if float(va["f1_macro"]) > best_val_f1:
            best_val_f1 = float(va["f1_macro"])
            best_epoch = int(epoch)
            torch.save({"model": model.state_dict(), "epoch": epoch}, best_path)

        # warning colapso a bg
        if float(va["pred_bg_frac"]) > max(0.95, bg_va + 0.12):
            log_line(
                f"[WARN] posible colapso a BG: "
                f"val pred_bg_frac={va['pred_bg_frac']:.3f} (bg_gt≈{bg_va:.3f})",
                run_log
            )

        nb_str = _format_neighbor_console(va, neighbor_list)

        log_line(
            f"[{epoch:03d}/{int(args.epochs)}] "
            f"train loss={tr['loss']:.4f} f1m={tr['f1_macro']:.3f} ioum={tr['iou_macro']:.3f} "
            f"acc_all={tr['acc_all']:.3f} acc_no_bg={tr['acc_no_bg']:.3f} | "
            f"val loss={va['loss']:.4f} f1m={va['f1_macro']:.3f} ioum={va['iou_macro']:.3f} "
            f"acc_all={va['acc_all']:.3f} acc_no_bg={va['acc_no_bg']:.3f} | "
            f"d21(cls) acc={va['d21_acc']:.3f} f1={va['d21_f1']:.3f} iou={va['d21_iou']:.3f} | "
            f"d21(bin all) acc={va['d21_bin_acc_all']:.3f} | "
            f"pred_bg_frac(train)={tr['pred_bg_frac']:.3f} pred_bg_frac(val)={va['pred_bg_frac']:.3f} "
            f"lr={lr_now:.2e} sec={sec:.1f}"
            f"{nb_str}",
            run_log
        )

        if neighbor_list:
            parts_full = []
            for name, _ in neighbor_list:
                parts_full.append(
                    f"{name}_acc={va.get(f'{name}_acc', 0.0):.3f} "
                    f"{name}_f1={va.get(f'{name}_f1', 0.0):.3f} "
                    f"{name}_iou={va.get(f'{name}_iou', 0.0):.3f} "
                    f"{name}_bin_acc_all={va.get(f'{name}_bin_acc_all', 0.0):.3f}"
                )
            log_line("[neighbors val] " + " | ".join(parts_full), run_log)

    save_json(dict(history), out_dir / "history.json")


    # =========================================================
    # TEST (best checkpoint)
    # =========================================================
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        log_line(f"[test] Cargado best.pt (epoch={ckpt.get('epoch','?')})", run_log)
    else:
        log_line("[test] best.pt no encontrado, usando modelo final (last.pt si existe)", run_log)
        if last_path.exists():
            ckpt = torch.load(last_path, map_location=device)
            model.load_state_dict(ckpt["model"])

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
        neighbor_list=neighbor_list,
        train_metrics_eval=False,
    )

    log_line(
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
        f"pred_bg_frac(test)={test_metrics['pred_bg_frac']:.3f}",
        run_log
    )

    test_neighbors = {}
    if neighbor_list:
        split_mode = str(args.neighbor_eval_split).lower()
        if split_mode in ("test", "both"):
            test_neighbors = {k: float(test_metrics[k]) for k in test_metrics.keys() if any(k.startswith(f"{n}_") for n, _ in neighbor_list)}

            parts = []
            for name, _ in neighbor_list:
                parts.append(
                    f"{name}_acc={test_neighbors.get(f'{name}_acc', 0.0):.3f} "
                    f"{name}_f1={test_neighbors.get(f'{name}_f1', 0.0):.3f} "
                    f"{name}_iou={test_neighbors.get(f'{name}_iou', 0.0):.3f} "
                    f"{name}_bin_acc_all={test_neighbors.get(f'{name}_bin_acc_all', 0.0):.3f}"
                )
            log_line("[test neighbors] " + " | ".join(parts), run_log)

            for k, v in test_neighbors.items():
                log_line(f"[test neighbors] {k} = {v:.4f}", run_log)

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
            plot_train_val(
                name=f"{name}_{met}",
                y_tr=history[f"train_{name}_{met}"],
                y_va=history[f"val_{name}_{met}"],
                out_png=plot_dir / f"{name}_{met}.png",
                best_epoch=best_epoch,
            )

    # =========================================================
    # INFERENCIA TRAZABLE
    # =========================================================
    if bool(args.do_infer):
        infer_nw = args.infer_num_workers
        if infer_nw is None:
            infer_nw = min(int(args.num_workers), 2)

        infer_loader = make_infer_loader(
            data_dir=data_dir,
            split=args.infer_split,
            bs=1,
            nw=infer_nw,
            normalize=(not args.no_normalize),
        )

        if args.index_csv:
            idx_map = _read_index_csv(Path(args.index_csv))
        else:
            auto_idx = _discover_index_csv(data_dir, args.infer_split)
            idx_map = _read_index_csv(auto_idx) if auto_idx else None

        inf_root = ensure_dir(out_dir / "inference")
        inf_all = ensure_dir(inf_root / "inference_all")
        inf_err = ensure_dir(inf_root / "inference_errors")
        inf_d21 = ensure_dir(inf_root / "inference_d21")

        manifest_rows = []
        count = 0

        model.eval()
        with torch.no_grad():
            for xyz, y, row_i in infer_loader:
                if count >= int(args.infer_examples):
                    break

                xyz = xyz.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                logits = model(xyz)
                pred = logits.argmax(dim=-1)

                xyz_np = xyz[0].detach().cpu().numpy()
                y_np = y[0].detach().cpu().numpy()
                pr_np = pred[0].detach().cpu().numpy()

                ri = int(row_i.item())
                tag = f"{args.infer_split}_row{ri}"

                info = {}
                if idx_map and ri in idx_map:
                    info = idx_map[ri]
                    safe_name = _sanitize_tag(info.get("sample_name", ""))
                    safe_jaw = _sanitize_tag(info.get("jaw", ""))
                    if safe_name or safe_jaw:
                        suffix = "_".join([x for x in [safe_name, safe_jaw] if x])
                        tag = f"{tag}_{suffix}"

                png_all = inf_all / f"{tag}.png"
                png_err = inf_err / f"{tag}.png"
                png_d21 = inf_d21 / f"{tag}.png"

                plot_pointcloud_all_classes(
                    xyz_np, y_np, pr_np, png_all,
                    C=C, title=tag, errors_log=err_log
                )
                plot_errors(
                    xyz_np, y_np, pr_np, png_err,
                    bg=bg, title=tag, errors_log=err_log
                )
                plot_d21_focus(
                    xyz_np, y_np, pr_np, png_d21,
                    d21_idx=d21_int, bg=bg, title=tag, errors_log=err_log
                )

                manifest_rows.append({
                    "row_i": ri,
                    "tag": tag,
                    "sample_name": info.get("sample_name", "") if info else "",
                    "jaw": info.get("jaw", "") if info else "",
                    "path": info.get("path", "") if info else "",
                    "idx_global": info.get("idx_global", "") if info else "",
                    "png_all": str(png_all.relative_to(out_dir)),
                    "png_errors": str(png_err.relative_to(out_dir)),
                    "png_d21": str(png_d21.relative_to(out_dir)),
                })

                count += 1

        if len(manifest_rows) > 0:
            with open(inf_root / "inference_manifest.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=manifest_rows[0].keys())
                w.writeheader()
                for r in manifest_rows:
                    w.writerow(r)

    total_sec = time.time() - t0
    log_line(
        f"[done] Entrenamiento terminado en {_fmt_hms(total_sec)}. "
        f"best_epoch={best_epoch} best_val={best_val_f1:.4f}",
        run_log
    )


if __name__ == "__main__":
    main()



