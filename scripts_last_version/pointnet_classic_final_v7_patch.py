#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pointnet_classic_final_v7_patch.py

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

(PATCH sobre v7 original):
✅ NO elimina outputs ni funcionalidad existente
✅ Añade test_metrics_filtered.json excluyendo samples only_bg
✅ Añade ignored_test_samples.json
✅ Filtra SOLO la inferencia visual para omitir samples only_bg
✅ Añade inference/ignored_inference_samples.json
✅ Mantiene plots/, inference_manifest.csv y trazabilidad de paciente

Dataset esperado:
  data_dir/X_train.npz, Y_train.npz, X_val.npz, Y_val.npz, X_test.npz, Y_test.npz
  X: [B,N,3], Y: [B,N] con clases internas 0..C-1 (0=bg)

Ejemplo:
python3 pointnet_classic_final_v7_patch.py \
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
            return None

        def _pick(*cands):
            for c in cands:
                if c in fields:
                    return fields[c]
            return None

        k_name = _pick("sample_name", "sample", "name", "patient")
        k_jaw = _pick("jaw", "arch")
        k_path = _pick("path", "file_path")

        mp = {}
        for row in reader:
            try:
                ri = int(row[row_key])
            except:
                continue

            mp[ri] = {
                "sample_name": row.get(k_name, "") if k_name else "",
                "jaw": row.get(k_jaw, "") if k_jaw else "",
                "path": row.get(k_path, "") if k_path else "",
            }

        return mp if len(mp) > 0 else None


def _discover_index_csv(data_dir: Path, split: str) -> Optional[Path]:
    split = str(split).lower()
    fname = f"index_{split}.csv"

    # 1) directo
    p = data_dir / fname
    if p.exists():
        return p

    # 2) ancestros
    cur = data_dir
    for _ in range(10):
        p = cur / fname
        if p.exists():
            return p
        if cur.parent == cur:
            break
        cur = cur.parent

    return None


# ============================================================
# ENTRENAMIENTO / EVALUACIÓN (IDÉNTICO A v7)
# ============================================================
@torch.no_grad()
def _compute_metrics_from_logits(
    logits: torch.Tensor,
    y: torch.Tensor,
    C: int,
    d21_idx: int,
    bg: int,
) -> Dict[str, float]:

    pred = logits.argmax(dim=-1)

    acc_all = _acc_all(pred, y)

    mask = (y != int(bg))
    acc_no_bg = float((pred[mask] == y[mask]).float().mean().item()) if mask.sum() > 0 else 0.0

    f1m, ioum = macro_metrics_no_bg(pred, y, C=C, bg=bg)

    d21_acc, d21_f1, d21_iou = d21_metrics_binary(pred, y, d21_idx, bg, False)
    d21_bin_all, _, _ = d21_metrics_binary(pred, y, d21_idx, bg, True)

    pred_bg_frac = float((pred == int(bg)).float().mean().item())

    return {
        "acc_all": acc_all,
        "acc_no_bg": acc_no_bg,
        "f1_macro": f1m,
        "iou_macro": ioum,
        "d21_acc": d21_acc,
        "d21_f1": d21_f1,
        "d21_iou": d21_iou,
        "d21_bin_acc_all": d21_bin_all,
        "pred_bg_frac": pred_bg_frac,
    }


def run_epoch(
    model,
    loader,
    optimizer,
    loss_fn,
    C,
    d21_idx,
    device,
    bg,
    train,
    use_amp,
    grad_clip,
    neighbor_list=None,
    train_metrics_eval=False,
):
    neighbor_list = neighbor_list or []

    if train:
        model.train()
    else:
        model.eval()

    sums = defaultdict(float)
    neigh_sums = defaultdict(float)

    nb = 0
    scaler = torch.cuda.amp.GradScaler() if (train and use_amp) else None

    for batch in loader:
        xyz, y = batch

        xyz = xyz.to(device)
        y = y.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(xyz)
                    loss = loss_fn(logits.reshape(-1, C), y.reshape(-1))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(xyz)
                loss = loss_fn(logits.reshape(-1, C), y.reshape(-1))
                loss.backward()
                optimizer.step()

            logits_eval = model(xyz) if train_metrics_eval else logits.detach()

        else:
            logits = model(xyz)
            loss = loss_fn(logits.reshape(-1, C), y.reshape(-1))
            logits_eval = logits

        m = _compute_metrics_from_logits(logits_eval, y, C, d21_idx, bg)

        sums["loss"] += loss.item()
        for k, v in m.items():
            sums[k] += v

        pred = logits_eval.argmax(dim=-1)

        for name, idx in neighbor_list:
            nm = _tooth_metrics_binary(pred, y, idx, bg)
            for k, v in nm.items():
                neigh_sums[f"{name}_{k}"] += v

        nb += 1

    out = {k: v / nb for k, v in sums.items()}
    for k, v in neigh_sums.items():
        out[k] = v / nb

    return out

# ============================================================
# PATCH NUEVO: TEST FILTRADO (solo agrega funcionalidad)
# ============================================================
@torch.no_grad()
def run_epoch_filtered_only_bg(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    C: int,
    d21_idx: int,
    device: torch.device,
    bg: int,
):
    """
    Igual que evaluación normal, pero ignora samples cuyo GT es solo background.
    Usa loader con return_index=True para poder guardar ignored_rows.
    """
    model.eval()

    sums = defaultdict(float)
    n_valid = 0
    ignored_rows = []

    for batch_idx, batch in enumerate(loader):
        if len(batch) == 3:
            xyz, y, row_i = batch
            ri = int(row_i.item())
        else:
            xyz, y = batch
            ri = batch_idx

        y_np = y[0].detach().cpu().numpy()
        vals = np.unique(y_np)

        if len(vals) == 1 and int(vals[0]) == int(bg):
            ignored_rows.append(ri)
            continue

        xyz = xyz.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(xyz)
        loss = loss_fn(logits.reshape(-1, C), y.reshape(-1))

        m = _compute_metrics_from_logits(logits, y, C, d21_idx, bg)

        sums["loss"] += float(loss.item())
        for k, v in m.items():
            sums[k] += float(v)

        n_valid += 1

    n = max(1, n_valid)

    out = {k: v / n for k, v in sums.items()}
    out["n_valid_samples"] = int(n_valid)
    out["ignored_rows"] = ignored_rows
    return out


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
# MAIN (primera parte)
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
        "script_name": "pointnet_classic_final_v7_patch.py",
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
    # TRAIN LOOP (IDÉNTICO v7)
    # =========================================================
    history_jsonl = out_dir / "history_epoch.jsonl"

    for epoch in range(1, int(args.epochs) + 1):
        t_ep = time.time()

        tr = run_epoch(
            model, dl_tr, opt, loss_fn, C, d21_int, device, bg,
            train=True, use_amp=args.use_amp, grad_clip=args.grad_clip,
            neighbor_list=neighbor_list,
            train_metrics_eval=args.train_metrics_eval
        )

        va = run_epoch(
            model, dl_va, None, loss_fn, C, d21_int, device, bg,
            train=False, use_amp=False, grad_clip=None,
            neighbor_list=neighbor_list
        )

        sched.step()
        lr_now = float(opt.param_groups[0]["lr"])
        sec = float(time.time() - t_ep)

        # ---------- history ----------
        for k, v in tr.items():
            if f"train_{k}" in history:
                history[f"train_{k}"].append(v)
        for k, v in va.items():
            if f"val_{k}" in history:
                history[f"val_{k}"].append(v)

        append_jsonl(
            {"epoch": epoch, "train": tr, "val": va},
            history_jsonl
        )

        # ---------- CSV ----------
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            wcsv = csv.writer(f)

            def _row(split, m):
                base = [
                    epoch, split,
                    m["loss"],
                    m["acc_all"], m["acc_no_bg"],
                    m["f1_macro"], m["iou_macro"],
                    m["d21_acc"], m["d21_f1"], m["d21_iou"],
                    m["d21_bin_acc_all"],
                    m["pred_bg_frac"],
                    lr_now,
                    sec,
                ]
                for name, _ in neighbor_list:
                    base += [
                        m.get(f"{name}_acc", 0.0),
                        m.get(f"{name}_f1", 0.0),
                        m.get(f"{name}_iou", 0.0),
                        m.get(f"{name}_bin_acc_all", 0.0),
                    ]
                return base

            wcsv.writerow(_row("train", tr))
            wcsv.writerow(_row("val", va))

        # ---------- consola ----------
        msg = (
            f"[{epoch}/{args.epochs}] "
            f"train loss={tr['loss']:.4f} f1m={tr['f1_macro']:.3f} | "
            f"val loss={va['loss']:.4f} f1m={va['f1_macro']:.3f} "
            f"| d21 f1={va['d21_f1']:.3f}"
        )

        if neighbor_list:
            neigh_txt = []
            for name, _ in neighbor_list:
                neigh_txt.append(f"{name}_f1={va.get(f'{name}_f1',0):.3f}")
            msg += " | " + " ".join(neigh_txt)

        log_line(msg, run_log)

        # ---------- best ----------
        if va["f1_macro"] > best_val_f1:
            best_val_f1 = va["f1_macro"]
            best_epoch = epoch
            torch.save({"model": model.state_dict()}, best_path)

    # =========================================================
    # SAVE LAST
    # =========================================================
    torch.save({"model": model.state_dict()}, last_path)

    # =========================================================
    # PLOTS (INTACTOS)
    # =========================================================
    plot_train_val("loss", history["train_loss"], history["val_loss"], out_dir / "plots/loss.png", best_epoch)
    plot_train_val("f1_macro", history["train_f1_macro"], history["val_f1_macro"], out_dir / "plots/f1_macro.png", best_epoch)
    plot_train_val("iou_macro", history["train_iou_macro"], history["val_iou_macro"], out_dir / "plots/iou_macro.png", best_epoch)
    plot_train_val("d21_f1", history["train_d21_f1"], history["val_d21_f1"], out_dir / "plots/d21_f1.png", best_epoch)

    # =========================================================
    # TEST NORMAL (INTACTO)
    # =========================================================
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    te = run_epoch(
        model, dl_te, None, loss_fn, C, d21_int, device, bg,
        train=False, use_amp=False, grad_clip=None,
        neighbor_list=neighbor_list
    )

    te["best_epoch"] = int(best_epoch)
    save_json(te, out_dir / "test_metrics.json")

    # =========================================================
    # 🔥 PATCH 1: TEST FILTRADO (NO ROMPE NADA)
    # =========================================================
    dl_te_inf = make_infer_loader(
        data_dir=data_dir,
        split="test",
        bs=1,
        nw=0,
        normalize=(not args.no_normalize),
    )

    te_filtered = run_epoch_filtered_only_bg(
        model=model,
        loader=dl_te_inf,
        loss_fn=loss_fn,
        C=C,
        d21_idx=d21_int,
        device=device,
        bg=bg,
    )

    save_json(te_filtered, out_dir / "test_metrics_filtered.json")

    save_json(
        {
            "ignored_rows": te_filtered["ignored_rows"],
            "n_valid": te_filtered["n_valid_samples"],
        },
        out_dir / "ignored_test_samples.json",
    )

    # =========================================================
    # 🔥 PATCH 2: INFERENCIA FILTRADA (SIN ROMPER TRAZABILIDAD)
    # =========================================================
    if args.do_infer:
        infer_loader = make_infer_loader(
            data_dir=data_dir,
            split=args.infer_split,
            bs=1,
            nw=(args.infer_num_workers or 0),
            normalize=(not args.no_normalize),
        )

        index_csv = Path(args.index_csv) if args.index_csv else _discover_index_csv(data_dir, args.infer_split)
        index_map = _read_index_csv(index_csv) if index_csv else None

        inf_root = ensure_dir(out_dir / "inference")
        inf_all = ensure_dir(inf_root / "inference_all")
        inf_err = ensure_dir(inf_root / "inference_errors")
        inf_d21 = ensure_dir(inf_root / "inference_d21")

        manifest = []
        ignored_rows = []

        model.eval()
        with torch.no_grad():
            for xyz, y, row_i in infer_loader:
                ri = int(row_i.item())

                # 🔥 FILTRO
                y_np = y[0].cpu().numpy()
                if len(np.unique(y_np)) == 1 and np.unique(y_np)[0] == bg:
                    ignored_rows.append(ri)
                    continue

                xyz = xyz.to(device)
                y = y.to(device)

                logits = model(xyz)
                pred = logits.argmax(dim=-1)

                xyz_np = xyz[0].cpu().numpy()
                y_np = y[0].cpu().numpy()
                pr_np = pred[0].cpu().numpy()

                meta = index_map.get(ri, {}) if index_map else {}

                tag = f"{args.infer_split}_row{ri}"
                if meta.get("sample_name"):
                    tag += "_" + _sanitize_tag(meta["sample_name"])

                title = tag

                p_all = inf_all / f"{tag}.png"
                p_err = inf_err / f"{tag}.png"
                p_d21 = inf_d21 / f"{tag}.png"

                plot_pointcloud_all_classes(xyz_np, y_np, pr_np, p_all, C, title)
                plot_errors(xyz_np, y_np, pr_np, p_err, bg, title)
                plot_d21_focus(xyz_np, y_np, pr_np, p_d21, d21_int, bg, title)

                manifest.append({
                    "row_i": ri,
                    "tag": tag,
                    "sample_name": meta.get("sample_name", ""),
                    "jaw": meta.get("jaw", ""),
                    "path": meta.get("path", ""),
                    "png_all": str(p_all.relative_to(inf_root)),
                    "png_errors": str(p_err.relative_to(inf_root)),
                    "png_d21": str(p_d21.relative_to(inf_root)),
                })

        # 🔥 CLAVE: RESTAURA TRAZABILIDAD
        if manifest:
            with open(inf_root / "inference_manifest.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
                writer.writeheader()
                writer.writerows(manifest)

        save_json(
            {
                "ignored_rows": ignored_rows,
                "n_ignored": len(ignored_rows),
            },
            inf_root / "ignored_inference_samples.json",
        )

    log_line(f"[done] tiempo total: {_fmt_hms(time.time() - t0)}", run_log)


if __name__ == "__main__":
    main()