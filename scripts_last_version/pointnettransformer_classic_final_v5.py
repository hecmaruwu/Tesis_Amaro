#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pointnettransformer_classic_final_v5.py

PointNet-Transformer – Segmentación multiclase dental 3D (MISMA TRAZABILIDAD/OUTPUTS)

Objetivo:
- Mantener outputs/estructura idéntica a pointnet_classic_final_v4 + extras tipo DGCNN:
  ✅ history.json, metrics_epoch.csv, run_meta.json, best.pt, last.pt, test_metrics.json
  ✅ plots Train vs Val (sin línea horizontal de TEST)
  ✅ inferencia: inference_all / inference_errors / inference_d21 + manifest
  ✅ métricas macro sin bg + d21 binario (sin bg) + d21_bin_acc_all (incluye bg)
  ✅ métricas de dientes vecinos (lista arbitraria: "d11:idx,d22:idx,...")

Backbone:
- PointNet-style embedding inicial (MLP compartido)
- Transformer Encoder sobre tokens de puntos (con positional encoding opcional)
- Head punto-a-punto para logits [B,N,C]

Dataset esperado:
  data_dir/X_train.npz, Y_train.npz, X_val.npz, Y_val.npz, X_test.npz, Y_test.npz
  X: [B,N,3], Y: [B,N] con clases internas 0..C-1 (0=bg)

Ejemplo:
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python3 -u pointnettransformer_classic_final_v5.py \
  --data_dir /home/htaucare/Tesis_Amaro/data/Teeth_3ds/fixed_split/8192/upper_only_surf_global_excl_wisdom_seed42_aug2 \
  --out_dir  /home/htaucare/Tesis_Amaro/outputs/pointnet_transformer/gpu0_run_fast_v5 \
  --epochs 120 --batch_size 8 --lr 2e-4 --weight_decay 1e-4 \
  --num_workers 6 --device cuda --use_amp --grad_clip 1.0 \
  --bg_index 0 --bg_weight 0.03 --d21_internal 8 \
  --neighbor_teeth "d11:1,d22:9" \
  --token_subsample 2048 --prop_k 3 --prop_chunk 2048 \
  --do_infer --infer_examples 12 --infer_split test
"""

import os
import re
import json
import csv
import time
import math
import argparse
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict  # usado en history

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


def append_jsonl(obj, path: Path):
    """
    Guardado incremental por época (JSON Lines).
    Útil si matan el job (walltime / OOM / ctrl+c), igual queda trazabilidad.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _fmt_hms(seconds: float) -> str:
    seconds = float(seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:d}h {m:02d}m {s:05.2f}s"


def _get_lr(opt: torch.optim.Optimizer) -> float:
    try:
        return float(opt.param_groups[0].get("lr", 0.0))
    except Exception:
        return float("nan")


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(p: Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def log_line(msg: str, log_path: Optional[Path] = None, also_print: bool = True):
    """
    Log robusto:
    - imprime a stdout
    - opcionalmente guarda en errors.log (o run.log) para post-mortem
    """
    line = f"[{_now_str()}] {msg}"
    if also_print:
        print(line, flush=True)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


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

    ⚠️ (nota histórica)
    En inferencia, matplotlib + dataloader con muchos workers a veces da problemas silenciosos
    en algunos clusters. Para robustez, limitamos workers desde main usando --infer_num_workers.
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


def infer_num_classes_from_npz(data_dir: Path) -> int:
    """
    Inferimos C de forma robusta desde Y_train/val/test.
    """
    ys = []
    for split in ("train", "val", "test"):
        yp = data_dir / f"Y_{split}.npz"
        y = np.load(yp)["Y"].reshape(-1)
        ys.append(y)
    y_all = np.concatenate(ys, axis=0)
    return int(np.max(y_all)) + 1


# ============================================================
# MÉTRICAS (macro sin bg) + d21 binario + vecinos
# ============================================================
@torch.no_grad()
def macro_metrics_no_bg(pred: torch.Tensor, gt: torch.Tensor, C: int, bg: int = 0) -> Tuple[float, float]:
    """
    Macro-F1 e IoU macro calculados EXCLUYENDO BG (gt!=bg),
    promediando sobre clases 1..C-1 (omitimos clases sin soporte).
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
            # clase ausente -> omitimos del macro
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
    # Reutilizamos d21_metrics_binary para cualquier diente (binario tooth vs resto)
    acc, f1, iou = d21_metrics_binary(pred, gt, d21_idx=int(tooth_idx), bg=int(bg), include_bg=False)
    acc_all, f1_all, iou_all = d21_metrics_binary(pred, gt, d21_idx=int(tooth_idx), bg=int(bg), include_bg=True)
    return {
        "acc": float(acc), "f1": float(f1), "iou": float(iou),
        "bin_acc_all": float(acc_all), "bin_f1_all": float(f1_all), "bin_iou_all": float(iou_all),
    }


def parse_neighbor_teeth(s: Optional[str]) -> List[Tuple[str, int]]:
    """
    Parse string tipo: "d11:1,d22:9" -> [("d11",1), ("d22",9)]
    (nombre arbitrario, idx interno int)
    """
    if s is None:
        return []
    s = str(s).strip()
    if not s:
        return []
    out: List[Tuple[str, int]] = []
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if ":" not in p:
            continue
        name, idx = p.split(":", 1)
        name = str(name).strip()
        try:
            idxi = int(str(idx).strip())
        except Exception:
            continue
        if name:
            out.append((name, idxi))
    return out

# ============================================================
# (continúa) MÉTRICAS (macro sin bg) + d21 binario + vecinos
# ============================================================
@torch.no_grad()
def eval_neighbors_on_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    neighbor_list: List[Tuple[str, int]],
    bg: int
) -> Dict[str, float]:
    """
    Eval promedio en loader para vecinos (acumula por batch y promedia).
    Mantiene la misma API que tu versión previa.
    """
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
# TRAZABILIDAD (index_*.csv) – igual que v4
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
# FIXES DE NOMBRES (compatibilidad)
# - en tu log salió NameError: _set_seed -> alias directo
# ============================================================
def _set_seed(seed: int = 42):
    # alias por compatibilidad con versiones previas
    return set_seed(seed)


# ============================================================
# DATASET (NPZPointDataset) – compat "v3"
# Nota: lo dejamos tal cual para no romper nada tuyo.
# ============================================================
class NPZPointDataset(Dataset):
    """
    Dataset NPZ compatible con tu pipeline:
      - X_*.npz contiene {"X": [B,N,3]}
      - Y_*.npz contiene {"Y": [B,N]}
    Devuelve:
      - train/val/test: (xyz [N,3], y [N])
      - infer (opcional): (xyz, y, row_i) para trazabilidad (row dentro del split)
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
        xi = np.ascontiguousarray(self.X[i], dtype=np.float32)  # [N,3]
        yi = np.ascontiguousarray(self.Y[i], dtype=np.int64)    # [N]

        xyz = torch.as_tensor(xi, dtype=torch.float32)
        y   = torch.as_tensor(yi, dtype=torch.int64)

        if self.normalize:
            xyz = normalize_unit_sphere(xyz)

        if self.return_index:
            return xyz, y, torch.tensor(i, dtype=torch.int64)
        return xyz, y


def make_loaders_v3(data_dir: Path, bs: int, nw: int, normalize: bool = True):
    """
    Igual filosofía de make_loaders, pero usando NPZPointDataset (definido).
    Retorna: dl_tr, dl_va, dl_te
    """
    ds_tr = NPZPointDataset(data_dir / "X_train.npz", data_dir / "Y_train.npz", normalize=normalize, return_index=False)
    ds_va = NPZPointDataset(data_dir / "X_val.npz",   data_dir / "Y_val.npz",   normalize=normalize, return_index=False)
    ds_te = NPZPointDataset(data_dir / "X_test.npz",  data_dir / "Y_test.npz",  normalize=normalize, return_index=False)

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


def make_infer_loader_v3(data_dir: Path, split: str, bs: int, nw: int, normalize: bool = True) -> DataLoader:
    """
    Loader para infer trazable: retorna (xyz, y, row_i)
    """
    split = str(split).strip().lower()
    if split == "train":
        ds = NPZPointDataset(data_dir / "X_train.npz", data_dir / "Y_train.npz", normalize=normalize, return_index=True)
    elif split == "val":
        ds = NPZPointDataset(data_dir / "X_val.npz",   data_dir / "Y_val.npz",   normalize=normalize, return_index=True)
    elif split == "test":
        ds = NPZPointDataset(data_dir / "X_test.npz",  data_dir / "Y_test.npz",  normalize=normalize, return_index=True)
    else:
        raise ValueError(f"split inválido: {split}")

    common = dict(
        batch_size=int(bs),
        num_workers=int(nw),
        pin_memory=True,
        persistent_workers=(int(nw) > 0),
        drop_last=False,
    )
    if int(nw) > 0:
        common["prefetch_factor"] = 2

    return DataLoader(ds, shuffle=False, **common)

# ============================================================
# FAST kNN (chunked) + propagación token->full
# - evita O(N^2) sobre N=8192
# - atención solo en M tokens (M=token_subsample)
# - luego propaga logits/features a todos los puntos usando kNN (prop_k)
# ============================================================
def _as_fp32(x: torch.Tensor) -> torch.Tensor:
    return x.float() if x.dtype != torch.float32 else x


@torch.no_grad()
def knn_indices_chunked(q_xyz: torch.Tensor, k_xyz: torch.Tensor, k: int, chunk: int = 2048) -> torch.Tensor:
    """
    kNN desde queries a keys, CHUNKED (FP32).
      q_xyz: [B,Q,3]
      k_xyz: [B,K,3]
    return idx: [B,Q,k]  (idx sobre dimensión K)

    Nota:
    - Implementación por dist^2 = ||q||^2 + ||k||^2 - 2 q·k
    - Chunked en Q para evitar memoria gigante.
    """
    q_xyz = _as_fp32(q_xyz)
    k_xyz = _as_fp32(k_xyz)

    B, Q, _ = q_xyz.shape
    _, K, _ = k_xyz.shape
    k = int(k)
    chunk = int(chunk)

    device = q_xyz.device
    idx_out = torch.empty((B, Q, k), dtype=torch.long, device=device)

    # dist^2(a,b)=||a||^2 + ||b||^2 - 2 a·b
    q2 = (q_xyz ** 2).sum(dim=-1, keepdim=True)                   # [B,Q,1]
    k2 = (k_xyz ** 2).sum(dim=-1, keepdim=True).transpose(2, 1)   # [B,1,K]
    k_t = k_xyz.transpose(2, 1).contiguous()                      # [B,3,K]

    for s in range(0, Q, chunk):
        e = min(Q, s + chunk)
        q = q_xyz[:, s:e, :]                                      # [B,Qs,3]
        dot = torch.matmul(q, k_t)                                # [B,Qs,K]
        dist = q2[:, s:e, :] + k2 - 2.0 * dot                     # [B,Qs,K]
        dist = torch.clamp(dist, min=0.0)

        kk = min(k, K) if K > 0 else 1
        idx = torch.topk(dist, k=kk, dim=-1, largest=False).indices  # [B,Qs,kk]
        if idx.shape[-1] < k:
            pad = k - idx.shape[-1]
            idx = torch.cat([idx, idx[..., :1].repeat(1, 1, pad)], dim=-1)
        idx_out[:, s:e, :] = idx[:, :, :k]

    return idx_out


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    points: [B,N,C]
    idx:    [B,S] or [B,S,K]
    return: [B,S,C] or [B,S,K,C]
    """
    B, N, C = points.shape
    idx = torch.clamp(idx, 0, N - 1)
    device = points.device

    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1

    batch_indices = torch.arange(B, device=device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def farthest_point_sample_stub(xyz: torch.Tensor, M: int) -> torch.Tensor:
    """
    ⚠️ Manteniendo simplicidad/robustez: token subsample por RAND (no FPS),
    porque FPS puro en PyTorch puede ser lento / costoso en CPU/loader.
    - M << N => reduce O(N^2) a O(M^2).
    - Determinismo depende del seed global + estado RNG de torch.

    Si quieres FPS real después, lo cambiamos, pero este stub es suficiente para acelerar.
    """
    B, N, _ = xyz.shape
    M = int(min(M, N))
    device = xyz.device
    idx = torch.empty((B, M), dtype=torch.long, device=device)
    for b in range(B):
        perm = torch.randperm(N, device=device)
        idx[b] = perm[:M]
    return idx


# ============================================================
# POINTNET-TRANSFORMER (FAST) – tokens + propagación
# - embed inicial estilo PointNet
# - transformer encoder sobre tokens (M)
# - head en tokens -> logits_tokens [B,M,C]
# - propaga logits a N usando kNN (prop_k)
# ============================================================
class PointMLPEmbed(nn.Module):
    """
    Embedding inicial por punto (MLP compartido).
    Entrada:  xyz [B,N,3]
    Salida:   feat [B,N,D]
    """
    def __init__(self, d_model: int = 256, hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        self.d_model = int(d_model)
        self.net = nn.Sequential(
            nn.Linear(3, int(hidden)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), int(d_model)),
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        return self.net(xyz)


class PositionalEncodingMLP(nn.Module):
    """
    Positional encoding aprendido desde xyz.
    Suma un vector pe = f(xyz) a los tokens.
    """
    def __init__(self, d_model: int = 256, hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, int(hidden)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), int(d_model)),
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        return self.net(xyz)


class TransformerBlock(nn.Module):
    """
    Wrapper simple sobre TransformerEncoderLayer con batch_first=True.
    """
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_first: bool = True,
    ):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=int(d_model),
            nhead=int(nhead),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            activation=str(activation),
            batch_first=True,
            norm_first=bool(norm_first),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class PointNetTransformerSegFast(nn.Module):
    """
    Segmentación punto-a-punto FAST:
      xyz(N) -> tokens(M) -> transformer(M) -> head(M) -> logits_tokens
      logits_tokens -> propaga a N con kNN (N->M) => logits_full [B,N,C]

    Importante:
    - Este forward devuelve logits [B,N,C] SIEMPRE.
    - Eso simplifica training/eval/infer: no hay "modo tokens" en el exterior.
    """
    def __init__(
        self,
        num_classes: int,
        d_model: int = 256,
        embed_hidden: int = 128,
        depth: int = 4,
        nhead: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        use_pos_mlp: bool = True,
        pos_hidden: int = 128,
        norm_first: bool = True,
        activation: str = "gelu",
        head_hidden: int = 256,
        head_dropout: float = 0.1,
        token_subsample: int = 2048,
        prop_k: int = 3,
        prop_chunk: int = 2048,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.d_model = int(d_model)
        self.depth = int(depth)
        self.use_pos_mlp = bool(use_pos_mlp)

        self.token_subsample = int(token_subsample)
        self.prop_k = int(prop_k)
        self.prop_chunk = int(prop_chunk)

        self.embed = PointMLPEmbed(
            d_model=int(d_model),
            hidden=int(embed_hidden),
            dropout=float(dropout)
        )

        self.pos = (
            PositionalEncodingMLP(
                d_model=int(d_model),
                hidden=int(pos_hidden),
                dropout=float(dropout),
            )
            if self.use_pos_mlp
            else None
        )

        self.encoder = nn.ModuleList([
            TransformerBlock(
                d_model=int(d_model),
                nhead=int(nhead),
                dim_feedforward=int(dim_feedforward),
                dropout=float(dropout),
                activation=str(activation),
                norm_first=bool(norm_first),
            )
            for _ in range(int(depth))
        ])

        self.head = nn.Sequential(
            nn.Linear(int(d_model), int(head_hidden)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(head_dropout)),
            nn.Linear(int(head_hidden), int(num_classes)),
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        xyz: [B,N,3]
        return logits: [B,N,C]
        """
        B, N, _ = xyz.shape
        xyz = xyz.float()

        # --- token subsample (RAND por batch; determinismo depende del seed global) ---
        M = int(min(self.token_subsample, N))
        if M < N:
            tok_idx = farthest_point_sample_stub(xyz, M)          # [B,M]
            tok_xyz = index_points(xyz, tok_idx)                  # [B,M,3]
        else:
            tok_xyz = xyz

        # --- embed tokens ---
        x = self.embed(tok_xyz)                                   # [B,M,D]
        if self.pos is not None:
            x = x + self.pos(tok_xyz)

        # --- transformer sobre tokens ---
        for blk in self.encoder:
            x = blk(x)                                            # [B,M,D]

        logits_tok = self.head(x)                                 # [B,M,C]

        # --- si M==N, listo ---
        if M == N:
            return logits_tok

        # --- propaga logits tokens a todos los puntos N usando kNN: N->M ---
        # idx_nm: [B,N,prop_k] indices sobre tokens (M)
        #
        # ⚠️ Importante:
        # - hacemos este kNN en FP32 aunque uses AMP, para estabilidad numérica.
        # - knn_indices_chunked ya trabaja en FP32 por _as_fp32.
        with torch.amp.autocast(device_type="cuda", enabled=False):
            idx_nm = knn_indices_chunked(
                q_xyz=_as_fp32(xyz),
                k_xyz=_as_fp32(tok_xyz),
                k=self.prop_k,
                chunk=self.prop_chunk
            )

        # gather logits_tok: [B,N,k,C]
        logits_nk = index_points(logits_tok, idx_nm)              # [B,N,k,C]

        # agregación simple (mean) -> [B,N,C]
        logits_full = logits_nk.mean(dim=2)
        return logits_full

# ============================================================
# TRAIN / EVAL (estilo v4) + VECINOS
# - corregido para FAST backbone (logits ya [B,N,C])
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
        # dejamos estos 2 por compat/debug (no siempre se loguean en CSV)
        out[f"{name}_bin_f1_all"] = float(m["bin_f1_all"])
        out[f"{name}_bin_iou_all"] = float(m["bin_iou_all"])

    return out


def _merge_metrics(base: Dict[str, float], extra: Dict[str, float]) -> Dict[str, float]:
    out = dict(base)
    for k, v in (extra or {}).items():
        out[k] = float(v)
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
) -> Dict[str, float]:
    """
    run_epoch:
    - Si train=True: hace forward+backward con AMP opcional y grad clip, luego evalúa (mismo batch) sin dropout.
    - Si train=False: solo eval.
    - Acumula métricas por batch y promedia.

    Nota:
    - neighbor_list aquí opera "por batch" (barato) porque usa _tooth_metrics_binary.
    - Para evaluar vecinos en todo el split con menos varianza, puedes usar eval_neighbors_on_loader().
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
            model.train(True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
                logits_train = model(xyz)  # [B,N,C] (FAST ya propagado)
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

            # eval (mismo batch) para métricas comparables y sin dropout
            model.eval()
            with torch.no_grad():
                with torch.amp.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
                    logits_eval = model(xyz)

            metrics = _compute_metrics_from_logits(
                logits_eval, y, C=C, d21_idx=d21_idx, bg=bg
            )
            nb_metrics = _neighbors_metrics_from_logits(
                logits_eval, y, neighbor_list=neighbor_list, bg=bg
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


run_epoch.scaler = None

# ============================================================
# ARGPARSE
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="PointNet-Transformer FAST v5 (full trazabilidad)")

    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)

    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--infer_num_workers", type=int, default=None)

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_amp", action="store_true")
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--bg_index", type=int, default=0)
    p.add_argument("--bg_weight", type=float, default=0.03)
    p.add_argument("--d21_internal", type=int, required=True)

    p.add_argument("--neighbor_teeth", type=str, default="")

    # Transformer
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--embed_hidden", type=int, default=128)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--dim_feedforward", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--head_hidden", type=int, default=256)
    p.add_argument("--head_dropout", type=float, default=0.1)

    # FAST
    p.add_argument("--token_subsample", type=int, default=2048)
    p.add_argument("--prop_k", type=int, default=3)
    p.add_argument("--prop_chunk", type=int, default=2048)

    # Infer
    p.add_argument("--do_infer", action="store_true")
    p.add_argument("--infer_examples", type=int, default=12)
    p.add_argument("--infer_split", type=str, default="test")

    return p.parse_args()


# ============================================================
# MAIN
# ============================================================
def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "plots")
    ensure_dir(out_dir / "inference")

    run_log = out_dir / "run.log"
    err_log = out_dir / "errors.log"

    _set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Guardar run_meta
    save_json(vars(args), out_dir / "run_meta.json")

    # Loaders
    dl_tr, dl_va, dl_te = make_loaders_v3(
        data_dir, args.batch_size, args.num_workers, normalize=True
    )

    C = infer_num_classes_from_npz(data_dir)
    neighbor_list = parse_neighbor_teeth(args.neighbor_teeth)

    log_line(f"Num classes = {C}", run_log)
    log_line(f"Neighbors = {neighbor_list}", run_log)

    model = PointNetTransformerSegFast(
        num_classes=C,
        d_model=args.d_model,
        embed_hidden=args.embed_hidden,
        depth=args.depth,
        nhead=args.nhead,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        head_hidden=args.head_hidden,
        head_dropout=args.head_dropout,
        token_subsample=args.token_subsample,
        prop_k=args.prop_k,
        prop_chunk=args.prop_chunk,
    ).to(device)

    weights = torch.ones(C, dtype=torch.float32)
    weights[int(args.bg_index)] = float(args.bg_weight)
    weights = weights.to(device)

    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    history = defaultdict(list)
    best_val = -1.0
    best_epoch = -1

    # ========================================================
    # TRAIN LOOP
    # ========================================================
    for epoch in range(1, args.epochs + 1):

        tr = run_epoch(
            model, dl_tr, optimizer, loss_fn,
            C=C,
            d21_idx=args.d21_internal,
            device=device,
            bg=args.bg_index,
            train=True,
            use_amp=args.use_amp,
            grad_clip=args.grad_clip,
            neighbor_list=neighbor_list,
        )

        va = run_epoch(
            model, dl_va, None, loss_fn,
            C=C,
            d21_idx=args.d21_internal,
            device=device,
            bg=args.bg_index,
            train=False,
            use_amp=args.use_amp,
            grad_clip=None,
            neighbor_list=neighbor_list,
        )

        scheduler.step()

        # guardar history
        for k, v in tr.items():
            history[f"train_{k}"].append(v)
        for k, v in va.items():
            history[f"val_{k}"].append(v)

        # CSV row
        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in tr.items()},
            **{f"val_{k}": v for k, v in va.items()},
            "lr": scheduler.get_last_lr()[0],
        }

        csv_path = out_dir / "metrics_epoch.csv"
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                w.writeheader()
            w.writerow(row)

        append_jsonl(row, out_dir / "history_epoch.jsonl")

        console_nb = _format_neighbor_console(va, neighbor_list)

        log_line(
            f"[{epoch}/{args.epochs}] "
            f"train loss={tr['loss']:.4f} f1m={tr['f1_macro']:.3f} "
            f"| val loss={va['loss']:.4f} f1m={va['f1_macro']:.3f} "
            f"d21_f1={va['d21_f1']:.3f}"
            f"{console_nb}",
            run_log
        )

        # best model según val f1_macro
        if va["f1_macro"] > best_val:
            best_val = va["f1_macro"]
            best_epoch = epoch
            torch.save(model.state_dict(), out_dir / "best.pt")

    torch.save(model.state_dict(), out_dir / "last.pt")
    save_json(history, out_dir / "history.json")

    # ========================================================
    # TEST FINAL
    # ========================================================
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
    te = run_epoch(
        model, dl_te, None, loss_fn,
        C=C,
        d21_idx=args.d21_internal,
        device=device,
        bg=args.bg_index,
        train=False,
        use_amp=args.use_amp,
        grad_clip=None,
        neighbor_list=neighbor_list,
    )
    save_json(te, out_dir / "test_metrics.json")

    # ========================================================
    # PLOTS OBLIGATORIOS
    # ========================================================
    plots_dir = out_dir / "plots"

    plot_train_val("loss", history["train_loss"], history["val_loss"], plots_dir / "loss.png", best_epoch)
    plot_train_val("f1_macro", history["train_f1_macro"], history["val_f1_macro"], plots_dir / "f1_macro.png", best_epoch)
    plot_train_val("iou_macro", history["train_iou_macro"], history["val_iou_macro"], plots_dir / "iou_macro.png", best_epoch)
    plot_train_val("d21_f1", history["train_d21_f1"], history["val_d21_f1"], plots_dir / "d21_f1.png", best_epoch)

    plot_train_val("acc_all", history["train_acc_all"], history["val_acc_all"], plots_dir / "acc_all.png", best_epoch)
    plot_train_val("acc_no_bg", history["train_acc_no_bg"], history["val_acc_no_bg"], plots_dir / "acc_no_bg.png", best_epoch)

    # Vecinos plots
    for name, _ in neighbor_list:
        plot_train_val(f"{name}_acc",
                       history[f"train_{name}_acc"],
                       history[f"val_{name}_acc"],
                       plots_dir / f"{name}_acc.png",
                       best_epoch)

        plot_train_val(f"{name}_f1",
                       history[f"train_{name}_f1"],
                       history[f"val_{name}_f1"],
                       plots_dir / f"{name}_f1.png",
                       best_epoch)

        plot_train_val(f"{name}_iou",
                       history[f"train_{name}_iou"],
                       history[f"val_{name}_iou"],
                       plots_dir / f"{name}_iou.png",
                       best_epoch)

        plot_train_val(f"{name}_bin_acc_all",
                       history[f"train_{name}_bin_acc_all"],
                       history[f"val_{name}_bin_acc_all"],
                       plots_dir / f"{name}_bin_acc_all.png",
                       best_epoch)

    # ========================================================
    # INFERENCIA TRAZABLE
    # ========================================================
    if args.do_infer:
        infer_nw = args.infer_num_workers
        if infer_nw is None:
            infer_nw = min(args.num_workers, 2)

        infer_loader = make_infer_loader_v3(
            data_dir,
            split=args.infer_split,
            bs=1,
            nw=infer_nw,
            normalize=True
        )

        idx_csv = _discover_index_csv(data_dir, args.infer_split)
        idx_map = _read_index_csv(idx_csv) if idx_csv else None

        inf_root = out_dir / "inference"
        inf_all = ensure_dir(inf_root / "inference_all")
        inf_err = ensure_dir(inf_root / "inference_errors")
        inf_d21 = ensure_dir(inf_root / "inference_d21")

        manifest_rows = []
        count = 0

        model.eval()
        with torch.no_grad():
            for xyz, y, row_i in infer_loader:
                if count >= args.infer_examples:
                    break

                xyz = xyz.to(device)
                y = y.to(device)

                logits = model(xyz)
                pred = logits.argmax(dim=-1)

                xyz_np = xyz[0].cpu().numpy()
                y_np = y[0].cpu().numpy()
                p_np = pred[0].cpu().numpy()

                ri = int(row_i.item())
                tag = f"{args.infer_split}_row{ri}"

                if idx_map and ri in idx_map:
                    info = idx_map[ri]
                    name = _sanitize_tag(info.get("sample_name", ""))
                    jaw = _sanitize_tag(info.get("jaw", ""))
                    tag = f"{tag}_{name}_{jaw}"

                png_all = inf_all / f"{tag}.png"
                png_err = inf_err / f"{tag}.png"
                png_d21 = inf_d21 / f"{tag}.png"

                plot_pointcloud_all_classes(
                    xyz_np, y_np, p_np, png_all, C=C,
                    title=tag, errors_log=err_log
                )
                plot_errors(
                    xyz_np, y_np, p_np, png_err,
                    bg=args.bg_index,
                    title=tag, errors_log=err_log
                )
                plot_d21_focus(
                    xyz_np, y_np, p_np, png_d21,
                    d21_idx=args.d21_internal,
                    bg=args.bg_index,
                    title=tag, errors_log=err_log
                )

                manifest_rows.append({
                    "row_i": ri,
                    "tag": tag,
                    "png_all": str(png_all.relative_to(out_dir)),
                    "png_errors": str(png_err.relative_to(out_dir)),
                    "png_d21": str(png_d21.relative_to(out_dir)),
                })

                count += 1

        with open(inf_root / "inference_manifest.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=manifest_rows[0].keys())
            w.writeheader()
            for r in manifest_rows:
                w.writerow(r)


if __name__ == "__main__":
    main()
