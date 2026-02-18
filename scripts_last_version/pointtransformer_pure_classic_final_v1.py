#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pointtransformer_pure_classic_final_v1.py

Point Transformer (PURE) – Segmentación multiclase dental 3D (MISMA TRAZABILIDAD/OUTPUTS)

Objetivo:
- Mantener outputs/estructura idéntica a pointnet_classic_final_v4 + extras tipo DGCNN:
  ✅ history.json, metrics_epoch.csv, run_meta.json, best.pt, last.pt, test_metrics.json
  ✅ plots Train vs Val (sin línea horizontal de TEST)
  ✅ inferencia: inference_all / inference_errors / inference_d21 + manifest
  ✅ métricas macro sin bg + d21 binario (sin bg) + d21_bin_acc_all (incluye bg)
  ✅ métricas de dientes vecinos (lista arbitraria: "d11:idx,d22:idx,...")

Backbone (PURE Point Transformer):
- Embedding inicial por punto (MLP compartido)
- Point Transformer layers (atención vectorial local con kNN + positional encoding relativo)
- Head punto-a-punto para logits [B,N,C]

Estrategia de eficiencia (para N=8192):
- Atención SOLO en M tokens (token_subsample, M<<N) para evitar O(N^2).
- PointTransformer opera en tokens con kNN local (attn_k).
- Luego se propagan logits a todos los puntos usando kNN tokens->full (prop_k).

Dataset esperado:
  data_dir/X_train.npz, Y_train.npz, X_val.npz, Y_val.npz, X_test.npz, Y_test.npz
  X: [B,N,3], Y: [B,N] con clases internas 0..C-1 (0=bg)

Ejemplo:
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python3 -u pointtransformer_pure_classic_final_v1.py \
  --data_dir /home/htaucare/Tesis_Amaro/data/Teeth_3ds/fixed_split/8192/upper_only_surf_global_excl_wisdom_seed42_aug2 \
  --out_dir  /home/htaucare/Tesis_Amaro/outputs/pointtransformer_pure/gpu0_run_fast_v1 \
  --epochs 120 --batch_size 8 --lr 2e-4 --weight_decay 1e-4 \
  --num_workers 6 --device cuda --use_amp --grad_clip 1.0 \
  --bg_index 0 --bg_weight 0.03 --d21_internal 8 \
  --neighbor_teeth "d11:1,d22:9" \
  --token_subsample 2048 --attn_k 16 --prop_k 3 --prop_chunk 2048 \
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


def _set_seed(seed: int = 42):
    # alias por compatibilidad con versiones previas
    return set_seed(seed)


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

    ⚠️ FIX:
    En inferencia, matplotlib + dataloader con muchos workers a veces da problemas silenciosos
    en algunos clusters. Para robustez, limitamos workers a <=2 (y jamás persistent_workers
    si nw==0). Si quieres forzarlo, deja --infer_num_workers explícito.
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
# FAST kNN (chunked) + indexing helpers
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
    ⚠️ Manteniendo simplicidad/robustez: token subsample por RAND determinístico (no FPS),
    porque FPS puro en PyTorch puede ser lento / costoso en CPU/loader.
    - M << N => igual reduce O(N^2) a O(M^2).
    - Para determinismo REAL: usar seed global + torch.Generator por batch (no imprescindible).

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
# POINT TRANSFORMER "PURE" (Vector Attention local) + tokens + propagación
# - Esto SÍ es un Point Transformer real (no nn.TransformerEncoderLayer)
# - Atención local por kNN en espacio xyz (sobre tokens)
# - Positional encoding RELATIVO (xyz_i - xyz_j) aprendido
# - Vector attention (peso por canal) + agregación
#
# Paper base (idea): Point Transformer (Zhao et al.)
# Implementación aquí: robusta, simple, 100% PyTorch, sin dependencias externas
# ============================================================

class PointTransformerLayer(nn.Module):
    """
    Capa Point Transformer (local vector attention) para puntos 3D.

    Entrada:
      - xyz:  [B,M,3]
      - feat: [B,M,D]

    Proceso (resumen):
      1) kNN sobre xyz para cada punto (neighbors K)
      2) Proyecciones:
         q = Wq feat_i
         k = Wk feat_j
         v = Wv feat_j
      3) Positional relativo:
         delta_xyz = xyz_i - xyz_j
         pe = phi(delta_xyz)  (MLP)
      4) Attention "vectorial" por canal:
         att = gamma(q - k + pe)   -> [B,M,K,D]
         att = softmax(att, dim=K) (por canal D, softmax sobre vecinos)
      5) Agregación:
         out_i = sum_j att_ij * (v_j + pe_ij)   (todo en D canales)
      6) Proyección final + residual + norm

    Nota:
      - softmax vectorial: hacemos softmax sobre dimensión de vecinos (K) para cada canal D.
      - Esto es pesado si M grande; por eso lo usas sobre tokens (M = token_subsample).
    """
    def __init__(
        self,
        d_model: int = 256,
        k: int = 16,
        pe_hidden: int = 64,
        attn_hidden: int = 128,
        dropout: float = 0.0,
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.k = int(k)

        # Proyecciones QKV
        self.to_q = nn.Linear(int(d_model), int(d_model), bias=False)
        self.to_k = nn.Linear(int(d_model), int(d_model), bias=False)
        self.to_v = nn.Linear(int(d_model), int(d_model), bias=False)

        # Positional encoding relativo (xyz_i - xyz_j) -> D
        self.phi = nn.Sequential(
            nn.Linear(3, int(pe_hidden)),
            nn.ReLU(inplace=True),
            nn.Linear(int(pe_hidden), int(d_model)),
        )

        # gamma: produce "vector attention" por canal D (no un escalar)
        # input: (q_i - k_j + pe_ij) -> D
        # output: att_ij -> D
        self.gamma = nn.Sequential(
            nn.Linear(int(d_model), int(attn_hidden)),
            nn.ReLU(inplace=True),
            nn.Linear(int(attn_hidden), int(d_model)),
        )

        self.out_proj = nn.Sequential(
            nn.Linear(int(d_model), int(d_model)),
            nn.Dropout(float(dropout)),
        )

        self.norm = nn.LayerNorm(int(d_model)) if bool(use_layernorm) else nn.Identity()
        self.drop = nn.Dropout(float(dropout))

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        """
        xyz:  [B,M,3]
        feat: [B,M,D]
        return: [B,M,D]
        """
        B, M, _ = xyz.shape
        D = feat.shape[-1]
        assert int(D) == int(self.d_model), f"d_model mismatch: feat D={D} vs layer d_model={self.d_model}"

        # -----------------------------
        # kNN sobre tokens (M puntos)
        # idx: [B,M,K] indices en [0..M-1]
        # -----------------------------
        with torch.amp.autocast(device_type="cuda", enabled=False):
            idx = knn_indices_chunked(
                q_xyz=_as_fp32(xyz),
                k_xyz=_as_fp32(xyz),
                k=self.k,
                chunk=2048
            )

        # -----------------------------
        # gather neighbors
        # neigh_xyz:  [B,M,K,3]
        # neigh_feat: [B,M,K,D]
        # -----------------------------
        neigh_xyz = index_points(xyz, idx)       # [B,M,K,3]
        neigh_feat = index_points(feat, idx)     # [B,M,K,D]

        # -----------------------------
        # QKV
        # q: [B,M,D]
        # k: [B,M,K,D]
        # v: [B,M,K,D]
        # -----------------------------
        q = self.to_q(feat)                      # [B,M,D]
        k = self.to_k(neigh_feat)                # [B,M,K,D]
        v = self.to_v(neigh_feat)                # [B,M,K,D]

        # -----------------------------
        # Positional encoding relativo
        # delta = xyz_i - xyz_j  (broadcast)
        # pe: [B,M,K,D]
        # -----------------------------
        delta = xyz.unsqueeze(2) - neigh_xyz     # [B,M,K,3]
        pe = self.phi(delta)                     # [B,M,K,D]

        # -----------------------------
        # Vector attention
        # att_logits = gamma(q_i - k_j + pe_ij)  -> [B,M,K,D]
        # softmax over K (neighbors) for each channel D
        # -----------------------------
        q_exp = q.unsqueeze(2)                   # [B,M,1,D]
        att_logits = self.gamma(q_exp - k + pe)  # [B,M,K,D]

        # softmax sobre vecinos (dim=2), por canal D
        att = F.softmax(att_logits, dim=2)       # [B,M,K,D]
        att = self.drop(att)

        # -----------------------------
        # Aggregation
        # msg = (v_j + pe_ij)
        # out = sum_j att_ij * msg_ij  -> [B,M,D]
        # -----------------------------
        msg = v + pe                              # [B,M,K,D]
        out = (att * msg).sum(dim=2)              # [B,M,D]

        out = self.out_proj(out)                  # [B,M,D]

        # Residual + Norm
        out = self.norm(feat + out)               # [B,M,D]
        return out


class PointTransformerBackbone(nn.Module):
    """
    Backbone Point Transformer puro (stack de capas locales).

    Entrada:
      xyz_tok: [B,M,3]
    Salida:
      feat_tok: [B,M,D]

    Nota:
      - Aquí mantenemos un embedding inicial desde xyz (MLP) + opcional pos_mlp.
      - Luego pasamos por L capas PointTransformerLayer.
    """
    def __init__(
        self,
        d_model: int = 256,
        embed_hidden: int = 128,
        depth: int = 4,
        k_attn: int = 16,
        pe_hidden: int = 64,
        attn_hidden: int = 128,
        dropout: float = 0.1,
        use_pos_mlp: bool = True,
        pos_hidden: int = 128,
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.depth = int(depth)
        self.use_pos_mlp = bool(use_pos_mlp)

        self.embed = PointMLPEmbed(
            d_model=int(d_model),
            hidden=int(embed_hidden),
            dropout=float(dropout),
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

        self.layers = nn.ModuleList([
            PointTransformerLayer(
                d_model=int(d_model),
                k=int(k_attn),
                pe_hidden=int(pe_hidden),
                attn_hidden=int(attn_hidden),
                dropout=float(dropout),
                use_layernorm=bool(use_layernorm),
            )
            for _ in range(int(depth))
        ])

    def forward(self, xyz_tok: torch.Tensor) -> torch.Tensor:
        """
        xyz_tok: [B,M,3]
        return feat_tok: [B,M,D]
        """
        x = self.embed(xyz_tok)  # [B,M,D]
        if self.pos is not None:
            x = x + self.pos(xyz_tok)

        for lyr in self.layers:
            x = lyr(xyz_tok, x)

        return x


class PointTransformerSegPureFast(nn.Module):
    """
    Segmentación punto-a-punto usando Point Transformer PURO en TOKENS:
      xyz(N) -> tokens(M) -> PointTransformerBackbone(M) -> head(M)-> logits_tok
      logits_tok -> propaga a N con kNN (N->M) => logits_full [B,N,C]

    Importante:
      - forward SIEMPRE retorna logits_full [B,N,C]
      - Esto mantiene training/eval/infer idéntico al resto de tus scripts.
    """
    def __init__(
        self,
        num_classes: int,
        d_model: int = 256,
        embed_hidden: int = 128,
        depth: int = 4,
        dropout: float = 0.1,
        use_pos_mlp: bool = True,
        pos_hidden: int = 128,
        # Point Transformer local attention params
        k_attn: int = 16,
        pe_hidden: int = 64,
        attn_hidden: int = 128,
        use_layernorm: bool = True,
        # Head
        head_hidden: int = 256,
        head_dropout: float = 0.1,
        # FAST
        token_subsample: int = 2048,
        prop_k: int = 3,
        prop_chunk: int = 2048,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.d_model = int(d_model)

        self.token_subsample = int(token_subsample)
        self.prop_k = int(prop_k)
        self.prop_chunk = int(prop_chunk)

        self.backbone = PointTransformerBackbone(
            d_model=int(d_model),
            embed_hidden=int(embed_hidden),
            depth=int(depth),
            k_attn=int(k_attn),
            pe_hidden=int(pe_hidden),
            attn_hidden=int(attn_hidden),
            dropout=float(dropout),
            use_pos_mlp=bool(use_pos_mlp),
            pos_hidden=int(pos_hidden),
            use_layernorm=bool(use_layernorm),
        )

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

        # --- token subsample (RAND stub; puedes cambiar luego a FPS real si quieres) ---
        M = int(min(self.token_subsample, N))
        if M < N:
            tok_idx = farthest_point_sample_stub(xyz, M)     # [B,M]
            tok_xyz = index_points(xyz, tok_idx)             # [B,M,3]
        else:
            tok_idx = None
            tok_xyz = xyz

        # --- backbone puro ---
        feat_tok = self.backbone(tok_xyz)                    # [B,M,D]

        # --- logits en tokens ---
        logits_tok = self.head(feat_tok)                     # [B,M,C]

        # --- si M==N, listo ---
        if M == N:
            return logits_tok

        # --- propaga logits tokens a todos los puntos N usando kNN: N->M ---
        with torch.amp.autocast(device_type="cuda", enabled=False):
            idx_nm = knn_indices_chunked(
                q_xyz=_as_fp32(xyz),
                k_xyz=_as_fp32(tok_xyz),
                k=self.prop_k,
                chunk=self.prop_chunk
            )

        logits_nk = index_points(logits_tok, idx_nm)         # [B,N,k,C]
        logits_full = logits_nk.mean(dim=2)                  # [B,N,C]
        return logits_full

# ============================================================
# TRAIN / EVAL (estilo v4) + VECINOS
# - IMPORTANTE: aquí definimos TODO lo que te faltaba:
#   ✅ run_epoch
#   ✅ _format_neighbor_console
#   ✅ helpers de métricas desde logits
#   ✅ _merge_metrics, etc.
# ============================================================

@torch.no_grad()
def _compute_metrics_from_logits(
    logits: torch.Tensor,
    y: torch.Tensor,
    C: int,
    d21_idx: int,
    bg: int
) -> Dict[str, float]:
    """
    logits: [B,N,C]
    y:      [B,N]
    """
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
    """
    Calcula métricas binarias por diente vecino (lista arbitraria).
    Devuelve keys:
      {name}_acc, {name}_f1, {name}_iou, {name}_bin_acc_all
    """
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
    Corre 1 época train/val/test con:
      - loss promedio
      - métricas macro sin bg + d21 binario + pred_bg_frac
      - opcional métricas de dientes vecinos

    Nota:
      - En train: calculamos métricas en modo eval sobre el MISMO batch (como tu PointNet/DGCNN),
        para comparabilidad (dropout off).
      - Esto cuesta un forward extra, pero es lo que tú vienes usando como estándar.
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

    # vecinos (acumuladores)
    nb_sums: Dict[str, float] = {}
    if neighbor_list:
        for name, _ in neighbor_list:
            for k in ("acc", "f1", "iou", "bin_acc_all"):
                nb_sums[f"{name}_{k}"] = 0.0

    n_batches = 0

    if not train:
        model.eval()

    for batch in loader:
        # dataset devuelve (xyz,y) aquí
        xyz, y = batch
        xyz = xyz.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if train:
            model.train(True)
            assert optimizer is not None, "optimizer requerido en train"
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
                logits_train = model(xyz)  # [B,N,C]
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

            # métricas en modo eval (dropout off)
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
        acc_all_sum += metrics["acc_all"]
        acc_no_bg_sum += metrics["acc_no_bg"]
        f1m_sum += metrics["f1_macro"]
        ioum_sum += metrics["iou_macro"]
        d21_acc_sum += metrics["d21_acc"]
        d21_f1_sum += metrics["d21_f1"]
        d21_iou_sum += metrics["d21_iou"]
        d21_bin_all_sum += metrics["d21_bin_acc_all"]
        pred_bg_frac_sum += metrics["pred_bg_frac"]

        if neighbor_list:
            for name, _ in neighbor_list:
                for k in ("acc", "f1", "iou", "bin_acc_all"):
                    nb_sums[f"{name}_{k}"] += float(nb_metrics.get(f"{name}_{k}", 0.0))

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
            out[k] = v / n

    return out


run_epoch.scaler = None


# ============================================================
# ARGPARSE (idéntico espíritu v4, + hp de PointTransformer puro)
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Point Transformer PURE Segmentation FAST (v1) — mismo esquema outputs que PointNet/DGCNN"
    )

    # DATA / IO
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--index_csv", type=str, default=None)

    # TRAIN
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--use_amp", action="store_true")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)

    # LOGGING / ROBUSTEZ
    p.add_argument("--print_every", type=int, default=1)
    p.add_argument("--save_every", type=int, default=1)

    # BG / D21
    p.add_argument("--bg_index", type=int, default=0)
    p.add_argument("--bg_weight", type=float, default=0.03)
    p.add_argument("--d21_internal", type=int, default=8)

    # PURE POINT TRANSFORMER HP
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--embed_hidden", type=int, default=128)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--k_attn", type=int, default=16)
    p.add_argument("--pe_hidden", type=int, default=64)
    p.add_argument("--attn_hidden", type=int, default=128)
    p.add_argument("--pos_hidden", type=int, default=128)
    p.add_argument("--head_hidden", type=int, default=256)
    p.add_argument("--head_dropout", type=float, default=0.1)
    p.add_argument("--no_pos_mlp", action="store_true")

    # FAST OPTIONS
    p.add_argument("--token_subsample", type=int, default=2048)
    p.add_argument("--prop_k", type=int, default=3)
    p.add_argument("--prop_chunk", type=int, default=2048)

    # NEIGHBORS
    p.add_argument("--neighbor_teeth", type=str, default="")
    p.add_argument("--neighbor_eval_split", type=str, default="val")
    p.add_argument("--neighbor_every", type=int, default=1)

    # INFER
    p.add_argument("--do_infer", action="store_true")
    p.add_argument("--infer_examples", type=int, default=12)
    p.add_argument("--infer_split", type=str, default="test")
    p.add_argument("--infer_num_workers", type=int, default=None)

    return p.parse_args()

def main():
    args = parse_args()

    # seed (sí, puedes cambiarla con --seed)
    _set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    # logs
    errors_log = out_dir / "errors.log"
    run_log = out_dir / "run.log"

    log_line("=== RUN START (PURE PointTransformer v1) ===", run_log, also_print=True)
    log_line(f"data_dir={data_dir}", run_log, also_print=True)
    log_line(f"out_dir={out_dir}", run_log, also_print=True)
    log_line(f"device={device}", run_log, also_print=True)

    # ------------------------------------------------
    # DATA
    # ------------------------------------------------
    dl_tr, dl_va, dl_te = make_loaders_v3(
        data_dir=data_dir,
        bs=args.batch_size,
        nw=args.num_workers,
        normalize=True,
    )

    # ------------------------------------------------
    # NUM CLASSES
    # ------------------------------------------------
    C = infer_num_classes_from_npz(data_dir)
    log_line(f"num_classes (inferred) = {C}", run_log, also_print=True)

    # ------------------------------------------------
    # MODEL (PURE POINT TRANSFORMER, FAST tokens+prop)
    # ------------------------------------------------
    model = PointTransformerSegPureFast(
        num_classes=C,
        d_model=args.d_model,
        embed_hidden=args.embed_hidden,
        depth=args.depth,
        dropout=args.dropout,
        use_pos_mlp=not args.no_pos_mlp,
        pos_hidden=args.pos_hidden,
        # PointTransformer specifics
        k_attn=args.k_attn,
        pe_hidden=args.pe_hidden,
        attn_hidden=args.attn_hidden,
        # head
        head_hidden=args.head_hidden,
        head_dropout=args.head_dropout,
        # fast
        token_subsample=args.token_subsample,
        prop_k=args.prop_k,
        prop_chunk=args.prop_chunk,
    ).to(device)

    # ------------------------------------------------
    # LOSS
    # ------------------------------------------------
    weights = torch.ones(C, device=device)
    weights[int(args.bg_index)] = float(args.bg_weight)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    # ------------------------------------------------
    # OPT / SCHED
    # ------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(args.epochs)
    )

    # ------------------------------------------------
    # NEIGHBORS
    # ------------------------------------------------
    neighbor_list = parse_neighbor_teeth(args.neighbor_teeth)
    if neighbor_list:
        log_line(f"neighbor_teeth parsed: {neighbor_list}", run_log, also_print=True)

    # ------------------------------------------------
    # META
    # ------------------------------------------------
    meta = vars(args).copy()
    meta["num_classes"] = int(C)
    meta["script_name"] = "pointtransformer_pure_classic_final_v1.py"
    meta["start_time"] = _now_str()
    meta["torch_version"] = torch.__version__
    meta["cuda_available"] = bool(torch.cuda.is_available())
    if torch.cuda.is_available():
        try:
            meta["cuda_device_name"] = torch.cuda.get_device_name(0)
        except Exception:
            meta["cuda_device_name"] = ""
    save_json(meta, out_dir / "run_meta.json")

    history = defaultdict(list)
    best_val = -1.0
    best_epoch = 0

    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"
    csv_path = out_dir / "metrics_epoch.csv"
    jsonl_path = out_dir / "history_epoch.jsonl"

    # ------------------------------------------------
    # CSV HEADER
    # ------------------------------------------------
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = [
            "epoch",
            "lr",
            "train_loss", "val_loss",
            "train_f1m", "val_f1m",
            "train_ioum", "val_ioum",
            "train_acc_all", "val_acc_all",
            "train_acc_no_bg", "val_acc_no_bg",
            "train_d21_acc", "val_d21_acc",
            "train_d21_f1", "val_d21_f1",
            "train_d21_iou", "val_d21_iou",
            "train_d21_bin_acc_all", "val_d21_bin_acc_all",
            "train_pred_bg_frac", "val_pred_bg_frac",
        ]
        for name, _ in neighbor_list:
            header += [f"val_{name}_acc", f"val_{name}_f1", f"val_{name}_iou", f"val_{name}_bin_acc_all"]
        writer.writerow(header)

    # ------------------------------------------------
    # TRAIN LOOP
    # ------------------------------------------------
    t0 = time.time()

    for epoch in range(1, int(args.epochs) + 1):
        lr_now = _get_lr(optimizer)

        # TRAIN (sin vecinos por costo)
        train_metrics = run_epoch(
            model=model,
            loader=dl_tr,
            optimizer=optimizer,
            loss_fn=loss_fn,
            C=C,
            d21_idx=int(args.d21_internal),
            device=device,
            bg=int(args.bg_index),
            train=True,
            use_amp=bool(args.use_amp),
            grad_clip=float(args.grad_clip) if args.grad_clip is not None else None,
            neighbor_list=[],
        )

        # VAL (vecinos cada neighbor_every)
        do_nb = (bool(neighbor_list) and (epoch % int(args.neighbor_every) == 0))
        val_metrics = run_epoch(
            model=model,
            loader=dl_va,
            optimizer=None,
            loss_fn=loss_fn,
            C=C,
            d21_idx=int(args.d21_internal),
            device=device,
            bg=int(args.bg_index),
            train=False,
            use_amp=bool(args.use_amp),
            grad_clip=None,
            neighbor_list=neighbor_list if do_nb else [],
        )

        scheduler.step()

        # history
        for k, v in train_metrics.items():
            history[f"train_{k}"].append(float(v))
        for k, v in val_metrics.items():
            history[f"val_{k}"].append(float(v))
        history["lr"].append(float(lr_now))

        # JSONL por época
        ep_obj = {
            "epoch": int(epoch),
            "lr": float(lr_now),
            "train": {k: float(v) for k, v in train_metrics.items()},
            "val": {k: float(v) for k, v in val_metrics.items()},
        }
        append_jsonl(ep_obj, jsonl_path)

        # PRINT consola
        if epoch % int(args.print_every) == 0:
            nb_str = _format_neighbor_console(val_metrics, neighbor_list) if do_nb else ""
            msg = (
                f"[{epoch}/{args.epochs}] "
                f"train loss={train_metrics['loss']:.4f} "
                f"f1m={train_metrics['f1_macro']:.3f} ioum={train_metrics['iou_macro']:.3f} "
                f"acc_all={train_metrics['acc_all']:.3f} acc_no_bg={train_metrics['acc_no_bg']:.3f} | "
                f"val loss={val_metrics['loss']:.4f} "
                f"f1m={val_metrics['f1_macro']:.3f} ioum={val_metrics['iou_macro']:.3f} "
                f"acc_all={val_metrics['acc_all']:.3f} acc_no_bg={val_metrics['acc_no_bg']:.3f} | "
                f"d21 acc={val_metrics['d21_acc']:.3f} f1={val_metrics['d21_f1']:.3f} iou={val_metrics['d21_iou']:.3f} | "
                f"d21(bin all) acc={val_metrics['d21_bin_acc_all']:.3f} | "
                f"pred_bg_frac(val)={val_metrics['pred_bg_frac']:.3f} "
                f"lr={lr_now:.2e}"
                f"{nb_str}"
            )
            log_line(msg, run_log, also_print=True)

        # CSV row
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            row = [
                int(epoch),
                float(lr_now),
                float(train_metrics["loss"]), float(val_metrics["loss"]),
                float(train_metrics["f1_macro"]), float(val_metrics["f1_macro"]),
                float(train_metrics["iou_macro"]), float(val_metrics["iou_macro"]),
                float(train_metrics["acc_all"]), float(val_metrics["acc_all"]),
                float(train_metrics["acc_no_bg"]), float(val_metrics["acc_no_bg"]),
                float(train_metrics["d21_acc"]), float(val_metrics["d21_acc"]),
                float(train_metrics["d21_f1"]), float(val_metrics["d21_f1"]),
                float(train_metrics["d21_iou"]), float(val_metrics["d21_iou"]),
                float(train_metrics["d21_bin_acc_all"]), float(val_metrics["d21_bin_acc_all"]),
                float(train_metrics["pred_bg_frac"]), float(val_metrics["pred_bg_frac"]),
            ]
            for name, _ in neighbor_list:
                row += [
                    float(val_metrics.get(f"{name}_acc", 0.0)),
                    float(val_metrics.get(f"{name}_f1", 0.0)),
                    float(val_metrics.get(f"{name}_iou", 0.0)),
                    float(val_metrics.get(f"{name}_bin_acc_all", 0.0)),
                ]
            writer.writerow(row)

        # BEST
        if float(val_metrics["f1_macro"]) > float(best_val):
            best_val = float(val_metrics["f1_macro"])
            best_epoch = int(epoch)
            torch.save({"model": model.state_dict(), "epoch": int(epoch)}, best_path)

        # LAST + history parcial
        if epoch % int(args.save_every) == 0:
            torch.save({"model": model.state_dict(), "epoch": int(epoch)}, last_path)
            try:
                save_json(dict(history), out_dir / "history.json")
            except Exception as e:
                log_line(f"[WARN] save history.json falló: {repr(e)}", errors_log, also_print=True)

    # fin train
    save_json(dict(history), out_dir / "history.json")

    # ------------------------------------------------
    # LOAD BEST
    # ------------------------------------------------
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])

    # ------------------------------------------------
    # TEST
    # ------------------------------------------------
    test_metrics = run_epoch(
        model=model,
        loader=dl_te,
        optimizer=None,
        loss_fn=loss_fn,
        C=C,
        d21_idx=int(args.d21_internal),
        device=device,
        bg=int(args.bg_index),
        train=False,
        use_amp=bool(args.use_amp),
        grad_clip=None,
        neighbor_list=neighbor_list if str(args.neighbor_eval_split).lower() in ("test", "both") else [],
    )

    test_out = dict(test_metrics)
    test_out["best_epoch"] = int(best_epoch)
    test_out["best_val_f1_macro"] = float(best_val)
    save_json(test_out, out_dir / "test_metrics.json")

    nb_str_test = _format_neighbor_console(test_metrics, neighbor_list) if (
        neighbor_list and str(args.neighbor_eval_split).lower() in ("test", "both")
    ) else ""

    log_line(
        f"[TEST] loss={test_metrics['loss']:.4f} "
        f"f1m={test_metrics['f1_macro']:.3f} ioum={test_metrics['iou_macro']:.3f} "
        f"acc_all={test_metrics['acc_all']:.3f} acc_no_bg={test_metrics['acc_no_bg']:.3f} | "
        f"d21 acc={test_metrics['d21_acc']:.3f} f1={test_metrics['d21_f1']:.3f} iou={test_metrics['d21_iou']:.3f} | "
        f"d21(bin all) acc={test_metrics['d21_bin_acc_all']:.3f} | "
        f"pred_bg_frac(test)={test_metrics['pred_bg_frac']:.3f}"
        f"{nb_str_test}",
        run_log,
        also_print=True,
    )

    # ------------------------------------------------
    # PLOTS (Train vs Val)
    # ------------------------------------------------
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for k in ["loss", "f1_macro", "iou_macro", "d21_f1"]:
        try:
            if f"train_{k}" in history and f"val_{k}" in history:
                plot_train_val(
                    name=k,
                    y_tr=history[f"train_{k}"],
                    y_va=history[f"val_{k}"],
                    out_png=plot_dir / f"{k}.png",
                    best_epoch=best_epoch,
                )
        except Exception as e:
            log_line(f"[WARN] plot {k} falló: {repr(e)}", errors_log, also_print=True)

    # ------------------------------------------------
    # INFERENCIA TRAZABLE (robusta)
    # ------------------------------------------------
    if args.do_infer:
        infer_dir = out_dir / "inference"
        infer_dir.mkdir(parents=True, exist_ok=True)

        infer_all_dir = infer_dir / "inference_all"
        infer_err_dir = infer_dir / "inference_errors"
        infer_d21_dir = infer_dir / "inference_d21"
        infer_all_dir.mkdir(parents=True, exist_ok=True)
        infer_err_dir.mkdir(parents=True, exist_ok=True)
        infer_d21_dir.mkdir(parents=True, exist_ok=True)

        if args.infer_num_workers is None:
            infer_nw = int(min(int(args.num_workers), 2))
        else:
            infer_nw = int(args.infer_num_workers)

        log_line(
            f"[infer] split={args.infer_split} examples={args.infer_examples} infer_num_workers={infer_nw}",
            run_log,
            also_print=True,
        )

        loader = make_infer_loader_v3(
            data_dir=data_dir,
            split=args.infer_split,
            bs=args.batch_size,
            nw=infer_nw,
            normalize=True,
        )

        # index csv discovery
        index_csv_path = None
        if args.index_csv:
            index_csv_path = Path(args.index_csv)
            if not index_csv_path.exists():
                log_line(f"[WARN] args.index_csv no existe: {index_csv_path}", errors_log, also_print=True)
                index_csv_path = None
        else:
            index_csv_path = _discover_index_csv(data_dir, args.infer_split)

        if index_csv_path is not None:
            log_line(f"[infer] usando index_csv: {index_csv_path}", run_log, also_print=True)
        else:
            log_line(f"[infer] no se encontró index_csv (se usará tag por row_i)", run_log, also_print=True)

        index_map = _read_index_csv(index_csv_path) if index_csv_path else None

        manifest_path = infer_dir / "inference_manifest.csv"
        with open(manifest_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "split", "row_i", "tag",
                "sample_name", "jaw", "idx_global", "path",
                "png_all", "png_errors", "png_d21"
            ])

            model.eval()
            done = 0

            with torch.no_grad():
                for batch in loader:
                    try:
                        xyz, y, row_i = batch
                        xyz = xyz.to(device, non_blocking=True)
                        y = y.to(device, non_blocking=True)

                        with torch.amp.autocast(device_type=device.type, enabled=(args.use_amp and device.type == "cuda")):
                            logits = model(xyz)

                        pred = logits.argmax(dim=-1)

                        B = xyz.shape[0]
                        for b in range(B):
                            if done >= int(args.infer_examples):
                                break

                            row_idx = int(row_i[b].item())
                            tag = f"{args.infer_split}_sample_{row_idx}"

                            sample_name = ""
                            jaw = ""
                            idx_global = ""
                            src_path = ""

                            if index_map and row_idx in index_map:
                                sample_name = index_map[row_idx].get("sample_name", "")
                                jaw = index_map[row_idx].get("jaw", "")
                                idx_global = index_map[row_idx].get("idx_global", "")
                                src_path = index_map[row_idx].get("path", "")
                                if sample_name:
                                    tag = _sanitize_tag(sample_name)

                            xyz_np = xyz[b].detach().cpu().numpy()
                            y_np = y[b].detach().cpu().numpy()
                            pr_np = pred[b].detach().cpu().numpy()

                            png_all = infer_all_dir / f"{tag}_all.png"
                            png_err = infer_err_dir / f"{tag}_errors.png"
                            png_d21 = infer_d21_dir / f"{tag}_d21.png"

                            plot_pointcloud_all_classes(
                                xyz_np, y_np, pr_np,
                                png_all,
                                C=C,
                                title=tag,
                                errors_log=errors_log,
                            )

                            plot_errors(
                                xyz_np, y_np, pr_np,
                                png_err,
                                bg=int(args.bg_index),
                                title=tag,
                                errors_log=errors_log,
                            )

                            plot_d21_focus(
                                xyz_np, y_np, pr_np,
                                png_d21,
                                d21_idx=int(args.d21_internal),
                                bg=int(args.bg_index),
                                title=tag,
                                errors_log=errors_log,
                            )

                            writer.writerow([
                                str(args.infer_split),
                                row_idx,
                                tag,
                                sample_name,
                                jaw,
                                idx_global,
                                src_path,
                                str(png_all.name),
                                str(png_err.name),
                                str(png_d21.name),
                            ])

                            done += 1

                        if done >= int(args.infer_examples):
                            break

                    except Exception as e:
                        log_line(f"[WARN] infer batch falló: {repr(e)}", errors_log, also_print=True)
                        continue

        log_line(f"[infer] listo: se generaron {done} ejemplos en {infer_dir}", run_log, also_print=True)

    total_time = time.time() - t0
    log_line(
        f"[done] Entrenamiento terminado en {_fmt_hms(total_time)}. "
        f"best_epoch={best_epoch} best_val_f1={best_val:.4f}",
        run_log,
        also_print=True,
    )


if __name__ == "__main__":
    main()


"""
============================================================
EJEMPLOS DE EJECUCIÓN (PURE PointTransformer v1)
============================================================

1) Entrenar + inferencia (GPU 0), con vecinos en VAL:
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python3 -u pointtransformer_pure_classic_final_v1.py \
  --data_dir /home/htaucare/Tesis_Amaro/data/Teeth_3ds/fixed_split/8192/upper_only_surf_global_excl_wisdom_seed42_aug2 \
  --out_dir  /home/htaucare/Tesis_Amaro/outputs/point_transformer_pure/gpu0_run_v1 \
  --epochs 120 --batch_size 8 --lr 2e-4 --weight_decay 1e-4 \
  --num_workers 6 --device cuda --use_amp --grad_clip 1.0 --seed 42 \
  --bg_index 0 --bg_weight 0.03 --d21_internal 8 \
  --neighbor_teeth "d11:1,d22:9" --neighbor_every 1 --neighbor_eval_split val \
  --token_subsample 2048 --prop_k 3 --prop_chunk 2048 \
  --k_attn 16 --depth 4 --d_model 256 \
  --do_infer --infer_examples 12 --infer_split test

2) Máxima robustez IO/matplotlib en infer (0 workers):
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python3 -u pointtransformer_pure_classic_final_v1.py \
  --data_dir /home/htaucare/Tesis_Amaro/data/Teeth_3ds/fixed_split/8192/upper_only_surf_global_excl_wisdom_seed42_aug2 \
  --out_dir  /home/htaucare/Tesis_Amaro/outputs/point_transformer_pure/gpu0_run_v1_infer0w \
  --epochs 120 --batch_size 8 --lr 2e-4 --weight_decay 1e-4 \
  --num_workers 6 --device cuda --use_amp --grad_clip 1.0 --seed 42 \
  --bg_index 0 --bg_weight 0.03 --d21_internal 8 \
  --neighbor_teeth "d11:1,d22:9" --neighbor_every 1 --neighbor_eval_split val \
  --token_subsample 2048 --prop_k 3 --prop_chunk 2048 \
  --k_attn 16 --depth 4 --d_model 256 \
  --do_infer --infer_examples 12 --infer_split test \
  --infer_num_workers 0

3) Correr en GPU 1:
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python3 -u pointtransformer_pure_classic_final_v1.py \
  --data_dir /home/htaucare/Tesis_Amaro/data/Teeth_3ds/fixed_split/8192/upper_only_surf_global_excl_wisdom_seed42_aug2 \
  --out_dir  /home/htaucare/Tesis_Amaro/outputs/point_transformer_pure/gpu1_run_v1 \
  --epochs 120 --batch_size 8 --lr 2e-4 --weight_decay 1e-4 \
  --num_workers 6 --device cuda --use_amp --grad_clip 1.0 --seed 42 \
  --bg_index 0 --bg_weight 0.03 --d21_internal 8 \
  --neighbor_teeth "d11:1,d22:9" --neighbor_every 1 --neighbor_eval_split val \
  --token_subsample 2048 --prop_k 3 --prop_chunk 2048 \
  --k_attn 16 --depth 4 --d_model 256 \
  --do_infer --infer_examples 12 --infer_split test

Outputs esperados en out_dir:
- run_meta.json
- run.log, errors.log
- metrics_epoch.csv
- history.json + history_epoch.jsonl
- best.pt, last.pt
- test_metrics.json
- plots/*.png
- inference/inference_manifest.csv
- inference/inference_all/*.png
- inference/inference_errors/*.png
- inference/inference_d21/*.png
"""
