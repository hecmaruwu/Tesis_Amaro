#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dgcnn_infer_final_v1.py

Inferencia + visualización para DGCNN dental 3D
compatible con checkpoints guardados como:

1) state_dict puro:
   torch.save(model.state_dict(), "best.pt")

2) checkpoint tipo dict:
   torch.save({
       "epoch": ...,
       "model_state": model.state_dict(),
       ...
   }, "best.pt")

FIX aplicado:
✅ load_checkpoint_flexible(...) para soportar ambas variantes
✅ NO cambia outputs, plots ni funcionalidad general
"""

import os
import re
import csv
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# NORMALIZACIÓN
# ============================================================

def normalize_unit_sphere(xyz: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    xyz: [N,3]
    centra por media y escala por radio máximo.
    """
    center = xyz.mean(dim=0, keepdim=True)
    xyz = xyz - center
    scale = torch.norm(xyz, dim=1).max().clamp_min(eps)
    return xyz / scale


# ============================================================
# DATASET NPZ
# ============================================================

class NPZSplitDataset(Dataset):
    """
    Espera:
      X_{split}.npz con key "X": [M,N,3]
      Y_{split}.npz con key "Y": [M,N]
    """
    def __init__(self, data_dir: Path, split: str, normalize: bool = True):
        self.data_dir = Path(data_dir)
        self.split = str(split)

        xp = self.data_dir / f"X_{self.split}.npz"
        yp = self.data_dir / f"Y_{self.split}.npz"

        self.X = np.load(xp)["X"]
        self.Y = np.load(yp)["Y"]
        self.normalize = bool(normalize)

        assert self.X.shape[0] == self.Y.shape[0], "M mismatch X vs Y"
        assert self.X.shape[1] == self.Y.shape[1], "N mismatch X vs Y"
        assert self.X.shape[2] == 3, "X debe ser [M,N,3]"

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
    if isinstance(label_map, dict) and len(label_map) > 0:
        try:
            mx = max(int(v) for v in label_map.values())
            return int(mx + 1)
        except Exception:
            pass

    maxy = -1
    for split in ("train", "val", "test"):
        yp = Path(data_dir) / f"Y_{split}.npz"
        if yp.exists():
            y = np.load(yp)["Y"]
            maxy = max(maxy, int(y.max()))

    if maxy < 0:
        raise RuntimeError("No se pudo inferir num_classes.")
    return int(maxy + 1)


# ============================================================
# TRAZABILIDAD index_{split}.csv
# ============================================================

def discover_index_csv(data_dir: Path, split: str) -> Optional[Path]:
    """
    Busca index_{split}.csv en:
      1) data_dir/index_{split}.csv
      2) ancestros directos
      3) dentro de Teeth_3ds si aparece en el árbol
    """
    data_dir = Path(data_dir).resolve()
    target = f"index_{split}.csv"

    cand = data_dir / target
    if cand.exists():
        return cand

    parents = [data_dir] + list(data_dir.parents)

    for p in parents[:12]:
        cand = p / target
        if cand.exists():
            return cand

    for p in parents:
        if p.name.lower() == "teeth_3ds":
            try:
                for sub in p.rglob(target):
                    return sub
            except Exception:
                return None
            break

    return None


def read_index_csv(p: Optional[Path]) -> Optional[Dict[int, Dict[str, str]]]:
    """
    Retorna:
      { row_idx_int : {col: value, ...}, ... }
    """
    if p is None:
        return None
    p = Path(p)
    if not p.exists():
        return None

    out: Dict[int, Dict[str, str]] = {}
    try:
        with p.open("r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            rows = list(r)

        if not rows:
            return None

        idx_col = None
        for c in ("idx", "index", "row", "i", "row_i"):
            if c in rows[0]:
                idx_col = c
                break

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


def sanitize_tag(s: str, maxlen: int = 80) -> str:
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
# DGCNN
# ============================================================

def pairwise_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    x: [B,N,D]
    y: [B,M,D]
    retorna: [B,N,M]
    """
    xx = (x ** 2).sum(dim=-1, keepdim=True)
    yy = (y ** 2).sum(dim=-1, keepdim=True).transpose(1, 2)
    xy = torch.bmm(x, y.transpose(1, 2))
    return xx + yy - 2.0 * xy


@torch.no_grad()
def knn_indices_chunked(x: torch.Tensor, k: int, chunk_size: int = 1024) -> torch.Tensor:
    """
    x: [B,N,D]
    retorna idx: [B,N,k]
    """
    B, N, D = x.shape
    k = int(k)
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        chunk_size = N

    idx_out = torch.empty((B, N, k), device=x.device, dtype=torch.long)

    for s in range(0, N, chunk_size):
        e = min(N, s + chunk_size)
        xq = x[:, s:e, :]
        dist = pairwise_dist(xq, x)

        q = e - s
        ar = torch.arange(q, device=x.device)
        dist[:, ar, s + ar] = float("inf")

        _, idx = dist.topk(k=k, dim=-1, largest=False, sorted=False)
        idx_out[:, s:e, :] = idx

    return idx_out


def get_graph_feature(
    x: torch.Tensor,
    k: int,
    idx: Optional[torch.Tensor] = None,
    knn_chunk_size: int = 1024,
) -> torch.Tensor:
    """
    x: [B,C,N]
    retorna: [B,2C,N,k]
    """
    B, C, N = x.shape
    xt = x.transpose(1, 2).contiguous()

    if idx is None:
        idx = knn_indices_chunked(xt, k=int(k), chunk_size=int(knn_chunk_size))

    idx_base = torch.arange(0, B, device=x.device).view(-1, 1, 1) * N
    idx = idx + idx_base
    idx = idx.reshape(-1)

    neigh = xt.reshape(B * N, C)[idx, :].view(B, N, int(k), C)
    center = xt.view(B, N, 1, C).expand(-1, -1, int(k), -1)

    edge = torch.cat((neigh - center, center), dim=3)
    return edge.permute(0, 3, 1, 2).contiguous()


class EdgeConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2 * int(c_in), int(c_out), kernel_size=1, bias=False),
            nn.BatchNorm2d(int(c_out)),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, k: int, knn_chunk_size: int) -> torch.Tensor:
        feat = get_graph_feature(x, k=int(k), idx=None, knn_chunk_size=int(knn_chunk_size))
        feat = self.conv(feat)
        x = feat.max(dim=-1)[0]
        return x


class DGCNNSeg(nn.Module):
    def __init__(self, num_classes: int, k: int = 20, emb_dims: int = 1024, dropout: float = 0.5):
        super().__init__()
        self.k = int(k)
        self.emb_dims = int(emb_dims)

        self.ec1 = EdgeConvBlock(c_in=3,   c_out=64)
        self.ec2 = EdgeConvBlock(c_in=64,  c_out=64)
        self.ec3 = EdgeConvBlock(c_in=64,  c_out=128)
        self.ec4 = EdgeConvBlock(c_in=128, c_out=256)

        self.conv_global = nn.Sequential(
            nn.Conv1d(512, self.emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.emb_dims),
            nn.ReLU(inplace=True),
        )

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
        B, N, _ = xyz.shape
        x = xyz.transpose(1, 2).contiguous()

        x1 = self.ec1(x, k=self.k, knn_chunk_size=int(knn_chunk_size))
        x2 = self.ec2(x1, k=self.k, knn_chunk_size=int(knn_chunk_size))
        x3 = self.ec3(x2, k=self.k, knn_chunk_size=int(knn_chunk_size))
        x4 = self.ec4(x3, k=self.k, knn_chunk_size=int(knn_chunk_size))

        x_cat = torch.cat((x1, x2, x3, x4), dim=1)

        g = self.conv_global(x_cat)
        g = torch.max(g, dim=2, keepdim=True)[0]
        g = g.repeat(1, 1, N)

        feat = torch.cat((x_cat, g), dim=1)
        feat = self.conv1(feat)
        feat = self.dp1(feat)
        feat = self.conv2(feat)
        feat = self.dp2(feat)

        logits = self.classifier(feat).transpose(1, 2).contiguous()
        return logits

# ============================================================
# CHECKPOINT LOAD FLEXIBLE (FIX CLAVE)
# ============================================================

def load_checkpoint_flexible(model: nn.Module, model_path: Path, device: torch.device):
    """
    Soporta:
      1) state_dict puro
      2) dict con key 'model_state'
      3) dict con key 'model'
    """
    model_path = Path(model_path)
    obj = torch.load(model_path, map_location=device)

    if isinstance(obj, dict):
        if "model_state" in obj:
            model.load_state_dict(obj["model_state"], strict=True)
            return obj
        if "model" in obj:
            model.load_state_dict(obj["model"], strict=True)
            return obj

        # intento directo por si el dict YA es un state_dict
        try:
            model.load_state_dict(obj, strict=True)
            return {"raw_state_dict": True}
        except Exception:
            pass

    raise RuntimeError(
        f"No pude interpretar el checkpoint en {model_path}. "
        f"Esperaba state_dict puro o dict con 'model_state' / 'model'."
    )


# ============================================================
# MÉTRICAS
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
    """
    pred = logits.argmax(dim=-1)

    pred_f = pred.reshape(-1)
    tgt_f = target.reshape(-1)

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
    prec_list, rec_list, f1_list, iou_list = [], [], [], []

    for c in range(int(C)):
        if c == int(bg):
            continue

        pred_c = (pred_f == c)
        tgt_c = (tgt_f == c)

        tp = (pred_c & tgt_c).sum().item()
        fp = (pred_c & (~tgt_c)).sum().item()
        fn = ((~pred_c) & tgt_c).sum().item()

        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        f1 = 2.0 * prec * rec / (prec + rec + eps)
        iou = tp / (tp + fp + fn + eps)

        prec_list.append(prec)
        rec_list.append(rec)
        f1_list.append(f1)
        iou_list.append(iou)

    if len(f1_list) > 0:
        prec_m = float(np.mean(prec_list))
        rec_m = float(np.mean(rec_list))
        f1_m = float(np.mean(f1_list))
        iou_m = float(np.mean(iou_list))
    else:
        prec_m = rec_m = f1_m = iou_m = 0.0

    pred_bg_frac = float((pred_f == int(bg)).sum().item()) / float(total + 1e-9)

    return acc_all, acc_no_bg, prec_m, rec_m, f1_m, iou_m, pred_bg_frac


@torch.no_grad()
def compute_d21_binary_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    d21_idx: int,
):
    """
    d21 como positivo, resto negativo.
    """
    pred = logits.argmax(dim=-1)

    pred_bin = (pred == int(d21_idx))
    tgt_bin = (target == int(d21_idx))

    eps = 1e-9

    tp = (pred_bin & tgt_bin).sum().item()
    fp = (pred_bin & (~tgt_bin)).sum().item()
    fn = ((~pred_bin) & tgt_bin).sum().item()
    tn = ((~pred_bin) & (~tgt_bin)).sum().item()

    pos = tgt_bin.sum().item()
    if pos > 0:
        acc_cls = float(tp) / float(pos + eps)
    else:
        acc_cls = 0.0

    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = 2.0 * prec * rec / (prec + rec + eps)
    iou = tp / (tp + fp + fn + eps)
    bin_acc_all = float(tp + tn) / float(tp + tn + fp + fn + eps)

    return float(acc_cls), float(f1), float(iou), float(bin_acc_all)


# ============================================================
# PLOTS 3D
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
    Dibuja GT vs Pred con colores por clase.
    """
    xyz = to_np(xyz)
    gt = to_np(gt)
    pred = to_np(pred)

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
        ax.view_init(elev=20, azim=45)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_error_map(
    xyz,
    gt,
    pred,
    save_path: Path,
    title: str = "",
):
    """
    Gris = correcto, rojo = error.
    """
    xyz = to_np(xyz)
    gt = to_np(gt)
    pred = to_np(pred)

    ok = (gt == pred)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")

    if ok.any():
        ax.scatter(
            xyz[ok, 0], xyz[ok, 1], xyz[ok, 2],
            s=2, color="lightgray"
        )
    if (~ok).any():
        ax.scatter(
            xyz[~ok, 0], xyz[~ok, 1], xyz[~ok, 2],
            s=3, color="red"
        )

    ax.set_title("Errors (red)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=20, azim=45)

    fig.suptitle(title)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_d21_focus(
    xyz,
    gt,
    pred,
    d21_idx: int,
    save_path: Path,
    title: str = "",
):
    """
    Verde = TP del d21
    Rojo = FP/FN del d21
    Gris = resto
    """
    xyz = to_np(xyz)
    gt = to_np(gt)
    pred = to_np(pred)

    gt_d21 = (gt == int(d21_idx))
    pr_d21 = (pred == int(d21_idx))

    tp = gt_d21 & pr_d21
    fp = (~gt_d21) & pr_d21
    fn = gt_d21 & (~pr_d21)
    err = fp | fn

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")

    base = ~(tp | err)
    if base.any():
        ax.scatter(
            xyz[base, 0], xyz[base, 1], xyz[base, 2],
            s=1.5, color="lightgray", alpha=0.35
        )
    if tp.any():
        ax.scatter(
            xyz[tp, 0], xyz[tp, 1], xyz[tp, 2],
            s=3, color="green"
        )
    if err.any():
        ax.scatter(
            xyz[err, 0], xyz[err, 1], xyz[err, 2],
            s=3, color="red"
        )

    ax.set_title("D21 focus")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=20, azim=45)

    fig.suptitle(title)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


# ============================================================
# EVALUACIÓN SOBRE SPLIT
# ============================================================

@torch.no_grad()
def eval_split(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    C: int,
    bg: int,
    d21_idx: int,
    knn_chunk_size: int,
):
    model.eval()

    loss_fn = nn.CrossEntropyLoss()
    loss_sum = 0.0

    acc_all_s = acc_no_bg_s = 0.0
    prec_s = rec_s = f1_s = iou_s = 0.0
    pred_bg_frac_s = 0.0
    d21_acc_s = d21_f1_s = d21_iou_s = d21_bin_acc_all_s = 0.0
    nb = 0

    for xyz, y in loader:
        xyz = xyz.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(xyz, knn_chunk_size=int(knn_chunk_size))
        loss = loss_fn(logits.reshape(-1, int(C)), y.reshape(-1))

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
        rec_s += float(rec_m)
        f1_s += float(f1_m)
        iou_s += float(iou_m)
        pred_bg_frac_s += float(pred_bg_frac)
        d21_acc_s += float(d21_acc)
        d21_f1_s += float(d21_f1)
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
# INFERENCE COMPLETA (MISMO ESTILO POINTNET)
# ============================================================

@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    out_dir: Path,
    C: int,
    d21_idx: int,
    infer_examples: int,
    split: str,
):
    model.eval()

    out_root = Path(out_dir) / "inference"
    dir_all = out_root / "inference_all"
    dir_err = out_root / "inference_errors"
    dir_d21 = out_root / "inference_d21"

    for d in (dir_all, dir_err, dir_d21):
        d.mkdir(parents=True, exist_ok=True)

    manifest_path = out_root / "inference_manifest.csv"

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_idx",
            "png_all",
            "png_error",
            "png_d21"
        ])

        count = 0

        for i, (xyz, y) in enumerate(loader):
            if infer_examples > 0 and count >= infer_examples:
                break

            xyz = xyz.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(xyz)
            pred = logits.argmax(dim=-1)

            xyz_np = xyz[0].cpu()
            y_np = y[0].cpu()
            pred_np = pred[0].cpu()

            base = f"{split}_{i:04d}"

            p_all = dir_all / f"{base}.png"
            p_err = dir_err / f"{base}.png"
            p_d21 = dir_d21 / f"{base}.png"

            # --- ALL ---
            plot_3d_seg(
                xyz=xyz_np,
                gt=y_np,
                pred=pred_np,
                C=C,
                save_path=p_all,
                title=f"{split} {i} ALL"
            )

            # --- ERROR ---
            plot_error_map(
                xyz=xyz_np,
                gt=y_np,
                pred=pred_np,
                save_path=p_err,
                title=f"{split} {i} ERR"
            )

            # --- D21 ---
            plot_d21_focus(
                xyz=xyz_np,
                gt=y_np,
                pred=pred_np,
                d21_idx=d21_idx,
                save_path=p_d21,
                title=f"{split} {i} D21"
            )

            writer.writerow([
                i,
                str(p_all.relative_to(out_dir)),
                str(p_err.relative_to(out_dir)),
                str(p_d21.relative_to(out_dir)),
            ])

            count += 1

    print(f"[DONE] Generated {count} samples in {out_root}")


# ============================================================
# MAIN (FIX DEFINITIVO)
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)

    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument("--d21_internal", type=int, default=8)
    parser.add_argument("--infer_examples", type=int, default=12)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = Path(args.data_dir)
    model_path = Path(args.model_path)

    print(f"[setup] device={device}")
    print(f"[setup] model_path={model_path}")

    # ========================================================
    # DATA
    # ========================================================

    ds = NPZDataset(
        data_dir / f"X_{args.split}.npz",
        data_dir / f"Y_{args.split}.npz",
        normalize=True
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # ========================================================
    # MODEL
    # ========================================================

    # ⚠️ IMPORTANTE: inferir clases correctamente
    Y_sample = np.load(data_dir / f"Y_{args.split}.npz")["Y"]
    C = int(Y_sample.max()) + 1

    print(f"[setup] num_classes={C}")

    model = DGCNNSeg(num_classes=C).to(device)

    # ========================================================
    # 🔥 FIX CLAVE (tu error real)
    # ========================================================

    ckpt = load_checkpoint_flexible(model, model_path, device)

    print("[OK] Modelo cargado correctamente")

    # ========================================================
    # OUTPUT DIR
    # ========================================================

    out_dir = model_path.parent

    # ========================================================
    # EVAL
    # ========================================================

    metrics = eval_split(
        model=model,
        loader=loader,
        device=device,
        C=C,
        bg=0,
        d21_idx=args.d21_internal,
        knn_chunk_size=1024,
    )

    print("\n[TEST METRICS]")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ========================================================
    # INFERENCE
    # ========================================================

    run_inference(
        model=model,
        loader=loader,
        device=device,
        out_dir=out_dir,
        C=C,
        d21_idx=args.d21_internal,
        infer_examples=args.infer_examples,
        split=args.split,
    )


if __name__ == "__main__":
    main()