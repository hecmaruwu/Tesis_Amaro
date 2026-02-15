#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_dgcnn_classic_only_fixed_v4.py

DGCNN (EdgeConv) – Segmentación multiclase dental 3D
✅ MISMO SKELETON/OUTPUTS que pointnet_classic_final_v4.py
✅ BG incluido en loss (NO ignore)
✅ BG excluido SOLO en métricas macro (f1/iou/prec/rec implícitos vía TP/FP/FN)
✅ Métricas diente 21 explícitas BINARIO correcto (acc/f1/iou) + d21_bin_acc_all
✅ Estabilidad: bg downweight, weight_decay, grad clipping, CosineAnnealingLR
✅ RTX 3090 friendly: AMP, pin_memory, persistent_workers, non_blocking, cudnn.benchmark
✅ Inferencia: PNGs 3D (GT vs Pred) + errores + foco d21
✅ TRAZABILIDAD (v4): _discover_index_csv / _read_index_csv / _sanitize_tag
   + inference_manifest.csv (row_i -> paciente)

Uso:
cd /home/htaucare/Tesis_Amaro/scripts_last_version

python3 train_dgcnn_classic_only_fixed_v4.py \
  --data_dir .../upper_only_surf_global_excl_wisdom_seed42_aug2 \
  --out_dir  .../outputs/dgcnn/gpu1_run1_v4 \
  --epochs 120 --batch_size 16 --lr 2e-4 --weight_decay 1e-4 --dropout 0.5 \
  --num_workers 6 --device cuda --d21_internal 8 \
  --bg_weight 0.03 --grad_clip 1.0 --use_amp \
  --k 20 --emb_dims 1024 \
  --train_metrics_eval \
  --do_infer --infer_examples 12 --infer_split test \
  --index_csv .../index_test.csv
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
# DATASET (ROBUSTO: NO torch.from_numpy)
# ============================================================
class NPZDataset(Dataset):
    """
    Carga X_*.npz / Y_*.npz donde:
      X: [B,N,3] float32
      Y: [B,N]   int64

    FIX robusto: evita torch.from_numpy en workers (crash raro).
    """
    def __init__(self, Xp: Path, Yp: Path, normalize: bool = True):
        self.X = np.load(Xp)["X"]  # puede ser memmap/subclase/strides raros
        self.Y = np.load(Yp)["Y"]

        assert self.X.ndim == 3 and self.X.shape[-1] == 3, f"X shape inesperada: {self.X.shape}"
        assert self.Y.ndim == 2, f"Y shape inesperada: {self.Y.shape}"
        assert self.X.shape[0] == self.Y.shape[0], "B mismatch"
        assert self.X.shape[1] == self.Y.shape[1], "N mismatch"
        self.normalize = bool(normalize)

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, i: int):
        x = np.ascontiguousarray(np.asarray(self.X[int(i)]), dtype=np.float32)  # [N,3]
        y = np.ascontiguousarray(np.asarray(self.Y[int(i)]), dtype=np.int64)    # [N]
        xyz = torch.as_tensor(x, dtype=torch.float32)
        lab = torch.as_tensor(y, dtype=torch.int64)
        if self.normalize:
            xyz = normalize_unit_sphere(xyz)
        return xyz, lab


def make_loaders(data_dir: Path, bs: int, nw: int, normalize: bool = True):
    ds_tr = NPZDataset(data_dir / "X_train.npz", data_dir / "Y_train.npz", normalize=normalize)
    ds_va = NPZDataset(data_dir / "X_val.npz",   data_dir / "Y_val.npz",   normalize=normalize)
    ds_te = NPZDataset(data_dir / "X_test.npz",  data_dir / "Y_test.npz",  normalize=normalize)

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
# TRAZABILIDAD (v4): sanitize + read + discover index CSV
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
    Acepta encabezados flexibles (requiere row_i).
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

        k_idxg = _pick("idx_global", "idx", "global_idx", "global_id", "patient_global_idx")
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
# PARTE 2/4 — DGCNN (EdgeConv) para segmentación punto-a-punto
# ============================================================

def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    x: [B, C, N]
    return idx: [B, N, k]
    """
    # (x - y)^2 = x^2 + y^2 - 2xy
    xx = torch.sum(x ** 2, dim=1, keepdim=True)  # [B,1,N]
    pairwise = xx.transpose(2, 1) + xx - 2.0 * torch.matmul(x.transpose(2, 1), x)  # [B,N,N]
    _, idx = torch.topk(pairwise, k=k, dim=-1, largest=False, sorted=False)  # [B,N,k]
    return idx


def get_graph_feature(x: torch.Tensor, k: int, idx: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Edge feature:
      e_ij = concat( x_j - x_i , x_i )

    x: [B, C, N]
    return: [B, 2C, N, k]
    """
    B, C, N = x.shape
    if idx is None:
        idx = knn(x, k=k)  # [B,N,k]

    device = x.device
    idx_base = torch.arange(B, device=device).view(-1, 1, 1) * N  # [B,1,1]
    idx = idx + idx_base
    idx = idx.reshape(-1)  # [B*N*k]

    x_t = x.transpose(2, 1).contiguous()  # [B,N,C]
    neigh = x_t.reshape(B * N, C)[idx, :].view(B, N, k, C)  # [B,N,k,C]
    x_i = x_t.view(B, N, 1, C).repeat(1, 1, k, 1)           # [B,N,k,C]

    edge = torch.cat((neigh - x_i, x_i), dim=3)             # [B,N,k,2C]
    return edge.permute(0, 3, 1, 2).contiguous()            # [B,2C,N,k]


class EdgeConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, edge_feat: torch.Tensor) -> torch.Tensor:
        # edge_feat: [B,in_ch,N,k]
        x = self.net(edge_feat)          # [B,out_ch,N,k]
        x = torch.max(x, dim=-1)[0]      # [B,out_ch,N]
        return x


class DGCNNSeg(nn.Module):
    """
    DGCNN-style seg (equivalente estructuralmente al usado en tu pipeline):

      - 4 EdgeConv blocks
      - concat (64+64+128+256)=512
      - fuse -> 512
      - emb -> emb_dims (global)
      - concat local+global -> head -> logits [B,N,C]
    """

    def __init__(
        self,
        num_classes: int,
        k: int = 20,
        emb_dims: int = 1024,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.C = int(num_classes)
        self.k = int(k)
        self.emb_dims = int(emb_dims)

        # input coords C=3 -> edge 2C=6
        self.ec1 = EdgeConvBlock(6, 64)
        self.ec2 = EdgeConvBlock(2 * 64, 64)
        self.ec3 = EdgeConvBlock(2 * 64, 128)
        self.ec4 = EdgeConvBlock(2 * 128, 256)

        self.fuse = nn.Sequential(
            nn.Conv1d(64 + 64 + 128 + 256, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.emb = nn.Sequential(
            nn.Conv1d(512, self.emb_dims, 1, bias=False),
            nn.BatchNorm1d(self.emb_dims),
            nn.LeakyReLU(0.2, inplace=True),
        )

        head_in = 512 + self.emb_dims
        self.head = nn.Sequential(
            nn.Conv1d(head_in, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(float(dropout)),

            nn.Conv1d(256, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(float(dropout)),

            nn.Conv1d(256, self.C, 1, bias=True),
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        xyz: [B,N,3]
        return logits: [B,N,C]
        """
        B, N, _ = xyz.shape
        x = xyz.transpose(2, 1).contiguous()  # [B,3,N]

        # EdgeConv 1
        e1 = get_graph_feature(x, k=self.k)   # [B,6,N,k]
        x1 = self.ec1(e1)                     # [B,64,N]

        # EdgeConv 2
        e2 = get_graph_feature(x1, k=self.k)  # [B,128,N,k]
        x2 = self.ec2(e2)                     # [B,64,N]

        # EdgeConv 3
        e3 = get_graph_feature(x2, k=self.k)  # [B,128,N,k]
        x3 = self.ec3(e3)                     # [B,128,N]

        # EdgeConv 4
        e4 = get_graph_feature(x3, k=self.k)  # [B,256,N,k]
        x4 = self.ec4(e4)                     # [B,256,N]

        # Local concat
        x_cat = torch.cat((x1, x2, x3, x4), dim=1)  # [B,512,N]
        x_local = self.fuse(x_cat)                  # [B,512,N]

        # Global embedding
        x_emb = self.emb(x_local)                   # [B,emb_dims,N]
        x_global = torch.max(x_emb, dim=2, keepdim=True)[0]  # [B,emb_dims,1]
        x_global = x_global.repeat(1, 1, N)         # [B,emb_dims,N]

        # Final head
        x_final = torch.cat((x_local, x_global), dim=1)  # [B,512+emb_dims,N]
        logits = self.head(x_final)                      # [B,C,N]

        return logits.transpose(2, 1).contiguous()       # [B,N,C]

# ============================================================
# PARTE 3/4 — MÉTRICAS + VISUALIZACIÓN (idéntica a PointNet v4)
#           + run_epoch (v4-style: update en train(), métricas en eval())
# ============================================================

# --------- métricas (sin sklearn; mismas definiciones que tu PointNet v4) ---------

@torch.no_grad()
def macro_metrics_no_bg(pred: torch.Tensor, gt: torch.Tensor, C: int, bg: int = 0) -> Tuple[float, float]:
    """
    Macro-F1 e IoU macro calculados EXCLUYENDO BG (gt!=bg),
    promediando sobre clases 1..C-1, omitiendo clases ausentes (denom=0).
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

        denom = (tp + fp + fn)
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
    include_bg=False => excluye puntos bg (métrica principal)
    include_bg=True  => incluye bg (referencia)
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


def _mean_std(xs: List[float]) -> Tuple[float, float]:
    if xs is None or len(xs) == 0:
        return 0.0, 0.0
    a = np.asarray(xs, dtype=np.float64)
    return float(a.mean()), float(a.std(ddof=0))


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


# --------- visualización (robusta: igual fixes de PointNet v4) ---------

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
    s: float = 1.0
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
    c[tp, :] = (0.10, 0.75, 0.25, 1.0)
    c[err, :] = (0.85, 0.10, 0.10, 1.0)

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


# --------- run_epoch (idéntico a PointNet v4) ---------

def _get_autocast_ctx(device: torch.device, use_amp: bool):
    if use_amp and device.type == "cuda":
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
    grad_clip: Optional[float] = None,
    collect_batch_stats: bool = False,
) -> Dict[str, float]:
    """
    v4-style:
      - train=True: hace update con model.train() (dropout ON),
        pero calcula métricas con un segundo forward en model.eval() (dropout OFF)
      - train=False: eval normal (model.eval())

    collect_batch_stats=True => agrega *_std (útil para test)
    """
    scaler = run_epoch.scaler  # type: ignore
    if use_amp and train and (device.type == "cuda") and (scaler is None):
        scaler = torch.amp.GradScaler("cuda")
        run_epoch.scaler = scaler  # type: ignore

    loss_sum = 0.0
    sums = {
        "acc_all": 0.0, "acc_no_bg": 0.0, "f1_macro": 0.0, "iou_macro": 0.0,
        "d21_acc": 0.0, "d21_f1": 0.0, "d21_iou": 0.0, "d21_bin_acc_all": 0.0,
        "pred_bg_frac": 0.0,
    }
    n_batches = 0

    batch_stats: Dict[str, List[float]] = {k: [] for k in (["loss"] + list(sums.keys()))}

    if not bool(train):
        model.eval()

    for xyz, y in loader:
        xyz = xyz.to(device, non_blocking=True)  # [B,N,3]
        y = y.to(device, non_blocking=True)      # [B,N]

        if train:
            assert optimizer is not None
            model.train(True)
            optimizer.zero_grad(set_to_none=True)

            ctx = _get_autocast_ctx(device, bool(use_amp))
            with ctx:
                logits_train = model(xyz)                 # [B,N,C]
                loss = loss_fn(logits_train.reshape(-1, C), y.reshape(-1))

            if use_amp and device.type == "cuda":
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

            # métricas: segundo forward en eval() (dropout OFF)
            model.eval()
            with torch.no_grad():
                logits_eval = model(xyz)
            metrics = _compute_metrics_from_logits(logits_eval, y, C=C, d21_idx=d21_idx, bg=bg)

        else:
            ctx = _get_autocast_ctx(device, bool(use_amp) and (device.type == "cuda"))
            with torch.no_grad():
                with ctx:
                    logits = model(xyz)
                    loss = loss_fn(logits.reshape(-1, C), y.reshape(-1))
                metrics = _compute_metrics_from_logits(logits, y, C=C, d21_idx=d21_idx, bg=bg)

        loss_sum += float(loss.item())
        for k in sums.keys():
            sums[k] += float(metrics[k])
        n_batches += 1

        if collect_batch_stats:
            batch_stats["loss"].append(float(loss.item()))
            for k in sums.keys():
                batch_stats[k].append(float(metrics[k]))

    n = max(1, n_batches)
    out = {"loss": loss_sum / n}
    out.update({k: v / n for k, v in sums.items()})

    if collect_batch_stats:
        for k, v in batch_stats.items():
            _, std = _mean_std(v)
            out[f"{k}_std"] = float(std)

    return out


run_epoch.scaler = None  # type: ignore

# ============================================================
# PARTE 4/4 — MAIN (mismos outputs que PointNet v4)
#   - run_meta.json
#   - metrics_epoch.csv (misma grilla + "sec")
#   - history.json (mismas keys train_* / val_*)
#   - best.pt / last.pt
#   - test_metrics.json (incluye std por batch + tiempos)
#   - plots/* (Train vs Val, SIN línea test)
#   - infer (all/errors/d21) + inference_manifest.csv
#   - trazabilidad: index_{split}.csv + --index_csv (forzar)
# ============================================================

def _fmt_hms(seconds: float) -> str:
    seconds = float(seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:d}h {m:02d}m {s:05.2f}s"


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

    # DGCNN
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--emb_dims", type=int, default=1024)

    # task
    ap.add_argument("--d21_internal", type=int, required=True)
    ap.add_argument("--bg_index", type=int, default=0)
    ap.add_argument("--bg_weight", type=float, default=0.03)

    # stability
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--use_amp", action="store_true")
    ap.add_argument("--no_normalize", action="store_true")

    # (NEW) igual a PointNet v4
    ap.add_argument("--train_metrics_eval", action="store_true",
                    help="Calcula métricas de TRAIN con model.eval() (dropout OFF), "
                         "pero mantiene el forward train para backprop.")

    # trazabilidad
    ap.add_argument("--index_csv", type=str, default=None)

    # infer
    ap.add_argument("--do_infer", action="store_true")
    ap.add_argument("--infer_examples", type=int, default=12)
    ap.add_argument("--infer_split", type=str, default="test", choices=["test", "val", "train"])

    args = ap.parse_args()
    set_seed(args.seed)

    # device
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    print(f"[SANITY] num_classes C = {C}")
    print(f"[SANITY] bg_frac train/val/test = {bg_tr:.4f} {bg_va:.4f} {bg_te:.4f}")
    print(f"[SANITY] baseline acc_all (always-bg) train/val/test = {bg_tr:.4f} {bg_va:.4f} {bg_te:.4f}")

    if not (0 <= int(args.d21_internal) < C):
        raise ValueError(f"d21_internal fuera de rango: {args.d21_internal} (C={C})")

    d21_int = int(args.d21_internal)
    print(f"[INFO] d21_internal={d21_int}")

    # =========================================================
    # LOADERS
    # =========================================================
    dl_tr, dl_va, dl_te, ds_te = make_loaders(
        data_dir=data_dir,
        bs=args.batch_size,
        nw=args.num_workers,
        normalize=(not args.no_normalize),
    )

    # =========================================================
    # MODEL / OPT / LOSS
    # =========================================================
    model = DGCNNSeg(
        num_classes=C,
        k=int(args.k),
        emb_dims=int(args.emb_dims),
        dropout=float(args.dropout),
    ).to(device)

    w = torch.ones(C, device=device, dtype=torch.float32)
    w[bg] = float(args.bg_weight)
    loss_fn = nn.CrossEntropyLoss(weight=w)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=int(args.epochs), eta_min=1e-6
    )

    # =========================================================
    # META
    # =========================================================
    run_meta = {
        "script_name": "train_dgcnn_classic_only_fixed_v2.py",
        "model": "DGCNN-EdgeConv",
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
        "k": int(args.k),
        "emb_dims": int(args.emb_dims),
        "do_infer": bool(args.do_infer),
        "infer_examples": int(args.infer_examples),
        "infer_split": str(args.infer_split),
        "index_csv": str(args.index_csv) if args.index_csv else "",
    }
    save_json(run_meta, out_dir / "run_meta.json")

    # =========================================================
    # CSV por epoch (misma estructura que PointNet v4)
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
        ])

    # =========================================================
    # HISTORY (mismas keys que PointNet v4)
    # =========================================================
    history: Dict[str, List[float]] = {}
    def _mk(k: str):
        history[k] = []

    for k in (
        "loss", "acc_all", "acc_no_bg",
        "f1_macro", "iou_macro",
        "d21_acc", "d21_f1", "d21_iou",
        "d21_bin_acc_all",
        "pred_bg_frac",
    ):
        _mk(f"train_{k}")
        _mk(f"val_{k}")

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

        # 1) entrenamiento (update) (dropout ON), métricas internas ya son eval-forward en run_epoch(train=True)
        tr_backprop = run_epoch(
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
        )

        # 2) (opcional) métricas train comparables calculadas 100% en eval()
        if bool(args.train_metrics_eval):
            tr = run_epoch(
                model=model,
                loader=dl_tr,
                optimizer=None,
                loss_fn=loss_fn,
                C=C,
                d21_idx=d21_int,
                device=device,
                bg=bg,
                train=False,
                use_amp=False,
                grad_clip=None,
            )
        else:
            tr = tr_backprop

        # 3) validación
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
        )

        sched.step()
        lr_now = float(opt.param_groups[0]["lr"])
        sec = float(time.time() - e0)

        # history
        for k in (
            "loss", "acc_all", "acc_no_bg",
            "f1_macro", "iou_macro",
            "d21_acc", "d21_f1", "d21_iou",
            "d21_bin_acc_all",
            "pred_bg_frac",
        ):
            history[f"train_{k}"].append(float(tr[k]))
            history[f"val_{k}"].append(float(va[k]))

        # CSV
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            wcsv = csv.writer(f)
            wcsv.writerow([
                epoch, "train",
                tr["loss"], tr["acc_all"], tr["acc_no_bg"],
                tr["f1_macro"], tr["iou_macro"],
                tr["d21_acc"], tr["d21_f1"], tr["d21_iou"],
                tr["d21_bin_acc_all"],
                tr["pred_bg_frac"],
                lr_now,
                sec,
            ])
            wcsv.writerow([
                epoch, "val",
                va["loss"], va["acc_all"], va["acc_no_bg"],
                va["f1_macro"], va["iou_macro"],
                va["d21_acc"], va["d21_f1"], va["d21_iou"],
                va["d21_bin_acc_all"],
                va["pred_bg_frac"],
                lr_now,
                sec,
            ])

        # checkpoints
        torch.save({"model": model.state_dict(), "epoch": epoch}, last_path)

        if float(va["f1_macro"]) > best_val_f1:
            best_val_f1 = float(va["f1_macro"])
            best_epoch = int(epoch)
            torch.save({"model": model.state_dict(), "epoch": epoch}, best_path)

        # warning colapso a bg
        if float(va["pred_bg_frac"]) > max(0.95, bg_va + 0.12):
            print(
                f"[WARN] posible colapso a BG: "
                f"val pred_bg_frac={va['pred_bg_frac']:.3f} (bg_gt≈{bg_va:.3f})"
            )

        print(
            f"[{epoch:03d}/{int(args.epochs)}] "
            f"train loss={tr['loss']:.4f} f1m={tr['f1_macro']:.3f} ioum={tr['iou_macro']:.3f} "
            f"acc_all={tr['acc_all']:.3f} acc_no_bg={tr['acc_no_bg']:.3f} | "
            f"val loss={va['loss']:.4f} f1m={va['f1_macro']:.3f} ioum={va['iou_macro']:.3f} "
            f"acc_all={va['acc_all']:.3f} acc_no_bg={va['acc_no_bg']:.3f} | "
            f"d21(cls) acc={va['d21_acc']:.3f} f1={va['d21_f1']:.3f} iou={va['d21_iou']:.3f} | "
            f"d21(bin all) acc={va['d21_bin_acc_all']:.3f} | "
            f"pred_bg_frac(val)={va['pred_bg_frac']:.3f} lr={lr_now:.2e} sec={sec:.1f}"
        )

    save_json(history, out_dir / "history.json")

    # =========================================================
    # TEST (best) + PLOTS + INFER (con trazabilidad)
    # =========================================================
    ckpt = torch.load(best_path, map_location=device)
    best_epoch = int(ckpt.get("epoch", best_epoch if best_epoch > 0 else -1))
    model.load_state_dict(ckpt["model"])
    model.eval()

    te = run_epoch(
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
        collect_batch_stats=True,
    )

    total_sec = float(time.time() - t0)
    test_json = {
        "best_epoch": int(best_epoch),
        "total_sec": total_sec,
        "total_time_hms": _fmt_hms(total_sec),
        "num_classes": int(C),
        "bg_index": int(bg),
        "d21_internal": int(d21_int),
        "test": te,
        "best_val_f1_macro(no_bg)": float(best_val_f1),
    }
    save_json(test_json, out_dir / "test_metrics.json")

    # ---- plots (Train vs Val) ----
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_train_val("loss", history["train_loss"], history["val_loss"],
                   plots_dir / "loss_train_val.png", best_epoch=best_epoch)
    plot_train_val("f1_macro_no_bg", history["train_f1_macro"], history["val_f1_macro"],
                   plots_dir / "f1_macro_train_val.png", best_epoch=best_epoch)
    plot_train_val("iou_macro_no_bg", history["train_iou_macro"], history["val_iou_macro"],
                   plots_dir / "iou_macro_train_val.png", best_epoch=best_epoch)
    plot_train_val("acc_all", history["train_acc_all"], history["val_acc_all"],
                   plots_dir / "acc_all_train_val.png", best_epoch=best_epoch)
    plot_train_val("acc_no_bg", history["train_acc_no_bg"], history["val_acc_no_bg"],
                   plots_dir / "acc_no_bg_train_val.png", best_epoch=best_epoch)
    plot_train_val("pred_bg_frac", history["train_pred_bg_frac"], history["val_pred_bg_frac"],
                   plots_dir / "pred_bg_frac_train_val.png", best_epoch=best_epoch)

    plot_train_val("d21_f1", history["train_d21_f1"], history["val_d21_f1"],
                   plots_dir / "d21_f1_train_val.png", best_epoch=best_epoch)
    plot_train_val("d21_acc", history["train_d21_acc"], history["val_d21_acc"],
                   plots_dir / "d21_acc_train_val.png", best_epoch=best_epoch)
    plot_train_val("d21_iou", history["train_d21_iou"], history["val_d21_iou"],
                   plots_dir / "d21_iou_train_val.png", best_epoch=best_epoch)
    plot_train_val("d21_bin_acc_all", history["train_d21_bin_acc_all"], history["val_d21_bin_acc_all"],
                   plots_dir / "d21_bin_acc_all_train_val.png", best_epoch=best_epoch)

    # ---- infer (con index discovery + --index_csv forzado) ----
    if args.do_infer and int(args.infer_examples) > 0:
        model.eval()

        split = str(args.infer_split).strip().lower()
        if split == "test":
            ds_inf = ds_te
        elif split == "val":
            ds_inf = NPZDataset(
                data_dir / "X_val.npz",
                data_dir / "Y_val.npz",
                normalize=(not args.no_normalize),
            )
        elif split == "train":
            ds_inf = NPZDataset(
                data_dir / "X_train.npz",
                data_dir / "Y_train.npz",
                normalize=(not args.no_normalize),
            )
        else:
            raise ValueError(f"infer_split inválido: {split}")

        # trazabilidad
        index_path = None
        if args.index_csv:
            p = Path(args.index_csv)
            if p.exists():
                index_path = p
            else:
                print(f"[TRACE][WARN] --index_csv provisto pero no existe: {p}")

        if index_path is None:
            index_path = _discover_index_csv(data_dir, split)

        index_map = _read_index_csv(index_path) if index_path is not None else None

        if index_path is not None and index_map is not None:
            print(f"[TRACE] usando index CSV: {index_path}")
        else:
            print(f"[TRACE] sin index CSV (no encontrado/ilegible para split={split}) -> tag 'unknown'")

        # sampleo determinista
        k = min(int(args.infer_examples), len(ds_inf))
        rng = np.random.default_rng(int(args.seed) + 12345)
        idxs = rng.choice(len(ds_inf), size=k, replace=False)

        out_all = out_dir / "inference_all"
        out_err = out_dir / "inference_errors"
        out_d21 = out_dir / "inference_d21"
        out_all.mkdir(parents=True, exist_ok=True)
        out_err.mkdir(parents=True, exist_ok=True)
        out_d21.mkdir(parents=True, exist_ok=True)

        manifest_path = out_dir / "inference_manifest.csv"
        with open(manifest_path, "w", newline="", encoding="utf-8") as fman:
            wman = csv.writer(fman)
            wman.writerow([
                "split", "row_i",
                "idx_global", "sample_name", "jaw", "source_path", "has_labels",
                "png_all", "png_err", "png_d21",
            ])

            with torch.no_grad():
                for r, i in enumerate(idxs, start=1):
                    i = int(i)

                    xyz, y = ds_inf[i]
                    xyz_b = xyz.unsqueeze(0).to(device, non_blocking=True)

                    logits = model(xyz_b)[0]  # [N,C]
                    pred = logits.argmax(dim=-1)

                    xyz_np = xyz.detach().cpu().numpy().astype(np.float32, copy=False)
                    y_np = y.detach().cpu().numpy().astype(np.int32, copy=False)
                    pred_np = pred.detach().cpu().numpy().astype(np.int32, copy=False)

                    meta = {"idx_global": "", "sample_name": "", "jaw": "", "path": "", "has_labels": ""}
                    if index_map is not None and i in index_map:
                        meta = index_map[i]

                    sample = meta.get("sample_name", "") or ""
                    jaw = meta.get("jaw", "") or ""
                    idx_global = meta.get("idx_global", "") or ""
                    src_path = meta.get("path", "") or ""
                    has_labels = meta.get("has_labels", "") or ""

                    tag = _sanitize_tag(f"{sample}_{jaw}".strip("_")) or "unknown"

                    png_all = out_all / f"ex_{r:02d}_row_{i:05d}_{tag}.png"
                    png_err = out_err / f"ex_{r:02d}_row_{i:05d}_{tag}.png"
                    png_d21 = out_d21 / f"ex_{r:02d}_row_{i:05d}_{tag}.png"

                    title = (
                        f"{split} row={i} | sample={sample} | jaw={jaw} | idx_global={idx_global} | "
                        f"best_epoch={int(best_epoch)} | C={C} | d21={int(d21_int)} | k={int(args.k)}"
                    )

                    plot_pointcloud_all_classes(
                        xyz=xyz_np, y_gt=y_np, y_pr=pred_np,
                        out_png=png_all, C=C, title=title, s=1.0
                    )
                    plot_errors(
                        xyz=xyz_np, y_gt=y_np, y_pr=pred_np,
                        out_png=png_err, bg=bg, title=title, s=1.0
                    )
                    plot_d21_focus(
                        xyz=xyz_np, y_gt=y_np, y_pr=pred_np,
                        out_png=png_d21, d21_idx=int(d21_int),
                        bg=bg, title=title, s=1.2
                    )

                    wman.writerow([
                        split, i,
                        idx_global, sample, jaw, src_path, has_labels,
                        str(png_all), str(png_err), str(png_d21),
                    ])

        print(f"[INFER] manifest guardado en: {manifest_path}")
        print(f"[INFER] outputs: {out_all} | {out_err} | {out_d21}")

    print(
        f"[DONE] out_dir={out_dir} | total_sec={total_sec:.1f} ({_fmt_hms(total_sec)}) | "
        f"best_epoch={best_epoch} | best_val_f1_macro(no_bg)={best_val_f1:.4f} | "
        f"d21_f1_test={float(te.get('d21_f1', 0.0)):.4f}"
    )


if __name__ == "__main__":
    main()
