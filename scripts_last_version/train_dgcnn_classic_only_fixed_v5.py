#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_dgcnn_classic_only_fixed_v5.py

DGCNN (EdgeConv) – Segmentación multiclase dental 3D
✅ MISMO SKELETON/OUTPUTS que pointnet_classic_final_v4.py
✅ BG incluido en loss (NO ignore)
✅ BG excluido SOLO en métricas macro (f1/iou)
✅ Métricas diente 21 explícitas BINARIO correcto (acc/f1/iou) + d21_bin_acc_all
✅ Estabilidad: bg downweight, weight_decay, grad clipping, CosineAnnealingLR
✅ RTX 3090 friendly: AMP (torch.amp), pin_memory, persistent_workers, non_blocking, cudnn.benchmark
✅ Inferencia: PNGs 3D (GT vs Pred) + errores + foco d21
✅ TRAZABILIDAD (v4): _discover_index_csv / _read_index_csv / _sanitize_tag
   + inference_manifest.csv (row_i -> paciente)

NOTAS IMPORTANTES v5 (fixes):
- FIX KNN chunked: corrige mismatch (chunk vs N) en cálculo de distancias.
- FIX AMP deprecations: usa torch.amp.GradScaler("cuda") y torch.amp.autocast("cuda").

Dataset esperado:
  data_dir/X_train.npz, Y_train.npz, X_val.npz, Y_val.npz, X_test.npz, Y_test.npz
  X: [B,N,3], Y: [B,N] con clases internas 0..C-1 (0=bg)
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
    """
    xyz: [N,3]
    """
    c = xyz.mean(dim=0, keepdim=True)
    x = xyz - c
    r = torch.norm(x, dim=1).max().clamp_min(eps)
    return x / r


# ============================================================
# DATASET (ROBUSTO: evita torch.from_numpy en workers)
# ============================================================
class NPZDataset(Dataset):
    """
    Carga X_*.npz / Y_*.npz donde:
      X: [B,N,3] float32
      Y: [B,N]   int64

    FIX robusto: np.ascontiguousarray + torch.as_tensor
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

        k_idxg = _pick("idx_global", "global_idx", "global_id", "patient_global_idx", "idx")
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
#   - KNN normal o CHUNKED (FIX: dist shape [B,chunk,N])
#   - get_graph_feature (usa idx)
#   - DGCNNSeg (4 EdgeConv + fuse + emb + head)
# ============================================================

def knn_full(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    KNN completo (sin chunk):
      x: [B, C, N]
      return idx: [B, N, k]
    """
    # dist(i,j) = ||xi - xj||^2 = xi^2 + xj^2 - 2 xi·xj
    xx = torch.sum(x ** 2, dim=1, keepdim=True)                 # [B,1,N]
    dist = xx.transpose(2, 1) + xx - 2.0 * torch.matmul(x.transpose(2, 1), x)  # [B,N,N]
    _, idx = torch.topk(dist, k=int(k), dim=-1, largest=False, sorted=False)   # [B,N,k]
    return idx


def knn_chunked(x: torch.Tensor, k: int, chunk_size: int) -> torch.Tensor:
    """
    KNN chunked (reduce RAM):
      x: [B, C, N]
      Procesa queries por chunks de tamaño chunk_size.

    FIX CLAVE:
      dist debe quedar [B, chunk, N], NO [B,chunk,chunk] ni broadcast malo.
    """
    B, C, N = x.shape
    k = int(k)
    chunk_size = int(chunk_size)
    chunk_size = max(1, min(chunk_size, N))

    x_t = x.transpose(2, 1).contiguous()                        # [B,N,C]
    x2 = torch.sum(x_t ** 2, dim=2, keepdim=True)               # [B,N,1]
    x_t_T = x_t.transpose(2, 1).contiguous()                    # [B,C,N]

    idx_out = torch.empty((B, N, k), device=x.device, dtype=torch.long)

    # Loop por chunks de queries (dimension "query points")
    for start in range(0, N, chunk_size):
        end = min(N, start + chunk_size)
        q = x_t[:, start:end, :]                                # [B,chunk,C]
        q2 = torch.sum(q ** 2, dim=2, keepdim=True)             # [B,chunk,1]

        # qx = q @ x^T  => [B,chunk,N]
        qx = torch.matmul(q, x_t_T)                             # [B,chunk,N]

        # dist = ||q||^2 + ||x||^2 - 2 q·x
        # q2: [B,chunk,1]
        # x2^T: [B,1,N]
        dist = q2 + x2.transpose(2, 1) - 2.0 * qx               # [B,chunk,N]

        _, idx = torch.topk(dist, k=k, dim=-1, largest=False, sorted=False)  # [B,chunk,k]
        idx_out[:, start:end, :] = idx

    return idx_out


def knn(x: torch.Tensor, k: int, chunk_size: int = 0) -> torch.Tensor:
    """
    Wrapper: usa chunked si chunk_size>0, si no usa full.
    """
    if chunk_size is None or int(chunk_size) <= 0:
        return knn_full(x, k=int(k))
    return knn_chunked(x, k=int(k), chunk_size=int(chunk_size))


def get_graph_feature(
    x: torch.Tensor,
    k: int,
    idx: Optional[torch.Tensor] = None,
    knn_chunk_size: int = 0,
) -> torch.Tensor:
    """
    Edge feature:
      e_ij = concat( x_j - x_i , x_i )

    x: [B, C, N]
    return: [B, 2C, N, k]
    """
    B, C, N = x.shape
    k = int(k)

    if idx is None:
        idx = knn(x, k=k, chunk_size=int(knn_chunk_size))  # [B,N,k]

    device = x.device
    idx_base = torch.arange(B, device=device).view(-1, 1, 1) * N  # [B,1,1]
    idx = idx + idx_base
    idx = idx.reshape(-1)  # [B*N*k]

    x_t = x.transpose(2, 1).contiguous()  # [B,N,C]
    neigh = x_t.reshape(B * N, C)[idx, :].view(B, N, k, C)        # [B,N,k,C]
    x_i = x_t.view(B, N, 1, C).expand(-1, -1, k, -1)              # [B,N,k,C]

    edge = torch.cat((neigh - x_i, x_i), dim=3)                   # [B,N,k,2C]
    return edge.permute(0, 3, 1, 2).contiguous()                  # [B,2C,N,k]


class EdgeConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(int(in_ch), int(out_ch), 1, bias=False),
            nn.BatchNorm2d(int(out_ch)),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, edge_feat: torch.Tensor) -> torch.Tensor:
        # edge_feat: [B,in_ch,N,k]
        x = self.net(edge_feat)      # [B,out_ch,N,k]
        x = torch.max(x, dim=-1)[0]  # [B,out_ch,N]
        return x


class DGCNNSeg(nn.Module):
    """
    DGCNN-style seg:

      - 4 EdgeConv blocks
      - concat (64+64+128+256)=512
      - fuse -> 512
      - emb -> emb_dims (global)
      - concat local+global -> head -> logits [B,N,C]

    Incluye knn_chunk_size para reducir memoria si es necesario.
    """

    def __init__(
        self,
        num_classes: int,
        k: int = 20,
        emb_dims: int = 1024,
        dropout: float = 0.5,
        knn_chunk_size: int = 0,
    ):
        super().__init__()
        self.C = int(num_classes)
        self.k = int(k)
        self.emb_dims = int(emb_dims)
        self.knn_chunk_size = int(knn_chunk_size)

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
        e1 = get_graph_feature(x, k=self.k, idx=None, knn_chunk_size=self.knn_chunk_size)  # [B,6,N,k]
        x1 = self.ec1(e1)                                                                  # [B,64,N]

        # EdgeConv 2
        e2 = get_graph_feature(x1, k=self.k, idx=None, knn_chunk_size=self.knn_chunk_size) # [B,128,N,k]
        x2 = self.ec2(e2)                                                                  # [B,64,N]

        # EdgeConv 3
        e3 = get_graph_feature(x2, k=self.k, idx=None, knn_chunk_size=self.knn_chunk_size) # [B,128,N,k]
        x3 = self.ec3(e3)                                                                  # [B,128,N]

        # EdgeConv 4
        e4 = get_graph_feature(x3, k=self.k, idx=None, knn_chunk_size=self.knn_chunk_size) # [B,256,N,k]
        x4 = self.ec4(e4)                                                                  # [B,256,N]

        # Local concat
        x_cat = torch.cat((x1, x2, x3, x4), dim=1)  # [B,512,N]
        x_local = self.fuse(x_cat)                  # [B,512,N]

        # Global embedding
        x_emb = self.emb(x_local)                                       # [B,emb_dims,N]
        x_global = torch.max(x_emb, dim=2, keepdim=True)[0]             # [B,emb_dims,1]
        x_global = x_global.expand(-1, -1, N).contiguous()              # [B,emb_dims,N]

        # Head
        x_final = torch.cat((x_local, x_global), dim=1)                 # [B,512+emb_dims,N]
        logits = self.head(x_final)                                     # [B,C,N]
        return logits.transpose(2, 1).contiguous()                      # [B,N,C]

# ============================================================
# PARTE 3/4 — MÉTRICAS + VISUALIZACIÓN (idéntico a PointNet v4)
#           + run_epoch v4-style (update train(), métricas con eval())
#           + AMP MODERNO (sin FutureWarning)
# ============================================================

# --------- métricas (sin sklearn; mismas definiciones que PointNet v4) ---------

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


# --------- visualización (robusta: mismos fixes de PointNet v4) ---------

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


# --------- run_epoch (v4-style) + AMP moderno (sin warnings) ---------

def _autocast_ctx(device: torch.device, use_amp: bool):
    if bool(use_amp) and device.type == "cuda":
        return torch.amp.autocast("cuda", enabled=True)
    return torch.amp.autocast("cpu", enabled=False)


def _get_scaler(device: torch.device, use_amp: bool) -> Optional[torch.amp.GradScaler]:
    if (device.type == "cuda") and bool(use_amp):
        return torch.amp.GradScaler("cuda")
    return None


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
      - train=True: update con model.train() (dropout ON),
        métricas con segundo forward en model.eval() (dropout OFF)
      - train=False: eval normal (model.eval())

    collect_batch_stats=True => agrega *_std (útil para test)
    """
    # scaler lazy (solo si se usa)
    scaler = run_epoch.scaler  # type: ignore
    if train and (device.type == "cuda") and bool(use_amp) and (scaler is None):
        scaler = _get_scaler(device, use_amp=True)
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

            ctx = _autocast_ctx(device, bool(use_amp))
            with ctx:
                logits_train = model(xyz)                 # [B,N,C]
                loss = loss_fn(logits_train.reshape(-1, C), y.reshape(-1))

            if (device.type == "cuda") and bool(use_amp):
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

            # métricas: segundo forward en eval (dropout OFF)
            model.eval()
            with torch.no_grad():
                logits_eval = model(xyz)
            metrics = _compute_metrics_from_logits(logits_eval, y, C=C, d21_idx=d21_idx, bg=bg)

        else:
            ctx = _autocast_ctx(device, bool(use_amp) and (device.type == "cuda"))
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
# PARTE 4/4 — MAIN + TRAIN LOOP + SCHEDULER + INFERENCIA
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=200)
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
    parser.add_argument("--knn_chunk_size", type=int, default=0)

    parser.add_argument("--bg_weight", type=float, default=0.03)
    parser.add_argument("--d21_internal", type=int, default=8)

    parser.add_argument("--do_infer", action="store_true")
    parser.add_argument("--infer_examples", type=int, default=12)
    parser.add_argument("--infer_split", type=str, default="test")

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # --------------------------------------------------------
    # NUM_CLASSES (robusto: desde Y_train)
    # --------------------------------------------------------
    y_train_np = np.load(data_dir / "Y_train.npz")["Y"]
    C = int(y_train_np.max()) + 1
    bg = 0
    d21_idx = int(args.d21_internal)

    # --------------------------------------------------------
    # LOADERS
    # --------------------------------------------------------
    train_loader, val_loader, test_loader, test_ds = make_loaders(
        data_dir=data_dir,
        bs=int(args.batch_size),
        nw=int(args.num_workers),
        normalize=True,
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------
    model = DGCNNSeg(
        num_classes=C,
        k=int(args.k),
        emb_dims=int(args.emb_dims),
        dropout=float(args.dropout),
        knn_chunk_size=int(args.knn_chunk_size),
    ).to(device)

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------
    weights = torch.ones(C, device=device)
    weights[int(bg)] = float(args.bg_weight)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    # --------------------------------------------------------
    # OPTIMIZER + SCHEDULER
    # --------------------------------------------------------
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(args.epochs),
        eta_min=1e-6
    )

    # --------------------------------------------------------
    # TRAIN LOOP
    # --------------------------------------------------------
    history = {
        "train_loss": [], "val_loss": [],
        "train_f1": [], "val_f1": [],
        "train_iou": [], "val_iou": [],
        "train_acc_all": [], "val_acc_all": [],
        "train_acc_no_bg": [], "val_acc_no_bg": [],
    }

    best_val_f1 = -1.0
    best_epoch = -1

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = run_epoch(
            model, train_loader, optimizer, loss_fn,
            C=C, d21_idx=d21_idx, device=device, bg=bg,
            train=True, use_amp=args.use_amp,
            grad_clip=args.grad_clip
        )

        val_metrics = run_epoch(
            model, val_loader, None, loss_fn,
            C=C, d21_idx=d21_idx, device=device, bg=bg,
            train=False, use_amp=args.use_amp
        )

        scheduler.step()

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_f1"].append(train_metrics["f1_macro"])
        history["val_f1"].append(val_metrics["f1_macro"])
        history["train_iou"].append(train_metrics["iou_macro"])
        history["val_iou"].append(val_metrics["iou_macro"])
        history["train_acc_all"].append(train_metrics["acc_all"])
        history["val_acc_all"].append(val_metrics["acc_all"])
        history["train_acc_no_bg"].append(train_metrics["acc_no_bg"])
        history["val_acc_no_bg"].append(val_metrics["acc_no_bg"])

        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_epoch = epoch
            torch.save(model.state_dict(), out_dir / "best.pt")

        torch.save(model.state_dict(), out_dir / "last.pt")

        dt = time.time() - t0
        print(
            f"[{epoch}/{args.epochs}] "
            f"train loss={train_metrics['loss']:.4f} "
            f"val loss={val_metrics['loss']:.4f} "
            f"val f1={val_metrics['f1_macro']:.4f} "
            f"({ _fmt_hms(dt) })"
        )

    total_time = time.time() - start_time
    print(f"\nEntrenamiento completo en {_fmt_hms(total_time)}")
    print(f"Best epoch: {best_epoch} | Best val_f1: {best_val_f1:.4f}")

    save_json(history, out_dir / "history.json")

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))

    test_metrics = run_epoch(
        model, test_loader, None, loss_fn,
        C=C, d21_idx=d21_idx, device=device, bg=bg,
        train=False, use_amp=args.use_amp,
        collect_batch_stats=True
    )

    save_json(test_metrics, out_dir / "test_metrics.json")

    # --------------------------------------------------------
    # PLOTS
    # --------------------------------------------------------
    plots_dir = out_dir / "plots"
    plot_train_val("Loss", history["train_loss"], history["val_loss"], plots_dir / "loss.png", best_epoch)
    plot_train_val("F1_macro", history["train_f1"], history["val_f1"], plots_dir / "f1.png", best_epoch)
    plot_train_val("IoU_macro", history["train_iou"], history["val_iou"], plots_dir / "iou.png", best_epoch)

    # --------------------------------------------------------
    # INFERENCIA (con trazabilidad)
    # --------------------------------------------------------
    if args.do_infer:
        infer_split = args.infer_split.lower()
        if infer_split == "train":
            loader = train_loader
        elif infer_split == "val":
            loader = val_loader
        else:
            loader = test_loader

        idx_csv = _discover_index_csv(data_dir, infer_split)
        idx_map = _read_index_csv(idx_csv) if idx_csv else None

        infer_dir = out_dir / "inference" / infer_split
        infer_dir.mkdir(parents=True, exist_ok=True)

        manifest_rows = []

        model.eval()
        count = 0

        for batch_i, (xyz, y) in enumerate(loader):
            xyz = xyz.to(device)
            y = y.to(device)

            with torch.no_grad():
                logits = model(xyz)
                pred = logits.argmax(dim=-1)

            B = xyz.shape[0]
            for b in range(B):
                if count >= int(args.infer_examples):
                    break

                xyz_np = xyz[b].cpu().numpy()
                y_np = y[b].cpu().numpy()
                p_np = pred[b].cpu().numpy()

                tag = f"row_{count}"
                if idx_map and count in idx_map:
                    name = _sanitize_tag(idx_map[count].get("sample_name", ""))
                    if name:
                        tag = f"{count}_{name}"

                plot_pointcloud_all_classes(
                    xyz_np, y_np, p_np,
                    infer_dir / f"{tag}_all.png",
                    C=C,
                    title=tag
                )

                plot_errors(
                    xyz_np, y_np, p_np,
                    infer_dir / f"{tag}_errors.png",
                    bg=bg,
                    title=tag
                )

                plot_d21_focus(
                    xyz_np, y_np, p_np,
                    infer_dir / f"{tag}_d21.png",
                    d21_idx=d21_idx,
                    bg=bg,
                    title=tag
                )

                manifest_rows.append({
                    "row_i": count,
                    "tag": tag
                })

                count += 1

            if count >= int(args.infer_examples):
                break

        with open(infer_dir / "inference_manifest.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["row_i", "tag"])
            writer.writeheader()
            writer.writerows(manifest_rows)


if __name__ == "__main__":
    main()
