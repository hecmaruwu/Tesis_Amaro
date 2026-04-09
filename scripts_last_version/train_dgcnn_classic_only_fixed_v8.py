#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_dgcnn_classic_only_fixed_v8.py

DGCNN (EdgeConv) – Segmentación multiclase dental 3D

VERSIÓN v8 (v7 + workaround robusto contra NumPy "mezclado" en INFER)

Contexto del bug:
- En algunos entornos, NumPy puede fallar al construir __repr__/errores internos
  (p.ej. "module 'numpy.core.multiarray' has no attribute 'character'").
- Eso suele detonarse cuando un ufunc (como y_gt != y_pred) falla y NumPy intenta
  formatear arrays para el error (arrayprint/numerictypes).

Arreglo v8 (SIN tocar el entorno):
✔ En inferencia/plots, EVITAMOS ufuncs de NumPy para máscaras/comparaciones.
✔ Hacemos comparaciones con Torch y recién convertimos a numpy() al final.
✔ Agregamos chequeos explícitos de mismatch de tamaños para errores claros.

Mantiene TODO lo de v7:
✔ Sin conflictos de device (loss weight en GPU)
✔ Sin redefiniciones (limpieza v6)
✔ KNN chunk-safe correcto
✔ Vecinos configurables (neighbor_teeth)
✔ AMP moderno torch.amp
✔ Skeleton idéntico a tu PointNet v4 (outputs/trazabilidad)
✔ Compatible con dataset flat:
    X_train.npz / Y_train.npz  (misma raíz)

v7 ya incluía:
✔ Plots de vecinos en out_dir/plots/neighbors/:
    {name}_acc.png, {name}_f1.png, {name}_iou.png, {name}_bin_acc_all.png
  + summary plot

Outputs esperados:
- out_dir/run_meta.json
- out_dir/run.log (si rediriges stdout) + errors.log (si rediriges stderr)
- out_dir/metrics_epoch.csv
- out_dir/history.json
- out_dir/history_epoch.jsonl
- out_dir/best.pt / out_dir/last.pt
- out_dir/test_metrics.json
- out_dir/plots/*.png
- out_dir/inference/{train|val|test}/*.png   (si --do_infer)
"""

import os
import json
import csv
import time
import argparse
import random
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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


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

# alias (compat PointNet)
def set_seed(seed: int = 42):
    seed_all(seed)


# ============================================================
# NORMALIZACIÓN
# ============================================================

def normalize_unit_sphere(xyz: torch.Tensor, eps: float = 1e-9):
    """
    xyz: [N,3]
    - centra por media
    - escala por max norm (sphere)
    """
    center = xyz.mean(dim=0, keepdim=True)
    xyz = xyz - center
    scale = torch.norm(xyz, dim=1).max().clamp_min(eps)
    return xyz / scale


# ============================================================
# DATASET ROBUSTO
# ============================================================

class NPZDataset(Dataset):
    def __init__(self, X_path: Path, Y_path: Path, normalize: bool = True):
        self.X = np.load(X_path)["X"]
        self.Y = np.load(Y_path)["Y"]
        self.normalize = bool(normalize)

        assert self.X.shape[0] == self.Y.shape[0]
        assert self.X.shape[1] == self.Y.shape[1]

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, idx):
        x = np.ascontiguousarray(self.X[idx], dtype=np.float32)  # [N,3]
        y = np.ascontiguousarray(self.Y[idx], dtype=np.int64)    # [N]

        xyz = torch.as_tensor(x, dtype=torch.float32)
        lab = torch.as_tensor(y, dtype=torch.int64)

        if self.normalize:
            xyz = normalize_unit_sphere(xyz)

        return xyz, lab


def make_loaders(data_dir: Path, bs: int, nw: int, normalize: bool = True):
    """
    Retorna EXACTAMENTE 3 loaders (train/val/test).
    Dataset flat: X_{split}.npz / Y_{split}.npz en la raíz data_dir.
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

    dl_tr = DataLoader(ds_tr, shuffle=True, **common)
    dl_va = DataLoader(ds_va, shuffle=False, **common)
    dl_te = DataLoader(ds_te, shuffle=False, **common)
    return dl_tr, dl_va, dl_te


# ============================================================
# LABEL MAP + INFER NUM CLASSES
# ============================================================

def load_label_map(data_dir: Path):
    p = data_dir / "label_map.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def infer_num_classes(data_dir: Path, label_map: Optional[dict]):
    # 1) label_map
    if isinstance(label_map, dict) and len(label_map) > 0:
        try:
            mx = max(int(v) for v in label_map.values())
            return int(mx + 1)
        except Exception:
            pass

    # 2) scan Y_*.npz
    maxy = -1
    for split in ("train", "val", "test"):
        yp = data_dir / f"Y_{split}.npz"
        if yp.exists():
            y = np.load(yp)["Y"]
            maxy = max(maxy, int(y.max()))

    if maxy < 0:
        raise RuntimeError("No se pudo inferir num_classes (no Y_*.npz y/o label_map inválido).")
    return int(maxy + 1)


# ============================================================
# NEIGHBORS PARSER
# ============================================================

def parse_neighbors(s: str) -> List[Tuple[str, int]]:
    """
    --neighbor_teeth "d11:1,d22:9"
    Retorna: [("d11",1), ("d22",9)]
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
            raise ValueError(f"Formato inválido en neighbor_teeth: '{part}' (usa nombre:idx)")
        name, idx = part.split(":", 1)
        out.append((name.strip(), int(idx)))
    return out


# ============================================================
# TRAZABILIDAD INDEX CSV (index_{split}.csv)
# ============================================================

def _discover_index_csv(data_dir: Path, split: str) -> Optional[Path]:
    """
    Busca index_{split}.csv en:
      1) data_dir/index_{split}.csv
      2) ascendiendo hasta Teeth_3ds (si existe)
      3) primer match en ancestros (fallback suave)
    """
    split = str(split)
    cand = data_dir / f"index_{split}.csv"
    if cand.exists():
        return cand

    p = data_dir.resolve()
    parents = [p] + list(p.parents)

    # 2) hasta Teeth_3ds
    for parent in parents:
        if parent.name == "Teeth_3ds":
            cand2 = parent / f"index_{split}.csv"
            if cand2.exists():
                return cand2
            for sub in parent.rglob(f"index_{split}.csv"):
                return sub
            break

    # 3) fallback: ancestros directos
    for parent in parents:
        cand3 = parent / f"index_{split}.csv"
        if cand3.exists():
            return cand3

    return None


def _read_index_csv(p: Optional[Path]) -> Optional[Dict[int, Dict[str, str]]]:
    """
    Lee index_{split}.csv y retorna:
      { row_idx_int : {col: value, ...}, ... }
    Si falla, retorna None.
    """
    if p is None or (not Path(p).exists()):
        return None

    out: Dict[int, Dict[str, str]] = {}
    try:
        with Path(p).open("r", encoding="utf-8", newline="") as f:
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
                out[int(rid)] = {kk: ("" if row[kk] is None else str(row[kk])) for kk in row.keys()}
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

# ============================================================
# PARTE 2/5 — DGCNN (KNN chunk-safe) + EdgeConv + Modelo
# ============================================================

def knn(x: torch.Tensor, k: int, chunk_size: int = 0) -> torch.Tensor:
    """
    x: [B, C, N]
    return idx: [B, N, k]

    Chunk-safe real:
      - queries por bloques M, pero distancia contra TODO N
      - nunca mezcla dims

    Distancia:
      ||q - x||^2 = ||q||^2 + ||x||^2 - 2 q^T x
    """
    assert x.dim() == 3, f"x debe ser [B,C,N], got {tuple(x.shape)}"
    B, C, N = x.shape
    k = int(k)
    if k <= 0:
        raise ValueError("k debe ser > 0")
    k = min(k, N)

    # ||x||^2 para todos los puntos (keys)
    xx = (x ** 2).sum(dim=1)  # [B, N]

    chunk_size = int(chunk_size or 0)
    if chunk_size <= 0 or chunk_size >= N:
        # full queries
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
        knn_chunk_size: int = 0,
    ):
        super().__init__()
        self.k = int(k)
        self.knn_chunk_size = int(knn_chunk_size)
        self.emb_dims = int(emb_dims)
        self.dropout = float(dropout)

        # EdgeConv stacks (paper-style)
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

        e1 = get_graph_feature(x,  k=self.k, knn_chunk_size=self.knn_chunk_size)  # [B,6,N,k]
        x1 = self.ec1(e1)  # [B,64,N]

        e2 = get_graph_feature(x1, k=self.k, knn_chunk_size=self.knn_chunk_size)  # [B,128,N,k]
        x2 = self.ec2(e2)  # [B,64,N]

        e3 = get_graph_feature(x2, k=self.k, knn_chunk_size=self.knn_chunk_size)  # [B,128,N,k]
        x3 = self.ec3(e3)  # [B,128,N]

        e4 = get_graph_feature(x3, k=self.k, knn_chunk_size=self.knn_chunk_size)  # [B,256,N,k]
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
#   ✅ Incluye PLOTS de vecinos (v7)
# ============================================================

# ----------------------------
# LOSS: BG incluido, downweight bg (SIN ignore_index)
# ----------------------------
def make_loss_fn(num_classes: int, bg_class: int, bg_weight: float, device: torch.device) -> nn.Module:
    """
    CrossEntropy con pesos por clase:
      - bg_class tiene peso bg_weight (ej: 0.03)
      - resto 1.0

    ✅ FIX device: weight en el MISMO device que logits.
    """
    C = int(num_classes)
    bg = int(bg_class)
    w = torch.ones(C, dtype=torch.float32, device=device)
    if 0 <= bg < C:
        w[bg] = float(bg_weight)
    return nn.CrossEntropyLoss(weight=w)


# ----------------------------
# MÉTRICAS (macro sin BG, d21 binario, etc.)
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
    Binario para cualquier diente:
      positivo = clase d21_idx
      negativo = resto

    include_bg=False => excluye puntos con gt==bg (métrica informativa)
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

    f1m, ioum = macro_metrics_no_bg(pred, y, C=C, bg=bg)

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
# AMP moderno (cuda)
# ----------------------------
def _get_autocast_ctx(device: torch.device, use_amp: bool):
    if bool(use_amp) and device.type == "cuda":
        return torch.amp.autocast("cuda", enabled=True)
    return torch.amp.autocast("cpu", enabled=False)


# ----------------------------
# VECINOS (lista arbitraria)
# ----------------------------
@torch.no_grad()
def eval_neighbors_on_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    neighbor_list: List[Tuple[str, int]],
    bg: int,
    use_amp: bool,
) -> Dict[str, float]:
    """
    Retorna promedio por batch:
      {name}_acc, {name}_f1, {name}_iou, {name}_bin_acc_all

    - acc/f1/iou => include_bg=False
    - bin_acc_all => include_bg=True
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
    with torch.no_grad():
        for xyz, y in loader:
            xyz = xyz.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with ctx:
                logits = model(xyz)
            pred = logits.argmax(dim=-1)

            for name, idx in neighbor_list:
                acc, f1, iou = d21_metrics_binary(pred, y, d21_idx=int(idx), bg=int(bg), include_bg=False)
                acc_all, _, _ = d21_metrics_binary(pred, y, d21_idx=int(idx), bg=int(bg), include_bg=True)

                sums[f"{name}_acc"] += acc
                sums[f"{name}_f1"] += f1
                sums[f"{name}_iou"] += iou
                sums[f"{name}_bin_acc_all"] += acc_all

            nb += 1

    nb = max(1, nb)
    return {k: v / nb for k, v in sums.items()}


# ----------------------------
# PLOTS (idéntico estilo PointNet v4) + VECINOS (v7)
# ----------------------------
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


def plot_neighbor_curves(
    history: Dict[str, List[float]],
    neighbors: List[Tuple[str, int]],
    out_dir: Path,
    best_epoch: Optional[int] = None,
):
    """
    Genera curvas por vecino (val) en:
      out_dir/plots/neighbors/{name}_acc.png
      out_dir/plots/neighbors/{name}_f1.png
      out_dir/plots/neighbors/{name}_iou.png
      out_dir/plots/neighbors/{name}_bin_acc_all.png
    """
    if not neighbors:
        return

    n_dir = Path(out_dir) / "plots" / "neighbors"
    n_dir.mkdir(parents=True, exist_ok=True)

    for name, _ in neighbors:
        for metric in ("acc", "f1", "iou", "bin_acc_all"):
            key = f"val_{name}_{metric}"
            y = history.get(key, [])
            if not isinstance(y, list) or len(y) == 0:
                continue

            out_png = n_dir / f"{name}_{metric}.png"
            plt.figure(figsize=(7, 4))
            plt.plot(y, label=key)
            if best_epoch is not None and int(best_epoch) > 0:
                plt.axvline(int(best_epoch) - 1, linestyle=":", label=f"best_epoch={int(best_epoch)}")
            plt.xlabel("epoch")
            plt.ylabel(key)
            plt.title(f"{key} (Val)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_png, dpi=250)
            plt.close()


def plot_neighbors_summary(
    history: Dict[str, List[float]],
    neighbors: List[Tuple[str, int]],
    out_png: Path,
    best_epoch: Optional[int] = None,
):
    """
    Un plot con todas las curvas de vecinos encima (val).
    Guarda: plots/neighbors_summary.png
    """
    if not neighbors:
        return
    out_png.parent.mkdir(parents=True, exist_ok=True)

    metrics = ("acc", "f1", "iou", "bin_acc_all")
    plt.figure(figsize=(9, 6))

    for metric in metrics:
        for name, _ in neighbors:
            key = f"val_{name}_{metric}"
            y = history.get(key, [])
            if isinstance(y, list) and len(y) > 0:
                plt.plot(y, label=key)

    if best_epoch is not None and int(best_epoch) > 0:
        plt.axvline(int(best_epoch) - 1, linestyle=":", label=f"best_epoch={int(best_epoch)}")

    plt.xlabel("epoch")
    plt.ylabel("metric value")
    plt.title("Neighbors (Val) — todas las curvas")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=250)
    plt.close()


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
                loss = loss_fn(logits_train.reshape(-1, C), y.reshape(-1))

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
            metrics = _compute_metrics_from_logits(logits_eval, y, C=C, d21_idx=d21_idx, bg=bg)

        else:
            with torch.no_grad():
                with ctx:
                    logits = model(xyz)
                    loss = loss_fn(logits.reshape(-1, C), y.reshape(-1))
                metrics = _compute_metrics_from_logits(logits, y, C=C, d21_idx=d21_idx, bg=bg)

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
# PARTE 4/5 — MAIN + PARSER + LOGGING + CKPTS + TEST + INFER
# (NO cierra el archivo; el cierre + infer helpers van en PARTE 5/5)
# ============================================================

def get_lr(optim: torch.optim.Optimizer) -> float:
    for g in optim.param_groups:
        return float(g.get("lr", 0.0))
    return 0.0


def append_metrics_csv(csv_path: Path, row: Dict[str, Any], header_order: Optional[List[str]] = None):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    header = header_order if header_order is not None else list(row.keys())
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in header})


def save_history_json(out_dir: Path, history: Dict[str, List[float]]):
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


def save_history_epoch_jsonl(out_dir: Path, epoch_row: Dict[str, Any]):
    p = out_dir / "history_epoch.jsonl"
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(epoch_row, ensure_ascii=False) + "\n")


def save_test_metrics(out_dir: Path, metrics: Dict[str, float], extra: Optional[Dict[str, Any]] = None):
    payload = {"test": metrics}
    if extra:
        payload["extra"] = extra
    (out_dir / "test_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_run_meta(out_dir: Path, args: argparse.Namespace, extra: Dict[str, Any]):
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).name),
        "args": vars(args),
        "extra": extra,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def _safe_open_logs(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    run_log = out_dir / "run.log"
    err_log = out_dir / "errors.log"
    return run_log, err_log


def log_line(path: Path, msg: str):
    ts = datetime.now().isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")


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
    p.add_argument("--knn_chunk_size", type=int, default=1024)

    p.add_argument("--bg_class", type=int, default=0)
    p.add_argument("--bg_weight", type=float, default=0.03)
    p.add_argument("--d21_internal", type=int, default=8)

    p.add_argument("--neighbor_teeth", type=str, default="", help='ej: "d11:1,d22:9" (idx internos)')

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--normalize", action="store_true", help="normalize_unit_sphere por muestra (ON recomendado)")

    p.add_argument("--plot_every", type=int, default=10)

    p.add_argument("--do_infer", action="store_true")
    p.add_argument("--infer_examples", type=int, default=12)
    p.add_argument("--infer_split", type=str, default="test", choices=["train", "val", "test"])

    return p.parse_args()


def main():
    args = parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_log, err_log = _safe_open_logs(out_dir)
    log_line(run_log, f"START script={Path(__file__).name}")
    log_line(run_log, f"data_dir={data_dir}")
    log_line(run_log, f"out_dir={out_dir}")
    log_line(run_log, f"args={vars(args)}")

    # seed
    set_seed(int(args.seed))

    # device
    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            log_line(run_log, f"CUDA device name: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass

    # num classes
    label_map = load_label_map(data_dir)
    C = infer_num_classes(data_dir, label_map)
    bg = int(args.bg_class)
    d21_idx = int(args.d21_internal)
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

    optim = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=int(args.epochs), eta_min=max(1e-6, float(args.lr) * 0.02)
    )

    loss_fn = make_loss_fn(num_classes=int(C), bg_class=int(bg), bg_weight=float(args.bg_weight), device=device)

    # outputs
    ckpt_best = out_dir / "best.pt"
    ckpt_last = out_dir / "last.pt"
    metrics_csv = out_dir / "metrics_epoch.csv"
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

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

    header_order = [
        "epoch", "lr",
        "train_loss", "train_f1_macro", "train_iou_macro", "train_acc_all", "train_acc_no_bg",
        "train_d21_acc", "train_d21_f1", "train_d21_iou", "train_d21_bin_acc_all", "train_pred_bg_frac",
        "val_loss", "val_f1_macro", "val_iou_macro", "val_acc_all", "val_acc_no_bg",
        "val_d21_acc", "val_d21_f1", "val_d21_iou", "val_d21_bin_acc_all", "val_pred_bg_frac",
    ]
    for name, _ in neighbors:
        header_order += [f"val_{name}_acc", f"val_{name}_f1", f"val_{name}_iou", f"val_{name}_bin_acc_all"]

    best_key = "val_iou_macro"
    best_val = -1e9
    best_epoch = -1

    print(f"[setup] data_dir={data_dir}")
    print(f"[setup] out_dir={out_dir}")
    print(f"[setup] device={device} | C={C} | bg={bg} | d21={d21_idx}")
    if neighbors:
        print(f"[setup] neighbors={neighbors}")
    log_line(run_log, f"setup device={device} C={C} bg={bg} d21={d21_idx} neighbors={neighbors}")

    t0 = time.time()

    # ==========================
    # TRAIN LOOP
    # ==========================
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

        history["train_loss"].append(float(tr["loss"]))
        history["val_loss"].append(float(va["loss"]))

        for k in ("f1_macro", "iou_macro", "acc_all", "acc_no_bg", "d21_acc", "d21_f1", "d21_iou", "d21_bin_acc_all", "pred_bg_frac"):
            history[f"train_{k}"].append(float(tr[k]))
            history[f"val_{k}"].append(float(va[k]))

        for name, _ in neighbors:
            history[f"val_{name}_acc"].append(float(neigh_val.get(f"{name}_acc", 0.0)))
            history[f"val_{name}_f1"].append(float(neigh_val.get(f"{name}_f1", 0.0)))
            history[f"val_{name}_iou"].append(float(neigh_val.get(f"{name}_iou", 0.0)))
            history[f"val_{name}_bin_acc_all"].append(float(neigh_val.get(f"{name}_bin_acc_all", 0.0)))

        msg = (
            f"[{epoch:03d}/{args.epochs}] "
            f"train loss={tr['loss']:.4f} f1m={tr['f1_macro']:.3f} ioum={tr['iou_macro']:.3f} "
            f"acc_all={tr['acc_all']:.3f} acc_no_bg={tr['acc_no_bg']:.3f} | "
            f"val loss={va['loss']:.4f} f1m={va['f1_macro']:.3f} ioum={va['iou_macro']:.3f} "
            f"acc_all={va['acc_all']:.3f} acc_no_bg={va['acc_no_bg']:.3f} | "
            f"d21 acc={va['d21_acc']:.3f} f1={va['d21_f1']:.3f} iou={va['d21_iou']:.3f} | "
            f"d21(bin all) acc={va['d21_bin_acc_all']:.3f} | "
            f"pred_bg_frac(val)={va['pred_bg_frac']:.3f} lr={lr:.2e}"
        )
        print(msg)
        log_line(run_log, msg)

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

        epoch_row = dict(row)
        epoch_row["best_epoch_so_far"] = int(best_epoch)
        epoch_row["best_val_so_far"] = float(best_val)
        save_history_epoch_jsonl(out_dir, epoch_row)

        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optim.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "num_classes": int(C),
                "bg_class": int(bg),
                "d21_internal": int(d21_idx),
                "args": vars(args),
                "best_epoch": int(best_epoch),
                "best_val": float(best_val),
            },
            ckpt_last,
        )
        save_history_json(out_dir, history)

        cur = float(va["iou_macro"])
        if cur > best_val:
            best_val = cur
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optim_state": optim.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "num_classes": int(C),
                    "bg_class": int(bg),
                    "d21_internal": int(d21_idx),
                    "args": vars(args),
                    "best_epoch": int(best_epoch),
                    "best_val": float(best_val),
                },
                ckpt_best,
            )
            log_line(run_log, f"NEW BEST epoch={best_epoch} best_val={best_val:.6f}")

    dt = time.time() - t0
    print(f"[done] Entrenamiento terminado en {dt/60:.1f} min. best_epoch={best_epoch}")

    # ==========================
    # TEST
    # ==========================
    if ckpt_best.exists():
        ck = torch.load(ckpt_best, map_location="cpu")
        model.load_state_dict(ck["model_state"], strict=True)

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

    save_test_metrics(out_dir, metrics={k: float(v) for k, v in te.items()})

    # ==========================
    # INFER (definido en PARTE 5)
    # ==========================
    if bool(args.do_infer):
        infer_and_save_examples(
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
        )

# ============================================================
# PARTE 5/5 — INFERENCIA COMPLETA + PLOTS 3D + CIERRE
# ============================================================

# ============================================================
# PLOTS 3D (GT vs Pred / errores / foco d21)
# ============================================================

def _setup_3d_axes(ax):
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=20, azim=45)
    ax.grid(False)


def plot_pointcloud_all_classes(
    xyz: np.ndarray,
    y_gt: np.ndarray,
    y_pred: np.ndarray,
    out_png: Path,
    C: int,
    title: str,
    s: float = 1.0,
):
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=y_gt, s=s, cmap="tab20")
    ax1.set_title("GT")
    _setup_3d_axes(ax1)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=y_pred, s=s, cmap="tab20")
    ax2.set_title("Pred")
    _setup_3d_axes(ax2)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=250)
    plt.close(fig)


# ==========================
# FIX CRÍTICO v8 (NO NumPy ufunc)
# ==========================
def plot_errors(
    xyz: np.ndarray,
    y_gt: np.ndarray,
    y_pred: np.ndarray,
    out_png: Path,
    bg: int,
    title: str,
    s: float = 1.0,
):
    """
    Rojo = error
    Gris = correcto

    ✔ Comparación hecha en TORCH
    ✔ Chequeo explícito de mismatch
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)

    t_gt = torch.as_tensor(y_gt).reshape(-1)
    t_pr = torch.as_tensor(y_pred).reshape(-1)

    if t_gt.numel() != t_pr.numel():
        raise RuntimeError(
            f"[infer] mismatch en plot_errors: "
            f"y_gt tiene {t_gt.numel()} elems, y_pred tiene {t_pr.numel()}"
        )

    errors = (t_gt != t_pr).cpu().numpy()

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        xyz[~errors, 0],
        xyz[~errors, 1],
        xyz[~errors, 2],
        c="lightgray",
        s=s,
    )
    ax.scatter(
        xyz[errors, 0],
        xyz[errors, 1],
        xyz[errors, 2],
        c="red",
        s=s,
    )

    ax.set_title("Errors (red)")
    _setup_3d_axes(ax)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=250)
    plt.close(fig)


# ==========================
# FIX CRÍTICO v8 (NO NumPy ufunc)
# ==========================
def plot_d21_focus(
    xyz: np.ndarray,
    y_gt: np.ndarray,
    y_pred: np.ndarray,
    out_png: Path,
    d21_idx: int,
    bg: int,
    title: str,
    s: float = 1.2,
):
    """
    Foco en clase d21.
    Azul = GT d21
    Verde = Pred d21

    ✔ Comparaciones hechas en TORCH
    ✔ Sin ufunc NumPy
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)

    t_gt = torch.as_tensor(y_gt).reshape(-1)
    t_pr = torch.as_tensor(y_pred).reshape(-1)

    if t_gt.numel() != t_pr.numel():
        raise RuntimeError(
            f"[infer] mismatch en plot_d21_focus: "
            f"y_gt tiene {t_gt.numel()} elems, y_pred tiene {t_pr.numel()}"
        )

    gt_mask = (t_gt == int(d21_idx)).cpu().numpy()
    pred_mask = (t_pr == int(d21_idx)).cpu().numpy()

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        xyz[:, 0], xyz[:, 1], xyz[:, 2],
        c="lightgray",
        s=0.5,
        alpha=0.3
    )

    ax.scatter(
        xyz[gt_mask, 0], xyz[gt_mask, 1], xyz[gt_mask, 2],
        c="blue",
        s=s,
        label="GT d21"
    )

    ax.scatter(
        xyz[pred_mask, 0], xyz[pred_mask, 1], xyz[pred_mask, 2],
        c="green",
        s=s,
        label="Pred d21"
    )

    ax.legend()
    ax.set_title("d21 focus")
    _setup_3d_axes(ax)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=250)
    plt.close(fig)


# ============================================================
# INFERENCIA CON TRAZABILIDAD (v8 robusta)
# ============================================================

def infer_and_save_examples(
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
):
    Xp = data_dir / f"X_{split}.npz"
    Yp = data_dir / f"Y_{split}.npz"
    if (not Xp.exists()) or (not Yp.exists()):
        print(f"[infer] No existe {Xp.name} o {Yp.name} en {data_dir}, salto inferencia.")
        return

    X = np.load(Xp)["X"]
    Y = np.load(Yp)["Y"]
    M = int(X.shape[0])
    K = min(int(n_examples), M)

    idx_path = _discover_index_csv(data_dir, split=split)
    idx_map = _read_index_csv(idx_path) if idx_path is not None else None

    out_inf = out_dir / "inference" / str(split)
    out_inf.mkdir(parents=True, exist_ok=True)

    model.eval()
    ctx = _get_autocast_ctx(device, bool(use_amp))

    with torch.no_grad():
        for i in range(K):
            xyz = X[i].astype(np.float32, copy=False)
            ygt = Y[i].astype(np.int32, copy=False)

            label = f"i={i:04d}"
            if idx_map is not None and i in idx_map:
                label = f"i={i:04d} | {_pick_trace_label(idx_map[i])}"

            tx = torch.as_tensor(xyz, dtype=torch.float32, device=device).unsqueeze(0)

            with ctx:
                logits = model(tx)

            pred = logits.argmax(dim=-1).squeeze(0).detach().cpu().numpy().astype(np.int32)

            # ✔ Chequeo explícito anti-mismatch
            if ygt.shape[0] != pred.shape[0]:
                raise RuntimeError(
                    f"[infer] N mismatch en ejemplo {i}: "
                    f"gt={ygt.shape}, pred={pred.shape}"
                )

            base = out_inf / f"ex_{i:04d}"

            plot_pointcloud_all_classes(
                xyz, ygt, pred,
                base.with_suffix(".all.png"),
                C=C,
                title=label,
                s=1.0,
            )

            plot_errors(
                xyz, ygt, pred,
                base.with_suffix(".err.png"),
                bg=bg,
                title=label,
                s=1.0,
            )

            plot_d21_focus(
                xyz, ygt, pred,
                base.with_suffix(".d21.png"),
                d21_idx=d21_idx,
                bg=bg,
                title=label,
                s=1.2,
            )

    print(f"[infer] Guardado en: {out_inf}")


# ============================================================
# CIERRE DEL SCRIPT
# ============================================================

if __name__ == "__main__":
    main()
# ------------------------------------------------------------
# EJEMPLOS (COMENTADOS) DE CÓMO CORRER ESTE SCRIPT
# ------------------------------------------------------------
# Asumiendo:
#   - Estás en: /home/htaucare/Tesis_Amaro/scripts_last_version
#   - El script se llama: train_dgcnn_classic_only_fixed_v8.py
#   - Dataset (flat) está en:
#       /home/htaucare/Tesis_Amaro/data/Teeth_3ds/fixed_split/8192/upper_only_surf_global_excl_wisdom_seed42_aug2
#
# 0) Activar env + entrar a carpeta:
#   conda activate enviroment
#   cd /home/htaucare/Tesis_Amaro/scripts_last_version
#
# 1) RUN “BALANCEADO” (recomendado; AMP + KNN chunk pequeño para menos VRAM):
#   CUDA_VISIBLE_DEVICES=0 python3 -u train_dgcnn_classic_only_fixed_v8.py \
#     --data_dir /home/htaucare/Tesis_Amaro/data/Teeth_3ds/fixed_split/8192/upper_only_surf_global_excl_wisdom_seed42_aug2 \
#     --out_dir  /home/htaucare/Tesis_Amaro/outputs/dgcnn/gpu0_v8_balanced \
#     --epochs 120 --batch_size 16 --lr 2e-4 --weight_decay 1e-4 --dropout 0.5 \
#     --num_workers 6 --device cuda --use_amp --grad_clip 1.0 \
#     --k 20 --emb_dims 768 --knn_chunk_size 256 \
#     --bg_class 0 --bg_weight 0.03 --d21_internal 8 \
#     --neighbor_teeth "d11:1,d22:9" \
#     --normalize \
#     --do_infer --infer_examples 12 --infer_split test
#
# 2) RUN “ULTRA-SAFE VRAM” (si te preocupa OOM; baja batch_size):
#   CUDA_VISIBLE_DEVICES=0 python3 -u train_dgcnn_classic_only_fixed_v8.py \
#     --data_dir /home/htaucare/Tesis_Amaro/data/Teeth_3ds/fixed_split/8192/upper_only_surf_global_excl_wisdom_seed42_aug2 \
#     --out_dir  /home/htaucare/Tesis_Amaro/outputs/dgcnn/gpu0_v8_ultrasafe_bs12 \
#     --epochs 120 --batch_size 12 --lr 2e-4 --weight_decay 1e-4 --dropout 0.5 \
#     --num_workers 6 --device cuda --use_amp --grad_clip 1.0 \
#     --k 20 --emb_dims 768 --knn_chunk_size 256 \
#     --bg_class 0 --bg_weight 0.03 --d21_internal 8 \
#     --neighbor_teeth "d11:1,d22:9" \
#     --normalize \
#     --do_infer --infer_examples 12 --infer_split test
#
# 3) RUN “MÁXIMA CALIDAD” (más caro; típicamente más VRAM):
#   CUDA_VISIBLE_DEVICES=0 python3 -u train_dgcnn_classic_only_fixed_v8.py \
#     --data_dir /home/htaucare/Tesis_Amaro/data/Teeth_3ds/fixed_split/8192/upper_only_surf_global_excl_wisdom_seed42_aug2 \
#     --out_dir  /home/htaucare/Tesis_Amaro/outputs/dgcnn/gpu0_v8_max_quality \
#     --epochs 120 --batch_size 16 --lr 2e-4 --weight_decay 1e-4 --dropout 0.5 \
#     --num_workers 6 --device cuda --use_amp --grad_clip 1.0 \
#     --k 20 --emb_dims 1024 --knn_chunk_size 1024 \
#     --bg_class 0 --bg_weight 0.03 --d21_internal 8 \
#     --neighbor_teeth "d11:1,d22:9" \
#     --normalize \
#     --do_infer --infer_examples 12 --infer_split test
#
# 4) SOLO TRAIN+TEST (sin inferencia):
#   CUDA_VISIBLE_DEVICES=0 python3 -u train_dgcnn_classic_only_fixed_v8.py \
#     --data_dir /home/htaucare/Tesis_Amaro/data/Teeth_3ds/fixed_split/8192/upper_only_surf_global_excl_wisdom_seed42_aug2 \
#     --out_dir  /home/htaucare/Tesis_Amaro/outputs/dgcnn/gpu0_v8_noinfer \
#     --epochs 120 --batch_size 16 --lr 2e-4 --weight_decay 1e-4 --dropout 0.5 \
#     --num_workers 6 --device cuda --use_amp --grad_clip 1.0 \
#     --k 20 --emb_dims 768 --knn_chunk_size 256 \
#     --bg_class 0 --bg_weight 0.03 --d21_internal 8 \
#     --neighbor_teeth "d11:1,d22:9" \
#     --normalize
#
# Tips rápidos:
#   - Si te quedas sin VRAM: baja --batch_size o baja --k (20->16) o baja --emb_dims.
#   - Si quieres menos picos de memoria de KNN: baja --knn_chunk_size (ej 256 o 128).
#   - AMP (--use_amp) casi siempre conviene en RTX 3090.
# ------------------------------------------------------------