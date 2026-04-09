#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_dgcnn_classic_only_fixed_v8_patch.py

DGCNN (EdgeConv) – Segmentación multiclase dental 3D
(PATCH FINAL alineado con pointnet_classic_final_v8_patch.py
 y pointnettransformer_classic_final_v5_patch.py)

Objetivo:
✅ Mantener la arquitectura DGCNN original (EdgeConv + KNN chunk-safe)
✅ Mantener outputs/trazabilidad del script base
✅ Estandarizar con PointNet/Transformer:
   - history.json
   - history_epoch.jsonl
   - metrics_epoch.csv
   - run_meta.json
   - best.pt / last.pt
   - test_metrics.json
   - plots Train vs Val
   - inferencia trazable
✅ Agregar:
   - test_metrics_filtered.json
   - ignored_test_samples.json
   - inferencia filtrada (only_bg)
   - ignored_inference_samples.json
   - infer_examples funcionando de verdad
   - plots faltantes de d21
   - plots de neighbors
✅ Mantener las mismas librerías/estilo de graficación
✅ Aplicar grad clipping real
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
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ============================================================
# SEED / IO / LOG
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


def save_json(obj: Any, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def append_jsonl(obj: Any, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def ensure_dir(p: Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _fmt_hms(seconds: float) -> str:
    seconds = float(seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:d}h {m:02d}m {s:05.2f}s"


def log_line(msg: str, log_path: Optional[Path] = None, also_print: bool = True):
    line = f"[{_now_str()}] {msg}"
    if also_print:
        print(line, flush=True)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ============================================================
# AMP HELPERS
# ============================================================
def _amp_enabled(device: torch.device, use_amp: bool) -> bool:
    return bool(use_amp) and (device.type == "cuda") and torch.cuda.is_available()


def _make_grad_scaler(device: torch.device, use_amp: bool):
    enabled = _amp_enabled(device, use_amp)
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


class _AutocastCtx:
    def __init__(self, device: torch.device, enabled: bool):
        self.device = device
        self.enabled = bool(enabled)
        self.ctx = None

    def __enter__(self):
        if self.enabled and self.device.type == "cuda":
            try:
                self.ctx = torch.amp.autocast("cuda", enabled=True)
            except Exception:
                self.ctx = torch.cuda.amp.autocast(enabled=True)
        else:
            self.ctx = torch.autocast(device_type="cpu", enabled=False)
        return self.ctx.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.ctx.__exit__(exc_type, exc_val, exc_tb)


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
# DATASET
# ============================================================
class NPZDataset(Dataset):
    """
    X: [B,N,3]
    Y: [B,N]

    return_index:
      - False: (xyz, y)
      - True : (xyz, y, idx_local)
    """
    def __init__(self, X_path: Path, Y_path: Path, normalize: bool = True, return_index: bool = False):
        self.X = np.load(X_path)["X"]
        self.Y = np.load(Y_path)["Y"]
        self.normalize = bool(normalize)
        self.return_index = bool(return_index)

        assert self.X.shape[0] == self.Y.shape[0]
        assert self.X.shape[1] == self.Y.shape[1]

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, idx):
        idx = int(idx)
        x = np.ascontiguousarray(self.X[idx], dtype=np.float32)  # [N,3]
        y = np.ascontiguousarray(self.Y[idx], dtype=np.int64)    # [N]

        xyz = torch.as_tensor(x, dtype=torch.float32)
        lab = torch.as_tensor(y, dtype=torch.int64)

        if self.normalize:
            xyz = normalize_unit_sphere(xyz)

        if self.return_index:
            return xyz, lab, torch.tensor(idx, dtype=torch.int64)
        return xyz, lab


def _make_loader(ds: Dataset, bs: int, nw: int, shuffle: bool) -> DataLoader:
    common = dict(
        batch_size=int(bs),
        num_workers=int(nw),
        pin_memory=True,
        persistent_workers=(int(nw) > 0),
        drop_last=False,
        shuffle=bool(shuffle),
    )
    if int(nw) > 0:
        common["prefetch_factor"] = 2
    return DataLoader(ds, **common)


def make_loaders(data_dir: Path, bs: int, nw: int, normalize: bool = True):
    ds_tr = NPZDataset(data_dir / "X_train.npz", data_dir / "Y_train.npz", normalize=normalize, return_index=False)
    ds_va = NPZDataset(data_dir / "X_val.npz",   data_dir / "Y_val.npz",   normalize=normalize, return_index=False)
    ds_te = NPZDataset(data_dir / "X_test.npz",  data_dir / "Y_test.npz",  normalize=normalize, return_index=False)

    dl_tr = _make_loader(ds_tr, bs=bs, nw=nw, shuffle=True)
    dl_va = _make_loader(ds_va, bs=bs, nw=nw, shuffle=False)
    dl_te = _make_loader(ds_te, bs=bs, nw=nw, shuffle=False)
    return dl_tr, dl_va, dl_te


def make_infer_loader(data_dir: Path, split: str, bs: int, nw: int, normalize: bool = True) -> DataLoader:
    split = str(split).lower().strip()

    if split == "train":
        ds = NPZDataset(data_dir / "X_train.npz", data_dir / "Y_train.npz", normalize=normalize, return_index=True)
    elif split == "val":
        ds = NPZDataset(data_dir / "X_val.npz",   data_dir / "Y_val.npz",   normalize=normalize, return_index=True)
    elif split == "test":
        ds = NPZDataset(data_dir / "X_test.npz",  data_dir / "Y_test.npz",  normalize=normalize, return_index=True)
    else:
        raise ValueError(f"split inválido: {split}")

    return _make_loader(ds, bs=bs, nw=nw, shuffle=False)


# ============================================================
# LABEL MAP + NUM CLASSES
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
    if isinstance(label_map, dict) and len(label_map) > 0:
        try:
            mx = max(int(v) for v in label_map.values())
            return int(mx + 1)
        except Exception:
            pass

    maxy = -1
    for split in ("train", "val", "test"):
        yp = data_dir / f"Y_{split}.npz"
        if yp.exists():
            y = np.load(yp)["Y"]
            maxy = max(maxy, int(y.max()))

    if maxy < 0:
        raise RuntimeError("No se pudo inferir num_classes.")
    return int(maxy + 1)


# ============================================================
# NEIGHBORS PARSER
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
            raise ValueError(f"Formato inválido en neighbor_teeth: '{part}' (usa nombre:idx)")
        name, idx = part.split(":", 1)
        out.append((name.strip(), int(idx)))
    return out


# ============================================================
# TRAZABILIDAD index_{split}.csv
# ============================================================
def _discover_index_csv(data_dir: Path, split: str) -> Optional[Path]:
    """
    Busca index_{split}.csv en:
      1) data_dir/index_{split}.csv
      2) ancestros directos
      3) fallback en merged_*
    """
    split = str(split).lower()
    fname = f"index_{split}.csv"

    p = data_dir / fname
    if p.exists():
        return p

    cur = data_dir
    for _ in range(10):
        p = cur / fname
        if p.exists():
            return p
        if cur.parent == cur:
            break
        cur = cur.parent

    candidates = []
    cur = data_dir
    ancestors = [cur]
    for _ in range(10):
        if cur.parent == cur:
            break
        cur = cur.parent
        ancestors.append(cur)

    for anc in ancestors:
        for mg in anc.glob("merged_*"):
            p = mg / fname
            if p.exists():
                candidates.append(p)

    if len(candidates) == 0:
        return None

    candidates = sorted(candidates, key=lambda z: z.stat().st_mtime, reverse=True)
    return candidates[0]


def _read_index_csv(p: Optional[Path]) -> Optional[Dict[int, Dict[str, str]]]:
    """
    Lee index_{split}.csv y retorna:
      { row_idx_int : {col: value, ...}, ... }
    """
    if p is None or (not Path(p).exists()):
        return None

    out: Dict[int, Dict[str, str]] = {}
    try:
        with Path(p).open("r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for i, row in enumerate(r):
                rid = None
                for k in ("row_i", "idx", "index", "row", "i"):
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
# DGCNN (KNN chunk-safe) + EdgeConv + Modelo
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

    xx = (x ** 2).sum(dim=1)  # [B, N]

    chunk_size = int(chunk_size or 0)
    if chunk_size <= 0 or chunk_size >= N:
        qx = torch.bmm(x.transpose(2, 1), x)                  # [B, N, N]
        dist = xx.unsqueeze(2) + xx.unsqueeze(1) - 2.0 * qx   # [B, N, N]
        idx = dist.topk(k=k, dim=-1, largest=False, sorted=False).indices
        return idx

    idx_chunks: List[torch.Tensor] = []
    for s in range(0, N, chunk_size):
        e = min(N, s + chunk_size)
        q = x[:, :, s:e]                         # [B, C, M]
        qq = (q ** 2).sum(dim=1)                 # [B, M]
        qx = torch.bmm(q.transpose(2, 1), x)     # [B, M, N]
        dist = qq.unsqueeze(2) + xx.unsqueeze(1) - 2.0 * qx
        idx = dist.topk(k=k, dim=-1, largest=False, sorted=False).indices
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
    return: [B, 2C, N, k] = concat(x_j - x_i, x_i)
    """
    B, C, N = x.shape
    if idx is None:
        idx = knn(x, k=int(k), chunk_size=int(knn_chunk_size))  # [B,N,k]
    k = int(idx.shape[-1])

    device = x.device
    idx_base = torch.arange(B, device=device).view(B, 1, 1) * N
    idx = (idx + idx_base).reshape(-1)

    x_t = x.transpose(2, 1).contiguous()        # [B,N,C]
    feat = x_t.reshape(B * N, C)[idx, :]        # [B*N*k, C]
    feat = feat.view(B, N, k, C)                # [B,N,k,C]

    x_i = x_t.view(B, N, 1, C).expand(-1, -1, k, -1)
    edge = torch.cat((feat - x_i, x_i), dim=3)  # [B,N,k,2C]
    return edge.permute(0, 3, 1, 2).contiguous()  # [B,2C,N,k]


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

        # EdgeConv stacks (arquitectura intacta)
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
# LOSS
# ============================================================
def make_loss_fn(num_classes: int, bg_class: int, bg_weight: float, device: torch.device) -> nn.Module:
    """
    CrossEntropy con pesos por clase:
      - bg_class tiene peso bg_weight
      - resto 1.0

    BG se incluye en la loss, NO se ignora.
    """
    C = int(num_classes)
    bg = int(bg_class)
    w = torch.ones(C, dtype=torch.float32, device=device)
    if 0 <= bg < C:
        w[bg] = float(bg_weight)
    return nn.CrossEntropyLoss(weight=w)


# ============================================================
# MÉTRICAS
# ============================================================
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

    include_bg=False => excluye puntos con gt==bg
    include_bg=True  => incluye bg
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
def _tooth_metrics_binary(pred: torch.Tensor, gt: torch.Tensor, tooth_idx: int, bg: int = 0) -> Dict[str, float]:
    acc, f1, iou = d21_metrics_binary(
        pred=pred, gt=gt, d21_idx=int(tooth_idx), bg=int(bg), include_bg=False
    )
    acc_all, f1_all, iou_all = d21_metrics_binary(
        pred=pred, gt=gt, d21_idx=int(tooth_idx), bg=int(bg), include_bg=True
    )
    return {
        "acc": float(acc),
        "f1": float(f1),
        "iou": float(iou),
        "bin_acc_all": float(acc_all),
        "bin_f1_all": float(f1_all),
        "bin_iou_all": float(iou_all),
    }


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
    acc_no_bg = float((pred[mask] == y[mask]).float().mean().item()) if mask.any() else 0.0

    f1m, ioum = macro_metrics_no_bg(pred, y, C=C, bg=bg)

    d21_acc, d21_f1, d21_iou = d21_metrics_binary(
        pred, y, d21_idx=int(d21_idx), bg=int(bg), include_bg=False
    )
    d21_bin_acc_all, _, _ = d21_metrics_binary(
        pred, y, d21_idx=int(d21_idx), bg=int(bg), include_bg=True
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

    return out


@torch.no_grad()
def eval_neighbors_on_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    neighbor_list: List[Tuple[str, int]],
    bg: int,
    use_amp: bool,
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

            with _AutocastCtx(device, _amp_enabled(device, use_amp)):
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
# PLOTS
# ============================================================
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
# run_epoch
# ============================================================
@torch.no_grad()
def _mean_dict(accum: Dict[str, float], n: int) -> Dict[str, float]:
    n = max(1, int(n))
    return {k: v / n for k, v in accum.items()}


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
    train_metrics_eval: bool = False,
) -> Dict[str, float]:
    """
    train=True:
      - forward/backward con model.train()
      - métricas desde logits.detach() o segundo forward eval si train_metrics_eval=True

    train=False:
      - eval normal

    Devuelve métricas base + neighbors.
    """
    neighbor_list = neighbor_list or []

    if train:
        model.train()
    else:
        model.eval()

    sums = defaultdict(float)
    neigh_sums = defaultdict(float)
    nb = 0

    scaler = _make_grad_scaler(device, use_amp)

    for xyz, y in loader:
        xyz = xyz.to(device, non_blocking=True)   # [B,N,3]
        y = y.to(device, non_blocking=True)       # [B,N]

        if train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)

            with _AutocastCtx(device, _amp_enabled(device, use_amp)):
                logits = model(xyz)
                loss = loss_fn(logits.reshape(-1, C), y.reshape(-1))

            if scaler is not None:
                scaler.scale(loss).backward()

                if grad_clip is not None and grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))

                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                optimizer.step()

            logits_eval = model(xyz) if train_metrics_eval else logits.detach()

        else:
            with torch.no_grad():
                with _AutocastCtx(device, _amp_enabled(device, use_amp)):
                    logits = model(xyz)
                    loss = loss_fn(logits.reshape(-1, C), y.reshape(-1))
                logits_eval = logits

        m = _compute_metrics_from_logits(logits_eval, y, C, d21_idx, bg)

        sums["loss"] += float(loss.item())
        for k, v in m.items():
            sums[k] += float(v)

        if neighbor_list:
            nm = _neighbors_metrics_from_logits(logits_eval, y, neighbor_list, bg)
            for k, v in nm.items():
                neigh_sums[k] += float(v)

        nb += 1

    out = _mean_dict(sums, nb)

    for k, v in neigh_sums.items():
        out[k] = v / max(1, nb)

    return out


# ============================================================
# run_epoch_filtered_only_bg
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
    use_amp: bool,
):
    """
    Igual que test normal, pero ignora samples cuyo GT es solo background.
    Loader debe retornar (xyz, y, row_i) o (xyz, y).
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

        with _AutocastCtx(device, _amp_enabled(device, use_amp)):
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
    p = argparse.ArgumentParser()

    p.add_argument("--data_dir", type=str, required=True, help="carpeta con X_train.npz/Y_train.npz etc")
    p.add_argument("--out_dir", type=str, required=True)

    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.5)

    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--infer_num_workers", type=int, default=None)

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
    p.add_argument("--neighbor_eval_split", type=str, default="val", choices=["val", "test", "both", "none"])
    p.add_argument("--neighbor_every", type=int, default=1)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--normalize", action="store_true", help="normalize_unit_sphere por muestra (ON recomendado)")

    p.add_argument("--do_infer", action="store_true")
    p.add_argument("--infer_examples", type=int, default=12)
    p.add_argument("--infer_split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--index_csv", type=str, default=None)
    p.add_argument("--train_metrics_eval", action="store_true")

    return p.parse_args()


# ============================================================
# HELPERS neighbor control
# ============================================================
def should_eval_neighbors(split: str, mode: str) -> bool:
    mode = str(mode).lower()
    if mode == "none":
        return False
    if mode == "both":
        return True
    return split == mode


def merge_neighbor_metrics(base: Dict[str, float], neigh: Dict[str, float]) -> Dict[str, float]:
    out = dict(base)
    for k, v in neigh.items():
        out[k] = float(v)
    return out


# ============================================================
# MAIN
# ============================================================
def main():
    args = parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    ensure_dir(out_dir)
    ensure_dir(out_dir / "plots")
    ensure_dir(out_dir / "inference")

    run_log = out_dir / "run.log"
    err_log = out_dir / "errors.log"

    set_seed(int(args.seed))

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

    bg = int(args.bg_class)

    bg_tr = float((Ytr == bg).mean())
    bg_va = float((Yva == bg).mean())
    bg_te = float((Yte == bg).mean())

    label_map = load_label_map(data_dir)
    C = infer_num_classes(data_dir, label_map)
    d21_idx = int(args.d21_internal)

    log_line(f"[SANITY] num_classes C = {C}", run_log)
    log_line(f"[SANITY] bg_frac train/val/test = {bg_tr:.4f} {bg_va:.4f} {bg_te:.4f}", run_log)
    log_line(f"[SANITY] baseline acc_all (always-bg) train/val/test = {bg_tr:.4f} {bg_va:.4f} {bg_te:.4f}", run_log)

    if not (0 <= int(d21_idx) < C):
        raise ValueError(f"d21_internal fuera de rango: {d21_idx} (C={C})")

    # =========================================================
    # LOADERS
    # =========================================================
    dl_tr, dl_va, dl_te = make_loaders(
        data_dir=data_dir,
        bs=int(args.batch_size),
        nw=int(args.num_workers),
        normalize=bool(args.normalize),
    )

    # =========================================================
    # MODEL / OPT / LOSS
    # =========================================================
    model = DGCNN_Seg(
        num_classes=int(C),
        k=int(args.k),
        emb_dims=int(args.emb_dims),
        dropout=float(args.dropout),
        knn_chunk_size=int(args.knn_chunk_size),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(args.epochs),
        eta_min=1e-6,
    )

    loss_fn = make_loss_fn(
        num_classes=int(C),
        bg_class=int(bg),
        bg_weight=float(args.bg_weight),
        device=device,
    )

    # =========================================================
    # NEIGHBORS
    # =========================================================
    neighbors = parse_neighbors(args.neighbor_teeth)
    if neighbors:
        bad = [(n, i) for (n, i) in neighbors if not (0 <= int(i) < C)]
        if bad:
            raise ValueError(f"neighbor_teeth fuera de rango (C={C}): {bad}")
        log_line(f"[NEIGHBORS] parsed: {neighbors}", run_log)
    else:
        log_line("[NEIGHBORS] none", run_log)

    neighbor_cols = []
    for name, _ in neighbors:
        neighbor_cols += [
            f"{name}_acc", f"{name}_f1", f"{name}_iou", f"{name}_bin_acc_all"
        ]

    # =========================================================
    # RUN META
    # =========================================================
    run_meta = {
        "script_name": "train_dgcnn_classic_only_fixed_v8_patch.py",
        "start_time": datetime.now().isoformat(timespec="seconds"),
        "data_dir": str(data_dir),
        "out_dir": str(out_dir),
        "device": str(device),
        "seed": int(args.seed),
        "num_classes": int(C),
        "bg_class": int(bg),
        "bg_weight": float(args.bg_weight),
        "d21_internal": int(d21_idx),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "dropout": float(args.dropout),
        "grad_clip": float(args.grad_clip),
        "use_amp": bool(args.use_amp),
        "normalize_unit_sphere": bool(args.normalize),
        "train_metrics_eval": bool(args.train_metrics_eval),
        "infer_examples": int(args.infer_examples),
        "do_infer": bool(args.do_infer),
        "infer_split": str(args.infer_split),
        "infer_num_workers": int(args.infer_num_workers) if args.infer_num_workers is not None else None,
        "index_csv": str(args.index_csv) if args.index_csv else "",
        "neighbor_teeth": str(args.neighbor_teeth),
        "neighbor_eval_split": str(args.neighbor_eval_split),
        "neighbor_every": int(args.neighbor_every),
        "neighbor_parsed": neighbors,
        "k": int(args.k),
        "emb_dims": int(args.emb_dims),
        "knn_chunk_size": int(args.knn_chunk_size),
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

    base_metrics = (
        "loss", "acc_all", "acc_no_bg",
        "f1_macro", "iou_macro",
        "d21_acc", "d21_f1", "d21_iou",
        "d21_bin_acc_all",
        "pred_bg_frac",
    )
    for k in base_metrics:
        history[f"train_{k}"] = []
        history[f"val_{k}"] = []

    for name, _ in neighbors:
        for met in ("acc", "f1", "iou", "bin_acc_all"):
            history[f"train_{name}_{met}"] = []
            history[f"val_{name}_{met}"] = []

    best_val_d21_f1 = -1.0
    best_epoch = -1
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"

    t0 = time.time()
    history_jsonl = out_dir / "history_epoch.jsonl"


    print(f"[setup] data_dir={data_dir}")
    print(f"[setup] out_dir={out_dir}")
    print(f"[setup] device={device} | C={C} | bg={bg} | d21={d21_idx}")
    if neighbors:
        print(f"[setup] neighbors={neighbors}")

    log_line(f"[setup] device={device} C={C} bg={bg} d21={d21_idx} neighbors={neighbors}", run_log)

    # =========================================================
    # TRAIN LOOP
    # =========================================================
    for epoch in range(1, int(args.epochs) + 1):
        t_ep = time.time()

        tr = run_epoch(
            model=model,
            loader=dl_tr,
            optimizer=optimizer,
            loss_fn=loss_fn,
            C=int(C),
            d21_idx=int(d21_idx),
            device=device,
            bg=int(bg),
            train=True,
            use_amp=bool(args.use_amp),
            grad_clip=float(args.grad_clip) if float(args.grad_clip) > 0 else None,
            neighbor_list=[],
            train_metrics_eval=bool(args.train_metrics_eval),
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
            neighbor_list=[],
            train_metrics_eval=False,
        )

        do_neighbor_now = (
            len(neighbors) > 0
            and int(args.neighbor_every) > 0
            and (epoch % int(args.neighbor_every) == 0)
        )

        tr_neigh = {}
        va_neigh = {}

        if do_neighbor_now:
            tr_neigh = eval_neighbors_on_loader(
                model=model,
                loader=dl_tr,
                device=device,
                neighbor_list=neighbors,
                bg=int(bg),
                use_amp=bool(args.use_amp),
            )

            if should_eval_neighbors("val", args.neighbor_eval_split):
                va_neigh = eval_neighbors_on_loader(
                    model=model,
                    loader=dl_va,
                    device=device,
                    neighbor_list=neighbors,
                    bg=int(bg),
                    use_amp=bool(args.use_amp),
                )

        tr = merge_neighbor_metrics(tr, tr_neigh)
        va = merge_neighbor_metrics(va, va_neigh)

        scheduler.step()
        lr = float(optimizer.param_groups[0]["lr"])
        sec = float(time.time() - t_ep)

        # history
        for k, v in tr.items():
            key = f"train_{k}"
            if key in history:
                history[key].append(float(v))

        for k, v in va.items():
            key = f"val_{k}"
            if key in history:
                history[key].append(float(v))

        for name, _ in neighbors:
            for met in ("acc", "f1", "iou", "bin_acc_all"):
                kt = f"train_{name}_{met}"
                kv = f"val_{name}_{met}"

                if len(history[kt]) < epoch:
                    prev = history[kt][-1] if len(history[kt]) > 0 else 0.0
                    history[kt].append(float(prev))

                if len(history[kv]) < epoch:
                    prev = history[kv][-1] if len(history[kv]) > 0 else 0.0
                    history[kv].append(float(prev))

        append_jsonl({"epoch": epoch, "train": tr, "val": va}, history_jsonl)

        # csv
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            wcsv = csv.writer(f)

            def _row(split: str, m: Dict[str, float]):
                base = [
                    epoch, split,
                    m["loss"],
                    m["acc_all"], m["acc_no_bg"],
                    m["f1_macro"], m["iou_macro"],
                    m["d21_acc"], m["d21_f1"], m["d21_iou"],
                    m["d21_bin_acc_all"],
                    m["pred_bg_frac"],
                    lr,
                    sec,
                ]
                for name, _ in neighbors:
                    base += [
                        m.get(f"{name}_acc", 0.0),
                        m.get(f"{name}_f1", 0.0),
                        m.get(f"{name}_iou", 0.0),
                        m.get(f"{name}_bin_acc_all", 0.0),
                    ]
                return base

            wcsv.writerow(_row("train", tr))
            wcsv.writerow(_row("val", va))

        msg = (
            f"[{epoch:03d}/{args.epochs}] "
            f"train loss={tr['loss']:.4f} f1m={tr['f1_macro']:.3f} ioum={tr['iou_macro']:.3f} | "
            f"val loss={va['loss']:.4f} f1m={va['f1_macro']:.3f} ioum={va['iou_macro']:.3f} | "
            f"d21 acc={va['d21_acc']:.3f} f1={va['d21_f1']:.3f} iou={va['d21_iou']:.3f}"
        )

        if neighbors:
            neigh_txt = []
            for name, _ in neighbors:
                neigh_txt.append(f"{name}_f1={va.get(f'{name}_f1', 0.0):.3f}")
            msg += " | " + " ".join(neigh_txt)

        print(msg)
        log_line(msg, run_log)

        # save last
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "num_classes": int(C),
                "bg_class": int(bg),
                "d21_internal": int(d21_idx),
                "args": vars(args),
                "best_epoch": int(best_epoch),
                "best_val_d21_f1": float(best_val_d21_f1),
            },
            last_path,
        )

        save_json(dict(history), out_dir / "history.json")

        cur = float(va["d21_f1"])
        if cur > best_val_d21_f1:
            best_val_d21_f1 = cur
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "num_classes": int(C),
                    "bg_class": int(bg),
                    "d21_internal": int(d21_idx),
                    "args": vars(args),
                    "best_epoch": int(best_epoch),
                    "best_val_d21_f1": float(best_val_d21_f1),
                },
                best_path,
            )
            log_line(f"[BEST] epoch={best_epoch} best_val_d21_f1={best_val_d21_f1:.6f}", run_log)

    dt = time.time() - t0
    print(f"[done] Entrenamiento terminado en {dt/60:.1f} min. best_epoch={best_epoch}")
    log_line(f"[done] training finished in {_fmt_hms(dt)} | best_epoch={best_epoch}", run_log)

    # =========================================================
    # PLOTS
    # =========================================================
    plot_train_val("loss", history["train_loss"], history["val_loss"], out_dir / "plots/loss.png", best_epoch)
    plot_train_val("f1_macro", history["train_f1_macro"], history["val_f1_macro"], out_dir / "plots/f1_macro.png", best_epoch)
    plot_train_val("iou_macro", history["train_iou_macro"], history["val_iou_macro"], out_dir / "plots/iou_macro.png", best_epoch)

    plot_train_val("d21_acc", history["train_d21_acc"], history["val_d21_acc"], out_dir / "plots/d21_acc.png", best_epoch)
    plot_train_val("d21_f1", history["train_d21_f1"], history["val_d21_f1"], out_dir / "plots/d21_f1.png", best_epoch)
    plot_train_val("d21_iou", history["train_d21_iou"], history["val_d21_iou"], out_dir / "plots/d21_iou.png", best_epoch)

    for name, _ in neighbors:
        plot_train_val(
            f"{name}_acc",
            history[f"train_{name}_acc"],
            history[f"val_{name}_acc"],
            out_dir / f"plots/{name}_acc.png",
            best_epoch,
        )
        plot_train_val(
            f"{name}_f1",
            history[f"train_{name}_f1"],
            history[f"val_{name}_f1"],
            out_dir / f"plots/{name}_f1.png",
            best_epoch,
        )
        plot_train_val(
            f"{name}_iou",
            history[f"train_{name}_iou"],
            history[f"val_{name}_iou"],
            out_dir / f"plots/{name}_iou.png",
            best_epoch,
        )
        plot_train_val(
            f"{name}_bin_acc_all",
            history[f"train_{name}_bin_acc_all"],
            history[f"val_{name}_bin_acc_all"],
            out_dir / f"plots/{name}_bin_acc_all.png",
            best_epoch,
        )

    # =========================================================
    # TEST NORMAL
    # =========================================================
    if best_path.exists():
        ck = torch.load(best_path, map_location=device)
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
        neighbor_list=[],
        train_metrics_eval=False,
    )

    if len(neighbors) > 0 and should_eval_neighbors("test", args.neighbor_eval_split):
        te_neigh = eval_neighbors_on_loader(
            model=model,
            loader=dl_te,
            device=device,
            neighbor_list=neighbors,
            bg=int(bg),
            use_amp=bool(args.use_amp),
        )
        te = merge_neighbor_metrics(te, te_neigh)

    te["best_epoch"] = int(best_epoch)
    save_json(te, out_dir / "test_metrics.json")

    # =========================================================
    # TEST FILTRADO (ONLY_BG)
    # =========================================================
    dl_te_inf = make_infer_loader(
        data_dir=data_dir,
        split="test",
        bs=1,
        nw=0,
        normalize=bool(args.normalize),
    )

    te_filtered = run_epoch_filtered_only_bg(
        model=model,
        loader=dl_te_inf,
        loss_fn=loss_fn,
        C=int(C),
        d21_idx=int(d21_idx),
        device=device,
        bg=int(bg),
        use_amp=bool(args.use_amp),
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
    # INFERENCIA FILTRADA Y TRAZABLE
    # =========================================================
    if bool(args.do_infer):
        infer_nw = int(args.infer_num_workers) if args.infer_num_workers is not None else min(int(args.num_workers), 2)

        infer_loader = make_infer_loader(
            data_dir=data_dir,
            split=str(args.infer_split),
            bs=1,
            nw=infer_nw,
            normalize=bool(args.normalize),
        )

        index_csv = Path(args.index_csv) if args.index_csv else _discover_index_csv(data_dir, args.infer_split)
        index_map = _read_index_csv(index_csv) if index_csv else None

        inf_root = ensure_dir(out_dir / "inference")
        inf_all = ensure_dir(inf_root / "inference_all")
        inf_err = ensure_dir(inf_root / "inference_errors")
        inf_d21 = ensure_dir(inf_root / "inference_d21")

        manifest = []
        ignored_rows = []
        done_examples = 0
        max_examples = int(args.infer_examples)

        model.eval()
        with torch.no_grad():
            for xyz, y, row_i in infer_loader:
                ri = int(row_i.item())

                # filtra only_bg
                y_np_pre = y[0].cpu().numpy()
                vals = np.unique(y_np_pre)
                if len(vals) == 1 and int(vals[0]) == int(bg):
                    ignored_rows.append(ri)
                    continue

                # respeta infer_examples
                if done_examples >= max_examples:
                    break

                xyz = xyz.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                with _AutocastCtx(device, _amp_enabled(device, bool(args.use_amp))):
                    logits = model(xyz)
                pred = logits.argmax(dim=-1)

                xyz_np = xyz[0].detach().cpu().numpy()
                y_np = y[0].detach().cpu().numpy()
                pr_np = pred[0].detach().cpu().numpy()

                meta = index_map.get(ri, {}) if index_map else {}

                tag = f"{args.infer_split}_row{ri}"
                if meta:
                    trace_label = _pick_trace_label(meta)
                    if trace_label:
                        tag += "_" + _sanitize_tag(trace_label)

                title_parts = [f"{args.infer_split} row={ri}"]
                if meta:
                    trace_label = _pick_trace_label(meta)
                    if trace_label:
                        title_parts.append(trace_label)
                title = " | ".join(title_parts)

                p_all = inf_all / f"{tag}.png"
                p_err = inf_err / f"{tag}.png"
                p_d21 = inf_d21 / f"{tag}.png"

                plot_pointcloud_all_classes(
                    xyz=xyz_np,
                    y_gt=y_np,
                    y_pred=pr_np,
                    out_png=p_all,
                    C=int(C),
                    title=title,
                    s=1.0,
                )
                plot_errors(
                    xyz=xyz_np,
                    y_gt=y_np,
                    y_pred=pr_np,
                    out_png=p_err,
                    bg=int(bg),
                    title=title,
                    s=1.0,
                )
                plot_d21_focus(
                    xyz=xyz_np,
                    y_gt=y_np,
                    y_pred=pr_np,
                    out_png=p_d21,
                    d21_idx=int(d21_idx),
                    bg=int(bg),
                    title=title,
                    s=1.2,
                )

                manifest.append({
                    "row_i": int(ri),
                    "tag": tag,
                    "trace_label": _pick_trace_label(meta) if meta else "",
                    "png_all": str(p_all.relative_to(inf_root)),
                    "png_errors": str(p_err.relative_to(inf_root)),
                    "png_d21": str(p_d21.relative_to(inf_root)),
                })

                done_examples += 1

        if len(manifest) > 0:
            with open(inf_root / "inference_manifest.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
                writer.writeheader()
                writer.writerows(manifest)
        else:
            with open(inf_root / "inference_manifest.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["row_i", "tag", "trace_label", "png_all", "png_errors", "png_d21"]
                )
                writer.writeheader()

        save_json(
            {
                "ignored_rows": ignored_rows,
                "n_ignored": len(ignored_rows),
                "n_rendered": int(done_examples),
                "infer_examples_requested": int(args.infer_examples),
                "infer_examples_rendered": int(done_examples),
                "index_csv_used": str(index_csv) if index_csv else "",
            },
            inf_root / "ignored_inference_samples.json",
        )

    # =========================================================
    # RUN META FINAL UPDATE
    # =========================================================
    run_meta_final = dict(run_meta)
    run_meta_final.update({
        "end_time": datetime.now().isoformat(timespec="seconds"),
        "elapsed_sec": float(time.time() - t0),
        "best_epoch": int(best_epoch),
        "best_val_d21_f1": float(best_val_d21_f1),
    })
    save_json(run_meta_final, out_dir / "run_meta.json")

    log_line(f"[done] best_epoch={best_epoch} best_val_d21_f1={best_val_d21_f1:.6f}", run_log)
    log_line(f"[done] tiempo total: {_fmt_hms(time.time() - t0)}", run_log)

# ============================================================
# PLOTS 3D (GT vs Pred / errores / foco d21)
# ============================================================
def _to_np(a) -> np.ndarray:
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    a = np.asarray(a)
    if not a.flags["C_CONTIGUOUS"]:
        a = np.ascontiguousarray(a)
    return a


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

    xyz = _to_np(xyz).astype(np.float32, copy=False)
    y_gt = _to_np(y_gt).astype(np.int32, copy=False).reshape(-1)
    y_pred = _to_np(y_pred).astype(np.int32, copy=False).reshape(-1)

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


def plot_errors(
    xyz: np.ndarray,
    y_gt: np.ndarray,
    y_pred: np.ndarray,
    out_png: Path,
    bg: int,
    title: str,
    s: float = 1.0,
):
    out_png.parent.mkdir(parents=True, exist_ok=True)

    xyz = _to_np(xyz).astype(np.float32, copy=False)
    y_gt = _to_np(y_gt).astype(np.int32, copy=False).reshape(-1)
    y_pred = _to_np(y_pred).astype(np.int32, copy=False).reshape(-1)

    errors = (y_gt != y_pred)

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
    out_png.parent.mkdir(parents=True, exist_ok=True)

    xyz = _to_np(xyz).astype(np.float32, copy=False)
    y_gt = _to_np(y_gt).astype(np.int32, copy=False).reshape(-1)
    y_pred = _to_np(y_pred).astype(np.int32, copy=False).reshape(-1)

    gt_mask = (y_gt == int(d21_idx))
    pred_mask = (y_pred == int(d21_idx))

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

if __name__ == "__main__":
    main()  