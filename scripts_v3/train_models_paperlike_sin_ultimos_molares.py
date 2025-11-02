#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_models_paperlike_sin_ultimos_molares.py
--------------------------------------------------------------
Versión académica del flujo de entrenamiento "paper-like" 
para segmentación dental 3D (PointNet, PointNet++, ToothFormer, etc.)
adaptada a datasets sin terceros molares (26 dientes + fondo).

Características:
  - Compatible con datasets generados por augmentation_and_split_v3.
  - num_classes = 27 (26 dientes + 1 fondo).
  - Ignora dientes 18, 28, 38, 48.
  - Compatible con RTX 3090 (~24 GB VRAM).
  - Entrenamiento reproducible, con early stopping, curvas y logging.

Autor: ChatGPT (GPT-5)
--------------------------------------------------------------
"""

# ==============================================================
# === Importaciones generales =================================
# ==============================================================

import os, sys, json, csv, time, random, gc
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==============================================================
# === Configuración general y utilidades =======================
# ==============================================================

def set_seed(seed: int = 42):
    """Establece semillas globales para reproducibilidad total."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_device(x, device: torch.device):
    """Envía tensores o listas de tensores al dispositivo especificado."""
    if isinstance(x, (tuple, list)):
        return [to_device(t, device) for t in x]
    return x.to(device, non_blocking=True)


def sanitize_tensor(t: torch.Tensor) -> torch.Tensor:
    """Reemplaza NaN/Inf en tensores para evitar pérdidas inválidas."""
    return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)


def normalize_cloud(x: torch.Tensor) -> torch.Tensor:
    """
    Normaliza una nube de puntos a esfera unitaria:
    centro en (0,0,0) y radio máximo 1.
    """
    c = x.mean(dim=1, keepdim=True)
    x = x - c
    r = (x.pow(2).sum(-1).sqrt()).max(dim=1, keepdim=True)[0].unsqueeze(-1)
    return x / (r + 1e-8)


def save_json(obj: Any, path: Path):
    """Guarda un objeto Python como JSON en la ruta indicada."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def plot_curves(history: Dict[str, List[float]], out_dir: Path, model_name: str):
    """Genera gráficos de pérdida y métricas por split (train/val)."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    keys = ["loss", "acc", "f1", "iou", "d21_acc", "d21_f1", "d21_iou"]
    for k in keys:
        plt.figure(figsize=(7, 4))
        for split in ["train", "val"]:
            kk = f"{split}_{k}"
            if kk in history and len(history[kk]) > 0:
                plt.plot(history[kk], label=split)
        plt.xlabel("Época"); plt.ylabel(k.upper())
        plt.title(f"{model_name} – {k.upper()}")
        plt.legend(); plt.tight_layout()
        plt.savefig(out_dir / f"{model_name}_{k}.png", dpi=300)
        plt.close()


# ==============================================================
# === Dataset y DataLoader =====================================
# ==============================================================

class CloudDataset(Dataset):
    """
    Dataset de nubes de puntos:
      - Carga X e Y desde archivos .npz con claves 'X' y 'Y'
      - Devuelve tensores (B,P,3) y (B,P)
    """
    def __init__(self, X_path: Path, Y_path: Path):
        self.X = np.load(X_path)["X"].astype(np.float32)
        self.Y = np.load(Y_path)["Y"].astype(np.int64)
        assert self.X.shape[0] == self.Y.shape[0], \
            f"Dimensión inconsistente entre {X_path.name} y {Y_path.name}"

    def __len__(self): return self.X.shape[0]

    def __getitem__(self, idx):
        X = torch.from_numpy(self.X[idx])
        Y = torch.from_numpy(self.Y[idx])
        return X, Y


def make_loaders(data_dir: Path, batch_size: int = 8, num_workers: int = 4) -> Dict[str, DataLoader]:
    """
    Genera DataLoaders para los splits train/val/test.
    Requiere archivos: X_train.npz / Y_train.npz / X_val.npz / Y_val.npz / X_test.npz / Y_test.npz
    """
    data_dir = Path(data_dir)
    splits = ["train", "val", "test"]
    loaders = {}
    for split in splits:
        Xp = data_dir / f"X_{split}.npz"
        Yp = data_dir / f"Y_{split}.npz"
        if not (Xp.exists() and Yp.exists()):
            raise FileNotFoundError(f"Faltan archivos para split {split}: {Xp}, {Yp}")
        ds = CloudDataset(Xp, Yp)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
    return loaders

# ==============================================================
# === Métricas y matrices de confusión =========================
# ==============================================================

@torch.no_grad()
def confusion_matrix(logits: torch.Tensor, y_true: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Calcula matriz de confusión C_ij = #pred=j siendo gt=i."""
    preds = logits.argmax(dim=-1)
    t = y_true.view(-1)
    p = preds.view(-1)
    valid = (t >= 0) & (t < num_classes)
    t = t[valid]; p = p[valid]
    idx = t * num_classes + p
    cm = torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return cm


def macro_from_cm(cm: torch.Tensor) -> Dict[str, float]:
    """Extrae accuracy, precision, recall, F1 e IoU promedio macro."""
    cm = cm.float()
    tp = torch.diag(cm)
    gt = cm.sum(1).clamp_min(1e-8)
    pd = cm.sum(0).clamp_min(1e-8)
    acc = (tp.sum() / cm.sum().clamp_min(1e-8)).item()
    prec = torch.mean(tp / pd).item()
    rec = torch.mean(tp / gt).item()
    f1 = torch.mean(2 * tp / (gt + pd)).item()
    iou = torch.mean(tp / (gt + pd - tp).clamp_min(1e-8)).item()
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "iou": iou}


def d21_metrics(logits: torch.Tensor, y_true: torch.Tensor, cls_id: int) -> Dict[str, float]:
    """Calcula métricas específicas para el diente de interés (por defecto 11)."""
    preds = logits.argmax(dim=-1).view(-1)
    t = y_true.view(-1)
    tp = ((preds == cls_id) & (t == cls_id)).sum().float()
    fp = ((preds == cls_id) & (t != cls_id)).sum().float()
    fn = ((preds != cls_id) & (t == cls_id)).sum().float()
    tn = ((preds != cls_id) & (t != cls_id)).sum().float()

    acc = ((tp + tn) / (tp + tn + fp + fn + 1e-8)).item()
    prec = (tp / (tp + fp + 1e-8)).item()
    rec = (tp / (tp + fn + 1e-8)).item()
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    iou = (tp / (tp + fp + fn + 1e-8)).item()
    return {"d21_acc": acc, "d21_f1": f1, "d21_iou": iou}


# ==============================================================
# === Utilidades geométricas (kNN, gather batched) =============
# ==============================================================

def knn_indices(query: torch.Tensor, ref: torch.Tensor, k: int) -> torch.Tensor:
    """Retorna índices de los k vecinos más cercanos en 'ref' para cada punto en 'query'."""
    d = torch.cdist(query, ref)  # (B, M, N)
    k = min(k, ref.size(1))
    idx = torch.topk(d, k=k, dim=-1, largest=False).indices
    return idx


def batched_gather(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Selecciona por batch y por índice (gather extendido)."""
    B, N, C = points.shape
    _, M, K = idx.shape
    b = torch.arange(B, device=points.device)[:, None, None].expand(B, M, K)
    out = points[b, idx, :]
    return out


# ==============================================================
# === Modelos: PointNet, PointNet++, DilatedToothSegNet, etc. ==
# ==============================================================

# -----------------------------
# PointNet (Qi et al., 2017)
# -----------------------------

class STN3d(nn.Module):
    """Spatial Transformer Network 3D (T-Net)."""
    def __init__(self, k: int = 3):
        super().__init__()
        self.k = k
        self.conv1, self.bn1 = nn.Conv1d(k, 64, 1), nn.BatchNorm1d(64)
        self.conv2, self.bn2 = nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128)
        self.conv3, self.bn3 = nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024)
        self.fc1, self.bn4 = nn.Linear(1024, 512), nn.BatchNorm1d(512)
        self.fc2, self.bn5 = nn.Linear(512, 256), nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, k * k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2)[0]
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x).view(B, self.k, self.k)
        iden = torch.eye(self.k, device=x.device).unsqueeze(0).repeat(B, 1, 1)
        return x + iden


class PointNetSeg(nn.Module):
    """PointNet para segmentación punto a punto."""
    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        super().__init__()
        self.input_tnet = STN3d(k=3)
        self.conv1, self.bn1 = nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64)
        self.conv2, self.bn2 = nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128)
        self.conv3, self.bn3 = nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024)
        self.fconv1, self.bn4 = nn.Conv1d(1152, 512, 1), nn.BatchNorm1d(512)
        self.fconv2, self.bn5 = nn.Conv1d(512, 256, 1), nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(dropout)
        self.fconv3 = nn.Conv1d(256, num_classes, 1)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        B, P, _ = xyz.shape
        x = xyz.transpose(2, 1)
        T = self.input_tnet(x)
        x = torch.bmm(T, x)
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x3 = F.relu(self.bn3(self.conv3(x2)))
        xg = torch.max(x3, 2, keepdim=True)[0].repeat(1, 1, P)
        x_cat = torch.cat([xg, x2], 1)
        x = F.relu(self.bn4(self.fconv1(x_cat)))
        x = F.relu(self.bn5(self.fconv2(x)))
        x = self.dropout(x)
        out = self.fconv3(x).transpose(2, 1)
        return out


# -----------------------------
# PointNet++ (Qi et al., 2017)
# -----------------------------

class MLP1d(nn.Module):
    """Bloque MLP sobre (B, C, N) usando conv1d + BN + ReLU."""
    def __init__(self, in_ch: int, mlp: List[int]):
        super().__init__()
        layers = []
        c = in_ch
        for oc in mlp:
            layers += [nn.Conv1d(c, oc, 1), nn.BatchNorm1d(oc), nn.ReLU(True)]
            c = oc
        self.net = nn.Sequential(*layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# -----------------------------
# Set Abstraction y Feature Propagation (PointNet++)
# -----------------------------

class SA_Layer(nn.Module):
    """Set Abstraction Layer (agrupación jerárquica)."""
    def __init__(self, nsample: int, in_ch: int, mlp: List[int]):
        super().__init__()
        self.nsample = nsample
        self.mlp = MLP1d(in_ch + 3, mlp)
        self.out_ch = mlp[-1]

    def forward(self, xyz: torch.Tensor, feats: Optional[torch.Tensor]):
        B, P, _ = xyz.shape
        M = max(1, P // 4)
        idx_center = torch.linspace(0, P - 1, M, device=xyz.device, dtype=torch.long)[None, :].repeat(B, 1)
        centers = torch.gather(xyz, 1, idx_center[..., None].expand(-1, -1, 3))
        idx_knn = knn_indices(centers, xyz, self.nsample)
        neigh_xyz = batched_gather(xyz, idx_knn)
        local_xyz = (neigh_xyz - centers[:, :, None, :]).permute(0, 3, 1, 2)
        if feats is not None:
            feats_perm = feats.transpose(1, 2).contiguous()
            neigh_f = batched_gather(feats_perm, idx_knn).permute(0, 3, 1, 2)
            cat = torch.cat([local_xyz, neigh_f], dim=1)
        else:
            cat = local_xyz
        Bm, Cm, Mm, Km = cat.shape
        cat = cat.reshape(Bm, Cm, Mm * Km)
        out = self.mlp(cat)
        out = out.view(Bm, -1, Mm, Km).max(dim=-1)[0]
        return centers, out


class FP_Layer(nn.Module):
    """Feature Propagation (interpola coarse→fine y fusiona con features previas)."""
    def __init__(self, in_ch: int, mlp: List[int]):
        super().__init__()
        self.mlp = MLP1d(in_ch, mlp)
        self.out_ch = mlp[-1]

    def forward(self, xyz1, xyz2, feats1, feats2):
        B, N1, _ = xyz1.shape
        _, C2, N2 = feats2.shape
        idx = knn_indices(xyz1, xyz2, k=min(3, N2))
        d = torch.cdist(xyz1, xyz2)
        knn_d = torch.gather(d, 2, idx).clamp_min(1e-8)
        w = (1.0 / knn_d); w = w / w.sum(dim=-1, keepdim=True)
        f2p = feats2.transpose(1, 2).contiguous()
        neigh = batched_gather(f2p, idx)
        out = (w[..., None] * neigh).sum(dim=2).transpose(1, 2).contiguous()
        if feats1 is not None:
            out = torch.cat([out, feats1], dim=1)
        return self.mlp(out)


class PointNet2Seg(nn.Module):
    """PointNet++ simplificado para segmentación punto a punto."""
    def __init__(self, num_classes: int = 10, nsample: int = 32):
        super().__init__()
        self.sa1 = SA_Layer(nsample=nsample, in_ch=0, mlp=[64, 128, 256])
        self.sa2 = SA_Layer(nsample=nsample, in_ch=256, mlp=[256, 512, 512])
        self.fp1 = FP_Layer(in_ch=512 + 256, mlp=[256, 256])
        self.fp2 = FP_Layer(in_ch=256, mlp=[256, 128])
        self.head = nn.Sequential(
            nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(128, num_classes, 1)
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        feats = None
        xyz1, f1 = self.sa1(xyz, feats)
        xyz2, f2 = self.sa2(xyz1, f1)
        f = self.fp1(xyz1, xyz2, f1, f2)
        f = self.fp2(xyz, xyz1, None, f)
        return self.head(f).transpose(2, 1)


# -----------------------------
# DilatedToothSegNet (bloques dilatados 1D)
# -----------------------------

class DilatedBlock(nn.Module):
    """Bloque dilatado simple (conv1d dilatada + BN + ReLU) x2."""
    def __init__(self, in_ch: int, out_ch: int, dilation: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, dilation=dilation),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(True),
            nn.Conv1d(out_ch, out_ch, 1, dilation=dilation),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(True),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DilatedToothSegNet(nn.Module):
    """Backbone dilatado 1D + cabeza de segmentación."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.backbone = nn.Sequential(
            DilatedBlock(3, 64, dilation=1),
            DilatedBlock(64, 128, dilation=2),
            DilatedBlock(128, 256, dilation=4),
            DilatedBlock(256, 256, dilation=1),
        )
        self.head = nn.Sequential(
            nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(128, num_classes, 1),
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        x = xyz.transpose(2, 1)
        f = self.backbone(x)
        return self.head(f).transpose(2, 1)


# -----------------------------
# Transformer3D (con Fourier Positional Encoding)
# -----------------------------

class FourierPE(nn.Module):
    """Positional Encoding Fourier (sinusoidal sobre xyz escalado)."""
    def __init__(self, num_feats: int = 32, scale: float = 10.0):
        super().__init__()
        self.num_feats = num_feats
        self.scale = scale
    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        x = xyz * self.scale
        k = torch.arange(self.num_feats, device=xyz.device).float()
        freqs = (2.0 ** k)[None, None, :]
        sin = torch.sin(x.unsqueeze(-1) / freqs)
        cos = torch.cos(x.unsqueeze(-1) / freqs)
        pe = torch.cat([sin, cos], dim=-1).reshape(xyz.size(0), xyz.size(1), -1)
        return pe


class Transformer3D(nn.Module):
    """Transformer encoder sobre tokens de punto."""
    def __init__(self, num_classes: int = 10, d_model: int = 128, nhead: int = 4,
                 depth: int = 4, dim_ff: int = 256):
        super().__init__()
        self.pe = FourierPE(num_feats=d_model // 6)
        in_dim = 3 + (3 * 2 * (d_model // 6))
        self.lin = nn.Linear(in_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(d_model, num_classes)
        )
    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        pe = self.pe(xyz)
        x = torch.cat([xyz, pe], -1)
        x = self.lin(x)
        x = self.enc(x)
        return self.head(x)


# -----------------------------
# ToothFormer (académico-lite)
# -----------------------------

class PatchEmbed(nn.Module):
    """Extrae embeddings por patch local (MLP sobre vecinos kNN)."""
    def __init__(self, in_ch: int = 3, emb_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_ch, 64, 1), nn.ReLU(True),
            nn.Conv2d(64, 128, 1), nn.ReLU(True),
            nn.Conv2d(128, emb_dim, 1)
        )

    def forward(self, xyz: torch.Tensor, centers: torch.Tensor, idx_knn: torch.Tensor) -> torch.Tensor:
        neigh = batched_gather(xyz, idx_knn)
        local = neigh - centers[:, :, None, :]
        x = local.permute(0, 3, 1, 2).contiguous()
        f = self.mlp(x)
        f = torch.max(f, dim=-1)[0]
        return f.permute(0, 2, 1).contiguous()


class LearnablePE(nn.Module):
    """Positional Encoding aprendible por patch."""
    def __init__(self, dim: int, max_patches: int = 256):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_patches, dim) * 0.02)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        M = x.size(1)
        return x + self.pe[:, :M, :]


class ToothFormer(nn.Module):
    """ToothFormer jerárquico-lite (para 26 dientes)."""
    def __init__(self, num_classes: int = 27, emb_dim: int = 256,
                 nhead: int = 8, depth: int = 6, dim_ff: int = 512,
                 num_patches: int = 64, k_per_patch: int = 128):
        super().__init__()
        self.num_patches = num_patches
        self.k = k_per_patch
        self.patch_embed = PatchEmbed(in_ch=3, emb_dim=emb_dim)
        self.pos = LearnablePE(dim=emb_dim, max_patches=num_patches)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=nhead, dim_feedforward=dim_ff,
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.proj_lin = nn.Linear(emb_dim, emb_dim)
        self.head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(emb_dim, num_classes)
        )

    @torch.no_grad()
    def _choose_centers(self, xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = xyz.shape
        idx = torch.linspace(0, N - 1, steps=self.num_patches, device=xyz.device).long()
        idx = idx.unsqueeze(0).repeat(B, 1)
        centers = torch.gather(xyz, 1, idx[..., None].expand(-1, -1, 3))
        return centers, idx

    @torch.no_grad()
    def _knn_per_center(self, centers: torch.Tensor, xyz: torch.Tensor, k: int) -> torch.Tensor:
        return knn_indices(centers, xyz, k=k)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        centers, _ = self._choose_centers(xyz)
        idx_knn = self._knn_per_center(centers, xyz, self.k)
        tokens = self.patch_embed(xyz, centers, idx_knn)
        tokens = self.pos(tokens)
        tokens = self.encoder(tokens)
        idx_pc = knn_indices(xyz, centers, k=1).squeeze(-1)
        b = torch.arange(xyz.size(0), device=xyz.device)[:, None].expand_as(idx_pc)
        feats = tokens[b, idx_pc, :]
        feats = self.proj_lin(feats)
        return self.head(feats)


# ==============================================================
# === Fábrica de modelos =======================================
# ==============================================================

def build_model(name: str, num_classes: int) -> nn.Module:
    """Devuelve la instancia del modelo solicitado."""
    n = name.lower()
    if n == "pointnet":
        return PointNetSeg(num_classes=num_classes)
    elif n == "pointnetpp":
        return PointNet2Seg(num_classes=num_classes)
    elif n == "dilatedtoothsegnet":
        return DilatedToothSegNet(num_classes=num_classes)
    elif n == "transformer3d":
        return Transformer3D(num_classes=num_classes,
                             d_model=128, nhead=4, depth=4, dim_ff=256)
    elif n == "toothformer":
        return ToothFormer(num_classes=num_classes, emb_dim=256, nhead=8,
                           depth=6, dim_ff=512, num_patches=64, k_per_patch=128)
    else:
        raise ValueError(f"Modelo no reconocido: {name}")

# ==============================================================
# === Early Stopping y gestión de checkpoints ==================
# ==============================================================

class EarlyStopping:
    """Detiene el entrenamiento si la pérdida de validación no mejora tras 'patience' épocas."""
    def __init__(self, patience: int = 20, delta: float = 1e-4, ckpt_dir: Optional[Path] = None):
        self.patience = patience
        self.delta = delta
        self.ckpt_dir = Path(ckpt_dir) if ckpt_dir else None
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss: float, model: nn.Module, epoch: int) -> bool:
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.ckpt_dir:
                self.ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save({"model": model.state_dict(), "epoch": epoch},
                           self.ckpt_dir / "best.pt")
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


# ==============================================================
# === Carga de pesos de clase =================================
# ==============================================================

def load_class_weights(artifacts_dir: Path, num_classes: int) -> Optional[torch.Tensor]:
    """Lee un archivo JSON con pesos por clase (dict o lista)."""
    f = artifacts_dir / "class_weights.json"
    if not f.exists():
        return None
    try:
        w = json.loads(f.read_text())
        if isinstance(w, dict):
            arr = np.zeros((num_classes,), dtype=np.float32)
            for k, v in w.items():
                idx = int(k)
                if 0 <= idx < num_classes:
                    arr[idx] = float(v)
        else:
            arr = np.array(w, dtype=np.float32)
        if arr.shape[0] != num_classes:
            raise ValueError("Tamaño de class_weights.json no coincide con num_classes")
        return torch.tensor(arr, dtype=torch.float32)
    except Exception as e:
        print(f"[WARN] Error cargando class_weights.json: {e}")
        return None


# ==============================================================
# === Bucle de entrenamiento y evaluación ======================
# ==============================================================

def train_one_epoch(model, loader, optimizer, criterion, device, num_classes, d21_id):
    model.train()
    loss_sum = 0.0
    cm = torch.zeros(num_classes, num_classes, device=device)
    d21 = {"d21_acc": 0.0, "d21_f1": 0.0, "d21_iou": 0.0}
    batches = 0

    for X, Y in loader:
        X, Y = to_device((X, Y), device)
        X = sanitize_tensor(normalize_cloud(sanitize_tensor(X)))
        logits = sanitize_tensor(model(X))
        Y = torch.clamp(Y, 0, num_classes - 1)
        loss = criterion(logits.transpose(2, 1), Y)
        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad(set_to_none=True)
            continue
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_sum += float(loss.detach().cpu())
        cm += confusion_matrix(logits, Y, num_classes)
        d21m = d21_metrics(logits, Y, d21_id)
        for k in d21: d21[k] += d21m[k]
        batches += 1

    macro = macro_from_cm(cm)
    for k in d21: d21[k] /= max(1, batches)
    return loss_sum / max(1, batches), {**macro, **d21}


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes, d21_id):
    model.eval()
    loss_sum = 0.0
    cm = torch.zeros(num_classes, num_classes, device=device)
    d21 = {"d21_acc": 0.0, "d21_f1": 0.0, "d21_iou": 0.0}
    batches = 0

    for X, Y in loader:
        X, Y = to_device((X, Y), device)
        X = sanitize_tensor(normalize_cloud(sanitize_tensor(X)))
        logits = sanitize_tensor(model(X))
        Y = torch.clamp(Y, 0, num_classes - 1)
        loss = criterion(logits.transpose(2, 1), Y)
        loss_sum += float(loss.detach().cpu())
        cm += confusion_matrix(logits, Y, num_classes)
        d21m = d21_metrics(logits, Y, d21_id)
        for k in d21: d21[k] += d21m[k]
        batches += 1

    macro = macro_from_cm(cm)
    for k in d21: d21[k] /= max(1, batches)
    return loss_sum / max(1, batches), {**macro, **d21}


# ==============================================================
# === Logging y guardado ======================================
# ==============================================================

def update_history(history, prefix, stats):
    for k, v in stats.items():
        history.setdefault(f"{prefix}_{k}", []).append(float(v))


def print_epoch(ep, epochs, tr_loss, va_loss, va_stats):
    msg = (
        f"[Ep {ep:03d}/{epochs}] "
        f"tr={tr_loss:.4f}  va={va_loss:.4f}  "
        f"acc={va_stats['acc']:.3f}  f1={va_stats['f1']:.3f}  "
        f"iou={va_stats['iou']:.3f}  d21_f1={va_stats['d21_f1']:.3f}"
    )
    print(msg)


def save_history_csv(history, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    any_key = next(iter(history.keys()))
    T = len(history[any_key]) if any_key else 0
    keys = sorted(history.keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch"] + keys)
        for ep in range(T):
            row = [ep + 1] + [history[k][ep] if ep < len(history[k]) else "" for k in keys]
            w.writerow(row)


def save_run_summary(run_dir, args, num_classes, n_params, best_val, last_epoch, te_loss, te_stats, history):
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json({
        "args": args,
        "num_classes": num_classes,
        "params": int(n_params),
        "best_val_loss": float(best_val),
        "last_epoch": int(last_epoch),
        "test": te_stats,
        "history_keys": list(history.keys())
    }, run_dir / "summary.json")
    save_history_csv(history, run_dir / "history.csv")


# ==============================================================
# === MAIN =====================================================
# ==============================================================

def main():
    parser = argparse.ArgumentParser(description="Entrenamiento sin terceros molares (26 dientes + fondo)")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model", default="toothformer",
                        choices=["pointnet", "pointnetpp", "dilatedtoothsegnet", "transformer3d", "toothformer"])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--ignore_index", type=int, default=0, help="ID del fondo")
    parser.add_argument("--d21_id", type=int, default=11, help="ID remapeado del diente 11")
    parser.add_argument("--eval_best", action="store_true")
    args = parser.parse_args()

    # Semillas y dispositivo
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    # DataLoaders
    loaders = make_loaders(Path(args.data_dir), batch_size=args.batch_size, num_workers=args.num_workers)

    # Inferir clases (limitadas a 27: 26 dientes + fondo)
    Ytr = np.load(Path(args.data_dir) / "Y_train.npz")["Y"]
    num_classes = min(int(np.max(Ytr)) + 1, 27)
    print(f"[INFO] num_classes={num_classes} (sin terceros molares)")

    # Modelo
    model = build_model(args.model, num_classes=num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] {args.model}  params={n_params/1e6:.2f}M")

    # Pérdida
    artifacts = Path(args.data_dir) / "artifacts"
    class_w = load_class_weights(artifacts, num_classes)
    if class_w is not None:
        class_w = class_w.to(device)
        print("[INFO] Usando class_weights.json")

    criterion = nn.CrossEntropyLoss(weight=class_w, ignore_index=args.ignore_index)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Directorio de salida
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_dir) / f"{args.model}_{stamp}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    print(f"[RUN] {run_dir}")

    stopper = EarlyStopping(patience=args.patience, delta=1e-4, ckpt_dir=run_dir / "checkpoints")

    history = {}
    best_val, last_epoch = float("inf"), 0

    for ep in range(1, args.epochs + 1):
        tr_loss, tr_stats = train_one_epoch(model, loaders["train"], optimizer, criterion, device, num_classes, args.d21_id)
        va_loss, va_stats = evaluate(model, loaders["val"], criterion, device, num_classes, args.d21_id)
        update_history(history, "train", {"loss": tr_loss, **tr_stats})
        update_history(history, "val", {"loss": va_loss, **va_stats})
        print_epoch(ep, args.epochs, tr_loss, va_loss, va_stats)
        improved = stopper(va_loss, model, ep)
        if improved:
            best_val = va_loss
        if stopper.early_stop:
            print("[EARLY] Parada temprana activada.")
            last_epoch = ep
            break
        last_epoch = ep

    torch.save({"model": model.state_dict(), "epoch": last_epoch}, run_dir / "checkpoints" / "final_model.pt")
    plot_curves(history, run_dir, args.model)

    ckpt = run_dir / "checkpoints" / ("best.pt" if args.eval_best and (run_dir / "checkpoints" / "best.pt").exists() else "final_model.pt")
    print(f"[EVAL] Cargando {ckpt.name} para evaluación en test.")
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"])

    te_loss, te_stats = evaluate(model, loaders["test"], criterion, device, num_classes, args.d21_id)
    print(f"[TEST] loss={te_loss:.4f}  acc={te_stats['acc']:.3f}  f1={te_stats['f1']:.3f}  iou={te_stats['iou']:.3f}  d21_f1={te_stats['d21_f1']:.3f}")

    save_run_summary(run_dir, vars(args), num_classes, n_params, best_val, last_epoch, te_loss, te_stats, history)
    print(f"[DONE] Resultados guardados en: {run_dir}")


if __name__ == "__main__":
    main()
