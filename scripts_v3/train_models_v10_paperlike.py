#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_models_v10_paperlike.py
--------------------------------------------------------------
Versión académica y modular del flujo de entrenamiento "paper-like" 
para segmentación de dientes (3D point clouds), compatible con tus 
scripts v8/v9 y datasets generados por augmentation_and_split_v3.

Modelos integrados:
  1. PointNet (Qi et al., 2017)
  2. PointNet++ (Qi et al., 2017)
  3. DilatedToothSegNet (bloques dilatados)
  4. Transformer3D (Fourier Positional Encoding)
  5. ToothFormer (versión académica-lite jerárquica por patches)

Características:
  - Normalización robusta de nubes (centro=0, escala unitaria)
  - Control de NaN/Inf y validación de etiquetas fuera de rango
  - Métricas macro + específicas para el diente 21
  - Soporte de class_weights.json y early stopping
  - Logging extendido, guardado de curvas y resumen JSON
  - Compatible con RTX 3090 (entrenamiento ToothFormer ~24 GB VRAM)
--------------------------------------------------------------
Autor: Adaptado por ChatGPT (GPT-5) para flujo de investigación dental
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
        X = torch.tensor(self.X[idx], dtype=torch.float32)
        Y = torch.tensor(self.Y[idx], dtype=torch.long)
        return X, Y


def make_loaders(data_dir: Path, batch_size: int = 8, num_workers: int = 4) -> Dict[str, DataLoader]:
    """
    Genera DataLoaders para los splits train/val/test.
    Requiere archivos: X_train.npz / Y_train.npz / X_val.npz / Y_val.npz / X_test.npz / Y_test.npz
    """
    data_dir = Path(data_dir)
    splits = {"train": "train", "val": "val", "test": "test"}
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
    """
    Calcula matriz de confusión C_ij = #pred=j siendo gt=i
    """
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
    """
    Calcula métricas específicas para el diente 21 (o id indicado).
    """
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
    """
    Retorna índices de los k vecinos más cercanos en 'ref' para cada punto en 'query'.
    query: (B, M, 3)
    ref:   (B, N, 3)
    out:   (B, M, k)
    """
    # Distancias euclidianas batched
    d = torch.cdist(query, ref)  # (B, M, N)
    k = min(k, ref.size(1))
    idx = torch.topk(d, k=k, dim=-1, largest=False).indices  # (B, M, k)
    return idx


def batched_gather(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Selecciona por batch y por índice.
    points: (B, N, C)
    idx:    (B, M, K)
    out:    (B, M, K, C)
    """
    B, N, C = points.shape
    _, M, K = idx.shape
    b = torch.arange(B, device=points.device)[:, None, None].expand(B, M, K)
    out = points[b, idx, :]  # (B, M, K, C)
    return out


# ==============================================================
# === Modelos: PointNet, PointNet++, DilatedToothSegNet, etc. ==
# ==============================================================

# -----------------------------
# PointNet (Qi et al., 2017)
# -----------------------------

class STN3d(nn.Module):
    """
    Spatial Transformer Network 3D:
      Aprende una matriz (3x3) para alinear la nube antes del backbone.
    """
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
        # x: (B, 3, P)
        B = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))            # (B, 1024, P)
        x = torch.max(x, 2)[0]                 # (B, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x).view(B, self.k, self.k)

        # Sesgo a identidad para estabilidad
        iden = torch.eye(self.k, device=x.device).unsqueeze(0).repeat(B, 1, 1)
        return x + iden


class PointNetSeg(nn.Module):
    """
    PointNet para segmentación punto a punto.
    - T-Net de entrada (3x3)
    - Convoluciones 1D + max-pool global
    - Concatenación (global + local) y cabeza de segmentación
    """
    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        super().__init__()
        self.input_tnet = STN3d(k=3)

        self.conv1, self.bn1 = nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64)
        self.conv2, self.bn2 = nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128)
        self.conv3, self.bn3 = nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024)

        # 1024 (global) + 128 (local)
        self.fconv1, self.bn4 = nn.Conv1d(1152, 512, 1), nn.BatchNorm1d(512)
        self.fconv2, self.bn5 = nn.Conv1d(512, 256, 1), nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(dropout)
        self.fconv3 = nn.Conv1d(256, num_classes, 1)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        # xyz: (B, P, 3)
        B, P, _ = xyz.shape
        x = xyz.transpose(2, 1)                # (B, 3, P)
        T = self.input_tnet(x)                 # (B, 3, 3)
        x = torch.bmm(T, x)                    # (B, 3, P)

        x1 = F.relu(self.bn1(self.conv1(x)))   # (B, 64, P)
        x2 = F.relu(self.bn2(self.conv2(x1)))  # (B,128, P)
        x3 = F.relu(self.bn3(self.conv3(x2)))  # (B,1024,P)

        xg = torch.max(x3, 2, keepdim=True)[0].repeat(1, 1, P)  # (B,1024,P)
        x_cat = torch.cat([xg, x2], 1)                          # (B,1152,P)

        x = F.relu(self.bn4(self.fconv1(x_cat)))
        x = F.relu(self.bn5(self.fconv2(x)))
        x = self.dropout(x)
        out = self.fconv3(x).transpose(2, 1)    # (B, P, C)
        return out


# -----------------------------
# PointNet++ (Qi et al., 2017)
#   SA (Set Abstraction) + FP (Feature Propagation)
#   Implementación lite sin libs externas
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


class SA_Layer(nn.Module):
    """
    Set Abstraction:
      - Selecciona centros (submuestreo uniforme determinista para reproducibilidad)
      - kNN en xyz
      - concat xyz_local (+ feats si existen) -> MLP -> max pool
    """
    def __init__(self, nsample: int, in_ch: int, mlp: List[int]):
        super().__init__()
        self.nsample = nsample
        self.mlp = MLP1d(in_ch + 3, mlp)   # añadimos xyz_local
        self.out_ch = mlp[-1]

    def forward(self, xyz: torch.Tensor, feats: Optional[torch.Tensor]):
        """
        xyz:   (B, P, 3)
        feats: (B, C, P) o None
        return:
          xyz_down: (B, M, 3)
          feats_down (B, C', M)
        """
        B, P, _ = xyz.shape
        M = max(1, P // 4)

        # Selección determinista de centros (equispaciado) para reproducibilidad
        idx_center = torch.linspace(0, P - 1, M, device=xyz.device, dtype=torch.long)[None, :].repeat(B, 1)
        centers = torch.gather(xyz, 1, idx_center[..., None].expand(-1, -1, 3))  # (B, M, 3)

        # Vecinos kNN alrededor de cada centro
        idx_knn = knn_indices(centers, xyz, self.nsample)            # (B, M, K)
        neigh_xyz = batched_gather(xyz, idx_knn)                     # (B, M, K, 3)
        local_xyz = (neigh_xyz - centers[:, :, None, :]).permute(0, 3, 1, 2)  # (B, 3, M, K)

        if feats is not None:
            feats_perm = feats.transpose(1, 2).contiguous()          # (B, P, C)
            neigh_f = batched_gather(feats_perm, idx_knn).permute(0, 3, 1, 2)  # (B, C, M, K)
            cat = torch.cat([local_xyz, neigh_f], dim=1)             # (B, 3+C, M, K)
        else:
            cat = local_xyz                                          # (B, 3,   M, K)

        # MLP por patch y max-pool en K
        Bm, Cm, Mm, Km = cat.shape
        cat = cat.reshape(Bm, Cm, Mm * Km)                           # (B, C, M*K)
        # aplicamos conv1d sobre la dimensión "puntos" reordenada (trick simple)
        out = self.mlp(cat)                                          # (B, C', M*K)
        out = out.view(Bm, -1, Mm, Km).max(dim=-1)[0]                # (B, C', M)
        return centers, out


class FP_Layer(nn.Module):
    """
    Feature Propagation:
      Interpola (inverse distance) características de coarse->fine y fusiona con feats previas.
    """
    def __init__(self, in_ch: int, mlp: List[int]):
        super().__init__()
        self.mlp = MLP1d(in_ch, mlp)
        self.out_ch = mlp[-1]

    def forward(self, xyz1, xyz2, feats1, feats2):
        """
        xyz1:   (B, N1, 3)  puntos destino (fine)
        xyz2:   (B, N2, 3)  puntos fuente (coarse)
        feats1: (B, C1, N1) feats destino a concatenar (previas), o None
        feats2: (B, C2, N2) feats fuente a interpolar
        """
        B, N1, _ = xyz1.shape
        _, C2, N2 = feats2.shape

        # k=3 vecinos más cercanos de cada punto destino en los puntos fuente
        idx = knn_indices(xyz1, xyz2, k=min(3, N2))  # (B, N1, 3)
        d = torch.cdist(xyz1, xyz2)                  # (B, N1, N2)
        knn_d = torch.gather(d, 2, idx).clamp_min(1e-8)  # (B, N1, 3)
        w = (1.0 / knn_d); w = w / w.sum(dim=-1, keepdim=True)

        f2p = feats2.transpose(1, 2).contiguous()    # (B, N2, C2)
        neigh = batched_gather(f2p, idx)             # (B, N1, 3, C2)
        out = (w[..., None] * neigh).sum(dim=2).transpose(1, 2).contiguous()  # (B, C2, N1)

        if feats1 is not None:
            out = torch.cat([out, feats1], dim=1)    # (B, C2+C1, N1)

        return self.mlp(out)                         # (B, C', N1)


class PointNet2Seg(nn.Module):
    """PointNet++ para segmentación (lite)."""
    def __init__(self, num_classes: int = 10, nsample: int = 32):
        super().__init__()
        self.sa1 = SA_Layer(nsample=nsample, in_ch=0,    mlp=[64, 128, 256])
        self.sa2 = SA_Layer(nsample=nsample, in_ch=256,  mlp=[256, 512, 512])

        self.fp1 = FP_Layer(in_ch=512 + 256, mlp=[256, 256])  # de sa2->sa1
        self.fp2 = FP_Layer(in_ch=256,       mlp=[256, 128])  # de sa1->xyz original

        self.head = nn.Sequential(
            nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(128, num_classes, 1)
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        # xyz: (B, P, 3)
        feats = None
        xyz1, f1 = self.sa1(xyz, feats)          # (B, M1,3), (B, C1, M1)
        xyz2, f2 = self.sa2(xyz1, f1)            # (B, M2,3), (B, C2, M2)

        f = self.fp1(xyz1, xyz2, f1, f2)         # (B, 256, M1)
        f = self.fp2(xyz, xyz1, None, f)         # (B, 128, P)

        out = self.head(f).transpose(2, 1)       # (B, P, C)
        return out


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
    """
    Backbone dilatado 1D + cabeza de segmentación.
    La entrada se trata como "señal" por punto, con canales 3 (xyz).
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.backbone = nn.Sequential(
            DilatedBlock(3,   64, dilation=1),
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
        # xyz: (B, P, 3)
        x = xyz.transpose(2, 1)              # (B, 3, P)
        f = self.backbone(x)                 # (B, 256, P)
        out = self.head(f).transpose(2, 1)   # (B, P, C)
        return out


# -----------------------------
# Transformer3D (con Fourier PE)
# -----------------------------

class FourierPE(nn.Module):
    """
    Positional Encoding Fourier (sinusoidal sobre xyz escalado).
    num_feats: número de frecuencias por eje.
    """
    def __init__(self, num_feats: int = 32, scale: float = 10.0):
        super().__init__()
        self.num_feats = num_feats
        self.scale = scale

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        # xyz: (B, P, 3)
        x = xyz * self.scale
        k = torch.arange(self.num_feats, device=xyz.device).float()
        freqs = (2.0 ** k)[None, None, :]  # (1,1,F)
        sin = torch.sin(x.unsqueeze(-1) / freqs)  # (B,P,3,F)
        cos = torch.cos(x.unsqueeze(-1) / freqs)  # (B,P,3,F)
        pe = torch.cat([sin, cos], dim=-1).reshape(xyz.size(0), xyz.size(1), -1)  # (B,P, 3*2*F)
        return pe


class Transformer3D(nn.Module):
    """
    Transformer encoder sobre tokens de punto:
      input = [xyz || PE_fourier] -> proyección lineal -> encoder -> cabeza
    """
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
        pe = self.pe(xyz)              # (B,P,PE)
        x = torch.cat([xyz, pe], -1)   # (B,P,in_dim)
        x = self.lin(x)                # (B,P,D)
        x = self.enc(x)                # (B,P,D)
        return self.head(x)            # (B,P,C)


# -----------------------------
# ToothFormer (académico-lite)
#   Jerárquico por patches:
#     - Centros (aprox-FPS equiespaciado) + kNN por centro
#     - PatchEmbed (MLP 2D sobre K vecinos)
#     - Encoder Transformer sobre M tokens de patch
#     - Proyección de tokens a puntos: punto -> centro NN
# -----------------------------

class PatchEmbed(nn.Module):
    """
    Extrae embeddings por patch a partir de vecinos locales:
      Input:  xyz (B,N,3), centers (B,M,3), idx_knn (B,M,K)
      Salida: (B, M, E)  E=emb_dim
    """
    def __init__(self, in_ch: int = 3, emb_dim: int = 256):
        super().__init__()
        # Tratamos la vecindad como "imagen" (canal=in_ch, ancho=K, alto=M)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_ch, 64, 1), nn.ReLU(True),
            nn.Conv2d(64, 128, 1),   nn.ReLU(True),
            nn.Conv2d(128, emb_dim, 1)
        )

    def forward(self, xyz: torch.Tensor, centers: torch.Tensor, idx_knn: torch.Tensor) -> torch.Tensor:
        # Vecinos: (B, M, K, 3)
        neigh = batched_gather(xyz, idx_knn)
        # Coordenadas locales respecto del centro
        local = neigh - centers[:, :, None, :]          # (B, M, K, 3)
        x = local.permute(0, 3, 1, 2).contiguous()      # (B, 3, M, K)
        f = self.mlp(x)                                 # (B, E, M, K)
        f = torch.max(f, dim=-1, keepdim=False)[0]      # (B, E, M)
        f = f.permute(0, 2, 1).contiguous()             # (B, M, E)
        return f


class LearnablePE(nn.Module):
    """Positional Encoding aprendible por patch."""
    def __init__(self, dim: int, max_patches: int = 256):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_patches, dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, M, D)
        M = x.size(1)
        return x + self.pe[:, :M, :]


class ToothFormer(nn.Module):
    """
    ToothFormer académico-lite:
      - Selección de M centros (equispaciado determinista, aprox-FPS)
      - kNN (K vecinos) para cada centro
      - PatchEmbed -> PE aprendible -> TransformerEncoder
      - Proyección de tokens a puntos vía mapeo punto->centro NN
    """
    def __init__(self, num_classes: int = 10, emb_dim: int = 256,
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

        # Proyección a puntos
        self.proj_lin = nn.Linear(emb_dim, emb_dim)
        self.head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(emb_dim, num_classes)
        )

    @torch.no_grad()
    def _choose_centers(self, xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Selección determinista de centros (M índices equiespaciados).
        Retorna centros (B,M,3) e índices (B,M).
        """
        B, N, _ = xyz.shape
        idx = torch.linspace(0, N - 1, steps=self.num_patches, device=xyz.device).long()
        idx = idx.unsqueeze(0).repeat(B, 1)  # (B, M)
        centers = torch.gather(xyz, 1, idx[..., None].expand(-1, -1, 3))  # (B, M, 3)
        return centers, idx

    @torch.no_grad()
    def _knn_per_center(self, centers: torch.Tensor, xyz: torch.Tensor, k: int) -> torch.Tensor:
        """kNN de cada centro sobre los N puntos."""
        return knn_indices(centers, xyz, k=k)  # (B, M, K)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        # xyz: (B, N, 3)
        B, N, _ = xyz.shape

        # 1) centros y vecinos
        centers, _ = self._choose_centers(xyz)              # (B, M, 3)
        idx_knn = self._knn_per_center(centers, xyz, self.k)  # (B, M, K)

        # 2) patch embeddings + PE + encoder
        tokens = self.patch_embed(xyz, centers, idx_knn)    # (B, M, E)
        tokens = self.pos(tokens)                           # (B, M, E)
        tokens = self.encoder(tokens)                       # (B, M, E)

        # 3) punto -> centro más cercano (para proyectar token correspondiente)
        with torch.no_grad():
            idx_pc = knn_indices(xyz, centers, k=1).squeeze(-1)  # (B, N)

        b = torch.arange(B, device=xyz.device)[:, None].expand(B, N)
        picked = tokens[b, idx_pc, :]                       # (B, N, E)
        feats = self.proj_lin(picked)                       # (B, N, E)
        out = self.head(feats)                              # (B, N, C)
        return out

# ==============================================================
# === Fábrica de modelos =======================================
# ==============================================================

def build_model(name: str, num_classes: int) -> nn.Module:
    """
    Devuelve la instancia del modelo solicitado.
    """
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
    """
    Monitorea la pérdida de validación y detiene el entrenamiento
    si no mejora tras 'patience' épocas consecutivas.
    """
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
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch},
                    self.ckpt_dir / "best.pt"
                )
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
    """
    Lee un archivo JSON con pesos por clase (dict o lista).
    """
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

def train_one_epoch(model: nn.Module, loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    criterion, device: torch.device,
                    num_classes: int, d21_id: int) -> Tuple[float, Dict[str, float]]:
    """
    Entrena el modelo durante una época completa.
    Devuelve pérdida media y métricas macro + d21.
    """
    model.train()
    loss_sum = 0.0
    cm = torch.zeros(num_classes, num_classes, device=device)
    d21 = {"d21_acc": 0.0, "d21_f1": 0.0, "d21_iou": 0.0}
    batches = 0

    for X, Y in loader:
        X, Y = to_device((X, Y), device)
        X = sanitize_tensor(normalize_cloud(sanitize_tensor(X)))

        logits = sanitize_tensor(model(X))          # (B,P,C)
        Y = torch.clamp(Y, 0, num_classes - 1)

        loss = criterion(logits.transpose(2, 1), Y)  # CrossEntropy espera (B,C,P)
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
def evaluate(model: nn.Module, loader: DataLoader, criterion,
             device: torch.device, num_classes: int, d21_id: int) -> Tuple[float, Dict[str, float]]:
    """
    Evalúa el modelo sin actualizar parámetros.
    """
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
# === Gestión de historiales y logging =========================
# ==============================================================

def update_history(history: Dict[str, List[float]], prefix: str, stats: Dict[str, float]):
    """Agrega métricas al historial global (dict de listas)."""
    for k, v in stats.items():
        history.setdefault(f"{prefix}_{k}", []).append(float(v))


def print_epoch(ep: int, epochs: int, tr_loss: float, va_loss: float,
                va_stats: Dict[str, float]) -> None:
    """
    Muestra un resumen de métricas por época (estilo paper).
    """
    msg = (
        f"[Ep {ep:03d}/{epochs}] "
        f"tr={tr_loss:.4f}  va={va_loss:.4f}  "
        f"acc={va_stats['acc']:.3f}  f1={va_stats['f1']:.3f}  "
        f"iou={va_stats['iou']:.3f}  d21_f1={va_stats['d21_f1']:.3f}"
    )
    print(msg)

# ==============================================================
# === Utilidades de guardado (CSV/JSON) ========================
# ==============================================================

def save_history_csv(history: Dict[str, List[float]], out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    # Transponer dict de listas a filas por época
    # Detectar número de épocas por alguna clave
    any_key = next(iter(history.keys()))
    T = len(history[any_key]) if any_key else 0
    keys = sorted(history.keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch"] + keys)
        for ep in range(T):
            row = [ep + 1] + [history[k][ep] if ep < len(history[k]) else "" for k in keys]
            w.writerow(row)


def save_run_summary(run_dir: Path, args: Dict[str, Any], num_classes: int,
                     n_params: int, best_val: float, last_epoch: int,
                     te_loss: float, te_stats: Dict[str, float], history: Dict[str, List[float]]):
    run_dir.mkdir(parents=True, exist_ok=True)
    # JSON
    save_json({
        "args": args,
        "num_classes": num_classes,
        "params": int(n_params),
        "best_val_loss": float(best_val),
        "last_epoch": int(last_epoch),
        "test": te_stats,
        "history_keys": list(history.keys())
    }, run_dir / "summary.json")
    # CSV
    save_history_csv(history, run_dir / "history.csv")


# ==============================================================
# === MAIN ======================================================
# ==============================================================

def main():
    parser = argparse.ArgumentParser(description="Entrenamiento paper-like v10 con ToothFormer")
    # Paths
    parser.add_argument("--data_dir", required=True, help="Carpeta con X_*.npz / Y_*.npz (+ artifacts/)")
    parser.add_argument("--out_dir", required=True, help="Carpeta base para runs (se crea subcarpeta por timestamp/modelo)")
    # Modelo
    parser.add_argument("--model", default="toothformer",
                        choices=["pointnet", "pointnetpp", "dilatedtoothsegnet", "transformer3d", "toothformer"])
    # Hparams
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    # Loss / labels
    parser.add_argument("--ignore_index", type=int, default=None, help="ID a ignorar en la pérdida (p.ej. fondo)")
    parser.add_argument("--d21_id", type=int, default=21, help="ID (remapeado) del diente 21 para métricas específicas")
    # Scheduler
    parser.add_argument("--use_cosine", action="store_true", help="Usar CosineAnnealingLR (T_max=epochs)")
    # Eval
    parser.add_argument("--eval_best", action="store_true", help="Evaluar test con el mejor checkpoint (best.pt)")
    args = parser.parse_args()

    # Semilla y dispositivo
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    # DataLoaders
    loaders = make_loaders(Path(args.data_dir), batch_size=args.batch_size, num_workers=args.num_workers)

    # Inferir num_classes
    Ytr = np.load(Path(args.data_dir) / "Y_train.npz")["Y"]
    num_classes = int(np.max(Ytr)) + 1
    print(f"[INFO] num_classes={num_classes}")

    # Modelo
    model = build_model(args.model, num_classes=num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] {args.model}  params={n_params/1e6:.2f}M")

    # Pérdida con pesos de clase (si existen)
    artifacts = Path(args.data_dir) / "artifacts"
    class_w = load_class_weights(artifacts, num_classes)
    if class_w is not None:
        class_w = class_w.to(device)
        print("[INFO] Usando class_weights.json")

    if args.ignore_index is not None:
        criterion = nn.CrossEntropyLoss(weight=class_w, ignore_index=args.ignore_index)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_w)

    # Optimizador y scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
                 if args.use_cosine else None)

    # Salida
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_dir) / f"{args.model}_{stamp}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    print(f"[RUN] {run_dir}")

    # Early stopping
    stopper = EarlyStopping(patience=args.patience, delta=1e-4, ckpt_dir=run_dir / "checkpoints")

    # Historial
    history: Dict[str, List[float]] = {}
    best_val = float("inf")
    last_epoch = 0

    # Entrenamiento
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_stats = train_one_epoch(model, loaders["train"], optimizer, criterion,
                                            device, num_classes, args.d21_id)
        va_loss, va_stats = evaluate(model, loaders["val"], criterion,
                                     device, num_classes, args.d21_id)

        if scheduler is not None:
            scheduler.step()

        update_history(history, "train", {"loss": tr_loss, **tr_stats})
        update_history(history, "val",   {"loss": va_loss, **va_stats})
        print_epoch(ep, args.epochs, tr_loss, va_loss, va_stats)

        improved = stopper(va_loss, model, ep)
        if improved:
            best_val = va_loss
        if stopper.early_stop:
            print("[EARLY] Parada temprana activada.")
            last_epoch = ep
            break

        last_epoch = ep

    # Guardar último checkpoint
    torch.save({"model": model.state_dict(), "epoch": last_epoch}, run_dir / "checkpoints" / "final_model.pt")

    # Curvas e historial
    plot_curves(history, run_dir, args.model)
    save_run_summary(run_dir, vars(args), num_classes, n_params, best_val, last_epoch,
                     te_loss=float("nan"), te_stats={}, history=history)

    # Evaluación en test (best o final)
    ckpt_to_eval = run_dir / "checkpoints" / ("best.pt" if args.eval_best and (run_dir / "checkpoints" / "best.pt").exists() else "final_model.pt")
    print(f"[EVAL] Cargando {ckpt_to_eval.name} para evaluación en test.")
    state = torch.load(ckpt_to_eval, map_location=device)
    model.load_state_dict(state["model"])

    te_loss, te_stats = evaluate(model, loaders["test"], criterion, device, num_classes, args.d21_id)
    print(f"[TEST] loss={te_loss:.4f}  acc={te_stats['acc']:.3f}  f1={te_stats['f1']:.3f}  iou={te_stats['iou']:.3f}  d21_f1={te_stats['d21_f1']:.3f}")

    # Actualizar summary con test
    save_run_summary(run_dir, vars(args), num_classes, n_params, best_val, last_epoch,
                     te_loss=te_loss, te_stats=te_stats, history=history)

    print(f"[DONE] Resultados guardados en: {run_dir}")


if __name__ == "__main__":
    main()

