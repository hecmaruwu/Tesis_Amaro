#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_models_v12_paperlike.py
--------------------------------------------------------------
Versión completa y estable (v12) del framework paper-like para
segmentación 3D de dientes mediante nubes de puntos.

Incluye:
  - PointNet
  - PointNet++ normal (Qi et al. 2017)
  - PointNet++ mejorado (SPFE + WSLFA + FP jerárquico)
  - DilatedToothSegNet
  - Transformer3D (Fourier PE)
  - ToothFormer (Académico-lite por patches)

Características clave:
  * Soporta datasets con XYZ o XYZ+features geométricas (normales,
    curvatura, extras). in_ch se detecta automáticamente.
  * Normalización robusta por nube (esfera unitaria).
  * Entrenamiento con CE + Dice, EarlyStopping, AdamW, Cosine/StepLR.
  * Métricas macro reales e independientes del desbalance.
  * Métrica especial por clase de enfoque (focus_id, p.ej. d21).
  * Guardado estilo paper (best.pt, final_model.pt, history, summary).
  * Visualización 3D de GT/predicción con Plotly.

Autor: Adaptado por ChatGPT (GPT-5)
--------------------------------------------------------------
"""

# ==============================================================
# === IMPORTS GENERALES ========================================
# ==============================================================

import os
import sys
import time
import json
import csv
import math
import random
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
# === SEMILLAS Y UTILIDADES ====================================
# ==============================================================

def set_seed(seed: int = 42):
    """
    Fija semillas para reproducibilidad total en CPU/GPU.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_device(x, device: torch.device):
    """
    Envía tensores o estructuras de tensores al dispositivo.
    """
    if isinstance(x, (tuple, list)):
        return [to_device(t, device) for t in x]
    return x.to(device, non_blocking=True)


def sanitize_tensor(t: torch.Tensor) -> torch.Tensor:
    """
    Reemplaza NaN/Inf por 0 (seguridad numérica).
    """
    return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)


def normalize_cloud(x: torch.Tensor) -> torch.Tensor:
    """
    Normaliza cada nube individualmente:
      - centra en (0,0,0)
      - escala a radio máximo = 1
    Solo normaliza XYZ (primeras 3 componentes). Las features
    adicionales se mantienen sin normalizar.
    """
    B, P, C = x.shape
    xyz = x[:, :, :3]
    feats = x[:, :, 3:] if C > 3 else None

    center = xyz.mean(dim=1, keepdim=True)
    xyz = xyz - center

    radius = (xyz.pow(2).sum(-1).sqrt()).max(dim=1, keepdim=True)[0]
    xyz = xyz / (radius.unsqueeze(-1) + 1e-8)

    if feats is not None:
        x = torch.cat([xyz, feats], dim=-1)
    else:
        x = xyz

    return x


def save_json(obj: Any, path: Path):
    """
    Guarda un diccionario en JSON con indentación.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_history_csv(history: Dict[str, List[float]], out_csv: Path):
    """
    Guarda el history completo en formato CSV (épocas × métricas).
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    keys = sorted(history.keys())
    T = len(history[keys[0]])

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch"] + keys)
        for ep in range(T):
            row = [ep + 1] + [history[k][ep] if ep < len(history[k]) else "" for k in keys]
            w.writerow(row)


def plot_curves(history: Dict[str, List[float]], out_dir: Path, model_name: str):
    """
    Genera gráficas PNG para loss, acc, f1, iou, precision, recall,
    d_focus_f1. Guarda un PNG por métrica.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = ["loss", "acc", "prec", "rec", "f1", "iou", "d_focus_f1"]

    for k in keys:
        plt.figure(figsize=(7, 4))
        for split in ["train", "val"]:
            kk = f"{split}_{k}"
            if kk in history and len(history[kk]) > 0:
                plt.plot(history[kk], label=split)
        plt.xlabel("Época")
        plt.ylabel(k.upper())
        plt.title(f"{model_name} – {k.upper()}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{model_name}_{k}.png", dpi=300)
        plt.close()


def update_history(history: Dict[str, List[float]], prefix: str,
                   stats: Dict[str, float], loss_value: Optional[float] = None):
    """
    Inserta métricas en el diccionario de history.
    """
    if loss_value is not None:
        history.setdefault(f"{prefix}_loss", []).append(float(loss_value))

    for k in ["acc", "prec", "rec", "f1", "iou",
              "d_focus_acc", "d_focus_prec", "d_focus_rec",
              "d_focus_f1", "d_focus_iou"]:
        if k in stats:
            history.setdefault(f"{prefix}_{k}", []).append(float(stats[k]))


# ==============================================================
# === DATASET Y DATA LOADERS ===================================
# ==============================================================

class CloudDataset(Dataset):
    """
    Dataset básico para X/Y en formato .npz
      X: (M, P, C)
      Y: (M, P)
    """
    def __init__(self, X_path: Path, Y_path: Path):
        X_np = np.load(X_path)["X"]
        Y_np = np.load(Y_path)["Y"]

        assert X_np.shape[0] == Y_np.shape[0], (
            f"Dim mismatch entre {X_path} y {Y_path}"
        )

        self.X = torch.from_numpy(X_np.astype(np.float32))
        self.Y = torch.from_numpy(Y_np.astype(np.int64))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def make_loaders(data_dir: Path, batch_size: int = 8, num_workers: int = 6):
    """
    Crea DataLoaders para train/val/test basados en:
      X_train.npz, Y_train.npz
      X_val.npz,   Y_val.npz
      X_test.npz,  Y_test.npz
    """
    data_dir = Path(data_dir)
    loaders = {}

    for split in ["train", "val", "test"]:
        Xp = data_dir / f"X_{split}.npz"
        Yp = data_dir / f"Y_{split}.npz"

        if not Xp.exists() or not Yp.exists():
            raise FileNotFoundError(f"Faltan archivos para split {split}: {Xp}, {Yp}")

        ds = CloudDataset(Xp, Yp)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )

    return loaders

# ==============================================================
# === MÉTRICAS (CONFUSION MATRIX, MACRO, POR-CLASE) ============
# ==============================================================

@torch.no_grad()
def confusion_matrix_from_logits(logits: torch.Tensor,
                                 y_true: torch.Tensor,
                                 num_classes: int) -> torch.Tensor:
    """
    Construye la matriz de confusión multi-clase.
    logits: (B,N,C)
    y_true: (B,N)
    retorna: (C,C) con conteos enteros
    """
    preds = logits.argmax(dim=-1)    # (B,N)
    t = y_true.reshape(-1)
    p = preds.reshape(-1)

    valid = (t >= 0) & (t < num_classes)
    t, p = t[valid], p[valid]

    idx = t * num_classes + p
    cm = torch.bincount(idx, minlength=num_classes ** 2)
    return cm.reshape(num_classes, num_classes)


def macro_stats_from_cm(cm: torch.Tensor) -> Dict[str, float]:
    """
    Calcula:
      - macro accuracy
      - macro precision
      - macro recall
      - macro f1
      - macro IoU
    Y también retorna vectores por clase para el summary.
    """
    cm = cm.float()
    tp = torch.diag(cm)
    gt = cm.sum(1).clamp_min(1e-8)  # ground truth por clase
    pd = cm.sum(0).clamp_min(1e-8)  # predicciones por clase
    tot = cm.sum().clamp_min(1e-8)

    acc = (tp.sum() / tot).item()

    precision_c = (tp / pd).cpu().numpy()
    recall_c    = (tp / gt).cpu().numpy()
    f1_c        = (2 * tp / (pd + gt)).cpu().numpy()
    iou_c       = (tp / (pd + gt - tp).clamp_min(1e-8)).cpu().numpy()

    return {
        "acc": float(acc),
        "precision": float(np.nanmean(precision_c)),
        "recall": float(np.nanmean(recall_c)),
        "f1": float(np.nanmean(f1_c)),
        "iou": float(np.nanmean(iou_c)),
        "_per_class": {
            "precision": precision_c.tolist(),
            "recall":    recall_c.tolist(),
            "f1":        f1_c.tolist(),
            "iou":       iou_c.tolist(),
            "support":   cm.sum(1).cpu().numpy().tolist()
        }
    }


def single_class_stats(logits: torch.Tensor,
                       y_true: torch.Tensor,
                       cls_id: int) -> Dict[str, float]:
    """
    Métricas binarizando solo una clase (focus_id, p.ej. d21).
    Retorna: d_focus_precision, d_focus_recall, d_focus_f1, d_focus_iou
    """
    preds = logits.argmax(dim=-1).reshape(-1)
    t = y_true.reshape(-1)

    tp = ((preds == cls_id) & (t == cls_id)).sum().float()
    fp = ((preds == cls_id) & (t != cls_id)).sum().float()
    fn = ((preds != cls_id) & (t == cls_id)).sum().float()
    tn = ((preds != cls_id) & (t != cls_id)).sum().float()

    acc  = ((tp + tn) / (tp + tn + fp + fn + 1e-8)).item()
    prec = (tp / (tp + fp + 1e-8)).item()
    rec  = (tp / (tp + fn + 1e-8)).item()
    f1   = 2 * prec * rec / (prec + rec + 1e-8)
    iou  = (tp / (tp + fp + fn + 1e-8)).item()

    return {
        "d_focus_acc": acc,
        "d_focus_prec": prec,
        "d_focus_rec": rec,
        "d_focus_f1": f1,
        "d_focus_iou": iou
    }


# ==============================================================
# === CLASS WEIGHTS (LECTURA / AUTO-CÁLCULO) ===================
# ==============================================================

def _try_load_class_weights(artifacts_dir: Path,
                            num_classes: int) -> Optional[torch.Tensor]:
    """
    Intenta cargar artifacts/class_weights.json
    Formatos aceptados:
        - [w0, w1, ..., wC]
        - { "0": w0, "1": w1, ... }
        - { "class_weights": {...} }
    """
    f = artifacts_dir / "class_weights.json"
    if not f.exists():
        return None

    try:
        obj = json.loads(f.read_text())

        # Formato { "class_weights": {...} }
        if isinstance(obj, dict) and "class_weights" in obj:
            obj = obj["class_weights"]

        # Formato dict por claves string
        if isinstance(obj, dict):
            arr = np.zeros((num_classes,), dtype=np.float32)
            for k, v in obj.items():
                i = int(k)
                if 0 <= i < num_classes:
                    arr[i] = float(v)
            return torch.tensor(arr, dtype=torch.float32)

        # Formato lista
        arr = np.array(obj, dtype=np.float32)
        if arr.shape[0] != num_classes:
            print("[WARN] class_weights.json con tamaño incorrecto, ignorando.")
            return None
        return torch.tensor(arr, dtype=torch.float32)

    except Exception as e:
        print(f"[WARN] No se pudo leer class_weights.json: {e}")
        return None


def _auto_class_weights_from_train(Ytr_path: Path,
                                   num_classes: int,
                                   clip_min: float = 0.2,
                                   clip_max: float = 5.0) -> torch.Tensor:
    """
    Esquema paper-like:
        w_c = 1 / log(1.2 + freq)
    Normalizado a media=1 y recortado a [clip_min, clip_max].
    """
    Y = np.load(Ytr_path)["Y"]
    freqs = np.bincount(Y.ravel(), minlength=num_classes).astype(np.float64)
    freqs = np.maximum(freqs, 1.0)

    inv_log = 1.0 / np.log(1.2 + freqs)
    inv_log /= inv_log.mean()
    inv_log = np.clip(inv_log, clip_min, clip_max)

    return torch.tensor(inv_log, dtype=torch.float32)

# ==============================================================
# === LOSS: CROSS ENTROPY + DICE =================================
# ==============================================================

class CombinedLoss(nn.Module):
    """
    Loss combinada:
      L = ce_weight * CrossEntropy + dice_weight * DiceLoss

    - class_weights: tensor [C] opcional (para desbalance)
    - ignore_index = -1 para masking
    """
    def __init__(self, num_classes: int,
                 ce_weight: float = 1.0,
                 dice_weight: float = 1.0,
                 class_weights=None,
                 ignore_index: int = -1):
        super().__init__()
        self.num_classes = num_classes
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)
        self.ce = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index
        )

    def forward(self, logits, y_true):
        """
        logits: (B,N,C)
        y_true: (B,N)
        """
        ce = self.ce(logits.transpose(1, 2), y_true)
        if self.dice_weight <= 0:
            return ce

        probs = F.softmax(logits, dim=-1)  # (B,N,C)
        y_onehot = F.one_hot(y_true, num_classes=self.num_classes).float()

        inter = (probs * y_onehot).sum(dim=(1, 2))
        union = probs.sum(dim=(1, 2)) + y_onehot.sum(dim=(1, 2))
        dice = 1.0 - (2 * inter + 1e-5) / (union + 1e-5)
        dice = dice.mean()

        return self.ce_weight * ce + self.dice_weight * dice


# ==============================================================
# === HELPERS COMPARTIDOS (MLP simple 1D) =======================
# ==============================================================

class MLP1d(nn.Module):
    """
    MLP de convoluciones 1×1 con BatchNorm y ReLU.
    Entrada: (B,C_in,N)
    """
    def __init__(self, in_ch, mlp):
        super().__init__()
        layers = []
        c = in_ch
        for oc in mlp:
            layers += [
                nn.Conv1d(c, oc, 1),
                nn.BatchNorm1d(oc),
                nn.ReLU(True)
            ]
            c = oc
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ==============================================================
# === T-NET 3D (PointNet original) =============================
# ==============================================================

class STN3d(nn.Module):
    """
    Spatial Transformer Network para alinear XYZ.
    Produce una matriz 3×3 por nube.
    """
    def __init__(self, k=3):
        super().__init__()
        self.k = k

        self.conv1 = nn.Conv1d(k, 64, 1)
        self.bn1   = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2   = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3   = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256, k * k)

    def forward(self, x):  # x: (B,3,N)
        B = x.size(0)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))              # (B,1024,N)
        x = torch.max(x, 2)[0]                   # (B,1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x).view(B, self.k, self.k)

        # Identidad para evitar colapsos
        iden = torch.eye(self.k, device=x.device).unsqueeze(0).repeat(B, 1, 1)
        return x + iden


# ==============================================================
# === POINTNET COMPLETO (SEGMENTACIÓN) =========================
# ==============================================================

class PointNetSeg(nn.Module):
    """
    PointNet original (Qi et al., 2017) para segmentación.

    - Acepta XYZ + features adicionales.
    - Aplica T-Net solo a XYZ (primeras 3 dims).
    - Arquitectura clásica con skip global/local.
    """
    def __init__(self,
                 num_classes: int,
                 dropout: float = 0.5,
                 in_ch: int = 3):
        super().__init__()

        self.in_ch = in_ch
        self.use_tnet = True
        self.tnet = STN3d(k=3)

        # Primer bloque convolucional recibe in_ch dinámico
        self.conv1 = nn.Conv1d(in_ch, 64, 1)
        self.bn1   = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2   = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3   = nn.BatchNorm1d(1024)

        # Local + Global concatenado → 128 + 1024 = 1152
        self.fconv1 = nn.Conv1d(1152, 512, 1)
        self.bn4    = nn.BatchNorm1d(512)

        self.fconv2 = nn.Conv1d(512, 256, 1)
        self.bn5    = nn.BatchNorm1d(256)

        self.dropout = nn.Dropout(dropout)
        self.fconv3  = nn.Conv1d(256, num_classes, 1)

    def forward(self, pts):  # pts: (B,N,C)
        B, N, C = pts.shape

        # Formato esperado por conv1d → (B,C,N)
        x = pts.transpose(2, 1)

        # ===== T-Net solo a XYZ =====
        if self.use_tnet:
            xyz = x[:, :3, :]                   # (B,3,N)
            T = self.tnet(xyz)                  # (B,3,3)
            xyz = torch.bmm(T, xyz)             # (B,3,N)

            if C > 3:  # volver a concatenar extras sin transformar
                x = torch.cat([xyz, x[:, 3:, :]], dim=1)
            else:
                x = xyz

        # ===== Bloques convolucionales =====
        x1 = F.relu(self.bn1(self.conv1(x)))          # (B,64,N)
        x2 = F.relu(self.bn2(self.conv2(x1)))         # (B,128,N)
        x3 = F.relu(self.bn3(self.conv3(x2)))         # (B,1024,N)

        # Global max pooling (shape expandido)
        xg = torch.max(x3, 2, keepdim=True)[0]        # (B,1024,1)
        xg = xg.repeat(1, 1, N)                       # (B,1024,N)

        # Concatenación global + local
        x_cat = torch.cat([xg, x2], dim=1)            # (B,1152,N)

        x = F.relu(self.bn4(self.fconv1(x_cat)))
        x = F.relu(self.bn5(self.fconv2(x)))
        x = self.dropout(x)
        x = self.fconv3(x)                            # (B,C_classes,N)

        return x.transpose(2, 1)                      # (B,N,C_classes)

# ==============================================================
# === POINTNET++ NORMAL (Qi et al., 2017) =======================
# ==============================================================

"""
Esta sección implementa PointNet++ en su versión clásica:

  - 2 niveles de Set Abstraction (SA)
  - 2 niveles de Feature Propagation (FP)

Compatible con:
    * XYZ solamente → in_ch = 3
    * XYZ + normales/curvatura → in_ch = 6 o más

El modelo detecta automáticamente cuántos canales tiene el input.
"""


# --------------------------------------------------------------
# FPS simple (uniforme, sin distancias) — estable y rápido
# --------------------------------------------------------------
@torch.no_grad()
def fps_uniform(xyz: torch.Tensor, M: int):
    """
    FPS extremadamente simple pero estable:
    Selecciona M índices equiespaciados en el rango [0, N-1].
    """
    B, N, _ = xyz.shape
    idx = torch.linspace(0, N - 1, steps=M, device=xyz.device).long()
    return idx.unsqueeze(0).repeat(B, 1)  # (B,M)


# --------------------------------------------------------------
# SA Layer clásico (PointNet++ original)
# --------------------------------------------------------------

class SA_Layer_PN2(nn.Module):
    """
    Set Abstraction clásico:
        - FPS
        - KNN
        - MLP (1×1 conv)
        - MaxPool sobre vecinos

    xyz:     (B,N,3)
    feat_in: (B,C_in,N) o None
    """

    def __init__(self, nsample: int, in_ch: int, mlp: List[int]):
        super().__init__()
        self.nsample = nsample

        layers = []
        last_ch = in_ch + 3  # coords relativas + feats
        for oc in mlp:
            layers.append(nn.Conv2d(last_ch, oc, 1))
            layers.append(nn.BatchNorm2d(oc))
            layers.append(nn.ReLU(True))
            last_ch = oc
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz, feat_in):
        """
        xyz:     (B,N,3)
        feat_in: (B,C_in,N) o None
        """

        B, N, _ = xyz.shape
        C_in = 0 if feat_in is None else feat_in.shape[1]

        # -------- FPS --------
        M = max(1, N // 4)  # normalmente 1/4 del total
        idx_center = fps_uniform(xyz, M)  # (B,M)
        centers_xyz = torch.gather(
            xyz, 1, idx_center[..., None].expand(-1, -1, 3)
        )  # (B,M,3)

        # -------- KNN de tamaño nsample --------
        d = torch.cdist(centers_xyz, xyz)  # (B,M,N)
        knn_idx = d.topk(k=min(self.nsample, N), dim=-1, largest=False)[1]  # (B,M,K)

        # -------- Coordenadas relativas --------------------------------
        neigh_xyz = torch.gather(
            xyz[:, None, :, :].expand(-1, M, -1, -1),
            2, knn_idx[..., None].expand(-1, -1, -1, 3)
        )  # (B,M,K,3)

        local_xyz = neigh_xyz - centers_xyz[:, :, None, :]  # (B,M,K,3)

        # -------- Features de los vecinos -------------------------------
        if feat_in is None:
            feat_T = torch.zeros((B, N, 0), device=xyz.device)
        else:
            feat_T = feat_in.transpose(2, 1)  # (B,N,C_in)

        neigh_f = torch.gather(
            feat_T[:, None, :, :].expand(-1, M, -1, -1),
            2, knn_idx[..., None].expand(-1, -1, -1, C_in)
        )  # (B,M,K,C_in)

        # -------- Concatenación coords+feats ----------------------------
        cat = torch.cat([local_xyz, neigh_f], dim=-1)  # (B,M,K,3+C_in)
        cat = cat.permute(0, 3, 1, 2).contiguous()      # (B,3+C_in,M,K)

        # -------- MLP y max pool ---------------------------------------
        f = self.mlp(cat)               # (B,mlp_last,M,K)
        f = torch.max(f, dim=-1)[0]     # (B,mlp_last,M)

        return centers_xyz, f  # XYZ reducido, features agregadas


# --------------------------------------------------------------
# FP Layer (Feature Propagation) clásico
# --------------------------------------------------------------

class FP_Layer_PN2(nn.Module):
    """
    Interpola features desde puntos "altos" (menos denso) hacia "bajos".

    xyz_low:  (B,N_low,3)
    xyz_high: (B,N_high,3)
    feat_low: (B,C_low,N_low) o None
    feat_high:(B,C_high,N_high)

    Resultado: (B,out_ch,N_low)
    """

    def __init__(self, in_ch, mlp: List[int]):
        super().__init__()
        layers = []
        last = in_ch
        for oc in mlp:
            layers.append(nn.Conv1d(last, oc, 1))
            layers.append(nn.BatchNorm1d(oc))
            layers.append(nn.ReLU(True))
            last = oc
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz_low, xyz_high, feat_low, feat_high):
        B, Nl, _ = xyz_low.shape
        Nh = xyz_high.shape[1]

        # Distancias (B,Nl,Nh)
        d = torch.cdist(xyz_low, xyz_high)
        idx = d.topk(k=min(3, Nh), dim=-1, largest=False)[1]     # (B,Nl,3)

        dist = torch.gather(d, 2, idx).clamp_min(1e-8)          # (B,Nl,3)
        w = (1.0 / dist)
        w = w / w.sum(dim=-1, keepdim=True)                     # (B,Nl,3)

        # Extrae features de high resol
        fT = feat_high.transpose(2, 1)                          # (B,Nh,C_high)
        neigh = torch.gather(
            fT, 1,
            idx[..., None].expand(-1, -1, -1, fT.shape[-1])
        )  # (B,Nl,3,C_high)

        f_interp = (w[..., None] * neigh).sum(dim=2)            # (B,Nl,C_high)
        f_interp = f_interp.transpose(2, 1)                     # (B,C_high,Nl)

        if feat_low is not None:
            f_cat = torch.cat([f_interp, feat_low], dim=1)      # (B,C_high+C_low,Nl)
        else:
            f_cat = f_interp

        return self.mlp(f_cat)  # (B,out_ch,Nl)


# --------------------------------------------------------------
# POINTNET++ NORMAL — MODELO COMPLETO
# --------------------------------------------------------------

class PointNet2Seg(nn.Module):
    """
    PointNet++ normal clásico (v12)
    --------------------------------
    Estructura:
        SA1 → SA2 → FP1 → FP2 → Head

    in_ch se detecta dinámicamente:
      - XYZ solamente: in_ch=3
      - XYZ+normales: in_ch=6
      - XYZ+extras:   in_ch=C
    """

    def __init__(self,
                 num_classes: int,
                 nsample: int = 32,
                 in_ch: int = 3,
                 dropout: float = 0.5):
        super().__init__()

        # SA1 usa in_ch dinámico
        self.sa1 = SA_Layer_PN2(
            nsample=nsample,
            in_ch=in_ch,
            mlp=[64, 128, 256]
        )

        # SA2 recibe salida=256 → + coords relativas
        self.sa2 = SA_Layer_PN2(
            nsample=nsample,
            in_ch=256,
            mlp=[256, 512, 512]
        )

        # Decoder
        self.fp1 = FP_Layer_PN2(in_ch=512 + 256, mlp=[256, 256])
        self.fp2 = FP_Layer_PN2(in_ch=256,       mlp=[256, 128])

        # Head final
        self.head = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Conv1d(128, num_classes, 1)
        )

    def forward(self, pts):  # pts: (B,N,C)
        xyz = pts[:, :, :3]

        # Si hay features extra → (B,N,C_extras) → (B,C_extras,N)
        feats0 = None
        if pts.shape[2] > 3:
            feats0 = pts[:, :, 3:].transpose(1, 2)

        # ====== Encoder ======
        xyz1, f1 = self.sa1(xyz, feats0)     # (B,M1,3), (B,256,M1)
        xyz2, f2 = self.sa2(xyz1, f1)        # (B,M2,3), (B,512,M2)

        # ====== Decoder ======
        f_up1 = self.fp1(xyz1, xyz2, f1, f2)  # (B,256,M1)
        f_up0 = self.fp2(xyz, xyz1, None, f_up1)  # (B,128,N)

        out = self.head(f_up0).transpose(2, 1)  # (B,N,C)
        return out

# ==============================================================
# === POINTNET++ MEJORADO (SPFE + WSLFA + FP FULL) =============
# ==============================================================

"""
Esta es la versión más robusta y potente del framework:

 - SPFE: extracción preliminar por punto (XYZ, XYZ centrado,
         normales, extras).
 - SA_WSLFA: agregación espacial con pesos aprendidos (similar a
   convolution sobre patch pero sin malla).
 - 3 niveles jerárquicos (M1, M2, M3)
 - 3 Feature Propagation (FP3 → FP2 → FP1)
 - Compatible con cualquier número de canales (detectado en runtime)

Este es el modelo recomendado para su tesis.
"""


# --------------------------------------------------------------
# SPFE: Single-Point Preliminary Feature Extraction
# --------------------------------------------------------------

class SPFE(nn.Module):
    """
    SPFE toma como entrada:
        concat([xyz, xyz_centrado, normales?, extras?])
    y produce 64 channels por punto.

    La MLP se construye dinámicamente cuando se ve la dimensionalidad real.
    """

    def __init__(self, in_ch_spfe=None, out_ch=64):
        super().__init__()
        self.mlp = None
        self.out_ch = out_ch

    def forward(self, xyz_all):
        B, N, C = xyz_all.shape

        xyz = xyz_all[:, :, :3]
        xyz_c = xyz - xyz.mean(dim=1, keepdim=True)

        normals = torch.zeros_like(xyz)
        extras = None

        if C >= 6:
            normals = xyz_all[:, :, 3:6]
            if C > 6:
                extras = xyz_all[:, :, 6:]
        elif C > 3:
            extras = xyz_all[:, :, 3:]

        parts = [xyz, xyz_c, normals]
        if extras is not None and extras.shape[-1] > 0:
            parts.append(extras)
        else:
            pass

        spfe_in = torch.cat(parts, dim=-1)  # (B,N,D)
        D = spfe_in.shape[-1]

        spfe_in = spfe_in.transpose(2, 1).contiguous()  # (B,D,N)

        # Crear MLP si no existe
        if self.mlp is None or list(self.mlp[0].weight.shape)[1] != D:
            self.mlp = nn.Sequential(
                nn.Conv1d(D, self.out_ch, 1),
                nn.BatchNorm1d(self.out_ch),
                nn.ReLU(True),
                nn.Conv1d(self.out_ch, self.out_ch, 1),
                nn.BatchNorm1d(self.out_ch),
                nn.ReLU(True)
            ).to(xyz_all.device)

        return self.mlp(spfe_in)  # (B,64,N)


# --------------------------------------------------------------
# SA_WSLFA — Weighted Summation via Learnable Feature Aggregation
# --------------------------------------------------------------

class SA_WSLFA(nn.Module):
    """
    SA con WSLFA:
        - FPS dinámico (M puntos)
        - KNN
        - MLP para features
        - MLP para alphas (pesos)
        - Suma ponderada → feature por centro

    Se construye dinámicamente al ver la dimensionalidad real.
    """

    def __init__(self, n_center, k_neighbors, in_ch, mlp_out):
        super().__init__()
        self.n_center = n_center
        self.k = k_neighbors
        self.mlp_out = mlp_out
        self.out_ch = mlp_out

        # Flags internas (para construcción dinámica)
        self._mlp_feat_ok = False
        self._mlp_alpha_ok = False

    # Construcción dinámica de la MLP que procesa coords+feats
    def _build_mlp_feat(self, D_cat, device):
        self.mlp_feat = nn.Sequential(
            nn.Conv2d(D_cat, self.out_ch, 1),
            nn.BatchNorm2d(self.out_ch),
            nn.ReLU(True)
        ).to(device)
        self._mlp_feat_ok = True

    # Construcción dinámica de la MLP que procesa diferencias para alpha
    def _build_mlp_alpha(self, D_alpha, device):
        self.mlp_alpha = nn.Sequential(
            nn.Conv2d(D_alpha, self.out_ch, 1),
            nn.BatchNorm2d(self.out_ch),
            nn.ReLU(True)
        ).to(device)
        self._mlp_alpha_ok = True

    def forward(self, xyz, feat_in):
        """
        xyz:     (B,N,3)
        feat_in: (B,C_in,N)
        """
        B, N, _ = xyz.shape
        C_in = feat_in.shape[1]
        M = self.n_center

        # -------------------- FPS --------------------
        idx_center = fps_uniform(xyz, M)
        centers_xyz = torch.gather(
            xyz, 1,
            idx_center[..., None].expand(-1, -1, 3)
        )  # (B,M,3)

        # -------------------- KNN --------------------
        d = torch.cdist(centers_xyz, xyz)   # (B,M,N)
        idx_knn = d.topk(k=min(self.k, N), dim=-1, largest=False)[1]  # (B,M,K)

        # coords de vecinos
        neigh_xyz = torch.gather(
            xyz[:, None, :, :].expand(-1, M, -1, -1),
            2,
            idx_knn[..., None].expand(-1, -1, -1, 3)
        )  # (B,M,K,3)

        local_xyz = neigh_xyz - centers_xyz[:, :, None, :]  # (B,M,K,3)

        # -------------------- feats vecinos --------------------
        feat_T = feat_in.transpose(2, 1)  # (B,N,C_in)
        neigh_f = torch.gather(
            feat_T[:, None, :, :].expand(-1, M, -1, -1),
            2,
            idx_knn[..., None].expand(-1, -1, -1, C_in)
        )  # (B,M,K,C_in)

        # concat coords+feats
        cat = torch.cat([local_xyz, neigh_f], dim=-1)  # (B,M,K,3+C_in)
        cat = cat.permute(0, 3, 1, 2).contiguous()      # (B,3+C_in,M,K)

        D_cat = 3 + C_in

        # -------------------- construir mlp_feat --------------------
        if (not self._mlp_feat_ok) or (self.mlp_feat[0].weight.shape[1] != D_cat):
            self._build_mlp_feat(D_cat, xyz.device)

        f_prime = self.mlp_feat(cat)            # (B,out_ch,M,K)
        f_mean = f_prime.mean(dim=-1, keepdim=True)

        # Input para alpha
        alpha_in = torch.cat([cat, f_prime - f_mean], dim=1)
        D_alpha = D_cat + self.out_ch

        # -------------------- construir mlp_alpha --------------------
        if (not self._mlp_alpha_ok) or (self.mlp_alpha[0].weight.shape[1] != D_alpha):
            self._build_mlp_alpha(D_alpha, xyz.device)

        alpha = self.mlp_alpha(alpha_in)        # (B,out_ch,M,K)
        w = torch.softmax(alpha, dim=-1)

        # Agregación por suma ponderada
        f_region = (w * f_prime).sum(dim=-1)    # (B,out_ch,M)

        return centers_xyz, f_region


# --------------------------------------------------------------
# FP Layer jerárquico (3-NN + concatenación)
# --------------------------------------------------------------

class FP_Layer(nn.Module):
    """
    Feature Propagation (FP) de PointNet++ — versión corregida y estable.
    Interpola características desde xyz_high -> xyz_low usando 3-NN.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(True)
        )

    def forward(self, xyz_low, xyz_high, feat_low, feat_high):
        """
        xyz_low:  (B, Nl, 3)
        xyz_high: (B, Nh, 3)
        feat_low: (B, Cl, Nl)  o None
        feat_high:(B, Ch, Nh)
        """
        B, Nl, _ = xyz_low.shape
        Nh = xyz_high.shape[1]
        Ch = feat_high.shape[1]

        # ---------------- 1) Distancias + indices 3-NN ----------------
        dist = torch.cdist(xyz_low, xyz_high)                   # (B, Nl, Nh)
        idx = dist.topk(k=min(3, Nh), dim=-1, largest=False)[1] # (B, Nl, 3)

        # ---------------- 2) Ponderaciones ----------------
        d3 = torch.gather(dist, 2, idx).clamp_min(1e-8)         # (B, Nl, 3)
        w = 1.0 / d3
        w = w / w.sum(dim=-1, keepdim=True)                     # (B, Nl, 3)

        # ---------------- 3) Gather features correctos ----------------
        # feat_high: (B, Ch, Nh) -> (B, Nh, Ch)
        fh = feat_high.transpose(2, 1)                          # (B, Nh, Ch)

        # Expandir idx → (B, Nl, 3, Ch)
        idx_exp = idx.unsqueeze(-1).expand(-1, -1, -1, Ch)

        # ---------------- ESTA ES LA LÍNEA CORREGIDA ----------------
        neigh = torch.gather(fh.unsqueeze(1).expand(B, Nl, Nh, Ch), 2, idx_exp)
        # neigh: (B, Nl, 3, Ch)

        # ---------------- 4) Interpolación ----------------
        f_interp = (w.unsqueeze(-1) * neigh).sum(dim=2)         # (B, Nl, Ch)
        f_interp = f_interp.transpose(2, 1)                     # (B, Ch, Nl)

        # ---------------- 5) Concatenación ----------------
        if feat_low is not None:
            f_cat = torch.cat([f_interp, feat_low], dim=1)      # (B, Ch+Cl, Nl)
        else:
            f_cat = f_interp

        # ---------------- 6) MLP final ----------------
        return self.mlp(f_cat)                                  # (B, out_ch, Nl)



# --------------------------------------------------------------
# POINTNET++ SPFE + WSLFA — MODELO COMPLETO
# --------------------------------------------------------------

class PointNet2Seg_SPFE_WSLFA(nn.Module):
    """
    PointNet++ mejorado v12:
        SPFE → SA1 → SA2 → SA3 → FP3 → FP2 → FP1 → Head
    """
    def __init__(self,
                 num_classes: int,
                 in_ch: int = 3,
                 k: int = 32,
                 M1_frac: float = 1/4,
                 M2_frac: float = 1/8,
                 M3_frac: float = 1/16,
                 dropout: float = 0.5):
        super().__init__()

        self.spfe = None   # se crea en runtime
        self.in_ch_raw = in_ch

        # SA dinámicas
        self.sa1 = SA_WSLFA(0, k_neighbors=k, in_ch=0, mlp_out=128)
        self.sa2 = SA_WSLFA(None, k_neighbors=k, in_ch=None, mlp_out=256)
        self.sa3 = SA_WSLFA(None, k_neighbors=k, in_ch=None, mlp_out=512)

        # FP layers
        self.fp3 = FP_Layer(512 + 256, 256)
        self.fp2 = FP_Layer(256 + 128, 128)
        self.fp1 = FP_Layer(128 + 64, 128)

        # Head final
        self.head = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Conv1d(128, num_classes, 1),
        )

        self.M1_frac = M1_frac
        self.M2_frac = M2_frac
        self.M3_frac = M3_frac

    def _build_spfe_input(self, X):
        B, N, C = X.shape
        xyz = X[:, :, :3]
        xyz_c = xyz - xyz.mean(dim=1, keepdim=True)

        if C >= 6:
            normals = X[:, :, 3:6]
            extras  = X[:, :, 6:] if C > 6 else None
        else:
            normals = torch.zeros_like(xyz)
            extras  = X[:, :, 3:] if C > 3 else None

        parts = [xyz, xyz_c, normals]
        if extras is not None and extras.shape[-1] > 0:
            parts.append(extras)

        return torch.cat(parts, dim=-1)  # (B,N,D)

    def forward(self, X):  # X: (B,N,C)
        B, N, C = X.shape
        xyz = X[:, :, :3]

        # ------------------------------------------------------
        # 1) Construir entrada SPFE (XYZ, XYZ centrado, normales, extras)
        # ------------------------------------------------------
        xyz_c = xyz - xyz.mean(dim=1, keepdim=True)

        if C >= 6:
            normals = X[:, :, 3:6]
            extras  = X[:, :, 6:] if C > 6 else None
        else:
            normals = torch.zeros_like(xyz)
            extras  = X[:, :, 3:] if C > 3 else None

        parts = [xyz, xyz_c, normals]
        if extras is not None and extras.shape[-1] > 0:
            parts.append(extras)

        spfe_in = torch.cat(parts, dim=-1)              # (B,N,D)
        D = spfe_in.shape[-1]                           # canales reales
        spfe_in = spfe_in.transpose(2, 1).contiguous()  # (B,D,N)

        # ------------------------------------------------------
        # 2) Inicializar SPFE solo UNA VEZ
        # ------------------------------------------------------
        if self.spfe is None:
            print(f"[INIT] SPFE creado con in_ch_spfe={D}")
            self.spfe = SPFE(in_ch_spfe=D).to(X.device)

        # Si la MLP no coincide → reconstruir SOLO la MLP
        elif (self.spfe.mlp is None or
              list(self.spfe.mlp[0].weight.shape)[1] != D):

            print(f"[INIT] SPFE.mlp reconstruida con in_ch_spfe={D}")

            self.spfe.mlp = nn.Sequential(
                nn.Conv1d(D, self.spfe.out_ch, 1),
                nn.BatchNorm1d(self.spfe.out_ch),
                    nn.ReLU(True),
            nn.Conv1d(self.spfe.out_ch, self.spfe.out_ch, 1),
            nn.BatchNorm1d(self.spfe.out_ch),
            nn.ReLU(True)
            ).to(X.device)

    # ------------------------------------------------------
    # 3) Forward SPFE
    # ------------------------------------------------------
        f0 = self.spfe.mlp(spfe_in)   # (B,64,N)

    # ------------------------------------------------------
    # 4) Encoder jerárquico (SA1-SA2-SA3)
    # ------------------------------------------------------
        M1 = max(1, int(N * self.M1_frac))
        M2 = max(1, int(N * self.M2_frac))
        M3 = max(1, int(N * self.M3_frac))

        self.sa1.n_center = M1
        self.sa2.n_center = M2
        self.sa3.n_center = M3

        xyz1, f1 = self.sa1(xyz, f0)
        xyz2, f2 = self.sa2(xyz1, f1)
        xyz3, f3 = self.sa3(xyz2, f2)

    # ------------------------------------------------------
    # 5) Decoder (FP3 → FP2 → FP1)
    # ------------------------------------------------------
        f_up2 = self.fp3(xyz2, xyz3, f2, f3)
        f_up1 = self.fp2(xyz1, xyz2, f1, f_up2)
        f_up0 = self.fp1(xyz,  xyz1, f0, f_up1)

    # ------------------------------------------------------
    # 6) Head final
    # ------------------------------------------------------
        logits = self.head(f_up0).transpose(2, 1)
        return logits


# ==============================================================
# === DILATED TOOTH SEG NET (Conv1D Dilatadas) =================
# ==============================================================

class DilatedConvBlock(nn.Module):
    """
    Bloque básico:
        Conv1D (kernel=3, dilatación=d)
        BatchNorm
        ReLU
    """
    def __init__(self, in_ch, out_ch, dilation):
        super().__init__()
        self.conv = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size=3,
            padding=dilation,
            dilation=dilation
        )
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DilatedToothSegNet(nn.Module):
    """
    Arquitectura liviana estilo Temporal Convolutional Network (TCN)
    adaptada a nubes de puntos.

    Super flexible: recibe XYZ o XYZ+features (in_ch detectado).
    """
    def __init__(self,
                 num_classes: int,
                 in_ch: int = 3,
                 base_ch: int = 64,
                 dropout: float = 0.3):
        super().__init__()

        self.encoder = nn.Sequential(
            DilatedConvBlock(in_ch,          base_ch,     dilation=1),
            DilatedConvBlock(base_ch,        base_ch * 2, dilation=2),
            DilatedConvBlock(base_ch * 2,    base_ch * 4, dilation=4),
        )

        self.bottleneck = nn.Sequential(
            nn.Conv1d(base_ch * 4, base_ch * 4, 1),
            nn.BatchNorm1d(base_ch * 4),
            nn.ReLU(True),
        )

        self.decoder = nn.Sequential(
            nn.Conv1d(base_ch * 4, base_ch * 2, 1),
            nn.BatchNorm1d(base_ch * 2),
            nn.ReLU(True),

            nn.Conv1d(base_ch * 2, base_ch, 1),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(True),

            nn.Dropout(dropout),
            nn.Conv1d(base_ch, num_classes, 1)
        )

    def forward(self, pts):  # (B,N,C)
        x = pts.transpose(1, 2)       # (B,C,N)
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x.transpose(1, 2)      # (B,N,C_classes)


# ==============================================================
# === TRANSFORMER3D (Fourier Positional Encoding) ==============
# ==============================================================

class FourierPE(nn.Module):
    """
    Positional Encoding para 3D basado en frecuencias 2^k.
    Produce senos y cosenos según XYZ.
    """
    def __init__(self, num_feats=32, scale=10.0):
        super().__init__()
        self.num_feats = num_feats
        self.scale = scale

    def forward(self, xyz):  # (B,N,3)
        x = xyz * self.scale
        k = torch.arange(self.num_feats, device=xyz.device).float()
        freqs = (2.0 ** k)[None, None, :]  # (1,1,num_feats)

        sin = torch.sin(x.unsqueeze(-1) / freqs)
        cos = torch.cos(x.unsqueeze(-1) / freqs)

        # (B,N,3,num_feats*2) → (B,N,6*num_feats)
        return torch.cat([sin, cos], dim=-1).reshape(xyz.size(0),
                                                     xyz.size(1),
                                                     -1)


class Transformer3D(nn.Module):
    """
    Modelo Transformer simple para nubes 3D:

     input_dim = in_ch + PE_dim
     encoder Transformer clásico
     head lineal final por punto
    """
    def __init__(self,
                 num_classes: int,
                 d_model: int = 128,
                 nhead: int = 4,
                 depth: int = 4,
                 dim_ff: int = 256,
                 in_ch: int = 3,
                 dropout: float = 0.5):
        super().__init__()

        # Fourier PE basado solo en XYZ
        self.pe = FourierPE(num_feats=d_model // 6)

        # Dimensión total de entrada
        in_dim = in_ch + (3 * 2 * (d_model // 6))

        self.lin = nn.Linear(in_dim, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, pts):  # (B,N,C)
        xyz = pts[:, :, :3]
        pe = self.pe(xyz)

        x = torch.cat([pts, pe], dim=-1)
        x = self.lin(x)
        x = self.encoder(x)
        return self.head(x)  # (B,N,C_classes)


# ==============================================================
# === TOOTHFORMER (Transformer por patches) ====================
# ==============================================================

class LearnablePE(nn.Module):
    """Positional embeddings aprendibles para tokens."""
    def __init__(self, dim, max_patches=256):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_patches, dim) * 0.02)

    def forward(self, x):  # (B,M,dim)
        return x + self.pe[:, :x.size(1), :]


class PatchEmbed(nn.Module):
    """
    Patch embedding para nubes 3D:
        - Selecciona vecindarios por KNN
        - Conv2D 1×1 (simulando VIT)
        - MaxPool sobre K vecinos
    """
    def __init__(self, in_ch_xyz=3, emb_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_ch_xyz, 64, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 1),
            nn.ReLU(True),
            nn.Conv2d(128, emb_dim, 1)
        )

    def forward(self, xyz, centers, idx_knn):
        """
        xyz:     (B,N,3)
        centers: (B,M,3)
        idx_knn: (B,M,K)
        """
        B, M, K = idx_knn.shape

        neigh = torch.gather(
            xyz[:, None, :, :].expand(-1, M, -1, -1),
            2,
            idx_knn[..., None].expand(-1, -1, -1, 3)
        )  # (B,M,K,3)

        local = neigh - centers[:, :, None, :]   # (B,M,K,3)
        x = local.permute(0, 3, 1, 2)             # (B,3,M,K)

        f = self.mlp(x)                           # (B,emb_dim,M,K)
        f = torch.max(f, dim=-1)[0]               # (B,emb_dim,M)
        return f.permute(0, 2, 1)                 # (B,M,emb_dim)


class ToothFormer(nn.Module):
    """
    Modelo Transformer basado en patches (VIT adaptado a nubes):

      - Selección uniforme de M centros
      - KNN por centro
      - Embedding por patch
      - TransformerEncoder
      - Reasignación punto→patch (1-NN)
      - Fusión con features extra si existen
    """

    def __init__(self,
                 num_classes: int,
                 emb_dim: int = 256,
                 nhead: int = 8,
                 depth: int = 6,
                 dim_ff: int = 512,
                 num_patches: int = 64,
                 k_per_patch: int = 128,
                 in_ch: int = 3,
                 dropout: float = 0.5):
        super().__init__()

        self.num_patches = num_patches
        self.k = k_per_patch
        self.in_ch = in_ch

        self.patch_embed = PatchEmbed(in_ch_xyz=3, emb_dim=emb_dim)
        self.pos = LearnablePE(emb_dim, max_patches=num_patches)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)

        # Proyección para extras
        self.proj_extras = nn.Linear(in_ch - 3, emb_dim) if in_ch > 3 else None
        self.proj_lin = nn.Linear(emb_dim * (2 if in_ch > 3 else 1), emb_dim)

        self.head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, num_classes)
        )

    @torch.no_grad()
    def _choose_centers(self, xyz):
        B, N, _ = xyz.shape
        idx = torch.linspace(0, N - 1,
                             steps=self.num_patches,
                             device=xyz.device).long()
        idx = idx.unsqueeze(0).repeat(B, 1)
        centers = torch.gather(xyz, 1, idx[..., None].expand(-1, -1, 3))
        return centers, idx

    @torch.no_grad()
    def _knn_per_center(self, centers, xyz, k):
        d = torch.cdist(centers, xyz)
        return d.topk(k=min(k, xyz.size(1)), dim=-1, largest=False)[1]

    def forward(self, pts):  # (B,N,C)
        B, N, C = pts.shape
        xyz = pts[:, :, :3]
        extras = pts[:, :, 3:] if C > 3 else None

        centers, idx = self._choose_centers(xyz)
        idx_knn = self._knn_per_center(centers, xyz, self.k)

        tokens = self.patch_embed(xyz, centers, idx_knn)
        tokens = self.encoder(self.pos(tokens))

        # reasignación punto→patch más cercano
        d_all = torch.cdist(xyz, centers)  # (B,N,M)
        idx_pc = d_all.topk(k=1, dim=-1,
                            largest=False)[1].squeeze(-1)  # (B,N)

        b = torch.arange(B, device=xyz.device)[:, None].expand(B, N)
        picked = tokens[b, idx_pc, :]  # (B,N,emb_dim)

        if extras is not None:
            ex = self.proj_extras(extras)
            picked = torch.cat([picked, ex], dim=-1)

        feats = self.proj_lin(picked)
        logits = self.head(feats)
        return logits

# ==============================================================
# === BUILDER DE MODELOS =======================================
# ==============================================================

def build_model(model_name: str,
                num_classes: int,
                in_ch: int,
                device: torch.device):
    """
    Devuelve el modelo solicitado, listo en el dispositivo.
    """
    model_name = model_name.lower()

    if model_name == "pointnet":
        return PointNetSeg(num_classes=num_classes, in_ch=in_ch).to(device)

    elif model_name == "pointnetpp":
        return PointNet2Seg(num_classes=num_classes,
                            in_ch=in_ch).to(device)

    elif model_name == "pointnetpp_improved":
        return PointNet2Seg_SPFE_WSLFA(num_classes=num_classes,
                                       in_ch=in_ch).to(device)

    elif model_name == "dilated":
        return DilatedToothSegNet(num_classes=num_classes,
                                  in_ch=in_ch).to(device)

    elif model_name == "transformer3d":
        return Transformer3D(num_classes=num_classes,
                             in_ch=in_ch).to(device)

    elif model_name == "toothformer":
        return ToothFormer(num_classes=num_classes,
                           in_ch=in_ch).to(device)

    else:
        raise ValueError(f"Modelo desconocido: {model_name}")


# ==============================================================
# === EVALUACIÓN ===============================================
# ==============================================================

@torch.no_grad()
def evaluate_model(model,
                   loader,
                   device: torch.device,
                   num_classes: int,
                   focus_id: int):
    """
    Evalúa el modelo en un loader (val o test).
    Retorna loss promedio y macro stats.
    """
    model.eval()
    loss_f = model.loss_fn
    total_loss = 0.0
    count = 0

    cm_total = torch.zeros((num_classes, num_classes),
                           device=device)

    focus_acc = 0; focus_prec = 0
    focus_rec = 0; focus_f1 = 0
    focus_iou = 0
    focus_batches = 0

    for X, Y in loader:
        X = to_device(X, device)
        Y = to_device(Y, device)

        X = normalize_cloud(X)
        logits = model(X)

        loss = loss_f(logits, Y)
        total_loss += loss.item()
        count += 1

        cm = confusion_matrix_from_logits(logits, Y, num_classes)
        cm_total += cm

        stats_f = single_class_stats(logits, Y, focus_id)
        focus_acc  += stats_f["d_focus_acc"]
        focus_prec += stats_f["d_focus_prec"]
        focus_rec  += stats_f["d_focus_rec"]
        focus_f1   += stats_f["d_focus_f1"]
        focus_iou  += stats_f["d_focus_iou"]
        focus_batches += 1

    macro = macro_stats_from_cm(cm_total)
    macro["d_focus_acc"]  = focus_acc  / max(1, focus_batches)
    macro["d_focus_prec"] = focus_prec / max(1, focus_batches)
    macro["d_focus_rec"]  = focus_rec  / max(1, focus_batches)
    macro["d_focus_f1"]   = focus_f1   / max(1, focus_batches)
    macro["d_focus_iou"]  = focus_iou  / max(1, focus_batches)

    return total_loss / max(1, count), macro


# ==============================================================
# === ENTRENAMIENTO COMPLETO ===================================
# ==============================================================

def train_model(model,
                loaders,
                device: torch.device,
                num_classes: int,
                focus_id: int,
                ce_weight: float,
                dice_weight: float,
                out_dir: Path,
                epochs: int = 300,
                lr: float = 1e-3,
                weight_decay: float = 1e-4,
                patience: int = 40,
                data_dir: Path = None):
    """
    Entrenamiento estilo 'paper-like' v12 corregido.
    Mantiene TODOS los valores por defecto originales.
    No requiere pasar data_dir (funciona igual).
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================================
    # 1) Detectar canales
    # ==========================================================
    X0, _ = next(iter(loaders["train"]))
    in_ch = X0.shape[2]

    # ==========================================================
    # 2) Class Weights — ruta real desde el dataset
    # ==========================================================
    ds_train = loaders["train"].dataset

    # Preferencia: ruta real proveniente del dataset
    Ytr_path = getattr(ds_train, "Y_path", None)

    # Fallback: uso de data_dir solo si dataset no expone Y_path
    if Ytr_path is None:
        if data_dir is None:
            raise RuntimeError(
                "ERROR: El dataset no expone Y_path y data_dir=None. "
                "Debe agregar self.Y_path en el dataset o pasar data_dir."
            )
        Ytr_path = Path(data_dir) / "Y_train.npz"

    Ytr_path = Path(Ytr_path)
    if not Ytr_path.exists():
        raise FileNotFoundError(f"[ERROR] No se encontró {Ytr_path}")

    # artifact path
    artifacts_dir = Ytr_path.parent / "artifacts"
    cw_file = artifacts_dir / "class_weights.json"

    # Cargar si existe
    W = None
    if cw_file.exists():
        W = _try_load_class_weights(artifacts_dir, num_classes)

    # Calcular si no existe
    if W is None:
        print("[WARN] No se encontró class_weights.json — usando auto-weights.")
        W = _auto_class_weights_from_train(Ytr_path, num_classes)

        # Validación de dimensión
        if W.shape[0] != num_classes:
            raise RuntimeError(
                f"[ERROR] class_weights tiene {W.shape[0]} entradas, "
                f"pero num_classes = {num_classes}"
            )

    # Enviar a device
    W = W.to(device)



    W = W.to(device)

    # ==========================================================
    # 3) Loss
    # ==========================================================
    loss_fn = CombinedLoss(
        num_classes=num_classes,
        ce_weight=ce_weight,
        dice_weight=dice_weight,
        class_weights=W
    )
    model.loss_fn = loss_fn

    # ==========================================================
    # 4) Optimizador + Scheduler
    # ==========================================================
    optim = torch.optim.AdamW(model.parameters(),
                              lr=lr,
                              weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim,
        T_max=epochs
    )

    # ==========================================================
    # 5) Early Stopping
    # ==========================================================
    best_val = float("inf")
    best_epoch = 0
    history = {}

    # ==========================================================
    # 6) LOOP DE ENTRENAMIENTO
    # ==========================================================
    for ep in range(1, epochs + 1):

        # ------------------- TRAIN -------------------
        model.train()
        train_loss = 0.0
        cm_total = torch.zeros((num_classes, num_classes), device=device)

        f_acc = f_prec = f_rec = f_f1 = f_iou = 0.0
        fb = 0

        for X, Y in loaders["train"]:
            X = to_device(X, device)
            Y = to_device(Y, device)

            X = normalize_cloud(X)
            logits = model(X)

            loss = loss_fn(logits, Y)
            optim.zero_grad()
            loss.backward()
            optim.step()

            train_loss += loss.item()
            cm_total += confusion_matrix_from_logits(logits, Y, num_classes)

            st = single_class_stats(logits, Y, focus_id)
            f_acc  += st["d_focus_acc"]
            f_prec += st["d_focus_prec"]
            f_rec  += st["d_focus_rec"]
            f_f1   += st["d_focus_f1"]
            f_iou  += st["d_focus_iou"]
            fb += 1

        macro_train = macro_stats_from_cm(cm_total)
        macro_train["d_focus_acc"]  = f_acc / max(1, fb)
        macro_train["d_focus_prec"] = f_prec / max(1, fb)
        macro_train["d_focus_rec"]  = f_rec / max(1, fb)
        macro_train["d_focus_f1"]   = f_f1 / max(1, fb)
        macro_train["d_focus_iou"]  = f_iou / max(1, fb)

        update_history(history, "train", macro_train, train_loss)

        # ------------------- VALIDACIÓN -------------------
        val_loss, macro_val = evaluate_model(
            model,
            loaders["val"],
            device,
            num_classes,
            focus_id
        )
        update_history(history, "val", macro_val, val_loss)

        # ------------------- SCHEDULER -------------------
        scheduler.step()

        # ------------------- LOG -------------------
        print(f"[Ep {ep:03d}/{epochs}] "
              f"tr={train_loss:.4f}  va={val_loss:.4f}  "
              f"acc={macro_val['acc']:.3f}  f1={macro_val['f1']:.3f}  "
              f"d_focus_f1={macro_val['d_focus_f1']:.3f}")

        # ------------------- EARLY STOPPING -------------------
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = ep
            torch.save(model.state_dict(), out_dir / "best.pt")
        elif ep - best_epoch >= patience:
            print("[EARLY] Parada temprana.")
            break

    # ==========================================================
    # 7) Guardado Final
    # ==========================================================
    torch.save(model.state_dict(), out_dir / "final_model.pt")

    save_json(history, out_dir / "history.json")
    plot_curves(history, out_dir, model.__class__.__name__)

    # ==========================================================
    # 8) Evaluación con best.pt
    # ==========================================================
    model.load_state_dict(torch.load(out_dir / "best.pt"))

    test_loss, macro_test = evaluate_model(
        model,
        loaders["test"],
        device,
        num_classes,
        focus_id
    )

    summary = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "test_loss": test_loss,
        **macro_test
    }

    save_json(summary, out_dir / "summary.json")

    print("[TEST]", summary)
    print("[DONE] Resultados guardados en:", out_dir)

    return model, history, summary

# ==============================================================
# === VISUALIZACIÓN 3D =========================================
# ==============================================================

def visualize_prediction(X, Y, pred, out_dir=None):
    """
    Guarda o muestra 3 visualizaciones interactivas en HTML usando Plotly:
        - Ground Truth (Viridis)
        - Predicción (Turbo)
        - Diferencia (Jet)

    Parámetros:
        X: np.array o tensor con forma (N, C) donde C >= 3 (XYZ)
        Y: etiquetas verdaderas (N,)
        pred: etiquetas predichas (N,)
        out_dir: carpeta donde guardar plots HTML (opcional)
    """

    import plotly.graph_objects as go
    from pathlib import Path

    # Usar solo XYZ
    xyz = X[:, :3]

    # -------------------------
    # Ground Truth
    # -------------------------
    fig_gt = go.Figure(
        data=[go.Scatter3d(
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
            mode="markers",
            marker=dict(size=2, color=Y, colorscale="Viridis"),
            name="GT"
        )]
    )
    fig_gt.update_layout(title="Ground Truth")

    # -------------------------
    # Predicción
    # -------------------------
    fig_pred = go.Figure(
        data=[go.Scatter3d(
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
            mode="markers",
            marker=dict(size=2, color=pred, colorscale="Turbo"),
            name="Pred"
        )]
    )
    fig_pred.update_layout(title="Predicción del Modelo")

    # -------------------------
    # Diferencia
    # -------------------------
    diff = (pred != Y).astype(int)

    fig_diff = go.Figure(
        data=[go.Scatter3d(
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
            mode="markers",
            marker=dict(size=2, color=diff, colorscale="Jet"),
            name="Diff"
        )]
    )
    fig_diff.update_layout(title="Diferencias (1 = error)")

    # -------------------------
    # Guardado o visualización
    # -------------------------
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        fig_gt.write_html(out_dir / "plot_gt.html")
        fig_pred.write_html(out_dir / "plot_pred.html")
        fig_diff.write_html(out_dir / "plot_diff.html")

        print(f"[PLOTS] Guardados en: {out_dir}")
    else:
        fig_gt.show()
        fig_pred.show()
        fig_diff.show()

# ==============================================================
# === MAIN + ARGPARSE ==========================================
# ==============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Entrenamiento paper-like v12 para segmentación dental 3D"
    )

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True,
                        choices=["pointnet", "pointnetpp", "pointnetpp_improved",
                                 "dilated", "transformer3d", "toothformer"])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--ce_weight", type=float, default=1.0)
    parser.add_argument("--dice_weight", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--focus_id", type=int, default=7)
    parser.add_argument("--tag", type=str, default="exp")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("[DEVICE]", device)

    data_dir = Path(args.data_dir)
    loaders = make_loaders(data_dir, batch_size=args.batch_size)

    # === DETECCIÓN GLOBAL DE CLASES (CORREGIDO) =====
    Ytrain_full = np.load(Path(args.data_dir) / "Y_train.npz")["Y"]
    num_classes = int(Ytrain_full.max() + 1)

    X0, _ = next(iter(loaders["train"]))
    in_ch = X0.shape[2]

    print(f"[INFO] Detectadas {num_classes} clases (global).")
    print(f"[INFO] Detectados {in_ch} canales por punto.")

    out_dir = Path("runs_v12") / f"{args.tag}"
    model = build_model(args.model, num_classes, in_ch, device)

    model, history, summary = train_model(
        model=model,
        loaders=loaders,
        device=device,
        num_classes=num_classes,
        focus_id=args.focus_id,
        ce_weight=args.ce_weight,
        dice_weight=args.dice_weight,
        out_dir=out_dir,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        data_dir=data_dir
    )

    print("[FINISHED] Entrenamiento completado.")
