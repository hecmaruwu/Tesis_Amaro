#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_models_v11_paperlike.py (PARTE 1/5)
--------------------------------------------------------------
Entrenamiento paper-like para segmentación por puntos con:
 - PointNet
 - PointNet++ (normal)
 - PointNet++ (mejorado)
 - Transformer3D (básico)
 - ToothFormer (académico-lite)

Compatibilidad v4:
 - Lee splits con XYZ o XYZ+features geométricas (normales, curvatura).
 - Auto-detecta in_ch desde X_train.npz.
 - Mantiene outputs estilo v10: checkpoints/best.pt, final_model.pt,
   history.csv, summary.json y plots PNG de métricas.

Esta PARTE 1 contiene:
  * Imports y utilidades generales
  * Dataset y DataLoaders
  * Métricas (macro y por clase “d21”)
  * Lectura / construcción de class weights
--------------------------------------------------------------
Autor: Adaptado por ChatGPT (GPT-5 Thinking)
"""

# ==============================================================
# === Importaciones generales ==================================
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
# === Utilidades generales =====================================
# ==============================================================

def set_seed(seed: int = 42):
    """Fija semillas para reproducibilidad (CPU/GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinismo (más lento, pero reproduce resultados)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_device(x, device: torch.device):
    """Envía tensores o colecciones de tensores al dispositivo."""
    if isinstance(x, (tuple, list)):
        return [to_device(t, device) for t in x]
    return x.to(device, non_blocking=True)


def sanitize_tensor(t: torch.Tensor) -> torch.Tensor:
    """Reemplaza NaN/Inf por 0 (seguridad numérica)."""
    return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)


def normalize_cloud(x: torch.Tensor) -> torch.Tensor:
    """
    Normaliza cada nube a esfera unitaria (centro=0, radio máx=1).
    x: (B, P, C>=3) — normaliza XYZ (primeras 3 cols) y deja features igual.
    """
    B, P, C = x.shape
    xyz = x[:, :, :3]
    feats = x[:, :, 3:] if C > 3 else None

    c = xyz.mean(dim=1, keepdim=True)
    xyz = xyz - c
    r = (xyz.pow(2).sum(-1).sqrt()).max(dim=1, keepdim=True)[0].unsqueeze(-1)
    xyz = xyz / (r + 1e-8)

    if feats is not None:
        x = torch.cat([xyz, feats], dim=-1)
    else:
        x = xyz
    return x


def save_json(obj: Any, path: Path):
    """Guarda un dict/objeto JSON con mkdir previa."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_history_csv(history: Dict[str, List[float]], out_csv: Path):
    """Guarda history.csv (mismas claves que v10)."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    any_key = next(iter(history.keys()))
    T = len(history[any_key])
    keys = sorted(history.keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch"] + keys)
        for ep in range(T):
            row = [ep + 1] + [history[k][ep] if ep < len(history[k]) else "" for k in keys]
            w.writerow(row)



def plot_curves(history: Dict[str, List[float]], out_dir: Path, model_name: str):

    """
    Genera PNGs de curvas (loss/acc/f1/iou/d21_f1/recall/precision)
    para train/val en formato similar a v10.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = ["loss", "acc", "f1", "iou", "prec", "rec", "d_focus_f1"]
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


def update_history(history: Dict[str, List[float]], prefix: str, stats: Dict[str, float]):
    """Actualiza el dict de historial (train_* / val_*)."""
    for k, v in stats.items():
        history.setdefault(f"{prefix}_{k}", []).append(float(v))


def print_epoch(ep: int, epochs: int, tr_loss: float, va_loss: float, va_stats: Dict[str, float]):
    """Imprime resumen por época (formato v10)."""
    msg = (
        f"[Ep {ep:03d}/{epochs}] "
        f"tr={tr_loss:.4f}  va={va_loss:.4f}  "
        f"acc={va_stats['acc']:.3f}  f1={va_stats['f1']:.3f}  "
        f"iou={va_stats['iou']:.3f}  prec={va_stats['precision']:.3f}  rec={va_stats['recall']:.3f}  "
        f"d21_f1={va_stats['d21_f1']:.3f}"
    )
    print(msg)


# ==============================================================
# === Dataset y DataLoaders ====================================
# ==============================================================

class CloudDataset(Dataset):
    """
    Lee arrays comprimidos .npz con claves:
      - X: (M, P, C>=3)
      - Y: (M, P) etiquetas enteras (0..num_classes-1), -1 opcional para padding
    """
    def __init__(self, X_path: Path, Y_path: Path):
        X_np = np.load(X_path)["X"]
        Y_np = np.load(Y_path)["Y"]
        assert X_np.shape[0] == Y_np.shape[0], f"Inconsistencia entre {X_path} y {Y_path}"
        self.X = torch.from_numpy(X_np.astype(np.float32))
        self.Y = torch.from_numpy(Y_np.astype(np.int64))

    def __len__(self): 
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def make_loaders(data_dir: Path, batch_size=8, num_workers=6) -> Dict[str, DataLoader]:
    """
    Crea DataLoaders para train/val/test.
    Archivos esperados: X_train.npz, Y_train.npz, etc.
    """
    data_dir = Path(data_dir)
    loaders = {}
    for split in ["train", "val", "test"]:
        Xp, Yp = data_dir / f"X_{split}.npz", data_dir / f"Y_{split}.npz"
        if not (Xp.exists() and Yp.exists()):
            raise FileNotFoundError(f"Faltan archivos para split {split}: {Xp}, {Yp}")
        ds = CloudDataset(Xp, Yp)
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=(split == "train"),
            num_workers=num_workers, pin_memory=True, drop_last=False
        )
    return loaders


# ==============================================================
# === Métricas (macro, por clase d21) ==========================
# ==============================================================

@torch.no_grad()
def confusion_matrix_from_logits(logits: torch.Tensor, y_true: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    CM multi-clase (B, P, C) vs (B, P) → (C, C).
    Usa argmax sobre logits. Ignora etiquetas -1 (padding).
    """
    preds = logits.argmax(dim=-1)
    t = y_true.view(-1); p = preds.view(-1)
    valid = (t >= 0) & (t < num_classes)
    t, p = t[valid], p[valid]
    idx = t * num_classes + p
    cm = torch.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return cm


def macro_stats_from_cm(cm: torch.Tensor) -> Dict[str, float]:
    """
    Devuelve métricas macro: accuracy, F1, IoU, precisión y recall.
    """
    cm = cm.float()
    tp = torch.diag(cm)
    gt = cm.sum(1).clamp_min(1e-8)
    pd = cm.sum(0).clamp_min(1e-8)
    acc = (tp.sum() / cm.sum().clamp_min(1e-8)).item()

    precision_c = (tp / pd).clamp(0, 1)
    recall_c    = (tp / gt).clamp(0, 1)
    f1_c        = (2 * precision_c * recall_c / (precision_c + recall_c + 1e-8)).clamp(0, 1)
    iou_c       = (tp / (gt + pd - tp).clamp_min(1e-8)).clamp(0, 1)

    return {
        "acc": acc,
        "precision": precision_c.mean().item(),
        "recall":    recall_c.mean().item(),
        "f1":        f1_c.mean().item(),
        "iou":       iou_c.mean().item(),
    }


def single_class_stats(logits: torch.Tensor, y_true: torch.Tensor, cls_id: int) -> Dict[str, float]:
    """
    Métricas por clase objetivo (p.ej., d21).
    Retorna d21_acc, d21_f1, d21_iou, d21_precision, d21_recall.
    """
    preds = logits.argmax(dim=-1).view(-1)
    t = y_true.view(-1)
    valid = (t >= 0)
    preds, t = preds[valid], t[valid]

    tp = ((preds == cls_id) & (t == cls_id)).sum().float()
    fp = ((preds == cls_id) & (t != cls_id)).sum().float()
    fn = ((preds != cls_id) & (t == cls_id)).sum().float()
    tn = ((preds != cls_id) & (t != cls_id)).sum().float()

    acc  = ((tp + tn) / (tp + tn + fp + fn + 1e-8)).item()
    prec = (tp / (tp + fp + 1e-8)).item()
    rec  = (tp / (tp + fn + 1e-8)).item()
    f1   = (2 * prec * rec / (prec + rec + 1e-8))
    iou  = (tp / (tp + fp + fn + 1e-8)).item()

    return {
        "d21_acc": acc,
        "d21_precision": prec,
        "d21_recall": rec,
        "d21_f1": f1,
        "d21_iou": iou
    }


# ==============================================================
# === Pesos por clase (lectura / auto-cálculo) =================
# ==============================================================

def _try_load_class_weights(artifacts_dir: Path, num_classes: int) -> Optional[torch.Tensor]:
    """
    Lee artifacts/class_weights.json si existe.
    Soporta tanto lista como dict { "0": w0, "1": w1, ... } o {"class_weights": {...}}.
    """
    f = artifacts_dir / "class_weights.json"
    if not f.exists():
        return None
    try:
        obj = json.loads(f.read_text())
        if isinstance(obj, dict) and "class_weights" in obj:
            obj = obj["class_weights"]
        if isinstance(obj, dict):
            arr = np.zeros((num_classes,), dtype=np.float32)
            for k, v in obj.items():
                i = int(k)
                if 0 <= i < num_classes:
                    arr[i] = float(v)
            return torch.tensor(arr, dtype=torch.float32)
        else:
            arr = np.array(obj, dtype=np.float32)
            if arr.shape[0] != num_classes:
                print("[WARN] Tamaño de class_weights.json no coincide; ignorando archivo.")
                return None
            return torch.tensor(arr, dtype=torch.float32)
    except Exception as e:
        print(f"[WARN] Error leyendo class_weights.json: {e}")
        return None


def _auto_class_weights_from_train(Ytr_path: Path,
                                   num_classes: int,
                                   clip_min: float = 0.2,
                                   clip_max: float = 5.0) -> torch.Tensor:
    """
    Esquema paper-like:
        w_c = 1 / log(1.2 + freq_c)
    Normaliza a media=1 y recorta a [clip_min, clip_max].
    """
    Ytr = np.load(Ytr_path)["Y"]
    freqs = np.bincount(Ytr.ravel(), minlength=num_classes).astype(np.float64)
    freqs = np.maximum(freqs, 1.0)
    inv_log = 1.0 / np.log(1.2 + freqs)
    inv_log /= inv_log.mean()
    inv_log = np.clip(inv_log, clip_min, clip_max)
    return torch.tensor(inv_log, dtype=torch.float32)

# ==============================================================
# === BLOQUE DE MODELOS ========================================
# ==============================================================

# --------------------------------------------------------------
# 0. Helpers comunes a varios modelos
# --------------------------------------------------------------
class MLP1d(nn.Module):
    def __init__(self, in_ch, mlp):
        super().__init__()
        layers, c = [], in_ch
        for oc in mlp:
            layers += [nn.Conv1d(c, oc, 1), nn.BatchNorm1d(oc), nn.ReLU(True)]
            c = oc
        self.net = nn.Sequential(*layers)
    def forward(self, x):  # x: (B, C, N)
        return self.net(x)

# --------------------------------------------------------------
# 1) PointNet (Qi et al., 2017) con T-Net de entrada
# --------------------------------------------------------------
class STN3d(nn.Module):
    """Spatial Transformer 3D — Alinea nubes antes del backbone."""
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.conv1, self.bn1 = nn.Conv1d(k, 64, 1), nn.BatchNorm1d(64)
        self.conv2, self.bn2 = nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128)
        self.conv3, self.bn3 = nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024)
        self.fc1, self.bn4 = nn.Linear(1024, 512), nn.BatchNorm1d(512)
        self.fc2, self.bn5 = nn.Linear(512, 256), nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, k * k)

    def forward(self, x):  # x: (B, k, N)
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
    """
    Segmentación punto a punto con T-Net. Acepta in_ch>=3 (XYZ + features).
    Si in_ch>3, el T-Net se aplica sobre las primeras 3 coords (XYZ) y
    luego se concatenan las features crudas al resto del pipeline.
    """
    def __init__(self, num_classes=26, dropout=0.5, in_ch=3):
        super().__init__()
        self.in_ch = in_ch
        self.use_tnet = True  # mantener paper-like
        self.input_tnet = STN3d(k=3)

        # El primer bloque conv recibe in_ch canales
        self.conv1, self.bn1 = nn.Conv1d(in_ch, 64, 1), nn.BatchNorm1d(64)
        self.conv2, self.bn2 = nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128)
        self.conv3, self.bn3 = nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024)

        # Skip global + local (x2) → 1024 + 128 = 1152
        self.fconv1, self.bn4 = nn.Conv1d(1152, 512, 1), nn.BatchNorm1d(512)
        self.fconv2, self.bn5 = nn.Conv1d(512, 256, 1), nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(dropout)
        self.fconv3 = nn.Conv1d(256, num_classes, 1)

    def forward(self, pts):  # pts: (B, P, C_in)
        B, P, C = pts.shape
        x = pts.transpose(2, 1)  # (B, C_in, N)

        if self.use_tnet:
            # aplicar T-Net solo a XYZ
            xyz = x[:, :3, :]                               # (B,3,N)
            T = self.input_tnet(xyz)                        # (B,3,3)
            xyz = torch.bmm(T, xyz)                         # (B,3,N)
            if C > 3:
                x = torch.cat([xyz, x[:, 3:, :]], dim=1)    # (B, C_in, N)
            else:
                x = xyz

        x1 = F.relu(self.bn1(self.conv1(x)))                # (B,64,N)
        x2 = F.relu(self.bn2(self.conv2(x1)))               # (B,128,N)
        x3 = F.relu(self.bn3(self.conv3(x2)))               # (B,1024,N)
        xg = torch.max(x3, 2, keepdim=True)[0].repeat(1, 1, P)
        x_cat = torch.cat([xg, x2], 1)                      # (B,1152,N)

        x = F.relu(self.bn4(self.fconv1(x_cat)))
        x = F.relu(self.bn5(self.fconv2(x)))
        x = self.dropout(x)
        x = self.fconv3(x)                                  # (B,num_classes,N)
        return x.transpose(2, 1)                            # (B,N,num_classes)

# --------------------------------------------------------------
# 2) PointNet++ (normal): SA(Multi-Scale) simplificada + FP
#    Acepta in_ch variable; primer SA usa in_ch.
# --------------------------------------------------------------

class PointNet2Seg(nn.Module):
    """
    PointNet++ (normal). Acepta in_ch; primera SA usa in_ch.
    """
    def __init__(self, num_classes=26, nsample=32, in_ch=3, dropout=0.5):
        super().__init__()
        self.sa1 = SA_Layer_PN2(nsample, in_ch=in_ch, mlp=[64, 128, 256])  # in_ch -> SA1
        self.sa2 = SA_Layer_PN2(nsample, in_ch=256,   mlp=[256, 512, 512])

        self.fp1 = FP_Layer_PN2(512 + 256, [256, 256])
        self.fp2 = FP_Layer_PN2(256,       [256, 128])

        self.head = nn.Sequential(
            nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Dropout(dropout), nn.Conv1d(128, num_classes, 1)
        )

    def forward(self, pts):  # pts: (B,N,C_in)
        xyz = pts[:, :, :3]
        feats0 = None if pts.shape[2] == 3 else pts[:, :, 3:].transpose(1, 2)  # (B,C_extras,N) o None

        xyz1, f1 = self.sa1(xyz, feats0)          # (B,M1,3), (B,256,M1)
        xyz2, f2 = self.sa2(xyz1, f1)             # (B,M2,3), (B,512,M2)
        f = self.fp1(xyz1, xyz2, f1, f2)          # (B,256,M1)
        f = self.fp2(xyz,  xyz1, None, f)         # (B,128,N)
        out = self.head(f).transpose(2, 1)        # (B,N,num_classes)
        return out


# --------------------------------------------------------------
# 3) PointNet++ (mejorado) — variaciones menores y robustez
# --------------------------------------------------------------

# ---------- Utilidad: FPS (simple) ----------
@torch.no_grad()
def fps_uniform(xyz, M):
    # xyz: (B,N,3)
    B, N, _ = xyz.shape
    idx = torch.linspace(0, N-1, steps=M, device=xyz.device).long()
    return idx.unsqueeze(0).repeat(B, 1)  # (B,M)

# ---------- SPFE: Single-Point Preliminary Feature Extraction ----------
class SPFE(nn.Module):
    """
    Toma como entrada: concat([xyz, xyz_c, normales?, extras?])
    y produce 64 dims por punto.
    Ajusta dinámicamente el número de canales de entrada.
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
            extras  = xyz_all[:, :, 6:] if C > 6 else None
        elif C > 3:
            extras = xyz_all[:, :, 3:]

        parts = [xyz, xyz_c, normals]
        if extras is not None and extras.shape[-1] > 0:
            parts.append(extras)

        spfe_in = torch.cat(parts, dim=-1).transpose(2, 1).contiguous()  # (B,D,N)
        D = spfe_in.shape[1]

        # Inicializa la MLP si aún no existe o si cambia el número de canales
        if self.mlp is None or list(self.mlp[0].weight.shape)[1] != D:
            self.mlp = nn.Sequential(
                nn.Conv1d(D, self.out_ch, 1),
                nn.BatchNorm1d(self.out_ch),
                nn.ReLU(True),
                nn.Conv1d(self.out_ch, self.out_ch, 1),
                nn.BatchNorm1d(self.out_ch),
                nn.ReLU(True)
            ).to(xyz_all.device)

        return self.mlp(spfe_in)  # (B, 64, N)


# ---------- SA (Set Abstraction) con WSLFA ----------
class SA_WSLFA(nn.Module):
    """
    Set Abstraction con WSLFA (Weighted Summation via Learnable Feature Aggregation)
    Funciona con cualquier número de canales dinámicos.
    """
    def __init__(self, n_center, k_neighbors, in_ch, mlp_out):
        super().__init__()
        self.n_center = n_center
        self.k = k_neighbors
        self.mlp_out = mlp_out
        self.out_ch = mlp_out

        # Definimos flags, pero NO creamos capas hasta forward()
        self._mlp_feat_ok = False
        self._mlp_alpha_ok = False

    def _build_mlp_feat(self, D_cat, device):
        """Construye la MLP de features una vez que se conoce D_cat."""
        self.mlp_feat = nn.Sequential(
            nn.Conv2d(D_cat, self.out_ch, 1),
            nn.BatchNorm2d(self.out_ch),
            nn.ReLU(True),
        ).to(device)
        self._mlp_feat_ok = True

    def _build_mlp_alpha(self, D_alpha, device):
        """Construye la MLP de pesos (alpha) una vez que se conoce D_alpha."""
        self.mlp_alpha = nn.Sequential(
            nn.Conv2d(D_alpha, self.out_ch, 1),
            nn.BatchNorm2d(self.out_ch),
            nn.ReLU(True),
        ).to(device)
        self._mlp_alpha_ok = True

    def forward(self, xyz, feat_in):
        """
        xyz:     (B, N, 3)
        feat_in: (B, C_in, N)
        """
        B, N, _ = xyz.shape
        C_in = feat_in.shape[1]
        M = self.n_center

        # -------------------- FPS --------------------
        idx_center = fps_uniform(xyz, M)
        centers_xyz = torch.gather(
            xyz, 1, idx_center[..., None].expand(-1, -1, 3)
        )  # (B, M, 3)

        # -------------------- KNN --------------------
        d = torch.cdist(centers_xyz, xyz)                      # (B, M, N)
        idx_knn = d.topk(k=min(self.k, N), dim=-1, largest=False)[1]   # (B, M, K)

        # -------------------- Coordenadas relativas --------------------
        neigh_xyz = torch.gather(
            xyz[:, None, :, :].expand(-1, M, -1, -1),
            2, idx_knn[..., None].expand(-1, -1, -1, 3)
        )  # (B,M,K,3)

        local_xyz = neigh_xyz - centers_xyz[:, :, None, :]     # (B,M,K,3)

        # -------------------- Features vecinos --------------------
        feat_T = feat_in.transpose(2, 1)  # (B,N,C_in)
        neigh_f = torch.gather(
            feat_T[:, None, :, :].expand(-1, M, -1, -1),
            2, idx_knn[..., None].expand(-1, -1, -1, C_in)
        )  # (B,M,K,C_in)

        # -------------------- Combinar coords + feats --------------------
        cat = torch.cat([local_xyz, neigh_f], dim=-1)          # (B,M,K,3+C_in)
        cat = cat.permute(0, 3, 1, 2).contiguous()             # (B,3+C_in,M,K)

        D_cat = 3 + C_in

        # -------------------- Build mlp_feat dinámicamente --------------------
        if (not self._mlp_feat_ok) or (self.mlp_feat[0].weight.shape[1] != D_cat):
            self._build_mlp_feat(D_cat, xyz.device)

        f_prime = self.mlp_feat(cat)  # (B,out_ch,M,K)
        f_mean = f_prime.mean(dim=-1, keepdim=True)

        # -------------------- Entrada para alpha --------------------
        alpha_in = torch.cat([cat, f_prime - f_mean], dim=1)  # (B,3+C_in+out_ch,M,K)
        D_alpha = D_cat + self.out_ch

        # -------------------- Build mlp_alpha dinámicamente --------------------
        if (not self._mlp_alpha_ok) or (self.mlp_alpha[0].weight.shape[1] != D_alpha):
            self._build_mlp_alpha(D_alpha, xyz.device)

        alpha = self.mlp_alpha(alpha_in)
        w = torch.softmax(alpha, dim=-1)

        # -------------------- Agregación --------------------
        f_region = (w * f_prime).sum(dim=-1)  # (B,out_ch,M)

        return centers_xyz, f_region


# ---------- FP (Feature Propagation) con interpolación 3-NN ----------
class FP_Layer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1), nn.BatchNorm1d(out_ch), nn.ReLU(True),
        )
        self.out_ch = out_ch

    def forward(self, xyz_low, xyz_high, feat_low, feat_high):
        """
        Interpola de (xyz_high, feat_high) -> xyz_low y concat con feat_low.
        xyz_low:  (B,Nl,3)
        xyz_high: (B,Nh,3)
        feat_low: (B,Cl,Nl)  o None
        feat_high:(B,Ch,Nh)
        """
        B, Nl, _ = xyz_low.shape
        Nh = xyz_high.shape[1]
        d = torch.cdist(xyz_low, xyz_high)                      # (B,Nl,Nh)
        idx = d.topk(k=min(3, Nh), dim=-1, largest=False)[1]    # (B,Nl,3)
        dist = torch.gather(d, 2, idx).clamp_min(1e-8)          # (B,Nl,3)
        w = (1.0 / dist); w = w / w.sum(dim=-1, keepdim=True)   # (B,Nl,3)
        f_high_T = feat_high.transpose(2,1)                     # (B,Nh,Ch)
        neigh = torch.gather(
            f_high_T, 1, idx[..., None].expand(-1, -1, -1, f_high_T.shape[-1])
        )                                                       # (B,Nl,3,Ch)
        f_interp = (w[..., None] * neigh).sum(dim=2).transpose(2,1)  # (B,Ch,Nl)
        if feat_low is not None:
            f_cat = torch.cat([f_interp, feat_low], dim=1)      # (B,Ch+Cl,Nl)
        else:
            f_cat = f_interp
        return self.mlp(f_cat)                                  # (B,out_ch,Nl)


class PointNet2Seg_SPFE_WSLFA(nn.Module):
    """
    PointNet++ mejorado al estilo del paper:
      - SPFE (64 dims) antes del encoder
      - SA con WSLFA (agregación por suma ponderada aprendida)
      - 3 niveles SA (M1/M2/M3) y 3 FP
      - Cabeza de segmentación por punto
    """
    def __init__(self, num_classes=26, in_ch=3, k=32,
                 M1_frac=1/4, M2_frac=1/8, M3_frac=1/16, dropout=0.5):
        super().__init__()
        # SPFE: calcula dinámicamente el in_ch efectivo (xyz + xyz_c + (normales o ceros) + extras)
        # Se inicializa en forward la primera vez según D real
        self.spfe = None

        # Encoder (dimensiones de salida por nivel)
        self.sa1 = SA_WSLFA(n_center=None, k_neighbors=k, in_ch=None, mlp_out=128)
        self.sa2 = SA_WSLFA(n_center=None, k_neighbors=k, in_ch=128 + 3, mlp_out=256)
        self.sa3 = SA_WSLFA(n_center=None, k_neighbors=k, in_ch=256 + 3, mlp_out=512)

        # Decoder (Feature Propagation)
        self.fp3 = FP_Layer(in_ch=512 + 256, out_ch=256)
        self.fp2 = FP_Layer(in_ch=256 + 128, out_ch=128)
        self.fp1 = FP_Layer(in_ch=128 + 64, out_ch=128)

        # Cabeza de segmentación
        self.head = nn.Sequential(
            nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Conv1d(128, num_classes, 1)
        )

        # Guardar fracciones (para decidir M1, M2, M3 en forward)
        self.M1_frac, self.M2_frac, self.M3_frac = M1_frac, M2_frac, M3_frac
        self.in_ch_raw = in_ch  # canales originales del input

    def _build_spfe_input(self, X):
        """Construye entrada para SPFE combinando xyz, coords centradas, normales y extras."""
        B, N, C = X.shape
        xyz = X[:, :, :3]
        xyz_c = xyz - xyz.mean(dim=1, keepdim=True)
        if C >= 6:
            normals = X[:, :, 3:6]
            extras = X[:, :, 6:] if C > 6 else None
        else:
            normals = torch.zeros_like(xyz)
            extras = X[:, :, 3:] if C > 3 else None
        parts = [xyz, xyz_c, normals]
        if extras is not None and extras.shape[-1] > 0:
            parts.append(extras)
        return torch.cat(parts, dim=-1)  # (B, N, D)

def forward(self, X):
    """
    X: (B, N, C) con C>=3
    """
    B, N, C = X.shape
    xyz = X[:, :, :3]

    # -------- SPFE: extrae features por punto --------
    spfe_in = self._build_spfe_input(X)        # (B, N, D)
    D = spfe_in.shape[-1]

    # Inicializa SPFE dinámicamente según D real
    if (self.spfe is None) or (self.spfe.mlp is None) or \
       (self.spfe.mlp[0].in_channels != D):
        print(f"[INIT] Reconfigurando SPFE con in_ch_spfe={D}")
        self.spfe = SPFE(in_ch_spfe=D).to(X.device)

    f0 = self.spfe(spfe_in)                   # (B, 64, N)

    # ==================================================
    #  ENCODER (3 niveles SA con WSLFA)
    # ==================================================

    # Cantidad de puntos por nivel
    M1 = max(1, int(N * self.M1_frac))
    M2 = max(1, int(N * self.M2_frac))
    M3 = max(1, int(N * self.M3_frac))

    self.sa1.n_center = M1
    self.sa2.n_center = M2
    self.sa3.n_center = M3

    # -------- SA1 --------
    xyz1, f1 = self.sa1(xyz, f0)              # xyz1: (B,M1,3), f1: (B,128,M1)

    # -------- SA2 --------
    xyz2, f2 = self.sa2(xyz1, f1)             # xyz2: (B,M2,3), f2: (B,256,M2)

    # -------- SA3 --------
    xyz3, f3 = self.sa3(xyz2, f2)             # xyz3: (B,M3,3), f3: (B,512,M3)

    # ==================================================
    # DECODER (3 etapas FP)
    # ==================================================

    # FP3: desde nivel 3 → 2
    f_up2 = self.fp3(xyz2, xyz3, f2, f3)      # (B,256,M2)

    # FP2: desde nivel 2 → 1
    f_up1 = self.fp2(xyz1, xyz2, f1, f_up2)   # (B,128,M1)

    # FP1: desde nivel 1 → puntos originales
    f_up0 = self.fp1(xyz,  xyz1, f0, f_up1)   # (B,128,N)

    # -------- Cabeza --------
    logits = self.head(f_up0).transpose(2, 1) # (B,N,C)

    return logits


 

# --------------------------------------------------------------
# 4) DilatedToothSegNet — Bloques 1D dilatados con skip connections
# --------------------------------------------------------------
class DilatedConvBlock(nn.Module):
    """Bloque dilatado 1D: Conv1d -> BN -> ReLU, con dilatación creciente."""
    def __init__(self, in_ch, out_ch, dilation):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(True)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DilatedToothSegNet(nn.Module):
    """
    Arquitectura ligera 1D con convoluciones dilatadas.
    Inspirada en redes tipo “TemporalConvNet”, adaptada a nubes 3D por punto.
    Usa features por punto concatenadas (XYZ + normales + curvatura).
    """
    def __init__(self, num_classes=26, in_ch=3, base_ch=64, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            DilatedConvBlock(in_ch, base_ch, dilation=1),
            DilatedConvBlock(base_ch, base_ch * 2, dilation=2),
            DilatedConvBlock(base_ch * 2, base_ch * 4, dilation=4),
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
            nn.Conv1d(base_ch, num_classes, 1),
        )

    def forward(self, pts):  # (B,N,C_in)
        x = pts.transpose(1, 2)  # (B,C_in,N)
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x.transpose(1, 2)  # (B,N,num_classes)


# --------------------------------------------------------------
# 4) Transformer3D básico (con Fourier Positional Encoding)
# --------------------------------------------------------------
class FourierPE(nn.Module):
    def __init__(self, num_feats=32, scale=10.0):
        super().__init__()
        self.num_feats = num_feats
        self.scale = scale
    def forward(self, xyz):  # (B,N,3)
        x = xyz * self.scale
        k = torch.arange(self.num_feats, device=xyz.device).float()
        freqs = (2.0 ** k)[None, None, :]
        sin = torch.sin(x.unsqueeze(-1) / freqs)
        cos = torch.cos(x.unsqueeze(-1) / freqs)
        return torch.cat([sin, cos], dim=-1).reshape(xyz.size(0), xyz.size(1), -1)


class Transformer3D(nn.Module):
    def __init__(self, num_classes=26, d_model=128, nhead=4, depth=4, dim_ff=256, in_ch=3, dropout=0.5):
        super().__init__()
        self.pe = FourierPE(num_feats=d_model // 6)
        in_dim = in_ch + (3 * 2 * (d_model // 6))
        self.lin = nn.Linear(in_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            batch_first=True, norm_first=True, dropout=dropout
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(True),
            nn.Dropout(dropout), nn.Linear(d_model, num_classes)
        )
    def forward(self, pts):  # (B,N,C_in)
        pe = self.pe(pts[:, :, :3])
        x = self.lin(torch.cat([pts, pe], -1))
        x = self.enc(x)
        return self.head(x)  # (B,N,num_classes)


# --------------------------------------------------------------
# 5) ToothFormer (académico-lite por patches)
# --------------------------------------------------------------
class LearnablePE(nn.Module):
    def __init__(self, dim, max_patches=256):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_patches, dim) * 0.02)
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class PatchEmbed(nn.Module):
    """
    Extrae tokens por patch (vecindario K por centro) usando MLP 1x1 en 2D simulado.
    Recibe solo XYZ; las features extra se proyectan al final (vía proj_lin) si se desea.
    """
    def __init__(self, in_ch_xyz=3, emb_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_ch_xyz, 64, 1), nn.ReLU(True),
            nn.Conv2d(64, 128, 1), nn.ReLU(True),
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
            xyz[:, None, :, :].expand(-1, M, -1, -1), 2,
            idx_knn[..., None].expand(-1, -1, -1, 3)
        )  # (B,M,K,3)
        local = neigh - centers[:, :, None, :]  # (B,M,K,3)
        x = local.permute(0, 3, 1, 2)           # (B,3,M,K)
        f = self.mlp(x)                         # (B,emb_dim,M,K)
        f = torch.max(f, dim=-1)[0].permute(0, 2, 1)  # (B,M,emb_dim)
        return f


class ToothFormer(nn.Module):
    def __init__(self, num_classes=26, emb_dim=256, nhead=8, depth=6, dim_ff=512,
                 num_patches=64, k_per_patch=128, in_ch=3, dropout=0.5):
        super().__init__()
        self.num_patches = num_patches
        self.k = k_per_patch
        self.in_ch = in_ch

        self.patch_embed = PatchEmbed(in_ch_xyz=3, emb_dim=emb_dim)
        self.pos = LearnablePE(emb_dim, max_patches=num_patches)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=nhead, dim_feedforward=dim_ff,
            batch_first=True, norm_first=True, dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)

        # Si hay features extra (in_ch>3), las fusionamos post-asignación
        self.proj_extras = nn.Linear(in_ch - 3, emb_dim) if in_ch > 3 else None
        self.proj_lin = nn.Linear(emb_dim * (2 if in_ch > 3 else 1), emb_dim)

        self.head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.ReLU(True),
            nn.Dropout(dropout), nn.Linear(emb_dim, num_classes)
        )

    @torch.no_grad()
    def _choose_centers(self, xyz):
        B, N, _ = xyz.shape
        idx = torch.linspace(0, N - 1, steps=self.num_patches, device=xyz.device).long()
        idx = idx.unsqueeze(0).repeat(B, 1)  # (B,M)
        centers = torch.gather(xyz, 1, idx[..., None].expand(-1, -1, 3))
        return centers, idx  # (B,M,3), (B,M)

    @torch.no_grad()
    def _knn_per_center(self, centers, xyz, k):
        d = torch.cdist(centers, xyz)  # (B,M,N)
        return d.topk(k=min(k, xyz.size(1)), dim=-1, largest=False)[1]  # (B,M,K)

    def forward(self, pts):  # pts: (B,N,C_in)
        B, N, C = pts.shape
        xyz = pts[:, :, :3]
        extras = pts[:, :, 3:] if C > 3 else None

        centers, _ = self._choose_centers(xyz)                 # (B,M,3)
        idx_knn = self._knn_per_center(centers, xyz, self.k)   # (B,M,K)
        tokens = self.patch_embed(xyz, centers, idx_knn)       # (B,M,emb_dim)
        tokens = self.encoder(self.pos(tokens))                # (B,M,emb_dim)

        # asignación punto→patch más cercano (1-NN desde centros)
        d_all = torch.cdist(xyz, centers)                      # (B,N,M)
        idx_pc = d_all.topk(k=1, dim=-1, largest=False)[1].squeeze(-1)  # (B,N)

        b = torch.arange(B, device=xyz.device)[:, None].expand(B, N)
        picked = tokens[b, idx_pc, :]                          # (B,N,emb_dim)

        if extras is not None:
            ex = self.proj_extras(extras)                      # (B,N,emb_dim)
            picked = torch.cat([picked, ex], dim=-1)           # (B,N,2*emb_dim)

        feats = self.proj_lin(picked)                          # (B,N,emb_dim)
        logits = self.head(feats)                              # (B,N,num_classes)
        return logits


# --------------------------------------------------------------
# 6) Fábrica de modelos
# --------------------------------------------------------------
# --------------------------------------------------------------
# 6) Fábrica de modelos (versión v11 - SPFE+WSLFA integrada)
# --------------------------------------------------------------
def build_model(name: str, num_classes: int, in_ch: int = 3) -> nn.Module:
    """
    Devuelve el modelo solicitado.
    - 'pointnetpp', 'pointnetpp_improved' o 'pointnetpp_spfe_wslfa' → usan la nueva arquitectura
      PointNet2Seg_SPFE_WSLFA (con SPFE + WSLFA + FP jerárquico).
    - Los demás modelos (PointNet, DilatedToothSegNet, Transformer3D, ToothFormer)
      se mantienen como en la versión original.
    """
    n = name.lower()

    if n == "pointnet":
        return PointNetSeg(num_classes=num_classes, in_ch=in_ch)

    # 🔹 Todos los alias apuntan al nuevo PointNet++ mejorado con SPFE+WSLFA
    elif n in ["pointnetpp", "pointnetpp_improved", "pointnetpp_spfe_wslfa"]:
        return PointNet2Seg_SPFE_WSLFA(num_classes=num_classes, in_ch=in_ch)

    elif n == "dilatedtoothsegnet":
        return DilatedToothSegNet(num_classes=num_classes, in_ch=in_ch)

    elif n == "transformer3d":
        return Transformer3D(num_classes=num_classes, in_ch=in_ch)

    elif n == "toothformer":
        return ToothFormer(num_classes=num_classes, in_ch=in_ch)

    else:
        raise ValueError(f"Modelo no reconocido: {name}")


# ==============================================================
# === LOSSES ===================================================
# ==============================================================

class CombinedLoss(nn.Module):
    """
    CrossEntropy + Dice (opcional).
    - class_weights: tensor [C] en el device (o None)
    - ce_weight / dice_weight: ponderaciones relativas
    """
    def __init__(self, num_classes, ce_weight=1.0, dice_weight=1.0, class_weights=None, ignore_index=-1):
        super().__init__()
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)
        self.num_classes = int(num_classes)
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)

    def forward(self, logits, y_true):
        # logits: (B, N, C), y_true: (B, N)
        loss_ce = self.ce(logits.transpose(1, 2), y_true)
        if self.dice_weight <= 0:
            return loss_ce

        probs = F.softmax(logits, dim=-1)                       # (B,N,C)
        y_onehot = F.one_hot(y_true, num_classes=self.num_classes).float()
        inter = (probs * y_onehot).sum(dim=(1, 2))              # (B,)
        union = probs.sum(dim=(1, 2)) + y_onehot.sum(dim=(1, 2))
        dice = 1.0 - (2.0 * inter + 1e-5) / (union + 1e-5)      # (B,)
        loss_dice = dice.mean()
        return self.ce_weight * loss_ce + self.dice_weight * loss_dice


# ==============================================================
# === MÉTRICAS ================================================
# ==============================================================

@torch.no_grad()
def confusion_matrix(logits: torch.Tensor, y_true: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    CM multi-clase: logits(B,N,C) vs y_true(B,N) → (C,C) con conteos.
    """
    preds = logits.argmax(dim=-1)               # (B,N)
    t = y_true.reshape(-1)
    p = preds.reshape(-1)
    valid = (t >= 0) & (t < num_classes)
    t, p = t[valid], p[valid]
    idx = t * num_classes + p
    cm = torch.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return cm


def macro_from_cm(cm: torch.Tensor) -> dict:
    """
    Devuelve macro-accuracy, macro-precision, macro-recall, macro-F1, macro-IoU.
    También retorna vectores por-clase (prec/rec/f1/iou) para el summary.json.
    """
    cm = cm.float()
    tp = torch.diag(cm)
    gt = cm.sum(1).clamp_min(1e-8)       # filas (verdaderos)
    pd = cm.sum(0).clamp_min(1e-8)       # columnas (predichos)
    tot = cm.sum().clamp_min(1e-8)

    acc = (tp.sum() / tot).item()

    prec_c = (tp / pd).cpu().numpy()
    rec_c  = (tp / gt).cpu().numpy()
    f1_c   = (2 * tp / (pd + gt)).cpu().numpy()
    iou_c  = (tp / (pd + gt - tp).clamp_min(1e-8)).cpu().numpy()

    out = {
        "acc": acc,
        "prec": float(np.nanmean(prec_c)),
        "rec":  float(np.nanmean(rec_c)),
        "f1":   float(np.nanmean(f1_c)),
        "iou":  float(np.nanmean(iou_c)),
        # Para guardar luego en summary.json:
        "_per_class": {
            "precision": prec_c.tolist(),
            "recall":    rec_c.tolist(),
            "f1":        f1_c.tolist(),
            "iou":       iou_c.tolist(),
            "support":   cm.sum(1).cpu().numpy().tolist()
        }
    }
    return out


def dclass_metrics(logits: torch.Tensor, y_true: torch.Tensor, cls_id: int) -> dict:
    """
    Métricas binarizando una clase concreta (p.ej., diente 21 remapeado).
    Retorna dX_acc, dX_prec, dX_rec, dX_f1, dX_iou.
    """
    preds = logits.argmax(dim=-1).reshape(-1)
    t = y_true.reshape(-1)

    tp = ((preds == cls_id) & (t == cls_id)).sum().float()
    fp = ((preds == cls_id) & (t != cls_id)).sum().float()
    fn = ((preds != cls_id) & (t == cls_id)).sum().float()
    tn = ((preds != cls_id) & (t != cls_id)).sum().float()

    acc = ((tp + tn) / (tp + tn + fp + fn + 1e-8)).item()
    prec = (tp / (tp + fp + 1e-8)).item()
    rec  = (tp / (tp + fn + 1e-8)).item()
    f1   = 2 * prec * rec / (prec + rec + 1e-8)
    iou  = (tp / (tp + fp + fn + 1e-8)).item()

    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "iou": iou}


# ==============================================================
# === EVALUACIÓN ===============================================
# ==============================================================

@torch.no_grad()
def evaluate_model(model, loader, loss_fn, device, num_classes, focus_id=21):
    model.eval()
    total_loss = 0.0
    cm = torch.zeros(num_classes, num_classes, device=device)
    n_batches = 0

    # Acumuladores por clase para promedio ponderado opcional (si quisieras)
    for X, Y in loader:
        X, Y = to_device([X, Y], device)
        X = sanitize_tensor(normalize_cloud(X))   # normaliza solo XYZ

        logits = model(X)                         # (B,N,C)
        loss = loss_fn(logits, Y)
        total_loss += loss.item()
        cm += confusion_matrix(logits, Y, num_classes)
        n_batches += 1

    total_loss /= max(1, n_batches)
    macro_stats = macro_from_cm(cm)
    fstats = dclass_metrics(logits, Y, cls_id=min(focus_id, num_classes - 1))

    # Compactamos salida en un dict listo para history/resumen
    stats = {
        "acc": macro_stats["acc"],
        "prec": macro_stats["prec"],
        "rec": macro_stats["rec"],
        "f1": macro_stats["f1"],
        "iou": macro_stats["iou"],
        "d_focus_acc": fstats["acc"],
        "d_focus_prec": fstats["prec"],
        "d_focus_rec": fstats["rec"],
        "d_focus_f1": fstats["f1"],
        "d_focus_iou": fstats["iou"],
        # También devolvemos los per-clase para el summary.json
        "_per_class": macro_stats["_per_class"],
    }
    return total_loss, stats


# ==============================================================
# === ENTRENAMIENTO ===========================================
# ==============================================================

def print_epoch(ep: int, epochs: int, tr_loss: float, va_loss: float, va_stats: dict, focus_name="d21"):
    msg = (
        f"[Ep {ep:03d}/{epochs}] "
        f"tr={tr_loss:.4f}  va={va_loss:.4f}  "
        f"acc={va_stats['acc']:.3f}  prec={va_stats['prec']:.3f}  rec={va_stats['rec']:.3f}  "
        f"f1={va_stats['f1']:.3f}  iou={va_stats['iou']:.3f}  "
        f"{focus_name}_f1={va_stats['d_focus_f1']:.3f}"
    )
    print(msg)


def update_history(history: dict, prefix: str, stats: dict, loss_value: float | None = None):
    """
    Guarda en history los campos estándar + opcionalmente el loss.
    Keys patrón: train_loss, val_loss, val_acc, val_prec, ...
    """
    if loss_value is not None:
        history.setdefault(f"{prefix}_loss", []).append(float(loss_value))
    for k in ["acc", "prec", "rec", "f1", "iou", "d_focus_acc", "d_focus_prec", "d_focus_rec", "d_focus_f1", "d_focus_iou"]:
        if k in stats:
            history.setdefault(f"{prefix}_{k}", []).append(float(stats[k]))


def train_model(model,
                loaders,
                device,
                num_classes,
                out_dir: Path,
                lr=1e-3, weight_decay=1e-4, epochs=400,
                patience=80, use_cosine=False,
                class_weights=None,
                ce_weight=1.0, dice_weight=1.0,
                focus_id=21,
                run_tag="model",
                seed=42):
    """
    Entrenamiento con early stopping y guardado 'paper-like':
      - checkpoints/best.pt, checkpoints/final_model.pt
      - history.csv + curvas PNG
      - summary.json con métricas macro y por-clase (precision, recall, f1, iou, support)
    """
    set_seed(seed)
    out_dir = Path(out_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    loss_fn = CombinedLoss(num_classes, ce_weight=ce_weight, dice_weight=dice_weight, class_weights=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        if use_cosine
        else torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    )

    best_val = float("inf")
    best_snapshot = None
    best_stats = None
    patience_ctr = 0
    history = {}

    for ep in range(1, epochs + 1):
        # ----------------- Train -----------------
        model.train()
        tr_loss = 0.0
        nb = 0
        for X, Y in loaders["train"]:
            X, Y = to_device([X, Y], device)
            X = sanitize_tensor(normalize_cloud(X))
            optimizer.zero_grad()
            logits = model(X)
            loss = loss_fn(logits, Y)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
            nb += 1
        tr_loss /= max(1, nb)

        # ----------------- Val -------------------
        va_loss, va_stats = evaluate_model(
            model, loaders["val"], loss_fn, device, num_classes, focus_id=focus_id
        )

        update_history(history, "train", va_stats, loss_value=tr_loss)  # guardo train_loss
        update_history(history, "val", va_stats, loss_value=va_loss)    # guardo val_* + val_loss
        print_epoch(ep, epochs, tr_loss, va_loss, va_stats, focus_name=f"d{focus_id}")

        if scheduler is not None:
            scheduler.step()

        # ----------------- Early Stopping --------
        if va_loss < best_val:
            best_val = va_loss
            best_stats = va_stats
            patience_ctr = 0
            best_snapshot = {"model": model.state_dict()}
            torch.save(best_snapshot, out_dir / "checkpoints" / "best.pt")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print("[EARLY] Parada temprana activada.")
                break

    # ----------------- Guardado final ------------
    torch.save({"model": model.state_dict()}, out_dir / "checkpoints" / "final_model.pt")

    # history + plots
    save_history_csv(history, out_dir / "history.csv")
    plot_curves(history, out_dir, model_name=run_tag)

    # ----------------- Test con best.pt ----------
    print("[EVAL] Cargando best.pt para evaluación en test.")
    state = torch.load(out_dir / "checkpoints" / "best.pt", map_location=device)
    model.load_state_dict(state["model"])
    test_loss, test_stats = evaluate_model(
        model, loaders["test"], loss_fn, device, num_classes, focus_id=focus_id
    )

    print(
        f"[TEST] loss={test_loss:.4f}  "
        f"acc={test_stats['acc']:.3f}  prec={test_stats['prec']:.3f}  rec={test_stats['rec']:.3f}  "
        f"f1={test_stats['f1']:.3f}  iou={test_stats['iou']:.3f}  "
        f"d{focus_id}_f1={test_stats['d_focus_f1']:.3f}"
    )

    # ----------------- Summary -------------------
    summary = {
        "best_val_loss": best_val,
        "best_val_stats": {
            "acc": best_stats["acc"],
            "prec": best_stats["prec"],
            "rec":  best_stats["rec"],
            "f1":   best_stats["f1"],
            "iou":  best_stats["iou"],
            "focus": {
                "class_id": int(focus_id),
                "acc": best_stats["d_focus_acc"],
                "prec": best_stats["d_focus_prec"],
                "rec":  best_stats["d_focus_rec"],
                "f1":   best_stats["d_focus_f1"],
                "iou":  best_stats["d_focus_iou"],
            },
            "per_class": best_stats["_per_class"],  # precision/recall/f1/iou/support por clase (listas)
        },
        "test_loss": test_loss,
        "test_stats": {
            "acc": test_stats["acc"],
            "prec": test_stats["prec"],
            "rec":  test_stats["rec"],
            "f1":   test_stats["f1"],
            "iou":  test_stats["iou"],
            "focus": {
                "class_id": int(focus_id),
                "acc": test_stats["d_focus_acc"],
                "prec": test_stats["d_focus_prec"],
                "rec":  test_stats["d_focus_rec"],
                "f1":   test_stats["d_focus_f1"],
                "iou":  test_stats["d_focus_iou"],
            },
            "per_class": test_stats["_per_class"],
        },
        "epochs_trained": ep,
        "seed": seed,
        "lr": lr,
        "weight_decay": weight_decay,
        "patience": patience,
        "scheduler": "cosine" if use_cosine else "step",
        "losses": {"ce_weight": ce_weight, "dice_weight": dice_weight},
        "num_classes": int(num_classes),
        "run_tag": run_tag,
    }
    save_json(summary, out_dir / "summary.json")
    print(f"[DONE] Resultados guardados en: {out_dir}")

    return model, history, summary

# ==============================================================
# === MAIN / ORQUESTADOR =======================================
# ==============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Entrenamiento paper-like para segmentación 3D (v11, flujo v4)."
    )
    parser.add_argument("--data_dir", required=True,
                        help="Ruta base con X_train.npz, Y_train.npz, etc.")
    parser.add_argument(
    "--model",
    required=True,
    choices=["pointnet", "pointnetpp", "pointnetpp_improved", "pointnetpp_spfe_wslfa", "dilatedtoothsegnet", "transformer3d", "toothformer"],
    help="Modelo a entrenar." )

    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--use_cosine", action="store_true", help="Usar scheduler CosineAnnealingLR.")
    parser.add_argument("--ce_weight", type=float, default=1.0)
    parser.add_argument("--dice_weight", type=float, default=1.0)
    parser.add_argument("--focus_id", type=int, default=7, help="Clase de enfoque (p.ej. d21).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default=None, help="Etiqueta personalizada para el experimento.")
    args = parser.parse_args()

    # ----------------------------------------------------------
    # Setup
    # ----------------------------------------------------------
    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    device = torch.device(args.device)
    artifacts = data_dir / "artifacts"

    # detecta num_classes desde Y_train
    y_path = data_dir / "Y_train.npz"
    if not y_path.exists():
        raise FileNotFoundError(f"No existe {y_path}")
    Y_train = np.load(y_path)["Y"]
    num_classes = int(np.max(Y_train) + 1)
    print(f"[INFO] Detectadas {num_classes} clases.")

    # detecta in_ch desde X_train
    x_path = data_dir / "X_train.npz"
    if not x_path.exists():
        raise FileNotFoundError(f"No existe {x_path}")
    X_train = np.load(x_path)["X"]
    in_ch = int(X_train.shape[2])
    print(f"[INFO] Detectados {in_ch} canales por punto (ej. XYZ+features).")

    # ----------------------------------------------------------
    # Carga class weights
    # ----------------------------------------------------------
    class_w = _try_load_class_weights(artifacts, num_classes)
    if class_w is None:
        print("[WARN] No se encontró class_weights.json, calculando auto-pesos con 1/log(1.2+freq).")
        class_w = _auto_class_weights_from_train(y_path, num_classes)
    class_w = class_w.to(device)

    # ----------------------------------------------------------
    # Construcción de modelo y loaders
    # ----------------------------------------------------------
    model = build_model(args.model, num_classes=num_classes, in_ch=in_ch).to(device)
    loaders = make_loaders(data_dir, batch_size=args.batch_size)

    # ----------------------------------------------------------
    # Output dirs
    # ----------------------------------------------------------
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    tag = args.tag or f"{args.model}_{timestamp}"
    run_dir = Path(f"runs_v11/{tag}")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[RUN] Guardando resultados en: {run_dir}")

    # ----------------------------------------------------------
    # Entrenamiento
    # ----------------------------------------------------------
    model, history, summary = train_model(
        model=model,
        loaders=loaders,
        device=device,
        num_classes=num_classes,
        out_dir=run_dir,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        use_cosine=args.use_cosine,
        class_weights=class_w,
        ce_weight=args.ce_weight,
        dice_weight=args.dice_weight,
        focus_id=args.focus_id,
        run_tag=args.model,
        seed=args.seed,
    )

    print(f"[FINISHED] {args.model} completado exitosamente.")
    print(f"[OUT] Resultados finales: {run_dir}/summary.json")

# ==============================================================
# === VISUALIZACIÓN COMPARATIVA ================================
# ==============================================================

def visualize_prediction(
    model: nn.Module,
    data_dir: Path,
    checkpoint_path: Path,
    sample_index: int = 0,
    num_classes: int = 26,
    focus_id: int = 21,
    out_dir: Path = None,
    device: str = "cuda"
):
    """
    Visualiza una muestra puntual de test: comparación entre Ground Truth y Predicción.
    Genera:
      - plot_gt.html : nube coloreada por etiquetas reales
      - plot_pred.html : nube coloreada por predicciones
      - plot_diff.html : nube coloreada por tipo de acierto/error
    """
    import plotly.graph_objects as go
    import plotly.express as px

    # -------------------- carga datos --------------------
    X = np.load(data_dir / "X_test.npz")["X"]
    Y = np.load(data_dir / "Y_test.npz")["Y"]
    num_classes = int(np.max(Y) + 1)
    pts = torch.tensor(X[sample_index:sample_index + 1], dtype=torch.float32).to(device)
    gt = torch.tensor(Y[sample_index], dtype=torch.int64).to(device)

    # -------------------- modelo --------------------
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    with torch.no_grad():
        pred_logits = model(pts)
        preds = pred_logits.argmax(dim=-1).cpu().numpy()[0]

    gt_np = gt.cpu().numpy()
    pts_np = pts.cpu().numpy()[0, :, :3]

    # -------------------- colores --------------------
    palette = px.colors.qualitative.Dark24
    colors = np.array([px.colors.hex_to_rgb(c) for c in palette]) / 255.0
    cmap = np.vstack([colors, np.random.rand(max(0, num_classes - len(colors)), 3)])  # auto-extiende

    def color_by_label(labels):
        return np.array([cmap[int(i) % cmap.shape[0]] for i in labels])

    # -------------------- gráficos --------------------
    def make_plot(points, labels, title):
        c = color_by_label(labels)
        return go.Figure(
            data=[go.Scatter3d(
                x=points[:, 0], y=points[:, 1], z=points[:, 2],
                mode="markers",
                marker=dict(size=2, color=[f"rgb({r*255:.0f},{g*255:.0f},{b*255:.0f})" for r, g, b in c])
            )],
            layout=go.Layout(title=title, scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)))
        )

    fig_gt = make_plot(pts_np, gt_np, f"Ground Truth (muestra {sample_index})")
    fig_pred = make_plot(pts_np, preds, f"Predicción (muestra {sample_index})")

    # -------------------- comparación: correcto / FP / FN --------------------
    correct = preds == gt_np
    false_pos = (preds == focus_id) & (gt_np != focus_id)
    false_neg = (preds != focus_id) & (gt_np == focus_id)

    diff_color = np.zeros((len(preds), 3))
    diff_color[correct] = [0.2, 0.8, 0.2]     # verde
    diff_color[false_pos] = [0.9, 0.1, 0.1]   # rojo
    diff_color[false_neg] = [0.1, 0.1, 0.9]   # azul

    fig_diff = go.Figure(
        data=[go.Scatter3d(
            x=pts_np[:, 0], y=pts_np[:, 1], z=pts_np[:, 2],
            mode="markers",
            marker=dict(size=2, color=[f"rgb({r*255:.0f},{g*255:.0f},{b*255:.0f})" for r, g, b in diff_color]),
        )],
        layout=go.Layout(
            title=f"Comparación Correcto/FP/FN (muestra {sample_index})",
            scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False))
        )
    )

    # -------------------- guardado --------------------
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fig_gt.write_html(out_dir / "plot_gt.html")
        fig_pred.write_html(out_dir / "plot_pred.html")
        fig_diff.write_html(out_dir / "plot_diff.html")
        print(f"[PLOTS] Guardados en {out_dir}")
    else:
        fig_gt.show()
        fig_pred.show()
        fig_diff.show()

if __name__ == "__main__":
    main()

