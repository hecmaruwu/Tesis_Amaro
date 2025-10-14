#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entrenamiento de segmentación por puntos MULTICLASE con modelos paper-fiel:
- PointNet (Qi et al., 2017)
- PointNet++ (Qi et al., 2017)
- DilatedToothSegNet (bloques dilatados)
- Transformer 3D (PE sinusoidales)

Compatibilidad con tu flujo:
Lee X_*.npz / Y_*.npz con claves "X" y "Y"
Normaliza cada nube a esfera unitaria.
Salida: out_dir/tag/<modelo>/...
"""

import os, json, time, csv, argparse
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

# torchmetrics opcional
try:
    from torchmetrics.classification import (
        MulticlassAccuracy, MulticlassPrecision, MulticlassRecall,
        MulticlassF1Score, MulticlassJaccardIndex
    )
    HAS_TORCHMETRICS = True
except Exception:
    HAS_TORCHMETRICS = False

# torch-cluster opcional
try:
    from torch_cluster import fps, radius
    HAS_CLUSTER = True
except Exception:
    HAS_CLUSTER = False

from einops import rearrange

# =====================================================================================
# --------------------------- UTILIDADES BÁSICAS --------------------------------------
# =====================================================================================

def get_device(cuda_index: Optional[int] = None) -> torch.device:
    """Selecciona el dispositivo CUDA disponible o CPU."""
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        if cuda_index is not None:
            if cuda_index < 0 or cuda_index >= n:
                print(f"[WARN] cuda:{cuda_index} no existe (GPUs={n}), uso cuda:0.")
                return torch.device("cuda:0")
            return torch.device(f"cuda:{cuda_index}")
        return torch.device("cuda:0")
    return torch.device("cpu")


def seed_everything(seed: int = 42):
    """Reproducibilidad total."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_cloud(x: torch.Tensor) -> torch.Tensor:
    """
    Normaliza cada nube de puntos a una esfera unitaria centrada en el origen.
    x: (B, P, 3) o (P, 3)
    """
    mean = x.mean(dim=-2, keepdim=True)
    x = x - mean
    r = torch.linalg.vector_norm(x, dim=-1, keepdim=True).amax(dim=-2, keepdim=True)
    return x / (r + 1e-6)


def load_splits(data_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Carga los splits .npz de entrenamiento, validación y test."""
    req = ["X_train.npz", "Y_train.npz", "X_val.npz", "Y_val.npz", "X_test.npz", "Y_test.npz"]
    miss = [r for r in req if not (data_path / r).exists()]
    if miss:
        raise FileNotFoundError(f"Faltan archivos en {data_path}:\n  " + "\n  ".join(miss))

    Xtr = np.load(data_path / "X_train.npz")["X"]
    Ytr = np.load(data_path / "Y_train.npz")["Y"]
    Xva = np.load(data_path / "X_val.npz")["X"]
    Yva = np.load(data_path / "Y_val.npz")["Y"]
    Xte = np.load(data_path / "X_test.npz")["X"]
    Yte = np.load(data_path / "Y_test.npz")["Y"]

    ncls = int(max(Ytr.max(), Yva.max(), Yte.max()) + 1)
    print(f"[DATA] Xtr:{Xtr.shape} Xva:{Xva.shape} Xte:{Xte.shape} ncls={ncls}")
    return Xtr, Ytr, Xva, Yva, Xte, Yte, ncls


def make_dataloaders(Xtr, Ytr, Xva, Yva, Xte, Yte, batch_size=8):
    """Crea los DataLoaders PyTorch."""
    tr_ds = TensorDataset(torch.from_numpy(Xtr).float(), torch.from_numpy(Ytr).long())
    va_ds = TensorDataset(torch.from_numpy(Xva).float(), torch.from_numpy(Yva).long())
    te_ds = TensorDataset(torch.from_numpy(Xte).float(), torch.from_numpy(Yte).long())
    return (
        DataLoader(tr_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True),
        DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True),
        DataLoader(te_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True),
    )

# =====================================================================================
# -------------------------------- MÉTRICAS -------------------------------------------
# =====================================================================================

class MetricsBundle:
    """Conjunto de métricas macro (acc, prec, rec, f1, iou). Usa torchmetrics si está disponible."""
    def __init__(self, num_classes: int, device: torch.device):
        self.num_classes = num_classes
        self.device = device
        self.reset()

        if HAS_TORCHMETRICS:
            self.tm = {
                "acc":  MulticlassAccuracy(num_classes=num_classes).to(device),
                "prec": MulticlassPrecision(num_classes=num_classes, average="macro").to(device),
                "rec":  MulticlassRecall(num_classes=num_classes, average="macro").to(device),
                "f1":   MulticlassF1Score(num_classes=num_classes, average="macro").to(device),
                "iou":  MulticlassJaccardIndex(num_classes=num_classes, average="macro").to(device),
            }
        else:
            self.tm = None

    def reset(self):
        self.cm = torch.zeros((self.num_classes, self.num_classes), dtype=torch.long)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, y: torch.Tensor):
        preds = logits.argmax(dim=-1)
        if self.tm is not None:
            p = preds.reshape(-1)
            t = y.reshape(-1)
            for m in self.tm.values():
                m.update(p, t)
        else:
            p = preds.reshape(-1).cpu()
            t = y.reshape(-1).cpu()
            for ti, pi in zip(t.tolist(), p.tolist()):
                if 0 <= ti < self.num_classes and 0 <= pi < self.num_classes:
                    self.cm[ti, pi] += 1

    def compute(self) -> Dict[str, float]:
        if self.tm is not None:
            res = {k: float(m.compute().item()) for k, m in self.tm.items()}
            for m in self.tm.values():
                m.reset()
            return res

        cm = self.cm.float()
        tp = torch.diag(cm)
        gt = cm.sum(dim=1)
        pr = cm.sum(dim=0)
        prec = torch.nan_to_num(tp / pr, nan=0.0)
        rec = torch.nan_to_num(tp / gt, nan=0.0)
        f1 = torch.nan_to_num(2 * prec * rec / (prec + rec), nan=0.0)
        iou = torch.nan_to_num(tp / (gt + pr - tp), nan=0.0)
        acc = (tp.sum() / cm.sum()) if cm.sum() > 0 else torch.tensor(0.0)

        self.reset()
        return {
            "acc": float(acc.item()),
            "prec": float(prec.mean().item()),
            "rec": float(rec.mean().item()),
            "f1": float(f1.mean().item()),
            "iou": float(iou.mean().item()),
        }

# =====================================================================================
# -------------------------------  PÉRDIDAS  ------------------------------------------
# =====================================================================================

class FocalLoss(nn.Module):
    """Implementación estable de Focal Loss para clasificación multiclase."""
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        """
        logits: (B*P, C)
        target: (B*P,)
        """
        logp = F.log_softmax(logits, dim=-1)
        p = logp.exp()
        pt = torch.gather(p, dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
        loss = -((1 - pt) ** self.gamma) * torch.gather(logp, -1, target.unsqueeze(-1)).squeeze(-1)

        if self.alpha is not None:
            a = self.alpha.to(logits.device)
            wt = torch.gather(a, 0, target.view(-1)).view_as(target)
            loss = loss * wt
        return loss.mean()


# =====================================================================================
# -------------------------------  POINTNET  ------------------------------------------
# =====================================================================================

class OrthogonalRegularizer:
    """||I − AAᵀ||² para penalizar desviación de ortogonalidad en T-Nets."""
    def __init__(self, K: int, strength: float = 0.001):
        self.K = K
        self.strength = strength

    def __call__(self, A: torch.Tensor) -> torch.Tensor:
        I = torch.eye(self.K, device=A.device).unsqueeze(0).expand_as(A)
        return self.strength * ((I - torch.bmm(A, A.transpose(1, 2))).pow(2).sum(dim=(1, 2))).mean()


class TNet(nn.Module):
    """Subred transformadora (T-Net) del PointNet original."""
    def __init__(self, K=3):
        super().__init__()
        self.K = K
        self.conv1 = nn.Conv1d(K, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, K * K)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):  # x: (B, K, P)
        B = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2)[0]                 # (B, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x).view(B, self.K, self.K)
        I = torch.eye(self.K, device=x.device).unsqueeze(0).expand(B, -1, -1)
        return x + I


class PointNetSeg(nn.Module):
    """PointNet original adaptado para segmentación por punto."""
    def __init__(self, num_classes=10, dropout=0.5, reg_strength=0.001, base_channels=64):
        super().__init__()
        C = base_channels
        # T-Net de entrada (K=3)
        self.input_tnet = TNet(K=3)
        self.feat_conv1 = nn.Conv1d(3, C, 1)
        self.bn1 = nn.BatchNorm1d(C)
        self.feat_conv2 = nn.Conv1d(C, C, 1)
        self.bn2 = nn.BatchNorm1d(C)

        # T-Net de características (K=C)
        self.feat_tnet = TNet(K=C)
        self.feat_conv3 = nn.Conv1d(C, C, 1)
        self.bn3 = nn.BatchNorm1d(C)
        self.feat_conv4 = nn.Conv1d(C, 2 * C, 1)
        self.bn4 = nn.BatchNorm1d(2 * C)
        self.global_conv = nn.Conv1d(2 * C, 16 * C, 1)
        self.bn_g = nn.BatchNorm1d(16 * C)

        # Cabezal de segmentación
        self.conv6 = nn.Conv1d(16 * C + 2 * C, 8 * C, 1)
        self.bn6 = nn.BatchNorm1d(8 * C)
        self.conv7 = nn.Conv1d(8 * C, 4 * C, 1)
        self.bn7 = nn.BatchNorm1d(4 * C)
        self.conv8 = nn.Conv1d(4 * C, 2 * C, 1)
        self.dp = nn.Dropout(dropout)
        self.conv9 = nn.Conv1d(2 * C, num_classes, 1)

        self.reg_in = OrthogonalRegularizer(3, reg_strength)
        self.reg_feat = OrthogonalRegularizer(C, reg_strength)

    def forward(self, xyz):  # xyz: (B, P, 3)
        # 🔧 CORRECCIÓN CLAVE → asegurar orden (B, C, P)
        if xyz.dim() != 3 or xyz.size(-1) != 3:
            raise ValueError(f"Esperaba (B, P, 3), recibí {xyz.shape}")
        x = xyz.permute(0, 2, 1)              # (B, 3, P) ← CORREGIDO

        # --- flujo PointNet original ---
        T1 = self.input_tnet(x)
        x = torch.bmm(T1, x)
        x = F.relu(self.bn1(self.feat_conv1(x)))
        x = F.relu(self.bn2(self.feat_conv2(x)))

        T2 = self.feat_tnet(x)
        x = torch.bmm(T2, x)
        x = F.relu(self.bn3(self.feat_conv3(x)))
        x = F.relu(self.bn4(self.feat_conv4(x)))

        g = F.relu(self.bn_g(self.global_conv(x)))
        gmax = torch.max(g, 2, keepdim=True)[0]
        gcat = torch.cat([x, gmax.expand(-1, -1, x.size(2))], dim=1)

        y = F.relu(self.bn6(self.conv6(gcat)))
        y = F.relu(self.bn7(self.conv7(y)))
        y = self.dp(F.relu(self.conv8(y)))
        y = self.conv9(y)

        y = y.transpose(2, 1)                 # (B, P, num_classes)
        reg = self.reg_in(T1) + self.reg_feat(T2)
        return y, reg

# =====================================================================================
# ------------------------------  PointNet++  (SSG)  ----------------------------------
# =====================================================================================

@torch.no_grad()
def three_nn_interp(xyz1, xyz2, feats2, k=3):
    """
    Interpolación 3-NN por inverso de la distancia.
    xyz1:  (B, N1, 3)  destino (más denso)
    xyz2:  (B, N2, 3)  fuente  (más escaso)
    feats2:(B, C2, N2) características en los puntos fuente
    return:(B, C2, N1)
    """
    B, N1, _ = xyz1.shape
    _, N2, _ = xyz2.shape
    d = torch.cdist(xyz1, xyz2)                        # (B, N1, N2)
    d = torch.clamp(d, min=1e-8)
    k = min(k, N2)
    knn_d, knn_i = torch.topk(d, k=k, dim=-1, largest=False)  # (B, N1, k)
    w = 1.0 / knn_d
    w = w / w.sum(dim=-1, keepdim=True)                # (B, N1, k)

    # feats2: (B, C2, N2) -> (B, N2, C2) para gather
    f2 = feats2.transpose(1, 2)                        # (B, N2, C2)
    Bidx = torch.arange(B, device=xyz1.device)[:, None, None]
    neigh = f2[Bidx, knn_i]                            # (B, N1, k, C2)
    out = (w[..., None] * neigh).sum(dim=2)            # (B, N1, C2)
    return out.transpose(1, 2)                         # (B, C2, N1)


class MLP1d(nn.Module):
    """Bloque MLP 1D: Conv1d(1x1)+BN+ReLU apilados."""
    def __init__(self, in_c, mlp):
        super().__init__()
        layers, c = [], in_c
        for oc in mlp:
            layers += [nn.Conv1d(c, oc, 1), nn.BatchNorm1d(oc), nn.ReLU(inplace=True)]
            c = oc
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # x: (B, C, L)
        return self.net(x)


class SA_Layer(nn.Module):
    """
    Set Abstraction (SSG) con FPS+radius (si torch-cluster disponible) o kNN fallback.
    Devuelve centros (B, M, 3) y features (B, C_out, M).
    """
    def __init__(self, radius_r, nsample, in_ch, mlp):
        super().__init__()
        self.r = float(radius_r)
        self.nsample = int(nsample)
        self.mlp = MLP1d(in_ch + 3, mlp)   # concatenamos coords locales (3) + feats
        self.out_ch = mlp[-1]

    def forward(self, xyz, feats):
        """
        xyz:   (B, P, 3)
        feats: (B, C, P) o None
        """
        B, P, _ = xyz.shape

        # ---- Selección de centros (FPS si está disponible; si no, muestreo regular)
        if HAS_CLUSTER:
            idx_center_list = []
            for b in range(B):
                # torch_cluster.fps espera (N,3)
                idx = fps(xyz[b], ratio=0.25, random_start=False)  # ~ P/4
                idx_center_list.append(idx)
            M = min(x.size(0) for x in idx_center_list)
            idx_center = torch.stack([x[:M] for x in idx_center_list], dim=0)  # (B, M)
        else:
            M = max(1, P // 4)
            idx_center = torch.linspace(0, P - 1, M, device=xyz.device, dtype=torch.long)\
                          .unsqueeze(0).repeat(B, 1)  # (B, M)

        centers = torch.gather(xyz, 1, idx_center.unsqueeze(-1).expand(-1, -1, 3))  # (B, M, 3)

        new_feats = []
        for b in range(B):
            if HAS_CLUSTER:
                # Vecindarios por radio centrados en cada centro
                ind = radius(xyz[b], centers[b], self.r, max_num_neighbors=self.nsample)  # (2, E)
                row, col = ind  # row: índice del centro, col: idx del punto en xyz[b]

                gathered = []
                for i in range(centers.size(1)):
                    mask = (row == i)
                    idxs = col[mask][:self.nsample]
                    if idxs.numel() < self.nsample:
                        pad = idxs.new_full((self.nsample - idxs.numel(),),
                                            idxs[0] if idxs.numel() > 0 else 0)
                        idxs = torch.cat([idxs, pad], dim=0)  # (nsample,)

                    # coords locales
                    xyz_local = (xyz[b, idxs] - centers[b, i]).T.unsqueeze(0)  # (1, 3, nsample)

                    if feats is not None:
                        f_local = feats[b, :, idxs].unsqueeze(0)               # (1, C, nsample)
                        cat = torch.cat([xyz_local, f_local], dim=1)           # (1, 3+C, nsample)
                    else:
                        cat = xyz_local                                        # (1, 3, nsample)

                    out_i = self.mlp(cat)               # (1, C_out, nsample)
                    out_i = torch.max(out_i, dim=-1)[0] # (1, C_out)
                    gathered.append(out_i)

                out = torch.cat(gathered, dim=0).T.unsqueeze(0)  # (1, C_out, M)
                new_feats.append(out)

            else:
                # Fallback kNN sobre todos los puntos (usa top-k por distancia)
                M = centers.size(1)
                d = torch.cdist(centers[b], xyz[b])  # (M, P)
                k = min(self.nsample, P)
                knn = torch.topk(d, k=k, largest=False, dim=-1).indices  # (M, k)

                # xyz_local: (M, k, 3) -> (M, 3, k)
                xyz_local = (xyz[b][knn] - centers[b].unsqueeze(1))      # (M, k, 3)
                xyz_local = xyz_local.permute(0, 2, 1)                   # (M, 3, k)

                if feats is not None:
                    f_local = feats[b][:, knn]      # (C, M, k)
                    # Apilamos M como "batch" para pasar por Conv1d: (M, 3+C, k)
                    cat = torch.cat([xyz_local, f_local.permute(1, 0, 2)], dim=1)
                else:
                    cat = xyz_local  # (M, 3, k)

                out = self.mlp(cat)                                  # (M, C_out, k)
                out = torch.max(out, dim=-1, keepdim=True)[0]         # (M, C_out, 1)
                out = out.permute(2, 1, 0)                            # (1, C_out, M)
                new_feats.append(out)

        new_feats = torch.cat(new_feats, dim=0)  # (B, C_out, M)
        return centers, new_feats


class FP_Layer(nn.Module):
    """Feature Propagation: interpola de (xyz2, feats2) -> (xyz1, *) y pasa MLP."""
    def __init__(self, in_ch, mlp):
        super().__init__()
        self.mlp = MLP1d(in_ch, mlp)
        self.out_ch = mlp[-1]

    def forward(self, xyz1, xyz2, feats1, feats2):
        """
        xyz1:  (B, N, 3) denso (destino)
        xyz2:  (B, S, 3) escaso (fuente)
        feats1:(B, C1, N) skip (puede ser None)
        feats2:(B, C2, S) a interpolar
        """
        interp = three_nn_interp(xyz1, xyz2, feats2)   # (B, C2, N)
        cat = torch.cat([interp, feats1], dim=1) if feats1 is not None else interp  # (B, C1+C2, N)
        return self.mlp(cat)                           # (B, C_out, N)


class PointNet2Seg(nn.Module):
    """PointNet++ SSG (3 SA + 3 FP) con cabeza de segmentación."""
    def __init__(self, num_classes=10, dropout=0.5):
        super().__init__()
        # Bloques SA (r, nsample, in_ch, mlp_outs)
        self.sa1 = SA_Layer(radius_r=0.2, nsample=32,  in_ch=0,   mlp=[64, 64, 128])
        self.sa2 = SA_Layer(radius_r=0.4, nsample=64,  in_ch=128, mlp=[128, 128, 256])
        self.sa3 = SA_Layer(radius_r=0.8, nsample=128, in_ch=256, mlp=[256, 512, 1024])

        # Bloques FP (propagación hacia resolución original)
        self.fp3 = FP_Layer(in_ch=1024 + 256, mlp=[256, 256])
        self.fp2 = FP_Layer(in_ch=256 + 128,  mlp=[256, 128])
        self.fp1 = FP_Layer(in_ch=128 + 0,    mlp=[128, 128, 128])

        # Cabeza de segmentación
        self.head = nn.Sequential(
            nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Conv1d(128, num_classes, 1)
        )

    def forward(self, xyz):  # (B, P, 3)
        if xyz.dim() != 3 or xyz.size(-1) != 3:
            raise ValueError(f"Esperaba (B, P, 3) y recibí {xyz.shape}")

        feats0 = None
        # SA: bajamos resolución con pooling por vecindario
        l1_xyz, l1_feats = self.sa1(xyz, feats0)        # (B, M1, 3), (B, 128, M1)
        l2_xyz, l2_feats = self.sa2(l1_xyz, l1_feats)   # (B, M2, 3), (B, 256, M2)
        l3_xyz, l3_feats = self.sa3(l2_xyz, l2_feats)   # (B, M3, 3), (B, 1024, M3)

        # FP: subimos e integramos skips
        l2_new = self.fp3(l2_xyz, l3_xyz, l2_feats, l3_feats)  # (B, 256, M2)
        l1_new = self.fp2(l1_xyz, l2_xyz, l1_feats, l2_new)    # (B, 128, M1)
        l0_new = self.fp1(xyz,    l1_xyz, feats0,   l1_new)    # (B, 128, P)

        out = self.head(l0_new).transpose(2, 1)  # (B, P, num_classes)
        reg = torch.tensor(0.0, device=out.device)
        return out, reg

# =====================================================================================
# ---------------------------  DilatedToothSegNet  ------------------------------------
# =====================================================================================

class DilatedToothSegNet(nn.Module):
    """Backbone 1D dilatado con skip-concat y cabeza de segmentación."""
    def __init__(self, num_classes=10, base=64, dropout=0.5):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv1d(3, base, 1, dilation=1), nn.BatchNorm1d(base), nn.ReLU(True))
        self.enc2 = nn.Sequential(nn.Conv1d(base, base*2, 1, dilation=2), nn.BatchNorm1d(base*2), nn.ReLU(True))
        self.enc3 = nn.Sequential(nn.Conv1d(base*2, base*4, 1, dilation=3), nn.BatchNorm1d(base*4), nn.ReLU(True))
        self.enc4 = nn.Sequential(nn.Conv1d(base*4, base*8, 1, dilation=4), nn.BatchNorm1d(base*8), nn.ReLU(True))

        self.head = nn.Sequential(
            nn.Conv1d(base*8 + base*4 + base*2 + base, base*4, 1),
            nn.BatchNorm1d(base*4), nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Conv1d(base*4, num_classes, 1)
        )

    def forward(self, xyz):  # (B, P, 3)
        if xyz.dim() != 3 or xyz.size(-1) != 3:
            raise ValueError(f"Esperaba (B, P, 3) y recibí {xyz.shape}")
        x = xyz.permute(0, 2, 1)  # (B, 3, P)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        cat = torch.cat([e1, e2, e3, e4], dim=1)
        out = self.head(cat).transpose(2, 1)  # (B, P, C)
        reg = torch.tensor(0.0, device=out.device)
        return out, reg


# =====================================================================================
# -------------------------------  Transformer 3D  ------------------------------------
# =====================================================================================

class PositionalEncoding(nn.Module):
    """PE sinusoidal estándar (se suma a las embeddings)."""
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)  # (max_len, d_model)

    def forward(self, x):  # (B, P, D)
        P = x.size(1)
        return x + self.pe[:P].unsqueeze(0).to(x.device)


class Transformer3DSeg(nn.Module):
    """Transformer encoder puro sobre XYZ proyectado, con cabeza de segmentación."""
    def __init__(self, num_classes=10, dim=128, heads=8, depth=6, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(3, dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim*4,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.pe = PositionalEncoding(dim)
        self.head = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes)
        )

    def forward(self, xyz):  # (B, P, 3)
        if xyz.dim() != 3 or xyz.size(-1) != 3:
            raise ValueError(f"Esperaba (B, P, 3) y recibí {xyz.shape}")
        x = self.proj(xyz)
        x = self.pe(x)
        x = self.encoder(x)          # (B, P, D)
        out = self.head(x)           # (B, P, C)
        reg = torch.tensor(0.0, device=out.device)
        return out, reg


# =====================================================================================
# ------------------------------  ENTRENAMIENTO  --------------------------------------
# =====================================================================================

def make_criterion(args, num_classes: int, device: torch.device):
    """Construye la loss (CrossEntropy o Focal) con pesos opcionales por clase."""
    weights = None
    if args.class_weights_json and Path(args.class_weights_json).exists():
        wj = json.loads(Path(args.class_weights_json).read_text())
        weights = torch.ones(num_classes, dtype=torch.float32)
        for k, v in wj.items():
            k = int(k)
            if 0 <= k < num_classes:
                weights[k] = float(v)
        weights = weights.to(device)

    if args.use_focal:
        return FocalLoss(gamma=2.0, alpha=weights)
    return nn.CrossEntropyLoss(weight=weights)


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    """Entrena una época completa."""
    model.train()
    running = 0.0
    for xb, yb in loader:
        xb = normalize_cloud(xb.to(device, non_blocking=True))  # (B, P, 3)
        yb = yb.to(device, non_blocking=True)                   # (B, P)

        optimizer.zero_grad(set_to_none=True)
        if scaler:
            with torch.cuda.amp.autocast(True):
                logits, reg = model(xb)  # (B, P, C)
                loss = criterion(logits.reshape(-1, logits.size(-1)), yb.reshape(-1)) + reg
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, reg = model(xb)
            loss = criterion(logits.reshape(-1, logits.size(-1)), yb.reshape(-1)) + reg
            loss.backward()
            optimizer.step()
        running += float(loss.item())
    return running / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, criterion, device, metrics: MetricsBundle):
    """Evalúa sobre un conjunto (val/test)."""
    model.eval()
    total = 0.0
    for xb, yb in loader:
        xb = normalize_cloud(xb.to(device, non_blocking=True))
        yb = yb.to(device, non_blocking=True)
        logits, reg = model(xb)
        loss = criterion(logits.reshape(-1, logits.size(-1)), yb.reshape(-1)) + reg
        total += float(loss.item())
        metrics.update(logits, yb)
    avg = total / max(1, len(loader))
    m = metrics.compute()
    return avg, m


def build_model(name: str, num_classes: int, args):
    """Devuelve el modelo según el nombre."""
    name = name.lower()
    if name == "pointnet":
        return PointNetSeg(num_classes=num_classes, dropout=args.dropout,
                           reg_strength=0.001, base_channels=args.base_channels)
    if name == "pointnetpp":
        return PointNet2Seg(num_classes=num_classes, dropout=args.dropout)
    if name == "dilated":
        return DilatedToothSegNet(num_classes=num_classes,
                                  base=args.base_channels, dropout=args.dropout)
    if name == "transformer":
        return Transformer3DSeg(num_classes=num_classes, dim=args.tr_dim,
                                heads=args.tr_heads, depth=args.tr_depth,
                                dropout=args.dropout)
    raise ValueError(f"Modelo no soportado: {name}")


def _safe_json(obj):
    """Convierte objetos (Path, numpy types, etc.) a tipos serializables."""
    import numpy as np
    if isinstance(obj, (np.integer, np.int32, np.int64)): return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)): return float(obj)
    if isinstance(obj, Path): return str(obj)
    if isinstance(obj, dict): return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_safe_json(v) for v in obj]
    return obj


def train_model(model_name: str, args, device, dls, num_classes: int, run_root: Path):
    """Entrena un modelo específico y guarda historial, pesos y métricas."""
    run_root.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_root / "tb"))
    history = {"train_loss": [], "val_loss": [], "val_acc": [],
               "val_prec": [], "val_rec": [], "val_f1": [], "val_iou": []}

    model = build_model(model_name, num_classes, args).to(device)

    # Multi-GPU opcional
    if args.multi_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = make_criterion(args, num_classes, device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=args.lr_patience,
        factor=args.lr_factor, min_lr=args.min_lr, verbose=True
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val = float("inf")
    best_path = run_root / "checkpoints" / "best.pt"
    (run_root / "checkpoints").mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    metrics = MetricsBundle(num_classes, device)

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, dls["train"], optimizer, criterion, device, scaler)
        val_loss, val_m = evaluate(model, dls["val"], criterion, device, metrics)

        scheduler.step(val_loss)

        # Logs TensorBoard
        writer.add_scalar("loss/train", tr_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        for k, v in val_m.items():
            writer.add_scalar(f"val/{k}", v, epoch)

        # Registro en memoria
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_m.get("acc", 0.0))
        history["val_prec"].append(val_m.get("prec", 0.0))
        history["val_rec"].append(val_m.get("rec", 0.0))
        history["val_f1"].append(val_m.get("f1", 0.0))
        history["val_iou"].append(val_m.get("iou", 0.0))

        print(f"[{model_name}] Epoch {epoch}/{args.epochs} | tr {tr_loss:.4f} | va {val_loss:.4f} | "
              f"acc {val_m.get('acc',0):.4f} f1 {val_m.get('f1',0):.4f} iou {val_m.get('iou',0):.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "epoch": epoch}, best_path)
            print(f"  ↳ [OK] best checkpoint @ {best_path}")

    # Tiempo total
    train_time = round(time.time() - t0, 2)

    # Guardar último modelo
    final_path = run_root / "final_model.pt"
    torch.save({"model": model.state_dict(), "epoch": args.epochs}, final_path)

    # Cargar mejor checkpoint para evaluar en test
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_loss, test_m = evaluate(model, dls["test"], criterion, device,
                                 MetricsBundle(num_classes, device))

    # Serialización segura
    meta = {
        "model": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "seed": int(args.seed),
        "device": str(device),
        "multi_gpu": bool(args.multi_gpu),
        "paper_params": bool(args.paper_params),
        "train_time_sec": float(train_time),
        "best_val_loss": float(best_val)
    }

    (run_root / "config.json").write_text(
        json.dumps({"args": _safe_json(vars(args)), "meta": _safe_json(meta)}, indent=2),
        encoding="utf-8"
    )
    (run_root / "history.json").write_text(
        json.dumps(_safe_json(history), indent=2), encoding="utf-8"
    )
    (run_root / "metrics_val_test.json").write_text(
        json.dumps(_safe_json({"test_loss": test_loss, **test_m}), indent=2),
        encoding="utf-8"
    )

    writer.close()
    return {"test_loss": test_loss, **test_m, **meta}


# =====================================================================================
# ----------------------------------  MAIN / CLI  -------------------------------------
# =====================================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--tag", required=True, type=str)

    ap.add_argument("--model", default="all", choices=["all","pointnet","pointnetpp","dilated","transformer"])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--base_channels", type=int, default=64)

    ap.add_argument("--tr_dim", type=int, default=128)
    ap.add_argument("--tr_heads", type=int, default=8)
    ap.add_argument("--tr_depth", type=int, default=6)

    ap.add_argument("--lr_patience", type=int, default=20, help="Paciencia ReduceLROnPlateau.")
    ap.add_argument("--lr_factor", type=float, default=0.5, help="Factor de reducción del LR.")
    ap.add_argument("--min_lr", type=float, default=1e-6, help="LR mínimo permitido.")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cuda", type=int, default=None)
    ap.add_argument("--multi_gpu", action="store_true")
    ap.add_argument("--paper_params", action="store_true")

    ap.add_argument("--use_focal", action="store_true")
    ap.add_argument("--class_weights_json", default=None)

    args = ap.parse_args()

    # Ajustes "paper-like"
    if args.paper_params:
        if args.model in ["pointnet","all"]:
            args.base_channels = 64
            args.dropout = 0.5
        if args.model in ["pointnetpp","all"]:
            args.dropout = 0.5
        if args.model in ["transformer","all"]:
            args.tr_dim = 128; args.tr_heads = 8; args.tr_depth = 6

    seed_everything(args.seed)
    device = get_device(args.cuda)
    print(f"[INFO] Dispositivo: {device}  (GPUs visibles: {torch.cuda.device_count()})")

    Xtr, Ytr, Xva, Yva, Xte, Yte, num_classes = load_splits(args.data_path)
    dtr, dva, dte = make_dataloaders(Xtr, Ytr, Xva, Yva, Xte, Yte, args.batch_size)
    dls = {"train": dtr, "val": dva, "test": dte}

    out_root = args.out_dir / args.tag
    out_root.mkdir(parents=True, exist_ok=True)

    models = [args.model] if args.model != "all" else ["pointnet","pointnetpp","dilated","transformer"]
    summary_rows = []

    for name in models:
        print(f"\n[TRAIN] {name}")
        run_dir = out_root / name
        res = train_model(name, args, device, dls, num_classes, run_dir)
        row = {
            "model": name, "timestamp": res["timestamp"], "epochs": res["epochs"], "batch_size": res["batch_size"],
            "lr": res["lr"], "seed": res["seed"], "device": res["device"], "train_time_sec": res["train_time_sec"],
            "test_loss": res["test_loss"], "acc": res.get("acc", 0.0), "prec_macro": res.get("prec", 0.0),
            "rec_macro": res.get("rec", 0.0), "f1_macro": res.get("f1", 0.0), "miou_macro": res.get("iou", 0.0)
        }
        summary_rows.append(row)

    # CSV resumen global
    csv_path = out_root / "summary_all_models.csv"
    if summary_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader(); w.writerows(summary_rows)
        print(f"[CSV] {csv_path}")


if __name__ == "__main__":
    main()

