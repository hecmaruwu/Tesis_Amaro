#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_models_paperlike_loss_balanceada.py
--------------------------------------------------------------
Versión con pérdida combinada (CrossEntropy + DiceLoss) y pesos
balanceados según papers (1 / log(1.2 + freq)).
Basada en ToothFormer (2024) y DilatedToothSegNet (2023).
--------------------------------------------------------------
Autor: Adaptado por ChatGPT (GPT-5)
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_device(x, device: torch.device):
    if isinstance(x, (tuple, list)):
        return [to_device(t, device) for t in x]
    return x.to(device, non_blocking=True)


def sanitize_tensor(t: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)


def normalize_cloud(x: torch.Tensor) -> torch.Tensor:
    c = x.mean(dim=1, keepdim=True)
    x = x - c
    r = (x.pow(2).sum(-1).sqrt()).max(dim=1, keepdim=True)[0].unsqueeze(-1)
    return x / (r + 1e-8)


def save_json(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def plot_curves(history: Dict[str, List[float]], out_dir: Path, model_name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = ["loss", "acc", "f1", "iou", "d21_f1"]
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
# === Dataset y DataLoaders ====================================
# ==============================================================

class CloudDataset(Dataset):
    def __init__(self, X_path: Path, Y_path: Path):
        self.X = np.load(X_path)["X"].astype(np.float32)
        self.Y = np.load(Y_path)["Y"].astype(np.int64)
        assert self.X.shape[0] == self.Y.shape[0], f"Inconsistencia entre {X_path} y {Y_path}"
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.Y[idx])


def make_loaders(data_dir: Path, batch_size=8, num_workers=4) -> Dict[str, DataLoader]:
    data_dir = Path(data_dir)
    loaders = {}
    for split in ["train", "val", "test"]:
        Xp, Yp = data_dir/f"X_{split}.npz", data_dir/f"Y_{split}.npz"
        if not (Xp.exists() and Yp.exists()):
            raise FileNotFoundError(f"Faltan archivos para {split}")
        ds = CloudDataset(Xp, Yp)
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=(split=="train"),
            num_workers=num_workers, pin_memory=True, drop_last=False)
    return loaders

# ==============================================================
# === Métricas ================================================
# ==============================================================

@torch.no_grad()
def confusion_matrix(logits, y_true, num_classes):
    preds = logits.argmax(dim=-1)
    t = y_true.view(-1); p = preds.view(-1)
    valid = (t>=0)&(t<num_classes)
    t,p = t[valid],p[valid]
    idx = t*num_classes + p
    cm = torch.bincount(idx, minlength=num_classes**2).reshape(num_classes,num_classes)
    return cm


def macro_from_cm(cm):
    cm = cm.float()
    tp = torch.diag(cm)
    gt = cm.sum(1).clamp_min(1e-8)
    pd = cm.sum(0).clamp_min(1e-8)
    acc = (tp.sum()/cm.sum().clamp_min(1e-8)).item()
    f1 = torch.mean(2*tp/(gt+pd)).item()
    iou = torch.mean(tp/(gt+pd-tp).clamp_min(1e-8)).item()
    return {"acc":acc,"f1":f1,"iou":iou}


def d21_metrics(logits,y_true,cls_id):
    preds = logits.argmax(dim=-1).view(-1)
    t = y_true.view(-1)
    tp = ((preds==cls_id)&(t==cls_id)).sum().float()
    fp = ((preds==cls_id)&(t!=cls_id)).sum().float()
    fn = ((preds!=cls_id)&(t==cls_id)).sum().float()
    tn = ((preds!=cls_id)&(t!=cls_id)).sum().float()
    acc=((tp+tn)/(tp+tn+fp+fn+1e-8)).item()
    prec=(tp/(tp+fp+1e-8)).item(); rec=(tp/(tp+fn+1e-8)).item()
    f1=2*prec*rec/(prec+rec+1e-8)
    iou=(tp/(tp+fp+fn+1e-8)).item()
    return {"d21_acc":acc,"d21_f1":f1,"d21_iou":iou}

# ==============================================================
# === Utilidades geométricas ==================================
# ==============================================================

def knn_indices(query: torch.Tensor, ref: torch.Tensor, k: int) -> torch.Tensor:
    """Índices de los k vecinos más cercanos (Euclídeo)."""
    d = torch.cdist(query, ref)  # (B, M, N)
    k = min(k, ref.size(1))
    return torch.topk(d, k=k, dim=-1, largest=False).indices


def batched_gather(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Selecciona puntos por batch e índice (B,N,C) → (B,M,K,C)."""
    B, N, C = points.shape
    _, M, K = idx.shape
    b = torch.arange(B, device=points.device)[:, None, None].expand(B, M, K)
    return points[b, idx, :]


# ==============================================================
# === PointNet (Qi et al., 2017) ==============================
# ==============================================================

class STN3d(nn.Module):
    """Spatial Transformer 3D para alinear la nube de entrada."""
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.conv1, self.bn1 = nn.Conv1d(k, 64, 1), nn.BatchNorm1d(64)
        self.conv2, self.bn2 = nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128)
        self.conv3, self.bn3 = nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024)
        self.fc1, self.bn4 = nn.Linear(1024, 512), nn.BatchNorm1d(512)
        self.fc2, self.bn5 = nn.Linear(512, 256), nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, k * k)

    def forward(self, x):
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
    """Segmentación punto a punto con T-Net y convoluciones 1D."""
    def __init__(self, num_classes=26, dropout=0.5):
        super().__init__()
        self.input_tnet = STN3d(3)
        self.conv1, self.bn1 = nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64)
        self.conv2, self.bn2 = nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128)
        self.conv3, self.bn3 = nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024)
        self.fconv1, self.bn4 = nn.Conv1d(1152, 512, 1), nn.BatchNorm1d(512)
        self.fconv2, self.bn5 = nn.Conv1d(512, 256, 1), nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(dropout)
        self.fconv3 = nn.Conv1d(256, num_classes, 1)

    def forward(self, xyz):
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
        return self.fconv3(x).transpose(2, 1)


# ==============================================================
# === PointNet++ (lite) ========================================
# ==============================================================

class MLP1d(nn.Module):
    def __init__(self, in_ch, mlp):
        super().__init__()
        layers, c = [], in_ch
        for oc in mlp:
            layers += [nn.Conv1d(c, oc, 1), nn.BatchNorm1d(oc), nn.ReLU(True)]
            c = oc
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)


class SA_Layer(nn.Module):
    """Set Abstraction: submuestreo + kNN + MLP + max-pool."""
    def __init__(self, nsample, in_ch, mlp):
        super().__init__()
        self.nsample = nsample
        self.mlp = MLP1d(in_ch + 3, mlp)
        self.out_ch = mlp[-1]

    def forward(self, xyz, feats):
        B, P, _ = xyz.shape
        M = max(1, P // 4)
        idx_center = torch.linspace(0, P - 1, M, device=xyz.device, dtype=torch.long)[None, :].repeat(B, 1)
        centers = torch.gather(xyz, 1, idx_center[..., None].expand(-1, -1, 3))
        idx_knn = knn_indices(centers, xyz, self.nsample)
        neigh_xyz = batched_gather(xyz, idx_knn)
        local_xyz = (neigh_xyz - centers[:, :, None, :]).permute(0, 3, 1, 2)
        if feats is not None:
            feats_perm = feats.transpose(1, 2)
            neigh_f = batched_gather(feats_perm, idx_knn).permute(0, 3, 1, 2)
            cat = torch.cat([local_xyz, neigh_f], dim=1)
        else:
            cat = local_xyz
        Bm, Cm, Mm, Km = cat.shape
        cat = cat.reshape(Bm, Cm, Mm * Km)
        out = self.mlp(cat).view(Bm, -1, Mm, Km).max(dim=-1)[0]
        return centers, out


class FP_Layer(nn.Module):
    def __init__(self, in_ch, mlp):
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
        f2p = feats2.transpose(1, 2)
        neigh = batched_gather(f2p, idx)
        out = (w[..., None] * neigh).sum(dim=2).transpose(1, 2)
        if feats1 is not None:
            out = torch.cat([out, feats1], dim=1)
        return self.mlp(out)


class PointNet2Seg(nn.Module):
    """PointNet++ lite para segmentación."""
    def __init__(self, num_classes=26, nsample=32):
        super().__init__()
        self.sa1 = SA_Layer(nsample, 0, [64, 128, 256])
        self.sa2 = SA_Layer(nsample, 256, [256, 512, 512])
        self.fp1 = FP_Layer(512 + 256, [256, 256])
        self.fp2 = FP_Layer(256, [256, 128])
        self.head = nn.Sequential(
            nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Dropout(0.5), nn.Conv1d(128, num_classes, 1)
        )
    def forward(self, xyz):
        feats = None
        xyz1, f1 = self.sa1(xyz, feats)
        xyz2, f2 = self.sa2(xyz1, f1)
        f = self.fp1(xyz1, xyz2, f1, f2)
        f = self.fp2(xyz, xyz1, None, f)
        return self.head(f).transpose(2, 1)


# ==============================================================
# === DilatedToothSegNet ======================================
# ==============================================================

class DilatedBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, dilation=dilation),
            nn.BatchNorm1d(out_ch), nn.ReLU(True),
            nn.Conv1d(out_ch, out_ch, 1, dilation=dilation),
            nn.BatchNorm1d(out_ch), nn.ReLU(True),
        )
    def forward(self, x): return self.net(x)


class DilatedToothSegNet(nn.Module):
    def __init__(self, num_classes=26):
        super().__init__()
        self.backbone = nn.Sequential(
            DilatedBlock(3, 64, 1),
            DilatedBlock(64, 128, 2),
            DilatedBlock(128, 256, 4),
            DilatedBlock(256, 256, 1),
        )
        self.head = nn.Sequential(
            nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Dropout(0.5), nn.Conv1d(128, num_classes, 1)
        )
    def forward(self, xyz):
        x = xyz.transpose(2, 1)
        f = self.backbone(x)
        return self.head(f).transpose(2, 1)


# ==============================================================
# === Transformer3D (con Fourier PE) ===========================
# ==============================================================

class FourierPE(nn.Module):
    def __init__(self, num_feats=32, scale=10.0):
        super().__init__()
        self.num_feats = num_feats; self.scale = scale
    def forward(self, xyz):
        x = xyz * self.scale
        k = torch.arange(self.num_feats, device=xyz.device).float()
        freqs = (2.0 ** k)[None, None, :]
        sin = torch.sin(x.unsqueeze(-1) / freqs)
        cos = torch.cos(x.unsqueeze(-1) / freqs)
        return torch.cat([sin, cos], dim=-1).reshape(xyz.size(0), xyz.size(1), -1)


class Transformer3D(nn.Module):
    def __init__(self, num_classes=26, d_model=128, nhead=4, depth=4, dim_ff=256):
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
            nn.Dropout(0.5), nn.Linear(d_model, num_classes)
        )
    def forward(self, xyz):
        pe = self.pe(xyz)
        x = self.lin(torch.cat([xyz, pe], -1))
        x = self.enc(x)
        return self.head(x)


# ==============================================================
# === ToothFormer (académico-lite) =============================
# ==============================================================

class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, emb_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_ch, 64, 1), nn.ReLU(True),
            nn.Conv2d(64, 128, 1), nn.ReLU(True),
            nn.Conv2d(128, emb_dim, 1)
        )
    def forward(self, xyz, centers, idx_knn):
        neigh = batched_gather(xyz, idx_knn)
        local = neigh - centers[:, :, None, :]
        x = local.permute(0, 3, 1, 2)
        f = self.mlp(x)
        f = torch.max(f, dim=-1)[0].permute(0, 2, 1)
        return f


class LearnablePE(nn.Module):
    def __init__(self, dim, max_patches=256):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_patches, dim) * 0.02)
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class ToothFormer(nn.Module):
    def __init__(self, num_classes=26, emb_dim=256, nhead=8, depth=6, dim_ff=512,
                 num_patches=64, k_per_patch=128):
        super().__init__()
        self.num_patches = num_patches; self.k = k_per_patch
        self.patch_embed = PatchEmbed(3, emb_dim)
        self.pos = LearnablePE(emb_dim, num_patches)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=nhead, dim_feedforward=dim_ff,
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.proj_lin = nn.Linear(emb_dim, emb_dim)
        self.head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.ReLU(True),
            nn.Dropout(0.5), nn.Linear(emb_dim, num_classes)
        )

    @torch.no_grad()
    def _choose_centers(self, xyz):
        B, N, _ = xyz.shape
        idx = torch.linspace(0, N - 1, steps=self.num_patches, device=xyz.device).long()
        idx = idx.unsqueeze(0).repeat(B, 1)
        centers = torch.gather(xyz, 1, idx[..., None].expand(-1, -1, 3))
        return centers, idx

    @torch.no_grad()
    def _knn_per_center(self, centers, xyz, k):
        return knn_indices(centers, xyz, k=k)

    def forward(self, xyz):
        B, N, _ = xyz.shape
        centers, _ = self._choose_centers(xyz)
        idx_knn = self._knn_per_center(centers, xyz, self.k)
        tokens = self.patch_embed(xyz, centers, idx_knn)
        tokens = self.encoder(self.pos(tokens))
        idx_pc = knn_indices(xyz, centers, k=1).squeeze(-1)
        b = torch.arange(B, device=xyz.device)[:, None].expand(B, N)
        picked = tokens[b, idx_pc, :]
        feats = self.proj_lin(picked)
        return self.head(feats)

# ==============================================================
# === Fábrica de modelos ======================================
# ==============================================================

def build_model(name: str, num_classes: int) -> nn.Module:
    n = name.lower()
    if n == "pointnet":
        return PointNetSeg(num_classes)
    elif n == "pointnetpp":
        return PointNet2Seg(num_classes)
    elif n == "dilatedtoothsegnet":
        return DilatedToothSegNet(num_classes)
    elif n == "transformer3d":
        return Transformer3D(num_classes)
    elif n == "toothformer":
        return ToothFormer(num_classes)
    else:
        raise ValueError(f"Modelo no reconocido: {name}")


# ==============================================================
# === Early stopping ==========================================
# ==============================================================

class EarlyStopping:
    """Detiene el entrenamiento si la pérdida de validación no mejora."""
    def __init__(self, patience=20, delta=1e-4, ckpt_dir=None):
        self.patience = patience
        self.delta = delta
        self.ckpt_dir = Path(ckpt_dir) if ckpt_dir else None
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss, model, epoch):
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
# === Carga / auto-cálculo de pesos por clase ==================
# ==============================================================

def load_class_weights(artifacts_dir: Path, num_classes: int) -> Optional[torch.Tensor]:
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
            print("[WARN] Tamaño de class_weights.json no coincide, ignorando archivo.")
            return None
        return torch.tensor(arr, dtype=torch.float32)
    except Exception as e:
        print(f"[WARN] Error al leer class_weights.json: {e}")
        return None


def auto_class_weights_from_freqs(y_train: np.ndarray, num_classes: int,
                                  clip_min: float = 0.2, clip_max: float = 5.0) -> np.ndarray:
    """
    Esquema paper-like: w_c = 1 / log(1.2 + freq_c)
    Normaliza a media=1 y limita en [clip_min, clip_max].
    """
    freqs = np.bincount(y_train.flatten(), minlength=num_classes).astype(np.float64)
    freqs = np.maximum(freqs, 1.0)
    inv_log = 1.0 / np.log(1.2 + freqs)
    inv_log /= inv_log.mean()
    inv_log = np.clip(inv_log, clip_min, clip_max)
    return inv_log


# ==============================================================
# === Pérdida combinada: CE + λ·DiceLoss =======================
# ==============================================================

def multiclass_dice_loss(logits: torch.Tensor,
                         targets: torch.Tensor,
                         ignore_index: Optional[int],
                         eps: float = 1e-6) -> torch.Tensor:
    """
    Dice loss (1 - Dice) promedio por clase (multiclase).
    - logits: (B, C, P)
    - targets: (B, P)
    Maneja ignore_index (enmascara puntos ignorados).
    """
    B, C, P = logits.shape
    probs = F.softmax(logits, dim=1)                      # (B, C, P)
    # one-hot (B, P) -> (B, P, C) -> (B, C, P)
    tgt_oh = F.one_hot(targets.clamp_min(0), num_classes=C).permute(0, 2, 1).float()

    if ignore_index is not None:
        mask = (targets != ignore_index).float()          # (B, P)
        mask = mask.unsqueeze(1)                          # (B, 1, P)
        probs = probs * mask
        tgt_oh = tgt_oh * mask

    intersect = (probs * tgt_oh).sum(dim=(0, 2))          # (C,)
    cardinal  = (probs + tgt_oh).sum(dim=(0, 2))          # (C,)
    dice_c = (2 * intersect + eps) / (cardinal + eps)     # (C,)
    dice_loss = 1.0 - dice_c.mean()
    return dice_loss


class CombinedLoss(nn.Module):
    """
    L = CrossEntropy(weight=class_w, ignore_index) + λ * DiceLoss
    """
    def __init__(self,
                 class_weights: torch.Tensor,
                 ignore_index: Optional[int] = None,
                 dice_lambda: float = 0.3):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        self.ignore_index = ignore_index
        self.dice_lambda = dice_lambda

    def forward(self, logits_bpc: torch.Tensor, targets_bp: torch.Tensor) -> torch.Tensor:
        """
        logits_bpc: (B, P, C)  -> se transpone a (B, C, P) para CE/Dice
        targets_bp: (B, P)
        """
        logits = logits_bpc.transpose(2, 1).contiguous()  # (B, C, P)
        ce_loss = self.ce(logits, targets_bp)
        dice_loss = multiclass_dice_loss(logits, targets_bp, self.ignore_index)
        return ce_loss + self.dice_lambda * dice_loss


# ==============================================================
# === Entrenamiento y evaluación ==============================
# ==============================================================

def train_one_epoch(model: nn.Module, loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    criterion: nn.Module,
                    device: torch.device,
                    num_classes: int, d21_id: int) -> Tuple[float, Dict[str, float]]:
    model.train()
    loss_sum = 0.0
    cm = torch.zeros(num_classes, num_classes, device=device)
    d21 = {"d21_acc": 0.0, "d21_f1": 0.0, "d21_iou": 0.0}
    batches = 0

    for X, Y in loader:
        X, Y = to_device((X, Y), device)
        X = sanitize_tensor(normalize_cloud(sanitize_tensor(X)))

        logits = sanitize_tensor(model(X))                # (B, P, C)
        Y = torch.clamp(Y, 0, num_classes - 1)

        loss = criterion(logits, Y)                       # CombinedLoss maneja transpose
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
def evaluate(model: nn.Module, loader: DataLoader,
             criterion: nn.Module,
             device: torch.device,
             num_classes: int, d21_id: int) -> Tuple[float, Dict[str, float]]:
    model.eval()
    loss_sum = 0.0
    cm = torch.zeros(num_classes, num_classes, device=device)
    d21 = {"d21_acc": 0.0, "d21_f1": 0.0, "d21_iou": 0.0}
    batches = 0

    for X, Y in loader:
        X, Y = to_device((X, Y), device)
        X = sanitize_tensor(normalize_cloud(sanitize_tensor(X)))

        logits = sanitize_tensor(model(X))                # (B, P, C)
        Y = torch.clamp(Y, 0, num_classes - 1)

        loss = criterion(logits, Y)
        loss_sum += float(loss.detach().cpu())

        cm += confusion_matrix(logits, Y, num_classes)
        d21m = d21_metrics(logits, Y, d21_id)
        for k in d21: d21[k] += d21m[k]
        batches += 1

    macro = macro_from_cm(cm)
    for k in d21: d21[k] /= max(1, batches)
    return loss_sum / max(1, batches), {**macro, **d21}

# ==============================================================
# === Logging / historial / guardado ===========================
# ==============================================================

def update_history(history: Dict[str, List[float]], prefix: str, stats: Dict[str, float]):
    for k, v in stats.items():
        history.setdefault(f"{prefix}_{k}", []).append(float(v))


def print_epoch(ep: int, epochs: int, tr_loss: float, va_loss: float, va_stats: Dict[str, float]):
    msg = (
        f"[Ep {ep:03d}/{epochs}] "
        f"tr={tr_loss:.4f}  va={va_loss:.4f}  "
        f"acc={va_stats['acc']:.3f}  f1={va_stats['f1']:.3f}  "
        f"iou={va_stats['iou']:.3f}  d21_f1={va_stats['d21_f1']:.3f}"
    )
    print(msg)


def save_history_csv(history: Dict[str, List[float]], out_csv: Path):
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


def save_run_summary(run_dir: Path, args: Dict[str, Any], num_classes: int,
                     n_params: int, best_val: float, last_epoch: int,
                     te_loss: float, te_stats: Dict[str, float], history: Dict[str, List[float]]):
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
    parser = argparse.ArgumentParser(
        description="Entrenamiento paper-like con pérdida balanceada (CE + λ·Dice) y pesos 1/log(1.2+freq)."
    )
    # Paths
    parser.add_argument("--data_dir", required=True, help="Carpeta con X_*.npz / Y_*.npz (+ artifacts/)")
    parser.add_argument("--out_dir", required=True, help="Carpeta base para runs")
    # Modelo
    parser.add_argument("--model", default="pointnetpp",
                        choices=["pointnet", "pointnetpp", "dilatedtoothsegnet", "transformer3d", "toothformer"])
    # Hparams
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)           # Adam, más estable en desbalance
    parser.add_argument("--weight_decay", type=float, default=5e-5)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=6)
    # Loss / labels
    parser.add_argument("--ignore_index", type=int, default=0, help="ID a ignorar en la pérdida (fondo=0)")
    parser.add_argument("--d21_id", type=int, default=1, help="ID remapeado del incisivo 21 en tu dataset")
    parser.add_argument("--dice_lambda", type=float, default=0.3, help="Peso de DiceLoss en la pérdida combinada")
    # Scheduler / eval
    parser.add_argument("--use_cosine", action="store_true", help="Usar CosineAnnealingLR(T_max=epochs)")
    parser.add_argument("--eval_best", action="store_true", help="Evaluar test con best.pt si existe")
    # Auto weights params
    parser.add_argument("--w_clip_min", type=float, default=0.2, help="Límite inferior para pesos automáticos")
    parser.add_argument("--w_clip_max", type=float, default=5.0, help="Límite superior para pesos automáticos")
    args = parser.parse_args()

    # Semilla y dispositivo
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    # Data
    data_dir = Path(args.data_dir)
    loaders = make_loaders(data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

    # Detección de clases
    Ytr = np.load(data_dir / "Y_train.npz")["Y"]
    unique_ids = np.unique(Ytr)
    num_classes = int(unique_ids.max()) + 1
    print(f"[INFO] Clases detectadas en train: {unique_ids.tolist()}")
    print(f"[INFO] num_classes={num_classes}")

    # Modelo
    model = build_model(args.model, num_classes=num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] {args.model}  params={n_params/1e6:.2f}M")

    # Pesos de clase: cargar o calcular automáticamente (1/log(1.2+freq), normalizado y con clip)
    artifacts = data_dir / "artifacts"
    class_w = load_class_weights(artifacts, num_classes)
    if class_w is not None:
        class_w = class_w.to(device)
        print("[INFO] Usando class_weights.json")
    else:
        print("[INFO] Generando class_weights automáticamente (1/log(1.2+freq), normalizados y con clip)")
        auto_w = auto_class_weights_from_freqs(
            Ytr, num_classes, clip_min=args.w_clip_min, clip_max=args.w_clip_max
        )
        class_w = torch.tensor(auto_w, dtype=torch.float32, device=device)
        for i, w in enumerate(auto_w):
            print(f"  Clase {i:02d}: {w:.3f}")

    # Pérdida combinada (CE + λ·Dice)
    criterion = CombinedLoss(class_weights=class_w,
                             ignore_index=args.ignore_index,
                             dice_lambda=args.dice_lambda)

    # Optimizador y (opcional) scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs) if args.use_cosine else None

    # Salida
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_dir) / f"{args.model}_{stamp}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    print(f"[RUN] {run_dir}")

    # Early stopping e historial
    stopper = EarlyStopping(patience=args.patience, delta=1e-4, ckpt_dir=run_dir / "checkpoints")
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

    # Evaluación en test con best o final
    ckpt = run_dir / "checkpoints" / ("best.pt" if args.eval_best and (run_dir / "checkpoints" / "best.pt").exists()
                                      else "final_model.pt")
    print(f"[EVAL] Cargando {ckpt.name} para evaluación en test.")
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"])

    te_loss, te_stats = evaluate(model, loaders["test"], criterion, device, num_classes, args.d21_id)
    print(f"[TEST] loss={te_loss:.4f}  acc={te_stats['acc']:.3f}  f1={te_stats['f1']:.3f}  "
          f"iou={te_stats['iou']:.3f}  d21_f1={te_stats['d21_f1']:.3f}")

    # Actualizar resumen con test
    save_run_summary(run_dir, vars(args), num_classes, n_params, best_val, last_epoch,
                     te_loss=te_loss, te_stats=te_stats, history=history)
    print(f"[DONE] Resultados guardados en: {run_dir}")


if __name__ == "__main__":
    main()
