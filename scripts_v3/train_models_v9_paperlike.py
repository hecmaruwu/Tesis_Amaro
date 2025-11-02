#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_models_v8_paperlike.py (v8-final)

- PointNet, PointNet++, DilatedToothSegNet, Transformer3D (mejorado con Fourier PE + submuestreo)
- Normalización consistente por lote (center + unit sphere)
- CrossEntropy con ignore_index=0 (fondo) y class_weights si existen
- Métricas macro (acc/prec/rec/f1/IoU) y métricas específicas para diente 21
- Checkpoints y carpetas de corrida con métricas en el nombre
- Gráficas por modelo y gráficas combinadas entre modelos
"""

import os, json, time, argparse, random, gc, csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from torchmetrics.classification import (
        MulticlassAccuracy, MulticlassPrecision,
        MulticlassRecall, MulticlassF1Score,
        MulticlassJaccardIndex
    )
    HAS_TORCHMETRICS = True
except Exception:
    HAS_TORCHMETRICS = False


# ===========================================================
# ----------------------  DATASET  --------------------------
# ===========================================================

class CloudDataset(Dataset):
    def __init__(self, X_path, Y_path):
        self.X = np.load(X_path)["X"].astype(np.float32)
        self.Y = np.load(Y_path)["Y"].astype(np.int64)
        assert self.X.shape[0] == self.Y.shape[0], "X e Y deben tener mismo N"
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.Y[i])


def make_loaders(data_path, batch_size=8, num_workers=4):
    data_path = Path(data_path)
    paths = {
        "train": (data_path/"X_train.npz", data_path/"Y_train.npz"),
        "val":   (data_path/"X_val.npz",   data_path/"Y_val.npz"),
        "test":  (data_path/"X_test.npz",  data_path/"Y_test.npz")
    }
    loaders = {}
    for split,(xp,yp) in paths.items():
        if not xp.exists() or not yp.exists():
            raise FileNotFoundError(f"Faltan archivos del split {split}: {xp} / {yp}")
        ds = CloudDataset(xp, yp)
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=(split=="train"),
            drop_last=False, num_workers=num_workers, pin_memory=True
        )
    return loaders


# ===========================================================
# ----------------------  UTILS  ----------------------------
# ===========================================================

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def normalize_cloud(x: torch.Tensor) -> torch.Tensor:
    """Normaliza cada nube del batch a esfera unitaria (B,P,3)."""
    c = x.mean(dim=1, keepdim=True)
    x = x - c
    r = (x.pow(2).sum(-1).sqrt()).max(dim=1, keepdim=True)[0].unsqueeze(-1)
    return x / (r + 1e-8)

def save_json(obj, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path,"w",encoding="utf-8") as f: json.dump(obj,f,indent=2)

def plot_curves(history: Dict[str, List[float]], out_dir: Path, model_name: str):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for m in ["loss","acc","f1","iou","d21_acc","d21_f1","d21_iou"]:
        plt.figure(figsize=(7,4))
        for split in ["train","val"]:
            key = f"{split}_{m}"
            if key in history and len(history[key]) > 0:
                plt.plot(history[key], label=split)
        plt.xlabel("Época"); plt.ylabel(m.upper())
        plt.title(f"{model_name} – {m.upper()}"); plt.legend(); plt.tight_layout()
        plt.savefig(out_dir/f"{model_name}_{m}.png", dpi=300); plt.close()


# ===========================================================
# ----------------------  MÉTRICAS  -------------------------
# ===========================================================

class MetricsBundle:
    def __init__(self,num_classes:int,device:torch.device,ignore_index:int=0):
        self.num_classes=num_classes; self.device=device; self.ignore=ignore_index
        self.has_tm=HAS_TORCHMETRICS
        if self.has_tm:
            self._acc  = MulticlassAccuracy(num_classes=num_classes,average="macro",ignore_index=self.ignore).to(device)
            self._prec = MulticlassPrecision(num_classes=num_classes,average="macro",ignore_index=self.ignore).to(device)
            self._rec  = MulticlassRecall(num_classes=num_classes,average="macro",ignore_index=self.ignore).to(device)
            self._f1   = MulticlassF1Score(num_classes=num_classes,average="macro",ignore_index=self.ignore).to(device)
            self._iou  = MulticlassJaccardIndex(num_classes=num_classes,average="macro",ignore_index=self.ignore).to(device)
        self.reset_cm()

    def reset_cm(self):
        self.cm=torch.zeros((self.num_classes,self.num_classes),
                            device=self.device,dtype=torch.long)

    @torch.no_grad()
    def update(self,logits,y_true):
        preds=logits.argmax(dim=-1)
        t=y_true.view(-1); p=preds.view(-1)
        valid=(t>=0)&(t<self.num_classes)
        t=t[valid]; p=p[valid]
        idx=t*self.num_classes+p
        binc=torch.bincount(idx,minlength=self.num_classes**2).reshape(self.num_classes,self.num_classes)
        self.cm+=binc.long()
        if self.has_tm:
            self._acc.update(p,t); self._prec.update(p,t)
            self._rec.update(p,t); self._f1.update(p,t); self._iou.update(p,t)

    def compute_macro(self):
        if self.has_tm:
            res={ "acc":float(self._acc.compute()),"prec":float(self._prec.compute()),
                  "rec":float(self._rec.compute()),"f1":float(self._f1.compute()),
                  "iou":float(self._iou.compute()) }
            for m in [self._acc,self._prec,self._rec,self._f1,self._iou]: m.reset()
            return res
        cm=self.cm.float(); tp=torch.diag(cm); gt=cm.sum(1); pd=cm.sum(0)
        acc=(tp.sum()/cm.sum()).item() if cm.sum()>0 else 0
        prec=torch.nanmean(tp/(pd+1e-8)).item()
        rec=torch.nanmean(tp/(gt+1e-8)).item()
        f1=torch.nanmean(2*tp/(gt+pd+1e-8)).item()
        iou=torch.nanmean(tp/(gt+pd-tp+1e-8)).item()
        return{"acc":acc,"prec":prec,"rec":rec,"f1":f1,"iou":iou}

def per_class_from_cm(cm: torch.Tensor) -> Dict[str, np.ndarray]:
    """devuelve dict con arrays por-clase: acc, prec, rec, f1, iou."""
    cm = cm.float()
    tp = torch.diag(cm)
    gt = cm.sum(1)
    pd = cm.sum(0)
    acc_c = torch.nan_to_num(tp/(cm.sum()+1e-8))
    prec  = torch.nan_to_num(tp/(pd+1e-8))
    rec   = torch.nan_to_num(tp/(gt+1e-8))
    f1    = torch.nan_to_num(2*prec*rec/(prec+rec+1e-8))
    iou   = torch.nan_to_num(tp/(gt+pd-tp+1e-8))
    return {k: v.detach().cpu().numpy() for k,v in
            dict(acc=acc_c, prec=prec, rec=rec, f1=f1, iou=iou).items()}


class EarlyStopping:
    def __init__(self, patience=20, delta=1e-4, ckpt_dir=None):
        self.patience = patience
        self.delta = delta
        self.ckpt_dir = Path(ckpt_dir) if ckpt_dir else None
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss, model, epoch):
        improved = val_loss < self.best_loss - self.delta
        if improved:
            self.best_loss = val_loss
            self.counter = 0
            if self.ckpt_dir:
                torch.save({"model": model.state_dict(), "epoch": epoch},
                           self.ckpt_dir/"best.pt")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return improved

# ===========================================================
# -----------  KNN / GATHER / INTERP GEOM HELPERS -----------
# ===========================================================

def knn_indices(query, ref, k):
    d = torch.cdist(query, ref)
    idx = torch.topk(d, k=min(k, ref.size(1)), dim=-1, largest=False).indices
    return idx

def batched_gather(points, idx):
    B, N, C = points.shape
    _, M, K = idx.shape
    batch = torch.arange(B, device=points.device)[:, None, None].expand(B, M, K)
    return points[batch, idx, :]

def three_nn_interp(xyz1, xyz2, feats2, k=3):
    B, N1, _ = xyz1.shape
    _, C2, N2 = feats2.shape
    idx = knn_indices(xyz1, xyz2, k=min(k, N2))
    d = torch.cdist(xyz1, xyz2)
    knn_d = torch.gather(d, 2, idx).clamp(min=1e-8)
    w = 1.0 / knn_d
    w = w / w.sum(dim=-1, keepdim=True)
    feats2_perm = feats2.transpose(1, 2).contiguous()
    neigh = batched_gather(feats2_perm, idx)
    out = (w[..., None] * neigh).sum(dim=2)
    return out.transpose(1, 2).contiguous()


# ===========================================================
# ---------------------  PointNet Core -----------------------
# ===========================================================

class STN3d(nn.Module):
    """T-Net clásico 3×3 (sin regularizador explícito)."""
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
    """PointNet de segmentación (Qi et al., 2017)."""
    def __init__(self, num_classes=10, dropout=0.5):
        super().__init__()
        self.input_tnet = STN3d(k=3)
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
        return self.fconv3(x).transpose(2, 1)  # (B,P,C)


# ===========================================================
# ----------------------  PointNet++ ------------------------
# ===========================================================

class MLP1d(nn.Module):
    def __init__(self, in_ch, mlp):
        super().__init__()
        layers = []
        c = in_ch
        for oc in mlp:
            layers += [nn.Conv1d(c, oc, 1), nn.BatchNorm1d(oc), nn.ReLU(True)]
            c = oc
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class SA_Layer(nn.Module):
    """Set Abstraction simplificado con kNN y mini-PointNet (sin CUDA extras)."""
    def __init__(self, nsample, in_ch, mlp):
        super().__init__()
        self.nsample = nsample
        self.mlp = MLP1d(in_ch + 3, mlp)
        self.out_ch = mlp[-1]

    def forward(self, xyz, feats):
        B, P, _ = xyz.shape
        M = max(1, P // 4)
        idx_center = torch.linspace(0, P - 1, M, device=xyz.device, dtype=torch.long)
        idx_center = idx_center[None, :].repeat(B, 1)
        centers = torch.gather(xyz, 1, idx_center[..., None].expand(-1, -1, 3))
        idx_knn = knn_indices(centers, xyz, self.nsample)
        neigh_xyz = batched_gather(xyz, idx_knn)
        local_xyz = neigh_xyz - centers[:, :, None, :]
        local_xyz = local_xyz.permute(0, 1, 3, 2).contiguous()

        if feats is not None:
            feats_perm = feats.transpose(1, 2).contiguous()
            neigh_f = batched_gather(feats_perm, idx_knn)
            neigh_f = neigh_f.permute(0, 1, 3, 2).contiguous()
            cat = torch.cat([local_xyz, neigh_f], dim=2)
        else:
            cat = local_xyz

        Bm, Mm, Cm, K = cat.shape
        cat_flat = cat.view(Bm * Mm, Cm, K)
        out = self.mlp(cat_flat)
        out = torch.max(out, dim=-1, keepdim=False)[0]
        out = out.view(Bm, Mm, -1).permute(0, 2, 1).contiguous()
        return centers, out

class FP_Layer(nn.Module):
    def __init__(self, in_ch, mlp):
        super().__init__()
        self.mlp = MLP1d(in_ch, mlp)
        self.out_ch = mlp[-1]
    def forward(self, xyz1, xyz2, feats1, feats2):
        interp = three_nn_interp(xyz1, xyz2, feats2)
        cat = torch.cat([interp, feats1], dim=1) if feats1 is not None else interp
        return self.mlp(cat)

class PointNet2Seg(nn.Module):
    """Encoder-decoder SSG (Qi et al., 2017b)."""
    def __init__(self, num_classes=10, dropout=0.5):
        super().__init__()
        self.sa1 = SA_Layer(nsample=32,  in_ch=0,   mlp=[64, 64, 128])
        self.sa2 = SA_Layer(nsample=64,  in_ch=128, mlp=[128, 128, 256])
        self.sa3 = SA_Layer(nsample=128, in_ch=256, mlp=[256, 512, 1024])
        self.fp3 = FP_Layer(in_ch=1024 + 256, mlp=[256, 256])
        self.fp2 = FP_Layer(in_ch=256 + 128,  mlp=[256, 128])
        self.fp1 = FP_Layer(in_ch=128,        mlp=[128, 128, 128])
        self.head = nn.Sequential(
            nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Dropout(dropout), nn.Conv1d(128, num_classes, 1)
        )

    def forward(self, xyz):
        feats0 = None
        l1_xyz, l1 = self.sa1(xyz, feats0)
        l2_xyz, l2 = self.sa2(l1_xyz, l1)
        l3_xyz, l3 = self.sa3(l2_xyz, l2)
        l2n = self.fp3(l2_xyz, l3_xyz, l2, l3)
        l1n = self.fp2(l1_xyz, l2_xyz, l1, l2n)
        l0n = self.fp1(xyz, l1_xyz, None, l1n)
        out = self.head(l0n).transpose(2, 1)
        return out


# ===========================================================
# -------------  DilatedToothSegNet & Transformer -----------
# ===========================================================

class DilatedToothSegNet(nn.Module):
    def __init__(self, num_classes=10, base=64, dropout=0.5):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv1d(3, base, 1, dilation=1),
                                  nn.BatchNorm1d(base), nn.ReLU(True))
        self.enc2 = nn.Sequential(nn.Conv1d(base, base*2, 1, dilation=2),
                                  nn.BatchNorm1d(base*2), nn.ReLU(True))
        self.enc3 = nn.Sequential(nn.Conv1d(base*2, base*4, 1, dilation=3),
                                  nn.BatchNorm1d(base*4), nn.ReLU(True))
        self.enc4 = nn.Sequential(nn.Conv1d(base*4, base*8, 1, dilation=4),
                                  nn.BatchNorm1d(base*8), nn.ReLU(True))
        self.head = nn.Sequential(
            nn.Conv1d(base*(1+2+4+8), base*4, 1),
            nn.BatchNorm1d(base*4), nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Conv1d(base*4, num_classes, 1)
        )

    def forward(self, xyz):  # (B, P, 3)
        x = xyz.transpose(2, 1)
        e1 = self.enc1(x); e2 = self.enc2(e1); e3 = self.enc3(e2); e4 = self.enc4(e3)
        cat = torch.cat([e1,e2,e3,e4], dim=1)
        return self.head(cat).transpose(2,1)


# -------- Transformer mejorado: Fourier PE + submuestreo ----

def fourier_pe(xyz: torch.Tensor, freqs=(1,2,4,8,16)) -> torch.Tensor:
    """
    xyz: (B,P,3) en [-1,1] aprox.
    Devuelve concat[ sin(B*xyz), cos(B*xyz) ] -> (B,P, 3*2*len(freqs))
    """
    B, P, _ = xyz.shape
    Bv = torch.tensor(freqs, device=xyz.device, dtype=xyz.dtype).view(1,1,-1)  # (1,1,F)
    x = xyz.unsqueeze(-1) * Bv  # (B,P,3,F)
    s = torch.sin(x); c = torch.cos(x)
    pe = torch.cat([s,c], dim=-1)          # (B,P,3,2F)
    pe = pe.view(B, P, -1)                 # (B,P, 3*2F)
    return pe

class Transformer3DSeg(nn.Module):
    """
    Proyección de [xyz || FourierPE(xyz)] -> d_model, encoder Transformer, head MLP.
    Opción de submuestrear a N puntos (p.ej. 2048) sólo para este modelo.
    """
    def __init__(self, num_classes=10, dim=128, heads=4, depth=4, dropout=0.3,
                 tr_points: Optional[int] = 2048, fourier_freqs=(1,2,4,8,16)):
        super().__init__()
        self.dim = dim
        self.tr_points = tr_points
        self.freqs = fourier_freqs

        in_dim = 3 + 3*2*len(self.freqs)
        self.embed = nn.Sequential(
            nn.Linear(in_dim, dim),
            nn.ReLU(True),
            nn.Linear(dim, dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, batch_first=True,
            dim_feedforward=dim*4, dropout=dropout, activation="relu"
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.head = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes)
        )

    def forward(self, xyz):  # (B,P,3)
        B, P, _ = xyz.shape

        # Submuestreo opcional (uniforme aleatorio) SOLO para el transformer
        if (self.tr_points is not None) and (P > self.tr_points):
            idx = torch.randperm(P, device=xyz.device)[:self.tr_points]
            xyz = xyz[:, idx, :]
            P = xyz.shape[1]

        pe = fourier_pe(xyz, self.freqs)            # (B,P, 3*2F)
        x = torch.cat([xyz, pe], dim=-1)            # (B,P, in_dim)
        x = self.embed(x)                           # (B,P, d)
        x = self.encoder(x)                         # (B,P, d)
        return self.head(x)                         # (B,P,C)

# ===========================================================
# ------------------  Fábrica de modelos  -------------------
# ===========================================================

def build_model(name: str, num_classes: int, args):
    n = name.lower()
    if n == "pointnet":
        return PointNetSeg(num_classes=num_classes, dropout=args.dropout)
    if n == "pointnetpp":
        return PointNet2Seg(num_classes=num_classes, dropout=args.dropout)
    if n == "dilated":
        return DilatedToothSegNet(num_classes=num_classes,
                                  base=args.base_channels, dropout=args.dropout)
    if n == "transformer":
        return Transformer3DSeg(num_classes=num_classes,
                                dim=args.tr_dim, heads=args.tr_heads,
                                depth=args.tr_depth, dropout=args.dropout,
                                tr_points=args.tr_points)
    raise ValueError(f"Modelo no soportado: {name}")

def model_output_classes(model: nn.Module) -> int:
    for m in reversed(list(model.modules())):
        if isinstance(m, nn.Conv1d) and getattr(m,"kernel_size",None)==(1,):
            return m.out_channels
        if isinstance(m, nn.Linear):
            return m.out_features
    raise RuntimeError("No se pudo inferir #clases.")

def compute_class_weights_from_json(artifacts_dir: Path, num_classes: int):
    cw_file = artifacts_dir / "class_weights.json"
    if cw_file.exists():
        data = json.load(open(cw_file))
        if "class_weights" in data:
            w = np.array([data["class_weights"].get(str(i), 1.0)
                          for i in range(num_classes)], dtype=np.float32)
            w[0] = 0.0  # Fondo sin peso
            print(f"[INFO] Cargados pesos desde {cw_file}")
            return torch.tensor(w, dtype=torch.float32)
    print("[WARN] No se encontró class_weights.json, usando pesos uniformes.")
    w = np.ones(num_classes, dtype=np.float32); w[0] = 0.0
    return torch.tensor(w)

def load_label_map_index_21(artifacts_dir: Path) -> Optional[int]:
    lm = artifacts_dir / "label_map.json"
    if not lm.exists(): return None
    data = json.load(open(lm))
    # buscamos qué idx corresponde al id original 21 (FDI)
    idx2id = data.get("idx2id", {})
    for k, v in idx2id.items():
        if int(v) == 21:
            return int(k)
    return None

@torch.no_grad()
def _sanity_check_labels(yb, num_classes, where="train"):
    ymin, ymax = int(yb.min().item()), int(yb.max().item())
    if ymin < 0 or ymax >= num_classes:
        raise RuntimeError(
            f"[LabelRangeError] {where}: etiquetas fuera de rango "
            f"(min={ymin}, max={ymax}, C={num_classes}). Verifique el split."
        )

def one_epoch(model, loader, optimizer, criterion, device, metrics_bundle=None,
              amp=True, clip_grad=1.0, scaler=None, track_d21: Optional[int]=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = 0.0

    # contadores por-época para d21
    d21_hist = {"acc":[], "f1":[], "iou":[]}

    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        xb = normalize_cloud(xb)  # ✅ normalización consistente
        _sanity_check_labels(yb, model_output_classes(model), "train" if is_train else "eval")

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=amp and (device.type=="cuda")):
            logits = model(xb)  # (B, P, C)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), yb.reshape(-1))

        if is_train:
            if scaler is not None and amp and device.type=="cuda":
                scaler.scale(loss).backward()
                if clip_grad > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward()
                if clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()

        total_loss += loss.item()
        if metrics_bundle is not None:
            metrics_bundle.update(logits, yb)

        # diente 21 por batch (si existe)
        if track_d21 is not None:
            with torch.no_grad():
                pred = logits.argmax(dim=-1)  # (B,P)
                t = yb
                mask = (t==track_d21) | (pred==track_d21)
                if mask.any():
                    tp = ((pred==track_d21) & (t==track_d21) & mask).sum().item()
                    fp = ((pred==track_d21) & (t!=track_d21) & mask).sum().item()
                    fn = ((pred!=track_d21) & (t==track_d21) & mask).sum().item()
                    tn = ((pred!=track_d21) & (t!=track_d21) & mask).sum().item()
                    acc = (tp+tn)/max(1,tp+tn+fp+fn)
                    prec = tp/max(1,tp+fp); rec = tp/max(1,tp+fn)
                    f1 = 2*prec*rec/max(1e-8,prec+rec)
                    iou = tp/max(1,tp+fp+fn)
                    d21_hist["acc"].append(acc); d21_hist["f1"].append(f1); d21_hist["iou"].append(iou)

    avg_loss = total_loss / max(1, len(loader))
    macro = metrics_bundle.compute_macro() if metrics_bundle is not None else {}
    d21_epoch = {k: (float(np.mean(v)) if len(v)>0 else np.nan) for k,v in d21_hist.items()}
    return avg_loss, macro, d21_epoch


def train_model(model_name, loaders, num_classes, args, device, base_outdir,
                class_weights, idx_tooth21: Optional[int]):
    # carpeta base preliminar (sin métricas)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = base_outdir / f"{model_name}_{timestamp}"
    ckpt_dir, plots_dir = run_dir / "checkpoints", run_dir / "plots"
    ckpt_dir.mkdir(parents=True, exist_ok=True); plots_dir.mkdir(exist_ok=True)

    model = build_model(model_name, num_classes, args).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{model_name}] Parámetros entrenables: {n_params:,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device),
                                    ignore_index=0, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience, min_lr=args.min_lr
    )
    early = EarlyStopping(patience=args.es_patience, delta=args.es_delta, ckpt_dir=ckpt_dir)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type=="cuda")

    # history con d21 también
    history = {f"{sp}_{m}":[] for sp in ["train","val"]
               for m in ["loss","acc","prec","rec","f1","iou","d21_acc","d21_f1","d21_iou"]}

    best_val, best_ep = float("inf"), -1
    best_name_suffix = ""
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        mb_tr = MetricsBundle(num_classes, device, ignore_index=0)
        tr_loss, tr_macro, tr_d21 = one_epoch(
            model, loaders["train"], optimizer, criterion, device, mb_tr,
            amp=args.amp, clip_grad=args.clip_grad, scaler=scaler, track_d21=idx_tooth21
        )
        mb_val = MetricsBundle(num_classes, device, ignore_index=0)
        val_loss, val_macro, val_d21 = one_epoch(
            model, loaders["val"], None, criterion, device, mb_val,
            amp=args.amp, clip_grad=args.clip_grad, scaler=None, track_d21=idx_tooth21
        )
        scheduler.step(val_loss)

        # registrar history
        for split, loss, mac, d21 in [
            ("train", tr_loss, tr_macro, tr_d21),
            ("val",   val_loss, val_macro, val_d21)
        ]:
            history[f"{split}_loss"].append(float(loss))
            for k in ["acc","prec","rec","f1","iou"]:
                history[f"{split}_{k}"].append(float(mac.get(k, np.nan)))
            history[f"{split}_d21_acc"].append(float(d21.get("acc", np.nan)))
            history[f"{split}_d21_f1"].append(float(d21.get("f1", np.nan)))
            history[f"{split}_d21_iou"].append(float(d21.get("iou", np.nan)))

        print(f"[{model_name}] Ep {ep:03d}/{args.epochs}  "
              f"tr={tr_loss:.4f}  va={val_loss:.4f}  "
              f"acc={val_macro.get('acc',0):.4f}  f1={val_macro.get('f1',0):.4f}  iou={val_macro.get('iou',0):.4f}  "
              f"d21_f1={val_d21.get('f1',np.nan):.4f}")

        improved = early(val_loss, model, ep)
        if improved:
            best_val, best_ep = val_loss, ep
            # renombrar best con sufijo de métricas para identificar fácilmente
            acc = val_macro.get('acc', 0.0); f1 = val_macro.get('f1', 0.0); iou = val_macro.get('iou', 0.0)
            d21f1 = val_d21.get('f1', np.nan)
            best_name_suffix = f"valA{acc:.3f}_F{f1:.3f}_I{iou:.3f}_D21F{(0 if np.isnan(d21f1) else d21f1):.3f}"
            # duplica el best a un nombre legible
            try:
                torch.save(torch.load(ckpt_dir/"best.pt"), ckpt_dir/f"best_{best_name_suffix}.pt")
            except Exception:
                pass

        if early.early_stop:
            print(f"[{model_name}] Early stopping activado @ {ep} (mejor={best_ep}).")
            break

    total_time = round(time.time() - t0, 2)
    torch.save({"model": model.state_dict(), "epoch": ep}, ckpt_dir / "final.pt")

    # cargar best para test
    if (ckpt_dir / "best.pt").exists():
        model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=device)["model"])
    else:
        print("[WARN] No se encontró best.pt, usando último modelo.")

    # Test final (+ d21)
    mb_te = MetricsBundle(num_classes, device, ignore_index=0)
    criterion_eval = nn.CrossEntropyLoss(weight=class_weights.to(device), ignore_index=0)
    test_loss, test_macro, test_d21 = one_epoch(
        model, loaders["test"], None, criterion_eval, device, mb_te,
        amp=args.amp, clip_grad=args.clip_grad, scaler=None, track_d21=idx_tooth21
    )
    oa = float(torch.diag(mb_te.cm).sum() / mb_te.cm.sum())

    # Renombrar carpeta run con métricas resumen
    suffix = f"A{test_macro.get('acc',0):.3f}_F{test_macro.get('f1',0):.3f}_I{test_macro.get('iou',0):.3f}"
    if idx_tooth21 is not None:
        suffix += f"_D21F{test_d21.get('f1',0):.3f}"
    new_run_dir = base_outdir / f"{model_name}_{timestamp}_{suffix}"
    try:
        run_dir.rename(new_run_dir)
        run_dir = new_run_dir
        ckpt_dir = run_dir / "checkpoints"; plots_dir = run_dir / "plots"
    except Exception:
        pass

    # Copia del best con métricas de TEST
    try:
        if (ckpt_dir/"best.pt").exists():
            torch.save(torch.load(ckpt_dir/"best.pt"),
                       ckpt_dir/f"best_TEST_{suffix}.pt")
    except Exception:
        pass

    # Guardar history y plots
    save_json(history, run_dir / "history.json")
    plot_curves(history, plots_dir, model_name)

    print(f"[{model_name}] Fin. Best val={best_val:.4f} ep={best_ep}  "
          f"Test F1={test_macro.get('f1',0):.4f}  D21F1={test_d21.get('f1',np.nan):.4f}")

    res = {
        "model": model_name, "run_dir": str(run_dir),
        "best_epoch": best_ep, "best_val_loss": float(best_val),
        "test_loss": float(test_loss),
        "test_acc": float(test_macro.get("acc", 0)),
        "test_f1": float(test_macro.get("f1", 0)),
        "test_iou": float(test_macro.get("iou", 0)),
        "overall_acc": float(oa),
        "d21_f1": float(test_d21.get("f1", np.nan)),
        "d21_iou": float(test_d21.get("iou", np.nan)),
        "train_time_sec": total_time
    }
    del model; torch.cuda.empty_cache(); gc.collect()
    return res


def plot_combined(all_histories: Dict[str, Dict[str, List[float]]], out_dir: Path):
    """
    Dibuja curvas combinadas (todos los modelos) para:
    - train/val: acc, f1, iou
    - train/val: d21_acc, d21_f1, d21_iou
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    groups = [
        (["acc","f1","iou"], "macro"),
        (["d21_acc","d21_f1","d21_iou"], "tooth21")
    ]
    for metrics, tag in groups:
        for split in ["train","val"]:
            plt.figure(figsize=(8,5))
            for model_name, hist in all_histories.items():
                for m in metrics:
                    key = f"{split}_{m}"
                    if key in hist and len(hist[key])>0:
                        plt.plot(hist[key], label=f"{model_name}-{m}")
            plt.xlabel("Época"); plt.ylabel("score")
            plt.title(f"{tag.upper()} – {split}")
            plt.legend(ncol=2, fontsize=8)
            plt.tight_layout()
            plt.savefig(out_dir/f"combined_{tag}_{split}.png", dpi=300)
            plt.close()

# ===========================================================
# -------------------------  MAIN  --------------------------
# ===========================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default="all", choices=["all","pointnet","pointnetpp","dilated","transformer"])

    # Hiperparámetros
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--base_channels", type=int, default=64)  # dilated

    # Transformer
    ap.add_argument("--tr_dim", type=int, default=128)
    ap.add_argument("--tr_heads", type=int, default=4)
    ap.add_argument("--tr_depth", type=int, default=4)
    ap.add_argument("--tr_points", type=int, default=2048, help="#puntos usados por el Transformer (<= P)")

    # Scheduler
    ap.add_argument("--lr_patience", type=int, default=10)
    ap.add_argument("--lr_factor", type=float, default=0.3)
    ap.add_argument("--min_lr", type=float, default=5e-6)

    # EarlyStopping
    ap.add_argument("--es_patience", type=int, default=35)
    ap.add_argument("--es_delta", type=float, default=1e-4)

    # Entrenamiento estable
    ap.add_argument("--amp", action="store_true", help="Mixed precision (float16) en CUDA")
    ap.add_argument("--clip_grad", type=float, default=1.0, help="Clipping del gradiente (0 = off)")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cuda", type=int, default=None)

    args = ap.parse_args()
    set_seed(args.seed)

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() and args.cuda is not None else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] Device: {device}  (GPUs visibles: {torch.cuda.device_count()})")

    # Loaders
    loaders = make_loaders(args.data_path, batch_size=args.batch_size)

    # === num_classes robusto (UNIÓN de splits) ===
    dp = Path(args.data_path)
    Ytr = np.load(dp/"Y_train.npz")["Y"]
    Yva = np.load(dp/"Y_val.npz")["Y"]
    Yte = np.load(dp/"Y_test.npz")["Y"]
    num_classes = int(max(Ytr.max(), Yva.max(), Yte.max()) + 1)
    print(f"[DATA] num_classes (global) = {num_classes}")

    # Pesos por clase (train) para balancear, ignorando el fondo
    class_weights = compute_class_weights_from_json(dp/"artifacts", num_classes)

    # Índice del diente 21 si existe en el label_map
    idx_tooth21 = load_label_map_index_21(dp/"artifacts")
    if idx_tooth21 is None:
        print("[WARN] No se encontró índice de diente 21 en label_map.json; métricas d21 saldrán como NaN.")

    # Out base
    base_outdir = Path(args.out_dir) / args.tag
    base_outdir.mkdir(parents=True, exist_ok=True)

    models = [args.model] if args.model != "all" else ["pointnet","pointnetpp","dilated","transformer"]

    results = []
    all_histories = {}

    # Entrenar cada modelo
    for m in models:
        res = train_model(m, loaders, num_classes, args, device, base_outdir, class_weights, idx_tooth21)
        results.append(res)

        # cargar history para gráficos combinados
        hfile = Path(res["run_dir"]) / "history.json"
        try:
            hist = json.load(open(hfile))
            all_histories[m] = hist
        except Exception:
            pass

    # CSV resumen
    csv_path = base_outdir / "summary_all_models.csv"
    keys = ["model","best_epoch","best_val_loss","test_loss",
            "test_acc","test_f1","test_iou","overall_acc","d21_f1","d21_iou","train_time_sec","run_dir"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in results: w.writerow({k: r.get(k, None) for k in keys})
    print(f"[CSV] Resumen -> {csv_path}")

    # Gráficas combinadas
    plot_combined(all_histories, base_outdir/"combined_plots")
    print(f"[PLOTS] Combinados -> {base_outdir/'combined_plots'}")


if __name__ == "__main__":
    main()
