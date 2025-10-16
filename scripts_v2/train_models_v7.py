#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_models_v7.py (versión corregida)

- Entrenamiento multicategoría con paciencia y early stopping.
- Corrección: num_classes calculado desde todos los splits (train, val, test).
- Modelos: PointNet, PointNet++, DilatedToothSegNet, Transformer 3D.
"""

import os, json, time, argparse, random
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Torchmetrics opcional
try:
    from torchmetrics.classification import (
        MulticlassAccuracy, MulticlassPrecision, MulticlassRecall,
        MulticlassF1Score, MulticlassJaccardIndex
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

def normalize_cloud(x):
    """Normaliza cada nube a esfera unitaria (por lote)."""
    centroid = x.mean(dim=1, keepdim=True)
    x = x - centroid
    furthest = torch.sqrt((x**2).sum(dim=-1))
    furthest = furthest.max(dim=1, keepdim=True)[0]
    return x / (furthest.unsqueeze(-1) + 1e-8)

def plot_curves(history, out_dir, model_name):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    metrics = ["loss","acc","f1","iou"]
    for m in metrics:
        plt.figure(figsize=(6,4))
        for split in ["train","val","test"]:
            key = f"{split}_{m}"
            if key in history and len(history[key])>0:
                plt.plot(history[key], label=split)
        plt.xlabel("Época"); plt.ylabel(m.upper())
        plt.title(f"{model_name} – {m.upper()}"); plt.legend(); plt.tight_layout()
        plt.savefig(out_dir/f"{model_name}_{m}.png", dpi=300); plt.close()

def save_json(obj, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path,"w",encoding="utf-8") as f: json.dump(obj,f,indent=2)


# ===========================================================
# ----------------------  MÉTRICAS  -------------------------
# ===========================================================

class MetricsBundle:
    def __init__(self,num_classes:int,device:torch.device):
        self.num_classes=num_classes; self.device=device
        self.has_tm=HAS_TORCHMETRICS
        if self.has_tm:
            self._acc  = MulticlassAccuracy(num_classes=num_classes,average="macro").to(device)
            self._prec = MulticlassPrecision(num_classes=num_classes,average="macro").to(device)
            self._rec  = MulticlassRecall(num_classes=num_classes,average="macro").to(device)
            self._f1   = MulticlassF1Score(num_classes=num_classes,average="macro").to(device)
            self._iou  = MulticlassJaccardIndex(num_classes=num_classes,average="macro").to(device)
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

    def compute_per_class(self):
        cm=self.cm.float(); tp=torch.diag(cm); gt=cm.sum(1); pd=cm.sum(0)
        prec=torch.nan_to_num(tp/(pd+1e-8)).cpu().numpy()
        rec=torch.nan_to_num(tp/(gt+1e-8)).cpu().numpy()
        f1=np.nan_to_num(2*prec*rec/(prec+rec+1e-8))
        iou=np.nan_to_num(tp.cpu().numpy()/(gt.cpu().numpy()+pd.cpu().numpy()-tp.cpu().numpy()+1e-8))
        oa=float(tp.sum()/max(1.,cm.sum()))
        return oa,prec,rec,f1,iou


CLASS_NAMES=["Encía"]+[f"Diente {i}" for i in [11,12,13,14,15,16,17,18,
                                                21,22,23,24,25,26,27,28,
                                                31,32,33,34,35,36,37,38,
                                                41,42,43,44,45,46,47,48]]


# ===========================================================
# -------------------  EARLY STOPPING  ----------------------
# ===========================================================

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
# ----------------------  MODELOS  --------------------------
# ===========================================================

class STN3d(nn.Module):
    """T-Net de PointNet (3×3 sin regularizador)."""
    def __init__(self,k=3):
        super().__init__()
        self.k=k
        self.conv1, self.bn1 = nn.Conv1d(k,64,1), nn.BatchNorm1d(64)
        self.conv2, self.bn2 = nn.Conv1d(64,128,1), nn.BatchNorm1d(128)
        self.conv3, self.bn3 = nn.Conv1d(128,1024,1), nn.BatchNorm1d(1024)
        self.fc1, self.bn4 = nn.Linear(1024,512), nn.BatchNorm1d(512)
        self.fc2, self.bn5 = nn.Linear(512,256), nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256,k*k)
    def forward(self,x):
        B=x.size(0)
        x=F.relu(self.bn1(self.conv1(x)))
        x=F.relu(self.bn2(self.conv2(x)))
        x=self.bn3(self.conv3(x)); x=torch.max(x,2)[0]
        x=F.relu(self.bn4(self.fc1(x))); x=F.relu(self.bn5(self.fc2(x)))
        x=self.fc3(x).view(B,self.k,self.k)
        iden=torch.eye(self.k,device=x.device).unsqueeze(0).repeat(B,1,1)
        return x+iden


class PointNetSeg(nn.Module):
    """PointNet (paper-faithful segmentation)."""
    def __init__(self,num_classes=10,dropout=0.5):
        super().__init__()
        self.input_tnet=STN3d(k=3)
        self.conv1, self.bn1 = nn.Conv1d(3,64,1), nn.BatchNorm1d(64)
        self.conv2, self.bn2 = nn.Conv1d(64,128,1), nn.BatchNorm1d(128)
        self.conv3, self.bn3 = nn.Conv1d(128,1024,1), nn.BatchNorm1d(1024)
        self.fconv1, self.bn4 = nn.Conv1d(1152,512,1), nn.BatchNorm1d(512)
        self.fconv2, self.bn5 = nn.Conv1d(512,256,1), nn.BatchNorm1d(256)
        self.dropout=nn.Dropout(dropout)
        self.fconv3=nn.Conv1d(256,num_classes,1)
    def forward(self,xyz):
        B,P,_=xyz.shape; x=xyz.transpose(2,1)
        T=self.input_tnet(x); x=torch.bmm(T,x)
        x1=F.relu(self.bn1(self.conv1(x)))
        x2=F.relu(self.bn2(self.conv2(x1)))
        x3=F.relu(self.bn3(self.conv3(x2)))
        xg=torch.max(x3,2,keepdim=True)[0].repeat(1,1,P)
        x_cat=torch.cat([xg,x2],1)
        x=F.relu(self.bn4(self.fconv1(x_cat)))
        x=F.relu(self.bn5(self.fconv2(x)))
        x=self.dropout(x)
        return self.fconv3(x).transpose(2,1)


# =================== helpers geom (PN++) ===================

def knn_indices(query, ref, k):
    """
    query: (B, Nq, 3)
    ref:   (B, Nr, 3)
    return: idx (B, Nq, k) índices en ref
    """
    d = torch.cdist(query, ref)  # (B, Nq, Nr)
    idx = torch.topk(d, k=min(k, ref.size(1)), dim=-1, largest=False).indices
    return idx  # (B, Nq, k)

def batched_gather(points, idx):
    """
    points: (B, N, C)
    idx:    (B, M, K)
    return: (B, M, K, C)
    """
    B, N, C = points.shape
    _, M, K = idx.shape
    batch = torch.arange(B, device=points.device)[:, None, None].expand(B, M, K)
    out = points[batch, idx, :]  # (B, M, K, C)
    return out

def three_nn_interp(xyz1, xyz2, feats2, k=3):
    """
    Interpolación inversa de distancias: xyz1 <- xyz2
    xyz1:   (B, N1, 3) puntos a interpolar
    xyz2:   (B, N2, 3) puntos origen
    feats2: (B, C2, N2) características en xyz2
    return: (B, C2, N1)
    """
    B, N1, _ = xyz1.shape
    _, C2, N2 = feats2.shape
    idx = knn_indices(xyz1, xyz2, k=min(k, N2))           # (B, N1, k)

    # distancias
    d = torch.cdist(xyz1, xyz2)                           # (B, N1, N2)
    knn_d = torch.gather(d, 2, idx)                       # (B, N1, k)
    knn_d = torch.clamp(knn_d, min=1e-8)
    w = 1.0 / knn_d
    w = w / w.sum(dim=-1, keepdim=True)                   # (B, N1, k)

    # recolectar feats2 en (B, N2, C2) -> vecindarios
    feats2_perm = feats2.transpose(1, 2).contiguous()     # (B, N2, C2)
    neigh = batched_gather(feats2_perm, idx)              # (B, N1, k, C2)

    # combinar pesos
    out = (w[..., None] * neigh).sum(dim=2)               # (B, N1, C2)
    return out.transpose(1, 2).contiguous()               # (B, C2, N1)

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
    """
    Set Abstraction:
     - muestreo de centros (stride/4)
     - vecindad k-NN
     - mini-PointNet sobre (coords locales [+ feats])
    """
    def __init__(self, nsample, in_ch, mlp):
        super().__init__()
        self.nsample = nsample
        self.mlp = MLP1d(in_ch + 3, mlp)
        self.out_ch = mlp[-1]

    def forward(self, xyz, feats):
        B, P, _ = xyz.shape
        M = max(1, P // 4)

        # centros (índices espaciados uniformemente)
        idx_center = torch.linspace(0, P - 1, M, device=xyz.device, dtype=torch.long)
        idx_center = idx_center[None, :].repeat(B, 1)     # (B, M)
        centers = torch.gather(xyz, 1, idx_center[..., None].expand(-1, -1, 3))  # (B, M, 3)

        # vecinos k-NN para cada centro
        idx_knn = knn_indices(centers, xyz, self.nsample)         # (B, M, K)
        neigh_xyz = batched_gather(xyz, idx_knn)                  # (B, M, K, 3)
        local_xyz = neigh_xyz - centers[:, :, None, :]            # (B, M, K, 3)

        # preparar tensor entrada MLP: (B*M, in, K)
        local_xyz = local_xyz.permute(0, 1, 3, 2).contiguous()    # (B, M, 3, K)

        if feats is not None:
            feats_perm = feats.transpose(1, 2).contiguous()       # (B, P, C)
            neigh_f = batched_gather(feats_perm, idx_knn)         # (B, M, K, C)
            neigh_f = neigh_f.permute(0, 1, 3, 2).contiguous()    # (B, M, C, K)
            cat = torch.cat([local_xyz, neigh_f], dim=2)          # (B, M, 3+C, K)
        else:
            cat = local_xyz                                       # (B, M, 3, K)

        Bm, Mm, Cm, K = cat.shape
        cat_flat = cat.view(Bm * Mm, Cm, K)                       # (B*M, in, K)
        out = self.mlp(cat_flat)                                  # (B*M, out_ch, K)
        out = torch.max(out, dim=-1, keepdim=False)[0]            # (B*M, out_ch)
        out = out.view(Bm, Mm, -1).permute(0, 2, 1).contiguous()  # (B, out_ch, M)
        return centers, out

class FP_Layer(nn.Module):
    def __init__(self, in_ch, mlp):
        super().__init__()
        self.mlp = MLP1d(in_ch, mlp)
        self.out_ch = mlp[-1]
    def forward(self, xyz1, xyz2, feats1, feats2):
        # Interpola desde xyz2->xyz1 y concat con feats1 (skip)
        interp = three_nn_interp(xyz1, xyz2, feats2)          # (B, C2, N1)
        cat = torch.cat([interp, feats1], dim=1) if feats1 is not None else interp
        return self.mlp(cat)                                   # (B, C_out, N1)

class PointNet2Seg(nn.Module):
    """
    Encoder-decoder SSG:
      SA1 -> SA2 -> SA3 -> FP3 -> FP2 -> FP1 -> head
    k-NN en SA y 3NN en FP; sin dependencias CUDA externas.
    """
    def __init__(self, num_classes=10, dropout=0.5):
        super().__init__()
        self.sa1 = SA_Layer(nsample=32,  in_ch=0,   mlp=[64, 64, 128])
        self.sa2 = SA_Layer(nsample=64,  in_ch=128, mlp=[128, 128, 256])
        self.sa3 = SA_Layer(nsample=128, in_ch=256, mlp=[256, 512, 1024])

        self.fp3 = FP_Layer(in_ch=1024 + 256, mlp=[256, 256])
        self.fp2 = FP_Layer(in_ch=256 + 128,  mlp=[256, 128])
        self.fp1 = FP_Layer(in_ch=128 + 0,    mlp=[128, 128, 128])

        self.head = nn.Sequential(
            nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Dropout(dropout), nn.Conv1d(128, num_classes, 1)
        )

    def forward(self, xyz):  # (B, P, 3)
        feats0 = None  # SA1 no tiene feats de entrada
        l1_xyz, l1 = self.sa1(xyz, feats0)          # (B, P/4, 3), (B,128,P/4)
        l2_xyz, l2 = self.sa2(l1_xyz, l1)           # (B, P/16,3), (B,256,P/16)
        l3_xyz, l3 = self.sa3(l2_xyz, l2)           # (B, P/64,3), (B,1024,P/64)

        # FP de coarse->fine
        l2n = self.fp3(l2_xyz, l3_xyz, l2, l3)      # (B,256,P/16)
        l1n = self.fp2(l1_xyz, l2_xyz, l1, l2n)     # (B,128,P/4)
        l0n = self.fp1(xyz,    l1_xyz, None, l1n)   # (B,128,P)

        out = self.head(l0n).transpose(2, 1)        # (B, P, C)
        return out


# ===========================================================
# ----------------  DilatedToothSegNet & Transformer --------
# ===========================================================

class DilatedToothSegNet(nn.Module):
    def __init__(self, num_classes=10, base=64, dropout=0.5):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv1d(3, base, 1, dilation=1), nn.BatchNorm1d(base), nn.ReLU(True))
        self.enc2 = nn.Sequential(nn.Conv1d(base, base*2, 1, dilation=2), nn.BatchNorm1d(base*2), nn.ReLU(True))
        self.enc3 = nn.Sequential(nn.Conv1d(base*2, base*4, 1, dilation=3), nn.BatchNorm1d(base*4), nn.ReLU(True))
        self.enc4 = nn.Sequential(nn.Conv1d(base*4, base*8, 1, dilation=4), nn.BatchNorm1d(base*8), nn.ReLU(True))
        self.head = nn.Sequential(
            nn.Conv1d(base*(1+2+4+8), base*4, 1), nn.BatchNorm1d(base*4), nn.ReLU(True),
            nn.Dropout(dropout), nn.Conv1d(base*4, num_classes, 1)
        )
    def forward(self, xyz):  # (B, P, 3)
        x = xyz.transpose(2,1)
        e1 = self.enc1(x); e2 = self.enc2(e1); e3 = self.enc3(e2); e4 = self.enc4(e3)
        cat = torch.cat([e1,e2,e3,e4], dim=1)
        return self.head(cat).transpose(2,1)

class Transformer3DSeg(nn.Module):
    def __init__(self, num_classes=10, dim=128, heads=8, depth=6, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(3, dim)
        enc = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, batch_first=True,
                                         dim_feedforward=dim*4, dropout=dropout)
        self.encoder = nn.TransformerEncoder(enc, num_layers=depth)
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(True), nn.Dropout(dropout),
                                  nn.Linear(dim, num_classes))
    def forward(self, xyz):  # (B, P, 3)
        x = self.proj(xyz)
        x = self.encoder(x)
        return self.head(x)


# ===========================================================
# ------------------  ENTRENAMIENTO  ------------------------
# ===========================================================

def build_model(name: str, num_classes: int, args):
    n = name.lower()
    if n == "pointnet":      return PointNetSeg(num_classes=num_classes, dropout=args.dropout)
    if n == "pointnetpp":    return PointNet2Seg(num_classes=num_classes, dropout=args.dropout)
    if n == "dilated":       return DilatedToothSegNet(num_classes=num_classes, base=args.base_channels, dropout=args.dropout)
    if n == "transformer":   return Transformer3DSeg(num_classes=num_classes, dim=args.tr_dim, heads=args.tr_heads, depth=args.tr_depth, dropout=args.dropout)
    raise ValueError(f"Modelo no soportado: {name}")

@torch.no_grad()
def _sanity_check_labels(yb, num_classes, where="train"):
    ymin = int(yb.min().item())
    ymax = int(yb.max().item())
    if ymin < 0 or ymax >= num_classes:
        raise RuntimeError(
            f"[LabelRangeError] {where}: etiquetas fuera de rango. "
            f"min={ymin}, max={ymax}, num_classes={num_classes}. "
            f"Verifica el remapeo global y que --data_path apunte al split correcto."
        )

def one_epoch(model, loader, optimizer, criterion, device, metrics_bundle: MetricsBundle=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total = 0.0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        xb = normalize_cloud(xb)  # (B,P,3)

        # Chequeo ANTES de la pérdida para evitar el assert de CUDA
        _sanity_check_labels(yb, model_output_classes(model), where=("train" if is_train else "eval"))

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(xb)  # (B,P,C)
        loss = criterion(logits.reshape(-1, logits.shape[-1]), yb.reshape(-1))

        if is_train:
            loss.backward()
            optimizer.step()

        total += float(loss.item())
        if metrics_bundle is not None:
            metrics_bundle.update(logits, yb)

    avg = total / max(1, len(loader))
    macro = metrics_bundle.compute_macro() if metrics_bundle is not None else {}
    return avg, macro

def model_output_classes(model: nn.Module) -> int:
    """Obtiene C a partir de la última capa del modelo."""
    # PointNet / Dilated -> Conv1d(..., C, 1)
    for m in reversed(list(model.modules())):
        if isinstance(m, nn.Conv1d) and m.kernel_size == (1,):
            return m.out_channels
        if isinstance(m, nn.Linear):
            # Para Transformer3DSeg (última Linear es C)
            return m.out_features
    raise RuntimeError("No se pudo inferir el número de clases de la cabeza del modelo.")

def evaluate_detailed_per_class(cm_bundle: MetricsBundle, num_classes: int):
    oa, prec, rec, f1, iou = cm_bundle.compute_per_class()
    per_class = []
    for i in range(num_classes):
        per_class.append({
            "class_id": i,
            "class_name": CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"class_{i}",
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "iou": float(iou[i]),
        })
    # diente 21
    tooth21_idx = None
    try:
        tooth21_idx = CLASS_NAMES.index("Diente 21")
    except ValueError:
        pass
    d21_metrics = None
    if tooth21_idx is not None and tooth21_idx < num_classes:
        d21_metrics = per_class[tooth21_idx]
    return oa, per_class, d21_metrics

def train_model(model_name: str, loaders, num_classes: int, args, device, base_outdir: Path):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = base_outdir / f"{model_name}_lr{args.lr}_bs{args.batch_size}_drop{args.dropout}_{timestamp}"
    ckpt_dir = run_dir / "checkpoints"
    plots_dir = run_dir / "plots"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Modelo y optimización
    model = build_model(model_name, num_classes, args).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{model_name}] Parámetros entrenables: {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience, min_lr=args.min_lr
    )
    early = EarlyStopping(patience=args.es_patience, delta=args.es_delta, ckpt_dir=ckpt_dir)

    # History
    history = {f"{sp}_{m}":[] for sp in ["train","val","test"] for m in ["loss","acc","prec","rec","f1","iou"]}

    best_val = float("inf"); best_ep = -1
    t0 = time.time()

    for ep in range(1, args.epochs+1):
        # ---- Train
        mb_tr = MetricsBundle(num_classes, device)
        tr_loss, tr_macro = one_epoch(model, loaders["train"], optimizer, criterion, device, mb_tr)

        # ---- Val
        mb_val = MetricsBundle(num_classes, device)
        val_loss, val_macro = one_epoch(model, loaders["val"], None, criterion, device, mb_val)
        scheduler.step(val_loss)

        # ---- Test (cada N épocas para curva)
        if args.eval_test_every > 0 and (ep % args.eval_test_every == 0 or ep == args.epochs):
            mb_te = MetricsBundle(num_classes, device)
            te_loss, te_macro = one_epoch(model, loaders["test"], None, criterion, device, mb_te)
        else:
            te_loss, te_macro = (np.nan, {"acc":np.nan,"prec":np.nan,"rec":np.nan,"f1":np.nan,"iou":np.nan})

        # Guardar history
        for sp, loss, mac in [("train", tr_loss, tr_macro), ("val", val_loss, val_macro), ("test", te_loss, te_macro)]:
            history[f"{sp}_loss"].append(float(loss))
            for k in ["acc","prec","rec","f1","iou"]:
                history[f"{sp}_{k}"].append(float(mac.get(k, np.nan)))

        # Log consola
        print(f"[{model_name}] Ep {ep:03d}/{args.epochs}  tr={tr_loss:.4f}  va={val_loss:.4f}  "
              f"acc={val_macro.get('acc',0):.4f}  f1={val_macro.get('f1',0):.4f}  miou={val_macro.get('iou',0):.4f}")

        # Best + EarlyStopping por paciencia
        improved = early(val_loss, model, ep)
        if improved:
            best_val = val_loss
            best_ep = ep

        if early.early_stop:
            print(f"[{model_name}] Early stopping activado en la época {ep} (best @ {best_ep}).")
            break

    # Final
    total_time = round(time.time() - t0, 2)
    torch.save({"model": model.state_dict(), "epoch": ep}, ckpt_dir/"final_model.pt")

    # Cargar mejor y evaluar test completo
    best_ckpt = ckpt_dir/"best.pt"
    if best_ckpt.exists():
        best = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(best["model"])
    else:
        print("[WARN] No se encontró best.pt, usando último estado.")
    mb_test_full = MetricsBundle(num_classes, device)
    test_loss, test_macro = one_epoch(model, loaders["test"], None, criterion, device, mb_test_full)
    oa, per_class, d21 = evaluate_detailed_per_class(mb_test_full, num_classes)

    # Guardar métricas/curvas/config
    metrics_val_test = {
        "best_epoch": int(best_ep),
        "best_val_loss": float(best_val) if best_ep > 0 else float('inf'),
        "test_loss": float(test_loss),
        **{f"test_{k}": float(v) for k,v in test_macro.items()},
        "overall_accuracy_test": float(oa),
        "tooth_21": d21 if d21 is not None else {}
    }
    save_json(metrics_val_test, run_dir/"metrics_val_test.json")
    save_json({"per_class_test": per_class}, run_dir/"metrics_detailed_test.json")
    save_json(history, run_dir/"history.json")

    # Plots
    plot_curves(history, plots_dir, model_name)

    # Resumen legible
    summary_txt = (
        f"Model: {model_name}\n"
        f"Timestamp: {timestamp}\n"
        f"Device: {device}\n"
        f"Epochs: {args.epochs}  Batch: {args.batch_size}\n"
        f"LR: {args.lr}  WD: {args.weight_decay}\n"
        f"Dropout: {args.dropout}\n"
        f"Train time (s): {total_time}\n"
        f"Best epoch: {best_ep}  Best val_loss: {best_val:.4f}\n"
        f"Test loss: {test_loss:.4f}  Test F1: {test_macro.get('f1',0):.4f}  Test mIoU: {test_macro.get('iou',0):.4f}\n"
        f"Diente 21: {json.dumps(d21) if d21 else 'N/A'}\n"
        f"\nArchivos:\n - {ckpt_dir/'best.pt'}\n - {ckpt_dir/'final_model.pt'}\n"
        f" - {run_dir/'metrics_val_test.json'}\n - {run_dir/'metrics_detailed_test.json'}\n - {run_dir/'history.json'}\n - {plots_dir}\n"
    )
    (run_dir/"run_summary.txt").write_text(summary_txt, encoding="utf-8")

    return {
        "run_dir": str(run_dir),
        "model": model_name,
        "timestamp": timestamp,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "dropout": float(args.dropout),
        "train_time_sec": float(total_time),
        "best_epoch": int(best_ep),
        "best_val_loss": float(best_val) if best_ep > 0 else float('inf'),
        "test_loss": float(test_loss),
        **{f"test_{k}": float(v) for k,v in test_macro.items()},
        "overall_accuracy_test": float(oa),
        "tooth_21_f1": float(d21["f1"]) if d21 else None,
        "tooth_21_iou": float(d21["iou"]) if d21 else None
    }


# ===========================================================
# -------------------------  MAIN  --------------------------
# ===========================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--tag", required=True, type=str)

    ap.add_argument("--model", default="all", choices=["all","pointnet","pointnetpp","dilated","transformer"])

    # Hiperparámetros
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--base_channels", type=int, default=64)  # dilated

    # Transformer
    ap.add_argument("--tr_dim", type=int, default=128)
    ap.add_argument("--tr_heads", type=int, default=8)
    ap.add_argument("--tr_depth", type=int, default=6)

    # Scheduler
    ap.add_argument("--lr_patience", type=int, default=15)
    ap.add_argument("--lr_factor", type=float, default=0.5)
    ap.add_argument("--min_lr", type=float, default=1e-6)

    # EarlyStopping
    ap.add_argument("--es_patience", type=int, default=30, help="Paciencia para early stopping (val_loss).")
    ap.add_argument("--es_delta", type=float, default=1e-4, help="Mejora mínima para resetear paciencia.")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cuda", type=int, default=None)

    # Frecuencia para medir test en curvas
    ap.add_argument("--eval_test_every", type=int, default=10, help="Evalúa test cada N épocas. 0 = sólo al final")

    args = ap.parse_args()

    set_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.cuda}") if args.cuda is not None else torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[INFO] Device: {device}  (GPUs visibles: {torch.cuda.device_count()})")

    # Loaders
    loaders = make_loaders(args.data_path, batch_size=args.batch_size)

    # === num_classes robusto (UNIÓN de splits) ===
    dp = Path(args.data_path)
    Ytr = np.load(dp/"Y_train.npz")["Y"]
    Yva = np.load(dp/"Y_val.npz")["Y"]
    Yte = np.load(dp/"Y_test.npz")["Y"]
    # chequeo informativo
    print(f"[LABELS] train: min={Ytr.min()} max={Ytr.max()} uniq={len(np.unique(Ytr))}")
    print(f"[LABELS] val  : min={Yva.min()} max={Yva.max()} uniq={len(np.unique(Yva))}")
    print(f"[LABELS] test : min={Yte.min()} max={Yte.max()} uniq={len(np.unique(Yte))}")
    num_classes = int(max(Ytr.max(), Yva.max(), Yte.max()) + 1)
    print(f"[DATA] num_classes (global) = {num_classes}")

    base_outdir = Path(args.out_dir) / args.tag
    base_outdir.mkdir(parents=True, exist_ok=True)

    models = [args.model] if args.model != "all" else ["pointnet","pointnetpp","dilated","transformer"]
    rows = []
    for m in models:
        res = train_model(m, loaders, num_classes, args, device, base_outdir)
        rows.append(res)

    # CSV resumen
    import csv
    csv_path = base_outdir / "summary_all_models.csv"
    keys = ["run_dir","model","timestamp","epochs","batch_size","lr","weight_decay","dropout",
            "train_time_sec","best_epoch","best_val_loss","test_loss",
            "test_acc","test_prec","test_rec","test_f1","test_iou",
            "overall_accuracy_test","tooth_21_f1","tooth_21_iou"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows: w.writerow({k: r.get(k, None) for k in keys})
    print(f"[CSV] Resumen -> {csv_path}")

if __name__ == "__main__":
    main()
