#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_models_v3.py
Entrenamiento y evaluación de modelos paper-faithful para segmentación dental.

Incluye:
- PointNet (Qi et al., 2017)
- PointNet++ (SSG simplificado)
- DilatedToothSegNet (bloques dilatados)
- Transformer3D (encoder simple)
- Pérdidas: CE, CE ponderada, Focal, Dice+CE

Estructura de salida:
 out_dir/tag/<modelo>/
   ├── checkpoints/
   │    ├── best.pt
   │    └── final_model.pt
   ├── history.json
   ├── metrics_val_test.json
   ├── metrics_detailed_test.json
   ├── plots/
   └── run_summary.txt
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

# Torchmetrics (opcional)
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
    """
    Normaliza cada nube a esfera unitaria por lote.
    """
    centroid = x.mean(dim=1, keepdim=True)
    x = x - centroid
    furthest = torch.sqrt((x**2).sum(dim=-1))
    furthest = furthest.max(dim=1, keepdim=True)[0]
    return x / (furthest.unsqueeze(-1) + 1e-8)

def plot_curves(history, out_dir, model_name):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    metrics = ["loss", "acc", "f1", "iou"]
    for m in metrics:
        plt.figure(figsize=(6,4))
        for split in ["train","val","test"]:
            key = f"{split}_{m}"
            if key in history and len(history[key])>0:
                plt.plot(history[key], label=split)
        plt.xlabel("Época"); plt.ylabel(m.upper()); plt.title(f"{model_name} – {m.upper()}")
        plt.legend(); plt.tight_layout()
        plt.savefig(out_dir/f"{model_name}_{m}.png", dpi=300)
        plt.close()

def save_json(obj, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

# ===========================================================
# ----------------------  MÉTRICAS  -------------------------
# ===========================================================

class MetricsBundle:
    """Métricas macro (OA, Prec, Rec, F1, mIoU) + matriz de confusión."""
    def __init__(self, num_classes: int, device: torch.device):
        self.num_classes = num_classes
        self.device = device
        self.has_tm = HAS_TORCHMETRICS
        if self.has_tm:
            self._acc  = MulticlassAccuracy(num_classes=num_classes, average="macro").to(device)
            self._prec = MulticlassPrecision(num_classes=num_classes, average="macro").to(device)
            self._rec  = MulticlassRecall(num_classes=num_classes, average="macro").to(device)
            self._f1   = MulticlassF1Score(num_classes=num_classes, average="macro").to(device)
            self._iou  = MulticlassJaccardIndex(num_classes=num_classes, average="macro").to(device)
        self.reset_cm()

    def reset_cm(self):
        self.cm = torch.zeros((self.num_classes, self.num_classes), device=self.device, dtype=torch.long)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, y_true: torch.Tensor):
        preds = logits.argmax(dim=-1)
        t = y_true.view(-1)
        p = preds.view(-1)
        valid = (t >= 0) & (t < self.num_classes)
        t = t[valid]; p = p[valid]

        idx = t * self.num_classes + p
        binc = torch.bincount(idx, minlength=self.num_classes*self.num_classes).reshape(self.num_classes, self.num_classes)
        self.cm += binc.long()

        if self.has_tm:
            self._acc.update(p, t); self._prec.update(p, t)
            self._rec.update(p, t); self._f1.update(p, t); self._iou.update(p, t)

    def compute_macro(self):
        if self.has_tm:
            res = {
                "acc":  float(self._acc.compute().item()),
                "prec": float(self._prec.compute().item()),
                "rec":  float(self._rec.compute().item()),
                "f1":   float(self._f1.compute().item()),
                "iou":  float(self._iou.compute().item()),
            }
            self._acc.reset(); self._prec.reset(); self._rec.reset(); self._f1.reset(); self._iou.reset()
            return res
        else:
            cm = self.cm.float()
            tp = torch.diag(cm)
            gt = cm.sum(1)
            pd = cm.sum(0)
            acc = (tp.sum() / cm.sum()).item() if cm.sum() > 0 else 0.0
            prec = torch.nanmean(tp / (pd + 1e-8)).item()
            rec  = torch.nanmean(tp / (gt + 1e-8)).item()
            f1   = torch.nanmean(2 * tp / (gt + pd + 1e-8)).item()
            iou  = torch.nanmean(tp / (gt + pd - tp + 1e-8)).item()
            return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "iou": iou}

    def compute_per_class(self):
        cm = self.cm.float()
        tp = torch.diag(cm)
        gt = cm.sum(1)
        pd = cm.sum(0)
        prec = torch.nan_to_num(tp / (pd + 1e-8)).cpu().numpy()
        rec  = torch.nan_to_num(tp / (gt + 1e-8)).cpu().numpy()
        f1   = np.nan_to_num(2*prec*rec / (prec+rec+1e-8))
        iou  = np.nan_to_num((tp.cpu().numpy()) / (gt.cpu().numpy() + pd.cpu().numpy() - tp.cpu().numpy() + 1e-8))
        oa   = float(tp.sum().item() / max(1.0, cm.sum().item()))
        return oa, prec, rec, f1, iou


# Clases dentales (para reportes detallados)
CLASS_NAMES = ["Encía"] + [f"Diente {i}" for i in
                           [11,12,13,14,15,16,17,18,
                            21,22,23,24,25,26,27,28,
                            31,32,33,34,35,36,37,38,
                            41,42,43,44,45,46,47,48]]


# ===========================================================
# ----------------------  MODELOS  --------------------------
# ===========================================================

# ---------- PointNet ----------
class STN3d(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        B = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=False)[0]
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x).view(B, self.k, self.k)
        iden = torch.eye(self.k, device=x.device).unsqueeze(0).repeat(B,1,1)
        return x + iden


class PointNetSeg(nn.Module):
    def __init__(self, num_classes=10, dropout=0.5):
        super().__init__()
        self.input_tnet = STN3d(k=3)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fconv1 = nn.Conv1d(1024+128, 512, 1)
        self.fconv2 = nn.Conv1d(512, 256, 1)
        self.dropout = nn.Dropout(p=dropout)
        self.fconv3 = nn.Conv1d(256, num_classes, 1)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, xyz):
        B,P,_ = xyz.shape
        x = xyz.transpose(2,1)
        T = self.input_tnet(x)
        x = torch.bmm(T, x)
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x3 = F.relu(self.bn3(self.conv3(x2)))
        xg = torch.max(x3, 2, keepdim=True)[0].repeat(1,1,P)
        x_cat = torch.cat([xg, x2], 1)
        x = F.relu(self.bn4(self.fconv1(x_cat)))
        x = F.relu(self.bn5(self.fconv2(x)))
        x = self.dropout(x)
        x = self.fconv3(x).transpose(2,1)
        return x


# ---------- PointNet++ ----------
def three_nn_interp(xyz1, xyz2, feats2, k=3):
    d = torch.cdist(xyz1, xyz2)
    d = torch.clamp(d, min=1e-8)
    knn_d, knn_i = torch.topk(d, k=min(k, xyz2.size(1)), dim=-1, largest=False)
    w = 1.0 / knn_d
    w = w / w.sum(dim=-1, keepdim=True)
    f2 = feats2.transpose(1,2)
    B = xyz1.size(0)
    neigh = f2[torch.arange(B)[:,None,None], knn_i]
    out = (w[...,None] * neigh).sum(dim=2)
    return out.transpose(1,2)

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
    def __init__(self, radius_r, nsample, in_ch, mlp):
        super().__init__()
        self.nsample = nsample
        self.mlp = MLP1d(in_ch+3, mlp)
        self.out_ch = mlp[-1]

    def forward(self, xyz, feats):
        B,P,_ = xyz.shape
        M = max(1, P//4)
        idx_center = torch.linspace(0, P-1, M, device=xyz.device, dtype=torch.long)[None,:].repeat(B,1)
        centers = torch.gather(xyz, 1, idx_center[...,None].expand(-1,-1,3))
        d = torch.cdist(centers, xyz)
        knn = torch.topk(d, k=min(self.nsample,P), dim=-1, largest=False).indices
        Bidx = torch.arange(B, device=xyz.device)[:,None,None]
        neigh_xyz = xyz[Bidx.squeeze(-1), knn]
        neigh_xyz = neigh_xyz.permute(0,1,3,2)
        if feats is not None:
            f2 = feats.transpose(1,2)
            neigh_f = f2[Bidx.squeeze(-1), knn]
            neigh_f = neigh_f.permute(0,1,3,2)
            cat = torch.cat([neigh_xyz, neigh_f], dim=2)
        else:
            cat = neigh_xyz
        Bm, Mm, Cm, K = cat.shape
        cat_flat = cat.reshape(Bm*Mm, Cm, K)
        out = self.mlp(cat_flat)
        out = torch.max(out, dim=-1, keepdim=True)[0]
        out = out.view(Bm, Mm, -1).permute(0,2,1)
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
    def __init__(self, num_classes=10, dropout=0.5):
        super().__init__()
        self.sa1 = SA_Layer(0.2, 32, 0,   [64,64,128])
        self.sa2 = SA_Layer(0.4, 64, 128, [128,128,256])
        self.sa3 = SA_Layer(0.8, 128,256, [256,512,1024])
        self.fp3 = FP_Layer(1024+256, [256,256])
        self.fp2 = FP_Layer(256+128,  [256,128])
        self.fp1 = FP_Layer(128+0,    [128,128,128])
        self.head = nn.Sequential(
            nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Conv1d(128, num_classes, 1)
        )
    def forward(self, xyz):
        feats0 = None
        l1_xyz, l1 = self.sa1(xyz, feats0)
        l2_xyz, l2 = self.sa2(l1_xyz, l1)
        l3_xyz, l3 = self.sa3(l2_xyz, l2)
        l2n = self.fp3(l2_xyz, l3_xyz, l2, l3)
        l1n = self.fp2(l1_xyz, l2_xyz, l1, l2n)
        l0n = self.fp1(xyz,    l1_xyz, feats0, l1n)
        out = self.head(l0n).transpose(2,1)
        return out


# ---------- DilatedToothSegNet ----------
class DilatedToothSegNet(nn.Module):
    def __init__(self, num_classes=10, base=64, dropout=0.5):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv1d(3, base, 1, dilation=1), nn.BatchNorm1d(base), nn.ReLU(True))
        self.enc2 = nn.Sequential(nn.Conv1d(base, base*2, 1, dilation=2), nn.BatchNorm1d(base*2), nn.ReLU(True))
        self.enc3 = nn.Sequential(nn.Conv1d(base*2, base*4, 1, dilation=3), nn.BatchNorm1d(base*4), nn.ReLU(True))
        self.enc4 = nn.Sequential(nn.Conv1d(base*4, base*8, 1, dilation=4), nn.BatchNorm1d(base*8), nn.ReLU(True))
        self.head = nn.Sequential(
            nn.Conv1d(base*(1+2+4+8), base*4, 1),
            nn.BatchNorm1d(base*4), nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Conv1d(base*4, num_classes, 1)
        )
    def forward(self, xyz):
        x = xyz.transpose(2,1)
        e1 = self.enc1(x); e2 = self.enc2(e1); e3 = self.enc3(e2); e4 = self.enc4(e3)
        cat = torch.cat([e1,e2,e3,e4], dim=1)
        out = self.head(cat).transpose(2,1)
        return out


# ---------- Transformer 3D ----------
class Transformer3DSeg(nn.Module):
    def __init__(self, num_classes=10, dim=128, heads=8, depth=6, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(3, dim)
        enc = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, batch_first=True,
                                         dim_feedforward=dim*4, dropout=dropout)
        self.encoder = nn.TransformerEncoder(enc, num_layers=depth)
        self.head = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(True), nn.Dropout(dropout),
            nn.Linear(dim, num_classes)
        )
    def forward(self, xyz):
        x = self.proj(xyz)
        x = self.encoder(x)
        return self.head(x)

# ===========================================================
# ------------------  FUNCIONES DE PÉRDIDA  -----------------
# ===========================================================

def build_class_weights(data_path, num_classes):
    """Intenta inferir pesos de clase desde Y_train.npz si existe."""
    y_train_path = Path(data_path) / "Y_train.npz"
    if not y_train_path.exists():
        return None
    y = np.load(y_train_path)["Y"].reshape(-1)
    unique, counts = np.unique(y, return_counts=True)
    weights = np.ones(num_classes, dtype=np.float32)
    inv_freq = 1.0 / (counts + 1e-6)
    inv_freq = inv_freq / inv_freq.mean()
    for i, u in enumerate(unique):
        weights[u] = inv_freq[i]
    return torch.tensor(weights, dtype=torch.float32)


def build_criterion(args, class_weights, num_classes):
    """Construye la función de pérdida elegida desde argumentos."""
    if args.loss == "ce":
        return nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    if args.loss == "ce_weighted":
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    if args.loss == "focal":
        class Focal(nn.Module):
            def __init__(self, gamma=2.0, weight=None, ls=0.0):
                super().__init__()
                self.gamma = gamma
                self.ce = nn.CrossEntropyLoss(weight=weight, label_smoothing=ls)
            def forward(self, logits, targets):
                ce = self.ce(logits, targets)
                with torch.no_grad():
                    pt = torch.exp(-ce)
                return ((1-pt)**self.gamma)*ce
        return Focal(gamma=2.0, weight=class_weights, ls=args.label_smoothing)
    if args.loss == "dice_ce":
        class DiceCE(nn.Module):
            def __init__(self, weight=None, ls=0.0, smooth=1.0):
                super().__init__()
                self.ce = nn.CrossEntropyLoss(weight=weight, label_smoothing=ls)
                self.smooth = smooth
            def forward(self, logits, y):
                ce = self.ce(logits, y)
                C = logits.shape[-1]
                y_one = F.one_hot(y.clamp_min(0), num_classes=C).float()
                probs = F.softmax(logits, dim=-1)
                num = 2*(probs*y_one).sum(dim=0) + self.smooth
                den = probs.pow(2).sum(dim=0) + y_one.pow(2).sum(dim=0) + self.smooth
                dice = 1 - (num/den).mean()
                return 0.5*ce + 0.5*dice
        return DiceCE(weight=class_weights, ls=args.label_smoothing)
    raise ValueError(f"Pérdida no soportada: {args.loss}")


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


def one_epoch(model, loader, optimizer, criterion, device, metrics_bundle=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total = 0.0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        xb = normalize_cloud(xb)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(xb)
        loss = criterion(
            logits.reshape(-1, logits.shape[-1]),
            yb.reshape(-1)
        )
        if is_train:
            loss.backward()
            optimizer.step()

        total += float(loss.item())
        if metrics_bundle is not None:
            metrics_bundle.update(logits, yb)

    avg = total / max(1, len(loader))
    macro = metrics_bundle.compute_macro() if metrics_bundle is not None else {}
    return avg, macro


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
    try:
        idx21 = CLASS_NAMES.index("Diente 21")
        d21 = per_class[idx21] if idx21 < num_classes else None
    except ValueError:
        d21 = None
    return oa, per_class, d21

def train_model(model_name: str, loaders, num_classes: int, args, device, base_outdir: Path):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = base_outdir / f"{model_name}_lr{args.lr}_bs{args.batch_size}_drop{args.dropout}_{timestamp}"
    ckpt_dir = run_dir / "checkpoints"
    plots_dir = run_dir / "plots"
    run_dir.mkdir(parents=True, exist_ok=True); ckpt_dir.mkdir(exist_ok=True); plots_dir.mkdir(exist_ok=True)

    model = build_model(model_name, num_classes, args).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{model_name}] Parámetros entrenables: {n_params:,}")

    class_weights = build_class_weights(args.data_path, num_classes)
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = build_criterion(args, class_weights, num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_factor,
        patience=args.lr_patience, min_lr=args.min_lr
    )

    history = {f"{sp}_{m}":[] for sp in ["train","val","test"] for m in ["loss","acc","prec","rec","f1","iou"]}
    best_val = float("inf"); best_ep = -1
    t0 = time.time()

    for ep in range(1, args.epochs+1):
        mb_tr = MetricsBundle(num_classes, device)
        tr_loss, tr_macro = one_epoch(model, loaders["train"], optimizer, criterion, device, mb_tr)

        mb_val = MetricsBundle(num_classes, device)
        val_loss, val_macro = one_epoch(model, loaders["val"], None, criterion, device, mb_val)
        scheduler.step(val_loss)

        if args.eval_test_every > 0 and (ep % args.eval_test_every == 0 or ep == args.epochs):
            mb_te = MetricsBundle(num_classes, device)
            te_loss, te_macro = one_epoch(model, loaders["test"], None, criterion, device, mb_te)
        else:
            te_loss, te_macro = (np.nan, {"acc":np.nan,"prec":np.nan,"rec":np.nan,"f1":np.nan,"iou":np.nan})

        for sp, loss, mac in [("train", tr_loss, tr_macro), ("val", val_loss, val_macro), ("test", te_loss, te_macro)]:
            history[f"{sp}_loss"].append(float(loss))
            for k in ["acc","prec","rec","f1","iou"]:
                history[f"{sp}_{k}"].append(float(mac.get(k, np.nan)))

        print(f"[{model_name}] Ep {ep:03d}/{args.epochs}  tr={tr_loss:.4f}  va={val_loss:.4f}  acc={val_macro.get('acc',0):.4f}  f1={val_macro.get('f1',0):.4f}  miou={val_macro.get('iou',0):.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_ep = ep
            torch.save({"model": model.state_dict(), "epoch": ep}, ckpt_dir/"best.pt")

    total_time = round(time.time() - t0, 2)
    torch.save({"model": model.state_dict(), "epoch": args.epochs}, ckpt_dir/"final_model.pt")

    best = torch.load(ckpt_dir/"best.pt", map_location=device)
    model.load_state_dict(best["model"])

    mb_test_full = MetricsBundle(num_classes, device)
    test_loss, test_macro = one_epoch(model, loaders["test"], None, criterion, device, mb_test_full)
    oa, per_class, d21 = evaluate_detailed_per_class(mb_test_full, num_classes)

    metrics_val_test = {
        "best_epoch": int(best_ep),
        "best_val_loss": float(best_val),
        "test_loss": float(test_loss),
        **{f"test_{k}": float(v) for k,v in test_macro.items()},
        "overall_accuracy_test": float(oa),
        "tooth_21": d21 if d21 else {}
    }
    save_json(metrics_val_test, run_dir/"metrics_val_test.json")
    save_json({"per_class_test": per_class}, run_dir/"metrics_detailed_test.json")
    save_json(history, run_dir/"history.json")

    plot_curves(history, plots_dir, model_name)

    summary_txt = (
        f"Model: {model_name}\nDevice: {device}\nEpochs: {args.epochs}\n"
        f"LR: {args.lr}  WD: {args.weight_decay}\nDropout: {args.dropout}\n"
        f"Best epoch: {best_ep}  Best val_loss: {best_val:.4f}\n"
        f"Test F1: {test_macro.get('f1',0):.4f}  Test mIoU: {test_macro.get('iou',0):.4f}\n"
        f"Diente 21: {json.dumps(d21) if d21 else 'N/A'}\n"
    )
    (run_dir/"run_summary.txt").write_text(summary_txt, encoding="utf-8")

    return {
        "run_dir": str(run_dir),
        "model": model_name,
        "best_epoch": best_ep,
        "test_loss": float(test_loss),
        "test_f1": float(test_macro.get("f1",0)),
        "test_iou": float(test_macro.get("iou",0)),
        "overall_accuracy": float(oa),
        "tooth_21_f1": float(d21["f1"]) if d21 else None,
        "tooth_21_iou": float(d21["iou"]) if d21 else None
    }


# ===========================================================
# -------------------------  MAIN  ---------------------------
# ===========================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--tag", required=True, type=str)
    ap.add_argument("--model", default="all", choices=["all","pointnet","pointnetpp","dilated","transformer"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--base_channels", type=int, default=64)
    ap.add_argument("--tr_dim", type=int, default=128)
    ap.add_argument("--tr_heads", type=int, default=8)
    ap.add_argument("--tr_depth", type=int, default=6)
    ap.add_argument("--lr_patience", type=int, default=10)
    ap.add_argument("--lr_factor", type=float, default=0.5)
    ap.add_argument("--min_lr", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cuda", type=int, default=None)
    ap.add_argument("--eval_test_every", type=int, default=10)
    ap.add_argument("--loss", default="ce_weighted", choices=["ce","ce_weighted","focal","dice_ce"])
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() and args.cuda is not None else "cpu")
    print(f"[INFO] Device: {device} ({torch.cuda.device_count()} GPU visibles)")

    loaders = make_loaders(args.data_path, batch_size=args.batch_size)
    Yte = np.load(Path(args.data_path)/"Y_test.npz")["Y"]
    num_classes = int(Yte.max() + 1)
    print(f"[DATA] num_classes={num_classes}")

    base_outdir = Path(args.out_dir) / args.tag
    base_outdir.mkdir(parents=True, exist_ok=True)

    models = [args.model] if args.model != "all" else ["pointnet","pointnetpp","dilated","transformer"]
    rows = []
    for m in models:
        res = train_model(m, loaders, num_classes, args, device, base_outdir)
        rows.append(res)

    import csv
    csv_path = base_outdir / "summary_all_models.csv"
    keys = ["run_dir","model","best_epoch","test_loss","test_f1","test_iou","overall_accuracy","tooth_21_f1","tooth_21_iou"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows: w.writerow(r)
    print(f"[CSV] Resumen -> {csv_path}")

if __name__ == "__main__":
    main()

