#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_toothformer.py (v1.0) – ToothFormer ligero compatible con el flujo v2_patience

- Compatible con data_path que contiene: X_train.npz, Y_train.npz, X_val.npz, Y_val.npz, X_test.npz, Y_test.npz (claves "X"/"Y")
- Salidas idénticas al resto de scripts:
    runs/<tag>/toothformer_lr{...}_bs{...}_drop{...}_{timestamp}/
      - checkpoints/best.pt, final_model.pt
      - metrics_val_test.json
      - metrics_detailed_test.json (por clase, incluye diente 21 si existe)
      - history.json
      - plots/*.png
      - run_summary.txt

- Arquitectura: ToothFormer ligero
  * Proyección inicial a d_model
  * Bloques locales con atención sobre k-NN (relpos)
  * Downsample jerárquico (stride=4) con pooling
  * Upsample por interpolación 3-NN (como PointNet++)
  * Cabeza punto a punto (segm)

- RTX 3090: usar --batch_size 4..8 con --amp para velocidad/VRAM
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
        assert self.X.shape[0] == self.Y.shape[0], "X e Y deben tener el mismo N"
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
    """Normaliza cada nube a esfera unitaria por lote (B,P,3)."""
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
    """Métricas macro + matriz de confusión por clase."""
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
# --------------------  GEOM HELPERS  -----------------------
# ===========================================================

def knn_indices(query, ref, k):
    """
    query: (B, Nq, 3)
    ref:   (B, Nr, 3)
    return: idx (B, Nq, k) índices en ref
    """
    d = torch.cdist(query, ref)  # (B, Nq, Nr)
    idx = torch.topk(d, k=min(k, ref.size(1)), dim=-1, largest=False).indices
    return idx

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

    d = torch.cdist(xyz1, xyz2)                           # (B, N1, N2)
    knn_d = torch.gather(d, 2, idx)                       # (B, N1, k)
    knn_d = torch.clamp(knn_d, min=1e-8)
    w = 1.0 / knn_d
    w = w / w.sum(dim=-1, keepdim=True)                   # (B, N1, k)

    feats2_perm = feats2.transpose(1, 2).contiguous()     # (B, N2, C2)
    neigh = batched_gather(feats2_perm, idx)              # (B, N1, k, C2)

    out = (w[..., None] * neigh).sum(dim=2)               # (B, N1, C2)
    return out.transpose(1, 2).contiguous()               # (B, C2, N1)


# ===========================================================
# ------------------  TOOTHFORMER LIGERO  -------------------
# ===========================================================

class LocalSelfAttention(nn.Module):
    """
    Atención local sobre vecindarios k-NN con codificación posicional relativa simple.
    """
    def __init__(self, dim, heads=8, k=32, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.k = k
        self.scale = (dim // heads) ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        # Positional encoding (MLP sobre delta xyz)
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, dim), nn.ReLU(True),
            nn.Linear(dim, dim)
        )

    def forward(self, xyz, feats):
        """
        xyz:   (B, P, 3)
        feats: (B, P, D)
        """
        B, P, D = feats.shape
        idx = knn_indices(xyz, xyz, k=min(self.k, P))     # (B, P, K)
        neigh_xyz = batched_gather(xyz, idx)              # (B, P, K, 3)
        center = xyz[:, :, None, :].expand_as(neigh_xyz)  # (B, P, K, 3)
        relpos = neigh_xyz - center                       # (B, P, K, 3)

        q = self.to_q(feats).view(B, P, self.heads, D//self.heads)  # (B,P,H,d)
        k = self.to_k(feats).view(B, P, self.heads, D//self.heads)
        v = self.to_v(feats).view(B, P, self.heads, D//self.heads)

        # gather k y v por vecinos
        # -> primero a (B,P,D) para usar batched_gather
        k_full = k.reshape(B, P, D)
        v_full = v.reshape(B, P, D)
        k_nei = batched_gather(k_full, idx).view(B, P, self.k, self.heads, D//self.heads)  # (B,P,K,H,d)
        v_nei = batched_gather(v_full, idx).view(B, P, self.k, self.heads, D//self.heads)

        # relpos embedding
        pos_bias = self.pos_mlp(relpos)  # (B,P,K,D)
        pos_bias = pos_bias.view(B, P, self.k, self.heads, D//self.heads)

        # attn
        q = q[:, :, None, :, :]                      # (B,P,1,H,d)
        attn = (q * (k_nei + pos_bias)).sum(-1) * self.scale   # (B,P,K,H)
        attn = F.softmax(attn, dim=2)
        attn = self.dropout(attn)

        out = (attn[..., None] * (v_nei + pos_bias)).sum(dim=2)  # (B,P,H,d)
        out = out.reshape(B, P, D)
        return self.proj(out)


class TFBlock(nn.Module):
    def __init__(self, dim, heads=8, k=32, mlp_ratio=4, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = LocalSelfAttention(dim, heads=heads, k=k, dropout=drop)
        self.norm2 = nn.LayerNorm(dim)
        hidden = dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, dim), nn.Dropout(drop)
        )
    def forward(self, xyz, x):
        x = x + self.attn(xyz, self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Downsample(nn.Module):
    """
    Downsample por stride fijo (1/4) + proyección.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
    def forward(self, xyz, x):
        B, P, D = x.shape
        M = max(1, P // 4)
        idx = torch.linspace(0, P-1, M, device=x.device, dtype=torch.long)[None, :].repeat(B,1)
        xyz_ds = torch.gather(xyz, 1, idx[..., None].expand(-1,-1,3))      # (B,M,3)
        x_ds   = torch.gather(x,   1, idx[..., None].expand(-1,-1,D))      # (B,M,D)
        return xyz_ds, self.proj(x_ds)


class UpSampleFP(nn.Module):
    """Upsample por interpolación 3-NN (igual a FP de PointNet++)."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
    def forward(self, xyz_low, xyz_high, feat_low, feat_high_skip=None):
        # Interpola feat_low -> xyz_high
        # feat_low: (B,M,C_low)   xyz_low: (B,M,3)
        # xyz_high: (B,N,3)
        B, N, _ = xyz_high.shape
        C_low = feat_low.shape[-1]
        feat_low_ch = feat_low.transpose(1, 2).contiguous()  # (B,C_low,M)
        interp = three_nn_interp(xyz_high, xyz_low, feat_low_ch.transpose(1,2))  # (B,C_low,N)
        interp = interp.transpose(1, 2).contiguous()  # (B,N,C_low)
        if feat_high_skip is not None:
            cat = torch.cat([interp, feat_high_skip], dim=-1)
        else:
            cat = interp
        return self.proj(cat)  # (B,N,out_dim)


class ToothFormerSeg(nn.Module):
    """
    ToothFormer ligero jerárquico:
      - Embedding inicial
      - [TFBlock]*L1 -> Down -> [TFBlock]*L2 -> Down -> [TFBlock]*L3
      - Up con FP -> TFBlock -> Up con FP -> Head

    Dimensiones por defecto pensadas para 8192 puntos y batch 4–8 en 24GB.
    """
    def __init__(self, num_classes=10, dim=128, heads=8, depth=(2,2,2), k=(32,48,64), mlp_ratio=4, drop=0.1):
        super().__init__()
        self.embed = nn.Linear(3, dim)

        self.stage1 = nn.ModuleList([TFBlock(dim, heads=heads, k=k[0], mlp_ratio=mlp_ratio, drop=drop) for _ in range(depth[0])])
        self.down1  = Downsample(dim, dim*2)

        self.stage2 = nn.ModuleList([TFBlock(dim*2, heads=heads, k=k[1], mlp_ratio=mlp_ratio, drop=drop) for _ in range(depth[1])])
        self.down2  = Downsample(dim*2, dim*4)

        self.stage3 = nn.ModuleList([TFBlock(dim*4, heads=heads, k=k[2], mlp_ratio=mlp_ratio, drop=drop) for _ in range(depth[2])])

        self.up2 = UpSampleFP(in_dim=dim*4 + dim*2, out_dim=dim*2)
        self.up1 = UpSampleFP(in_dim=dim*2 + dim,   out_dim=dim)

        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(True),
            nn.Dropout(drop),
            nn.Linear(dim, num_classes)
        )

    def forward(self, xyz):  # (B,P,3)
        x0 = self.embed(xyz)  # (B,P,dim)

        # Stage 1
        x1 = x0
        for blk in self.stage1: x1 = blk(xyz, x1)
        xyz2, x2d = self.down1(xyz, x1)

        # Stage 2
        x2 = x2d
        for blk in self.stage2: x2 = blk(xyz2, x2)
        xyz3, x3d = self.down2(xyz2, x2)

        # Stage 3
        x3 = x3d
        for blk in self.stage3: x3 = blk(xyz3, x3)

        # Upsample (FP)
        up2 = self.up2(xyz3, xyz2, x3, feat_high_skip=x2)   # (B,P/4,dim*2)
        up1 = self.up1(xyz2, xyz, up2,  feat_high_skip=x1)  # (B,P,dim)

        out = self.head(up1)  # (B,P,C)
        return out


# ===========================================================
# ------------------  ENTRENAMIENTO  ------------------------
# ===========================================================

def one_epoch(model, loader, optimizer, criterion, device, metrics_bundle: MetricsBundle=None, use_amp=True):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total = 0.0
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and (device.type=="cuda")))
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        xb = normalize_cloud(xb)  # (B,P,3)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(use_amp and (device.type=="cuda"))):
            logits = model(xb)  # (B,P,C)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), yb.reshape(-1))

        if is_train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

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


class EarlyStopping:
    """Early stopping sobre val_loss con guardado del mejor checkpoint."""
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


def train_model(loaders, num_classes: int, args, device, base_outdir: Path):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = "toothformer"
    run_dir = base_outdir / f"{model_name}_lr{args.lr}_bs{args.batch_size}_drop{args.dropout}_{timestamp}"
    ckpt_dir = run_dir / "checkpoints"
    plots_dir = run_dir / "plots"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Modelo
    model = ToothFormerSeg(
        num_classes=num_classes,
        dim=args.dim, heads=args.heads,
        depth=(args.depth1, args.depth2, args.depth3),
        k=(args.k1, args.k2, args.k3),
        mlp_ratio=args.mlp_ratio, drop=args.dropout
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[ToothFormer] Parámetros entrenables: {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience, min_lr=args.min_lr
    )
    early = EarlyStopping(patience=args.es_patience, delta=args.es_delta, ckpt_dir=ckpt_dir)

    history = {f"{sp}_{m}":[] for sp in ["train","val","test"] for m in ["loss","acc","prec","rec","f1","iou"]}
    best_val = float("inf"); best_ep = -1
    t0 = time.time()

    for ep in range(1, args.epochs+1):
        # ---- Train
        mb_tr = MetricsBundle(num_classes, device)
        tr_loss, tr_macro = one_epoch(model, loaders["train"], optimizer, criterion, device, mb_tr, use_amp=args.amp)

        # ---- Val
        mb_val = MetricsBundle(num_classes, device)
        val_loss, val_macro = one_epoch(model, loaders["val"], None, criterion, device, mb_val, use_amp=args.amp)
        scheduler.step(val_loss)

        # ---- Test para curva
        if args.eval_test_every > 0 and (ep % args.eval_test_every == 0 or ep == args.epochs):
            mb_te = MetricsBundle(num_classes, device)
            te_loss, te_macro = one_epoch(model, loaders["test"], None, criterion, device, mb_te, use_amp=args.amp)
        else:
            te_loss, te_macro = (np.nan, {"acc":np.nan,"prec":np.nan,"rec":np.nan,"f1":np.nan,"iou":np.nan})

        # History
        for sp, loss, mac in [("train", tr_loss, tr_macro), ("val", val_loss, val_macro), ("test", te_loss, te_macro)]:
            history[f"{sp}_loss"].append(float(loss))
            for k in ["acc","prec","rec","f1","iou"]:
                history[f"{sp}_{k}"].append(float(mac.get(k, np.nan)))

        # Log
        print(f"[ToothFormer] Ep {ep:03d}/{args.epochs}  tr={tr_loss:.4f}  va={val_loss:.4f}  "
              f"acc={val_macro.get('acc',0):.4f}  f1={val_macro.get('f1',0):.4f}  miou={val_macro.get('iou',0):.4f}")

        improved = early(val_loss, model, ep)
        if improved:
            best_val = val_loss
            best_ep = ep

        if early.early_stop:
            print(f"[ToothFormer] Early stopping en la época {ep} (best @ {best_ep}).")
            break

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
    test_loss, test_macro = one_epoch(model, loaders["test"], None, criterion, device, mb_test_full, use_amp=args.amp)
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
    plot_curves(history, plots_dir, model_name="toothformer")

    # Resumen legible
    summary_txt = (
        f"Model: ToothFormer (ligero)\n"
        f"Timestamp: {timestamp}\n"
        f"Device: {device}\n"
        f"Epochs: {args.epochs}  Batch: {args.batch_size}\n"
        f"LR: {args.lr}  WD: {args.weight_decay}\n"
        f"Dropout: {args.dropout}\n"
        f"AMP: {args.amp}\n"
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
        "model": "toothformer",
        "timestamp": timestamp,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "dropout": float(args.dropout),
        "amp": bool(args.amp),
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

    # Hiperparámetros generales
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--batch_size", type=int, default=6)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--dropout", type=float, default=0.1)

    # ToothFormer dims
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--depth1", type=int, default=2)
    ap.add_argument("--depth2", type=int, default=2)
    ap.add_argument("--depth3", type=int, default=2)
    ap.add_argument("--k1", type=int, default=32)
    ap.add_argument("--k2", type=int, default=48)
    ap.add_argument("--k3", type=int, default=64)
    ap.add_argument("--mlp_ratio", type=int, default=4)

    # Scheduler
    ap.add_argument("--lr_patience", type=int, default=15)
    ap.add_argument("--lr_factor", type=float, default=0.5)
    ap.add_argument("--min_lr", type=float, default=1e-6)

    # EarlyStopping
    ap.add_argument("--es_patience", type=int, default=30)
    ap.add_argument("--es_delta", type=float, default=1e-4)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cuda", type=int, default=None)
    ap.add_argument("--eval_test_every", type=int, default=10, help="Evalúa test cada N épocas. 0 = sólo al final")
    ap.add_argument("--amp", action="store_true", help="Usar AMP (mixed precision)")

    args = ap.parse_args()

    set_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.cuda}") if args.cuda is not None else torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[INFO] Device: {device}  (GPUs visibles: {torch.cuda.device_count()})")

    # Loaders
    loaders = make_loaders(args.data_path, batch_size=args.batch_size)

    # num_classes desde Y_test
    Yte = np.load(Path(args.data_path)/"Y_test.npz")["Y"]
    num_classes = int(Yte.max() + 1)
    print(f"[DATA] num_classes={num_classes}")

    base_outdir = Path(args.out_dir) / args.tag
    base_outdir.mkdir(parents=True, exist_ok=True)

    res = train_model(loaders, num_classes, args, device, base_outdir)

    # CSV resumen único (para homogeneidad con v2_patience cuando ejecutas varios)
    import csv
    csv_path = base_outdir / "summary_toothformer.csv"
    keys = ["run_dir","model","timestamp","epochs","batch_size","lr","weight_decay","dropout","amp",
            "train_time_sec","best_epoch","best_val_loss","test_loss",
            "test_acc","test_prec","test_rec","test_f1","test_iou",
            "overall_accuracy_test","tooth_21_f1","tooth_21_iou"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        w.writerow({k: res.get(k, None) for k in keys})
    print(f"[CSV] Resumen -> {csv_path}")

if __name__ == "__main__":
    main()
