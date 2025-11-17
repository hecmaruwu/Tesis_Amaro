#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PointNet++ Binary CLASSIFICATION (v18)
- Corrige head a (B,2) (no (B,1,2))
- SA con canales dinámicos (3, 3+128, 3+256)
- Métricas por muestra + barrido de umbral en validación
- Guarda métricas (CSV) y curvas (PNG)
"""

import argparse, random, csv
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# ------------------ Utils ------------------
def set_seed(sd=42):
    random.seed(sd); np.random.seed(sd)
    torch.manual_seed(sd); torch.cuda.manual_seed_all(sd)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

# ------------------ Dataset ------------------
class UFRNBinaryPoints(Dataset):
    """
    Espera estructura:
      data_dir/paciente_xx/X.npy  (N,3) normalizado a esfera
      data_dir/paciente_xx/Y.npy  (N,)  etiquetas {0,1} por punto
    Para CLASIFICACIÓN por nube: gt = 1 si existe algún punto con 1, si no 0.
    """
    def __init__(self, root, augment=False, point_dropout=0.05):
        self.root = Path(root)
        self.ids = sorted([d.name for d in self.root.iterdir() if d.is_dir() and d.name.startswith("paciente_")])
        self.augment = augment
        self.point_dropout = point_dropout

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]
        X = np.load(self.root/pid/"X.npy").astype(np.float32)   # (N,3)
        Yp = np.load(self.root/pid/"Y.npy").astype(np.int64).reshape(-1)  # (N,)
        Yp = np.clip(Yp, 0, 1)
        # label por nube
        y = np.int64(Yp.sum() > 0)  # 0/1
        if self.augment:
            X = self._augment_points(X)
        return X, int(y), pid

    def _augment_points(self, X):
        # rotación Z
        th = np.random.uniform(0, 2*np.pi)
        R = np.array([[np.cos(th), -np.sin(th), 0],
                      [np.sin(th),  np.cos(th), 0],
                      [0, 0, 1]], dtype=np.float32)
        X = X @ R.T
        # jitter
        X += np.random.normal(0, 0.01, X.shape).astype(np.float32)
        # dropout de puntos (opcional)
        if np.random.rand() < self.point_dropout:
            keep = np.random.rand(X.shape[0]) > self.point_dropout
            if keep.sum() > 32:
                X = X[keep]
        # renormalizar
        c = X.mean(0, keepdims=True); X -= c
        r = np.linalg.norm(X, axis=1).max(); 
        if r > 0: X /= r
        return X

def collate(batch):
    Xs, ys, pids = zip(*batch)
    max_n = max(x.shape[0] for x in Xs)
    Xb = []
    for X in Xs:
        pad = max_n - X.shape[0]
        if pad > 0:
            X = np.pad(X, ((0, pad), (0, 0)), constant_values=0)
        Xb.append(X)
    Xb = torch.tensor(np.stack(Xb), dtype=torch.float32)  # (B,Nmax,3)
    yb = torch.tensor(ys, dtype=torch.long)               # (B,)
    return Xb, yb, pids

# ------------------ Geometría segura ------------------
def square_distance(src, dst):
    B,N,_=src.shape; _,M,_=dst.shape
    dist = -2*torch.matmul(src, dst.transpose(2,1))
    dist += torch.sum(src**2, -1, keepdim=True)
    dist += torch.sum(dst**2, -1).unsqueeze(1)
    return dist

def farthest_point_sample(xyz, npoint):
    B,N,_=xyz.shape; npoint=min(npoint,N)
    device=xyz.device
    centroids=torch.zeros(B,npoint,dtype=torch.long,device=device)
    distance=torch.full((B,N),1e10,device=device)
    farthest=torch.randint(0,N,(B,),device=device)
    for i in range(npoint):
        centroids[:,i]=farthest
        centroid=xyz[torch.arange(B,device=device),farthest,:].view(B,1,3)
        dist=torch.sum((xyz-centroid)**2,-1)
        mask=dist<distance; distance[mask]=dist[mask]
        farthest=distance.max(-1)[1]
    return centroids

def safe_batch_index(points, idx):
    """points:(B,N,C), idx:(B,S,K)->(B,S,K,C)"""
    B,N,C = points.shape
    S,K = idx.shape[1:3]
    out = torch.zeros((B,S,K,C), device=points.device, dtype=points.dtype)
    for b in range(B):
        valid = idx[b].clamp(0, N-1)
        out[b] = points[b][valid]
    return out

def sample_and_group(npoint, k, xyz, points=None):
    B,N,_=xyz.shape
    npoint=min(npoint,N); k=min(k,N)
    fps_idx=farthest_point_sample(xyz,npoint)
    new_xyz = safe_batch_index(xyz, fps_idx.unsqueeze(-1)).squeeze(2)  # (B,npoint,3)
    dists = square_distance(new_xyz, xyz)
    idx = dists.argsort()[:, :, :k]                                   # (B,npoint,k)
    grouped_xyz = safe_batch_index(xyz, idx)                          # (B,npoint,k,3)
    grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)
    if points is not None and points.shape[-1] > 0:
        grouped_pts = safe_batch_index(points, idx)                   # (B,npoint,k,C)
        new_points = torch.cat([grouped_xyz_norm, grouped_pts], dim=-1)
    else:
        new_points = grouped_xyz_norm
    return new_xyz, new_points  # (B,npoint,3), (B,npoint,k,3+C)

# ------------------ SA con canales dinámicos ------------------
class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, k, in_channels, out_channels):
        super().__init__()
        self.npoint=npoint; self.k=k
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, out_channels, 1), nn.BatchNorm2d(out_channels), nn.ReLU()
        )
    def forward(self, xyz, points=None):
        xyz=xyz.float(); 
        if points is not None: points=points.float()
        new_xyz, new_points = sample_and_group(self.npoint,self.k,xyz,points) # (B,S,K,3+C)
        new_points = new_points.permute(0,3,2,1).contiguous()                  # (B,C,K,S)
        feat = self.conv(new_points)                                          # (B,out,K,S)
        feat = torch.max(feat, 2)[0]                                          # (B,out,S)
        return new_xyz, feat

# ------------------ Modelo (head clasificador Bx2) ------------------
class PointNet2BinaryCLS(nn.Module):
    def __init__(self, k=2, dropout=0.5):
        super().__init__()
        self.sa1 = PointNetSetAbstraction(1024, 32, 3,     128)   # in=3
        self.sa2 = PointNetSetAbstraction(256,  32, 3+128, 256)   # in=3+128
        self.sa3 = PointNetSetAbstraction(64,   32, 3+256, 512)   # in=3+256
        self.head = nn.Sequential(
            nn.Linear(128+256+512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, k)
        )
    def forward(self, x):  # x:(B,N,3)
        l0_xyz = x.float(); l0_pts = None
        l1_xyz, l1_feat = self.sa1(l0_xyz, l0_pts)              # (B,128,S1)
        l2_xyz, l2_feat = self.sa2(l1_xyz, l1_feat.transpose(1,2))  # in (B,S1,3+128)
        l3_xyz, l3_feat = self.sa3(l2_xyz, l2_feat.transpose(1,2))  # in (B,S2,3+256)
        # Global pooling (mean) en cada nivel
        g1 = l1_feat.mean(dim=-1)   # (B,128)
        g2 = l2_feat.mean(dim=-1)   # (B,256)
        g3 = l3_feat.mean(dim=-1)   # (B,512)
        g  = torch.cat([g1,g2,g3], dim=1)  # (B,896)
        return self.head(g)  # (B,2)

# ------------------ Métricas por muestra ------------------
@torch.no_grad()
def metrics_from_logits(logits, y_true, thr=0.5):
    # logits: (B,2), y_true: (B,)
    p1 = F.softmax(logits, dim=-1)[:,1]      # (B,)
    pred = (p1 >= thr).long()
    y = y_true.long()

    tp = ((pred==1)&(y==1)).sum().item()
    tn = ((pred==0)&(y==0)).sum().item()
    fp = ((pred==1)&(y==0)).sum().item()
    fn = ((pred==0)&(y==1)).sum().item()

    acc = (tp+tn)/max(tp+tn+fp+fn,1)
    rec = tp / max(tp+fn,1)
    prec = tp / max(tp+fp,1)
    f1 = 2*prec*rec / max(prec+rec,1e-12)
    iou = tp / max(tp+fp+fn,1)
    return dict(acc=acc, recall=rec, f1=f1, iou=iou), p1.cpu().numpy()

def best_threshold(val_logits, val_y, grid=None):
    if grid is None:
        grid = np.linspace(0.10, 0.90, 33)
    best_f1, best_t = -1, 0.5
    for t in grid:
        p = (val_logits >= t).astype(np.int64)
        y = val_y.astype(np.int64)
        tp = ((p==1)&(y==1)).sum()
        fp = ((p==1)&(y==0)).sum()
        fn = ((p==0)&(y==1)).sum()
        prec = tp / max(tp+fp,1)
        rec  = tp / max(tp+fn,1)
        f1 = 2*prec*rec / max(prec+rec,1e-12)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1

# ------------------ Train ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_root", default="/home/htaucare/Tesis_Amaro/data/UFRN")
    ap.add_argument("--name", default="runs_pointnet2_binary_cls_v18")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--bs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--pos_weight", type=float, default=8.0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_root) / f"{args.name}_seed{args.seed}"
    ensure_dir(out_dir)

    # data
    full = UFRNBinaryPoints(args.data_dir, augment=False)
    ids = full.ids; n = len(ids)
    ntr = int(0.7*n); nva = int(0.15*n)
    tr_ids, va_ids, te_ids = ids[:ntr], ids[ntr:ntr+nva], ids[ntr+nva:]

    def subset(ids, aug):
        ds = UFRNBinaryPoints(args.data_dir, augment=aug)
        ds.ids = ids
        return ds

    dl_tr = DataLoader(subset(tr_ids, True),  batch_size=args.bs, shuffle=True,  num_workers=4, collate_fn=collate)
    dl_va = DataLoader(subset(va_ids, False), batch_size=args.bs, shuffle=False, num_workers=2, collate_fn=collate)
    dl_te = DataLoader(subset(te_ids, False), batch_size=args.bs, shuffle=False, num_workers=2, collate_fn=collate)

    # model/opt
    model = PointNet2BinaryCLS().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ce  = nn.CrossEntropyLoss(weight=torch.tensor([1.0, args.pos_weight], device=device))
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    hist = {"tr_loss":[], "va_acc":[], "va_f1":[], "va_rec":[], "va_iou":[]}
    best_f1, best_state = -1, None
    best_thr = 0.5
    patience, wait = 20, 0

    for ep in range(1, args.epochs+1):
        # -------- train --------
        model.train(); tr_loss = 0.0
        for X, y, _ in tqdm(dl_tr, desc=f"[Train {ep}/{args.epochs}]"):
            X, y = X.to(device), y.to(device)      # X:(B,N,3), y:(B,)
            opt.zero_grad()
            with torch.amp.autocast('cuda', enabled=args.amp, dtype=torch.float32):
                logits = model(X)                 # (B,2)
                loss = ce(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tr_loss += loss.item()

        # -------- val (acumular probs y labels para barrer umbral) --------
        model.eval()
        va_logits_all, va_y_all = [], []
        va_acc=va_f1=va_rec=va_iou=0.0; nb=0
        with torch.no_grad():
            for X, y, _ in dl_va:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                m, p1 = metrics_from_logits(logits, y, 0.5)  # se informan a 0.5, pero se ajusta luego
                va_acc += m["acc"]; va_f1 += m["f1"]; va_rec += m["recall"]; va_iou += m["iou"]; nb += 1
                va_logits_all.append(p1)                    # probs clase 1
                va_y_all.append(y.cpu().numpy())
        va_acc/=max(nb,1); va_f1/=max(nb,1); va_rec/=max(nb,1); va_iou/=max(nb,1)

        # buscar mejor umbral con TODO el set de validación
        va_logits_all = np.concatenate(va_logits_all, axis=0)
        va_y_all      = np.concatenate(va_y_all, axis=0)
        thr, f1_at_thr = best_threshold(va_logits_all, va_y_all)

        hist["tr_loss"].append(tr_loss/len(dl_tr))
        hist["va_acc"].append(va_acc)
        hist["va_f1"].append(va_f1)
        hist["va_rec"].append(va_rec)
        hist["va_iou"].append(va_iou)

        print(f"Ep{ep:03d}/{args.epochs} tr_loss={tr_loss/len(dl_tr):.4f} "
              f"va_acc={va_acc:.3f} va_f1@0.5={va_f1:.3f} va_rec={va_rec:.3f} va_iou={va_iou:.3f} "
              f"| best_thr={thr:.2f} (f1={f1_at_thr:.3f})")

        # early stop por F1 a umbral optimizado
        if f1_at_thr > best_f1:
            best_f1 = f1_at_thr
            best_state = { "model": model.state_dict(), "thr": thr, "ep": ep }
            torch.save(best_state, out_dir/"best.pt")
            wait = 0
            best_thr = thr
        else:
            wait += 1
            if wait > patience:
                print("[EARLY STOP]")
                break

    # -------- Test con mejor umbral --------
    ckpt = torch.load(out_dir/"best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    best_thr = ckpt["thr"]
    model.eval(); acc=f1=rec=iou=0.0; nb=0
    with torch.no_grad():
        for X, y, _ in dl_te:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            m,_ = metrics_from_logits(logits, y, best_thr)
            acc+=m["acc"]; f1+=m["f1"]; rec+=m["recall"]; iou+=m["iou"]; nb+=1
    acc/=nb; f1/=nb; rec/=nb; iou/=nb
    print(f"\n✅ TEST (thr={best_thr:.2f}) acc={acc:.3f} f1={f1:.3f} rec={rec:.3f} iou={iou:.3f}")

    # Curvas
    plt.figure(figsize=(10,5))
    plt.plot(hist["tr_loss"], label="train loss")
    plt.plot(hist["va_acc"], label="val acc@0.5")
    plt.plot(hist["va_f1"], label="val f1@0.5")
    plt.plot(hist["va_rec"], label="val rec@0.5")
    plt.plot(hist["va_iou"], label="val IoU@0.5")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(out_dir/"curves.png", dpi=160)

    # CSV
    with open(out_dir/"metrics.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["epoch","train_loss","val_acc@0.5","val_f1@0.5","val_rec@0.5","val_iou@0.5"])
        for i in range(len(hist["tr_loss"])):
            w.writerow([i+1, hist["tr_loss"][i], hist["va_acc"][i], hist["va_f1"][i], hist["va_rec"][i], hist["va_iou"][i]])
    print(f"[DONE] resultados → {out_dir}")
    

if __name__ == "__main__":
    main()
