#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PointNet++ Binary (v19b) — FIX duro a indexación y shapes:
- points se normaliza a (B, N, C) antes de agrupar
- npoint y k se acotan a N
- KNN con topk
- Split reproducible, métricas globales y búsqueda de threshold
"""

import os, random, argparse, numpy as np
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# -------------------- Utils --------------------
def set_seed(sd=42):
    random.seed(sd); np.random.seed(sd)
    torch.manual_seed(sd); torch.cuda.manual_seed_all(sd)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

# -------------------- Dataset --------------------
class UFRNBinaryDataset(Dataset):
    def __init__(self, root, augment=False):
        self.root = Path(root)
        self.ids = sorted([d.name for d in self.root.iterdir()
                           if d.is_dir() and d.name.startswith("paciente_")])
        self.augment = augment

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]
        X = np.load(self.root/pid/"X.npy").astype(np.float32)   # (N,3)
        Y = np.load(self.root/pid/"Y.npy").astype(np.int64).reshape(-1)  # (N,)
        Y = np.clip(Y, 0, 1)

        if self.augment:
            X, Y = self._augment(X, Y)

        return X, Y, pid

    def _augment(self, X, Y):
        # Rotación ligera Z
        theta = np.random.uniform(0, 2*np.pi)
        rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta),  np.cos(theta), 0],
                        [0, 0, 1]], dtype=np.float32)
        X = X @ rot.T
        # Jitter
        X += np.random.normal(0, 0.01, X.shape).astype(np.float32)
        # Re-normalizar a esfera unitaria (por si la rotación movió el centro)
        c = X.mean(axis=0, keepdims=True)
        X = X - c
        r = np.linalg.norm(X, axis=1).max()
        if r > 0: X /= r
        return X, Y

def collate(batch):
    Xs, Ys, pids = zip(*batch)
    max_n = max(x.shape[0] for x in Xs)
    Xb, Yb = [], []
    for X, Y in zip(Xs, Ys):
        pad = max_n - X.shape[0]
        if pad > 0:
            X = np.pad(X, ((0, pad), (0, 0)), mode='constant', constant_values=0)
            Y = np.pad(Y, (0, pad), mode='constant', constant_values=0)
        Xb.append(X); Yb.append(Y)
    Xb = torch.tensor(np.stack(Xb), dtype=torch.float32)   # (B, Nmax, 3)
    Yb = torch.tensor(np.stack(Yb), dtype=torch.long)      # (B, Nmax)
    return Xb, Yb, list(pids)

# -------------------- Geometry --------------------
def square_distance(src, dst):
    """
    src: (B,N,3), dst: (B,M,3) -> (B,N,M)
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.transpose(2, 1))
    dist += torch.sum(src ** 2, dim=-1, keepdim=True)      # (B,N,1)
    dist += torch.sum(dst ** 2, dim=-1).unsqueeze(1)       # (B,1,M)
    return dist

def farthest_point_sample(xyz, npoint):
    """
    xyz: (B,N,3) -> idx: (B,S) con S=min(npoint,N)
    """
    device = xyz.device
    B, N, _ = xyz.shape
    S = min(npoint, N)
    centroids = torch.zeros(B, S, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    for i in range(S):
        centroids[:, i] = farthest
        centroid = xyz[torch.arange(B, device=device), farthest, :].unsqueeze(1)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]
    return centroids

def index_points(points, idx):
    """
    points: (B,N,C), idx: (B,S) -> (B,S,C)
    """
    B = points.shape[0]
    out = []
    for b in range(B):
        out.append(points[b, idx[b], :])
    return torch.stack(out, dim=0)

def batch_gather(points, idx):
    """
    points: (B,N,C), idx: (B,S,K) -> (B,S,K,C)
    """
    B, N, C = points.shape
    _, S, K = idx.shape
    out = torch.zeros((B, S, K, C), device=points.device, dtype=points.dtype)
    for b in range(B):
        out[b] = points[b, idx[b], :]
    return out

def sample_and_group(npoint, k, xyz, points=None):
    """
    xyz:    (B,N,3)
    points: (B,N,C)  (si viene como B,C,N lo convertimos antes de llamarla)
    return:
      new_xyz:   (B,S,3)    S=min(npoint,N)
      new_pts:   (B,S,K,3+C)
    """
    B, N, _ = xyz.shape
    S = min(npoint, N)
    K = min(k, N)

    fps_idx = farthest_point_sample(xyz, S)     # (B,S)
    new_xyz = index_points(xyz, fps_idx)        # (B,S,3)

    dists = square_distance(new_xyz, xyz)       # (B,S,N)
    # Usar topk sobre -dists para los K más cercanos
    _, nn_idx = torch.topk(-dists, k=K, dim=-1) # (B,S,K)

    grouped_xyz = batch_gather(xyz, nn_idx)     # (B,S,K,3)
    grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)

    if points is not None:
        grouped_points = batch_gather(points, nn_idx)  # (B,S,K,C)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # (B,S,K,3+C)
    else:
        new_points = grouped_xyz_norm  # (B,S,K,3)

    return new_xyz, new_points

# -------------------- SA block --------------------
class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, k, mlp_channels):
        super().__init__()
        self.npoint = npoint
        self.k = k
        layers, last = [], None  # last se define dinámicamente cuando llega new_points
        self.mlp_channels = mlp_channels
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

    def _build_if_needed(self, in_channels):
        if len(self.convs) == 0:
            last = in_channels
            for o in self.mlp_channels:
                self.convs.append(nn.Conv2d(last, o, 1))
                self.bns.append(nn.BatchNorm2d(o))
                last = o

    def forward(self, xyz, points=None, sanity=False, tag=""):
        """
        xyz: (B,N,3)
        points: (B,N,C)  ó  (B,C,N) -> se normaliza a (B,N,C)
        """
        if points is not None and points.dim() == 3:
            # Si recibe (B,C,N), transpone a (B,N,C)
            if points.shape[1] != xyz.shape[1] and points.shape[2] == xyz.shape[1]:
                points = points.transpose(1, 2).contiguous()
            # Si ya viene (B,N,C), lo deja igual

        new_xyz, new_points = sample_and_group(self.npoint, self.k, xyz, points)  # (B,S,K,3+Cin)

        # Preparar para MLP: (B, C_in, K, S)
        B, S, K, Cin = new_points.shape
        new_points = new_points.permute(0, 3, 2, 1).contiguous()  # (B, Cin, K, S)

        # Construir MLP si hace falta
        self._build_if_needed(in_channels=Cin)

        # Aplicar MLP
        for conv, bn in zip(self.convs, self.bns):
            new_points = F.relu(bn(conv(new_points)))  # (B, Cout, K, S)

        # Max pooling por K: (B, Cout, S)
        new_points = torch.max(new_points, dim=2)[0]
        return new_xyz, new_points  # xyz: (B,S,3), points: (B,Cout,S)

# -------------------- Modelo --------------------
class PointNet2Binary(nn.Module):
    def __init__(self, k=2, dropout=0.5):
        super().__init__()
        self.sa1 = PointNetSetAbstraction(1024, 32, [64, 64, 128])
        self.sa2 = PointNetSetAbstraction(256,  32, [128, 128, 256])
        self.sa3 = PointNetSetAbstraction(64,   32, [256, 512, 1024])
        self.fc = nn.Sequential(
            nn.Conv1d(128+256+1024, 512, 1),
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, k, 1)
        )

    def forward(self, x, sanity=False):
        # x: (B,N,3)
        l0_xyz = x
        l0_points = None

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points, sanity=sanity, tag="sa1")     # (B,128,S1)
        # para SA2 y SA3, pasamos points como (B,N,C) -> transpose a (B,S,C) dentro del bloque
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points.transpose(1,2), sanity=sanity, tag="sa2")
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points.transpose(1,2), sanity=sanity, tag="sa3")

        # Global feature: concatenar promedios por S
        feat = torch.cat([
            l1_points.mean(dim=-1),   # (B,128)
            l2_points.mean(dim=-1),   # (B,256)
            l3_points.mean(dim=-1)    # (B,1024)
        ], dim=1).unsqueeze(-1)       # (B,1408,1)

        out = self.fc(feat).transpose(2,1)  # (B,1,2)
        return out

# -------------------- Métricas --------------------
@torch.no_grad()
def metrics_global(logits, y, threshold=0.5):
    p = F.softmax(logits, dim=-1)[..., 1]
    pred = (p >= threshold).long().view(-1)
    y = y.long().view(-1)
    tp = ((pred == 1) & (y == 1)).sum().item()
    tn = ((pred == 0) & (y == 0)).sum().item()
    fp = ((pred == 1) & (y == 0)).sum().item()
    fn = ((pred == 0) & (y == 1)).sum().item()
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    rec = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    iou = tp / max(tp + fp + fn, 1)
    return dict(acc=acc, f1=f1, recall=rec, iou=iou)

# -------------------- Train --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_root", default="/home/htaucare/Tesis_Amaro/data/UFRN")
    ap.add_argument("--name", default="runs_pointnet2_binary_cls_v19b")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--bs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--pos_weight", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--sanity", action="store_true", help="Imprime shapes del primer batch")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = Path(args.data_dir)
    out = Path(args.out_root) / f"{args.name}_seed{args.seed}"
    ensure_dir(out)

    full = UFRNBinaryDataset(data, augment=False)
    ids = full.ids.copy()
    random.shuffle(ids)
    n = len(ids)
    ntr = int(0.60 * n)
    nva = int(0.25 * n)
    tr, va, te = ids[:ntr], ids[ntr:ntr+nva], ids[ntr+nva:]

    def subset(_ids, aug):
        ds = UFRNBinaryDataset(data, augment=aug)
        ds.ids = _ids
        return ds

    trd, vad, ted = subset(tr, True), subset(va, False), subset(te, False)

    dl_tr = DataLoader(trd, batch_size=args.bs, shuffle=True,  num_workers=4, collate_fn=collate, pin_memory=True)
    dl_va = DataLoader(vad, batch_size=args.bs, shuffle=False, num_workers=2, collate_fn=collate, pin_memory=True)
    dl_te = DataLoader(ted, batch_size=args.bs, shuffle=False, num_workers=2, collate_fn=collate, pin_memory=True)

    model = PointNet2Binary().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss(weight=torch.tensor([1.0, args.pos_weight], device=device))

    hist = {"tr_loss": [], "va_loss": [], "va_f1": []}
    best_f1, best_thr, patience = -1, 0.5, 0
    printed_sanity = False

    for ep in range(1, args.epochs+1):
        model.train()
        tr_loss = 0.0
        for X, Y, _ in tqdm(dl_tr, desc=f"[Train {ep}/{args.epochs}]"):
            X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)

            if args.sanity and not printed_sanity:
                print(f"[SANITY] X={tuple(X.shape)} Y={tuple(Y.shape)}")
                printed_sanity = True

            opt.zero_grad()
            logits = model(X)  # (B,1,2)
            loss = ce(logits.reshape(-1,2), Y.reshape(-1))
            loss.backward()
            opt.step()
            tr_loss += loss.item()

        # --- Validación ---
        model.eval()
        va_loss, logits_all, y_all = 0.0, [], []
        with torch.no_grad():
            for X, Y, _ in dl_va:
                X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
                logits = model(X)
                loss = ce(logits.reshape(-1,2), Y.reshape(-1))
                va_loss += loss.item()
                logits_all.append(logits.cpu())
                y_all.append(Y.cpu())
        logits_all = torch.cat(logits_all, dim=0)
        y_all = torch.cat(y_all, dim=0)

        # Buscar mejor threshold
        local_best_f1, local_best_thr = -1, 0.5
        for thr in np.linspace(0.05, 0.95, 19):
            m = metrics_global(logits_all, y_all, threshold=thr)
            if m["f1"] > local_best_f1:
                local_best_f1, local_best_thr = m["f1"], thr
        m_best = metrics_global(logits_all, y_all, threshold=local_best_thr)

        hist["tr_loss"].append(tr_loss/len(dl_tr))
        hist["va_loss"].append(va_loss/len(dl_va))
        hist["va_f1"].append(m_best["f1"])

        print(f"Ep{ep:03d}/{args.epochs} tr_loss={hist['tr_loss'][-1]:.4f} "
              f"va_loss={hist['va_loss'][-1]:.4f} va_acc={m_best['acc']:.3f} "
              f"va_f1={m_best['f1']:.3f} va_rec={m_best['recall']:.3f} va_iou={m_best['iou']:.3f} "
              f"| best_thr={local_best_thr:.2f}")

        improved = local_best_f1 > best_f1
        if improved:
            best_f1, best_thr = local_best_f1, local_best_thr
            torch.save(model.state_dict(), out/"best.pt")
            patience = 0
        else:
            patience += 1
            if patience > 20:
                print("[EARLY STOP]")
                break

    # --- Test ---
    model.load_state_dict(torch.load(out/"best.pt", map_location=device))
    model.eval()
    logits_all, y_all = [], []
    with torch.no_grad():
        for X, Y, _ in dl_te:
            X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
            logits = model(X)
            logits_all.append(logits.cpu()); y_all.append(Y.cpu())
    logits_all = torch.cat(logits_all, dim=0)
    y_all = torch.cat(y_all, dim=0)
    m_final = metrics_global(logits_all, y_all, threshold=best_thr)
    print(f"✅ TEST acc={m_final['acc']:.3f} f1={m_final['f1']:.3f} rec={m_final['recall']:.3f} iou={m_final['iou']:.3f} (thr={best_thr:.2f})")

    # Curvas
    plt.figure(figsize=(8,4))
    plt.plot(hist["tr_loss"], label="Train Loss")
    plt.plot(hist["va_loss"], label="Val Loss")
    plt.plot(hist["va_f1"], label="Val F1")
    plt.legend(); plt.grid(True)
    plt.title("PointNet++ Binary (v19b)")
    plt.xlabel("Épocas"); plt.ylabel("Valor")
    plt.tight_layout()
    plt.savefig(out/"curves.png", dpi=160)
    print(f"[DONE] Resultados guardados en: {out}")

if __name__ == "__main__":
    main()
