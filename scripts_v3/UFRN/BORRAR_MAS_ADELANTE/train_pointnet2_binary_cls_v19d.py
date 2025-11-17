#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PointNet++ Binary (v19d)
Corrección final:
- Cada SA recibe el número correcto de canales de entrada (3, 131, 259)
- Sin errores de conv2d, ni mezcla CPU/GPU
- Split reproducible + métricas globales + búsqueda de threshold
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
        X = np.load(self.root/pid/"X.npy").astype(np.float32)
        Y = np.load(self.root/pid/"Y.npy").astype(np.int64).reshape(-1)
        Y = np.clip(Y, 0, 1)
        if self.augment: X, Y = self._augment(X, Y)
        return X, Y, pid

    def _augment(self, X, Y):
        theta = np.random.uniform(0, 2*np.pi)
        rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta),  np.cos(theta), 0],
                        [0, 0, 1]], dtype=np.float32)
        X = X @ rot.T
        X += np.random.normal(0, 0.01, X.shape).astype(np.float32)
        return X, Y

def collate(batch):
    Xs, Ys, pids = zip(*batch)
    max_n = max(x.shape[0] for x in Xs)
    Xb, Yb = [], []
    for X, Y in zip(Xs, Ys):
        pad = max_n - X.shape[0]
        if pad > 0:
            X = np.pad(X, ((0, pad), (0, 0)), constant_values=0)
            Y = np.pad(Y, (0, pad), constant_values=0)
        Xb.append(X); Yb.append(Y)
    return torch.tensor(np.stack(Xb), dtype=torch.float32), torch.tensor(np.stack(Yb), dtype=torch.long), pids

# -------------------- Geometry --------------------
def square_distance(src, dst):
    B, N, _ = src.shape; _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.transpose(2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def farthest_point_sample(xyz, npoint):
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
    return torch.stack([p[i, :] for p, i in zip(points, idx)], dim=0)

def sample_and_group(npoint, k, xyz, points=None):
    B, N, _ = xyz.shape
    S = min(npoint, N); K = min(k, N)
    fps_idx = farthest_point_sample(xyz, S)
    new_xyz = index_points(xyz, fps_idx)
    dists = square_distance(new_xyz, xyz)
    _, nn_idx = torch.topk(-dists, K, dim=-1)
    grouped_xyz = torch.stack([xyz[b, nn_idx[b], :] for b in range(B)], dim=0)
    grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)
    if points is not None:
        grouped_points = torch.stack([points[b, nn_idx[b], :] for b in range(B)], dim=0)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
    return new_xyz, new_points

# -------------------- SA block --------------------
class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, k, in_channels, mlp_channels):
        super().__init__()
        self.npoint, self.k = npoint, k
        layers, last = [], in_channels
        for o in mlp_channels:
            layers += [nn.Conv2d(last, o, 1), nn.BatchNorm2d(o), nn.ReLU()]
            last = o
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz, points=None):
        if points is not None and points.dim() == 3:
            if points.shape[1] != xyz.shape[1] and points.shape[2] == xyz.shape[1]:
                points = points.transpose(1, 2).contiguous()
        new_xyz, new_points = sample_and_group(self.npoint, self.k, xyz, points)
        B, S, K, Cin = new_points.shape
        new_points = new_points.permute(0, 3, 2, 1).contiguous()
        new_points = self.mlp(new_points)
        new_points = torch.max(new_points, dim=2)[0]
        return new_xyz, new_points

# -------------------- Modelo --------------------
class PointNet2Binary(nn.Module):
    def __init__(self, k=2, dropout=0.5):
        super().__init__()
        # Canal real de entrada por bloque
        self.sa1 = PointNetSetAbstraction(1024, 32, 3, [64, 64, 128])
        self.sa2 = PointNetSetAbstraction(256, 32, 128 + 3, [128, 128, 256])
        self.sa3 = PointNetSetAbstraction(64, 32, 256 + 3, [256, 512, 1024])
        self.fc = nn.Sequential(
            nn.Conv1d(128 + 256 + 1024, 512, 1),
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, k, 1)
        )

    def forward(self, x):
        l0_xyz, l0_points = x, None
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points.transpose(1, 2))
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points.transpose(1, 2))
        feat = torch.cat([l1_points.mean(-1), l2_points.mean(-1), l3_points.mean(-1)], dim=1).unsqueeze(-1)
        out = self.fc(feat)
        return out.transpose(2, 1)

# -------------------- Métricas --------------------
@torch.no_grad()
def metrics_global(logits, y, thr=0.5):
    p = F.softmax(logits, dim=-1)[..., 1]
    pred = (p >= thr).long().view(-1); y = y.view(-1)
    tp = ((pred==1)&(y==1)).sum().item()
    tn = ((pred==0)&(y==0)).sum().item()
    fp = ((pred==1)&(y==0)).sum().item()
    fn = ((pred==0)&(y==1)).sum().item()
    acc = (tp+tn)/max(tp+tn+fp+fn,1)
    rec = tp/max(tp+fn,1); prec = tp/max(tp+fp,1)
    f1 = 2*prec*rec/max(prec+rec,1e-9); iou = tp/max(tp+fp+fn,1)
    return dict(acc=acc,f1=f1,recall=rec,iou=iou)

# -------------------- Entrenamiento --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_root", default="/home/htaucare/Tesis_Amaro/data/UFRN")
    ap.add_argument("--name", default="runs_pointnet2_binary_cls_v19d")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--bs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--pos_weight", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=77)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = Path(args.data_dir)
    out = Path(args.out_root) / f"{args.name}_seed{args.seed}"
    ensure_dir(out)

    full = UFRNBinaryDataset(data, augment=False)
    ids = full.ids; random.shuffle(ids)
    n = len(ids)
    ntr, nva = int(0.6*n), int(0.25*n)
    tr, va, te = ids[:ntr], ids[ntr:ntr+nva], ids[ntr+nva:]

    def subset(ids, aug): ds = UFRNBinaryDataset(data, aug); ds.ids = ids; return ds
    trd, vad, ted = subset(tr, True), subset(va, False), subset(te, False)
    dl_tr = DataLoader(trd, batch_size=args.bs, shuffle=True, num_workers=4, collate_fn=collate)
    dl_va = DataLoader(vad, batch_size=args.bs, shuffle=False, num_workers=2, collate_fn=collate)
    dl_te = DataLoader(ted, batch_size=args.bs, shuffle=False, num_workers=2, collate_fn=collate)

    model = PointNet2Binary().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss(weight=torch.tensor([1.0, args.pos_weight], device=device))

    best_f1, best_thr, patience = 0, 0.5, 0
    hist = {"tr_loss": [], "va_loss": [], "va_f1": []}

    for ep in range(1, args.epochs+1):
        model.train(); tr_loss=0
        for X,Y,_ in tqdm(dl_tr, desc=f"[Train {ep}/{args.epochs}]"):
            X,Y=X.to(device),Y.to(device)
            opt.zero_grad()
            logits=model(X); loss=ce(logits.reshape(-1,2),Y.reshape(-1))
            loss.backward(); opt.step()
            tr_loss+=loss.item()

        model.eval(); va_loss=0; all_logits, all_y=[],[]
        with torch.no_grad():
            for X,Y,_ in dl_va:
                X,Y=X.to(device),Y.to(device)
                logits=model(X); va_loss+=ce(logits.reshape(-1,2),Y.reshape(-1)).item()
                all_logits.append(logits.cpu()); all_y.append(Y.cpu())
        logits_all=torch.cat(all_logits); y_all=torch.cat(all_y)
        local_best_f1, local_best_thr=-1,0.5
        for t in np.linspace(0.05,0.95,19):
            m=metrics_global(logits_all,y_all,t)
            if m["f1"]>local_best_f1:
                local_best_f1,local_best_thr=m["f1"],t
        m_best=metrics_global(logits_all,y_all,local_best_thr)
        print(f"Ep{ep:03d}/{args.epochs} tr_loss={tr_loss/len(dl_tr):.4f} va_loss={va_loss/len(dl_va):.4f} "
              f"va_acc={m_best['acc']:.3f} va_f1={m_best['f1']:.3f} va_rec={m_best['recall']:.3f} va_iou={m_best['iou']:.3f} | thr={local_best_thr:.2f}")

        hist["tr_loss"].append(tr_loss/len(dl_tr)); hist["va_loss"].append(va_loss/len(dl_va)); hist["va_f1"].append(m_best["f1"])

        if m_best["f1"]>best_f1:
            best_f1,best_thr=m_best["f1"],local_best_thr
            torch.save(model.state_dict(),out/"best.pt"); patience=0
        else:
            patience+=1
            if patience>20:
                print("[EARLY STOP]"); break

    # ---- TEST ----
    model.load_state_dict(torch.load(out/"best.pt", map_location=device))
    model.eval(); all_logits, all_y=[],[]
    with torch.no_grad():
        for X,Y,_ in dl_te:
            X,Y=X.to(device),Y.to(device)
            logits=model(X); all_logits.append(logits.cpu()); all_y.append(Y.cpu())
    logits_all=torch.cat(all_logits); y_all=torch.cat(all_y)
    m_final=metrics_global(logits_all,y_all,best_thr)
    print(f"✅ TEST acc={m_final['acc']:.3f} f1={m_final['f1']:.3f} rec={m_final['recall']:.3f} iou={m_final['iou']:.3f} (thr={best_thr:.2f})")

    plt.figure(figsize=(8,4))
    plt.plot(hist["tr_loss"],label="Train Loss")
    plt.plot(hist["va_loss"],label="Val Loss")
    plt.plot(hist["va_f1"],label="Val F1")
    plt.legend(); plt.grid(); plt.tight_layout()
    plt.savefig(out/"curves.png",dpi=160)
    print(f"[DONE] Resultados → {out}")

if __name__ == "__main__":
    main()
