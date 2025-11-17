#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PointNet++ Binary (v19)
Versión estable con:
- Split aleatorio reproducible (previene fuga de datos)
- Validación más grande (25%)
- Métricas globales (F1, recall, IoU, accuracy)
- Búsqueda de mejor threshold en validación
- Gráficos de pérdida y métricas
"""

import os, random, argparse, numpy as np
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# ============================================================
# 🔧 Utilidades
# ============================================================
def set_seed(sd=42):
    random.seed(sd)
    np.random.seed(sd)
    torch.manual_seed(sd)
    torch.cuda.manual_seed_all(sd)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

# ============================================================
# 📦 Dataset
# ============================================================
class UFRNBinaryDataset(Dataset):
    def __init__(self, root, augment=False):
        self.root = Path(root)
        self.ids = sorted([d.name for d in self.root.iterdir() if d.is_dir() and d.name.startswith("paciente_")])
        self.augment = augment

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]
        X = np.load(self.root/pid/"X.npy").astype(np.float32)
        Y = np.load(self.root/pid/"Y.npy").astype(np.int64).reshape(-1)
        Y = np.clip(Y, 0, 1)

        if self.augment:
            X, Y = self._augment(X, Y)

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
            Y = np.pad(Y, ((0, pad)), constant_values=0)
        Xb.append(X); Yb.append(Y)
    return torch.tensor(np.stack(Xb), dtype=torch.float32), torch.tensor(np.stack(Yb), dtype=torch.long), pids

# ============================================================
# 🧩 PointNet++ módulos
# ============================================================
def square_distance(src, dst):
    B, N, _ = src.shape; _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.transpose(2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[torch.arange(B, device=device), farthest, :].unsqueeze(1)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def index_points(points, idx):
    B = points.shape[0]
    out = []
    for b in range(B):
        out.append(points[b, idx[b], :])
    return torch.stack(out)

def sample_and_group(npoint, k, xyz, points=None):
    B, N, _ = xyz.shape
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    dists = square_distance(new_xyz, xyz)
    idx = dists.argsort()[:, :, :k]
    grouped_xyz = torch.zeros(B, npoint, k, 3, device=xyz.device)
    for b in range(B):
        grouped_xyz[b] = xyz[b, idx[b], :]
    grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)
    if points is not None:
        grouped_points = torch.zeros(B, npoint, k, points.shape[2], device=xyz.device)
        for b in range(B):
            grouped_points[b] = points[b, idx[b], :]
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
    return new_xyz, new_points

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, k, mlp):
        super().__init__()
        self.npoint, self.k = npoint, k
        layers, last = [], 3
        for o in mlp:
            layers += [nn.Conv2d(last, o, 1), nn.BatchNorm2d(o), nn.ReLU()]
            last = o
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz, points=None):
        new_xyz, new_points = sample_and_group(self.npoint, self.k, xyz, points)
        new_points = new_points.permute(0, 3, 2, 1)
        new_points = self.mlp(new_points)
        new_points = torch.max(new_points, 2)[0]
        return new_xyz, new_points

# ============================================================
# 🧠 Modelo completo PointNet++
# ============================================================
class PointNet2Binary(nn.Module):
    def __init__(self, k=2, dropout=0.5):
        super().__init__()
        self.sa1 = PointNetSetAbstraction(1024, 32, [64, 64, 128])
        self.sa2 = PointNetSetAbstraction(256, 32, [128, 128, 256])
        self.sa3 = PointNetSetAbstraction(64, 32, [256, 512, 1024])
        self.fc = nn.Sequential(
            nn.Conv1d(1408, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, k, 1)
        )

    def forward(self, x):
        l0_xyz, l0_points = x, None
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        feat = torch.cat([l1_points.mean(-1), l2_points.mean(-1), l3_points.mean(-1)], dim=1).unsqueeze(-1)
        out = self.fc(feat)
        return out.transpose(2, 1)

# ============================================================
# 📊 Métricas
# ============================================================
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

# ============================================================
# 🚀 Entrenamiento
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_root", default="/home/htaucare/Tesis_Amaro/data/UFRN")
    ap.add_argument("--name", default="runs_pointnet2_binary_v19")
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
    ids = full.ids
    n = len(ids)
    random.shuffle(ids)
    ntr, nva = int(0.6 * n), int(0.25 * n)
    tr, va, te = ids[:ntr], ids[ntr:ntr + nva], ids[ntr + nva:]

    def subset(ids, aug): ds = UFRNBinaryDataset(data, augment=aug); ds.ids = ids; return ds
    trd, vad, ted = subset(tr, True), subset(va, False), subset(te, False)

    dl_tr = DataLoader(trd, batch_size=args.bs, shuffle=True, num_workers=4, collate_fn=collate)
    dl_va = DataLoader(vad, batch_size=args.bs, shuffle=False, num_workers=2, collate_fn=collate)
    dl_te = DataLoader(ted, batch_size=args.bs, shuffle=False, num_workers=2, collate_fn=collate)

    model = PointNet2Binary().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss(weight=torch.tensor([1.0, args.pos_weight], device=device))

    best_f1, patience, hist = 0, 0, {"train": [], "val_f1": [], "val_loss": []}

    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        for X, Y, _ in tqdm(dl_tr, desc=f"[Train {ep}/{args.epochs}]"):
            X, Y = X.to(device), Y.to(device)
            opt.zero_grad()
            logits = model(X)
            loss = ce(logits.reshape(-1, 2), Y.reshape(-1))
            loss.backward()
            opt.step()
            total_loss += loss.item()

        model.eval()
        val_loss, agg_logits, agg_y = 0, [], []
        with torch.no_grad():
            for X, Y, _ in dl_va:
                X, Y = X.to(device), Y.to(device)
                logits = model(X)
                val_loss += ce(logits.reshape(-1, 2), Y.reshape(-1)).item()
                agg_logits.append(logits.cpu()); agg_y.append(Y.cpu())
        logits_all = torch.cat(agg_logits)
        y_all = torch.cat(agg_y)
        best_thr, best_f1 = 0.5, 0
        for thr in np.linspace(0.05, 0.95, 19):
            m = metrics_global(logits_all, y_all, thr)
            if m["f1"] > best_f1:
                best_f1, best_thr = m["f1"], thr
        m_best = metrics_global(logits_all, y_all, best_thr)

        print(f"Ep{ep:03d}/{args.epochs} tr_loss={total_loss/len(dl_tr):.4f} va_acc={m_best['acc']:.3f} va_f1={m_best['f1']:.3f} va_rec={m_best['recall']:.3f} va_iou={m_best['iou']:.3f} | best_thr={best_thr:.2f}")
        hist["train"].append(total_loss/len(dl_tr))
        hist["val_f1"].append(m_best["f1"])
        hist["val_loss"].append(val_loss/len(dl_va))

        if m_best["f1"] > best_f1:
            torch.save(model.state_dict(), out/"best.pt")
            patience = 0
        else:
            patience += 1
            if patience > 20:
                print("[EARLY STOP]")
                break

    # ---- Evaluación final ----
    model.load_state_dict(torch.load(out/"best.pt", map_location=device))
    model.eval()
    agg_logits, agg_y = [], []
    with torch.no_grad():
        for X, Y, _ in dl_te:
            X, Y = X.to(device), Y.to(device)
            logits = model(X)
            agg_logits.append(logits.cpu()); agg_y.append(Y.cpu())
    logits_all = torch.cat(agg_logits)
    y_all = torch.cat(agg_y)
    m_final = metrics_global(logits_all, y_all, best_thr)
    print(f"✅ TEST acc={m_final['acc']:.3f} f1={m_final['f1']:.3f} rec={m_final['recall']:.3f} iou={m_final['iou']:.3f}")

    # ---- Graficar curvas ----
    plt.figure(figsize=(8,4))
    plt.plot(hist["train"], label="Train Loss")
    plt.plot(hist["val_loss"], label="Val Loss")
    plt.plot(hist["val_f1"], label="Val F1")
    plt.legend(); plt.grid(True)
    plt.title("Entrenamiento PointNet++ Binary (v19)")
    plt.xlabel("Épocas")
    plt.ylabel("Valor")
    plt.tight_layout()
    plt.savefig(out/"curves.png", dpi=160)
    print(f"[DONE] Resultados guardados en: {out}")

if __name__ == "__main__":
    main()
