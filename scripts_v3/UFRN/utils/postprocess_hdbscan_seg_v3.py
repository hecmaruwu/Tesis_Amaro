#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-procesamiento con HDBSCAN (v3 - Compatible con modelo entrenado PointNet2BinarySeg)
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import hdbscan
from sklearn.metrics import f1_score, recall_score, accuracy_score, jaccard_score

# ============================================================
# 🔹 Dataset simple
# ============================================================
class UFRNBinaryDataset:
    def __init__(self, root):
        self.root = Path(root)
        self.ids = sorted([d.name for d in self.root.iterdir() if d.is_dir() and d.name.startswith("paciente_")])
    def __len__(self): return len(self.ids)
    def __getitem__(self, idx):
        pid = self.ids[idx]
        X = np.load(self.root / pid / "X.npy").astype(np.float32)
        Y = np.load(self.root / pid / "Y.npy").astype(np.int64).reshape(-1)
        Y = np.clip(Y, 0, 1)
        return X, Y, pid

# ============================================================
# 🔹 Funciones geométricas (idénticas a entrenamiento)
# ============================================================
def square_distance(src, dst):
    B, N, _ = src.shape; _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.transpose(2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def farthest_point_sample(xyz, npoint):
    device = xyz.device; B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[torch.arange(B), farthest, :].unsqueeze(1)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance; distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def index_points(points, idx):
    return torch.stack([p[i, :] for p, i in zip(points, idx)], dim=0)

def sample_and_group(npoint, k, xyz, points=None):
    B, N, _ = xyz.shape
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    dists = square_distance(new_xyz, xyz)
    _, nn_idx = torch.topk(-dists, k, dim=-1)
    grouped_xyz = torch.stack([xyz[b, nn_idx[b], :] for b in range(B)], dim=0)
    grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)
    if points is not None:
        grouped_points = torch.stack([points[b, nn_idx[b], :] for b in range(B)], dim=0)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
    return new_xyz, new_points

def three_nn(xyz1, xyz2):
    dist = square_distance(xyz1, xyz2)
    dist, idx = dist.sort(dim=-1)
    dist = dist[:, :, :3]; idx = idx[:, :, :3]
    return dist, idx

def interpolate(features2, xyz1, xyz2):
    dist, idx = three_nn(xyz1, xyz2)
    dist_recip = 1.0 / (dist + 1e-8)
    norm = torch.sum(dist_recip, dim=2, keepdim=True)
    weight = dist_recip / norm
    interpolated = torch.sum(index_points(features2.transpose(1,2), idx) * weight.unsqueeze(-1), dim=2)
    return interpolated.transpose(1,2)

# ============================================================
# 🔹 Modelo PointNet++ completo
# ============================================================
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
        new_points = new_points.permute(0, 3, 2, 1).contiguous()
        new_points = self.mlp(new_points)
        new_points = torch.max(new_points, dim=2)[0]
        return new_xyz, new_points

class PointNet2BinarySeg(nn.Module):
    def __init__(self, k=2, dropout=0.5):
        super().__init__()
        self.sa1 = PointNetSetAbstraction(1024, 32, 3, [64, 64, 128])
        self.sa2 = PointNetSetAbstraction(256, 32, 131, [128, 128, 256])
        self.sa3 = PointNetSetAbstraction(64, 32, 259, [256, 512, 1024])
        self.fc = nn.Sequential(
            nn.Conv1d(128 + 256 + 1024, 512, 1),
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(256, k, 1)
        )
    def forward(self, x):
        l0_xyz, l0_points = x, None
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points.transpose(1, 2))
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points.transpose(1, 2))
        l3_interp = interpolate(l3_points, l0_xyz, l3_xyz)
        l2_interp = interpolate(l2_points, l0_xyz, l2_xyz)
        l1_interp = interpolate(l1_points, l0_xyz, l1_xyz)
        feat = torch.cat([l1_interp, l2_interp, l3_interp], dim=1)
        out = self.fc(feat)
        return out.transpose(1, 2)

# ============================================================
# 🔹 Métricas y main
# ============================================================
def metrics_global(pred, y_true):
    acc = accuracy_score(y_true, pred)
    f1 = f1_score(y_true, pred, zero_division=0)
    rec = recall_score(y_true, pred, zero_division=0)
    iou = jaccard_score(y_true, pred, zero_division=0)
    return dict(acc=acc, f1=f1, recall=rec, iou=iou)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--model_ckpt", required=True)
    ap.add_argument("--thr", type=float, default=0.65)
    ap.add_argument("--min_cluster_size", type=int, default=30)
    ap.add_argument("--min_samples", type=int, default=5)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks_raw").mkdir(parents=True, exist_ok=True)

    # Modelo
    print(f"\n🚀 Cargando modelo completo desde {args.model_ckpt}")
    model = PointNet2BinarySeg().to(device)
    ckpt = torch.load(args.model_ckpt, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    ds = UFRNBinaryDataset(args.data_dir)
    print(f"📦 Pacientes detectados: {len(ds)}\n")

    all_metrics_raw, all_metrics_clean = [], []

    for X, Y, pid in tqdm(ds, desc="Post HDBSCAN"):
        try:
            X_t = torch.tensor(X[None, ...], dtype=torch.float32, device=device)
            with torch.no_grad():
                logits = model(X_t).cpu().squeeze(0)

            p = F.softmax(logits, dim=-1)[:, 1].numpy()
            mask_raw = (p >= args.thr).astype(np.uint8)
            np.save(out_dir / "masks_raw" / f"{pid}_raw.npy", mask_raw)

            pts_pos = X[mask_raw == 1]
            if len(pts_pos) > 0:
                clusterer = hdbscan.HDBSCAN(min_cluster_size=args.min_cluster_size,
                                            min_samples=args.min_samples)
                labels = clusterer.fit_predict(pts_pos)
                mask_clean = np.zeros_like(mask_raw)
                if np.any(labels >= 0):
                    main_cluster = np.argmax(np.bincount(labels[labels >= 0]))
                    keep_idx = np.where(mask_raw == 1)[0][labels == main_cluster]
                    mask_clean[keep_idx] = 1
            else:
                mask_clean = mask_raw
            np.save(out_dir / "masks" / f"{pid}_clean.npy", mask_clean)

            m_raw = metrics_global(mask_raw, Y)
            m_clean = metrics_global(mask_clean, Y)
            all_metrics_raw.append(list(m_raw.values()))
            all_metrics_clean.append(list(m_clean.values()))
        except Exception as e:
            print(f"⚠️ Error en {pid}: {e}")

    all_metrics_raw = np.array(all_metrics_raw)
    all_metrics_clean = np.array(all_metrics_clean)
    mean_raw = all_metrics_raw.mean(0)
    mean_clean = all_metrics_clean.mean(0)

    print("\n====== MÉTRICAS GLOBALES ======")
    print(f"ANTES  (thr={args.thr:.2f})  acc={mean_raw[0]:.3f} f1={mean_raw[1]:.3f} rec={mean_raw[2]:.3f} iou={mean_raw[3]:.3f}")
    print(f"DESPUÉS(HDBSCAN)          acc={mean_clean[0]:.3f} f1={mean_clean[1]:.3f} rec={mean_clean[2]:.3f} iou={mean_clean[3]:.3f}")
    print(f"[DONE] Salidas guardadas en: {out_dir}")

if __name__ == "__main__":
    main()
