#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inferencia visual con PointNet++ Binario (v9)
---------------------------------------------
Visualiza predicciones sobre nubes 3D (Diente 21 vs resto)
- HTML interactivo (Plotly)
- PNG estático (si kaleido está instalado)
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import plotly.graph_objects as go

# ======================================================
# 🔹 Modelo PointNet++ Binario (igual al usado en train)
# ======================================================
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

# --- Bloques mínimos reutilizados ---
def square_distance(src, dst):
    return torch.sum((src.unsqueeze(2) - dst.unsqueeze(1)) ** 2, dim=-1)

def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device, dtype=xyz.dtype)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_idx = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_idx, farthest, :].unsqueeze(1)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]
    return centroids

def index_points(points, idx):
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1]*(len(view_shape)-1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_idx = torch.arange(B, dtype=torch.long, device=points.device).view(view_shape).repeat(repeat_shape)
    return points[batch_idx, idx, :]

def sample_and_group(npoint, k, xyz, points):
    B, N, _ = xyz.shape
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    dist = square_distance(new_xyz, xyz)
    idx = dist.argsort()[:, :, :k]
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
    return new_xyz, new_points

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, k, in_channel, mlp):
        super().__init__()
        self.npoint = npoint
        self.k = k
        layers = []
        last = in_channel
        for out in mlp:
            layers += [nn.Conv2d(last, out, 1), nn.BatchNorm2d(out), nn.ReLU(inplace=True)]
            last = out
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz, points):
        new_xyz, new_points = sample_and_group(self.npoint, self.k, xyz, points)
        new_points = new_points.permute(0, 3, 2, 1)
        new_points = self.mlp(new_points)
        new_points = torch.max(new_points, 2)[0]
        return new_xyz, new_points

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super().__init__()
        layers = []
        last = in_channel
        for out in mlp:
            layers += [nn.Conv1d(last, out, 1), nn.BatchNorm1d(out), nn.ReLU(inplace=True)]
            last = out
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz1, xyz2, points1, points2):
        B, N, _ = xyz1.shape
        _, S, _ = xyz2.shape
        dists = square_distance(xyz1, xyz2)
        dists, idx = dists.sort(dim=-1)
        idx = idx[:, :, :3]
        dists = dists[:, :, :3]
        dist_recip = 1.0 / (dists + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        pts2 = points2.permute(0, 2, 1)
        interpolated = torch.sum(index_points(pts2, idx) * weight.unsqueeze(-1), dim=2)
        if points1 is not None:
            new_points = torch.cat([points1.permute(0, 2, 1), interpolated], dim=2)
        else:
            new_points = interpolated
        return self.mlp(new_points.permute(0, 2, 1))

class PointNet2Binary(nn.Module):
    def __init__(self, k=2):
        super().__init__()
        self.sa1 = PointNetSetAbstraction(512, 16, 3, [64, 64, 128])
        self.sa2 = PointNetSetAbstraction(128, 16, 128 + 3, [128, 128, 256])
        self.sa3 = PointNetSetAbstraction(32, 16, 256 + 3, [256, 256, 512])
        self.fp3 = PointNetFeaturePropagation(512 + 256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(256 + 128, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 64])
        self.classifier = nn.Sequential(
            nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Conv1d(64, k, 1)
        )

    def forward(self, x):
        x = x.float()
        l0_xyz, l0_points = x, None
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l1_points_pts = l1_points.permute(0, 2, 1)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points_pts)
        l2_points_pts = l2_points.permute(0, 2, 1)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points_pts)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)
        logits = self.classifier(l0_points).permute(0, 2, 1)
        return logits

# ======================================================
# 🔹 Función de inferencia visual
# ======================================================
def visualize_inference(points, preds, out_html, out_png, pid):
    pts_rest = points[preds == 0]
    pts_tooth = points[preds == 1]

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=pts_rest[:,0], y=pts_rest[:,1], z=pts_rest[:,2],
        mode="markers", marker=dict(size=2, color="blue", opacity=0.35),
        name="Resto (Clase 0)"
    ))
    fig.add_trace(go.Scatter3d(
        x=pts_tooth[:,0], y=pts_tooth[:,1], z=pts_tooth[:,2],
        mode="markers", marker=dict(size=3, color="red", opacity=0.9),
        name="Diente 21 (Clase 1)"
    ))
    fig.update_layout(
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
        title=f"Inferencia — {pid}",
        template="plotly_dark",
        showlegend=True
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html))
    try:
        import kaleido  # asegúrate de tenerlo instalado
        fig.write_image(str(out_png))
    except Exception:
        pass
    print(f"[OK] {pid}: guardado → {out_html.name}")

# ======================================================
# 🔹 Inferencia principal
# ======================================================
def infer_and_visualize(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PointNet2Binary().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    data_dir = Path(args.data_dir)
    figs_dir = Path(args.out_dir) / "figures_infer"
    figs_dir.mkdir(parents=True, exist_ok=True)

    samples = sorted(data_dir.glob("paciente_*/X.npy"))
    if args.max_samples > 0:
        samples = samples[:args.max_samples]

    for sample in tqdm(samples, desc="[INFERENCIA]"):
        pid = sample.parent.name
        X = np.load(sample).astype(np.float32)
        X_tensor = torch.from_numpy(X).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(X_tensor)
            preds = logits.argmax(2).cpu().numpy().squeeze()
        visualize_inference(X, preds,
                            figs_dir / f"{pid}_pred.html",
                            figs_dir / f"{pid}_pred.png",
                            pid)
    print(f"\n✅ Visualizaciones completadas. Guardadas en: {figs_dir}")

# ======================================================
# 🔹 Main
# ======================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, help="Ruta al modelo .pt")
    ap.add_argument("--data_dir", required=True, help="Carpeta con paciente_*/X.npy")
    ap.add_argument("--out_dir", default="/home/htaucare/Tesis_Amaro/scripts_v3/UFRN/inference")
    ap.add_argument("--max_samples", type=int, default=5, help="Cantidad de pacientes a visualizar (0=Todos)")
    args = ap.parse_args()
    infer_and_visualize(args)
