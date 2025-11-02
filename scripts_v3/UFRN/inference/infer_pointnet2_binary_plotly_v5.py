#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inferencia visual con Plotly para PointNet2Binary (segmentación diente 21)
Guarda visualizaciones HTML y PNG.
Compatible con el modelo entrenado en train_pointnet2_binary_with_split_v5.py
"""

import torch
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from tqdm import tqdm
import argparse
import plotly.io as pio

# ======================================================
# 🔹 Modelo idéntico al usado en entrenamiento
# ======================================================
import torch.nn as nn

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, k, in_channel, mlp):
        super().__init__()
        self.npoint, self.k = npoint, k
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channel + 3, mlp[0], 1),
            nn.BatchNorm2d(mlp[0]),
            nn.ReLU(),
            nn.Conv2d(mlp[0], mlp[1], 1),
            nn.BatchNorm2d(mlp[1]),
            nn.ReLU(),
            nn.Conv2d(mlp[1], mlp[2], 1),
            nn.BatchNorm2d(mlp[2]),
            nn.ReLU()
        )

    def forward(self, xyz, points):
        new_xyz, new_points = sample_and_group(self.npoint, self.k, xyz, points)
        new_points = new_points.permute(0, 3, 2, 1)
        new_points = self.mlp(new_points)
        new_points = torch.max(new_points, 2)[0]
        new_points = new_points.permute(0, 2, 1)
        return new_xyz, new_points

def square_distance(src, dst):
    return torch.sum((src.unsqueeze(2) - dst.unsqueeze(1)) ** 2, dim=-1)

def index_points(points, idx):
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_idx = torch.arange(B, dtype=torch.long, device=points.device).view(view_shape).repeat(repeat_shape)
    return points[batch_idx, idx, :]

def farthest_point_sample(xyz, npoint):
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
    distance = torch.ones(B, N, device=xyz.device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=xyz.device)
    batch_indices = torch.arange(B, dtype=torch.long, device=xyz.device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].unsqueeze(1)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def knn_point(k, xyz, new_xyz):
    dist = square_distance(new_xyz, xyz)
    idx = dist.topk(k=k, dim=-1, largest=False)[1]
    return idx

def sample_and_group(npoint, k, xyz, points):
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    idx = knn_point(k, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
    return new_xyz, new_points

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channel, mlp[0], 1),
            nn.BatchNorm1d(mlp[0]),
            nn.ReLU(),
            nn.Conv1d(mlp[0], mlp[1], 1),
            nn.BatchNorm1d(mlp[1]),
            nn.ReLU()
        )

    def forward(self, xyz1, xyz2, points1, points2):
        dist = square_distance(xyz1, xyz2)
        dist, idx = dist.sort(dim=-1)
        weight = 1.0 / (dist[:, :, :3] + 1e-8)
        weight = weight / torch.sum(weight, -1, keepdim=True)
        interpolated_points = torch.sum(index_points(points2, idx[:, :, :3]) * weight.unsqueeze(-1), dim=2)
        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points
        new_points = new_points.permute(0, 2, 1)
        new_points = self.mlp(new_points)
        return new_points.permute(0, 2, 1)

class PointNet2Binary(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.sa1 = PointNetSetAbstraction(1024, 32, 0, [64, 64, 128])
        self.sa2 = PointNetSetAbstraction(256, 32, 128, [128, 128, 256])
        self.sa3 = PointNetSetAbstraction(64, 32, 256, [256, 256, 512])
        self.fp3 = PointNetFeaturePropagation(512 + 256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(256 + 128, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128])
        self.head = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, num_classes, 1)
        )

    def forward(self, x):
        l0_xyz, l0_points = x, None
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
        x = self.head(l0_points.permute(0, 2, 1)).permute(0, 2, 1)
        return x

# ======================================================
# 🔹 Funciones auxiliares
# ======================================================
def load_model(model_path, device):
    model = PointNet2Binary().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def plot_inference(points, preds, out_html, pid):
    """Genera visualización 3D con colores binarios"""
    pts_rest = points[preds == 0]
    pts_tooth21 = points[preds == 1]

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=pts_rest[:, 0], y=pts_rest[:, 1], z=pts_rest[:, 2],
        mode='markers', marker=dict(size=2, color='blue', opacity=0.35),
        name='Resto (Clase 0)'
    ))
    fig.add_trace(go.Scatter3d(
        x=pts_tooth21[:, 0], y=pts_tooth21[:, 1], z=pts_tooth21[:, 2],
        mode='markers', marker=dict(size=3, color='red', opacity=0.9),
        name='Diente 21 (Clase 1)'
    ))

    fig.update_layout(
        title=f"Inferencia — {pid}",
        template="plotly_dark",
        showlegend=True,
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
            aspectmode="data",
            camera=dict(eye=dict(x=0, y=-1.8, z=0.4))  # vista frontal
        )
    )

    fig.write_html(str(out_html))
    print(f"[OK] {pid} → guardado {out_html.name}")

# ======================================================
# 🔹 Proceso principal de inferencia
# ======================================================
def infer_and_visualize(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model_path, device)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
        plot_inference(X, preds, out_dir / f"{pid}_pred.html", pid)

    print(f"\n✅ Visualizaciones completadas en: {out_dir}")

# ======================================================
# 🔹 Main
# ======================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, help="Ruta al modelo .pt entrenado")
    ap.add_argument("--data_dir", required=True, help="Carpeta con X.npy/Y.npy")
    ap.add_argument("--out_dir", default="/home/htaucare/Tesis_Amaro/scripts_v3/UFRN/inference/results_pointnet2")
    ap.add_argument("--max_samples", type=int, default=5, help="N° de pacientes a visualizar (0 = todos)")
    args = ap.parse_args()
    infer_and_visualize(args)
