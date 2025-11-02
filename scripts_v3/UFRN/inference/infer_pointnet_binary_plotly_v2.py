#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inferencia visual con Plotly para PointNet Binary (segmentación diente 21)
Detecta inversión de clases automáticamente y genera HTML interactivos.
"""

import torch
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from tqdm import tqdm
import argparse
import torch.nn as nn

# ======================================================
# 🔹 Modelo (idéntico al usado en entrenamiento)
# ======================================================
class PointNetSeg(nn.Module):
    def __init__(self, k=2):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU())

        self.seg = nn.Sequential(
            nn.Conv1d(1088, 512, 1), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Conv1d(512, 256, 1),  nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 128, 1),  nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, k, 1)
        )

    def forward(self, x):
        x = x.transpose(2, 1)
        x = self.conv1(x)
        pointfeat = self.conv2(x)
        x = self.conv3(pointfeat)
        x = self.conv4(x)
        x = self.conv5(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.repeat(1, 1, pointfeat.shape[2])
        x = torch.cat([pointfeat, x], dim=1)
        x = self.seg(x)
        return x.transpose(2, 1)

# ======================================================
# 🔹 Utilidades
# ======================================================
def load_model(model_path: Path, device: str):
    model = PointNetSeg(k=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def plot_inference(points: np.ndarray, preds: np.ndarray, out_html: Path, pid: str):
    """
    points: (N,3)
    preds : (N,)  0 = fondo, 1 = diente 21
    """

    # 🔍 Auto-corrección si las etiquetas están invertidas
    if preds.mean() > 0.5:
        print(f"[!] {pid}: inversión detectada → corrigiendo etiquetas")
        preds = 1 - preds

    pts_normal = points[preds == 0]
    pts_tooth21 = points[preds == 1]

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=pts_normal[:,0], y=pts_normal[:,1], z=pts_normal[:,2],
        mode='markers', marker=dict(size=2, color='blue', opacity=0.35),
        name='Resto (Clase 0)'
    ))
    fig.add_trace(go.Scatter3d(
        x=pts_tooth21[:,0], y=pts_tooth21[:,1], z=pts_tooth21[:,2],
        mode='markers', marker=dict(size=3, color='red', opacity=0.9),
        name='Diente 21 (Clase 1)'
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            aspectmode='data',
            camera=dict(eye=dict(x=1.2, y=1.2, z=0.5))
        ),
        title=f"Inferencia — {pid}",
        template="plotly_dark",
        showlegend=True
    )

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html))
    print(f"[OK] {pid}: guardado → {out_html.name}")


# ======================================================
# 🔹 Inferencia principal
# ======================================================
def infer_and_visualize(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = Path(args.model_path)
    data_dir = Path(args.data_dir)
    figs_dir = Path(args.out_dir) / "figures_infer"
    figs_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(model_path, device)
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

        plot_inference(
            X,
            preds,
            figs_dir / f"{pid}_pred.html",
            pid
        )

    print(f"\n✅ Visualizaciones completadas. Guardadas en: {figs_dir}")


# ======================================================
# 🔹 Main
# ======================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, help="Ruta al modelo entrenado (.pt)")
    ap.add_argument("--data_dir", required=True, help="Carpeta con X.npy/Y.npy (test o completo)")
    ap.add_argument("--out_dir", default="/home/htaucare/Tesis_Amaro/scripts_v3/UFRN/inference")
    ap.add_argument("--max_samples", type=int, default=5, help="Cantidad de pacientes a visualizar (0=Todos)")
    args = ap.parse_args()
    infer_and_visualize(args)

