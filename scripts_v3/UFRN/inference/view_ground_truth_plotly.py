#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualización de ground truth (Y.npy) con Plotly 3D
Muestra puntos del diente 21 (rojos) y resto del maxilar (azules).
"""

import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from tqdm import tqdm
import argparse

# ======================================================
# 🔹 Función para graficar el ground truth
# ======================================================
def plot_ground_truth(points: np.ndarray, labels: np.ndarray, out_html: Path, pid: str):
    pts_bg = points[labels == 0]
    pts_d21 = points[labels == 1]

    fig = go.Figure()

    # Fondo azul (resto del maxilar)
    fig.add_trace(go.Scatter3d(
        x=pts_bg[:, 0], y=pts_bg[:, 1], z=pts_bg[:, 2],
        mode='markers',
        marker=dict(size=2, color='blue', opacity=0.3),
        name='Resto (Clase 0)'
    ))

    # Diente 21 en rojo
    fig.add_trace(go.Scatter3d(
        x=pts_d21[:, 0], y=pts_d21[:, 1], z=pts_d21[:, 2],
        mode='markers',
        marker=dict(size=3, color='red', opacity=0.9),
        name='Diente 21 (Clase 1)'
    ))

    fig.update_layout(
        title=f"Ground Truth — {pid} (puntos positivos = {labels.sum()})",
        template="plotly_dark",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',
            camera=dict(eye=dict(x=1.2, y=1.2, z=0.6))
        ),
        showlegend=True
    )

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html))
    print(f"[OK] {pid}: figura guardada → {out_html.name}")


# ======================================================
# 🔹 Script principal
# ======================================================
def main(args):
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = sorted(data_dir.glob("paciente_*/X.npy"))
    if args.max_samples > 0:
        samples = samples[:args.max_samples]

    for sample in tqdm(samples, desc="[GROUND TRUTH]"):
        pid = sample.parent.name
        X = np.load(sample)
        Y = np.load(sample.parent / "Y.npy")

        plot_ground_truth(
            X,
            Y,
            out_dir / f"{pid}_ground_truth.html",
            pid
        )

    print(f"\n✅ Visualizaciones de Ground Truth completadas. Guardadas en: {out_dir}")


# ======================================================
# 🔹 Main
# ======================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True,
                    help="Carpeta con X.npy / Y.npy (por ejemplo: processed_pseudolabels_icp/8192/upper)")
    ap.add_argument("--out_dir", default="/home/htaucare/Tesis_Amaro/scripts_v3/UFRN/figures/ground_truth")
    ap.add_argument("--max_samples", type=int, default=5, help="Cantidad de pacientes a visualizar (0=Todos)")
    args = ap.parse_args()
    main(args)
