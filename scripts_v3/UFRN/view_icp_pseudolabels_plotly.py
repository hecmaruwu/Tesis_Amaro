#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualiza las etiquetas ICP del diente 21 con Plotly Express.
Colores:
 - Azul = Fondo (no diente)
 - Rojo = Diente 21

Guarda los resultados en:
  /home/htaucare/Tesis_Amaro/scripts_v3/UFRN/figures/
"""

import numpy as np
import pandas as pd
import plotly.express as px
from pathlib import Path
from tqdm import tqdm

# === CONFIG ===
data_root = Path("/home/htaucare/Tesis_Amaro/data/UFRN/processed_pseudolabels_icp/8192/upper")
fig_root = Path("/home/htaucare/Tesis_Amaro/scripts_v3/UFRN/figures")
fig_root.mkdir(parents=True, exist_ok=True)


def plot_icp_labels_plotly(pid, X, Y, out_html):
    """Crea gráfico 3D interactivo y lo guarda como HTML + PNG."""
    df = pd.DataFrame(X, columns=["X", "Y", "Z"])
    df["Etiqueta"] = np.where(Y == 1, "Diente 21", "Fondo")

    # --- Paleta personalizada ---
    color_map = {"Diente 21": "red", "Fondo": "blue"}

    fig = px.scatter_3d(
        df,
        x="X", y="Y", z="Z",
        color="Etiqueta",
        color_discrete_map=color_map,
        opacity=0.7,
        title=f"{pid} — Diente 21 (rojo) / Fondo (azul)",
        height=800,
    )

    # Ajuste de cámara para vista frontal anatómica
    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            camera=dict(eye=dict(x=0, y=1.8, z=0.6)),  # Frontal anatómica
            xaxis=dict(range=[-40, 40]),
            yaxis=dict(range=[-40, 40]),
            zaxis=dict(range=[-5, 15]),
        ),
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # Guarda versión interactiva (HTML)
    html_path = fig_root / f"{pid}_plotly.html"
    fig.write_html(str(html_path))

    # Guarda imagen estática (si tiene kaleido instalado)
    try:
        png_path = fig_root / f"{pid}_plotly.png"
        fig.write_image(str(png_path), scale=2)
        print(f"✅ Guardado: {png_path}")
    except Exception as e:
        print(f"[WARN] No se pudo guardar PNG (instale 'kaleido'): {e}")

    return fig


# === LOOP ===
patients = sorted([p for p in data_root.iterdir() if (p/"X.npy").exists() and (p/"Y.npy").exists()])
print(f"[INFO] Pacientes detectados: {len(patients)}")

ok = 0
for p in tqdm(patients, desc="[PLOTLY] Generando visualizaciones"):
    try:
        X = np.load(p/"X.npy")
        Y = np.load(p/"Y.npy")
        pid = p.name
        out_html = fig_root / f"{pid}_plotly.html"
        plot_icp_labels_plotly(pid, X, Y, out_html)
        ok += 1
    except Exception as e:
        print(f"[WARN] {p.name}: {e}")

print(f"\n✅ Figuras Plotly generadas correctamente: {ok}/{len(patients)}")
print(f"Guardadas en: {fig_root}")
