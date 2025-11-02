#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualiza las etiquetas ICP generadas para el diente 21
y guarda los gráficos comparativos en /UFRN/figures/.

Entrada:
  data/UFRN/processed_pseudolabels_icp/8192/upper/paciente_X/{X.npy, Y.npy}

Salida:
  scripts_v3/UFRN/figures/paciente_X_icp_labels.png
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# === CONFIG ===
data_root = Path("/home/htaucare/Tesis_Amaro/data/UFRN/processed_pseudolabels_icp/8192/upper")
fig_root = Path("/home/htaucare/Tesis_Amaro/scripts_v3/UFRN/figures")
fig_root.mkdir(parents=True, exist_ok=True)


def plot_icp_labels(pid, X, Y, out_path):
    """Genera gráfico 3D del paciente y guarda en PNG (vista frontal)."""
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')

    # Fondo (gris) y puntos positivos (rojos)
    ax.scatter(X[Y==0,0], X[Y==0,1], X[Y==0,2], s=1, c='blue', alpha=0.25, label="Fondo")
    ax.scatter(X[Y==1,0], X[Y==1,1], X[Y==1,2], s=4, c='red', alpha=0.8, label="Diente 21")

    ax.set_title(f"{pid} — Puntos positivos = {int(Y.sum())}", fontsize=12)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.legend(loc="upper right")

    # --- Vista FRONTAL anatómica ---
    ax.view_init(elev=15, azim=90)   # elev=ángulo vertical, azim=ángulo horizontal

    # Opcional: límites más centrados
    ax.set_xlim([-40, 40])
    ax.set_ylim([-40, 40])
    ax.set_zlim([-5, 15])

    plt.tight_layout()
    plt.savefig(out_path, dpi=600)
    plt.close(fig)


# === LOOP ===
patients = sorted([p for p in data_root.iterdir() if (p/"X.npy").exists() and (p/"Y.npy").exists()])
print(f"[INFO] Pacientes detectados: {len(patients)}")

ok = 0
for p in tqdm(patients, desc="[PLOT] Generando figuras"):
    try:
        X = np.load(p/"X.npy")
        Y = np.load(p/"Y.npy")
        pid = p.name
        out_path = fig_root / f"{pid}_icp_labels.png"
        plot_icp_labels(pid, X, Y, out_path)
        ok += 1
    except Exception as e:
        print(f"[WARN] {p.name}: {e}")

print(f"\n✅ Figuras generadas correctamente: {ok}/{len(patients)}")
print(f"Guardadas en: {fig_root}")
