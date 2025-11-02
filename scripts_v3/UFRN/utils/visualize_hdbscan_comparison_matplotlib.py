#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualizador comparativo HDBSCAN (Matplotlib)
---------------------------------------------
Genera imágenes PNG lado a lado:
  - Izquierda: predicción original (raw)
  - Derecha: post-procesada con HDBSCAN (clean)
Compatible con entornos sin display (headless).
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ---------------------------------------------------------------
# Función para crear cada subfigura
# ---------------------------------------------------------------
def plot_cloud(ax, X, mask, title, elev=25, azim=35):
    ax.view_init(elev=elev, azim=azim)
    ax.axis("off")
    ax.set_title(title, fontsize=11)
    bg = X[mask == 0]
    fg = X[mask == 1]
    ax.scatter(bg[:, 0], bg[:, 1], bg[:, 2], c="lightgray", s=0.3, alpha=0.7)
    if len(fg) > 0:
        ax.scatter(fg[:, 0], fg[:, 1], fg[:, 2], c="red", s=1.0, alpha=0.9)


# ---------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True,
                    help="Carpeta con X.npy originales")
    ap.add_argument("--mask_raw", required=True,
                    help="Carpeta con las máscaras crudas (predicciones del modelo)")
    ap.add_argument("--mask_clean", required=True,
                    help="Carpeta con las máscaras post-HDBSCAN")
    ap.add_argument("--out_dir", required=True,
                    help="Carpeta donde se guardarán las comparaciones")
    ap.add_argument("--sample", type=int, default=-1,
                    help="Número de pacientes a renderizar (-1 = todos)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    raw_dir = Path(args.mask_raw)
    clean_dir = Path(args.mask_clean)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    patients = sorted([p.name for p in data_dir.iterdir()
                       if p.is_dir() and p.name.startswith("paciente_")])
    if args.sample > 0:
        patients = patients[:args.sample]

    print(f"🖼 Renderizando {len(patients)} comparaciones...")

    for pid in patients:
        try:
            X = np.load(data_dir / pid / "X.npy").astype(np.float32)
            Y_raw = np.load(raw_dir / f"{pid}_raw.npy")
            Y_clean = np.load(clean_dir / f"{pid}_clean.npy")

            fig = plt.figure(figsize=(10, 5))
            ax1 = fig.add_subplot(121, projection='3d')
            ax2 = fig.add_subplot(122, projection='3d')

            plot_cloud(ax1, X, Y_raw, "Predicción original")
            plot_cloud(ax2, X, Y_clean, "Después de HDBSCAN")

            plt.tight_layout()
            out_img = out_dir / f"{pid}_comparison.png"
            plt.savefig(out_img, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"   ✓ {pid} → {out_img.name}")
        except Exception as e:
            print(f"   ⚠️ Error en {pid}: {e}")

    print(f"\n✅ Comparaciones guardadas en: {out_dir}")

if __name__ == "__main__":
    main()
