#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualizador 3D sin OpenGL (compatible con servidores)
-------------------------------------------------------
Renderiza nubes de puntos con Matplotlib 3D y guarda las imágenes en PNG.
Colores:
  - Gris = fondo
  - Rojo = diente 21
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")  # <- evita el uso de display
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def make_and_save_plot(X, mask, out_path, elev=25, azim=35):
    """
    Genera y guarda una imagen 3D de la nube de puntos.
    """
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=elev, azim=azim)
    ax.axis("off")

    # Fondo gris, diente rojo
    bg = X[mask == 0]
    fg = X[mask == 1]

    ax.scatter(bg[:, 0], bg[:, 1], bg[:, 2], c="lightgray", s=0.3, alpha=0.7)
    if len(fg) > 0:
        ax.scatter(fg[:, 0], fg[:, 1], fg[:, 2], c="red", s=1.0, alpha=0.9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True,
                    help="Directorio con X.npy originales")
    ap.add_argument("--mask_dir", required=True,
                    help="Directorio con las máscaras *_clean.npy o *_raw.npy")
    ap.add_argument("--out_dir", required=True,
                    help="Carpeta donde se guardarán las imágenes PNG")
    ap.add_argument("--mode", choices=["raw", "clean"], default="clean",
                    help="Modo de visualización")
    ap.add_argument("--sample", type=int, default=-1,
                    help="Número de pacientes a renderizar (-1 = todos)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    mask_dir = Path(args.mask_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    patients = sorted([p.name for p in data_dir.iterdir()
                       if p.is_dir() and p.name.startswith("paciente_")])
    if args.sample > 0:
        patients = patients[:args.sample]

    print(f"🖼 Renderizando {len(patients)} pacientes (modo={args.mode})...")

    for pid in patients:
        try:
            X = np.load(data_dir / pid / "X.npy").astype(np.float32)
            mask = np.load(mask_dir / f"{pid}_{args.mode}.npy")
            out_img = out_dir / f"{pid}_{args.mode}.png"
            make_and_save_plot(X, mask, out_img)
            print(f"   ✓ {pid} → {out_img.name}")
        except Exception as e:
            print(f"   ⚠️ Error en {pid}: {e}")

    print(f"\n✅ Imágenes guardadas en: {out_dir}")

if __name__ == "__main__":
    main()
