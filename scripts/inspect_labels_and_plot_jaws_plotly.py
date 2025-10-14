#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualización tipo paper ULTRA DENSIDAD
- Puntos grandes y muy visibles (modo publicación)
- Alta resolución (1600 dpi)
- Encía translúcida pero más marcada
- Fondo gris claro, vista 3D inmersiva
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from pathlib import Path
import json

LABEL_COLORS = {
    0: 'red', 11: 'blue', 12: 'green', 13: 'orange', 14: 'purple',
    15: 'cyan', 16: 'magenta', 17: 'yellow', 18: 'brown', 21: 'lime',
    22: 'navy', 23: 'teal', 24: 'violet', 25: 'salmon', 26: 'gold',
    27: 'lightblue', 28: 'coral', 31: 'olive', 32: 'silver', 33: 'gray',
    34: 'black', 35: 'darkred', 36: 'darkgreen', 37: 'darkblue',
    38: 'darkviolet', 41: 'peru', 42: 'chocolate', 43: 'mediumvioletred',
    44: 'lightskyblue', 45: 'lightpink', 46: 'plum', 47: 'khaki',
    48: 'powderblue',
}


def load_npz_split(split_dir: Path):
    X = np.load(split_dir / "X_test.npz")["X"]
    Y = np.load(split_dir / "Y_test.npz")["Y"]
    print(f"[DATA] X={X.shape}, Y={Y.shape}")
    meta = {}
    meta_path = split_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    return X, Y, meta


def inspect_labels(Y):
    uniq, counts = np.unique(Y, return_counts=True)
    print("\n[ETIQUETAS ÚNICAS EN Y_TEST]:")
    for u, c in zip(uniq, counts):
        print(f"  Clase {int(u):2d} → {c} puntos")
    return uniq


def plot_upper_lower_side_by_side(X, Y, sample_name, out_dir, axis="y", dpi=1600):
    out_dir.mkdir(parents=True, exist_ok=True)

    ax_idx = {"x": 0, "y": 1, "z": 2}[axis]
    median_plane = np.median(X[..., ax_idx])
    mask_upper = X[..., ax_idx] > median_plane
    mask_lower = X[..., ax_idx] <= median_plane

    pts_upper, lbl_upper = X[mask_upper], Y[mask_upper]
    pts_lower, lbl_lower = X[mask_lower], Y[mask_lower]

    print(f"[INFO] Superior: {pts_upper.shape[0]} pts | Inferior: {pts_lower.shape[0]} pts")

    fig = plt.figure(figsize=(26, 12), dpi=dpi)
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    def _scatter(ax, pts, lbl, title):
        colors, sizes = [], []
        for l in lbl:
            if int(l) == 0:  # encía
                colors.append(to_rgba('red', alpha=0.45))
                sizes.append(15)
            else:
                colors.append(to_rgba(LABEL_COLORS.get(int(l), "gray"), alpha=1.0))
                sizes.append(35)

        colors = np.array(colors)
        sizes = np.array(sizes)

        ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            c=colors, s=sizes, edgecolors='none'
        )

        # Zoom más cerrado
        ax.set_xlim(np.percentile(pts[:, 0], 2), np.percentile(pts[:, 0], 98))
        ax.set_ylim(np.percentile(pts[:, 1], 5), np.percentile(pts[:, 1], 98))
        ax.set_zlim(np.percentile(pts[:, 2], 2), np.percentile(pts[:, 2], 98))

        ax.view_init(elev=35, azim=-50)
        ax.set_box_aspect([1, 1, 0.7])
        ax.set_title(title, fontsize=18, pad=20)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_facecolor("#f6f6f6")

    _scatter(ax1, pts_upper, lbl_upper, f"Upper {sample_name}")
    _scatter(ax2, pts_lower, lbl_lower, f"Lower {sample_name}")

    plt.tight_layout()
    out_path = out_dir / f"upper_lower_{sample_name}_ultradense.png"
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[OK] Figura guardada en {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", required=True,
                    help="Carpeta con X_test.npz / Y_test.npz / meta.json")
    ap.add_argument("--sample", type=int, default=0,
                    help="Índice del paciente en test (0 por defecto)")
    ap.add_argument("--axis", choices=["y", "z"], default="y",
                    help="Eje usado para dividir maxilares")
    ap.add_argument("--out_dir", default="figures_ultradense",
                    help="Carpeta de salida")
    ap.add_argument("--dpi", type=int, default=1600,
                    help="Resolución de imagen")
    args = ap.parse_args()

    split_dir = Path(args.split_dir)
    out_dir = Path(args.out_dir)

    X, Y, meta = load_npz_split(split_dir)
    inspect_labels(Y)

    sample_name = f"paciente_{args.sample}"
    if meta and "cases_test" in meta and args.sample < len(meta["cases_test"]):
        pid = meta["cases_test"][args.sample].get("pid", sample_name)
        sample_name = pid

    print(f"[INFO] Visualizando muestra {args.sample} ({sample_name}) en modo ULTRA DENSO")
    plot_upper_lower_side_by_side(X[args.sample], Y[args.sample],
                                  sample_name, out_dir, axis=args.axis, dpi=args.dpi)


if __name__ == "__main__":
    main()
