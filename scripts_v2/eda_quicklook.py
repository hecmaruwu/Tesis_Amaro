#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def load_split(root: Path, split: str):
    Xp = root / f"X_{split}.npz"
    Yp = root / f"Y_{split}.npz"
    if not Xp.exists():
        raise FileNotFoundError(f"No existe {Xp}")
    X = np.load(Xp)["X"]  # [N, P, 3]
    Y = None
    if Yp.exists():
        Y = np.load(Yp)["Y"]  # [N, P]
    return X, Y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", required=True, help="data/Teeth_3ds/merged_dataset")
    ap.add_argument("--split", default="train", choices=["train","val","test"])
    ap.add_argument("--sample_idx", type=int, default=0, help="índice de muestra a visualizar")
    ap.add_argument("--max_points_plot", type=int, default=5000, help="limitar puntos en figura")
    args = ap.parse_args()

    root = Path(args.npz_dir)
    X, Y = load_split(root, args.split)
    print(f"[OK] Cargado: X_{args.split}.npz  shape={X.shape}")
    if Y is not None:
        print(f"[OK] Cargado: Y_{args.split}.npz  shape={Y.shape}")
    else:
        print("[WARN] No hay Y; solo puntos")

    N, P, _ = X.shape
    s = max(0, min(args.sample_idx, N-1))
    pts = X[s]  # [P,3]
    lbl = Y[s] if Y is not None else None

    # Estadísticas rápidas
    print(f"\n=== Muestra {s} ===")
    print(f"Puntos: {pts.shape[0]}")
    if lbl is not None:
        classes, counts = np.unique(lbl, return_counts=True)
        print("Clases:", classes.tolist())
        print("Frecuencias:", counts.tolist())

    out_dir = root / "eda_figs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Scatter 3D (submuestreo para plot)
    vis_idx = np.arange(pts.shape[0])
    if pts.shape[0] > args.max_points_plot:
        vis_idx = np.random.default_rng(0).choice(pts.shape[0], args.max_points_plot, replace=False)
    pvis = pts[vis_idx]
    cvis = None
    if lbl is not None:
        cvis = lbl[vis_idx]

    fig = plt.figure(figsize=(8,6), dpi=800)
    ax = fig.add_subplot(111, projection="3d")
    if cvis is None:
        ax.scatter(pvis[:,0], pvis[:,1], pvis[:,2], s=1)
    else:
        sc = ax.scatter(pvis[:,0], pvis[:,1], pvis[:,2], c=cvis, s=1, cmap="tab20")
        plt.colorbar(sc, ax=ax, shrink=0.6)
    ax.set_title(f"Scatter 3D ({args.split}) idx={s}")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    fig.tight_layout()
    fig_path = out_dir / f"scatter3d_{args.split}_{s}.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    print("[OK] Figura:", fig_path)

    # Histograma de clases
    if lbl is not None:
        fig = plt.figure(figsize=(8,6), dpi=800)
        plt.hist(lbl, bins=np.arange(lbl.min(), lbl.max()+2)-0.5)
        plt.xlabel("Clase"); plt.ylabel("Cuenta")
        plt.title(f"Histograma de clases — {args.split} idx={s}")
        fig.tight_layout()
        hist_path = out_dir / f"class_hist_{args.split}_{s}.png"
        fig.savefig(hist_path, dpi=200)
        plt.close(fig)
        print("[OK] Figura:", hist_path)

if __name__ == "__main__":
    main()
