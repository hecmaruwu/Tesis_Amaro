#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, argparse, json, math
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# -------- Normalización idéntica a train --------
def normalize_cloud_np(x):
    # x: (P,3) float
    x = x.astype(np.float32)
    mean = x.mean(axis=0, keepdims=True)
    x = x - mean
    r = np.linalg.norm(x, axis=1, keepdims=True).max(axis=0, keepdims=True)  # (1,1)
    x = x / (r + 1e-6)
    return x

def load_best_or_final(run_dir: Path, which_model: str|None):
    # Detecta run_single/…
    rs = run_dir / "run_single"
    if not rs.exists():
        raise FileNotFoundError(f"No se encontró run_single bajo: {run_dir}")
    # Decide best/final
    m = which_model
    if m is None:
        if (rs/"checkpoints/best").exists(): m = "best"
        elif (rs/"final_model").exists():   m = "final"
        else: raise FileNotFoundError(f"No existe best/final en {rs}")
    mdl_path = rs/"checkpoints/best" if m=="best" else rs/"final_model"
    print("[LOAD] Keras load_model desde:", mdl_path)
    model = keras.models.load_model(mdl_path, compile=False)
    return model, rs

def scatter3(ax, P, c, s=4, alpha=0.8):
    ax.scatter(P[:,0], P[:,1], P[:,2], c=c, s=s, alpha=alpha, depthshade=False)
    ax.set_axis_off()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)     # padre del run_single
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--which_model", default=None, choices=[None,"best","final"])
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--point_size", type=float, default=5.0)
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--elev", type=float, default=20.0)
    ap.add_argument("--azim", type=float, default=-60.0)
    # highlight opcional
    ap.add_argument("--highlight_tooth", type=int, default=None)
    ap.add_argument("--alpha_highlight", type=float, default=1.0)
    ap.add_argument("--size_highlight", type=float, default=2.5)
    ap.add_argument("--outline_highlight", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    data_dir = Path(args.data_dir)
    model, run_single = load_best_or_final(run_dir, args.which_model)

    Xte = np.load(data_dir/"X_test.npz")["X"]  # (N,P,3)
    Yte = np.load(data_dir/"Y_test.npz")["Y"]  # (N,P)
    N, P, _ = Xte.shape
    print(f"[DATA] Test: {Xte.shape}, classes≈{int(Yte.max()+1)}")

    # --- NORMALIZAR cada nube como en train ---
    Xte_norm = np.empty_like(Xte, dtype=np.float32)
    for i in range(N):
        Xte_norm[i] = normalize_cloud_np(Xte[i])

    # Predict
    probs = model.predict(Xte_norm, verbose=0)
    ypred = probs.argmax(axis=-1)  # (N,P)

    # Dump distribución (sanity check)
    counts = np.bincount(ypred.reshape(-1), minlength=probs.shape[-1])
    topk = np.argsort(counts)[::-1][:5]
    print("[PRED] top clases:", [(int(k), int(counts[k])) for k in topk])

    # Carpeta salida
    out_dir = run_single/"vis_compare"
    out_err = out_dir/"errors"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_err.mkdir(parents=True, exist_ok=True)

    # Colormap discreto
    cmap = plt.cm.get_cmap('tab20', probs.shape[-1])

    # Muestras
    take = min(args.samples, N)
    idxs = np.linspace(0, N-1, take, dtype=int)

    for j, i in enumerate(idxs, 1):
        pts_raw = Xte[i]
        pts     = Xte_norm[i]
        gt      = Yte[i]
        pr      = ypred[i]

        fig = plt.figure(figsize=(12,6), dpi=args.dpi)
        ax1 = fig.add_subplot(121, projection='3d'); ax1.view_init(args.elev, args.azim)
        ax2 = fig.add_subplot(122, projection='3d'); ax2.view_init(args.elev, args.azim)

        scatter3(ax1, pts_raw, c=cmap(gt % cmap.N), s=args.point_size, alpha=args.alpha)
        ax1.set_title("Real")

        scatter3(ax2, pts_raw, c=cmap(pr % cmap.N), s=args.point_size, alpha=args.alpha)
        ax2.set_title("Predicho")

        fig.tight_layout()
        png = out_dir/f"{j:03d}.png"
        fig.savefig(png, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

        # capas de error (dónde difiere)
        err_mask = (gt != pr)
        if err_mask.any():
            fig2 = plt.figure(figsize=(6,6), dpi=args.dpi)
            axe = fig2.add_subplot(111, projection='3d'); axe.view_init(args.elev, args.azim)
            c_err = np.where(err_mask, 'crimson', 'lightgray')
            scatter3(axe, pts_raw, c=c_err, s=args.point_size, alpha=1.0)
            axe.set_title("Errores (rojo)")
            fig2.tight_layout()
            fig2.savefig(out_err/f"{j:03d}_errors.png", dpi=args.dpi, bbox_inches="tight")
            plt.close(fig2)

        print(f"[OK] {png}")

    print(f"✅ Listo. Revisa: {out_dir}")

if __name__ == "__main__":
    main()
