#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inferencia por lotes + visualización (GT vs Pred) y mapas de error,
con opciones de submuestreo visual y resaltado de una clase.

Ejemplo:
  python -m scripts.infer_and_save_vis \
    --run_dir runs_grid/mi_experimento \
    --data_dir data/3dteethseg/splits/npz_4096_pairs \
    --which_model best \
    --num_samples_vis 8 --vis_subsample 3000 --save_svg
"""
import argparse
from pathlib import Path
import numpy as np
from tensorflow import keras

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------- Utilidades de carga robusta -----------------
def _load_array_any(p: Path, key_candidates=("X","Y","arr_0")):
    p = Path(p)
    if p.suffix == ".npy":
        return np.load(p, allow_pickle=False)
    elif p.suffix == ".npz":
        with np.load(p, allow_pickle=False) as z:
            for k in key_candidates:
                if k in z: 
                    return z[k]
            return z[list(z.keys())[0]]
    else:
        raise FileNotFoundError(f"No puedo leer: {p}")

def load_split_any(data_dir: Path, split="test"):
    data_dir = Path(data_dir)
    Xp = data_dir / f"X_{split}.npz"
    if not Xp.exists():
        Xp = data_dir / f"X_{split}.npy"
    Yp = data_dir / f"Y_{split}.npz"
    if not Yp.exists():
        Yp = data_dir / f"Y_{split}.npy"
    if not Xp.exists() or not Yp.exists():
        raise FileNotFoundError(f"Faltan X_{split} o Y_{split} en {data_dir}")
    X = _load_array_any(Xp).astype(np.float32)
    Y = _load_array_any(Yp).astype(np.int64)
    assert X.ndim == 3 and X.shape[:2] == Y.shape[:2], f"Shapes incompatibles: X{X.shape} Y{Y.shape}"
    return X, Y

def normalize_cloud_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mean = x.mean(axis=0, keepdims=True)
    x = x - mean
    r = np.linalg.norm(x, axis=1, keepdims=True).max(axis=0, keepdims=True)  # (1,1)
    x = x / (r + 1e-6)
    return x

def find_model_path(run_dir: Path, which: str | None):
    cands = []
    if which is None:
        cands += [run_dir/"checkpoints/best", run_dir/"final_model"]
        cands += [run_dir/"run_single"/"checkpoints/best", run_dir/"run_single"/"final_model"]
    elif which == "best":
        cands += [run_dir/"checkpoints/best", run_dir/"run_single"/"checkpoints/best"]
    else:
        cands += [run_dir/"final_model", run_dir/"run_single"/"final_model"]

    for p in cands:
        if p.exists():
            return p
    raise FileNotFoundError(f"No se encontró modelo (best/final) bajo {run_dir}")

def scatter3(ax, P, c, s=6.0, alpha=0.95):
    ax.scatter(P[:,0], P[:,1], P[:,2], c=c, s=s, alpha=alpha, depthshade=False)
    ax.set_axis_off()

# ------------------------------ MAIN ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Carpeta del experimento (contiene checkpoints/ o run_single/)")
    ap.add_argument("--data_dir", required=True, help="Carpeta con X_*.npz/.npy y Y_*.npz/.npy")
    ap.add_argument("--which_model", default=None, choices=[None,"best","final"])
    ap.add_argument("--batch_size", type=int, default=16)

    # Visual
    ap.add_argument("--num_samples_vis", type=int, default=6)
    ap.add_argument("--dpi", type=int, default=180)
    ap.add_argument("--point_size", type=float, default=6.0)
    ap.add_argument("--alpha", type=float, default=0.95)
    ap.add_argument("--vis_subsample", type=int, default=0,
                    help="Si >0, submuestrea esa cantidad de puntos por nube para dibujar.")
    ap.add_argument("--elev", type=float, default=20.0)
    ap.add_argument("--azim", type=float, default=-60.0)
    ap.add_argument("--save_svg", action="store_true")

    # Resaltado opcional
    ap.add_argument("--highlight_tooth", type=int, default=None)
    ap.add_argument("--alpha_highlight", type=float, default=1.0)
    ap.add_argument("--size_highlight", type=float, default=2.5)
    ap.add_argument("--highlight_color", type=str, default="lime")
    ap.add_argument("--outline_highlight", action="store_true")

    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    data_dir = Path(args.data_dir)

    # Datos
    Xte, Yte = load_split_any(data_dir, "test")
    N, P, _ = Xte.shape
    ncls = int(Yte.max(initial=0) + 1)
    print(f"[DATA] Test: {Xte.shape}, clases≈{ncls}")

    # Modelo
    mdl_path = find_model_path(run_dir, args.which_model)
    print(f"[LOAD] {mdl_path}")
    model = keras.models.load_model(mdl_path, compile=False)

    # Normalizar por nube
    Xte_norm = np.empty_like(Xte, dtype=np.float32)
    for i in range(N):
        Xte_norm[i] = normalize_cloud_np(Xte[i])

    # Inferencia
    probs = model.predict(Xte_norm, batch_size=args.batch_size, verbose=1)
    ypred = probs.argmax(axis=-1)
    ncls_pred = probs.shape[-1]
    print("[PRED] clases modelo:", ncls_pred)

    # Salida
    out_parent = run_dir/"run_single" if (run_dir/"run_single").exists() else run_dir
    out_dir = out_parent / "vis_compare"
    out_err = out_dir / "errors"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_err.mkdir(parents=True, exist_ok=True)

    # Índices a graficar
    k = min(args.num_samples_vis, N)
    idxs = np.linspace(0, N-1, k, dtype=int)

    # Colormap
    cmap = plt.cm.get_cmap("tab20", ncls_pred)

    def maybe_subsample(P3, *arrays):
        """Submuestrea para dibujar si vis_subsample > 0."""
        if args.vis_subsample and P3.shape[0] > args.vis_subsample:
            sel = np.random.choice(P3.shape[0], size=args.vis_subsample, replace=False)
            return (P3[sel],) + tuple(a[sel] for a in arrays)
        return (P3,) + arrays

    for j, i in enumerate(idxs, 1):
        pts_raw = Xte[i]
        gt = Yte[i]
        pr = ypred[i]

        pts_draw, gt_draw, pr_draw = maybe_subsample(pts_raw, gt, pr)

        # Figura GT vs Pred
        fig = plt.figure(figsize=(12, 6), dpi=args.dpi)
        ax1 = fig.add_subplot(121, projection="3d"); ax1.view_init(args.elev, args.azim)
        ax2 = fig.add_subplot(122, projection="3d"); ax2.view_init(args.elev, args.azim)

        scatter3(ax1, pts_draw, c=cmap(gt_draw % cmap.N), s=args.point_size, alpha=args.alpha)
        ax1.set_title("Real")

        scatter3(ax2, pts_draw, c=cmap(pr_draw % cmap.N), s=args.point_size, alpha=args.alpha)
        ax2.set_title("Predicho")

        # Resaltar clase en la predicción (opcional)
        if args.highlight_tooth is not None:
            m = (pr_draw == int(args.highlight_tooth))
            if m.any():
                ax2.scatter(
                    pts_draw[m,0], pts_draw[m,1], pts_draw[m,2],
                    c=args.highlight_color,
                    s=args.point_size * args.size_highlight,
                    alpha=args.alpha_highlight,
                    depthshade=False,
                    edgecolors="k" if args.outline_highlight else None
                )

        fig.tight_layout()
        base = f"{j:03d}"
        png_path = out_dir / f"{base}.png"
        fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
        if args.save_svg:
            fig.savefig(out_dir / f"{base}.svg", bbox_inches="tight")
        plt.close(fig)

        # Mapa de errores
        err_mask = (gt != pr)
        if err_mask.any():
            pts_err, err_draw = maybe_subsample(pts_raw, err_mask.astype(bool))
            fig2 = plt.figure(figsize=(6, 6), dpi=args.dpi)
            axe = fig2.add_subplot(111, projection="3d"); axe.view_init(args.elev, args.azim)
            colors = np.where(err_draw, "crimson", "lightgray")
            scatter3(axe, pts_err, c=colors, s=args.point_size, alpha=1.0)
            axe.set_title("Errores (rojo)")
            fig2.tight_layout()
            fig2.savefig(out_err / f"{base}_errors.png", dpi=args.dpi, bbox_inches="tight")
            if args.save_svg:
                fig2.savefig(out_err / f"{base}_errors.svg", bbox_inches="tight")
            plt.close(fig2)

        print(f"[OK] {png_path}")

    print(f"✅ Listo. Carpeta: {out_dir}")

if __name__ == "__main__":
    main()
