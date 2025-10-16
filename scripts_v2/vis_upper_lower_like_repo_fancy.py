#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualización automática de arcadas (upper/lower) con paleta personalizada y leyenda mejorada.

✔ Usa paleta de colores personalizada
✔ Resalta el diente 21 en verde ("lime")
✔ Leyendas grandes con cuadrados de color
✔ Distribución limpia tipo paper
"""

import argparse, json, random
from pathlib import Path
import numpy as np
import matplotlib.patches as mpatches

# Paleta de colores personalizada
PALETTE = {
    0: 'red', 11: 'blue', 12: 'green', 13: 'orange', 14: 'purple',
    15: 'cyan', 16: 'magenta', 17: 'yellow', 18: 'brown', 21: 'lime',
    22: 'navy', 23: 'teal', 24: 'violet', 25: 'salmon', 26: 'gold',
    27: 'lightblue', 28: 'coral', 31: 'olive', 32: 'silver', 33: 'gray',
    34: 'black', 35: 'darkred', 36: 'darkgreen', 37: 'darkblue',
    38: 'darkviolet', 41: 'peru', 42: 'chocolate', 43: 'mediumvioletred',
    44: 'lightskyblue', 45: 'lightpink', 46: 'plum', 47: 'khaki', 48: 'powderblue'
}

def fix_points_shape(pts):
    pts = np.asarray(pts)
    if pts.ndim == 2 and pts.shape[1] == 3: return pts
    if pts.ndim == 2 and pts.shape[0] == 3: return pts.T
    if pts.ndim == 1 and pts.size % 3 == 0: return pts.reshape(-1, 3)
    raise ValueError(f"Forma inesperada de puntos: {pts.shape}")

def load_split(splits_dir, split="test"):
    X = np.load(splits_dir / f"X_{split}.npz")["X"]
    Y = np.load(splits_dir / f"Y_{split}.npz")["Y"].astype("int32")
    meta_path = splits_dir / "meta.json"
    meta = json.load(open(meta_path)) if meta_path.exists() else {}
    cases = meta.get(f"cases_{split}", [])
    return X, Y, cases

def try_infer_patient_from_path(path):
    name = str(path)
    for token in name.split("/"):
        if token.lower().startswith("paciente") or token.lower().startswith("p"):
            return token
    return "unknown_patient"

def mpl_style_axes(ax):
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
    ax.grid(False)

def mpl_equalize_axes(ax, pts, pad=0.05):
    x,y,z = pts[:,0], pts[:,1], pts[:,2]
    cx,cy,cz = (x.max()+x.min())/2, (y.max()+y.min())/2, (z.max()+z.min())/2
    r = max(x.max()-x.min(), y.max()-y.min(), z.max()-z.min())/2 * (1+pad)
    ax.set_xlim(cx-r, cx+r); ax.set_ylim(cy-r, cy+r); ax.set_zlim(cz-r, cz+r)

def mpl_scatter_by_label(ax, pts, labels, s=2.5, alpha=0.7):
    uniq = np.unique(labels)
    for l in uniq:
        m = (labels==l)
        if m.any():
            ax.scatter(pts[m,0], pts[m,1], pts[m,2],
                       c=PALETTE.get(int(l), "#C8C8C8"),
                       label=str(int(l)), s=s, alpha=alpha,
                       depthshade=False, edgecolors="none")

def visualize_mpl(upper, lower, label_upper, label_lower, info_upper, info_lower,
                  elev=20, azim=-60, s=2.5, alpha=0.7, dpi=300, out=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12,6), dpi=dpi)

    # --- Upper ---
    ax1 = fig.add_subplot(121, projection="3d")
    mpl_style_axes(ax1); mpl_scatter_by_label(ax1, upper, label_upper, s=s, alpha=alpha)
    ax1.view_init(elev=elev, azim=azim); mpl_equalize_axes(ax1, upper)
    ax1.set_title(f"Upper — {info_upper}", pad=10, fontsize=12)

    # --- Lower ---
    ax2 = fig.add_subplot(122, projection="3d")
    mpl_style_axes(ax2); mpl_scatter_by_label(ax2, lower, label_lower, s=s, alpha=alpha)
    ax2.view_init(elev=elev, azim=azim); mpl_equalize_axes(ax2, lower)
    ax2.set_title(f"Lower — {info_lower}", pad=10, fontsize=12)

    # --- Leyenda tipo cuadrado ---
    unique_labels = sorted(list(PALETTE.keys()))
    patches = [mpatches.Patch(color=PALETTE[l], label=str(l)) for l in unique_labels]
    fig.legend(handles=patches, loc="center right", title="Dientes",
               bbox_to_anchor=(1.18, 0.5), frameon=False, fontsize=9,
               title_fontsize=10, ncol=2)

    fig.tight_layout(rect=[0, 0, 0.9, 1])  # espacio para la leyenda
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
        print(f"[OK] Guardado -> {out}")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir", required=True)
    ap.add_argument("--which_split", default="test")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--point_size", type=float, default=2.5)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--elev", type=float, default=20.0)
    ap.add_argument("--azim", type=float, default=-60.0)
    ap.add_argument("--out_png", default="figures/upper_lower_fancy.png")
    args = ap.parse_args()

    splits = Path(args.splits_dir)
    X, Y, cases = load_split(splits, args.which_split)

    if not cases:
        print("[WARN] meta.json no tiene 'cases_*'. Seleccionando dos muestras aleatorias.")
        iu, il = 0, 1
        info_u = try_infer_patient_from_path(splits)
        info_l = try_infer_patient_from_path(splits)
    else:
        upper_candidates = [i for i,c in enumerate(cases) if c.get("jaw") == "upper"]
        lower_candidates = [i for i,c in enumerate(cases) if c.get("jaw") == "lower"]
        if upper_candidates and lower_candidates:
            iu = random.choice(upper_candidates)
            il = random.choice(lower_candidates)
            info_u = cases[iu].get("pid","?")+" ("+cases[iu].get("jaw","?")+")"
            info_l = cases[il].get("pid","?")+" ("+cases[il].get("jaw","?")+")"
        else:
            iu, il = 0, 1
            info_u, info_l = "auto_upper", "auto_lower"

    up_pts, up_lab = fix_points_shape(X[iu]), Y[iu]
    lo_pts, lo_lab = fix_points_shape(X[il]), Y[il]

    visualize_mpl(up_pts, lo_pts, up_lab, lo_lab, info_u, info_l,
                  elev=args.elev, azim=args.azim, s=args.point_size, alpha=args.alpha,
                  dpi=args.dpi, out=args.out_png)

if __name__ == "__main__":
    main()
