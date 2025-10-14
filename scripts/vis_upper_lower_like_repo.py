#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualización estilo repo base:
- Dos subplots (Upper y Lower) del split elegido
- Sin ejes, sin grid, paneles transparentes
- Leyenda con números de diente
- Paleta personalizada (0 piel; 21 resaltado)
- Auto-arreglo de forma de puntos: (P,3), (3,P) o vector plano 3P
- --subsample para muestrear puntos sin tocar el dataset
- Opción --plotly (HTML interactivo)
"""
import argparse, json, random
from pathlib import Path
import numpy as np

import argparse, json, random
from pathlib import Path
import numpy as np

PALETTE = {
    0:  "#DECBBA",
    11: "#0000FF", 12: "#008000", 13: "#FF8C00",
    14: "#800080", 15: "#00FFFF", 16: "#FF00FF",
    17: "#FFFF00", 18: "#A52A2A", 21: "#00FF66",
    22: "#000080", 23: "#008080", 24: "#EE82EE",
    25: "#FA8072", 26: "#FFD700", 27: "#ADD8E6",
    28: "#FF7F50", 31: "#808000", 32: "#C0C0C0",
    33: "#808080", 34: "#000000", 35: "#8B0000",
    36: "#006400", 37: "#00008B", 38: "#9400D3",
    41: "#CD853F", 42: "#D2691E", 43: "#C71585",
    44: "#87CEFA", 45: "#FFB6C1", 46: "#DDA0DD",
    47: "#F0E68C", 48: "#B0E0E6",
}

def fix_points_shape(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts)
    if pts.ndim == 2 and pts.shape[1] == 3: return pts.astype(np.float32, copy=False)
    if pts.ndim == 2 and pts.shape[0] == 3: return pts.T.astype(np.float32, copy=False)
    if pts.ndim == 1 and pts.size % 3 == 0: return pts.reshape(-1,3).astype(np.float32, copy=False)
    raise ValueError(f"Forma de puntos inesperada: {pts.shape}")

def load_split(splits_dir: Path, split="test"):
    X = np.load(splits_dir / f"X_{split}.npz")["X"]
    Y = np.load(splits_dir / f"Y_{split}.npz")["Y"].astype("int32")
    meta_path = splits_dir / "meta.json"
    meta = json.load(open(meta_path)) if meta_path.exists() else {}
    cases = meta.get(f"cases_{split}", [])
    return X, Y, cases

def pick_random_by_jaw(cases, jaw):
    idxs = [i for i,c in enumerate(cases) if c.get("jaw")==jaw]
    return (random.choice(idxs) if idxs else None)

def mpl_style_axes(ax):
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
    ax.grid(False)

def mpl_equalize_axes(ax, pts, pad=0.05):
    x,y,z = pts[:,0], pts[:,1], pts[:,2]
    cx,cy,cz = (x.max()+x.min())/2, (y.max()+y.min())/2, (z.max()+z.min())/2
    r = max(x.max()-x.min(), y.max()-y.min(), z.max()-z.min())/2 * (1+pad)
    ax.set_xlim(cx-r, cx+r); ax.set_ylim(cy-r, cy+r); ax.set_zlim(cz-r, cz+r)

def mpl_scatter_by_label(ax, pts, labels, s=1.8, alpha=0.6):
    pts = fix_points_shape(pts); labels = np.asarray(labels).reshape(-1)
    if pts.shape[0] != labels.shape[0]: raise ValueError(f"#puntos {pts.shape[0]} ≠ #labels {labels.shape[0]}")
    uniq = np.unique(labels)
    for l in uniq:
        m = (labels==l)
        if m.any():
            ax.scatter(pts[m,0], pts[m,1], pts[m,2],
                       c=PALETTE.get(int(l), "#C8C8C8"),
                       label=str(int(l)), s=s, depthshade=False, alpha=alpha, edgecolors="none")

def visualize_mpl(upper, lower, label_upper, label_lower, info_upper, info_lower,
                  elev=20, azim=-60, s=1.8, alpha=0.6, dpi=300, out=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    upper = fix_points_shape(upper); lower = fix_points_shape(lower)
    fig = plt.figure(figsize=(12,6), dpi=dpi)

    ax1 = fig.add_subplot(121, projection="3d")
    mpl_style_axes(ax1); mpl_scatter_by_label(ax1, upper, label_upper, s=s, alpha=alpha)
    ax1.view_init(elev=elev, azim=azim); mpl_equalize_axes(ax1, upper)
    ax1.set_title(f"Upper — {info_upper}", pad=10)
    ax1.legend(loc="upper left", bbox_to_anchor=(1.02,1.0), frameon=False, title="labels")

    ax2 = fig.add_subplot(122, projection="3d")
    mpl_style_axes(ax2); mpl_scatter_by_label(ax2, lower, label_lower, s=s, alpha=alpha)
    ax2.view_init(elev=elev, azim=azim); mpl_equalize_axes(ax2, lower)
    ax2.set_title(f"Lower — {info_lower}", pad=10)
    ax2.legend(loc="upper left", bbox_to_anchor=(1.02,1.0), frameon=False, title="labels")

    fig.tight_layout()
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

def visualize_plotly(upper, lower, label_upper, label_lower, info_upper, info_lower, out_html=None):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    upper = fix_points_shape(upper); lower = fix_points_shape(lower)
    color_map = {str(k): v for k,v in PALETTE.items()}

    fig = make_subplots(rows=1, cols=2, specs=[[{"type":"scene"}, {"type":"scene"}]],
                        subplot_titles=(f"Upper — {info_upper}", f"Lower — {info_lower}"))

    labs_u = label_upper.astype(int).astype(str)
    fig.add_trace(go.Scatter3d(x=upper[:,0], y=upper[:,1], z=upper[:,2],
                               mode="markers", marker=dict(size=2, color=[color_map.get(l, "#C8C8C8") for l in labs_u]),
                               hoverinfo="skip"), row=1, col=1)
    labs_l = label_lower.astype(int).astype(str)
    fig.add_trace(go.Scatter3d(x=lower[:,0], y=lower[:,1], z=lower[:,2],
                               mode="markers", marker=dict(size=2, color=[color_map.get(l, "#C8C8C8") for l in labs_l]),
                               hoverinfo="skip"), row=1, col=2)
    for col in (1,2):
        fig.update_scenes(dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                               aspectmode="data", camera=dict(eye=dict(x=1.5,y=1.5,z=0.8))), row=1, col=col)
    fig.update_layout(margin=dict(l=0,r=0,t=40,b=0))
    if out_html:
        Path(out_html).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(out_html, include_plotlyjs="cdn")
    return fig

def maybe_subsample(pts, labels, step):
    pts = fix_points_shape(pts); labels = np.asarray(labels).reshape(-1)
    if step <= 1: return pts, labels
    idx = np.arange(pts.shape[0])[::step]; return pts[idx], labels[idx]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir", required=True)
    ap.add_argument("--which_split", default="test", choices=["train","val","test"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--point_size", type=float, default=1.8)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--elev", type=float, default=20.0)
    ap.add_argument("--azim", type=float, default=-60.0)
    ap.add_argument("--subsample", type=int, default=1)
    ap.add_argument("--out_png", default="figures/upper_lower_like_repo.png")
    ap.add_argument("--plotly", action="store_true")
    ap.add_argument("--out_html", default="figures/upper_lower_like_repo.html")
    args = ap.parse_args()

    random.seed(args.seed)
    splits = Path(args.splits_dir)
    X, Y, cases = load_split(splits, args.which_split)

    iu = pick_random_by_jaw(cases, "upper"); il = pick_random_by_jaw(cases, "lower")
    if iu is None or il is None: raise SystemExit("No encontré muestras upper/lower en el split.")
    up_pts, up_lab = fix_points_shape(X[iu]), Y[iu]
    lo_pts, lo_lab = fix_points_shape(X[il]), Y[il]
    up_pts, up_lab = maybe_subsample(up_pts, up_lab, args.subsample)
    lo_pts, lo_lab = maybe_subsample(lo_pts, lo_lab, args.subsample)

    up_info = f"PID {cases[iu].get('pid','?')} • {cases[iu].get('jaw','?')} • {cases[iu].get('case_id','?')}"
    lo_info = f"PID {cases[il].get('pid','?')} • {cases[il].get('jaw','?')} • {cases[il].get('case_id','?')}"

    if args.plotly:
        try:
            visualize_plotly(up_pts, lo_pts, up_lab, lo_lab, up_info, lo_info, out_html=args.out_html)
            print(f"[OK] HTML -> {args.out_html}")
        except ModuleNotFoundError:
            raise SystemExit("Falta plotly. Instala con: conda install -c conda-forge plotly  (o pip install plotly)")
    else:
        visualize_mpl(up_pts, lo_pts, up_lab, lo_lab, up_info, lo_info,
                      elev=args.elev, azim=args.azim, s=args.point_size, alpha=args.alpha,
                      dpi=args.dpi, out=args.out_png)
        print(f"[OK] PNG -> {args.out_png}")

if __name__ == "__main__":
    main()
