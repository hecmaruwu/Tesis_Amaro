#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inferencia visual (GT vs Pred) para modelos PyTorch del flujo v2.
Soporta: PointNet, PointNet++, DilatedToothSegNet, Transformer3DSeg.
Lee modelos desde run_dir/checkpoints/best.pt y datos desde X_test/Y_test.
Genera: figuras PNG en run_dir/vis_compare/ con el diente 21 resaltado.
"""

import argparse, json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Función de normalización local (sin dependencias externas)
# ------------------------------------------------------------
def normalize_cloud(x: np.ndarray) -> np.ndarray:
    """Normaliza nube (N,3): centra en el origen y escala a radio unitario."""
    x = np.asarray(x, dtype=np.float32)
    centroid = x.mean(axis=0, keepdims=True)
    x = x - centroid
    furthest_distance = np.sqrt((x ** 2).sum(axis=1)).max()
    if furthest_distance > 0:
        x = x / (furthest_distance + 1e-6)
    return x

# ------------------------------------------------------------
# Modelos disponibles (importes locales)
# ------------------------------------------------------------
from train_models_v2 import (
    PointNetSeg, PointNet2Seg, DilatedToothSegNet, Transformer3DSeg, CLASS_NAMES
)

# Paleta de colores
PALETTE = matplotlib.colormaps.get_cmap("tab20")

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def load_npz_auto(path):
    with np.load(path, allow_pickle=False) as z:
        if "X" in z:
            return z["X"]
        elif "Y" in z:
            return z["Y"]
        else:
            return z[list(z.keys())[0]]

def load_split(data_dir, split="test"):
    X = load_npz_auto(Path(data_dir)/f"X_{split}.npz")
    Y = load_npz_auto(Path(data_dir)/f"Y_{split}.npz").astype(np.int64)
    assert X.shape[0] == Y.shape[0], f"Shapes incompatibles: X{X.shape}, Y{Y.shape}"
    return X, Y

def build_model(name, num_classes):
    if name == "pointnet": return PointNetSeg(num_classes)
    if name == "pointnetpp": return PointNet2Seg(num_classes)
    if name == "dilated": return DilatedToothSegNet(num_classes)
    if name == "transformer": return Transformer3DSeg(num_classes)
    raise ValueError(f"Modelo no soportado: {name}")

def scatter3(ax, P, c, s=4.0, alpha=0.9):
    ax.scatter(P[:,0], P[:,1], P[:,2], c=c, s=s, alpha=alpha, depthshade=False)
    ax.set_axis_off()

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--which_model", default="best", choices=["best","final"])
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_samples_vis", type=int, default=6)
    ap.add_argument("--vis_subsample", type=int, default=4000)
    ap.add_argument("--highlight_tooth", type=int, default=21)
    ap.add_argument("--cuda", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    run_dir = Path(args.run_dir)
    model_name = run_dir.name.split("_")[0]  # ej. pointnet

    ckpt_pt = run_dir / "checkpoints" / f"{args.which_model}.pt"
    ckpt_pth = run_dir / "checkpoints" / f"{args.which_model}.pth"
    ckpt = ckpt_pt if ckpt_pt.exists() else ckpt_pth

    if not ckpt.exists():
        raise FileNotFoundError(f"No se encontró checkpoint ni .pt ni .pth en {run_dir}/checkpoints")

    print(f"[LOAD] {model_name} desde {ckpt}")

    # Datos
    X, Y = load_split(args.data_dir, "test")
    ncls = int(Y.max() + 1)

    # Modelo
    model = build_model(model_name, ncls).to(device)
    state = torch.load(ckpt, map_location=device)
    state_dict = state["model"] if "model" in state else state
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    out_dir = run_dir / "vis_compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Inferencia visual
    idxs = np.linspace(0, X.shape[0]-1, args.num_samples_vis, dtype=int)
    cmap = matplotlib.colormaps.get_cmap("tab20")


    for j, i in enumerate(idxs):
        pts_np = normalize_cloud(X[i])
        gt = Y[i]

        pts_t = torch.from_numpy(pts_np).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(pts_t)
            pred = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()

        # Submuestreo visual
        if args.vis_subsample and pts_np.shape[0] > args.vis_subsample:
            sel = np.random.choice(pts_np.shape[0], args.vis_subsample, replace=False)
        else:
            sel = np.arange(pts_np.shape[0])

        pts_draw = pts_np[sel]
        gt_sel = gt[sel]
        pr_sel = pred[sel]

        fig = plt.figure(figsize=(12,6))
        ax1 = fig.add_subplot(121, projection="3d")
        ax2 = fig.add_subplot(122, projection="3d")
        scatter3(ax1, pts_draw, c=cmap(gt_sel % cmap.N)); ax1.set_title("Ground Truth")
        scatter3(ax2, pts_draw, c=cmap(pr_sel % cmap.N)); ax2.set_title("Predicción")

        # Resalta diente 21
        if args.highlight_tooth in np.unique(pr_sel):
            mask = (pr_sel == args.highlight_tooth)
            ax2.scatter(pts_draw[mask,0], pts_draw[mask,1], pts_draw[mask,2],
                        c="lime", s=10, edgecolors="k", linewidths=0.3)

        fig.tight_layout()
        fig.savefig(out_dir / f"sample_{j:02d}.png", dpi=200)
        plt.close(fig)

    print(f"[OK] Visualizaciones -> {out_dir}")

# ------------------------------------------------------------
if __name__ == "__main__":
    main()
