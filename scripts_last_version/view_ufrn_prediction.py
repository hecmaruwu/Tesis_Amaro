#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

base = Path("data/UFRN/inference_8192/paciente_2/upper_full")
out  = Path("outputs/ufrn_inference/pointnet_fps_aug8192/paciente_2")

pts = np.load(base / "point_cloud_8192_raw.npy", allow_pickle=True).astype(np.float32)
pred = np.load(out / "pred_labels.npy", allow_pickle=True).astype(np.int32)

bg = pred == 0
other = (pred != 0) & (pred != 8)
d21 = pred == 8

views = [
    ("Vista XY", 0, 1),
    ("Vista XZ", 0, 2),
    ("Vista YZ", 1, 2),
]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, (title, a, b) in zip(axes, views):
    ax.scatter(pts[bg, a], pts[bg, b], c="lightgray", s=1, alpha=0.20)
    ax.scatter(pts[other, a], pts[other, b], c="blue", s=4, alpha=0.75)
    ax.scatter(pts[d21, a], pts[d21, b], c="red", s=25, alpha=1.0)

    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)

fig.suptitle("UFRN PointNet Prediction | Rojo=d21 | Azul=otras clases | Gris=background")
plt.tight_layout()

png_path = out / "ufrn_prediction_3views.png"
plt.savefig(png_path, dpi=300, bbox_inches="tight")
print("[OK] Imagen guardada en:", png_path)