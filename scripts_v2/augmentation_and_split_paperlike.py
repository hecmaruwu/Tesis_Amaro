#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocesamiento 'paper-like' inspirado en PointNet/TeethSeg:
 - Normalización global por arcada (no por diente)
 - Muestreo FPS en lugar de aleatorio
 - Fondo limitado a ~10%
 - Augmentación leve: rotación Z ±15°, jitter 0.01, escala 0.9–1.1
 - Remapeo global 0..C-1 y balanceo por clase
 - Split por paciente (si se incluye metadata)
"""

import os, json, argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

# -------------------------------
# FPS Sampling (Farthest Point Sampling)
# -------------------------------
def farthest_point_sampling(points, n_samples):
    """Selecciona n_samples puntos por FPS."""
    N = points.shape[0]
    centroids = np.zeros((n_samples,), dtype=np.int64)
    distances = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(n_samples):
        centroids[i] = farthest
        centroid = points[farthest, :]
        dist = np.sum((points - centroid) ** 2, axis=1)
        mask = dist < distances
        distances[mask] = dist[mask]
        farthest = np.argmax(distances)
    return points[centroids]

# -------------------------------
# Normalización global por arcada
# -------------------------------
def normalize_cloud(points):
    mean = points.mean(axis=0, keepdims=True)
    centered = points - mean
    scale = np.linalg.norm(centered, axis=1).max()
    return centered / (scale + 1e-6)

# -------------------------------
# Augmentaciones suaves
# -------------------------------
def augment_cloud(points, rotate_deg=15, jitter_sigma=0.01, scale_min=0.9, scale_max=1.1):
    # rotación alrededor de Z
    theta = np.radians(np.random.uniform(-rotate_deg, rotate_deg))
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    points = points @ R.T
    # jitter leve
    noise = np.random.normal(0, jitter_sigma, points.shape)
    points += noise
    # escalado global
    scale = np.random.uniform(scale_min, scale_max)
    points *= scale
    return points

# -------------------------------
# Limitación de fondo
# -------------------------------
def cap_background(points, labels, N, cap_frac=0.1):
    """Limita el fondo (label=0) a una fracción cap_frac del total."""
    mask_bg = labels == 0
    idx_bg = np.where(mask_bg)[0]
    idx_fg = np.where(~mask_bg)[0]

    n_bg = int(N * cap_frac)
    n_fg = N - n_bg

    sel_bg = np.array([], dtype=np.int64)
    sel_fg = np.array([], dtype=np.int64)

    if len(idx_bg) > 0:
        sel_bg = np.random.choice(idx_bg, size=min(len(idx_bg), n_bg), replace=False)
    if len(idx_fg) > 0:
        sel_fg = np.random.choice(idx_fg, size=min(len(idx_fg), n_fg), replace=False)

    sel = np.concatenate([sel_bg, sel_fg], axis=0).astype(np.int64)

    # Si faltan puntos, rellenar con reemplazo
    if len(sel) < N:
        extra = np.random.choice(np.arange(len(points)), N - len(sel), replace=True)
        sel = np.concatenate([sel, extra]).astype(np.int64)

    return points[sel], labels[sel]


# -------------------------------
# Main pipeline
# -------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True, help="Dataset con X_train/val/test.npz")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--N", type=int, default=8192)
    ap.add_argument("--cap_bg", type=float, default=0.1)
    ap.add_argument("--use_augmentation", action="store_true")
    ap.add_argument("--augment_times", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    np.random.seed(args.seed)

    out = Path(args.out_dir)
    (out / "artifacts").mkdir(parents=True, exist_ok=True)

    splits = {}
    all_labels = []
    for s in ["train", "val", "test"]:
        Xp = Path(args.dataset_dir) / f"X_{s}.npz"
        Yp = Path(args.dataset_dir) / f"Y_{s}.npz"
        if Xp.exists() and Yp.exists():
            X = np.load(Xp)["X"]
            Y = np.load(Yp)["Y"]
            splits[s] = (X, Y)
            all_labels.append(Y)
            print(f"[OK] {s}: {X.shape}")

    # Remapeo global
    all_unique = np.unique(np.concatenate(all_labels))
    id2idx = {int(v): i for i, v in enumerate(sorted(all_unique))}
    idx2id = {i: int(v) for i, v in enumerate(sorted(all_unique))}
    print(f"[MAP] Global labels: {len(id2idx)} clases")

    # Procesamiento
    for split, (X, Y) in splits.items():
        X_out, Y_out = [], []
        for i in tqdm(range(X.shape[0]), desc=f"Procesando {split}"):
            pts = normalize_cloud(X[i])
            lbl = np.vectorize(id2idx.get)(Y[i])
            if args.use_augmentation and split == "train":
                for _ in range(args.augment_times):
                    aug = augment_cloud(pts)
                    aug, lbl_aug = cap_background(aug, lbl, args.N, cap_frac=args.cap_bg)
                    aug = farthest_point_sampling(aug, args.N)
                    X_out.append(aug)
                    Y_out.append(lbl_aug)
            pts, lbl = cap_background(pts, lbl, args.N, args.cap_bg)
            pts = farthest_point_sampling(pts, args.N)
            X_out.append(pts)
            Y_out.append(lbl)
        np.savez_compressed(out / f"X_{split}.npz", X=np.stack(X_out))
        np.savez_compressed(out / f"Y_{split}.npz", Y=np.stack(Y_out))
        print(f"[SAVE] {split}: {len(X_out)} muestras")

    # Pesos balanceados
    y_train = np.concatenate([y.ravel() for y in Y_out])
    uniques, counts = np.unique(y_train, return_counts=True)
    weights = {str(int(u)): float(np.sum(counts) / (len(uniques) * c)) for u, c in zip(uniques, counts)}
    json.dump(weights, open(out / "artifacts/class_weights.json", "w"), indent=2)

    # Guardar metadata
    json.dump({"id2idx": id2idx, "idx2id": idx2id}, open(out / "artifacts/label_map.json", "w"), indent=2)
    print(f"[DONE] Dataset guardado en {out}")

if __name__ == "__main__":
    main()
