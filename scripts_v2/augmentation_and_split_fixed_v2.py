#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
augmentation_and_split_fixed_v2.py
----------------------------------
Versión robusta con:
 - cobertura total de clases entre splits
 - augmentación geométrica avanzada
 - reporte EDA automático

Usar después de generar fps_aug_fixed_v4.
"""

import os, json, argparse
from pathlib import Path
import numpy as np

# =======================================================
# --------- Utilidades de augmentación y normalización --
# =======================================================

def normalize_cloud_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mean = x.mean(axis=0, keepdims=True)
    x = x - mean
    r = np.linalg.norm(x, axis=1, keepdims=True).max()
    return x / (r + 1e-6)

def rotate_z(points, max_deg=45.0):
    theta = np.random.uniform(-max_deg, max_deg) * (np.pi / 180.0)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0.0],
                  [s,  c, 0.0],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    return points.dot(R)

def jitter(points, sigma=0.01, clip=0.05):
    noise = np.clip(np.random.normal(0, sigma, points.shape), -clip, clip)
    return points + noise

def scale(points, min_s=0.9, max_s=1.1):
    s = np.random.uniform(min_s, max_s)
    return points * s

def translate(points, max_shift=0.05):
    shift = np.random.uniform(-max_shift, max_shift, 3)
    return points + shift

def augment(points, rotate_deg=45, jitter_sigma=0.01,
            scale_min=0.9, scale_max=1.1, translate_shift=0.05,
            dropout_rate=0.05, n_target=None, rng=None):
    """Augmentación geométrica + remuestreo"""
    x = rotate_z(points, max_deg=rotate_deg)
    x = jitter(x, sigma=jitter_sigma)
    x = scale(x, min_s=scale_min, max_s=scale_max)
    x = translate(x, max_shift=translate_shift)
    x = normalize_cloud_np(x)
    if n_target is not None and x.shape[0] != n_target:
        rng = rng or np.random.default_rng()
        idx = rng.choice(x.shape[0], n_target, replace=(x.shape[0] < n_target))
        x = x[idx]
    return x

# =======================================================
# ------------------ Procesamiento base -----------------
# =======================================================

def process_split(X, Y, N, cap_bg, seed):
    rng = np.random.default_rng(seed)
    X_proc = np.empty((X.shape[0], N, 3), dtype=np.float32)
    Y_proc = np.empty((X.shape[0], N), dtype=np.int32)

    for i in range(X.shape[0]):
        pts, lbs = X[i], Y[i]
        if len(pts) == 0:
            continue

        if cap_bg is not None:
            mask_bg = (lbs == 0)
            idx_bg = np.where(mask_bg)[0]
            idx_fg = np.where(~mask_bg)[0]
            n_bg = int(min(len(idx_bg), N * cap_bg))
            n_fg = max(0, N - n_bg)
            sel_bg = rng.choice(idx_bg, size=n_bg, replace=(len(idx_bg) < n_bg)) if len(idx_bg) > 0 else np.empty((0,), np.int64)
            sel_fg = rng.choice(idx_fg, size=n_fg, replace=(len(idx_fg) < n_fg)) if len(idx_fg) > 0 else np.empty((0,), np.int64)
            idx = np.concatenate([sel_bg, sel_fg])
        else:
            idx = rng.choice(len(pts), N, replace=(len(pts) < N))
        X_proc[i] = pts[idx]
        Y_proc[i] = lbs[idx]

    return X_proc, Y_proc

# =======================================================
# -------------------- Main Pipeline --------------------
# =======================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--N", type=int, default=8192)
    ap.add_argument("--cap_bg", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_augmentation", action="store_true")
    ap.add_argument("--augment_times", type=int, default=1)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "artifacts").mkdir(exist_ok=True)

    # --- Load splits ---
    splits = {}
    for s in ["train", "val", "test"]:
        Xp = Path(args.dataset_dir) / f"X_{s}.npz"
        Yp = Path(args.dataset_dir) / f"Y_{s}.npz"
        X = np.load(Xp)["X"]; Y = np.load(Yp)["Y"]
        splits[s] = (X, Y)
        print(f"[OK] Loaded {s}: {X.shape}")

    # --- Procesar y remuestrear ---
    for s, (X, Y) in splits.items():
        X_proc, Y_proc = process_split(X, Y, args.N, args.cap_bg, args.seed)
        splits[s] = (X_proc, Y_proc)

    # --- Aumentar train si se pide ---
    if args.use_augmentation:
        Xo, Yo = splits["train"]
        aug_X, aug_Y = [], []
        for i in range(Xo.shape[0]):
            for _ in range(args.augment_times):
                Xa = augment(Xo[i], n_target=args.N, rng=rng)
                aug_X.append(Xa)
                aug_Y.append(Yo[i])
        aug_X = np.stack(aug_X)
        aug_Y = np.stack(aug_Y)
        splits["train"] = (np.concatenate([Xo, aug_X]), np.concatenate([Yo, aug_Y]))
        print(f"[AUG] Added {aug_X.shape[0]} augmented samples")

    # ===================================================
    # --------- Garantizar cobertura total -------------
    # ===================================================
    all_labels = set(np.unique(np.concatenate([Y for _, Y in splits.values()])))
    missing_per_split = {}
    for s in ["val", "test"]:
        present = set(np.unique(splits[s][1]))
        missing = sorted(list(all_labels - present))
        missing_per_split[s] = missing
        for m in missing:
            # buscar una muestra con esa clase en train
            Xtr, Ytr = splits["train"]
            idx = np.where(np.any(Ytr == m, axis=1))[0]
            if len(idx) > 0:
                sel = idx[0]
                Xadd, Yadd = Xtr[sel][None, ...], Ytr[sel][None, ...]
                splits[s] = (np.concatenate([splits[s][0], Xadd]),
                             np.concatenate([splits[s][1], Yadd]))
        if missing:
            print(f"[FIX] {s}: se añadieron muestras para clases faltantes {missing}")
        else:
            print(f"[OK] {s}: todas las clases presentes")

    # --- Guardar ---
    for s, (X, Y) in splits.items():
        np.savez_compressed(out / f"X_{s}.npz", X=X)
        np.savez_compressed(out / f"Y_{s}.npz", Y=Y)
        print(f"[SAVE] {s}: {X.shape}")

    # --- EDA ---
    print("\n[EDA] Cobertura final:")
    for s, (_, Y) in splits.items():
        print(f"  {s}: {len(np.unique(Y))} clases, min={Y.min()} max={Y.max()}")

    print(f"\n[DONE] Splits guardados en {out}")

if __name__ == "__main__":
    main()
