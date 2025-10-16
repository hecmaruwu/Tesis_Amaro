#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unifica augmentación + split + remapeo GLOBAL + fixed N + control de fondo.

Compatible con merged_8192 → fixed_split_aug → train_models_v2_patience.py
Asegura que todas las clases estén remapeadas 0..C-1 globalmente.
"""

import os
import json
import argparse
from pathlib import Path
import numpy as np

# ===========================
# --- Augmentation Utils ---
# ===========================

def normalize_cloud_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mean = x.mean(axis=0, keepdims=True)
    x = x - mean
    r = np.linalg.norm(x, axis=1, keepdims=True).max()
    return x / (r + 1e-6)

def rotate_z(points, max_deg=15.0):
    theta = np.random.uniform(-max_deg, max_deg) * (np.pi / 180.0)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0.0],
                  [s,  c, 0.0],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    return points.dot(R)

def jitter(points, sigma=0.005, clip=0.02):
    noise = np.clip(np.random.normal(0, sigma, points.shape), -clip, clip)
    return points + noise

def scale(points, min_s=0.95, max_s=1.05):
    s = np.random.uniform(min_s, max_s)
    return points * s

def dropout_points(points, drop_rate=0.05, min_keep=32):
    keep_mask = np.random.rand(points.shape[0]) > drop_rate
    if not keep_mask.any() or keep_mask.sum() < min_keep:
        keep_mask[np.random.choice(points.shape[0], min_keep, replace=False)] = True
    return points[keep_mask]

def augment(points,
            rotate_deg=15,
            jitter_sigma=0.005, jitter_clip=0.02,
            scale_min=0.95, scale_max=1.05,
            dropout_rate=0.05,
            n_target=None,
            rng=None):
    """Aplica augmentación y re-muestrea a n_target si se especifica"""
    x = rotate_z(points, max_deg=rotate_deg)
    x = jitter(x, sigma=jitter_sigma, clip=jitter_clip)
    x = scale(x, min_s=scale_min, max_s=scale_max)
    x = dropout_points(x, drop_rate=dropout_rate)
    x = normalize_cloud_np(x)
    if n_target is not None and x.shape[0] != n_target:
        rng = rng or np.random.default_rng()
        idx = rng.choice(x.shape[0], n_target, replace=(x.shape[0] < n_target))
        x = x[idx]
    return x

# ===========================
# --- Label and Sampling ---
# ===========================

def remap_labels_global(Y_all):
    """Crea un mapeo global de todas las etiquetas 0..C-1"""
    vals = sorted(set(np.unique(np.concatenate(Y_all))))
    id2idx = {int(v): i for i, v in enumerate(vals)}
    idx2id = {i: int(v) for i, v in enumerate(vals)}
    Y_remapped = [np.vectorize(id2idx.__getitem__)(Y) for Y in Y_all]
    return Y_remapped, id2idx, idx2id

# ===========================
# --- Main Pipeline ---
# ===========================

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
            idx = np.concatenate([sel_bg, sel_fg]).astype(np.int64)
        else:
            idx = rng.choice(len(pts), N, replace=(len(pts) < N)).astype(np.int64)

        X_proc[i] = pts[idx]
        Y_proc[i] = lbs[idx]

    return X_proc, Y_proc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--N", type=int, default=8192)
    ap.add_argument("--cap_bg", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_augmentation", action="store_true")
    ap.add_argument("--augment_times", type=int, default=1)
    ap.add_argument("--rotate_deg", type=float, default=15)
    ap.add_argument("--jitter_sigma", type=float, default=0.005)
    ap.add_argument("--scale_min", type=float, default=0.95)
    ap.add_argument("--scale_max", type=float, default=1.05)
    ap.add_argument("--dropout_rate", type=float, default=0.05)
    ap.add_argument("--merge_aug", action="store_true")
    ap.add_argument("--save_aug_separate", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "artifacts").mkdir(exist_ok=True)

    # --- Load original splits ---
    splits = {}
    all_Y_for_remap = []
    for s in ["train", "val", "test"]:
        Xp = Path(args.dataset_dir) / f"X_{s}.npz"
        Yp = Path(args.dataset_dir) / f"Y_{s}.npz"
        if Xp.exists() and Yp.exists():
            X = np.load(Xp)["X"]
            Y = np.load(Yp)["Y"]
            all_Y_for_remap.append(Y)
            splits[s] = (X, Y)
            print(f"[OK] Loaded split {s}: {X.shape}")

    # --- Global remap ---
    Y_remapped, id2idx, idx2id = remap_labels_global([Y for _, Y in splits.values()])
    for i, key in enumerate(["train", "val", "test"]):
        X, _ = splits[key]
        splits[key] = (X, Y_remapped[i])
    print(f"[MAP] Global label remap: {len(id2idx)} clases totales")

    # --- Apply subsampling & cap background ---
    for s, (X, Y) in splits.items():
        X, Y = process_split(X, Y, args.N, args.cap_bg, args.seed)
        splits[s] = (X, Y)

    # --- Augment only train ---
    if args.use_augmentation and "train" in splits:
        Xo, Yo = splits["train"]
        aug_X_list, aug_Y_list = [], []
        for i in range(Xo.shape[0]):
            for _ in range(args.augment_times):
                X_aug = augment(
                    Xo[i],
                    rotate_deg=args.rotate_deg,
                    jitter_sigma=args.jitter_sigma,
                    scale_min=args.scale_min,
                    scale_max=args.scale_max,
                    dropout_rate=args.dropout_rate,
                    n_target=args.N,
                    rng=rng
                )
                aug_X_list.append(X_aug)
                aug_Y_list.append(Yo[i])
        aug_X = np.stack(aug_X_list, axis=0).astype(np.float32)
        aug_Y = np.stack(aug_Y_list, axis=0).astype(np.int32)
        print(f"[AUG] Generated {aug_X.shape[0]} augmented samples ({args.N} pts each).")

        if args.merge_aug:
            splits["train"] = (np.concatenate([Xo, aug_X], axis=0),
                               np.concatenate([Yo, aug_Y], axis=0))
        elif args.save_aug_separate:
            np.savez_compressed(out / "X_train_aug.npz", X=aug_X)
            np.savez_compressed(out / "Y_train_aug.npz", Y=aug_Y)
            print(f"[SEP] Saved separate augmented split.")

    # --- Save outputs ---
    for s, (X, Y) in splits.items():
        np.savez_compressed(out / f"X_{s}.npz", X=X)
        np.savez_compressed(out / f"Y_{s}.npz", Y=Y)
        print(f"[SAVE] {s}: {X.shape}")

    # --- Save metadata ---
    meta = {
        "N": args.N,
        "use_augmentation": args.use_augmentation,
        "augment_times": args.augment_times,
        "merge_aug": args.merge_aug,
        "save_aug_separate": args.save_aug_separate,
        "label_map": {"id2idx": id2idx, "idx2id": idx2id},
    }
    json.dump(meta, open(out / "artifacts" / "meta.json", "w"), indent=2)
    json.dump({"id2idx": id2idx, "idx2id": idx2id}, open(out / "artifacts" / "label_map.json", "w"), indent=2)

    # --- Validación global ---
    for s, (_, Y) in splits.items():
        y_min, y_max = Y.min(), Y.max()
        print(f"[CHECK] {s}: min={y_min}, max={y_max}, unique={len(np.unique(Y))}")
        assert y_min == 0, f"{s} contiene etiquetas negativas"
        assert y_max < len(id2idx), f"{s} tiene etiquetas fuera de rango ({y_max} >= {len(id2idx)})"

    print(f"[DONE] All splits saved to {out}")

if __name__ == "__main__":
    main()
