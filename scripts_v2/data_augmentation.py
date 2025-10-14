#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aplicación de data augmentation a nubes de puntos 3D dentales.
- Rotación aleatoria en Z
- Jitter (ruido Gaussiano)
- Escalado global
- Dropout de puntos
Genera carpeta data_augmentation/<n_points>/upper/<case>/ y lower/...
"""

import os
import numpy as np
import json
from pathlib import Path

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

def dropout_points(points, drop_rate=0.05):
    keep_mask = np.random.rand(points.shape[0]) > drop_rate
    return points[keep_mask] if keep_mask.any() else points

def augment(points,
            rotate_deg=15,
            jitter_sigma=0.005, jitter_clip=0.02,
            scale_min=0.95, scale_max=1.05,
            dropout_rate=0.05):
    x = rotate_z(points, max_deg=rotate_deg)
    x = jitter(x, sigma=jitter_sigma, clip=jitter_clip)
    x = scale(x, min_s=scale_min, max_s=scale_max)
    x = dropout_points(x, drop_rate=dropout_rate)
    return normalize_cloud_np(x)

def augment_case(src_case_dir, dst_case_dir, augment_times=1):
    src_cloud = np.load(src_case_dir / "point_cloud.npy")
    labels = np.load(src_case_dir / "labels.npy") if (src_case_dir / "labels.npy").exists() else None
    inst = np.load(src_case_dir / "instances.npy") if (src_case_dir / "instances.npy").exists() else None
    meta = json.load(open(src_case_dir / "meta.json")) if (src_case_dir / "meta.json").exists() else {}
    dst_case_dir.mkdir(parents=True, exist_ok=True)
    for aidx in range(augment_times):
        aug_cloud = augment(src_cloud)
        np.save(dst_case_dir / f"point_cloud_aug{aidx}.npy", aug_cloud)
        if labels is not None:
            np.save(dst_case_dir / f"labels_aug{aidx}.npy", labels[:aug_cloud.shape[0]])
        if inst is not None:
            np.save(dst_case_dir / f"instances_aug{aidx}.npy", inst[:aug_cloud.shape[0]])
    meta["augmented"] = True
    json.dump(meta, open(dst_case_dir / "meta.json", "w"), indent=2)

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--src_root", required=True, help="Folder raíz, ej: processed_struct/8192")
    p.add_argument("--out_root", required=True, help="Folder salida, ej: data_augmentation/8192")
    p.add_argument("--augment_times", type=int, default=1, help="Veces a aplicar augmentation")
    args = p.parse_args()

    src_root = Path(args.src_root)
    out_root = Path(args.out_root)

    for jaw in ["upper", "lower"]:
        in_jaw = src_root / jaw
        out_jaw = out_root / jaw
        if not in_jaw.exists():
            continue
        for case in sorted(in_jaw.iterdir()):
            if not case.is_dir():
                continue
            out_case = out_jaw / case.name
            augment_case(case, out_case, augment_times=args.augment_times)
            print(f"[OK] Augmentado: {case}")

if __name__ == "__main__":
    main()
