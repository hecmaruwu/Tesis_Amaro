#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Augmentación y balanceo de nubes de puntos binarias (UFRN pseudo-labels)
- Rotaciones aleatorias
- Escalado
- Jitter (ruido gaussiano)
- Réplicas para balancear clases (21 vs resto)

Salida:
  /data/UFRN/processed_augmented/8192/upper/paciente_X_augY/{X.npy, Y.npy}
"""

import numpy as np
from pathlib import Path
import random, json, shutil
from tqdm import tqdm

def rotate_points(points):
    """Rotación aleatoria en torno al eje Z."""
    theta = random.uniform(0, 2*np.pi)
    rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta),  np.cos(theta), 0],
                    [0, 0, 1]], dtype=np.float32)
    return points @ rot.T

def jitter_points(points, sigma=0.005, clip=0.02):
    """Ruido gaussiano pequeño."""
    jitter = np.clip(sigma * np.random.randn(*points.shape), -clip, clip)
    return points + jitter

def scale_points(points, scale_range=(0.9, 1.1)):
    """Escalado aleatorio."""
    s = np.random.uniform(scale_range[0], scale_range[1])
    return points * s

def augment_sample(X, Y):
    X = rotate_points(X)
    X = scale_points(X)
    X = jitter_points(X)
    return X, Y.copy()

def main():
    root_in = Path("/home/htaucare/Tesis_Amaro/data/UFRN/processed_pseudolabels_targets/8192/upper")
    root_out = Path("/home/htaucare/Tesis_Amaro/data/UFRN/processed_augmented/8192/upper")
    root_out.mkdir(parents=True, exist_ok=True)

    samples = sorted(root_in.glob("paciente_*/X.npy"))
    if not samples:
        print("[ERR] No se encontraron nubes fuente."); return

    for sp in tqdm(samples, desc="[AUG]"):
        pid = sp.parent.name
        X = np.load(sp)
        Y = np.load(sp.with_name("Y.npy"))
        meta = json.loads((sp.parent / "meta.json").read_text())

        # Copia original
        dst = root_out / pid
        dst.mkdir(parents=True, exist_ok=True)
        np.save(dst/"X.npy", X)
        np.save(dst/"Y.npy", Y)
        shutil.copy(sp.parent / "meta.json", dst/"meta.json")

        # Balance: más ejemplos donde Y tiene mayor proporción de 1
        frac_21 = Y.mean()
        n_aug = 3 if frac_21 < 0.03 else 1  # 3x si tiene pocos puntos de diente
        for i in range(n_aug):
            X_aug, Y_aug = augment_sample(X, Y)
            dsta = root_out / f"{pid}_aug{i+1}"
            dsta.mkdir(exist_ok=True)
            np.save(dsta/"X.npy", X_aug)
            np.save(dsta/"Y.npy", Y_aug)
            (dsta/"meta.json").write_text(json.dumps({**meta, "aug_idx": i+1}, indent=2), encoding="utf-8")

    print(f"[DONE] Aumentación y balanceo listos en: {root_out}")

if __name__ == "__main__":
    main()
