#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
augmentation_and_split_v4.py
--------------------------------------------------------------
Aumentación y generación de splits estratificados para nubes dentales FPS.

Compatible con dataset extendido (XYZ o XYZ+normales+curvatura).
Solo aplica transformaciones geométricas a las coordenadas (x,y,z).

Basado en el flujo paper-like (PointNet++) y adaptado a v4.
--------------------------------------------------------------
Autor: Adaptado por ChatGPT (GPT-5)
Para la tesis de H. Taucare (Universidad de Chile)
--------------------------------------------------------------
"""

import argparse
import json
import numpy as np
import random
import gc
from pathlib import Path
from tqdm import tqdm


# ============================================================
# === Utilidades generales ===================================
# ============================================================

def seed_all(seed: int = 42):
    """Fija semilla global para reproducibilidad."""
    np.random.seed(seed)
    random.seed(seed)


def jitter_points(X, sigma=0.01, clip=0.05):
    """Añade ruido gaussiano pequeño a XYZ."""
    noise = np.clip(sigma * np.random.randn(*X[:, :3].shape), -clip, clip)
    X[:, :3] += noise
    return X


def random_scale(X, scale_min=0.95, scale_max=1.05):
    """Escalado aleatorio uniforme en XYZ."""
    s = np.random.uniform(scale_min, scale_max)
    X[:, :3] *= s
    return X


def random_rotation_z(X):
    """Rotación aleatoria alrededor del eje Z (coherente para dentición horizontal)."""
    theta = np.random.uniform(0, 2 * np.pi)
    rot = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ], dtype=np.float32)
    X[:, :3] = X[:, :3] @ rot.T
    return X


def random_dropout(X, Y, max_dropout_ratio=0.1):
    """Elimina aleatoriamente puntos (simula escaneo parcial)."""
    dropout_ratio = np.random.random() * max_dropout_ratio
    keep_idx = np.random.rand(X.shape[0]) >= dropout_ratio
    if np.sum(keep_idx) < 10:  # evita eliminar todo
        return X, Y
    return X[keep_idx], Y[keep_idx]


def normalize_xyz_inplace(X: np.ndarray) -> np.ndarray:
    """Re-normaliza XYZ a esfera unitaria después de augmentación."""
    coords = X[:, :3]
    coords -= coords.mean(axis=0, keepdims=True)
    r = np.linalg.norm(coords, axis=1).max()
    if r > 0:
        coords /= r
    X[:, :3] = coords
    return X


# ============================================================
# === Aumentación =============================================
# ============================================================

def augment_cloud(X: np.ndarray, Y: np.ndarray,
                  scale_min=0.95, scale_max=1.05,
                  jitter_sigma=0.01, jitter_clip=0.05,
                  dropout_rate=0.0):
    """Aplica pipeline de augmentación sobre XYZ y normaliza."""
    X = random_scale(X, scale_min, scale_max)
    X = random_rotation_z(X)
    X = jitter_points(X, jitter_sigma, jitter_clip)
    if dropout_rate > 0:
        X, Y = random_dropout(X, Y, dropout_rate)
    X = normalize_xyz_inplace(X)
    return X, Y


# ============================================================
# === MAIN ====================================================
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Aumentación y split estratificado (v4, compatible con features geométricas)."
    )
    parser.add_argument("--in_root", required=True,
                        help="Ruta de entrada merged_xyz_v4 (con X_*.npz, Y_*.npz).")
    parser.add_argument("--out_dir", required=True,
                        help="Ruta de salida para los splits augmentados.")
    parser.add_argument("--scale_min", type=float, default=0.95,
                        help="Escalado mínimo para augmentación (default=0.95).")
    parser.add_argument("--scale_max", type=float, default=1.05,
                        help="Escalado máximo para augmentación (default=1.05).")
    parser.add_argument("--jitter_sigma", type=float, default=0.01,
                        help="Desviación estándar del ruido gaussiano (default=0.01).")
    parser.add_argument("--jitter_clip", type=float, default=0.05,
                        help="Límite del ruido gaussiano (default=0.05).")
    parser.add_argument("--dropout_rate", type=float, default=0.0,
                        help="Proporción de puntos eliminados aleatoriamente (default=0.0).")
    parser.add_argument("--ensure_coverage", action="store_true",
                        help="Asegura cobertura espacial re-normalizando cada nube.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla RNG para reproducibilidad.")
    args = parser.parse_args()

    # --------------------------------------------------------
    # Inicialización
    # --------------------------------------------------------
    seed_all(args.seed)
    in_root = Path(args.in_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Carga de splits
    # --------------------------------------------------------
    X_train = np.load(in_root / "X_train.npz")["X"]
    Y_train = np.load(in_root / "Y_train.npz")["Y"]
    X_val = np.load(in_root / "X_val.npz")["X"]
    Y_val = np.load(in_root / "Y_val.npz")["Y"]
    X_test = np.load(in_root / "X_test.npz")["X"]
    Y_test = np.load(in_root / "Y_test.npz")["Y"]

    print(f"[OK] train: {X_train.shape}")
    print(f"[OK] val: {X_val.shape}")
    print(f"[OK] test: {X_test.shape}")

    # Detectar cantidad de canales (3 = XYZ, 7 = XYZ+normales+curvatura)
    n_channels = X_train.shape[-1]
    print(f"[INFO] Detectados {n_channels} canales por punto.")

    # --------------------------------------------------------
    # Aumentación de entrenamiento
    # --------------------------------------------------------
    aug_X, aug_Y = [], []

    for i in tqdm(range(X_train.shape[0]), desc="[AUG] train"):
        Xi = X_train[i].copy()
        Yi = Y_train[i].copy()
        X_aug, Y_aug = augment_cloud(
            Xi, Yi,
            scale_min=args.scale_min,
            scale_max=args.scale_max,
            jitter_sigma=args.jitter_sigma,
            jitter_clip=args.jitter_clip,
            dropout_rate=args.dropout_rate
        )
        if args.ensure_coverage:
            X_aug = normalize_xyz_inplace(X_aug)
        # Alinear dimensión (si hubo dropout)
        n_pad = X_train.shape[1] - X_aug.shape[0]
        if n_pad > 0:
            pad_X = np.zeros((n_pad, n_channels), np.float32)
            pad_Y = np.full((n_pad,), -1, np.int32)
            X_aug = np.concatenate([X_aug, pad_X], axis=0)
            Y_aug = np.concatenate([Y_aug, pad_Y], axis=0)
        elif n_pad < 0:
            X_aug = X_aug[:X_train.shape[1]]
            Y_aug = Y_aug[:Y_train.shape[1]]
        aug_X.append(X_aug)
        aug_Y.append(Y_aug)

    aug_X = np.stack(aug_X, axis=0)
    aug_Y = np.stack(aug_Y, axis=0)
    print(f"[AUG] train: +{aug_X.shape[0]} muestras (x1)")

    # --------------------------------------------------------
    # Combinar datos originales y aumentados
    # --------------------------------------------------------
    X_train_final = np.concatenate([X_train, aug_X], axis=0)
    Y_train_final = np.concatenate([Y_train, aug_Y], axis=0)

    print(f"[FINAL] train: {X_train_final.shape}")
    print(f"[FINAL] val:   {X_val.shape}")
    print(f"[FINAL] test:  {X_test.shape}")

    # --------------------------------------------------------
    # Guardar splits
    # --------------------------------------------------------
    np.savez_compressed(out_dir / "X_train.npz", X=X_train_final)
    np.savez_compressed(out_dir / "Y_train.npz", Y=Y_train_final)
    np.savez_compressed(out_dir / "X_val.npz", X=X_val)
    np.savez_compressed(out_dir / "Y_val.npz", Y=Y_val)
    np.savez_compressed(out_dir / "X_test.npz", X=X_test)
    np.savez_compressed(out_dir / "Y_test.npz", Y=Y_test)

    # --------------------------------------------------------
    # EDA (validación rápida de cobertura)
    # --------------------------------------------------------
    def summary(Y, split_name):
        u, c = np.unique(Y[Y >= 0], return_counts=True)
        bg_ratio = np.mean(Y == -1)
        print(f"  {split_name}: min={u.min()} max={u.max()} unique={len(u)} bg={bg_ratio:.2%}")

    print("\n[EDA] Validación por split:")
    summary(Y_train_final, "train")
    summary(Y_val, "val")
    summary(Y_test, "test")

    print(f"\n[DONE] Splits listos en: {out_dir}")


# ============================================================
# === Ejecución directa ======================================
# ============================================================

if __name__ == "__main__":
    main()
