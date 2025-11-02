#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_teeth_upper_arcadas.py
--------------------------------------------------------------
Genera mini-arcadas artificiales de la arcada superior
para entrenar modelos de segmentación por diente (PointNet++, etc.).

✔ Balancea los puntos por clase (cada diente ≈ misma cantidad de puntos)
✔ Diente 21 (ID=1) recibe 1.5× más puntos
✔ Evita repeticiones exactas con oversampling aleatorio controlado
✔ Guarda splits X_train_upper.npz / Y_train_upper.npz / etc.

Autor: Adaptado por ChatGPT (GPT-5)
--------------------------------------------------------------
"""

import numpy as np
import random, json
from pathlib import Path
from tqdm import tqdm

# ==============================================================
# === Configuración ============================================
# ==============================================================

INPUT_ROOT = Path("/home/htaucare/Tesis_Amaro/data/Teeth_3ds/fixed_split/8192/fps_aug_paperlike_v3")
OUT_ROOT = Path("/home/htaucare/Tesis_Amaro/data/Teeth_3ds/fixed_split/8192/merged_upper_arcadas_v1")

TARGET_TOOTH = 1       # ID del diente 21
N_POINTS = 8192        # puntos por nube
D21_BOOST = 1.5        # 1.5× más puntos al 21
OVERSAMPLE_RATIO = 0.2 # fuerza a incluir el 21 en el 20% de las arcadas
UPPER_IDS = np.arange(1, 13)  # IDs 1-12 = arcada superior


# ==============================================================
# === Funciones auxiliares =====================================
# ==============================================================

def normalize_cloud(x: np.ndarray) -> np.ndarray:
    """Normaliza la nube a esfera unitaria (centro=0, radio=1)."""
    x = x - x.mean(0, keepdims=True)
    r = np.sqrt((x ** 2).sum(1)).max()
    return x / (r + 1e-8)


def _class_quotas(ids_present, N_total=8192, d21_id=1, d21_boost=1.5):
    """Cuotas de puntos por clase con refuerzo al diente 21."""
    weights = []
    for cid in ids_present:
        w = d21_boost if cid == d21_id else 1.0
        weights.append(w)
    W = float(sum(weights))
    quotas = [max(1, int(round(N_total * w / W))) for w in weights]
    diff = N_total - sum(quotas)
    if diff != 0:
        order = np.argsort(-np.array(weights))
        i = 0
        while diff != 0:
            k = order[i % len(order)]
            quotas[k] += 1 if diff > 0 else -1
            diff += -1 if diff > 0 else 1
            i += 1
    return dict(zip(list(ids_present), quotas))


def combine_teeth_balanced(X_all, Y_all, ids_sel, N_points=8192, d21_id=1, d21_boost=1.5):
    """Combina dientes específicos en una mini-arcada con balance de puntos."""
    mask = np.isin(Y_all, ids_sel)
    X = X_all[mask]
    Y = Y_all[mask]
    if X.size == 0:
        idx = np.random.choice(X_all.shape[0], N_points, replace=True)
        return normalize_cloud(X_all[idx]), Y_all[idx]

    ids_present = np.unique(Y)
    quotas = _class_quotas(ids_present, N_total=N_points, d21_id=d21_id, d21_boost=d21_boost)
    picked_idx = []
    for cid in ids_present:
        cls_idx = np.where(Y == cid)[0]
        q = quotas.get(int(cid), 0)
        if cls_idx.size >= q:
            sel = np.random.choice(cls_idx, q, replace=False)
        else:
            sel = np.random.choice(cls_idx, q, replace=True)
        picked_idx.append(sel)
    picked_idx = np.concatenate(picked_idx) if len(picked_idx) else np.array([], dtype=int)
    rng = np.random.permutation(picked_idx.size)
    picked_idx = picked_idx[rng]
    Xb, Yb = X[picked_idx], Y[picked_idx]
    return normalize_cloud(Xb), Yb


# ==============================================================
# === Pipeline principal =======================================
# ==============================================================

def process_split(split_name: str):
    """Genera mini-arcadas superiores para un split (train/val/test)."""
    print(f"\n=== Procesando split {split_name} ===")

    X_path = INPUT_ROOT / f"X_{split_name}.npz"
    Y_path = INPUT_ROOT / f"Y_{split_name}.npz"
    if not X_path.exists() or not Y_path.exists():
        print(f"[WARN] Faltan archivos para {split_name}")
        return

    X_all = np.load(X_path)["X"]
    Y_all = np.load(Y_path)["Y"]

    X_new, Y_new = [], []
    n_samples = X_all.shape[0]

    for i in tqdm(range(n_samples), desc=f"Generando {split_name}"):
        Xp, Yp = X_all[i], Y_all[i]
        ids_present = np.intersect1d(np.unique(Yp), UPPER_IDS)
        if ids_present.size < 2:
            continue

        # 20% de las arcadas forzadas a incluir el 21
        if random.random() < OVERSAMPLE_RATIO and TARGET_TOOTH not in ids_present:
            ids_present = np.append(ids_present, TARGET_TOOTH)
            print("[FORCED] Incluyendo diente 21 (ID 1) en esta arcada.")

        Xc, Yc = combine_teeth_balanced(
            Xp, Yp, ids_present,
            N_points=N_POINTS,
            d21_id=TARGET_TOOTH,
            d21_boost=D21_BOOST
        )
        X_new.append(Xc)
        Y_new.append(Yc)

    if not X_new:
        print(f"[ERROR] No se generaron muestras para {split_name}.")
        return

    X_new, Y_new = np.stack(X_new), np.stack(Y_new)
    np.savez_compressed(OUT_ROOT / f"X_{split_name}_upper.npz", X=X_new)
    np.savez_compressed(OUT_ROOT / f"Y_{split_name}_upper.npz", Y=Y_new)
    print(f"[OK] Guardado: {OUT_ROOT}/X_{split_name}_upper.npz  ({len(X_new)} muestras)")
    print(f"[DONE] Split {split_name} procesado ✅")


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        process_split(split)
    print("\n✅ Mini-arcadas superiores generadas correctamente en:")
    print(f"   {OUT_ROOT}")


if __name__ == "__main__":
    main()
