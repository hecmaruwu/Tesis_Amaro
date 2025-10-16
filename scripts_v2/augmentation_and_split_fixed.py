#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
augmentation_and_split_fixed.py

Versión robusta y completa:
- Remapeo global unificado de etiquetas 0..C−1
- Garantiza cobertura total de clases en TRAIN
- Split estratificado balanceado (train/val/test)
- Cap de fondo configurable
- Augmentación opcional y parametrizable
- Validación + EDA automático al final
"""

import os
import json
import argparse
from pathlib import Path
import numpy as np

# ============================================================
# ------------------- FUNCIONES AUXILIARES -------------------
# ============================================================

def normalize_cloud_np(x: np.ndarray) -> np.ndarray:
    """Normaliza nube de puntos a esfera unitaria centrada en el origen."""
    x = x.astype(np.float32)
    mean = x.mean(axis=0, keepdims=True)
    x = x - mean
    r = np.linalg.norm(x, axis=1, keepdims=True).max()
    return x / (r + 1e-6)

def rotate_z(points, max_deg=15.0):
    """Rotación aleatoria sobre el eje Z."""
    theta = np.random.uniform(-max_deg, max_deg) * (np.pi / 180.0)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0.0],
                  [s,  c, 0.0],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    return points.dot(R)

def jitter(points, sigma=0.005, clip=0.02):
    """Ruido gaussiano pequeño (perturbación local)."""
    noise = np.clip(np.random.normal(0, sigma, points.shape), -clip, clip)
    return points + noise

def scale(points, min_s=0.95, max_s=1.05):
    """Escala uniforme aleatoria."""
    s = np.random.uniform(min_s, max_s)
    return points * s

def dropout_points(points, drop_rate=0.05, min_keep=32):
    """Simula pérdida parcial de puntos (oclusiones)."""
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
    """Aplica augmentaciones geométricas básicas y remuestrea si es necesario."""
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

def cap_background(X, Y, N, cap_bg=0.4, seed=42):
    """Limita el % de puntos de fondo (label=0) por nube."""
    rng = np.random.default_rng(seed)
    X_out, Y_out = np.empty((X.shape[0], N, 3), dtype=np.float32), np.empty((X.shape[0], N), dtype=np.int32)
    for i in range(X.shape[0]):
        pts, lbs = X[i], Y[i]
        mask_bg = (lbs == 0)
        idx_bg, idx_fg = np.where(mask_bg)[0], np.where(~mask_bg)[0]
        n_bg = int(min(len(idx_bg), N * cap_bg))
        n_fg = max(0, N - n_bg)
        sel_bg = rng.choice(idx_bg, n_bg, replace=(len(idx_bg) < n_bg)) if len(idx_bg) else np.empty(0, np.int64)
        sel_fg = rng.choice(idx_fg, n_fg, replace=(len(idx_fg) < n_fg)) if len(idx_fg) else np.empty(0, np.int64)
        idx = np.concatenate([sel_bg, sel_fg]).astype(np.int64)
        X_out[i], Y_out[i] = pts[idx], lbs[idx]
    return X_out, Y_out

# ============================================================
# ------------------ REMAPEO Y SPLIT GLOBAL ------------------
# ============================================================

def remap_labels_global(Y_all):
    """
    Crea un mapeo global de todas las etiquetas únicas → 0..C-1.
    Devuelve las etiquetas remapeadas + diccionarios id2idx / idx2id.
    """
    vals = sorted(set(np.unique(np.concatenate(Y_all))))
    id2idx = {int(v): i for i, v in enumerate(vals)}
    idx2id = {i: int(v) for i, v in enumerate(vals)}
    Y_remapped = [np.vectorize(id2idx.__getitem__)(Y) for Y in Y_all]
    return Y_remapped, id2idx, idx2id


def ensure_all_classes_in_train(splits, seed=42):
    """
    Garantiza que todas las clases globales estén presentes en el split TRAIN.
    Si una clase existe solo en val/test, se mueve una muestra al train.
    """
    rng = np.random.default_rng(seed)
    all_classes = set(np.unique(np.concatenate([Y for _, Y in splits.values()])))
    train_classes = set(np.unique(splits["train"][1]))
    missing = sorted(list(all_classes - train_classes))
    if not missing:
        print("[CHECK] Train cubre todas las clases.")
        return splits

    print(f"[FIX] Clases ausentes en train: {missing}")
    moved = 0
    for cls in missing:
        # buscar una nube que contenga esa clase (de val o test)
        for split_name in ["val", "test"]:
            X, Y = splits[split_name]
            idx_candidates = [i for i in range(len(Y)) if cls in Y[i]]
            if idx_candidates:
                idx = rng.choice(idx_candidates)
                x_sel, y_sel = X[idx], Y[idx]
                # mover a train
                splits["train"] = (
                    np.concatenate([splits["train"][0], x_sel[None]], axis=0),
                    np.concatenate([splits["train"][1], y_sel[None]], axis=0)
                )
                # remover del split original
                X_new = np.delete(X, idx, axis=0)
                Y_new = np.delete(Y, idx, axis=0)
                splits[split_name] = (X_new, Y_new)
                moved += 1
                print(f"[MOVE] Clase {cls} movida de {split_name} → train")
                break
    print(f"[DONE] Se movieron {moved} muestras para cobertura total.")
    return splits


def stratified_split(X, Y, ratios=(0.8, 0.1, 0.1), seed=42):
    """
    Divide el dataset de forma estratificada por clases.
    ratios: (train, val, test)
    """
    rng = np.random.default_rng(seed)
    cls_to_idx = {}
    for i, y in enumerate(Y):
        for c in np.unique(y):
            cls_to_idx.setdefault(int(c), []).append(i)

    total = len(X)
    idx_train, idx_val, idx_test = set(), set(), set()

    for cls, idx_list in cls_to_idx.items():
        rng.shuffle(idx_list)
        n = len(idx_list)
        n_tr = max(1, int(n * ratios[0]))
        n_va = max(1, int(n * ratios[1]))
        train_ids = idx_list[:n_tr]
        val_ids = idx_list[n_tr:n_tr + n_va]
        test_ids = idx_list[n_tr + n_va:]
        idx_train.update(train_ids)
        idx_val.update(val_ids)
        idx_test.update(test_ids)

    def subset(idxs):
        ids = sorted(list(idxs))
        return X[ids], Y[ids]

    splits = {
        "train": subset(idx_train),
        "val": subset(idx_val),
        "test": subset(idx_test)
    }
    return splits

# ============================================================
# ---------------------- PIPELINE MAIN -----------------------
# ============================================================

def process_split(X, Y, N, cap_bg, seed):
    """Aplica muestreo con control de fondo opcional."""
    if cap_bg is not None:
        return cap_background(X, Y, N, cap_bg, seed)
    else:
        rng = np.random.default_rng(seed)
        X_proc = np.empty((X.shape[0], N, 3), dtype=np.float32)
        Y_proc = np.empty((X.shape[0], N), dtype=np.int32)
        for i in range(X.shape[0]):
            pts = X[i]
            idx = rng.choice(len(pts), N, replace=(len(pts) < N))
            X_proc[i] = pts[idx]
            Y_proc[i] = Y[i][idx]
        return X_proc, Y_proc


def main():
    ap = argparse.ArgumentParser(description="Genera splits balanceados con augmentación opcional.")
    ap.add_argument("--dataset_dir", required=True, help="Ruta al dataset combinado (X.npz/Y.npz o merged_8192).")
    ap.add_argument("--out_dir", required=True, help="Carpeta de salida para el split final.")
    ap.add_argument("--N", type=int, default=8192, help="Número de puntos por nube.")
    ap.add_argument("--cap_bg", type=float, default=None, help="Porcentaje máximo de puntos de fondo.")
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

    # --------------------------------------------------------
    # --- CARGA BASE UNIFICADA (permite merged o flat) -------
    # --------------------------------------------------------
    files = sorted(Path(args.dataset_dir).glob("X_*.npz"))
    if not files:
        raise FileNotFoundError(f"No se encontraron archivos X_*.npz en {args.dataset_dir}")

    splits = {}
    for s in ["train", "val", "test"]:
        Xp = Path(args.dataset_dir) / f"X_{s}.npz"
        Yp = Path(args.dataset_dir) / f"Y_{s}.npz"
        if not (Xp.exists() and Yp.exists()):
            continue
        X = np.load(Xp)["X"]
        Y = np.load(Yp)["Y"]
        splits[s] = (X, Y)
        print(f"[OK] Loaded split {s}: {X.shape}")

    # Si no existen splits, intenta generar uno estratificado
    if not splits:
        X_all = np.load(Path(args.dataset_dir) / "X.npz")["X"]
        Y_all = np.load(Path(args.dataset_dir) / "Y.npz")["Y"]
        splits = stratified_split(X_all, Y_all, seed=args.seed)
        print(f"[AUTO] Split estratificado generado automáticamente.")

    # --------------------------------------------------------
    # --- REMAPEO GLOBAL DE ETIQUETAS ------------------------
    # --------------------------------------------------------
    Y_remapped, id2idx, idx2id = remap_labels_global([Y for _, Y in splits.values()])
    for i, key in enumerate(splits.keys()):
        X, _ = splits[key]
        splits[key] = (X, Y_remapped[i])
    print(f"[MAP] Remapeo global: {len(id2idx)} clases.")

    # --------------------------------------------------------
    # --- GARANTIZAR COBERTURA TOTAL EN TRAIN ----------------
    # --------------------------------------------------------
    splits = ensure_all_classes_in_train(splits, seed=args.seed)

    # --------------------------------------------------------
    # --- PROCESAR (FPS-like / random sampling) ---------------
    # --------------------------------------------------------
    for s, (X, Y) in splits.items():
        X, Y = process_split(X, Y, args.N, args.cap_bg, args.seed)
        splits[s] = (X, Y)

    # --------------------------------------------------------
    # --- AUMENTACIÓN (solo TRAIN) ----------------------------
    # --------------------------------------------------------
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
            splits["train"] = (
                np.concatenate([Xo, aug_X], axis=0),
                np.concatenate([Yo, aug_Y], axis=0)
            )
        elif args.save_aug_separate:
            np.savez_compressed(out / "X_train_aug.npz", X=aug_X)
            np.savez_compressed(out / "Y_train_aug.npz", Y=aug_Y)
            print(f"[SEP] Augmentación guardada por separado.")

    # --------------------------------------------------------
    # --- GUARDAR SPLITS Y METADATOS --------------------------
    # --------------------------------------------------------
    for s, (X, Y) in splits.items():
        np.savez_compressed(out / f"X_{s}.npz", X=X)
        np.savez_compressed(out / f"Y_{s}.npz", Y=Y)
        print(f"[SAVE] {s}: {X.shape}")

    meta = {
        "N": args.N,
        "cap_bg": args.cap_bg,
        "use_augmentation": args.use_augmentation,
        "augment_times": args.augment_times,
        "merge_aug": args.merge_aug,
        "save_aug_separate": args.save_aug_separate,
        "label_map": {"id2idx": id2idx, "idx2id": idx2id},
        "total_classes": len(id2idx)
    }
    json.dump(meta, open(out / "artifacts" / "meta.json", "w"), indent=2)
    json.dump({"id2idx": id2idx, "idx2id": idx2id}, open(out / "artifacts" / "label_map.json", "w"), indent=2)

    # --------------------------------------------------------
    # --- VALIDACIÓN GLOBAL Y EDA BÁSICO ----------------------
    # --------------------------------------------------------
    print("\n[EDA] Validación global de splits:")
    all_classes = list(range(len(id2idx)))
    missing_per_split = {}

    for s, (_, Y) in splits.items():
        y_min, y_max = int(Y.min()), int(Y.max())
        uniq = np.unique(Y)
        missing = sorted(list(set(all_classes) - set(uniq)))
        bg_ratio = (Y == 0).sum() / Y.size * 100
        print(f"  [{s.upper()}] min={y_min} max={y_max} unique={len(uniq)} bg={bg_ratio:.2f}%")
        missing_per_split[s] = missing

        # Validaciones duras
        assert y_min >= 0, f"{s} contiene etiquetas negativas"
        assert y_max < len(id2idx), f"{s} tiene etiquetas fuera de rango ({y_max} >= {len(id2idx)})"

    # --------------------------------------------------------
    # --- REPORTE DE CLASES FALTANTES -------------------------
    # --------------------------------------------------------
    print("\n[CLASES FALTANTES POR SPLIT]:")
    for s, missing in missing_per_split.items():
        if missing:
            print(f"  {s}: {missing}")
        else:
            print(f"  {s}: OK (todas las clases presentes)")

    # Guardar resumen EDA en JSON
    eda_summary = {
        s: {
            "num_samples": int(X.shape[0]),
            "num_points": int(X.shape[1]),
            "unique_labels": int(len(np.unique(Y))),
            "missing_labels": missing_per_split[s],
            "bg_percentage": float((Y == 0).sum() / Y.size * 100)
        }
        for s, (X, Y) in splits.items()
    }
    json.dump(eda_summary, open(out / "artifacts" / "eda_summary.json", "w"), indent=2)

    print(f"\n[DONE] Splits guardados y verificados en: {out}")
    print(f"[META] Etiquetas totales: {len(id2idx)} clases")
    print(f"[EDA] Resumen -> {out/'artifacts'/'eda_summary.json'}")

if __name__ == "__main__":
    main()
