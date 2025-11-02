#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
augmentation_and_split_fixed_v3.py

Pipeline robusto:
- Remapeo GLOBAL 0..C-1 (artifacts/label_map.json)
- Submuestreo a N puntos con límite de fondo (--cap_bg)
- Augmentación de train (--use_augmentation, --augment_times, --merge_aug)
- ensure_coverage: mueve mínimas muestras entre splits para cobertura total
- Guarda EDA y meta en artifacts/

Entrada esperada (modo A, apilado):
  X_train.npz (X: (M,N,3)), Y_train.npz (Y: (M,N))
  X_val.npz, Y_val.npz
  X_test.npz, Y_test.npz

Salida:
  X_{split}.npz, Y_{split}.npz + artifacts/{label_map.json, meta.json, eda_summary.json, coverage_moves.json}
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# ---------------------------
# Augment helpers
# ---------------------------

def normalize_cloud_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mean = x.mean(axis=0, keepdims=True)
    x = x - mean
    r = np.linalg.norm(x, axis=1, keepdims=True).max()
    return x / (r + 1e-6)

def rotate_z(points: np.ndarray, max_deg: float = 15.0) -> np.ndarray:
    theta = np.random.uniform(-max_deg, max_deg) * (np.pi / 180.0)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0.0],
                  [s,  c, 0.0],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    return points.dot(R)

def jitter(points: np.ndarray, sigma: float = 0.005, clip: float = 0.02) -> np.ndarray:
    noise = np.clip(np.random.normal(0, sigma, points.shape), -clip, clip)
    return points + noise

def scale(points: np.ndarray, min_s: float = 0.95, max_s: float = 1.05) -> np.ndarray:
    s = np.random.uniform(min_s, max_s)
    return points * s

def dropout_points(points: np.ndarray, drop_rate: float = 0.05, min_keep: int = 32) -> np.ndarray:
    keep = np.random.rand(points.shape[0]) > drop_rate
    if keep.sum() < min_keep:
        idx = np.random.choice(points.shape[0], min_keep, replace=False)
        keep[idx] = True
    return points[keep]

def augment(points: np.ndarray,
            rotate_deg: float = 15,
            jitter_sigma: float = 0.005, jitter_clip: float = 0.02,
            scale_min: float = 0.95, scale_max: float = 1.05,
            dropout_rate: float = 0.05,
            n_target: int = None,
            rng: np.random.Generator = None) -> np.ndarray:
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

# ---------------------------
# IO helpers
# ---------------------------

def _load_npz_array(path: Path, key_fallbacks=("X","Y","arr_0")):
    with np.load(path, allow_pickle=False) as z:
        for k in key_fallbacks:
            if k in z:
                return z[k]
        # último recurso: primer key
        return z[list(z.keys())[0]]

def save_npz(path: Path, key: str, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{key: arr})

# ---------------------------
# Label mapping
# ---------------------------

def build_global_label_map(Y_train: np.ndarray, Y_val: np.ndarray, Y_test: np.ndarray):
    vals = sorted(set(np.unique(Y_train)).union(np.unique(Y_val)).union(np.unique(Y_test)))
    id2idx = {int(v): i for i, v in enumerate(vals)}
    idx2id = {i: int(v) for i, v in enumerate(vals)}
    return id2idx, idx2id

def remap_array_labels(Y: np.ndarray, id2idx: Dict[int, int]) -> np.ndarray:
    return np.vectorize(id2idx.__getitem__)(Y).astype(np.int32)

# ---------------------------
# Subsample with background cap
# ---------------------------

def subsample_with_cap(points: np.ndarray, labels: np.ndarray, N: int,
                       bg_idx_set: np.ndarray = None,
                       cap_bg_frac: float = None,
                       rng: np.random.Generator = None):
    rng = rng or np.random.default_rng()
    n = points.shape[0]
    idx_all = np.arange(n, dtype=np.int64)

    if cap_bg_frac is not None and bg_idx_set is not None and len(bg_idx_set) > 0:
        is_bg = np.isin(labels, bg_idx_set)
        idx_bg = idx_all[is_bg]
        idx_fg = idx_all[~is_bg]
        max_bg = int(round(cap_bg_frac * N))
        take_bg = min(max_bg, idx_bg.size)
        rem = max(0, N - take_bg)
        sel_bg = rng.choice(idx_bg, size=take_bg, replace=False) if take_bg > 0 else np.empty((0,), np.int64)
        pool = idx_fg if idx_fg.size > 0 else idx_bg
        sel_fg = rng.choice(pool, size=rem, replace=(pool.size < rem))
        sel = np.concatenate([sel_bg, sel_fg], axis=0)
    else:
        sel = rng.choice(idx_all, size=N, replace=(n < N))

    return points[sel], labels[sel]

def process_split_arrays(X: np.ndarray, Y: np.ndarray, N: int,
                         cap_bg: float, bg_idx_set: np.ndarray,
                         seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    M = X.shape[0]
    Xo = np.empty((M, N, 3), dtype=np.float32)
    Yo = np.empty((M, N), dtype=np.int32)
    for i in range(M):
        Xi, Yi = subsample_with_cap(X[i], Y[i].ravel(), N, bg_idx_set, cap_bg, rng)
        Xo[i] = Xi; Yo[i] = Yi
    return Xo, Yo

# ---------------------------
# Coverage ensure (move minimal samples)
# ---------------------------

def classes_present(Y: np.ndarray, C: int) -> np.ndarray:
    present = np.zeros(C, dtype=bool)
    uniq = np.unique(Y)
    present[uniq] = True
    return present

def pick_sample_indices_with_class(Y: np.ndarray, target_class: int) -> List[int]:
    # devuelve índices de muestras que contienen target_class
    hits = []
    for i in range(Y.shape[0]):
        if (Y[i] == target_class).any():
            hits.append(i)
    return hits

def ensure_coverage_all_splits(splits: Dict[str, Tuple[np.ndarray, np.ndarray]],
                               C: int,
                               prefer_from: List[str] = ["train", "val", "test"]) -> Dict:
    """
    Garantiza que train/val/test tengan TODAS las C clases.
    Mueve la MÍNIMA cantidad de muestras entre splits.
    Registro de movimientos devuelto.
    """
    moves = []
    order = ["train", "val", "test"]
    for target in order:
        X_t, Y_t = splits[target]
        pres_t = classes_present(Y_t, C)
        missing = np.where(~pres_t)[0].tolist()
        if not missing:
            continue

        for cls in missing:
            moved = False
            # buscar donante en preferencia (train>val>test, excluyendo target)
            for donor in prefer_from:
                if donor == target or donor not in splits:
                    continue
                X_d, Y_d = splits[donor]
                cand = pick_sample_indices_with_class(Y_d, cls)
                if not cand:
                    continue
                # elegir la primera muestra candidata
                j = cand[0]
                # mover X_d[j], Y_d[j] al target
                x_sample = X_d[j:j+1].copy()
                y_sample = Y_d[j:j+1].copy()
                splits[target] = (np.concatenate([splits[target][0], x_sample], axis=0),
                                  np.concatenate([splits[target][1], y_sample], axis=0))
                splits[donor] = (np.delete(X_d, j, axis=0), np.delete(Y_d, j, axis=0))
                moves.append({"class": int(cls), "from": donor, "to": target, "donor_idx": int(j)})
                moved = True
                break
            if not moved:
                # No se encontró donante con esa clase; quedará faltante
                moves.append({"class": int(cls), "from": None, "to": target, "donor_idx": None, "note": "no donor found"})
    return {"moves": moves,
            "final_counts": {k: int(v[0].shape[0]) for k, v in splits.items()}}

# ---------------------------
# EDA
# ---------------------------

def summarize_split(name: str, X: np.ndarray, Y: np.ndarray, bg_idx: int):
    y = Y
    uniq = np.unique(y)
    return {
        "split": name,
        "num_samples": int(X.shape[0]),
        "num_points": int(X.shape[1]),
        "min": int(y.min()),
        "max": int(y.max()),
        "unique": [int(u) for u in uniq.tolist()],
        "num_unique": int(len(uniq)),
        "bg_percentage": float((y == bg_idx).sum() / y.size * 100.0) if bg_idx is not None else None
    }

# ---------------------------
# MAIN
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True, help="Carpeta fuente con X/Y_{train,val,test}.npz")
    ap.add_argument("--out_dir",     required=True, help="Carpeta destino del split procesado")
    ap.add_argument("--N",           type=int, default=8192, help="N puntos por nube")
    ap.add_argument("--cap_bg",      type=float, default=None, help="Fracción máxima de fondo por nube (0..1). Ej: 0.4")
    ap.add_argument("--seed",        type=int, default=42)

    # Augmentación
    ap.add_argument("--use_augmentation", action="store_true")
    ap.add_argument("--augment_times", type=int, default=1)
    ap.add_argument("--rotate_deg", type=float, default=15)
    ap.add_argument("--jitter_sigma", type=float, default=0.005)
    ap.add_argument("--scale_min", type=float, default=0.95)
    ap.add_argument("--scale_max", type=float, default=1.05)
    ap.add_argument("--dropout_rate", type=float, default=0.05)
    ap.add_argument("--merge_aug", action="store_true")
    ap.add_argument("--save_aug_separate", action="store_true")

    # Cobertura
    ap.add_argument("--ensure_coverage", action="store_true", help="Garantiza que cada split tenga todas las clases (mueve muestras).")

    args = ap.parse_args()
    np.random.seed(args.seed)

    src = Path(args.dataset_dir)
    out = Path(args.out_dir)
    art = out / "artifacts"
    out.mkdir(parents=True, exist_ok=True)
    art.mkdir(parents=True, exist_ok=True)

    # 1) Cargar splits originales
    paths = {s: (src / f"X_{s}.npz", src / f"Y_{s}.npz") for s in ["train", "val", "test"]}
    for s, (xp, yp) in paths.items():
        if not xp.exists() or not yp.exists():
            raise FileNotFoundError(f"Falta {s}: {xp} o {yp}")

    X_train = _load_npz_array(paths["train"][0]).astype(np.float32)  # (M,N,3)
    Y_train = _load_npz_array(paths["train"][1]).astype(np.int64)    # (M,N)
    X_val   = _load_npz_array(paths["val"][0]).astype(np.float32)
    Y_val   = _load_npz_array(paths["val"][1]).astype(np.int64)
    X_test  = _load_npz_array(paths["test"][0]).astype(np.float32)
    Y_test  = _load_npz_array(paths["test"][1]).astype(np.int64)

    print(f"[OK] Loaded train: {X_train.shape}")
    print(f"[OK] Loaded val  : {X_val.shape}")
    print(f"[OK] Loaded test : {X_test.shape}")

    # 2) Remapeo GLOBAL
    id2idx, idx2id = build_global_label_map(Y_train, Y_val, Y_test)
    C = len(id2idx)
    # índice de fondo (original id 0)
    bg_idx = id2idx.get(0, None)

    Y_train_m = remap_array_labels(Y_train, id2idx)
    Y_val_m   = remap_array_labels(Y_val,   id2idx)
    Y_test_m  = remap_array_labels(Y_test,  id2idx)

    print(f"[MAP] Global: {C} clases totales (0..{C-1})")
    art.joinpath("label_map.json").write_text(json.dumps({"id2idx": id2idx, "idx2id": idx2id}, indent=2), encoding="utf-8")

    # 3) Submuestreo con cap de fondo
    bg_idx_set = np.array([bg_idx], dtype=np.int64) if bg_idx is not None else None
    X_train_p, Y_train_p = process_split_arrays(X_train, Y_train_m, args.N, args.cap_bg, bg_idx_set, args.seed)
    X_val_p,   Y_val_p   = process_split_arrays(X_val,   Y_val_m,   args.N, args.cap_bg, bg_idx_set, args.seed)
    X_test_p,  Y_test_p  = process_split_arrays(X_test,  Y_test_m,  args.N, args.cap_bg, bg_idx_set, args.seed)

    # 4) Augment sólo train
    if args.use_augmentation:
        rng = np.random.default_rng(args.seed)
        aug_X_list, aug_Y_list = [], []
        for i in range(X_train_p.shape[0]):
            for _ in range(args.augment_times):
                X_aug = augment(X_train_p[i],
                                rotate_deg=args.rotate_deg,
                                jitter_sigma=args.jitter_sigma,
                                scale_min=args.scale_min,
                                scale_max=args.scale_max,
                                dropout_rate=args.dropout_rate,
                                n_target=args.N,
                                rng=rng)
                aug_X_list.append(X_aug)
                aug_Y_list.append(Y_train_p[i])
        aug_X = np.stack(aug_X_list, axis=0).astype(np.float32) if aug_X_list else None
        aug_Y = np.stack(aug_Y_list, axis=0).astype(np.int32) if aug_Y_list else None
        print(f"[AUG] Generated {0 if aug_X is None else aug_X.shape[0]} augmented samples ({args.N} pts each).")

        if args.merge_aug and aug_X is not None:
            X_train_p = np.concatenate([X_train_p, aug_X], axis=0)
            Y_train_p = np.concatenate([Y_train_p, aug_Y], axis=0)
        elif args.save_aug_separate and aug_X is not None:
            save_npz(out / "X_train_aug.npz", "X", aug_X)
            save_npz(out / "Y_train_aug.npz", "Y", aug_Y)
            print("[SEP] Saved separate augmented split.")

    # 5) ensure_coverage (opcional)
    splits = {"train": (X_train_p, Y_train_p),
              "val":   (X_val_p,   Y_val_p),
              "test":  (X_test_p,  Y_test_p)}
    coverage_info = {"enabled": bool(args.ensure_coverage), "moves": [], "final_counts": {}}
    if args.ensure_coverage:
        coverage_info = ensure_coverage_all_splits(splits, C, prefer_from=["train", "val", "test"])
        print("[COVER] Movimientos realizados:", len(coverage_info["moves"]))
        art.joinpath("coverage_moves.json").write_text(json.dumps(coverage_info, indent=2), encoding="utf-8")

    # 6) Guardar splits
    for s, (X, Y) in splits.items():
        save_npz(out / f"X_{s}.npz", "X", X)
        save_npz(out / f"Y_{s}.npz", "Y", Y)
        print(f"[SAVE] {s}: {X.shape}")

    # 7) EDA + validación
    print("\n[EDA] Validación por split:")
    eda = {}
    for s, (X, Y) in splits.items():
        summ = summarize_split(s, X, Y, bg_idx)
        eda[s] = summ
        print(f"  {s}: min={summ['min']} max={summ['max']} unique={summ['num_unique']} "
              f"bg={summ['bg_percentage']:.2f}% puntos={summ['num_samples']*summ['num_points']:,}")
        # validaciones
        assert summ["min"] >= 0, f"{s} tiene etiquetas negativas"
        assert summ["max"] < C, f"{s} tiene etiquetas fuera de rango ({summ['max']} >= {C})"
    art.joinpath("eda_summary.json").write_text(json.dumps(eda, indent=2), encoding="utf-8")

    # 8) Meta
    meta = {
        "N": int(args.N),
        "cap_bg": None if args.cap_bg is None else float(args.cap_bg),
        "seed": int(args.seed),
        "use_augmentation": bool(args.use_augmentation),
        "augment_times": int(args.augment_times),
        "rotate_deg": float(args.rotate_deg),
        "jitter_sigma": float(args.jitter_sigma),
        "scale_min": float(args.scale_min),
        "scale_max": float(args.scale_max),
        "dropout_rate": float(args.dropout_rate),
        "merge_aug": bool(args.merge_aug),
        "save_aug_separate": bool(args.save_aug_separate),
        "ensure_coverage": bool(args.ensure_coverage),
        "num_classes": int(C),
        "bg_idx": None if bg_idx is None else int(bg_idx),
        "coverage_info": coverage_info
    }
    art.joinpath("meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\n[DONE] Splits y artefactos listos en: {out}")
    print(f"[META] Clases totales: {C}  (label_map.json, meta.json, eda_summary.json)")

if __name__ == "__main__":
    main()
