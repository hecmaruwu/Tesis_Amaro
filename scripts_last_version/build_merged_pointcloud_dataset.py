#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_merged_pointcloud_dataset.py

Construye un dataset global (PointNet-like) desde nubes preprocesadas por jaw.

Entrada (soporta 2 layouts):
A) processed_struct_safe/<N>/<jaw>/<sample>/{point_cloud.npy, labels.npy?}
B) processed_struct_safe/<jaw>/<sample>/{point_cloud.npy, labels.npy?}

Salida:
  out_dir/
    X_train.npz (key "X")
    Y_train.npz (key "Y")
    X_val.npz
    Y_val.npz
    X_test.npz
    Y_test.npz
    index_train.csv
    index_val.csv
    index_test.csv
    artifacts/
      label_map.json   (id2idx / idx2id)
      meta.json

Notas:
- Este script NO submuestrea (no baja 200k->8192). Solo empaqueta a NPZ.
- La exclusión de muelas del juicio ya debe venir aplicada desde preprocess.
"""

import argparse
import csv
import gc
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# ------------------------- Utils -------------------------

def seed_all(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def normalize(points: np.ndarray) -> np.ndarray:
    """
    Normaliza a esfera unitaria:
      - centra en promedio 0
      - escala por radio máximo
    """
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        return points.astype(np.float32, copy=False)

    if not np.isfinite(points).all():
        points = np.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)

    points = points - points.mean(axis=0, keepdims=True)
    d = np.linalg.norm(points, axis=1)
    m = float(d.max()) if d.size else 1.0
    if np.isfinite(m) and m > 0:
        points = points / m
    return points


def load_sample(folder: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carga:
      - point_cloud.npy obligatorio
      - labels.npy opcional (si falta: ceros)
    """
    pc_path = folder / "point_cloud.npy"
    if not pc_path.exists():
        raise FileNotFoundError(f"No existe {pc_path}")

    X = np.load(pc_path).astype(np.float32)
    if X.ndim != 2 or X.shape[1] != 3 or X.shape[0] == 0:
        raise ValueError(f"point_cloud.npy inválido en {folder} (shape={getattr(X,'shape',None)})")

    y_path = folder / "labels.npy"
    if y_path.exists():
        Y = np.load(y_path).astype(np.int32).reshape(-1)
        # Ajuste defensivo si hay mismatch (idealmente no debería pasar)
        if Y.shape[0] != X.shape[0]:
            m = min(Y.shape[0], X.shape[0])
            Y = Y[:m]
            if m < X.shape[0]:
                pad_val = int(Y[-1]) if Y.size else 0
                Y = np.pad(Y, (0, X.shape[0] - m), mode="constant", constant_values=pad_val)
    else:
        Y = np.zeros((X.shape[0],), dtype=np.int32)

    return X, Y


def build_label_map(all_labels: np.ndarray) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Construye mapeo global (sobre TODO el dataset) a [0..C-1]
    """
    unique = sorted(int(x) for x in np.unique(all_labels))
    id2idx = {int(v): i for i, v in enumerate(unique)}
    idx2id = {i: int(v) for i, v in enumerate(unique)}
    return id2idx, idx2id


def split_indices(n: int, ratios=(0.8, 0.1, 0.1), seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split simple reproducible: shuffle + cortes.
    """
    if n <= 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)

    seed_all(seed)
    idx = np.arange(n)
    np.random.shuffle(idx)

    r_train, r_val, r_test = ratios
    n_train = int(r_train * n)
    n_val = int(r_val * n)

    itrain = idx[:n_train]
    ival = idx[n_train:n_train + n_val]
    itest = idx[n_train + n_val:]
    return itrain, ival, itest


def write_index_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def resolve_base_dir(in_root: Path, n_points: int, jaws: List[str]) -> Tuple[Path, bool]:
    """
    Detecta layout:
      A) in_root/<n_points>/<jaw>/...
      B) in_root/<jaw>/...

    Retorna (base_dir, uses_npoints_subdir)
    """
    cand_a = in_root / str(n_points)
    if cand_a.exists() and all((cand_a / j).exists() for j in jaws):
        return cand_a, True

    if all((in_root / j).exists() for j in jaws):
        return in_root, False

    # fallback: devolver A para construir error útil
    return cand_a, True


# ------------------------- Main -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", required=True, help="processed_struct_safe (raíz)")
    ap.add_argument("--out_dir", required=True, help="Directorio de salida merged_*")
    ap.add_argument("--n_points", type=int, default=200000, help="N esperado en carpeta (ej: 200000)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ratios", type=float, nargs=3, default=(0.8, 0.1, 0.1), help="train val test")
    ap.add_argument("--jaws", type=str, default="upper,lower", help="Ej: upper,lower o solo upper")
    ap.add_argument("--require_labels", action="store_true",
                    help="Si se activa, descarta muestras sin labels.npy (en vez de rellenar con 0).")
    args = ap.parse_args()

    seed_all(args.seed)

    in_root = Path(args.in_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "artifacts").mkdir(exist_ok=True)

    jaws = [j.strip() for j in args.jaws.split(",") if j.strip()]
    if not jaws:
        raise SystemExit("[ERROR] --jaws vacío. Ej: --jaws upper,lower")

    base_dir, uses_npoints_subdir = resolve_base_dir(in_root, args.n_points, jaws)

    print("[INFO] in_root:", in_root)
    print("[INFO] base_dir:", base_dir, ("(layout A: /<n_points>/<jaw>)" if uses_npoints_subdir else "(layout B: /<jaw>)"))
    print("[INFO] jaws:", jaws)
    print("[INFO] ratios:", args.ratios)

    all_X: List[np.ndarray] = []
    all_Y: List[np.ndarray] = []
    meta_rows: List[Dict] = []

    found = 0
    skipped = 0

    for jaw in jaws:
        jaw_dir = base_dir / jaw
        if not jaw_dir.exists():
            print(f"[WARN] No existe jaw_dir: {jaw_dir}")
            continue

        for case_dir in sorted(jaw_dir.iterdir()):
            if not case_dir.is_dir():
                continue

            pc = case_dir / "point_cloud.npy"
            if not pc.exists():
                skipped += 1
                continue

            lb = case_dir / "labels.npy"
            if args.require_labels and (not lb.exists()):
                skipped += 1
                continue

            try:
                X, Y = load_sample(case_dir)
                X = normalize(X)

                if Y.shape[0] != X.shape[0]:
                    raise ValueError(f"labels no calzan con puntos (Y={Y.shape[0]}, X={X.shape[0]})")

                # warning si N no coincide (no aborta)
                if args.n_points is not None and X.shape[0] != int(args.n_points):
                    print(f"[WARN] {jaw}/{case_dir.name}: X tiene {X.shape[0]} pts, esperado {args.n_points}. Igual lo incluyo.")

                all_X.append(X.astype(np.float32, copy=False))
                all_Y.append(Y.astype(np.int32, copy=False))
                meta_rows.append({
                    "sample_name": case_dir.name,
                    "jaw": jaw,
                    "path": str(case_dir),
                    "n_points": int(X.shape[0]),
                    "has_labels": int((case_dir / "labels.npy").exists()),
                })
                found += 1

            except Exception as e:
                print(f"[WARN] No se pudo leer {case_dir}: {e}")
                skipped += 1
                continue

    if found == 0:
        exp_a = in_root / str(args.n_points) / jaws[0] / "<CASE>/point_cloud.npy"
        exp_b = in_root / jaws[0] / "<CASE>/point_cloud.npy"
        raise SystemExit(
            "[ERROR] No se encontró ninguna muestra (all_X vacío).\n"
            "Revisa:\n"
            "  - --in_root correcto\n"
            "  - que existan carpetas de jaw y contengan point_cloud.npy\n"
            "  - que el layout sea A o B\n\n"
            f"Esperado layout A: {exp_a}\n"
            f"Esperado layout B: {exp_b}\n"
            f"in_root={in_root}\n"
            f"n_points={args.n_points}\n"
            f"jaws={jaws}\n"
        )

    # Stack
    X = np.stack(all_X, axis=0).astype(np.float32, copy=False)
    Y = np.stack(all_Y, axis=0).astype(np.int32, copy=False)
    del all_X, all_Y
    gc.collect()

    print(f"[OK] Dataset cargado: {X.shape[0]} muestras | P={X.shape[1]} | skipped={skipped}")

    # Label map global + remap
    id2idx, idx2id = build_label_map(Y)
    with (out_dir / "artifacts" / "label_map.json").open("w", encoding="utf-8") as f:
        json.dump({"id2idx": id2idx, "idx2id": idx2id}, f, indent=2)

    Y_remap = np.vectorize(id2idx.get)(Y).astype(np.int32, copy=False)

    # Splits
    itrain, ival, itest = split_indices(X.shape[0], ratios=tuple(args.ratios), seed=args.seed)

    def save_split(name: str, idxs: np.ndarray):
        np.savez_compressed(out_dir / f"X_{name}.npz", X=X[idxs])
        np.savez_compressed(out_dir / f"Y_{name}.npz", Y=Y_remap[idxs])
        print(f"[SAVE] {name}: {len(idxs)} muestras")

    save_split("train", itrain)
    save_split("val", ival)
    save_split("test", itest)

    # Index CSV
    idx_rows = [{"idx": int(i), **meta_rows[int(i)]} for i in range(len(meta_rows))]
    write_index_csv(out_dir / "index_train.csv", [idx_rows[int(i)] for i in itrain])
    write_index_csv(out_dir / "index_val.csv",   [idx_rows[int(i)] for i in ival])
    write_index_csv(out_dir / "index_test.csv",  [idx_rows[int(i)] for i in itest])

    # Meta
    meta = {
        "in_root": str(in_root),
        "base_dir": str(base_dir),
        "uses_npoints_subdir": bool(uses_npoints_subdir),
        "n_points_expected": int(args.n_points),
        "n_samples": int(X.shape[0]),
        "n_points_actual": int(X.shape[1]),
        "n_classes": int(len(id2idx)),
        "seed": int(args.seed),
        "ratios": {"train": float(args.ratios[0]), "val": float(args.ratios[1]), "test": float(args.ratios[2])},
        "jaws": list(jaws),
        "require_labels": bool(args.require_labels),
        "labels_unique_original": sorted([int(k) for k in id2idx.keys()]),
    }
    with (out_dir / "artifacts" / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[DONE] Guardado en: {out_dir}")
    print(f"       Clases (originales): {meta['labels_unique_original']}")
    print(f"       n_classes remapeadas: {meta['n_classes']}")


if __name__ == "__main__":
    main()
