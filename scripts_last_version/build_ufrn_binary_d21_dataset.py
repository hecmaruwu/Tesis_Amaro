#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_ufrn_binary_d21_dataset.py

Crea dataset UFRN binario para fine-tuning d21 vs fondo.

Soporta dos modos de split:

1) Split fijo por rango:
   train: paciente_1  .. paciente_train_end
   val:   paciente_train_end+1 .. paciente_val_end
   test:  paciente_val_end+1 .. final

2) Split aleatorio por semilla:
   --split_mode random
   --split_seed 42
   --n_train 35
   --n_val 8
   --n_test 8

Entrada:
- data/UFRN/targets_export/paciente_X/stl/upper_full.stl
- data/UFRN/pseudolabels_dbscan_all/paciente_X/gt_removed_d21_dbscan.npy

Salida:
- X_train.npz, Y_train.npz
- X_val.npz, Y_val.npz
- X_test.npz, Y_test.npz
- index_train.csv, index_val.csv, index_test.csv
- all_index.csv
- artifacts/meta.json

Etiqueta:
- 1 = diente 21 removido según pseudo-label clínica DBSCAN
- 0 = resto de la arcada
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree


# ============================================================
# Carga y muestreo
# ============================================================

def load_mesh_safe(mesh_path: Path):
    mesh = trimesh.load(mesh_path, process=False, force="mesh")

    if isinstance(mesh, trimesh.Scene):
        geoms = list(mesh.geometry.values())
        if len(geoms) == 0:
            raise RuntimeError(f"Scene vacía: {mesh_path}")
        mesh = trimesh.util.concatenate(geoms)

    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"No se pudo cargar como Trimesh: {mesh_path}")

    if mesh.vertices is None or len(mesh.vertices) == 0:
        raise RuntimeError(f"Malla sin vértices: {mesh_path}")

    return mesh


def sample_mesh(mesh_path: Path, n_points: int, seed: int):
    """
    Muestrea puntos en la superficie de la malla.
    Usa seed específico por paciente para reproducibilidad.
    """
    np.random.seed(int(seed))
    mesh = load_mesh_safe(mesh_path)

    try:
        pts, _ = trimesh.sample.sample_surface(mesh, int(n_points))
    except Exception:
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        replace = verts.shape[0] < int(n_points)
        idx = np.random.choice(verts.shape[0], int(n_points), replace=replace)
        pts = verts[idx]

    pts = np.asarray(pts, dtype=np.float32)

    if not np.isfinite(pts).all():
        pts = np.nan_to_num(pts, nan=0.0, posinf=0.0, neginf=0.0)

    return pts.astype(np.float32)


def normalize_unit_sphere(points: np.ndarray):
    """
    Normaliza la nube a esfera unitaria:
      x_norm = (x_raw - center) / scale

    Se guardan center y scale para reconstruir:
      x_raw = x_norm * scale + center
    """
    points = np.asarray(points, dtype=np.float32)

    center = points.mean(axis=0, keepdims=True)
    x = points - center

    scale = np.linalg.norm(x, axis=1).max()

    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    x = x / (scale + 1e-8)

    return x.astype(np.float32), center.astype(np.float32), float(scale)


def make_binary_labels(points_raw: np.ndarray, gt_d21: np.ndarray, radius: float):
    """
    Etiqueta como 1 los puntos de upper_full que están dentro del radio
    respecto a la pseudo-label d21 generada por diferencia full vs rec_21.
    """
    points_raw = np.asarray(points_raw, dtype=np.float32)
    gt_d21 = np.asarray(gt_d21, dtype=np.float32)

    if gt_d21.shape[0] == 0:
        return np.zeros(points_raw.shape[0], dtype=np.int64), {
            "positive_points": 0,
            "positive_frac": 0.0,
            "radius": float(radius),
        }

    tree = cKDTree(gt_d21)
    dists, _ = tree.query(points_raw, k=1)

    y = (dists <= float(radius)).astype(np.int64)

    info = {
        "positive_points": int(y.sum()),
        "positive_frac": float(y.mean()),
        "radius": float(radius),
        "min_dist": float(dists.min()),
        "mean_dist": float(dists.mean()),
        "max_dist": float(dists.max()),
    }

    return y, info


# ============================================================
# Pacientes y splits
# ============================================================

def patient_number(pid: str):
    try:
        return int(pid.replace("paciente_", ""))
    except Exception:
        return 10**9


def get_patients(ufrn_root: Path):
    patients = sorted(
        [
            p for p in ufrn_root.iterdir()
            if p.is_dir() and p.name.startswith("paciente_")
        ],
        key=lambda p: patient_number(p.name)
    )

    return patients


def build_fixed_split_map(patients, train_end: int, val_end: int):
    """
    Modo clásico:
      paciente_1..train_end = train
      paciente_train_end+1..val_end = val
      resto = test
    """
    split_map = {}

    for pdir in patients:
        pid = pdir.name
        pnum = patient_number(pid)

        if pnum <= int(train_end):
            split = "train"
        elif pnum <= int(val_end):
            split = "val"
        else:
            split = "test"

        split_map[pid] = split

    return split_map


def build_random_split_map(
    patients,
    split_seed: int,
    n_train: int,
    n_val: int,
    n_test: int,
):
    """
    Split aleatorio por paciente.

    Importante:
    - No mezcla puntos.
    - Cada paciente queda completo en un único split.
    - Mantiene trazabilidad por patient_id.
    """
    rng = np.random.default_rng(int(split_seed))

    patient_ids = [p.name for p in patients]
    patient_ids_sorted = sorted(patient_ids, key=patient_number)

    total_requested = int(n_train) + int(n_val) + int(n_test)

    if total_requested > len(patient_ids_sorted):
        raise RuntimeError(
            f"n_train+n_val+n_test={total_requested}, "
            f"pero solo hay {len(patient_ids_sorted)} pacientes."
        )

    shuffled = list(patient_ids_sorted)
    rng.shuffle(shuffled)

    train_ids = shuffled[:int(n_train)]
    val_ids = shuffled[int(n_train):int(n_train) + int(n_val)]
    test_ids = shuffled[int(n_train) + int(n_val):int(n_train) + int(n_val) + int(n_test)]

    split_map = {}

    for pid in train_ids:
        split_map[pid] = "train"

    for pid in val_ids:
        split_map[pid] = "val"

    for pid in test_ids:
        split_map[pid] = "test"

    # Si sobran pacientes, los dejamos fuera explícitamente.
    # En su caso 35+8+8 = 51, así que no debería sobrar ninguno.
    selected = set(split_map.keys())
    excluded = [pid for pid in patient_ids_sorted if pid not in selected]

    return split_map, {
        "split_seed": int(split_seed),
        "n_train_requested": int(n_train),
        "n_val_requested": int(n_val),
        "n_test_requested": int(n_test),
        "train_ids": sorted(train_ids, key=patient_number),
        "val_ids": sorted(val_ids, key=patient_number),
        "test_ids": sorted(test_ids, key=patient_number),
        "excluded_ids": sorted(excluded, key=patient_number),
    }


# ============================================================
# Guardado
# ============================================================

def save_split(out_dir: Path, split: str, X_list, Y_list, rows):
    X = np.stack(X_list, axis=0).astype(np.float32)
    Y = np.stack(Y_list, axis=0).astype(np.int64)

    np.savez_compressed(out_dir / f"X_{split}.npz", X=X)
    np.savez_compressed(out_dir / f"Y_{split}.npz", Y=Y)

    index_path = out_dir / f"index_{split}.csv"

    with open(index_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "row",
            "patient_id",
            "split",
            "upper_full_stl",
            "gt_d21_npy",
            "n_points",
            "positive_points",
            "positive_frac",
            "label_radius",
            "center_x",
            "center_y",
            "center_z",
            "scale",
            "sampling_seed",
            "split_seed",
            "split_mode",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, r in enumerate(rows):
            r = dict(r)
            r["row"] = i
            writer.writerow(r)

    print(f"[SAVE] {split}: X={X.shape}, Y={Y.shape}, index={index_path}")


def write_all_index(out_dir: Path, global_rows):
    all_index_path = out_dir / "all_index.csv"

    with open(all_index_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "patient_id",
            "split",
            "upper_full_stl",
            "gt_d21_npy",
            "n_points",
            "positive_points",
            "positive_frac",
            "label_radius",
            "center_x",
            "center_y",
            "center_z",
            "scale",
            "sampling_seed",
            "split_seed",
            "split_mode",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in global_rows:
            writer.writerow(r)

    print("[SAVE] all_index:", all_index_path)


def split_positive_summary(rows):
    if len(rows) == 0:
        return {
            "mean_positive_frac": None,
            "std_positive_frac": None,
            "min_positive_frac": None,
            "max_positive_frac": None,
            "mean_positive_points": None,
        }

    fracs = np.asarray([r["positive_frac"] for r in rows], dtype=np.float64)
    pts = np.asarray([r["positive_points"] for r in rows], dtype=np.float64)

    return {
        "mean_positive_frac": float(fracs.mean()),
        "std_positive_frac": float(fracs.std(ddof=0)),
        "min_positive_frac": float(fracs.min()),
        "max_positive_frac": float(fracs.max()),
        "mean_positive_points": float(pts.mean()),
    }


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--ufrn_root", required=True)
    ap.add_argument("--pseudolabel_root", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--n_points", type=int, default=8192)
    ap.add_argument("--label_radius", type=float, default=1.25)

    # Seed de muestreo geométrico.
    # Para que el mismo paciente tenga el mismo muestreo si el seed base no cambia.
    ap.add_argument("--seed", type=int, default=42)

    # Modo de split
    ap.add_argument(
        "--split_mode",
        choices=["fixed", "random"],
        default="fixed",
        help="fixed = rangos por número de paciente; random = split aleatorio por paciente."
    )

    # Split fijo clásico
    ap.add_argument("--train_end", type=int, default=35)
    ap.add_argument("--val_end", type=int, default=43)

    # Split aleatorio
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--n_train", type=int, default=35)
    ap.add_argument("--n_val", type=int, default=8)
    ap.add_argument("--n_test", type=int, default=8)

    args = ap.parse_args()

    ufrn_root = Path(args.ufrn_root)
    pseudolabel_root = Path(args.pseudolabel_root)
    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    patients = get_patients(ufrn_root)

    if len(patients) == 0:
        raise RuntimeError(f"No se encontraron pacientes en {ufrn_root}")

    print("[INFO] Pacientes encontrados:", len(patients))

    # ------------------------------------------------------------
    # Construir split map
    # ------------------------------------------------------------

    if args.split_mode == "fixed":
        split_map = build_fixed_split_map(
            patients=patients,
            train_end=args.train_end,
            val_end=args.val_end,
        )

        split_info = {
            "split_mode": "fixed",
            "train_rule": f"paciente_1..paciente_{args.train_end}",
            "val_rule": f"paciente_{args.train_end + 1}..paciente_{args.val_end}",
            "test_rule": f"paciente_{args.val_end + 1}..end",
            "split_seed": None,
            "train_ids": sorted(
                [pid for pid, s in split_map.items() if s == "train"],
                key=patient_number
            ),
            "val_ids": sorted(
                [pid for pid, s in split_map.items() if s == "val"],
                key=patient_number
            ),
            "test_ids": sorted(
                [pid for pid, s in split_map.items() if s == "test"],
                key=patient_number
            ),
            "excluded_ids": [],
        }

    else:
        split_map, split_info = build_random_split_map(
            patients=patients,
            split_seed=args.split_seed,
            n_train=args.n_train,
            n_val=args.n_val,
            n_test=args.n_test,
        )

        split_info["split_mode"] = "random"

    print("[INFO] split_mode:", args.split_mode)
    print("[INFO] n_train:", len(split_info["train_ids"]))
    print("[INFO] n_val:", len(split_info["val_ids"]))
    print("[INFO] n_test:", len(split_info["test_ids"]))

    if len(split_info.get("excluded_ids", [])) > 0:
        print("[WARN] Pacientes excluidos:", split_info["excluded_ids"])

    splits = {
        "train": {"X": [], "Y": [], "rows": []},
        "val": {"X": [], "Y": [], "rows": []},
        "test": {"X": [], "Y": [], "rows": []},
    }

    global_rows = []
    failed = []

    # ------------------------------------------------------------
    # Procesar pacientes
    # ------------------------------------------------------------

    for pdir in patients:
        pid = pdir.name
        pnum = patient_number(pid)

        if pid not in split_map:
            print("[SKIP]", pid, "no seleccionado por split.")
            continue

        split = split_map[pid]

        try:
            print("=" * 80)
            print(f"[INFO] Procesando {pid} -> {split}")

            upper_full_stl = pdir / "stl" / "upper_full.stl"
            gt_d21_npy = pseudolabel_root / pid / "gt_removed_d21_dbscan.npy"

            if not upper_full_stl.exists():
                raise FileNotFoundError(upper_full_stl)

            if not gt_d21_npy.exists():
                raise FileNotFoundError(gt_d21_npy)

            sampling_seed = int(args.seed) + int(pnum)

            pts_raw = sample_mesh(
                upper_full_stl,
                n_points=args.n_points,
                seed=sampling_seed,
            )

            gt_d21 = np.load(
                gt_d21_npy,
                allow_pickle=True
            ).astype(np.float32)

            y, label_info = make_binary_labels(
                points_raw=pts_raw,
                gt_d21=gt_d21,
                radius=args.label_radius,
            )

            pts_norm, center, scale = normalize_unit_sphere(pts_raw)

            row = {
                "patient_id": pid,
                "split": split,
                "upper_full_stl": str(upper_full_stl),
                "gt_d21_npy": str(gt_d21_npy),
                "n_points": int(args.n_points),
                "positive_points": int(label_info["positive_points"]),
                "positive_frac": float(label_info["positive_frac"]),
                "label_radius": float(args.label_radius),
                "center_x": float(center[0, 0]),
                "center_y": float(center[0, 1]),
                "center_z": float(center[0, 2]),
                "scale": float(scale),
                "sampling_seed": int(sampling_seed),
                "split_seed": int(args.split_seed) if args.split_mode == "random" else "",
                "split_mode": str(args.split_mode),
            }

            splits[split]["X"].append(pts_norm)
            splits[split]["Y"].append(y)
            splits[split]["rows"].append(row)
            global_rows.append(row)

            print(
                "[OK]",
                pid,
                "positivos:",
                row["positive_points"],
                "frac:",
                row["positive_frac"],
            )

        except Exception as e:
            print("[ERROR]", pid, e)
            failed.append({
                "patient_id": pid,
                "split": split,
                "error": str(e),
            })

    # ------------------------------------------------------------
    # Guardar splits
    # ------------------------------------------------------------

    for split in ["train", "val", "test"]:
        if len(splits[split]["X"]) == 0:
            raise RuntimeError(f"No hay muestras para split {split}")

        save_split(
            out_dir=out_dir,
            split=split,
            X_list=splits[split]["X"],
            Y_list=splits[split]["Y"],
            rows=splits[split]["rows"],
        )

    write_all_index(out_dir, global_rows)

    # ------------------------------------------------------------
    # Meta
    # ------------------------------------------------------------

    meta = {
        "ufrn_root": str(ufrn_root),
        "pseudolabel_root": str(pseudolabel_root),
        "out_dir": str(out_dir),
        "n_points": int(args.n_points),
        "label_radius": float(args.label_radius),
        "seed_sampling_base": int(args.seed),
        "split_info": split_info,
        "n_train": int(len(splits["train"]["X"])),
        "n_val": int(len(splits["val"]["X"])),
        "n_test": int(len(splits["test"]["X"])),
        "n_failed": int(len(failed)),
        "failed": failed,
        "positive_summary": {
            "train": split_positive_summary(splits["train"]["rows"]),
            "val": split_positive_summary(splits["val"]["rows"]),
            "test": split_positive_summary(splits["test"]["rows"]),
        },
    }

    with open(out_dir / "artifacts" / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("=" * 80)
    print("[DONE]")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()