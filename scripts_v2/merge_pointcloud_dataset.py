#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Construye un dataset "por paciente" (pid) combinando upper+lower (si existen),
detectando automáticamente las carpetas correctas según el modo de muestreo
(FPS, estratificado, global).

Entrada (como processed_flat):
  data_dir/
    fps_vertices_base_upper/<CASE_U>/{point_cloud.npy, labels.npy?}
    fps_vertices_base_lower/<CASE_L>/{point_cloud.npy, labels.npy?}
  o también:
    stratified_sample_vert_labels_upper/
    surf_sample_global_lower/

- Empareja por pid (via --id_regex o --split_char/_index).
- --allow_unpaired para aceptar pacientes con 1 jaw.
- Puntos totales por paciente: --points_total (ej: 8192).
- --balance_jaws reparte P/2 y P/2 si hay ambos jaws.

Salida:
  out_dir/
    X_train.npz, Y_train.npz, J_train.npz, meta.json, index_*.csv
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np
import csv


# ---------------------------------------------------------------------
# Utilidades auxiliares
# ---------------------------------------------------------------------

def extract_pid(case_name: str,
                id_regex: Optional[str],
                split_char: Optional[str],
                split_index: int) -> str:
    """Extrae el identificador de paciente (pid) desde el nombre del caso."""
    if id_regex:
        m = re.search(id_regex, case_name)
        if m and m.groups():
            return m.group(1)
    if split_char is not None:
        parts = case_name.split(split_char)
        if 0 <= split_index < len(parts):
            return parts[split_index]
    return case_name


def find_jaw_dirs(dataset_dir: Path) -> Dict[str, Path]:
    """
    Detecta automáticamente carpetas que contengan 'upper' o 'lower' en el nombre.
    Ejemplo: fps_vertices_base_upper, surf_sample_global_lower, etc.
    """
    jaw_dirs = {"upper": None, "lower": None}
    for sub in dataset_dir.iterdir():
        if not sub.is_dir():
            continue
        name = sub.name.lower()
        if "upper" in name:
            jaw_dirs["upper"] = sub
        elif "lower" in name:
            jaw_dirs["lower"] = sub
    return jaw_dirs


def list_cases(dataset_dir: Path, id_regex, split_char, split_index) -> Dict[str, Dict[str, List[Dict]]]:
    """Devuelve dict por pid: {'upper':[...], 'lower':[...]} con paths y case_id."""
    jaw_dirs = find_jaw_dirs(dataset_dir)
    if not any(jaw_dirs.values()):
        raise SystemExit(f"[ERROR] No se encontraron carpetas 'upper' ni 'lower' en {dataset_dir}")

    by_pid: Dict[str, Dict[str, List[Dict]]] = {}
    for jaw, d in jaw_dirs.items():
        if d is None:
            continue
        for case in sorted(d.iterdir()):
            if not case.is_dir():
                continue
            pc = case / "point_cloud.npy"
            if not pc.exists():
                continue
            lb = case / "labels.npy"
            pid = extract_pid(case.name, id_regex, split_char, split_index)
            by_pid.setdefault(pid, {"upper": [], "lower": []})
            by_pid[pid][jaw].append({
                "case_id": case.name,
                "points_path": str(pc),
                "labels_path": str(lb) if lb.exists() else None
            })
    return by_pid


def load_np(path):
    return np.load(path)


def sample_points(pts: np.ndarray, P: int, rng: np.random.Generator):
    N = pts.shape[0]
    idx = rng.choice(N, P, replace=(N < P))
    return pts[idx], idx


def build_sample_for_pid(rec, points_total: int, balance_jaws: bool, rng: np.random.Generator,
                         allow_unpaired: bool):
    """
    Construye una muestra (una fila) para un pid:
      - Concat upper + lower (si existen). Si balance_jaws, reparte P/2 y P/2.
      - Devuelve (points [P,3], labels [P] or None, jaw_mask [P]), meta por sample.
    """
    have_u = len(rec["upper"]) > 0
    have_l = len(rec["lower"]) > 0
    if not allow_unpaired and not (have_u and have_l):
        return None

    u = rec["upper"][0] if have_u else None
    l = rec["lower"][0] if have_l else None

    # --- Carga upper ---
    pts_u = lbs_u = None
    if u is not None:
        pts_u = load_np(u["points_path"]).astype(np.float32)
        if u["labels_path"] is not None:
            lbs_u = load_np(u["labels_path"]).astype(np.int64).reshape(-1)
            if lbs_u.shape[0] != pts_u.shape[0]:
                m = min(lbs_u.shape[0], pts_u.shape[0])
                lbs_u = lbs_u[:m]
                if m < pts_u.shape[0]:
                    lbs_u = np.pad(lbs_u, (0, pts_u.shape[0] - m), mode='edge')

    # --- Carga lower ---
    pts_l = lbs_l = None
    if l is not None:
        pts_l = load_np(l["points_path"]).astype(np.float32)
        if l["labels_path"] is not None:
            lbs_l = load_np(l["labels_path"]).astype(np.int64).reshape(-1)
            if lbs_l.shape[0] != pts_l.shape[0]:
                m = min(lbs_l.shape[0], pts_l.shape[0])
                lbs_l = lbs_l[:m]
                if m < pts_l.shape[0]:
                    lbs_l = np.pad(lbs_l, (0, pts_l.shape[0] - m), mode='edge')

    if have_u and have_l:
        if balance_jaws:
            Pu = points_total // 2
            Pl = points_total - Pu
        else:
            Nu = pts_u.shape[0]
            Nl = pts_l.shape[0]
            tot = Nu + Nl
            Pu = max(1, int(round(points_total * Nu / tot)))
            Pl = points_total - Pu

        pts_u_s, idx_u = sample_points(pts_u, Pu, rng)
        pts_l_s, idx_l = sample_points(pts_l, Pl, rng)
        lbs_u_s = (lbs_u[idx_u] if lbs_u is not None else None)
        lbs_l_s = (lbs_l[idx_l] if lbs_l is not None else None)

        pts = np.concatenate([pts_u_s, pts_l_s], axis=0)
        if (lbs_u_s is not None) or (lbs_l_s is not None):
            if lbs_u_s is None:
                lbs_u_s = np.zeros(Pu, dtype=np.int64)
            if lbs_l_s is None:
                lbs_l_s = np.zeros(Pl, dtype=np.int64)
            lbs = np.concatenate([lbs_u_s, lbs_l_s], axis=0)
        else:
            lbs = None
        jaw_mask = np.concatenate(
            [np.zeros(Pu, dtype=np.int64), np.ones(Pl, dtype=np.int64)], axis=0
        )

    else:
        single = u if have_u else l
        pts_all = load_np(single["points_path"]).astype(np.float32)
        pts_s, idx_s = sample_points(pts_all, points_total, rng)
        if single["labels_path"] is not None:
            lbs_all = load_np(single["labels_path"]).astype(np.int64).reshape(-1)
            if lbs_all.shape[0] != pts_all.shape[0]:
                m = min(lbs_all.shape[0], pts_all.shape[0])
                lbs_all = lbs_all[:m]
                if m < pts_all.shape[0]:
                    lbs_all = np.pad(lbs_all, (0, pts_all.shape[0] - m), mode='edge')
            lbs_s = lbs_all[idx_s]
        else:
            lbs_s = None
        pts = pts_s
        lbs = lbs_s
        jaw_mask = np.zeros(points_total, dtype=np.int64) if have_u else np.ones(points_total, dtype=np.int64)

    meta = {
        "case_upper": u["case_id"] if u else None,
        "case_lower": l["case_id"] if l else None,
    }
    return pts, lbs, jaw_mask, meta


def save_npz(path: Path, key: str, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{key: arr})


def write_index_csv(path: Path, meta_rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(meta_rows[0].keys()))
        w.writeheader()
        for r in meta_rows:
            w.writerow(r)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True, help="Ruta de processed_flat/8192/")
    ap.add_argument("--out_dir", required=True, help="Ruta destino merged_dataset/")
    ap.add_argument("--points_total", type=int, default=8192, help="P total por paciente")
    ap.add_argument("--balance_jaws", action="store_true", help="P mitad/mitad si hay upper+lower")
    ap.add_argument("--allow_unpaired", action="store_true", help="Permite pacientes con 1 jaw")
    ap.add_argument("--id_regex", default=None)
    ap.add_argument("--split_char", default=None)
    ap.add_argument("--split_index", type=int, default=0)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    dataset_dir = Path(args.dataset_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    by_pid = list_cases(dataset_dir, args.id_regex, args.split_char, args.split_index)
    pids = sorted(by_pid.keys())
    print(f"[INFO] Pacientes detectados: {len(pids)}")

    X_list, Y_list, J_list, meta_cases = [], [], [], []
    for pid in pids:
        rec = by_pid[pid]
        have_u = len(rec["upper"]) > 0
        have_l = len(rec["lower"]) > 0
        if not args.allow_unpaired and not (have_u and have_l):
            continue

        sample = build_sample_for_pid(rec, args.points_total, args.balance_jaws, rng, args.allow_unpaired)
        if sample is None:
            continue

        pts, lbs, jaw_mask, meta = sample
        X_list.append(pts.astype(np.float32))
        J_list.append(jaw_mask.astype(np.int64))
        if lbs is None:
            lbs = np.zeros(pts.shape[0], dtype=np.int64)
        Y_list.append(lbs.astype(np.int64))
        meta_cases.append({"pid": pid, **meta})

    if len(X_list) == 0:
        raise SystemExit("No se pudo construir ninguna muestra. Revisa flags/pairs/formato de entrada.")

    X = np.stack(X_list, axis=0)   # [N,P,3]
    Y = np.stack(Y_list, axis=0)   # [N,P]
    J = np.stack(J_list, axis=0)   # [N,P]
    N = X.shape[0]
    print(f"[OK] Dataset: N={N}  P={X.shape[1]}")

    # Splits
    idx = np.arange(N)
    rng.shuffle(idx)
    n_test = int(round(args.test_ratio * N))
    n_val = int(round(args.val_ratio * N))
    n_train = N - n_val - n_test

    Itr, Iva, Ite = idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]

    # Guardar NPZ
    save_npz(out_dir / "X_train.npz", "X", X[Itr])
    save_npz(out_dir / "X_val.npz", "X", X[Iva])
    save_npz(out_dir / "X_test.npz", "X", X[Ite])
    save_npz(out_dir / "Y_train.npz", "Y", Y[Itr])
    save_npz(out_dir / "Y_val.npz", "Y", Y[Iva])
    save_npz(out_dir / "Y_test.npz", "Y", Y[Ite])
    save_npz(out_dir / "J_train.npz", "J", J[Itr])
    save_npz(out_dir / "J_val.npz", "J", J[Iva])
    save_npz(out_dir / "J_test.npz", "J", J[Ite])

    meta = {
        "points_total": int(args.points_total),
        "balance_jaws": bool(args.balance_jaws),
        "allow_unpaired": bool(args.allow_unpaired),
        "N_total": int(N),
        "splits": {"train": len(Itr), "val": len(Iva), "test": len(Ite)}
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[OK] Guardado en: {out_dir}")
    print("  - X_*.npz, Y_*.npz, J_*.npz, meta.json")

if __name__ == "__main__":
    main()
