#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Crea un dataset (por caso) a partir de processed_flat:
  data/Teeth_3ds/processed_flat/
    upper/<CASE>/{point_cloud.npy, labels.npy?}
    lower/<CASE>/{point_cloud.npy, labels.npy?}

Opcional: exigir pares upper+lower por pid.
Guarda NPZ con claves "X" y "Y" (compatibles con train_models.py) más meta.json.
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np

# -------------------- helpers --------------------

def extract_pid(case_name: str,
                id_regex: Optional[str],
                split_char: Optional[str],
                split_index: int) -> str:
    if id_regex:
        m = re.search(id_regex, case_name)
        if not m or not m.groups():
            return case_name
        return m.group(1)
    if split_char is not None:
        parts = case_name.split(split_char)
        if 0 <= split_index < len(parts):
            return parts[split_index]
    return case_name

def pad_or_sample(points: np.ndarray,
                  labels: Optional[np.ndarray],
                  P: int,
                  rng: np.random.Generator
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points debe ser [N,3], recibido {points.shape}")
    labels_arr = None
    if labels is not None:
        labels_arr = np.asarray(labels).reshape(-1).astype(np.int64, copy=False)
        if labels_arr.shape[0] != points.shape[0]:
            k = min(labels_arr.shape[0], points.shape[0])
            labels_arr = labels_arr[:k]
            if k < points.shape[0]:
                labels_arr = np.pad(labels_arr, (0, points.shape[0] - k), mode='edge')
    N = points.shape[0]
    idx = rng.choice(N, P, replace=(N < P))
    points = points[idx]
    labels_arr = (labels_arr[idx] if labels_arr is not None else None)
    return points, labels_arr

def scan_cases(dataset_dir: Path,
               id_regex: Optional[str],
               split_char: Optional[str],
               split_index: int) -> List[Dict]:
    out = []
    for jaw in ["upper", "lower"]:
        jaw_dir = dataset_dir / jaw
        if not jaw_dir.is_dir():
            continue
        for case in sorted(jaw_dir.iterdir()):
            if not case.is_dir():
                continue
            pc = case / "point_cloud.npy"
            if not pc.exists():
                continue
            lb = case / "labels.npy"
            pid = extract_pid(case.name, id_regex, split_char, split_index)
            out.append({
                "jaw": jaw,
                "case_id": case.name,
                "pid": pid,
                "points_path": str(pc),
                "labels_path": str(lb) if lb.exists() else None
            })
    return out

def filter_paired(cases: List[Dict]) -> Tuple[List[Dict], Dict]:
    by_pid: Dict[str, Dict[str, List[Dict]]] = {}
    for c in cases:
        by_pid.setdefault(c["pid"], {"upper": [], "lower": []})
        by_pid[c["pid"]][c["jaw"]].append(c)
    paired, unpaired = [], {"upper_only": [], "lower_only": []}
    for pid, d in by_pid.items():
        has_u = len(d["upper"]) > 0
        has_l = len(d["lower"]) > 0
        if has_u and has_l:
            paired.extend(d["upper"] + d["lower"])
        else:
            if has_u:
                unpaired["upper_only"].append(d["upper"][0])
            if has_l:
                unpaired["lower_only"].append(d["lower"][0])
    report = {
        "total_cases": len(cases),
        "total_paired_cases": len(paired),
        "num_upper_only_pids": len(unpaired["upper_only"]),
        "num_lower_only_pids": len(unpaired["lower_only"]),
        "examples_upper_only": [{"pid": x["pid"], "case_id": x["case_id"]} for x in unpaired["upper_only"][:10]],
        "examples_lower_only": [{"pid": x["pid"], "case_id": x["case_id"]} for x in unpaired["lower_only"][:10]],
    }
    return paired, report

# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True, help="data/Teeth_3ds/processed_flat")
    ap.add_argument("--out_dir", required=True, help="data/Teeth_3ds/pointcloud_dataset")
    ap.add_argument("--points_per_cloud", type=int, default=0, help="Si 0 → usa P mínimo global")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--require_pairs", action="store_true")
    ap.add_argument("--id_regex", type=str, default=None)
    ap.add_argument("--split_char", type=str, default=None)
    ap.add_argument("--split_index", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    dataset_dir = Path(args.dataset_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_cases = scan_cases(dataset_dir, args.id_regex, args.split_char, args.split_index)
    if not raw_cases:
        raise SystemExit(f"No se encontraron casos en {dataset_dir}")

    if args.require_pairs:
        cases, rep = filter_paired(raw_cases)
        print(f"[PAIRS] {json.dumps(rep, indent=2)}")
    else:
        cases = raw_cases

    if not cases:
        raise SystemExit("Sin casos después de aplicar filtros (require_pairs, etc.).")

    if args.points_per_cloud > 0:
        P = int(args.points_per_cloud)
        print(f"[INFO] Usando P={P} puntos por nube (forzado).")
    else:
        lengths = [np.load(c["points_path"]).shape[0] for c in cases]
        P = int(min(lengths))
        print(f"[INFO] Usando P mínimo global = {P} puntos por nube.")

    X_list, Y_list, meta_cases = [], [], []
    for c in cases:
        pts = np.load(c["points_path"])
        lbs = np.load(c["labels_path"]) if c["labels_path"] is not None else None
        ptsP, lbsP = pad_or_sample(pts, lbs, P, rng)
        if lbsP is None:
            lbsP = np.zeros(P, dtype=np.int64)
        X_list.append(ptsP.astype(np.float32))
        Y_list.append(lbsP.astype(np.int64))
        meta_cases.append({"jaw": c["jaw"], "case_id": c["case_id"], "pid": c["pid"]})

    X = np.stack(X_list, axis=0)
    Y = np.stack(Y_list, axis=0)
    N = X.shape[0]

    # Splits robustos para datasets pequeños
    idx = np.arange(N)
    rng.shuffle(idx)
    n_test = int(round(args.test_ratio * N))
    n_val = int(round(args.val_ratio * N))

    # Garantiza que val/test no devoren todo
    if n_test + n_val >= N:
        # deja al menos 1 para train (si se puede)
        n_test = min(n_test, max(0, N - 1))
        n_val = min(n_val, max(0, N - 1 - n_test))

    # Si hay suficientes muestras, asegura al menos 1 en val/test
    if N >= 3:
        n_test = max(1, n_test)
        n_val = max(1, n_val)
        if n_test + n_val >= N:
            n_val = max(1, min(n_val, N - 2))
            n_test = max(1, min(n_test, N - 1 - n_val))

    n_train = N - n_val - n_test
    if n_train <= 0:
        # último recurso: todo a train
        n_train, n_val, n_test = N, 0, 0

    i_tr = idx[:n_train]
    i_va = idx[n_train:n_train + n_val]
    i_te = idx[n_train + n_val:]

    # Guardado NPZ con claves "X"/"Y"
    def save_npz(path: Path, arr: np.ndarray, key="X"):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{key: arr})

    save_npz(out_dir / "X_train.npz", X[i_tr], key="X")
    save_npz(out_dir / "X_val.npz", X[i_va], key="X")
    save_npz(out_dir / "X_test.npz", X[i_te], key="X")
    np.savez_compressed(out_dir / "Y_train.npz", Y=Y[i_tr])
    np.savez_compressed(out_dir / "Y_val.npz", Y=Y[i_va])
    np.savez_compressed(out_dir / "Y_test.npz", Y=Y[i_te])

    meta = {
        "P": int(P),
        "N_total": int(N),
        "splits": {"train": len(i_tr), "val": len(i_va), "test": len(i_te)},
        "cases_train": [meta_cases[k] for k in i_tr.tolist()],
        "cases_val": [meta_cases[k] for k in i_va.tolist()],
        "cases_test": [meta_cases[k] for k in i_te.tolist()],
        "require_pairs": bool(args.require_pairs),
        "pid_extraction": {"id_regex": args.id_regex, "split_char": args.split_char, "split_index": args.split_index}
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("[OK] Guardado en:", out_dir)
    print("  - X_* / Y_* .npz y meta.json")

if __name__ == "__main__":
    main()
