#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Construye un dataset "por paciente" (pid) combinando upper+lower (si existen),
y *opcionalmente* añade características geométricas a cada punto:
  - Normales (nx, ny, nz)
  - Curvatura (surface variation)

Entrada esperada (tipo processed_flat):
  dataset_dir/
    upper/<CASE_U>/{point_cloud.npy, labels.npy?}
    lower/<CASE_L>/{point_cloud.npy, labels.npy?}

Salida:
  out_dir/
    X_{train,val,test}.npz  (X: [N,P,C], C=3 o 7)
    Y_{train,val,test}.npz  (Y: [N,P], si hay etiquetas; fondo=0 si faltan)
    J_{train,val,test}.npz  (J: [N,P], 0=upper, 1=lower)
    meta.json
    index_{train,val,test}.csv
"""

import os
import re
import json
import csv
import argparse
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np

# SciPy KDTree para kNN rápido (evitamos O(N^2))
try:
    from scipy.spatial import cKDTree as KDTree
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# ----------------------------- Utilidades -----------------------------

def extract_pid(case_name: str,
                id_regex: Optional[str],
                split_char: Optional[str],
                split_index: int) -> str:
    if id_regex:
        m = re.search(id_regex, case_name)
        if m and m.groups():
            return m.group(1)
    if split_char is not None:
        parts = case_name.split(split_char)
        if 0 <= split_index < len(parts):
            return parts[split_index]
    return case_name


def list_cases(dataset_dir: Path, id_regex, split_char, split_index) -> Dict[str, Dict[str, List[Dict]]]:
    """Devuelve dict por pid: {'upper':[...], 'lower':[...]} con paths y case_id."""
    by_pid: Dict[str, Dict[str, List[Dict]]] = {}
    for jaw in ["upper", "lower"]:
        d = dataset_dir / jaw
        if not d.is_dir():
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


def load_np(path: str) -> np.ndarray:
    return np.load(path)


def normalize_cloud_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mean = x.mean(axis=0, keepdims=True)
    x = x - mean
    r = np.linalg.norm(x, axis=1, keepdims=True).max()
    return x / (r + 1e-8)


def sample_points(pts: np.ndarray, P: int, rng: np.random.Generator):
    N = pts.shape[0]
    idx = rng.choice(N, P, replace=(N < P))
    return pts[idx], idx


def _safe_labels_align(lbs: Optional[np.ndarray], pts_len: int) -> Optional[np.ndarray]:
    if lbs is None:
        return None
    lbs = lbs.reshape(-1)
    if lbs.shape[0] == pts_len:
        return lbs
    m = min(lbs.shape[0], pts_len)
    out = np.zeros((pts_len,), dtype=np.int64)
    out[:m] = lbs[:m]
    if m < pts_len:
        out[m:] = out[m-1] if m > 0 else 0
    return out


# ------------------- Geometría: normales y curvatura -------------------

def _require_scipy():
    if not _HAS_SCIPY:
        raise SystemExit(
            "Este script necesita SciPy para k-NN (scipy.spatial.cKDTree). "
            "Instala:  conda install -y scipy  (o)  pip install scipy"
        )

def estimate_normals_and_curvature(points_xyz: np.ndarray, k: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estima normales y curvatura por punto usando PCA local sobre k vecinos.
    Curvatura = λ_min / (λ1 + λ2 + λ3)  (surface variation)
    """
    _require_scipy()
    pts = points_xyz.astype(np.float32)
    n = pts.shape[0]
    k = int(max(3, min(k, n)))

    tree = KDTree(pts)
    # idx: (n, k) con los vecinos más cercanos (incluye el propio punto)
    _, idx = tree.query(pts, k=k)

    normals = np.zeros((n, 3), dtype=np.float32)
    curvature = np.zeros((n,), dtype=np.float32)

    for i in range(n):
        neigh = pts[idx[i]]  # (k, 3)
        c = neigh.mean(axis=0, keepdims=True)
        Q = neigh - c
        # Covarianza (3x3)
        C = (Q.T @ Q) / (Q.shape[0] - 1 + 1e-8)
        # Autovalores/Autovectores
        w, v = np.linalg.eigh(C)  # w ascendente
        # normal: eigenvector del menor autovalor
        nrm = v[:, 0]
        # curvatura: surface variation
        w_sum = float(np.clip(w.sum(), 1e-8, None))
        curv = float(np.clip(w[0] / w_sum, 0.0, 1.0))
        normals[i] = nrm
        curvature[i] = curv

    # Orientación de normales: opción simple (hacia +Z global)
    flip = normals[:, 2] < 0
    normals[flip] *= -1.0

    return normals, curvature


def maybe_add_geo_features(points: np.ndarray, add_geo: bool, k_nn: int) -> np.ndarray:
    """
    Si add_geo=True y puntos tienen 3 canales -> añade (nx,ny,nz,curv) => (N,7).
    Si ya tienen >=6 canales, se dejan como están (a menos que quieras forzar recomputar).
    """
    C = points.shape[-1]
    if not add_geo:
        return points

    if C >= 6:
        # Ya parecen tener normales; si también traen curvatura, respetamos.
        # Sólo normalizamos XYZ por consistencia al final del pipeline.
        return points

    # Caso típico: XYZ
    normals, curv = estimate_normals_and_curvature(points[:, :3], k=k_nn)
    feat = np.concatenate([points[:, :3], normals, curv[:, None]], axis=1).astype(np.float32)
    return feat


# -------------------- Construcción por paciente --------------------

def build_sample_for_pid(rec,
                        points_total: int,
                        balance_jaws: bool,
                        rng: np.random.Generator,
                        allow_unpaired: bool,
                        add_geo: bool,
                        k_nn: int):
    """
    Devuelve (points [P,C], labels [P] or None, jaw_mask [P]), meta por sample.
    points puede ser (P,3) o (P,7) si add_geo=True y había solo XYZ.
    """
    have_u = len(rec["upper"]) > 0
    have_l = len(rec["lower"]) > 0
    if not allow_unpaired and not (have_u and have_l):
        return None

    u = rec["upper"][0] if have_u else None
    l = rec["lower"][0] if have_l else None

    def load_case(case_entry):
        pts_all = load_np(case_entry["points_path"]).astype(np.float32)
        lbs_all = load_np(case_entry["labels_path"]).astype(np.int64) if case_entry["labels_path"] else None
        # Si vienen XYZ+algo, dejamos “algo”; si vienen sólo XYZ y add_geo=True, generamos normales/curvatura
        pts_all = maybe_add_geo_features(pts_all, add_geo=add_geo, k_nn=k_nn)
        # Normalizamos SIEMPRE XYZ (canales 0:3) a esfera unitaria por paciente
        pts_all[:, :3] = normalize_cloud_np(pts_all[:, :3])
        if lbs_all is not None:
            lbs_all = _safe_labels_align(lbs_all, pts_all.shape[0])
        return pts_all, lbs_all

    # Cargar casos
    pts_u = lbs_u = None
    if u is not None:
        pts_u, lbs_u = load_case(u)

    pts_l = lbs_l = None
    if l is not None:
        pts_l, lbs_l = load_case(l)

    # Asignación de puntos por jaw
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

        # Muestrear
        idx_u = rng.choice(pts_u.shape[0], Pu, replace=(pts_u.shape[0] < Pu))
        idx_l = rng.choice(pts_l.shape[0], Pl, replace=(pts_l.shape[0] < Pl))
        P_u = pts_u[idx_u]
        P_l = pts_l[idx_l]
        L_u = (lbs_u[idx_u] if lbs_u is not None else None)
        L_l = (lbs_l[idx_l] if lbs_l is not None else None)

        pts = np.concatenate([P_u, P_l], axis=0)
        C = pts.shape[1]
        if (L_u is not None) or (L_l is not None):
            if L_u is None: L_u = np.zeros((Pu,), dtype=np.int64)
            if L_l is None: L_l = np.zeros((Pl,), dtype=np.int64)
            lbs = np.concatenate([L_u, L_l], axis=0)
        else:
            lbs = None
        jaw_mask = np.concatenate([np.zeros(Pu, dtype=np.int64), np.ones(Pl, dtype=np.int64)], axis=0)

    else:
        single = u if have_u else l
        pts_all, lbs_all = load_case(single)
        idx_s = rng.choice(pts_all.shape[0], points_total, replace=(pts_all.shape[0] < points_total))
        pts = pts_all[idx_s]
        lbs = (lbs_all[idx_s] if lbs_all is not None else None)
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


# ----------------------------- MAIN -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True, help="Carpeta con upper/ y lower/ (processed_flat)")
    ap.add_argument("--out_dir",     required=True, help="Carpeta destino del dataset mergeado")
    ap.add_argument("--points_total", type=int, default=50000, help="Puntos totales por paciente")
    ap.add_argument("--balance_jaws", action="store_true", help="Repartir P/2 y P/2 si hay upper+lower")
    ap.add_argument("--allow_unpaired", action="store_true", help="Permite pacientes con 1 jaw disponible")
    ap.add_argument("--id_regex", default=None)
    ap.add_argument("--split_char", default=None)
    ap.add_argument("--split_index", type=int, default=0)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)

    # >>> NUEVO: generar características geométricas
    ap.add_argument("--with_geo", action="store_true",
                    help="Si se activa y el cloud trae sólo XYZ, añade (nx,ny,nz,curvatura) => 7 canales")
    ap.add_argument("--k_normals", type=int, default=30, help="k vecinos para normales/curvatura")

    args = ap.parse_args()

    # Requisito SciPy si se pidió geo
    if args.with_geo and not _HAS_SCIPY:
        _require_scipy()

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

        sample = build_sample_for_pid(rec,
                                      points_total=args.points_total,
                                      balance_jaws=args.balance_jaws,
                                      rng=rng,
                                      allow_unpaired=args.allow_unpaired,
                                      add_geo=args.with_geo,
                                      k_nn=args.k_normals)
        if sample is None:
            continue

        pts, lbs, jaw_mask, meta = sample
        X_list.append(pts.astype(np.float32))
        J_list.append(jaw_mask.astype(np.int64))
        if lbs is None:
            lbs = np.zeros((pts.shape[0],), dtype=np.int64)
        Y_list.append(lbs.astype(np.int64))
        meta_cases.append({"pid": pid, **meta})

    if len(X_list) == 0:
        raise SystemExit("No se pudo construir ninguna muestra. Revisa flags/pairs/formato de entrada.")

    X = np.stack(X_list, axis=0)   # [N,P,C]
    Y = np.stack(Y_list, axis=0)   # [N,P]
    J = np.stack(J_list, axis=0)   # [N,P]
    N, P, C = X.shape
    print(f"[OK] Dataset: N={N}  P={P}  C={C}  (C=3 → XYZ; C=7 → XYZ+normales+curv)")

    # Splits robustos por paciente
    idx = np.arange(N)
    rng.shuffle(idx)
    n_test = int(round(args.test_ratio * N))
    n_val = int(round(args.val_ratio * N))
    if n_test + n_val >= N:
        n_test = min(n_test, max(0, N - 1))
        n_val = min(n_val, max(0, N - 1 - n_test))
    if N >= 3:
        n_test = max(1, n_test)
        n_val = max(1, n_val)
        if n_test + n_val >= N:
            n_val = max(1, min(n_val, N - 2))
            n_test = max(1, min(n_test, N - 1 - n_val))
    n_train = N - n_val - n_test
    if n_train <= 0:
        n_train, n_val, n_test = N, 0, 0

    Itr, Iva, Ite = idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]

    def subset(arr, I): return arr[I]

    # Guardar NPZ
    save_npz(out_dir / "X_train.npz", "X", subset(X, Itr))
    save_npz(out_dir / "X_val.npz",   "X", subset(X, Iva))
    save_npz(out_dir / "X_test.npz",  "X", subset(X, Ite))
    save_npz(out_dir / "Y_train.npz", "Y", subset(Y, Itr))
    save_npz(out_dir / "Y_val.npz",   "Y", subset(Y, Iva))
    save_npz(out_dir / "Y_test.npz",  "Y", subset(Y, Ite))
    save_npz(out_dir / "J_train.npz", "J", subset(J, Itr))
    save_npz(out_dir / "J_val.npz",   "J", subset(J, Iva))
    save_npz(out_dir / "J_test.npz",  "J", subset(J, Ite))

    meta = {
        "points_total": int(P),
        "features": int(C),
        "with_geo": bool(args.with_geo),
        "k_normals": int(args.k_normals) if args.with_geo else None,
        "balance_jaws": bool(args.balance_jaws),
        "allow_unpaired": bool(args.allow_unpaired),
        "N_total": int(N),
        "splits": {"train": len(Itr), "val": len(Iva), "test": len(Ite)},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    idx_rows = [{"idx": int(i), **meta_cases[i]} for i in range(N)]
    idx_train = [idx_rows[i] for i in Itr]
    idx_val   = [idx_rows[i] for i in Iva]
    idx_test  = [idx_rows[i] for i in Ite]
    write_index_csv(out_dir / "index_train.csv", idx_train)
    write_index_csv(out_dir / "index_val.csv",   idx_val)
    write_index_csv(out_dir / "index_test.csv",  idx_test)

    print("[DONE] Guardado en:", str(out_dir))
    print("  - X_*.npz, Y_*.npz, J_*.npz, meta.json, index_*.csv")


if __name__ == "__main__":
    main()
