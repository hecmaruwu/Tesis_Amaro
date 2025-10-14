#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
refine_tooth21_clustering.py

Grid search de clustering (DBSCAN y opcional HDBSCAN) sobre la región faltante
(upper_full vs upper_rec_21), para aislar el diente 21 removido y evaluar
métricas contra la región faltante "ground truth" (derivada de upper_rec_21).

Requisitos:
- numpy, sklearn (DBSCAN)
- hdbscan (opcional; si no está, se ignora)
- tensorflow (opcional; solo si quieres calentar el modelo; no es necesario)

Salidas:
- CSV global con resultados por combinación de hiperparámetros (grid_metrics.csv)
- CSV por paciente con el “mejor” resultado elegido (best_by_patient.csv)
- (Opcional) PLY del mejor clúster por paciente (si pasas --export_best_ply)
"""

import os, json, argparse, csv, itertools
from pathlib import Path
import numpy as np

# ---- Intentamos HDBSCAN (opcional) ----
_HAS_HDBSCAN = True
try:
    import hdbscan  # type: ignore
except Exception:
    _HAS_HDBSCAN = False

# ---- DBSCAN de sklearn ----
from sklearn.cluster import DBSCAN

# ===========================
# Utilidades de nube / PLY
# ===========================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_cloud(path: Path) -> np.ndarray:
    pts = np.load(path).astype(np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Esperaba (N,3) en {path}, obtuve {pts.shape}")
    return pts

def sample_to_n(pts: np.ndarray, n_points: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = pts.shape[0]
    if n == n_points:
        return pts
    if n > n_points:
        idx = rng.choice(n, size=n_points, replace=False)
        return pts[idx]
    reps = int(np.ceil(n_points / n))
    tiled = np.tile(pts, (reps, 1))
    idx = rng.choice(tiled.shape[0], size=n_points, replace=False)
    return tiled[idx]

def save_ply(points: np.ndarray, colors: np.ndarray, path: Path):
    ensure_dir(path.parent)
    N = points.shape[0]
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors.astype(np.uint8)):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")

# ===========================
# Distancias (sin scipy)
# ===========================
def chunked_nn_dists(A: np.ndarray, B: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """d_min(A->B) por punto (dist mínima)."""
    if B.shape[0] == 0:
        return np.full((A.shape[0],), np.inf, dtype=np.float32)
    B2 = np.sum(B * B, axis=1)  # (M,)
    d_min = np.full((A.shape[0],), np.inf, dtype=np.float32)
    for i in range(0, A.shape[0], chunk):
        a = A[i:i+chunk]
        a2 = np.sum(a*a, axis=1, keepdims=True)  # (c,1)
        G = a @ B.T                              # (c,M)
        d2 = a2 + B2[None, :] - 2.0 * G
        d2 = np.maximum(d2, 0.0)
        d = np.sqrt(np.min(d2, axis=1))
        d_min[i:i+chunk] = d
    return d_min

def chamfer_bidirectional(P: np.ndarray, G: np.ndarray) -> float:
    """Chamfer simple: mean(d(P->G)) + mean(d(G->P))."""
    if P.shape[0] == 0 or G.shape[0] == 0:
        return float('inf')
    d_pg = chunked_nn_dists(P, G)
    d_gp = chunked_nn_dists(G, P)
    return float(d_pg.mean() + d_gp.mean())

def prf1_by_match(P: np.ndarray, G: np.ndarray, tau: float) -> tuple[float,float,float]:
    """Precision/Recall/F1 por matching con umbral tau."""
    if P.shape[0] == 0 and G.shape[0] == 0:
        return 1.0, 1.0, 1.0
    if P.shape[0] == 0:
        return 0.0, 0.0, 0.0
    if G.shape[0] == 0:
        return 0.0, 1.0, 0.0
    d_pg = chunked_nn_dists(P, G)  # para precision
    d_gp = chunked_nn_dists(G, P)  # para recall
    tp = float(np.sum(d_pg <= tau))
    precision = tp / max(1.0, float(P.shape[0]))
    tp_g = float(np.sum(d_gp <= tau))
    recall = tp_g / max(1.0, float(G.shape[0]))
    f1 = 0.0 if (precision+recall)==0 else 2.0*precision*recall/(precision+recall)
    return precision, recall, f1

# ===========================
# Localización de archivos
# ===========================
def find_upper_full(struct_root: Path, pid: str) -> Path:
    candidates = [
        struct_root/"upper"/f"{pid}_full"/"point_cloud.npy",
        struct_root/"upper"/f"{pid}_upper_full"/"point_cloud.npy",
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(f"No upper_full para {pid}: {candidates}")

def find_upper_rec21(struct_root: Path, pid: str) -> Path:
    candidates = [
        struct_root/"upper"/f"{pid}_upper_rec_21"/"point_cloud.npy",
        struct_root/"upper"/f"{pid}_rec_21"/"point_cloud.npy",
        struct_root/"upper"/f"{pid}_21"/"point_cloud.npy",
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(f"No upper_rec_21 para {pid}: {candidates}")

# ===========================
# Clustering helpers
# ===========================
def cluster_dbscan(X: np.ndarray, eps: float, min_samples: int):
    lab = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    return lab

def cluster_hdbscan(X: np.ndarray, min_cluster_size: int, min_samples: int | None):
    if not _HAS_HDBSCAN:
        raise RuntimeError("hdbscan no está instalado")
    cl = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                         min_samples=min_samples if min_samples is not None else None)
    lab = cl.fit_predict(X)
    return lab

def extract_cluster_points(X: np.ndarray, labels: np.ndarray, strategy: str = "largest") -> np.ndarray:
    """
    strategy:
      - 'largest': mayor número de puntos (recomendado y robusto).
      - (puedes añadir otras estrategias según geometría si lo necesitas)
    """
    mask_valid = labels >= 0
    if not np.any(mask_valid):
        return X[:0]
    labs, counts = np.unique(labels[mask_valid], return_counts=True)
    if strategy == "largest":
        pick = labs[np.argmax(counts)]
    else:
        pick = labs[np.argmax(counts)]
    return X[labels == pick]

# ===========================
# Pipeline por paciente
# ===========================
def get_missing_sets(upper_full_path: Path, upper_rec21_path: Path, n_points: int, removed_thresh: float):
    X_full = sample_to_n(load_cloud(upper_full_path), n_points)
    X_rec  = sample_to_n(load_cloud(upper_rec21_path), n_points)
    # Candidatos a "missing": puntos de full lejos de recortado
    d_full_to_rec = chunked_nn_dists(X_full, X_rec)
    miss_mask = d_full_to_rec > removed_thresh
    X_cand = X_full[miss_mask]
    # Ground truth (para métricas): el mismo missing_mask (proxy gt)
    X_gt = X_cand.copy()
    return X_full, X_cand, X_gt

def run_patient_grid(pid: str, struct_root: Path,
                     n_points: int, removed_thresh: float,
                     method_list: list[str],
                     dbscan_eps_list: list[float], dbscan_min_list: list[int],
                     hdbscan_min_cluster_size: list[int], hdbscan_min_samples: list[int],
                     match_tau: float,
                     out_best_ply_dir: Path | None) -> tuple[list[dict], dict]:

    up_full = find_upper_full(struct_root, pid)
    up_rec  = find_upper_rec21(struct_root, pid)
    X_full, X_cand, X_gt = get_missing_sets(up_full, up_rec, n_points, removed_thresh)

    results = []
    best = {"score": float('inf'), "row": None}  # minimiza Chamfer

    # Si no hay candidatos, devolvemos vacío
    if X_cand.shape[0] == 0:
        return results, {"patient_id": pid, "note": "no_missing_candidates"}

    # ---------- DBSCAN grid ----------
    if "dbscan" in method_list:
        for eps, ms in itertools.product(dbscan_eps_list, dbscan_min_list):
            labels = cluster_dbscan(X_cand, eps=eps, min_samples=ms)
            X_sel = extract_cluster_points(X_cand, labels, strategy="largest")
            ch = chamfer_bidirectional(X_sel, X_gt)
            p, r, f1 = prf1_by_match(X_sel, X_gt, tau=match_tau)
            row = {
                "patient_id": pid, "method": "dbscan",
                "eps": eps, "min_samples": ms,
                "min_cluster_size": "", "hdb_min_samples": "",
                "n_pred": int(X_sel.shape[0]),
                "chamfer": ch, "precision": p, "recall": r, "f1": f1
            }
            results.append(row)
            if ch < best["score"]:
                best["score"] = ch
                best["row"] = row
                best["best_points"] = X_sel

    # ---------- HDBSCAN grid (opcional) ----------
    if _HAS_HDBSCAN and ("hdbscan" in method_list):
        for mcs, ms in itertools.product(hdbscan_min_cluster_size, hdbscan_min_samples or [None]):
            try:
                labels = cluster_hdbscan(X_cand, min_cluster_size=mcs, min_samples=ms)
            except Exception as e:
                # si falla por params, continúa
                continue
            X_sel = extract_cluster_points(X_cand, labels, strategy="largest")
            ch = chamfer_bidirectional(X_sel, X_gt)
            p, r, f1 = prf1_by_match(X_sel, X_gt, tau=match_tau)
            row = {
                "patient_id": pid, "method": "hdbscan",
                "eps": "", "min_samples": "",
                "min_cluster_size": mcs, "hdb_min_samples": (ms if ms is not None else ""),
                "n_pred": int(X_sel.shape[0]),
                "chamfer": ch, "precision": p, "recall": r, "f1": f1
            }
            results.append(row)
            if ch < best["score"]:
                best["score"] = ch
                best["row"] = row
                best["best_points"] = X_sel

    # Export del mejor clúster como PLY (opcional)
    if out_best_ply_dir is not None and best.get("best_points") is not None:
        X_sel = best["best_points"]
        col = np.tile(np.array([255, 80, 80], dtype=np.uint8), (X_sel.shape[0], 1))  # rojo
        save_ply(X_sel, col, out_best_ply_dir/pid/f"{pid}_tooth21_best.ply")

    return results, best["row"] if best["row"] is not None else {"patient_id": pid, "note": "no_best"}

# ===========================
# CLI / Main
# ===========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ufrn_root", required=True)
    ap.add_argument("--struct_rel", default="processed_struct")
    ap.add_argument("--patients", default="all",
                    help="'all' o lista separada por comas: paciente_19,paciente_20")
    ap.add_argument("--n_points", type=int, default=8192)
    ap.add_argument("--removed_thresh", type=float, default=0.02,
                    help="Umbral para definir candidatos a 'missing' (mismo criterio que diagnóstico).")

    # Grid DBSCAN
    ap.add_argument("--dbscan_eps", default="0.015,0.02,0.025")
    ap.add_argument("--dbscan_min", default="10,20,30")

    # Grid HDBSCAN (opcional)
    ap.add_argument("--use_hdbscan", action="store_true")
    ap.add_argument("--hdbscan_min_cluster_size", default="40,60,90")
    ap.add_argument("--hdbscan_min_samples", default="5,10")

    # Métrica de matching
    ap.add_argument("--match_tau", type=float, default=0.01,
                    help="Umbral (en coords normalizadas) para TP/FP/FN.")

    # Salidas
    ap.add_argument("--out_dir", default="refine_tooth21")
    ap.add_argument("--export_best_ply", action="store_true",
                    help="Exporta el mejor clúster por paciente como PLY.")
    args = ap.parse_args()

    ufrn_root = Path(args.ufrn_root)
    struct_root = ufrn_root / args.struct_rel
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # Parse de listas
    db_eps = [float(x) for x in args.dbscan_eps.split(",") if x.strip()]
    db_min = [int(x) for x in args.dbscan_min.split(",") if x.strip()]
    use_hdb = bool(args.use_hdbscan) and _HAS_HDBSCAN
    hdb_mcs = [int(x) for x in args.hdbscan_min_cluster_size.split(",") if x.strip()]
    hdb_ms  = [int(x) for x in args.hdbscan_min_samples.split(",") if x.strip()]

    # Pacientes
    if args.patients.lower() != "all":
        pids = [p.strip() for p in args.patients.split(",") if p.strip()]
    else:
        csv_path = ufrn_root/"ufrn_por_paciente.csv"
        rows = [l.strip().split(",")[0] for l in csv_path.read_text(encoding="utf-8").splitlines()[1:] if l.strip()]
        pids = sorted(set(rows))

    methods = ["dbscan"]
    if use_hdb:
        methods.append("hdbscan")

    all_rows = []
    best_rows = []
    out_best_ply = out_dir/"ply" if args.export_best_ply else None

    print(f"[INFO] Pacientes: {len(pids)} | métodos: {methods}")
    for pid in pids:
        rows, best = run_patient_grid(
            pid, struct_root,
            n_points=args.n_points, removed_thresh=args.removed_thresh,
            method_list=methods,
            dbscan_eps_list=db_eps, dbscan_min_list=db_min,
            hdbscan_min_cluster_size=hdb_mcs, hdbscan_min_samples=hdb_ms,
            match_tau=args.match_tau,
            out_best_ply_dir=out_best_ply
        )
        all_rows.extend(rows)
        best_rows.append(best)

    # CSV global (toda la grilla)
    grid_csv = out_dir/"grid_metrics.csv"
    with open(grid_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "patient_id","method","eps","min_samples","min_cluster_size","hdb_min_samples",
            "n_pred","chamfer","precision","recall","f1"
        ])
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"[CSV] Grid completo: {grid_csv}")

    # CSV mejores por paciente
    best_csv = out_dir/"best_by_patient.csv"
    with open(best_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "patient_id","method","eps","min_samples","min_cluster_size","hdb_min_samples",
            "n_pred","chamfer","precision","recall","f1","note"
        ])
        w.writeheader()
        for r in best_rows:
            if r is None:
                continue
            if "note" not in r:
                r["note"] = ""
            w.writerow(r)
    print(f"[CSV] Mejores por paciente: {best_csv}")

    # Resumen simple en consola
    valid = [r for r in best_rows if r and "chamfer" in r]
    if valid:
        mean_ch = float(np.mean([r["chamfer"] for r in valid]))
        mean_f1 = float(np.mean([r["f1"]      for r in valid]))
        print(f"[SUMMARY] best_per_patient -> Chamfer(mean)={mean_ch:.6f}  F1(mean)={mean_f1:.3f}")
    else:
        print("[SUMMARY] No hubo resultados válidos (¿no hubo candidatos?)")

if __name__ == "__main__":
    main()
