#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
refine_tooth21_allinone.py

Post-procesamiento por clustering (DBSCAN/HDBSCAN) de la región faltante (tooth 21)
en UFRN. Incluye:
- Grid search interno (y sobreescribible por CLI)
- Métricas detalladas (Chamfer, Hausdorff, P/R/F1, IoU, centroid dist, AABB IoU/volumen)
- Selección mejor por paciente
- Selección "mejor global" (max F1 medio) y re-export (PLY)
- Plots con Plotly (opcional)

Dependencias mínimas: numpy, pandas, scikit-learn
Opcionales: hdbscan, plotly

Uso típico:
  export PYTHONPATH="$PWD:$PYTHONPATH"
  UFRN=/home/htaucare/Tesis_dientes_original/data/UFRN

  # DBSCAN solo (sin HDBSCAN)
  python scripts/refine_tooth21_allinone.py \
    --ufrn_root "$UFRN" \
    --struct_rel processed_struct \
    --out_dir refine_allinone_dbscan \
    --n_points 8192 --removed_thresh 0.02 --match_tau 0.01 \
    --export_best_ply --max_patients_3d 12

  # DBSCAN + HDBSCAN (si está instalado hdbscan)
  python scripts/refine_tooth21_allinone.py \
    --ufrn_root "$UFRN" \
    --struct_rel processed_struct \
    --out_dir refine_allinone_hdb \
    --n_points 8192 --removed_thresh 0.02 --match_tau 0.01 \
    --use_hdbscan --export_best_ply --max_patients_3d 12
"""

import os, argparse, json, itertools, re
from pathlib import Path
import numpy as np
import pandas as pd

# --- opcional plotly ---
_HAS_PLOTLY = True
try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    _HAS_PLOTLY = False

# --- opcional hdbscan ---
_HAS_HDBSCAN = True
try:
    import hdbscan  # type: ignore
except Exception:
    _HAS_HDBSCAN = False

from sklearn.cluster import DBSCAN

# =========================
# CONFIG por defecto (editable aquí)
# =========================
DEFAULT_DBSCAN_EPS = [0.015, 0.02, 0.025]
DEFAULT_DBSCAN_MIN = [10, 20, 30]

DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE = [40, 60, 90]
DEFAULT_HDBSCAN_MIN_SAMPLES      = [5, 10]

# =========================
# Utilidades de FS y PLY
# =========================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_json(obj, path: Path):
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

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

# =========================
# Carga / sampleo de nubes
# =========================
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

# =========================
# Distancias y métricas
# =========================
def chunked_nn_dists(A: np.ndarray, B: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """Dist mínima por punto de A hacia B."""
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
    if P.shape[0] == 0 or G.shape[0] == 0:
        return float('inf')
    d_pg = chunked_nn_dists(P, G)
    d_gp = chunked_nn_dists(G, P)
    return float(d_pg.mean() + d_gp.mean())

def hausdorff_bidirectional(P: np.ndarray, G: np.ndarray) -> float:
    if P.shape[0] == 0 or G.shape[0] == 0:
        return float('inf')
    d_pg = chunked_nn_dists(P, G)
    d_gp = chunked_nn_dists(G, P)
    return float(max(d_pg.max(), d_gp.max()))

def prf1_iou(P: np.ndarray, G: np.ndarray, tau: float):
    """TP/FP/FN por matching con umbral.
       IoU = TP/(TP+FP+FN). Accuracy no es bien definida (no hay TN).
    """
    if P.shape[0] == 0 and G.shape[0] == 0:
        return 1.0, 1.0, 1.0, 1.0
    if P.shape[0] == 0:
        return 0.0, 0.0, 0.0, 0.0
    if G.shape[0] == 0:
        return 0.0, 1.0, 0.0, 0.0
    d_pg = chunked_nn_dists(P, G)  # para precision
    d_gp = chunked_nn_dists(G, P)  # para recall
    tp  = float(np.sum(d_pg <= tau))
    fp  = float(P.shape[0]) - tp
    tp2 = float(np.sum(d_gp <= tau))
    fn  = float(G.shape[0]) - tp2
    # consolidamos TP como el mínimo de ambos contajes
    tp_final = min(tp, tp2)
    precision = tp_final / max(1.0, float(P.shape[0]))
    recall    = tp_final / max(1.0, float(G.shape[0]))
    f1 = 0.0 if (precision+recall)==0 else 2.0*precision*recall/(precision+recall)
    iou = tp_final / max(1.0, tp_final + fp + fn)
    return precision, recall, f1, iou

def aabb(points: np.ndarray):
    if points.shape[0] == 0:
        return None
    mn = points.min(axis=0)
    mx = points.max(axis=0)
    ctr = 0.5*(mn+mx)
    vol = float(np.prod(np.maximum(mx-mn, 0.0)))
    return {"min": mn, "max": mx, "center": ctr, "volume": vol}

def aabb_iou3d(b1, b2) -> float:
    if b1 is None or b2 is None:
        return 0.0
    mn = np.maximum(b1["min"], b2["min"])
    mx = np.minimum(b1["max"], b2["max"])
    side = np.maximum(mx - mn, 0.0)
    inter = float(np.prod(side))
    union = b1["volume"] + b2["volume"] - inter
    return inter/union if union > 0 else 0.0

# =========================
# Localización de archivos
# =========================
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

# =========================
# Candidatos "missing"
# =========================
def get_missing_sets(upper_full_path: Path, upper_rec21_path: Path, n_points: int, removed_thresh: float):
    X_full = sample_to_n(load_cloud(upper_full_path), n_points)
    X_rec  = sample_to_n(load_cloud(upper_rec21_path), n_points)
    d_full_to_rec = chunked_nn_dists(X_full, X_rec)
    miss_mask = d_full_to_rec > removed_thresh
    X_cand = X_full[miss_mask]
    X_gt   = X_cand.copy()  # proxy de ground truth
    return X_full, X_cand, X_gt

# =========================
# Clustering
# =========================
def run_dbscan(X: np.ndarray, eps: float, min_samples: int):
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    return labels

def run_hdbscan(X: np.ndarray, min_cluster_size: int, min_samples: int | None):
    if not _HAS_HDBSCAN:
        raise RuntimeError("hdbscan no está instalado")
    cl = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                         min_samples=min_samples if min_samples is not None else None)
    labels = cl.fit_predict(X)
    return labels

def pick_cluster(X: np.ndarray, labels: np.ndarray, strategy="largest") -> np.ndarray:
    valid = labels >= 0
    if not np.any(valid):
        return X[:0]
    labs, cnt = np.unique(labels[valid], return_counts=True)
    if strategy == "largest":
        sel = labs[np.argmax(cnt)]
    else:
        sel = labs[np.argmax(cnt)]
    return X[labels == sel]

# =========================
# Una pasada por paciente
# =========================
def eval_one(pid: str, struct_root: Path,
             n_points: int, removed_thresh: float,
             method_list: list[str],
             grid_db_eps: list[float], grid_db_min: list[int],
             grid_hdb_mcs: list[int], grid_hdb_ms: list[int],
             match_tau: float,
             out_best_dir: Path | None):
    up_full = find_upper_full(struct_root, pid)
    up_rec  = find_upper_rec21(struct_root, pid)
    X_full, X_cand, X_gt = get_missing_sets(up_full, up_rec, n_points, removed_thresh)

    rows = []
    best = {"score": float('inf'), "row": None, "points": None}

    if X_cand.shape[0] == 0:
        return rows, {"patient_id": pid, "note": "no_missing_candidates"}

    # DBSCAN
    if "dbscan" in method_list:
        for eps, ms in itertools.product(grid_db_eps, grid_db_min):
            lab = run_dbscan(X_cand, eps=eps, min_samples=ms)
            P = pick_cluster(X_cand, lab, "largest")
            ch = chamfer_bidirectional(P, X_gt)
            hd = hausdorff_bidirectional(P, X_gt)
            p, r, f1, iou = prf1_iou(P, X_gt, tau=match_tau)
            # bbox
            bP = aabb(P); bG = aabb(X_gt)
            bb_iou = aabb_iou3d(bP, bG)
            ctr_dist = float(np.linalg.norm((bP["center"]-bG["center"])) if bP and bG else np.inf)
            row = {"patient_id": pid, "method": "dbscan",
                   "eps": eps, "min_samples": ms,
                   "min_cluster_size": "", "hdb_min_samples": "",
                   "n_pred": int(P.shape[0]),
                   "chamfer": ch, "hausdorff": hd,
                   "precision": p, "recall": r, "f1": f1, "iou": iou,
                   "bbox_iou": bb_iou,
                   "bbox_vol_pred": (bP["volume"] if bP else 0.0),
                   "bbox_vol_gt":   (bG["volume"] if bG else 0.0),
                   "bbox_ctr_dist": ctr_dist}
            rows.append(row)
            if ch < best["score"]:
                best.update(score=ch, row=row, points=P)

    # HDBSCAN
    if _HAS_HDBSCAN and ("hdbscan" in method_list):
        for mcs, ms in itertools.product(grid_hdb_mcs, grid_hdb_ms or [None]):
            try:
                lab = run_hdbscan(X_cand, min_cluster_size=mcs, min_samples=ms)
            except Exception:
                continue
            P = pick_cluster(X_cand, lab, "largest")
            ch = chamfer_bidirectional(P, X_gt)
            hd = hausdorff_bidirectional(P, X_gt)
            p, r, f1, iou = prf1_iou(P, X_gt, tau=match_tau)
            bP = aabb(P); bG = aabb(X_gt)
            bb_iou = aabb_iou3d(bP, bG)
            ctr_dist = float(np.linalg.norm((bP["center"]-bG["center"])) if bP and bG else np.inf)
            row = {"patient_id": pid, "method": "hdbscan",
                   "eps": "", "min_samples": "",
                   "min_cluster_size": mcs, "hdb_min_samples": (ms if ms is not None else ""),
                   "n_pred": int(P.shape[0]),
                   "chamfer": ch, "hausdorff": hd,
                   "precision": p, "recall": r, "f1": f1, "iou": iou,
                   "bbox_iou": bb_iou,
                   "bbox_vol_pred": (bP["volume"] if bP else 0.0),
                   "bbox_vol_gt":   (bG["volume"] if bG else 0.0),
                   "bbox_ctr_dist": ctr_dist}
            rows.append(row)
            if ch < best["score"]:
                best.update(score=ch, row=row, points=P)

    # export mejor por paciente
    if out_best_dir is not None and best["points"] is not None:
        col = np.tile(np.array([255, 80, 80], dtype=np.uint8), (best["points"].shape[0], 1))
        save_ply(best["points"], col, out_best_dir/pid/f"{pid}_tooth21_best.ply")

    return rows, best["row"] if best["row"] is not None else {"patient_id": pid, "note": "no_best"}

# =========================
# Plots (opcional)
# =========================
def save_plot(fig, path: Path):
    ensure_dir(path.parent)
    fig.write_html(str(path), include_plotlyjs="cdn")

def plot_heatmaps(grid: pd.DataFrame, out_dir: Path):
    if not _HAS_PLOTLY: return
    # DBSCAN heatmap F1
    g = grid[grid["method"]=="dbscan"].copy()
    if not g.empty:
        g["eps"] = pd.to_numeric(g["eps"], errors="coerce")
        g["min_samples"] = pd.to_numeric(g["min_samples"], errors="coerce")
        pv = g.pivot_table(index="eps", columns="min_samples", values="f1", aggfunc="mean")
        fig = px.imshow(pv, text_auto=".2f", aspect="auto", title="DBSCAN — F1 medio")
        save_plot(fig, out_dir/"figs/dbscan_heatmap_f1.html")
    # HDBSCAN heatmap F1
    h = grid[grid["method"]=="hdbscan"].copy()
    if not h.empty:
        h["min_cluster_size"] = pd.to_numeric(h["min_cluster_size"], errors="coerce")
        h["hdb_min_samples"]  = pd.to_numeric(h["hdb_min_samples"],  errors="coerce")
        pv = h.pivot_table(index="min_cluster_size", columns="hdb_min_samples", values="f1", aggfunc="mean")
        fig = px.imshow(pv, text_auto=".2f", aspect="auto", title="HDBSCAN — F1 medio")
        save_plot(fig, out_dir/"figs/hdbscan_heatmap_f1.html")

def plot_histograms(best: pd.DataFrame, out_dir: Path):
    if not _HAS_PLOTLY: return
    for col, ttl in [("chamfer","Refined Chamfer"), ("hausdorff","Refined Hausdorff"),
                     ("f1","Refined F1"), ("iou","Refined IoU")]:
        if col in best.columns:
            fig = px.histogram(best, x=col, nbins=30, title=ttl)
            save_plot(fig, out_dir/f"figs/hist_{col}.html")

def plot_3d_subset(best: pd.DataFrame, struct_root: Path, refine_best_dir: Path, out_dir: Path, max_n: int = 12):
    if not _HAS_PLOTLY: return
    def parse_ply_ascii(ply_path: Path) -> np.ndarray:
        with open(ply_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip()=="end_header": break
            pts=[]
            for line in f:
                if not line.strip(): continue
                s=line.split()
                if len(s)<3: continue
                pts.append((float(s[0]),float(s[1]),float(s[2])))
        return np.array(pts, dtype=np.float32)

    def find_upper_full(struct_root: Path, pid: str):
        cands = [
            struct_root/"upper"/f"{pid}_full"/"point_cloud.npy",
            struct_root/"upper"/f"{pid}_upper_full"/"point_cloud.npy",
        ]
        for c in cands:
            if c.is_file(): return c
        return None

    picked = best.sort_values("chamfer", ascending=False)["patient_id"].tolist()[:max_n]
    for pid in picked:
        npy = find_upper_full(struct_root, pid)
        ply = refine_best_dir/pid/f"{pid}_tooth21_best.ply"
        if npy is None or not ply.is_file(): continue
        X = np.load(npy).astype(np.float32)
        if X.shape[0] > 8192:
            idx = np.random.default_rng(42).choice(X.shape[0], 8192, replace=False)
            X = X[idx]
        P = parse_ply_ascii(ply)

        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=X[:,0],y=X[:,1],z=X[:,2],
                                   mode='markers', marker=dict(size=2, color="#B0B0B0", opacity=0.4),
                                   name="upper_full"))
        fig.add_trace(go.Scatter3d(x=P[:,0],y=P[:,1],z=P[:,2],
                                   mode='markers', marker=dict(size=3, color="#E74C3C", opacity=0.9),
                                   name="best_cluster (21)"))
        fig.update_layout(title=f"{pid} — upper_full + best_cluster", height=700)
        save_plot(fig, out_dir/f"figs/{pid}_3d.html")

# =========================
# Mejor config global
# =========================
def select_global_best(grid: pd.DataFrame):
    """Devuelve (method, params_dict) con mejor F1 medio global."""
    # DBSCAN
    db = grid[grid["method"]=="dbscan"].copy()
    best_db = None
    if not db.empty:
        db["eps"] = pd.to_numeric(db["eps"], errors="coerce")
        db["min_samples"] = pd.to_numeric(db["min_samples"], errors="coerce")
        grp = db.groupby(["eps","min_samples"])["f1"].mean().reset_index()
        if not grp.empty:
            best_db = grp.sort_values("f1", ascending=False).iloc[0].to_dict()

    # HDBSCAN
    hb = grid[grid["method"]=="hdbscan"].copy()
    best_hb = None
    if not hb.empty:
        hb["min_cluster_size"] = pd.to_numeric(hb["min_cluster_size"], errors="coerce")
        hb["hdb_min_samples"]  = pd.to_numeric(hb["hdb_min_samples"], errors="coerce")
        grp = hb.groupby(["min_cluster_size","hdb_min_samples"])["f1"].mean().reset_index()
        if not grp.empty:
            best_hb = grp.sort_values("f1", ascending=False).iloc[0].to_dict()

    # elegir mejor entre ambos
    if best_db is None and best_hb is None:
        return None

    if best_hb is None or (best_db is not None and best_db["f1"] >= best_hb["f1"]):
        return ("dbscan", {"eps": float(best_db["eps"]), "min_samples": int(best_db["min_samples"]),
                           "f1_mean": float(best_db["f1"])})
    else:
        return ("hdbscan", {"min_cluster_size": int(best_hb["min_cluster_size"]),
                            "min_samples": int(best_hb["hdb_min_samples"]),
                            "f1_mean": float(best_hb["f1"])})

def rerun_global(method: str, params: dict, pids: list[str], struct_root: Path,
                 n_points: int, removed_thresh: float, match_tau: float,
                 out_dir: Path):
    rows=[]
    ply_dir = out_dir/"global_best_ply"
    for pid in pids:
        up_full = find_upper_full(struct_root, pid)
        up_rec  = find_upper_rec21(struct_root, pid)
        X_full, X_cand, X_gt = get_missing_sets(up_full, up_rec, n_points, removed_thresh)
        if X_cand.shape[0]==0:
            rows.append({"patient_id": pid, "note":"no_missing_candidates"})
            continue
        if method=="dbscan":
            lab = run_dbscan(X_cand, eps=params["eps"], min_samples=params["min_samples"])
        else:
            lab = run_hdbscan(X_cand, min_cluster_size=params["min_cluster_size"], min_samples=params["min_samples"])
        P = pick_cluster(X_cand, lab, "largest")
        ch = chamfer_bidirectional(P, X_gt)
        hd = hausdorff_bidirectional(P, X_gt)
        p, r, f1, iou = prf1_iou(P, X_gt, tau=match_tau)
        bP = aabb(P); bG = aabb(X_gt)
        bb_iou = aabb_iou3d(bP, bG)
        ctr_dist = float(np.linalg.norm((bP["center"]-bG["center"])) if bP and bG else np.inf)
        rows.append({"patient_id": pid, "method": method, **params,
                     "n_pred": int(P.shape[0]), "chamfer": ch, "hausdorff": hd,
                     "precision": p, "recall": r, "f1": f1, "iou": iou,
                     "bbox_iou": bb_iou, "bbox_vol_pred": (bP["volume"] if bP else 0.0),
                     "bbox_vol_gt": (bG["volume"] if bG else 0.0),
                     "bbox_ctr_dist": ctr_dist})
        # export ply
        col = np.tile(np.array([80, 160, 255], dtype=np.uint8), (P.shape[0], 1))  # azul para global
        save_ply(P, col, ply_dir/pid/f"{pid}_tooth21_globalbest.ply")
    df = pd.DataFrame(rows)
    df.to_csv(out_dir/"global_best_by_patient.csv", index=False)
    return df

# =========================
# CLI
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ufrn_root", required=True)
    ap.add_argument("--struct_rel", default="processed_struct")
    ap.add_argument("--patients", default="all",
                    help="'all' o lista separada por comas: paciente_19,paciente_20")
    ap.add_argument("--out_dir", default="refine_allinone_out")

    ap.add_argument("--n_points", type=int, default=8192)
    ap.add_argument("--removed_thresh", type=float, default=0.02)
    ap.add_argument("--match_tau", type=float, default=0.01)

    # usar HDBSCAN?
    ap.add_argument("--use_hdbscan", action="store_true")

    # (Opcional) override de grillas por CLI
    ap.add_argument("--dbscan_eps", default=",".join(str(x) for x in DEFAULT_DBSCAN_EPS))
    ap.add_argument("--dbscan_min", default=",".join(str(x) for x in DEFAULT_DBSCAN_MIN))
    ap.add_argument("--hdbscan_min_cluster_size", default=",".join(str(x) for x in DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE))
    ap.add_argument("--hdbscan_min_samples", default=",".join(str(x) for x in DEFAULT_HDBSCAN_MIN_SAMPLES))

    ap.add_argument("--export_best_ply", action="store_true")
    ap.add_argument("--max_patients_3d", type=int, default=12)

    args = ap.parse_args()

    ufrn_root = Path(args.ufrn_root)
    struct_root = ufrn_root / args.struct_rel
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir/"figs")

    # Parse grillas
    grid_db_eps = [float(x) for x in args.dbscan_eps.split(",") if x.strip()]
    grid_db_min = [int(x)   for x in args.dbscan_min.split(",") if x.strip()]
    grid_hdb_mcs= [int(x)   for x in args.hdbscan_min_cluster_size.split(",") if x.strip()]
    grid_hdb_ms = [int(x)   for x in args.hdbscan_min_samples.split(",") if x.strip()]

    # Pacientes
    if args.patients.lower() != "all":
        pids = [p.strip() for p in args.patients.split(",") if p.strip()]
    else:
        csv_path = ufrn_root/"ufrn_por_paciente.csv"
        rows = [l.strip().split(",")[0] for l in csv_path.read_text(encoding="utf-8").splitlines()[1:] if l.strip()]
        pids = sorted(set(rows))

    methods = ["dbscan"]
    if args.use_hdbscan and _HAS_HDBSCAN:
        methods.append("hdbscan")

    # Grid completo y mejores por paciente
    grid_rows = []
    best_rows = []
    best_ply_dir = out_dir/"best_ply" if args.export_best_ply else None

    print(f"[INFO] Pacientes: {len(pids)} | métodos: {methods}")
    for pid in pids:
        rows, best = eval_one(
            pid, struct_root,
            n_points=args.n_points, removed_thresh=args.removed_thresh,
            method_list=methods,
            grid_db_eps=grid_db_eps, grid_db_min=grid_db_min,
            grid_hdb_mcs=grid_hdb_mcs, grid_hdb_ms=grid_hdb_ms,
            match_tau=args.match_tau,
            out_best_dir=best_ply_dir
        )
        grid_rows.extend(rows)
        best_rows.append(best)

    grid_df = pd.DataFrame(grid_rows)
    best_df = pd.DataFrame(best_rows)
    grid_df.to_csv(out_dir/"grid_metrics.csv", index=False)
    best_df.to_csv(out_dir/"best_by_patient.csv", index=False)
    print(f"[CSV] Grid completo: {out_dir/'grid_metrics.csv'}")
    print(f"[CSV] Mejor por paciente: {out_dir/'best_by_patient.csv'}")

    # Resumen simple
    if "chamfer" in best_df.columns:
        mean_ch = float(best_df["chamfer"].mean())
        mean_f1 = float(best_df["f1"].mean())
        print(f"[SUMMARY] best_per_patient -> Chamfer(mean)={mean_ch:.6f}  F1(mean)={mean_f1:.3f}")

    # Heatmaps & histos
    plot_heatmaps(grid_df, out_dir)
    plot_histograms(best_df, out_dir)
    if args.export_best_ply:
        plot_3d_subset(best_df, struct_root, best_ply_dir, out_dir, args.max_patients_3d)

    # -------- Mejor config global (max F1 medio) --------
    choice = select_global_best(grid_df)
    if choice is not None:
        method, params = choice
        save_json({"method": method, "params": params}, out_dir/"global_choice.json")
        print(f"[GLOBAL] Mejor config: {method} {params}")
        global_df = rerun_global(method, params, pids, struct_root,
                                 n_points=args.n_points, removed_thresh=args.removed_thresh,
                                 match_tau=args.match_tau, out_dir=out_dir)
        print(f"[CSV] Global best por paciente: {out_dir/'global_best_by_patient.csv'}")
        if args.export_best_ply:
            # 3D con global best (azul)
            if _HAS_PLOTLY:
                # reusar 3D subset pero apuntando a global_best_ply
                plot_3d_subset(global_df, struct_root, out_dir/"global_best_ply", out_dir, args.max_patients_3d)
    else:
        print("[GLOBAL] No se pudo elegir una config global (grid vacío).")

    # Índice
    index = [
        "# Reporte de refinamiento (DBSCAN/HDBSCAN)",
        "- [grid_metrics.csv](grid_metrics.csv)",
        "- [best_by_patient.csv](best_by_patient.csv)",
        "- [global_choice.json](global_choice.json) (si existe)",
        "- [global_best_by_patient.csv](global_best_by_patient.csv) (si existe)",
        "- Carpeta de PLYs (si exportaste): best_ply/ y global_best_ply/",
        "- Figuras en figs/ (si Plotly disponible)"
    ]
    (out_dir/"INDEX.md").write_text("\n".join(index), encoding="utf-8")
    print(f"[DONE] Reporte en: {out_dir}")

if __name__ == "__main__":
    main()

