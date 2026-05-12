#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_paper_figure_models_selectable_v17_dental_icp_paper_final.py

Versión v17:
- Mantiene salidas:
    PNG resumen 2x3
    HTML Plotly filtrable
    PNG individuales
    summary.json

- Mantiene alineación dental:
    1) ancla dental de malla
    2) foreground de nube
    3) bbox/PCA coarse
    4) ICP rígido/similarity restringido a zona dental

- Mejora visual final:
    - beige cálido visible (#E8C39E)
    - malla más limpia y tenue
    - errores rojos más visibles
    - TP d21 verde visible
    - fondo/background NO se reduce más en tamaño
    - leyenda Plotly grande y extendida
    - vistas focales limpias para tesis/paper

Lógica:
    Vista 1: GT multiclase
    Vista 2: predicción multiclase
    Vista 3: errores multiclase
    Vista 4: GT d21 + vecinos
    Vista 5: pred d21 + vecinos
    Vista 6: error d21
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

try:
    import trimesh
    HAS_TRIMESH = True
except Exception:
    HAS_TRIMESH = False

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False


# ============================================================
# CONFIG
# ============================================================

MODEL_ALIASES = {
    "pointnet": "PointNet",
    "pointnetpp": "PointNet++",
    "dgcnn": "DGCNN",
    "pointnettransformer": "Transformer",
    "transformer": "Transformer",
}

MODEL_DIR_ALIASES = {
    "pointnet": "pointnet",
    "pointnetpp": "pointnetpp",
    "dgcnn": "dgcnn",
    "pointnettransformer": "pointnettransformer",
    "transformer": "pointnettransformer",
}

DEFAULT_MODELS = ["pointnet"]
DEFAULT_NEIGHBORS = "d11:1,d22:9"
D21_DEFAULT = 8

CLASS_NAMES = {
    0: "background",
    1: "d11",
    2: "d12",
    3: "d13",
    4: "d14",
    5: "d15",
    6: "d16",
    7: "d17",
    8: "d21",
    9: "d22",
    10: "d23",
    11: "d24",
    12: "d25",
    13: "d26",
    14: "d27",
}

CLASS_COLORS = {
    0:  "#000000",
    1:  "#2CA02C",
    2:  "#17BECF",
    3:  "#1F77B4",
    4:  "#9467BD",
    5:  "#E377C2",
    6:  "#D62728",
    7:  "#FF7F0E",
    8:  "#FF8C00",
    9:  "#FF69B4",
    10: "#BFEF45",
    11: "#2CA58D",
    12: "#00BCD4",
    13: "#C5B0D5",
    14: "#7F7F7F",
}

COLOR_MESH = "#9E9E9E"
COLOR_BG = "#000000"

# Beige cálido final.
COLOR_OTHER = "#E8C39E"
COLOR_CORRECT_TOOTH = "#E8C39E"
COLOR_CORRECT_BG = "#000000"

COLOR_D21 = "#FF8C00"
COLOR_LEFT_NEIGH = "#0070C0"
COLOR_RIGHT_NEIGH = "#FF69B4"
COLOR_ERROR = "#FF0000"
COLOR_TP = "#00B050"

# Alphas finales.
ALPHA_BG_MAIN = 0.56
ALPHA_BG_FOCUS = 0.07
ALPHA_BG_ERROR = 0.08
ALPHA_OTHER_FOCUS = 0.92
ALPHA_OTHER_ERROR = 0.90
ALPHA_ERROR = 0.98
ALPHA_TP = 1.00
ALPHA_FOCAL = 1.00


# ============================================================
# BASIC UTILS
# ============================================================

def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Dict:
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: Path):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def sanitize_model_name(name: str) -> str:
    name = str(name).strip().lower()
    return MODEL_DIR_ALIASES.get(name, name)


def pretty_model_name(name: str) -> str:
    return MODEL_ALIASES.get(sanitize_model_name(name), name)


def parse_models(s: str) -> List[str]:
    if not s:
        return DEFAULT_MODELS

    out = []
    for x in s.replace(",", " ").split():
        x = sanitize_model_name(x)
        if x:
            out.append(x)

    return out or DEFAULT_MODELS


def parse_neighbor_teeth(s: str) -> List[Tuple[str, int]]:
    out = []
    if not s:
        return out

    for part in s.replace(";", ",").split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        name, val = part.split(":", 1)
        try:
            out.append((name.strip(), int(val)))
        except Exception:
            pass

    return out


def parse_class_filter(s: str) -> Optional[List[int]]:
    if not s or s.strip().lower() == "all":
        return None

    s = s.strip().lower()

    if s == "teeth":
        return list(range(1, 15))

    out = []
    for part in s.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except Exception:
            pass

    return out if out else None


def finite_points(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != 3:
        X = X.reshape(-1, 3)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def unwrap_object_array(arr) -> np.ndarray:
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)

    if arr.dtype != object:
        return arr

    if arr.shape == ():
        try:
            return np.asarray(arr.item())
        except Exception:
            pass

    if arr.size == 1:
        try:
            return np.asarray(arr.reshape(-1)[0])
        except Exception:
            pass

    try:
        return np.asarray(arr.tolist())
    except Exception:
        return np.asarray(arr)


def load_any_array(path: Path, preferred_keys=None) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe archivo: {path}")

    preferred_keys = preferred_keys or [
        "arr_0", "pred", "prediction", "pred_labels",
        "labels", "gt", "gt_labels", "y", "Y",
        "xyz", "points", "X",
    ]

    try:
        obj = np.load(path, allow_pickle=False)
    except ValueError:
        obj = np.load(path, allow_pickle=True)

    if isinstance(obj, np.lib.npyio.NpzFile):
        keys = list(obj.keys())
        chosen = None
        for k in preferred_keys:
            if k in keys:
                chosen = k
                break
        if chosen is None:
            chosen = keys[0]
        arr = obj[chosen]
    else:
        arr = obj

    return unwrap_object_array(arr)


def load_pred_labels(path: Path) -> np.ndarray:
    arr = load_any_array(
        path,
        preferred_keys=["pred", "prediction", "pred_labels", "labels", "y", "Y", "arr_0"]
    )
    arr = np.asarray(arr).squeeze()
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr.astype(np.int64)


def load_gt_labels(path: Path) -> np.ndarray:
    arr = load_any_array(
        path,
        preferred_keys=["gt", "gt_labels", "labels", "y", "Y", "arr_0"]
    )
    arr = np.asarray(arr).squeeze()
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr.astype(np.int64)


def load_xyz(path: Path) -> np.ndarray:
    arr = load_any_array(
        path,
        preferred_keys=["xyz", "points", "X", "arr_0"]
    )
    arr = np.asarray(arr).squeeze()
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    return finite_points(arr)


def fix_length_match(X, pred, gt=None, strict=False):
    X = finite_points(X)
    pred = np.asarray(pred).reshape(-1).astype(np.int64)
    gt_arr = None if gt is None else np.asarray(gt).reshape(-1).astype(np.int64)

    lengths = [X.shape[0], pred.shape[0]]
    if gt_arr is not None:
        lengths.append(gt_arr.shape[0])

    mn = min(lengths)
    mx = max(lengths)

    if mn != mx:
        msg = (
            f"Longitudes distintas: "
            f"X={X.shape[0]}, pred={pred.shape[0]}, "
            f"gt={None if gt_arr is None else gt_arr.shape[0]}"
        )
        if strict:
            raise ValueError(msg)
        print(f"[WARN] {msg}. Recortando a {mn}.")

    X = X[:mn]
    pred = pred[:mn]
    if gt_arr is not None:
        gt_arr = gt_arr[:mn]

    return X, pred, gt_arr


# ============================================================
# CASE LOADING
# ============================================================

def load_common_bundle(case_dir: Path) -> Dict[str, np.ndarray]:
    bundle_path = Path(case_dir) / "common" / "bundle_case_data.npz"
    if not bundle_path.exists():
        return {}

    data = np.load(bundle_path, allow_pickle=True)
    return {k: np.asarray(unwrap_object_array(data[k])) for k in data.keys()}


def load_model_case(case_dir: Path, model: str, require_pred_labels=False) -> Dict:
    case_dir = Path(case_dir)
    model_key = sanitize_model_name(model)
    model_dir = case_dir / model_key

    pred_path = model_dir / "pred_labels.npy"
    gt_path = model_dir / "gt_labels.npy"
    xyz_path = model_dir / "xyz_labels.npy"

    if require_pred_labels and not pred_path.exists():
        raise FileNotFoundError(f"Falta pred_labels.npy para {model_key}: {pred_path}")

    if not pred_path.exists():
        return {
            "model": model_key,
            "available": False,
            "reason": f"No existe {pred_path}",
        }

    pred = load_pred_labels(pred_path)
    gt = load_gt_labels(gt_path) if gt_path.exists() else None
    xyz = load_xyz(xyz_path) if xyz_path.exists() else None

    if xyz is not None:
        xyz, pred, gt = fix_length_match(xyz, pred, gt, strict=False)

    return {
        "model": model_key,
        "pretty": pretty_model_name(model_key),
        "available": True,
        "pred": pred,
        "gt": gt,
        "xyz": xyz,
        "model_dir": str(model_dir),
        "pred_path": str(pred_path),
        "gt_path": str(gt_path) if gt_path.exists() else "",
        "xyz_path": str(xyz_path) if xyz_path.exists() else "",
    }


# ============================================================
# RAW MESH
# ============================================================

def find_raw_mesh_from_meta(case_dir: Path) -> Optional[Path]:
    meta = read_json(Path(case_dir) / "meta.json")

    candidates = [
        meta.get("raw_mesh_path", ""),
        meta.get("mesh_path", ""),
        meta.get("raw_path", ""),
    ]

    for raw in candidates:
        if raw:
            p = Path(raw)
            if p.exists():
                return p

    return None


def load_raw_mesh_vertices_faces(raw_mesh_path: Optional[Path]):
    if raw_mesh_path is None:
        return None, None

    if not HAS_TRIMESH:
        print("[WARN] trimesh no disponible.")
        return None, None

    raw_mesh_path = Path(raw_mesh_path)
    if not raw_mesh_path.exists():
        print(f"[WARN] raw mesh no existe: {raw_mesh_path}")
        return None, None

    try:
        loaded = trimesh.load(str(raw_mesh_path), force="mesh", process=False)
    except Exception:
        try:
            loaded = trimesh.load(str(raw_mesh_path), force="mesh", process=True)
        except Exception as e:
            print(f"[WARN] No pude cargar raw mesh: {e}")
            return None, None

    if isinstance(loaded, trimesh.Scene):
        geos = list(loaded.dump().geometry.values())
        if not geos:
            return None, None
        loaded = trimesh.util.concatenate(geos)

    if not isinstance(loaded, trimesh.Trimesh):
        return None, None

    V = finite_points(np.asarray(loaded.vertices, dtype=np.float32))
    F = np.asarray(loaded.faces, dtype=np.int64) if loaded.faces is not None else None

    if F is not None and F.size == 0:
        F = None

    return V, F


# ============================================================
# ALIGNMENT HELPERS v17
# ============================================================

def robust_bounds(X: np.ndarray, q_low=0.01, q_high=0.99):
    X = finite_points(X)
    lo = np.quantile(X, q_low, axis=0)
    hi = np.quantile(X, q_high, axis=0)
    return lo.astype(np.float32), hi.astype(np.float32)


def bbox_center_scale(X: np.ndarray, robust=True):
    X = finite_points(X)

    if X.shape[0] == 0:
        return np.zeros(3, dtype=np.float32), 1.0

    if robust:
        lo, hi = robust_bounds(X)
    else:
        lo, hi = X.min(axis=0), X.max(axis=0)

    center = 0.5 * (lo + hi)
    diag = np.linalg.norm(hi - lo)

    if not np.isfinite(diag) or diag <= 0:
        diag = 1.0

    return center.astype(np.float32), float(diag)


def bbox_center_scale_axiswise(X: np.ndarray, robust=True):
    X = finite_points(X)

    if X.shape[0] == 0:
        return np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32)

    if robust:
        lo, hi = robust_bounds(X)
    else:
        lo, hi = X.min(axis=0), X.max(axis=0)

    center = 0.5 * (lo + hi)
    scale_xyz = hi - lo
    scale_xyz = np.where(np.abs(scale_xyz) < 1e-9, 1.0, scale_xyz)

    return center.astype(np.float32), scale_xyz.astype(np.float32)


def top_axis_subset(V: np.ndarray, axis: int = 2, q: float = 0.45) -> np.ndarray:
    V = finite_points(V)

    if V.shape[0] < 64:
        return V

    qv = np.quantile(V[:, axis], float(q))
    S = V[V[:, axis] >= qv]

    if S.shape[0] < 64:
        return V

    return S


def target_foreground_subset(X: np.ndarray, labels: Optional[np.ndarray]) -> np.ndarray:
    X = finite_points(X)

    if labels is None:
        return X

    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    n = min(X.shape[0], labels.shape[0])

    Xf = X[:n][labels[:n] != 0]

    if Xf.shape[0] < 64:
        return X

    return Xf


def target_focus_subset(X: np.ndarray,
                        labels: Optional[np.ndarray],
                        d21_class: int = 8,
                        neighbor_ids: Optional[List[int]] = None) -> np.ndarray:
    X = finite_points(X)

    if labels is None:
        return X

    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    n = min(X.shape[0], labels.shape[0])
    Xn = X[:n]
    yn = labels[:n]

    ids = [int(d21_class)]
    if neighbor_ids:
        ids += [int(v) for v in neighbor_ids]

    mask = np.isin(yn, np.asarray(ids, dtype=np.int64))
    Xf = Xn[mask]

    if Xf.shape[0] >= 64:
        return Xf

    return target_foreground_subset(Xn, yn)


def random_sample_points(X: np.ndarray, max_points: int = 4096, seed: int = 42) -> np.ndarray:
    X = finite_points(X)
    if X.shape[0] <= max_points:
        return X

    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=max_points, replace=False)
    return X[idx]


def pca_basis(X: np.ndarray):
    X = finite_points(X)
    c = X.mean(axis=0)
    Xc = X - c.reshape(1, 3)

    C = np.cov(Xc.T)
    eigvals, eigvecs = np.linalg.eigh(C)

    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    eigvals = eigvals[order]

    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, -1] *= -1

    return c.astype(np.float32), eigvecs.astype(np.float32), eigvals.astype(np.float32)


def score_alignment_nn(source_aligned: np.ndarray, target: np.ndarray) -> float:
    source_aligned = random_sample_points(source_aligned, max_points=4096, seed=123)
    target = random_sample_points(target, max_points=4096, seed=456)

    if not HAS_SCIPY:
        cs, ss = bbox_center_scale(source_aligned, robust=True)
        ct, st = bbox_center_scale(target, robust=True)
        return float(np.linalg.norm(cs - ct) + abs(ss - st))

    tree = cKDTree(target)
    dist, _ = tree.query(source_aligned, k=1, workers=-1)
    return float(np.median(dist))


def dental_bbox_transform(V_all, V_anchor, X_anchor, axiswise=False):
    V_all = finite_points(V_all)
    V_anchor = finite_points(V_anchor)
    X_anchor = finite_points(X_anchor)

    if axiswise:
        c_src, s_src = bbox_center_scale_axiswise(V_anchor, robust=True)
        c_tgt, s_tgt = bbox_center_scale_axiswise(X_anchor, robust=True)

        V_out = (V_all - c_src.reshape(1, 3)) / (s_src.reshape(1, 3) + 1e-9)
        V_out = V_out * s_tgt.reshape(1, 3) + c_tgt.reshape(1, 3)

        meta = {
            "method": "dental_bbox_axiswise",
            "source_center": c_src.tolist(),
            "source_scale_xyz": s_src.tolist(),
            "target_center": c_tgt.tolist(),
            "target_scale_xyz": s_tgt.tolist(),
        }
    else:
        c_src, s_src = bbox_center_scale(V_anchor, robust=True)
        c_tgt, s_tgt = bbox_center_scale(X_anchor, robust=True)

        V_out = (V_all - c_src.reshape(1, 3)) / (s_src + 1e-9)
        V_out = V_out * s_tgt + c_tgt.reshape(1, 3)

        meta = {
            "method": "dental_bbox",
            "source_center": c_src.tolist(),
            "source_scale": float(s_src),
            "target_center": c_tgt.tolist(),
            "target_scale": float(s_tgt),
        }

    return V_out.astype(np.float32), meta


def dental_pca_transform(V_all, V_anchor, X_anchor):
    V_anchor = finite_points(V_anchor)
    X_anchor = finite_points(X_anchor)
    V_all = finite_points(V_all)

    cs, Bs, _ = pca_basis(V_anchor)
    ct, Bt, _ = pca_basis(X_anchor)

    _, s_src = bbox_center_scale(V_anchor, robust=True)
    _, s_tgt = bbox_center_scale(X_anchor, robust=True)
    scale = s_tgt / (s_src + 1e-9)

    sign_options = [
        np.diag([sx, sy, sz]).astype(np.float32)
        for sx in [-1, 1]
        for sy in [-1, 1]
        for sz in [-1, 1]
    ]

    best = None

    for S in sign_options:
        M = Bs @ S @ Bt.T
        Va = ((V_anchor - cs.reshape(1, 3)) @ M) * scale + ct.reshape(1, 3)
        score = score_alignment_nn(Va, X_anchor)

        if best is None or score < best["score"]:
            best = {
                "M": M,
                "S": S,
                "score": score,
            }

    M = best["M"]
    V_out = ((V_all - cs.reshape(1, 3)) @ M) * scale + ct.reshape(1, 3)

    meta = {
        "method": "dental_pca",
        "source_center": cs.tolist(),
        "target_center": ct.tolist(),
        "scale": float(scale),
        "score": float(best["score"]),
        "sign_matrix": best["S"].tolist(),
    }

    return V_out.astype(np.float32), meta


def estimate_kabsch_similarity(src: np.ndarray,
                               dst: np.ndarray,
                               allow_scaling: bool = False):
    src = finite_points(src)
    dst = finite_points(dst)

    if src.shape[0] < 3 or dst.shape[0] < 3:
        return np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32), 1.0

    cs = src.mean(axis=0)
    cd = dst.mean(axis=0)

    X = src - cs.reshape(1, 3)
    Y = dst - cd.reshape(1, 3)

    H = X.T @ Y

    try:
        U, S, Vt = np.linalg.svd(H)
        R = U @ Vt

        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = U @ Vt

        scale = 1.0
        if allow_scaling:
            denom = float(np.sum(X ** 2)) + 1e-9
            scale = float(np.sum(S) / denom)
            if not np.isfinite(scale) or scale <= 0:
                scale = 1.0

        t = cd - scale * (cs @ R)

        return R.astype(np.float32), t.astype(np.float32), float(scale)

    except Exception:
        return np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32), 1.0


def apply_similarity(X: np.ndarray, R: np.ndarray, t: np.ndarray, scale: float = 1.0):
    X = finite_points(X)
    return (float(scale) * (X @ R) + t.reshape(1, 3)).astype(np.float32)


def icp_refine_mesh_to_cloud(
    V_all_aligned: np.ndarray,
    V_anchor_aligned: np.ndarray,
    X_anchor: np.ndarray,
    n_iter: int = 35,
    max_corr: float = 0.08,
    trim_q: float = 0.85,
    allow_scaling: bool = False,
    source_sample: int = 8000,
    target_sample: int = 8000,
    seed: int = 42,
):
    meta = {
        "enabled": False,
        "method": "internal_trimmed_icp",
        "has_scipy": bool(HAS_SCIPY),
        "n_iter_requested": int(n_iter),
        "max_corr": float(max_corr),
        "trim_q": float(trim_q),
        "allow_scaling": bool(allow_scaling),
    }

    V_all_aligned = finite_points(V_all_aligned)
    src0 = finite_points(V_anchor_aligned)
    tgt0 = finite_points(X_anchor)

    if not HAS_SCIPY:
        meta["reason"] = "scipy_not_available"
        return V_all_aligned, meta

    if src0.shape[0] < 64 or tgt0.shape[0] < 64:
        meta["reason"] = f"not_enough_points src={src0.shape[0]} tgt={tgt0.shape[0]}"
        return V_all_aligned, meta

    src = random_sample_points(src0, max_points=int(source_sample), seed=seed)
    tgt = random_sample_points(tgt0, max_points=int(target_sample), seed=seed + 1)

    tree = cKDTree(tgt)

    current_src = src.copy()
    current_all = V_all_aligned.copy()

    history = []
    total_scale = 1.0

    for it in range(int(n_iter)):
        dist, idx = tree.query(current_src, k=1, workers=-1)
        dist = np.asarray(dist, dtype=np.float32)
        idx = np.asarray(idx, dtype=np.int64)

        finite_mask = np.isfinite(dist)
        if not np.any(finite_mask):
            break

        dist_f = dist[finite_mask]
        src_f = current_src[finite_mask]
        dst_f = tgt[idx[finite_mask]]

        if max_corr is not None and float(max_corr) > 0:
            mask_corr = dist_f <= float(max_corr)
        else:
            mask_corr = np.ones_like(dist_f, dtype=bool)

        if mask_corr.sum() < 20:
            mask_corr = np.ones_like(dist_f, dtype=bool)

        src_c = src_f[mask_corr]
        dst_c = dst_f[mask_corr]
        dist_c = dist_f[mask_corr]

        if src_c.shape[0] < 20:
            break

        tq = float(np.clip(trim_q, 0.05, 1.0))
        thr = np.quantile(dist_c, tq)
        keep = dist_c <= thr

        src_k = src_c[keep]
        dst_k = dst_c[keep]

        if src_k.shape[0] < 20:
            src_k = src_c
            dst_k = dst_c

        R, t, s = estimate_kabsch_similarity(
            src_k,
            dst_k,
            allow_scaling=bool(allow_scaling),
        )

        current_src = apply_similarity(current_src, R, t, s)
        current_all = apply_similarity(current_all, R, t, s)
        total_scale *= float(s)

        med = float(np.median(dist_c))
        mean = float(np.mean(dist_c))
        history.append({
            "iter": int(it),
            "n_corr": int(src_k.shape[0]),
            "median_dist_before": med,
            "mean_dist_before": mean,
            "scale_step": float(s),
        })

        if len(history) >= 4:
            prev = history[-4]["median_dist_before"]
            cur = history[-1]["median_dist_before"]
            if abs(prev - cur) < 1e-6:
                break

    meta.update({
        "enabled": True,
        "n_iter_done": int(len(history)),
        "history_tail": history[-8:],
        "median_before_first": float(history[0]["median_dist_before"]) if history else None,
        "median_before_last": float(history[-1]["median_dist_before"]) if history else None,
        "total_scale": float(total_scale),
        "source_sample_n": int(src.shape[0]),
        "target_sample_n": int(tgt.shape[0]),
    })

    return current_all.astype(np.float32), meta


def refine_translation_nn(V_all_aligned, V_anchor_aligned, X_anchor):
    if not HAS_SCIPY:
        return V_all_aligned, {
            "enabled": False,
            "reason": "scipy_not_available",
        }

    V_anchor_aligned = random_sample_points(V_anchor_aligned, max_points=4096)
    X_anchor = random_sample_points(X_anchor, max_points=4096)

    tree = cKDTree(V_anchor_aligned)
    _, idx = tree.query(X_anchor, k=1, workers=-1)

    residual = X_anchor - V_anchor_aligned[idx]
    shift = np.median(residual, axis=0).astype(np.float32)

    V_out = finite_points(V_all_aligned) + shift.reshape(1, 3)

    return V_out.astype(np.float32), {
        "enabled": True,
        "method": "median_nn_translation",
        "shift": shift.tolist(),
    }


def align_raw_mesh_dental(
    raw_vertices,
    raw_faces,
    target_xyz,
    target_labels,
    align_mode="dental_pca_refine",
    mesh_top_axis=2,
    mesh_top_q=0.45,
    d21_class=8,
    neighbor_ids=None,
    icp_iter=35,
    icp_max_corr=0.08,
    icp_trim_q=0.85,
    icp_allow_scaling=False,
    icp_focus="foreground",
):
    if raw_vertices is None:
        return None, raw_faces, {"mode": "missing_raw"}

    V_all = finite_points(raw_vertices)
    X = finite_points(target_xyz)

    V_anchor = top_axis_subset(V_all, axis=mesh_top_axis, q=mesh_top_q)

    if str(icp_focus).lower() in {"d21", "focus", "neighbors", "d21_neighbors"}:
        X_anchor = target_focus_subset(
            X,
            target_labels,
            d21_class=d21_class,
            neighbor_ids=neighbor_ids or [],
        )
        target_anchor_kind = "d21_plus_neighbors"
    else:
        X_anchor = target_foreground_subset(X, target_labels)
        target_anchor_kind = "foreground"

    meta = {
        "align_mode": align_mode,
        "mesh_anchor_n": int(V_anchor.shape[0]),
        "target_anchor_n": int(X_anchor.shape[0]),
        "target_anchor_kind": target_anchor_kind,
        "mesh_top_axis": int(mesh_top_axis),
        "mesh_top_q": float(mesh_top_q),
        "icp_iter": int(icp_iter),
        "icp_max_corr": float(icp_max_corr),
        "icp_trim_q": float(icp_trim_q),
        "icp_allow_scaling": bool(icp_allow_scaling),
        "icp_focus": str(icp_focus),
    }

    if align_mode == "none":
        return V_all, raw_faces, {**meta, "method": "none"}

    if align_mode == "robust_bbox":
        V_out, m = dental_bbox_transform(V_all, V_all, X, axiswise=False)
        meta.update(m)
        return V_out, raw_faces, meta

    if align_mode == "dental_bbox":
        V_out, m = dental_bbox_transform(V_all, V_anchor, X_anchor, axiswise=False)
        meta.update(m)
        return V_out, raw_faces, meta

    if align_mode == "dental_bbox_axiswise":
        V_out, m = dental_bbox_transform(V_all, V_anchor, X_anchor, axiswise=True)
        meta.update(m)
        return V_out, raw_faces, meta

    if align_mode == "dental_pca":
        V_out, m = dental_pca_transform(V_all, V_anchor, X_anchor)
        meta.update(m)
        return V_out, raw_faces, meta

    if align_mode in {"dental_icp", "dental_bbox_icp"}:
        V_coarse, m = dental_bbox_transform(V_all, V_anchor, X_anchor, axiswise=False)
        V_anchor_coarse, _ = dental_bbox_transform(V_anchor, V_anchor, X_anchor, axiswise=False)

        V_out, r = icp_refine_mesh_to_cloud(
            V_all_aligned=V_coarse,
            V_anchor_aligned=V_anchor_coarse,
            X_anchor=X_anchor,
            n_iter=icp_iter,
            max_corr=icp_max_corr,
            trim_q=icp_trim_q,
            allow_scaling=icp_allow_scaling,
        )

        meta.update(m)
        meta["refine"] = r
        return V_out, raw_faces, meta

    if align_mode in {"dental_pca_refine", "dental_pca_icp"}:
        V_pca, m = dental_pca_transform(V_all, V_anchor, X_anchor)
        V_anchor_pca, _ = dental_pca_transform(V_anchor, V_anchor, X_anchor)

        V_out, r = icp_refine_mesh_to_cloud(
            V_all_aligned=V_pca,
            V_anchor_aligned=V_anchor_pca,
            X_anchor=X_anchor,
            n_iter=icp_iter,
            max_corr=icp_max_corr,
            trim_q=icp_trim_q,
            allow_scaling=icp_allow_scaling,
        )

        meta.update(m)
        meta["refine"] = r
        return V_out, raw_faces, meta

    raise ValueError(f"align_mode no reconocido: {align_mode}")


# ============================================================
# COLORS / METRICS
# ============================================================

def hex_to_rgba01(hex_color: str, alpha: float = 1.0):
    h = hex_color.strip().lstrip("#")
    return (
        int(h[0:2], 16) / 255.0,
        int(h[2:4], 16) / 255.0,
        int(h[4:6], 16) / 255.0,
        float(alpha),
    )


def rgba_to_plotly(c):
    r = int(np.clip(c[0] * 255, 0, 255))
    g = int(np.clip(c[1] * 255, 0, 255))
    b = int(np.clip(c[2] * 255, 0, 255))
    a = float(np.clip(c[3], 0, 1))
    return f"rgba({r},{g},{b},{a:.3f})"


def hex_to_plotly_rgba(hex_color: str, alpha: float = 1.0):
    return rgba_to_plotly(hex_to_rgba01(hex_color, alpha))


def colors_to_plotly(colors):
    return [c if isinstance(c, str) else rgba_to_plotly(c) for c in colors]


def labels_to_colors(labels, alpha_bg=ALPHA_BG_MAIN, alpha_fg=1.0):
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    out = []

    for y in labels:
        y = int(y)
        if y == 0:
            out.append(hex_to_rgba01(COLOR_BG, alpha_bg))
        else:
            out.append(hex_to_rgba01(CLASS_COLORS.get(y, "#444444"), alpha_fg))

    return out


def error_colors_all(pred, gt, alpha_ok=ALPHA_OTHER_ERROR):
    pred = np.asarray(pred).reshape(-1).astype(np.int64)
    gt = np.asarray(gt).reshape(-1).astype(np.int64)
    n = min(len(pred), len(gt))
    pred, gt = pred[:n], gt[:n]

    out = []
    for p, g in zip(pred, gt):
        if int(p) == int(g):
            if int(g) == 0:
                out.append(hex_to_rgba01(COLOR_CORRECT_BG, ALPHA_BG_ERROR))
            else:
                out.append(hex_to_rgba01(COLOR_CORRECT_TOOTH, alpha_ok))
        else:
            out.append(hex_to_rgba01(COLOR_ERROR, ALPHA_ERROR))

    return out


def neighbor_focus_colors(labels, d21_class, neighbors, alpha_bg=ALPHA_BG_FOCUS):
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    neigh_ids = [v for _, v in neighbors]

    out = []
    for y in labels:
        y = int(y)

        if y == int(d21_class):
            out.append(hex_to_rgba01(COLOR_D21, ALPHA_FOCAL))
        elif len(neigh_ids) >= 1 and y == neigh_ids[0]:
            out.append(hex_to_rgba01(COLOR_LEFT_NEIGH, ALPHA_FOCAL))
        elif len(neigh_ids) >= 2 and y == neigh_ids[1]:
            out.append(hex_to_rgba01(COLOR_RIGHT_NEIGH, ALPHA_FOCAL))
        elif y == 0:
            out.append(hex_to_rgba01(COLOR_BG, alpha_bg))
        else:
            out.append(hex_to_rgba01(COLOR_OTHER, ALPHA_OTHER_FOCUS))

    return out


def d21_tp_error_colors(pred, gt, d21_class, neighbors, alpha_bg=ALPHA_BG_FOCUS):
    pred = np.asarray(pred).reshape(-1).astype(np.int64)
    gt = np.asarray(gt).reshape(-1).astype(np.int64)

    n = min(len(pred), len(gt))
    pred, gt = pred[:n], gt[:n]

    neigh_ids = [v for _, v in neighbors]

    out = []
    for p, g in zip(pred, gt):
        p, g = int(p), int(g)

        if p == d21_class and g == d21_class:
            out.append(hex_to_rgba01(COLOR_TP, ALPHA_TP))
        elif (p == d21_class) != (g == d21_class):
            out.append(hex_to_rgba01(COLOR_ERROR, ALPHA_ERROR))
        elif len(neigh_ids) >= 1 and g == neigh_ids[0]:
            out.append(hex_to_rgba01(COLOR_LEFT_NEIGH, ALPHA_FOCAL))
        elif len(neigh_ids) >= 2 and g == neigh_ids[1]:
            out.append(hex_to_rgba01(COLOR_RIGHT_NEIGH, ALPHA_FOCAL))
        elif g == 0:
            out.append(hex_to_rgba01(COLOR_BG, alpha_bg))
        else:
            out.append(hex_to_rgba01(COLOR_OTHER, ALPHA_OTHER_FOCUS))

    return out


def compute_simple_metrics(pred, gt, d21_class=8):
    pred = np.asarray(pred).reshape(-1)
    gt = np.asarray(gt).reshape(-1)
    n = min(len(pred), len(gt))
    pred, gt = pred[:n], gt[:n]

    mask_no_bg = gt != 0
    acc_no_bg = float((pred[mask_no_bg] == gt[mask_no_bg]).mean()) if mask_no_bg.any() else 0.0

    d21_gt = gt == int(d21_class)
    d21_pr = pred == int(d21_class)

    tp = int(np.logical_and(d21_gt, d21_pr).sum())
    fp = int(np.logical_and(~d21_gt, d21_pr).sum())
    fn = int(np.logical_and(d21_gt, ~d21_pr).sum())

    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    iou = tp / (tp + fp + fn + 1e-9)

    return {
        "acc_no_bg": acc_no_bg,
        "d21_precision": float(prec),
        "d21_recall": float(rec),
        "d21_f1": float(f1),
        "d21_iou": float(iou),
        "n_points": int(n),
    }


def filter_points_by_classes(X, labels_for_filter, colors, classes):
    if classes is None:
        return X, labels_for_filter, colors

    labels_for_filter = np.asarray(labels_for_filter).reshape(-1).astype(np.int64)
    mask = np.isin(labels_for_filter, np.asarray(classes, dtype=np.int64))

    Xf = finite_points(X)[mask]
    yf = labels_for_filter[mask]
    cf = [c for c, keep in zip(colors, mask) if bool(keep)]

    return Xf, yf, cf


# ============================================================
# DATA PREP
# ============================================================

def prepare_case_data(case_dir, model_key, d21_class, neighbors, align_mode,
                      mesh_top_q, icp_iter, icp_max_corr, icp_trim_q,
                      icp_allow_scaling, icp_focus):
    case_dir = Path(case_dir)
    common = load_common_bundle(case_dir)
    md = load_model_case(case_dir, model_key, require_pred_labels=True)

    X = md.get("xyz")
    pred = md["pred"]
    gt = md.get("gt")

    if X is None:
        if "xyz_8192" in common:
            X = finite_points(common["xyz_8192"])
        else:
            raise RuntimeError("No hay xyz_labels.npy ni xyz_8192 en common.")

    if gt is None:
        if "y_8192" in common:
            gt = np.asarray(common["y_8192"]).reshape(-1).astype(np.int64)
        else:
            raise RuntimeError("No hay gt_labels.npy ni y_8192 en common.")

    X, pred, gt = fix_length_match(X, pred, gt, strict=False)

    raw_path = find_raw_mesh_from_meta(case_dir)
    V_raw, F_raw = load_raw_mesh_vertices_faces(raw_path)

    neighbor_ids = [int(v) for _, v in neighbors]

    V, F, align_meta = align_raw_mesh_dental(
        raw_vertices=V_raw,
        raw_faces=F_raw,
        target_xyz=X,
        target_labels=gt,
        align_mode=align_mode,
        mesh_top_axis=2,
        mesh_top_q=mesh_top_q,
        d21_class=d21_class,
        neighbor_ids=neighbor_ids,
        icp_iter=icp_iter,
        icp_max_corr=icp_max_corr,
        icp_trim_q=icp_trim_q,
        icp_allow_scaling=icp_allow_scaling,
        icp_focus=icp_focus,
    )

    colors = {
        "gt_all": labels_to_colors(gt),
        "pred_all": labels_to_colors(pred),
        "err_all": error_colors_all(pred, gt),
        "gt_focus": neighbor_focus_colors(gt, d21_class, neighbors),
        "pred_focus": neighbor_focus_colors(pred, d21_class, neighbors),
        "d21_error": d21_tp_error_colors(pred, gt, d21_class, neighbors),
    }

    labels = {
        "gt_all": gt,
        "pred_all": pred,
        "err_all": gt,
        "gt_focus": gt,
        "pred_focus": pred,
        "d21_error": gt,
    }

    return {
        "case_dir": str(case_dir),
        "model_key": model_key,
        "pretty": md.get("pretty", pretty_model_name(model_key)),
        "X": X,
        "pred": pred,
        "gt": gt,
        "V": V,
        "F": F,
        "raw_path": str(raw_path) if raw_path else "",
        "align_meta": align_meta,
        "metrics": compute_simple_metrics(pred, gt, d21_class=d21_class),
        "colors": colors,
        "labels": labels,
    }


# ============================================================
# MATPLOTLIB RENDER
# ============================================================

def set_axes_from_points(ax, X, zoom=1.50, pad=0.10):
    X = finite_points(X)

    lo = np.quantile(X, 0.005, axis=0)
    hi = np.quantile(X, 0.995, axis=0)
    center = 0.5 * (lo + hi)
    radius = 0.5 * float(np.max(hi - lo)) * (1.0 + pad)
    radius = radius / max(float(zoom), 1e-6)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def apply_view(ax, X, elev=25, azim=-55, zoom=1.50):
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_facecolor("white")

    try:
        ax.dist = 8.4
    except Exception:
        pass

    set_axes_from_points(ax, X, zoom=zoom)


def add_mesh_static(ax, V, F, mesh_alpha=0.07):
    if V is None or F is None:
        return

    V = finite_points(V)
    F = np.asarray(F, dtype=np.int64)

    poly = Poly3DCollection(
        V[F],
        facecolors=hex_to_rgba01(COLOR_MESH, mesh_alpha),
        edgecolors=(0.42, 0.42, 0.42, 0.010),
        linewidths=0.006,
        antialiased=True,
    )
    ax.add_collection3d(poly)
    ax.auto_scale_xyz(V[:, 0], V[:, 1], V[:, 2])


def add_points_static(ax, X, colors, point_size=8.8):
    X = finite_points(X)

    ax.scatter(
        X[:, 0], X[:, 1], X[:, 2],
        c=colors,
        s=float(point_size),
        linewidths=0,
        depthshade=False,
    )


def scatter_by_mask_static(ax, X, mask, color, point_size, alpha=1.0):
    X = finite_points(X)
    mask = np.asarray(mask).reshape(-1).astype(bool)
    n = min(X.shape[0], mask.shape[0])
    X = X[:n]
    mask = mask[:n]

    if not np.any(mask):
        return

    ax.scatter(
        X[mask, 0],
        X[mask, 1],
        X[mask, 2],
        c=[hex_to_rgba01(color, alpha)],
        s=float(point_size),
        linewidths=0,
        depthshade=False,
    )


def plot_static_panel(ax, V, F, X, labels_for_filter, colors, title,
                      elev, azim, zoom, mesh_alpha, point_size, show_classes):
    Xp, _, Cp = filter_points_by_classes(X, labels_for_filter, colors, show_classes)

    add_mesh_static(ax, V, F, mesh_alpha=mesh_alpha)
    add_points_static(ax, Xp, Cp, point_size=point_size)

    ax.set_title(title, fontsize=12, fontweight="bold", pad=2)
    apply_view(ax, X, elev=elev, azim=azim, zoom=zoom)


def plot_static_panel_ordered(ax, V, F, X, pred, gt, key, d21_class, neighbors,
                              title, elev, azim, zoom, mesh_alpha, point_size):
    X = finite_points(X)
    pred = np.asarray(pred).reshape(-1).astype(np.int64)
    gt = np.asarray(gt).reshape(-1).astype(np.int64)

    n = min(X.shape[0], pred.shape[0], gt.shape[0])
    X = X[:n]
    pred = pred[:n]
    gt = gt[:n]

    neigh_ids = [v for _, v in neighbors]

    add_mesh_static(ax, V, F, mesh_alpha=mesh_alpha)

    if key == "err_all":
        bg_ok = (pred == gt) & (gt == 0)
        tooth_ok = (pred == gt) & (gt != 0)
        err = pred != gt

        # Background mantiene el mismo tamaño base; solo alpha bajo.
        scatter_by_mask_static(ax, X, bg_ok, COLOR_BG, point_size, ALPHA_BG_ERROR)
        scatter_by_mask_static(ax, X, tooth_ok, COLOR_OTHER, point_size * 1.00, ALPHA_OTHER_ERROR)
        scatter_by_mask_static(ax, X, err, COLOR_ERROR, point_size * 1.35, ALPHA_ERROR)

    elif key == "gt_focus":
        labels = gt
        bg = labels == 0
        other = labels != 0
        d21 = labels == int(d21_class)

        n1 = labels == neigh_ids[0] if len(neigh_ids) >= 1 else np.zeros_like(bg, dtype=bool)
        n2 = labels == neigh_ids[1] if len(neigh_ids) >= 2 else np.zeros_like(bg, dtype=bool)

        other = other & (~d21) & (~n1) & (~n2)

        scatter_by_mask_static(ax, X, bg, COLOR_BG, point_size, ALPHA_BG_FOCUS)
        scatter_by_mask_static(ax, X, other, COLOR_OTHER, point_size * 1.00, ALPHA_OTHER_FOCUS)
        scatter_by_mask_static(ax, X, n1, COLOR_LEFT_NEIGH, point_size * 1.13, ALPHA_FOCAL)
        scatter_by_mask_static(ax, X, n2, COLOR_RIGHT_NEIGH, point_size * 1.13, ALPHA_FOCAL)
        scatter_by_mask_static(ax, X, d21, COLOR_D21, point_size * 1.18, ALPHA_FOCAL)

    elif key == "pred_focus":
        labels = pred
        bg = labels == 0
        other = labels != 0
        d21 = labels == int(d21_class)

        n1 = labels == neigh_ids[0] if len(neigh_ids) >= 1 else np.zeros_like(bg, dtype=bool)
        n2 = labels == neigh_ids[1] if len(neigh_ids) >= 2 else np.zeros_like(bg, dtype=bool)

        other = other & (~d21) & (~n1) & (~n2)

        scatter_by_mask_static(ax, X, bg, COLOR_BG, point_size, ALPHA_BG_FOCUS)
        scatter_by_mask_static(ax, X, other, COLOR_OTHER, point_size * 1.00, ALPHA_OTHER_FOCUS)
        scatter_by_mask_static(ax, X, n1, COLOR_LEFT_NEIGH, point_size * 1.13, ALPHA_FOCAL)
        scatter_by_mask_static(ax, X, n2, COLOR_RIGHT_NEIGH, point_size * 1.13, ALPHA_FOCAL)
        scatter_by_mask_static(ax, X, d21, COLOR_D21, point_size * 1.18, ALPHA_FOCAL)

    elif key == "d21_error":
        bg = gt == 0
        d21_tp = (pred == int(d21_class)) & (gt == int(d21_class))
        d21_err = ((pred == int(d21_class)) != (gt == int(d21_class)))

        n1 = gt == neigh_ids[0] if len(neigh_ids) >= 1 else np.zeros_like(bg, dtype=bool)
        n2 = gt == neigh_ids[1] if len(neigh_ids) >= 2 else np.zeros_like(bg, dtype=bool)

        other = (gt != 0) & (~d21_tp) & (~d21_err) & (~n1) & (~n2)

        scatter_by_mask_static(ax, X, bg, COLOR_BG, point_size, ALPHA_BG_FOCUS)
        scatter_by_mask_static(ax, X, other, COLOR_OTHER, point_size * 1.00, ALPHA_OTHER_FOCUS)
        scatter_by_mask_static(ax, X, n1, COLOR_LEFT_NEIGH, point_size * 1.08, ALPHA_FOCAL)
        scatter_by_mask_static(ax, X, n2, COLOR_RIGHT_NEIGH, point_size * 1.08, ALPHA_FOCAL)
        scatter_by_mask_static(ax, X, d21_tp, COLOR_TP, point_size * 1.18, ALPHA_TP)
        scatter_by_mask_static(ax, X, d21_err, COLOR_ERROR, point_size * 1.35, ALPHA_ERROR)

    else:
        raise ValueError(f"Vista ordenada no reconocida: {key}")

    ax.set_title(title, fontsize=12, fontweight="bold", pad=2)
    apply_view(ax, X, elev=elev, azim=azim, zoom=zoom)


def make_summary_png(data, out_png, d21_class, neighbors, elev, azim,
                     zoom, zoom_focus, mesh_alpha, point_size, dpi,
                     show_classes, title=""):
    out_png = Path(out_png)
    ensure_dir(out_png.parent)

    X, V, F = data["X"], data["V"], data["F"]
    pred, gt = data["pred"], data["gt"]
    colors, labels = data["colors"], data["labels"]
    pretty, metrics = data["pretty"], data["metrics"]

    fig = plt.figure(figsize=(15.5, 9.4), dpi=int(dpi), facecolor="white")
    gs = GridSpec(2, 3, figure=fig, wspace=0.015, hspace=0.10)

    metric_txt = f"\nacc_nb={metrics['acc_no_bg']:.3f} | d21_f1={metrics['d21_f1']:.3f}"

    panel_specs = [
        ("GT\n(todas las clases)", "gt_all", zoom, point_size, False),
        (f"{pretty}\n(predicción multiclase){metric_txt}", "pred_all", zoom, point_size, False),
        ("Errores multiclase\nrojo = pred ≠ GT", "err_all", zoom, point_size * 1.02, True),
        ("GT: d21 + vecinos\nazul/naranja/rosado", "gt_focus", zoom_focus, point_size * 1.08, True),
        (f"{pretty}: d21 + vecinos\nazul/naranja/rosado", "pred_focus", zoom_focus, point_size * 1.08, True),
        ("Error d21\nverde=TP | rojo=FP/FN", "d21_error", zoom_focus, point_size * 1.10, True),
    ]

    for i, (ttl, key, z, ps, ordered) in enumerate(panel_specs):
        ax = fig.add_subplot(gs[i // 3, i % 3], projection="3d")

        if ordered:
            plot_static_panel_ordered(
                ax=ax,
                V=V,
                F=F,
                X=X,
                pred=pred,
                gt=gt,
                key=key,
                d21_class=d21_class,
                neighbors=neighbors,
                title=ttl,
                elev=elev,
                azim=azim,
                zoom=z,
                mesh_alpha=mesh_alpha,
                point_size=ps,
            )
        else:
            plot_static_panel(
                ax, V, F, X, labels[key], colors[key], ttl,
                elev, azim, z, mesh_alpha, ps, show_classes
            )

    fig.suptitle(
        title or "Vista paper — alineación dental + nubes coloreadas",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )

    fig.text(
        0.5,
        0.962,
        f"modelo={pretty} | d21={d21_class} | vecinos={neighbors}",
        ha="center",
        va="top",
        fontsize=10.5,
        color="#333333",
    )

    legend_items = [
        ("background", "#000000"),
        ("otros dientes", COLOR_OTHER),
        ("vecino izq.", COLOR_LEFT_NEIGH),
        ("d21", COLOR_D21),
        ("vecino der.", COLOR_RIGHT_NEIGH),
        ("error", COLOR_ERROR),
        ("TP d21", COLOR_TP),
        ("malla raw", COLOR_MESH),
    ]

    x0, y0, dx = 0.055, 0.035, 0.112
    for i, (lab, col) in enumerate(legend_items):
        x = x0 + i * dx
        fig.text(x, y0, "■", color=col, fontsize=14, ha="left", va="center")
        fig.text(x + 0.017, y0, lab, color="black", fontsize=8.8, ha="left", va="center")

    fig.text(
        0.5,
        0.012,
        "Alineación v17: beige cálido final; errores destacados; fondo tenue sin reducir tamaño.",
        ha="center",
        fontsize=9.5,
        color="#333333",
    )

    fig.savefig(
        out_png,
        bbox_inches="tight",
        pad_inches=0.025,
        dpi=int(dpi),
        facecolor="white",
    )
    plt.close(fig)

    return out_png


def make_individual_pngs(data, out_dir, elev, azim, zoom, zoom_focus,
                         mesh_alpha, point_size, dpi, show_classes,
                         d21_class, neighbors):
    out_dir = ensure_dir(Path(out_dir))

    X, V, F = data["X"], data["V"], data["F"]
    pred, gt = data["pred"], data["gt"]
    colors, labels = data["colors"], data["labels"]
    pretty, metrics = data["pretty"], data["metrics"]

    metric_txt = f"\nacc_nb={metrics['acc_no_bg']:.3f} | d21_f1={metrics['d21_f1']:.3f}"

    panels = [
        ("01_gt_multiclase", "GT\n(todas las clases)", "gt_all", zoom, False),
        ("02_pred_multiclase", f"{pretty}\n(predicción multiclase){metric_txt}", "pred_all", zoom, False),
        ("03_errores_multiclase", "Errores multiclase\nrojo = pred ≠ GT", "err_all", zoom, True),
        ("04_gt_d21_vecinos", "GT: d21 + vecinos\nazul/naranja/rosado", "gt_focus", zoom_focus, True),
        ("05_pred_d21_vecinos", f"{pretty}: d21 + vecinos\nazul/naranja/rosado", "pred_focus", zoom_focus, True),
        ("06_error_d21", "Error d21\nverde=TP | rojo=FP/FN", "d21_error", zoom_focus, True),
    ]

    saved = []
    for stem, ttl, key, z, ordered in panels:
        fig = plt.figure(figsize=(7.2, 5.6), dpi=int(dpi), facecolor="white")
        ax = fig.add_subplot(111, projection="3d")

        if ordered:
            plot_static_panel_ordered(
                ax=ax,
                V=V,
                F=F,
                X=X,
                pred=pred,
                gt=gt,
                key=key,
                d21_class=d21_class,
                neighbors=neighbors,
                title=ttl,
                elev=elev,
                azim=azim,
                zoom=z,
                mesh_alpha=mesh_alpha,
                point_size=point_size,
            )
        else:
            plot_static_panel(
                ax, V, F, X, labels[key], colors[key], ttl,
                elev, azim, z, mesh_alpha, point_size, show_classes,
            )

        out = out_dir / f"{stem}.png"
        fig.savefig(out, bbox_inches="tight", pad_inches=0.02, dpi=int(dpi), facecolor="white")
        plt.close(fig)
        saved.append(str(out))

    return saved


# ============================================================
# PLOTLY
# ============================================================

def add_plotly_mesh(fig, row, col, V, F, mesh_opacity):
    if V is None or F is None or mesh_opacity <= 0:
        return

    V = finite_points(V)
    F = np.asarray(F, dtype=np.int64)

    fig.add_trace(
        go.Mesh3d(
            x=V[:, 0],
            y=V[:, 1],
            z=V[:, 2],
            i=F[:, 0],
            j=F[:, 1],
            k=F[:, 2],
            color=COLOR_MESH,
            opacity=float(mesh_opacity),
            showscale=False,
            hoverinfo="skip",
            name="raw mesh",
            legendgroup="mesh",
            showlegend=False,
        ),
        row=row,
        col=col,
    )


def add_plotly_points_by_class(fig, row, col, X, labels_for_filter,
                               colors, trace_prefix, point_size, show_classes):
    X = finite_points(X)
    labels_for_filter = np.asarray(labels_for_filter).reshape(-1).astype(np.int64)

    if show_classes is None:
        classes = sorted(int(c) for c in np.unique(labels_for_filter))
    else:
        unique_set = set(np.unique(labels_for_filter).tolist())
        classes = [int(c) for c in show_classes if int(c) in unique_set]

    colors_arr = np.asarray(colors, dtype=object)

    for cls in classes:
        mask = labels_for_filter == cls
        if not np.any(mask):
            continue

        Xc = X[mask]
        color_c = colors_to_plotly([colors_arr[mask][0]])[0]
        cls_name = CLASS_NAMES.get(cls, f"class_{cls}")

        fig.add_trace(
            go.Scatter3d(
                x=Xc[:, 0],
                y=Xc[:, 1],
                z=Xc[:, 2],
                mode="markers",
                marker=dict(
                    size=float(point_size),
                    color=color_c,
                    opacity=1.0,
                ),
                name=f"{trace_prefix}: {cls} - {cls_name}",
                legendgroup=f"{trace_prefix}_{cls}",
                showlegend=True,
                visible=True,
            ),
            row=row,
            col=col,
        )


def add_plotly_mask_trace(fig, row, col, X, mask, name, color, point_size,
                          opacity=1.0, showlegend=True):
    X = finite_points(X)
    mask = np.asarray(mask).reshape(-1).astype(bool)

    n = min(X.shape[0], mask.shape[0])
    X = X[:n]
    mask = mask[:n]

    if not np.any(mask):
        return

    fig.add_trace(
        go.Scatter3d(
            x=X[mask, 0],
            y=X[mask, 1],
            z=X[mask, 2],
            mode="markers",
            marker=dict(
                size=float(point_size),
                color=color,
                opacity=float(opacity),
            ),
            name=name,
            showlegend=showlegend,
        ),
        row=row,
        col=col,
    )


def add_plotly_error_points(fig, row, col, X, pred, gt, point_size):
    X = finite_points(X)
    pred = np.asarray(pred).reshape(-1).astype(np.int64)
    gt = np.asarray(gt).reshape(-1).astype(np.int64)

    n = min(X.shape[0], pred.shape[0], gt.shape[0])
    X, pred, gt = X[:n], pred[:n], gt[:n]

    bg_ok = (pred == gt) & (gt == 0)
    non_bg_ok = (pred == gt) & (gt != 0)
    err = pred != gt

    add_plotly_mask_trace(
        fig, row, col, X, bg_ok,
        "errores: bg correcto",
        hex_to_plotly_rgba(COLOR_BG, ALPHA_BG_ERROR),
        point_size,
    )
    add_plotly_mask_trace(
        fig, row, col, X, non_bg_ok,
        "errores: correcto",
        hex_to_plotly_rgba(COLOR_OTHER, ALPHA_OTHER_ERROR),
        point_size,
    )
    add_plotly_mask_trace(
        fig, row, col, X, err,
        "errores: pred ≠ GT",
        hex_to_plotly_rgba(COLOR_ERROR, ALPHA_ERROR),
        point_size * 1.35,
    )


def add_plotly_focus_points(fig, row, col, X, labels, d21_class, neighbors,
                            trace_prefix, point_size):
    X = finite_points(X)
    labels = np.asarray(labels).reshape(-1).astype(np.int64)

    n = min(X.shape[0], labels.shape[0])
    X, labels = X[:n], labels[:n]

    neigh_ids = [v for _, v in neighbors]

    bg = labels == 0
    d21 = labels == int(d21_class)
    n1 = labels == neigh_ids[0] if len(neigh_ids) >= 1 else np.zeros_like(bg, dtype=bool)
    n2 = labels == neigh_ids[1] if len(neigh_ids) >= 2 else np.zeros_like(bg, dtype=bool)
    other = (labels != 0) & (~d21) & (~n1) & (~n2)

    add_plotly_mask_trace(
        fig, row, col, X, bg,
        f"{trace_prefix}: background",
        hex_to_plotly_rgba(COLOR_BG, ALPHA_BG_FOCUS),
        point_size,
    )
    add_plotly_mask_trace(
        fig, row, col, X, other,
        f"{trace_prefix}: otros dientes",
        hex_to_plotly_rgba(COLOR_OTHER, ALPHA_OTHER_FOCUS),
        point_size,
    )
    if len(neigh_ids) >= 1:
        add_plotly_mask_trace(
            fig, row, col, X, n1,
            f"{trace_prefix}: vecino izq. {neigh_ids[0]}",
            hex_to_plotly_rgba(COLOR_LEFT_NEIGH, ALPHA_FOCAL),
            point_size * 1.12,
        )
    if len(neigh_ids) >= 2:
        add_plotly_mask_trace(
            fig, row, col, X, n2,
            f"{trace_prefix}: vecino der. {neigh_ids[1]}",
            hex_to_plotly_rgba(COLOR_RIGHT_NEIGH, ALPHA_FOCAL),
            point_size * 1.12,
        )

    add_plotly_mask_trace(
        fig, row, col, X, d21,
        f"{trace_prefix}: d21",
        hex_to_plotly_rgba(COLOR_D21, ALPHA_FOCAL),
        point_size * 1.18,
    )


def add_plotly_d21_error_points(fig, row, col, X, pred, gt, d21_class, neighbors, point_size):
    X = finite_points(X)
    pred = np.asarray(pred).reshape(-1).astype(np.int64)
    gt = np.asarray(gt).reshape(-1).astype(np.int64)

    n = min(X.shape[0], pred.shape[0], gt.shape[0])
    X, pred, gt = X[:n], pred[:n], gt[:n]

    neigh_ids = [v for _, v in neighbors]

    bg = gt == 0
    d21_tp = (pred == int(d21_class)) & (gt == int(d21_class))
    d21_err = ((pred == int(d21_class)) != (gt == int(d21_class)))
    n1 = gt == neigh_ids[0] if len(neigh_ids) >= 1 else np.zeros_like(bg, dtype=bool)
    n2 = gt == neigh_ids[1] if len(neigh_ids) >= 2 else np.zeros_like(bg, dtype=bool)
    other = (gt != 0) & (~d21_tp) & (~d21_err) & (~n1) & (~n2)

    add_plotly_mask_trace(
        fig, row, col, X, bg,
        "d21_error: background",
        hex_to_plotly_rgba(COLOR_BG, ALPHA_BG_FOCUS),
        point_size,
    )
    add_plotly_mask_trace(
        fig, row, col, X, other,
        "d21_error: otros dientes",
        hex_to_plotly_rgba(COLOR_OTHER, ALPHA_OTHER_FOCUS),
        point_size,
    )
    if len(neigh_ids) >= 1:
        add_plotly_mask_trace(
            fig, row, col, X, n1,
            f"d21_error: vecino izq. {neigh_ids[0]}",
            hex_to_plotly_rgba(COLOR_LEFT_NEIGH, ALPHA_FOCAL),
            point_size * 1.08,
        )
    if len(neigh_ids) >= 2:
        add_plotly_mask_trace(
            fig, row, col, X, n2,
            f"d21_error: vecino der. {neigh_ids[1]}",
            hex_to_plotly_rgba(COLOR_RIGHT_NEIGH, ALPHA_FOCAL),
            point_size * 1.08,
        )
    add_plotly_mask_trace(
        fig, row, col, X, d21_tp,
        "d21_error: TP d21",
        hex_to_plotly_rgba(COLOR_TP, ALPHA_TP),
        point_size * 1.18,
    )
    add_plotly_mask_trace(
        fig, row, col, X, d21_err,
        "d21_error: FP/FN d21",
        hex_to_plotly_rgba(COLOR_ERROR, ALPHA_ERROR),
        point_size * 1.35,
    )


def update_plotly_layout(fig, plotly_zoom):
    camera = dict(
        eye=dict(x=1.35 / plotly_zoom, y=-1.65 / plotly_zoom, z=0.90 / plotly_zoom),
        up=dict(x=0, y=0, z=1),
    )

    for k in fig.layout:
        if str(k).startswith("scene"):
            fig.layout[k].update(
                xaxis=dict(visible=False, showgrid=False, zeroline=False),
                yaxis=dict(visible=False, showgrid=False, zeroline=False),
                zaxis=dict(visible=False, showgrid=False, zeroline=False),
                bgcolor="white",
                aspectmode="data",
                camera=camera,
            )


def make_plotly_html(data, out_html, d21_class, neighbors, mesh_opacity,
                     point_size, plotly_zoom, show_classes, title=""):
    if not HAS_PLOTLY:
        print("[WARN] Plotly no está instalado. No se exportará HTML.")
        return None

    out_html = Path(out_html)
    ensure_dir(out_html.parent)

    X, V, F = data["X"], data["V"], data["F"]
    pred, gt = data["pred"], data["gt"]
    colors, labels = data["colors"], data["labels"]
    pretty = data["pretty"]

    fig = make_subplots(
        rows=2,
        cols=3,
        specs=[[{"type": "scene"} for _ in range(3)] for _ in range(2)],
        subplot_titles=[
            "GT multiclase",
            f"{pretty} predicción",
            "Errores multiclase",
            "GT d21 + vecinos",
            f"{pretty} d21 + vecinos",
            "Error d21: verde TP / rojo FP-FN",
        ],
        horizontal_spacing=0.010,
        vertical_spacing=0.030,
    )

    for r in [1, 2]:
        for c in [1, 2, 3]:
            add_plotly_mesh(fig, r, c, V, F, mesh_opacity=mesh_opacity)

    add_plotly_points_by_class(fig, 1, 1, X, labels["gt_all"], colors["gt_all"], "GT", point_size, show_classes)
    add_plotly_points_by_class(fig, 1, 2, X, labels["pred_all"], colors["pred_all"], "Pred", point_size, show_classes)
    add_plotly_error_points(fig, 1, 3, X, pred, gt, point_size)
    add_plotly_focus_points(fig, 2, 1, X, gt, d21_class, neighbors, "GT_focus", point_size * 1.08)
    add_plotly_focus_points(fig, 2, 2, X, pred, d21_class, neighbors, "Pred_focus", point_size * 1.08)
    add_plotly_d21_error_points(fig, 2, 3, X, pred, gt, d21_class, neighbors, point_size * 1.08)

    fig.update_layout(
        title=title or f"Vista interactiva filtrable — {pretty}",
        height=1120,
        width=2150,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black", size=13),
        margin=dict(l=5, r=440, t=80, b=5),
        legend=dict(
            x=1.012,
            y=0.985,
            xanchor="left",
            yanchor="top",
            orientation="v",
            bgcolor="rgba(255,255,255,0.97)",
            bordercolor="rgba(100,100,100,0.80)",
            borderwidth=1.5,
            font=dict(size=15, color="black"),
            itemsizing="constant",
            itemwidth=115,
            tracegroupgap=7,
        ),
    )

    update_plotly_layout(fig, plotly_zoom=plotly_zoom)
    fig.write_html(str(out_html), include_plotlyjs="cdn")

    return out_html


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--case_dir", required=True)
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--out_html", default="")
    ap.add_argument("--out_individual_dir", default="")

    ap.add_argument("--models", default="pointnet")
    ap.add_argument("--d21_class", type=int, default=D21_DEFAULT)
    ap.add_argument("--neighbor_teeth", default=DEFAULT_NEIGHBORS)

    ap.add_argument(
        "--align_mode",
        default="dental_pca_refine",
        choices=[
            "none",
            "robust_bbox",
            "dental_bbox",
            "dental_bbox_axiswise",
            "dental_pca",
            "dental_pca_refine",
            "dental_bbox_icp",
            "dental_pca_icp",
            "dental_icp",
        ],
    )

    ap.add_argument("--mesh_top_q", type=float, default=0.45)
    ap.add_argument("--show_classes", default="all")

    ap.add_argument("--icp_iter", type=int, default=35)
    ap.add_argument("--icp_max_corr", type=float, default=0.08)
    ap.add_argument("--icp_trim_q", type=float, default=0.85)
    ap.add_argument("--icp_allow_scaling", action="store_true")
    ap.add_argument(
        "--icp_focus",
        default="foreground",
        choices=["foreground", "d21", "focus", "neighbors", "d21_neighbors"],
    )

    ap.add_argument("--zoom", type=float, default=1.55)
    ap.add_argument("--zoom_focus", type=float, default=1.85)
    ap.add_argument("--elev", type=float, default=25.0)
    ap.add_argument("--azim", type=float, default=-55.0)

    ap.add_argument("--dpi", type=int, default=500)
    ap.add_argument("--point_size", type=float, default=8.8)
    ap.add_argument("--mesh_alpha", type=float, default=0.07)

    ap.add_argument("--plotly_mesh_opacity", type=float, default=0.05)
    ap.add_argument("--plotly_point_size", type=float, default=4.2)
    ap.add_argument("--plotly_zoom", type=float, default=1.28)

    ap.add_argument("--title", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no_plotly", action="store_true")
    ap.add_argument("--no_individual", action="store_true")

    args = ap.parse_args()

    case_dir = Path(args.case_dir).resolve()
    out_png = Path(args.out_png).resolve()

    if out_png.exists() and not args.force:
        raise FileExistsError(f"Ya existe {out_png}. Use --force para sobrescribir.")

    models = parse_models(args.models)
    model_key = models[0]
    neighbors = parse_neighbor_teeth(args.neighbor_teeth)
    show_classes = parse_class_filter(args.show_classes)

    print("[INFO] case_dir:", case_dir)
    print("[INFO] model:", model_key)
    print("[INFO] align_mode:", args.align_mode)
    print("[INFO] mesh_top_q:", args.mesh_top_q)
    print("[INFO] icp_iter:", args.icp_iter)
    print("[INFO] icp_max_corr:", args.icp_max_corr)
    print("[INFO] icp_trim_q:", args.icp_trim_q)
    print("[INFO] icp_allow_scaling:", args.icp_allow_scaling)
    print("[INFO] icp_focus:", args.icp_focus)
    print("[INFO] point_size:", args.point_size)
    print("[INFO] plotly_point_size:", args.plotly_point_size)
    print("[INFO] mesh_alpha:", args.mesh_alpha)
    print("[INFO] plotly_mesh_opacity:", args.plotly_mesh_opacity)
    print("[INFO] show_classes:", show_classes if show_classes is not None else "all")

    data = prepare_case_data(
        case_dir=case_dir,
        model_key=model_key,
        d21_class=args.d21_class,
        neighbors=neighbors,
        align_mode=args.align_mode,
        mesh_top_q=args.mesh_top_q,
        icp_iter=args.icp_iter,
        icp_max_corr=args.icp_max_corr,
        icp_trim_q=args.icp_trim_q,
        icp_allow_scaling=args.icp_allow_scaling,
        icp_focus=args.icp_focus,
    )

    make_summary_png(
        data=data,
        out_png=out_png,
        d21_class=args.d21_class,
        neighbors=neighbors,
        elev=args.elev,
        azim=args.azim,
        zoom=args.zoom,
        zoom_focus=args.zoom_focus,
        mesh_alpha=args.mesh_alpha,
        point_size=args.point_size,
        dpi=args.dpi,
        show_classes=show_classes,
        title=args.title,
    )
    print(f"[OK] PNG resumen: {out_png}")

    individual_saved = []
    if not args.no_individual:
        out_ind = (
            Path(args.out_individual_dir).resolve()
            if args.out_individual_dir
            else out_png.parent / "individual_panels_v17"
        )

        individual_saved = make_individual_pngs(
            data=data,
            out_dir=out_ind,
            elev=args.elev,
            azim=args.azim,
            zoom=args.zoom,
            zoom_focus=args.zoom_focus,
            mesh_alpha=args.mesh_alpha,
            point_size=args.point_size,
            dpi=args.dpi,
            show_classes=show_classes,
            d21_class=args.d21_class,
            neighbors=neighbors,
        )
        print(f"[OK] Paneles individuales: {out_ind}")

    out_html = None
    if not args.no_plotly:
        out_html = Path(args.out_html).resolve() if args.out_html else out_png.with_suffix(".html")

        make_plotly_html(
            data=data,
            out_html=out_html,
            d21_class=args.d21_class,
            neighbors=neighbors,
            mesh_opacity=args.plotly_mesh_opacity,
            point_size=args.plotly_point_size,
            plotly_zoom=args.plotly_zoom,
            show_classes=show_classes,
            title=args.title,
        )
        print(f"[OK] HTML Plotly: {out_html}")

    summary = {
        "case_dir": str(case_dir),
        "out_png": str(out_png),
        "out_html": str(out_html) if out_html else "",
        "individual_panels": individual_saved,
        "model": model_key,
        "d21_class": int(args.d21_class),
        "neighbors": neighbors,
        "align_mode": args.align_mode,
        "mesh_top_q": float(args.mesh_top_q),
        "icp_iter": int(args.icp_iter),
        "icp_max_corr": float(args.icp_max_corr),
        "icp_trim_q": float(args.icp_trim_q),
        "icp_allow_scaling": bool(args.icp_allow_scaling),
        "icp_focus": str(args.icp_focus),
        "point_size": float(args.point_size),
        "plotly_point_size": float(args.plotly_point_size),
        "mesh_alpha": float(args.mesh_alpha),
        "plotly_mesh_opacity": float(args.plotly_mesh_opacity),
        "show_classes": show_classes if show_classes is not None else "all",
        "raw_mesh_path": data.get("raw_path", ""),
        "align_meta": data.get("align_meta", {}),
        "metrics": data.get("metrics", {}),
    }

    save_json(summary, out_png.with_suffix(".summary.json"))
    print(f"[OK] Summary JSON: {out_png.with_suffix('.summary.json')}")
    print("\nListo ✅")


if __name__ == "__main__":
    main()