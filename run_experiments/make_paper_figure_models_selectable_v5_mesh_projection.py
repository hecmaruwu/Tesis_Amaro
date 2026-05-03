#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_paper_figure_models_selectable_v5_mesh_projection.py

Figura paper-level para segmentación dental 3D.

Mejora principal respecto a v4:
- Usa la malla raw como referencia en TODOS los paneles.
- Proyecta etiquetas de la nube 8192 hacia la malla usando nearest neighbor.
- Permite visualizar:
    1) raw mesh gris
    2) GT proyectado en malla
    3) diente 21 GT proyectado en malla
    4) predicción del modelo proyectada en malla
    5) errores proyectados en malla
    6) diente 21 predicho proyectado en malla
- Mantiene overlay opcional de puntos 8192.
- Exporta PNG de alta definición.
- Exporta HTML Plotly opcional.

Entrada esperada:
case_dir/
  meta.json
  common/
    bundle_case_data.npz
  pointnet/
    pred_labels.npy
    gt_labels.npy
    xyz_labels.npy

Compatible también con:
  pointnetpp/
  dgcnn/
  pointnettransformer/
"""

import argparse
import json
import re
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
# -------------------- MODELOS / ALIASES ---------------------
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


# ============================================================
# -------------------- COLORES -------------------------------
# ============================================================

TAB20 = plt.get_cmap("tab20", 20)

COLOR_RAW = np.array([0.72, 0.72, 0.72, 1.00], dtype=np.float32)
COLOR_BG = np.array([0.82, 0.82, 0.82, 0.45], dtype=np.float32)
COLOR_CORRECT = np.array([0.78, 0.78, 0.78, 0.45], dtype=np.float32)
COLOR_ERROR = np.array([1.00, 0.02, 0.01, 1.00], dtype=np.float32)
COLOR_D21 = np.array([0.00, 0.20, 1.00, 1.00], dtype=np.float32)
COLOR_TOOTH_OTHER = np.array([0.42, 0.42, 0.42, 0.55], dtype=np.float32)
COLOR_POINT_OVERLAY = np.array([0.05, 0.05, 0.05, 0.20], dtype=np.float32)


# ============================================================
# -------------------- UTILS ---------------------------------
# ============================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Dict:
    if not Path(path).exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: Path):
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def sanitize_model_name(name: str) -> str:
    name = str(name).strip().lower()
    return MODEL_DIR_ALIASES.get(name, name)


def pretty_model_name(name: str) -> str:
    key = sanitize_model_name(name)
    return MODEL_ALIASES.get(key, key)


def parse_models(s: str) -> List[str]:
    if not s:
        return DEFAULT_MODELS
    out = []
    for x in s.replace(",", " ").split():
        x = sanitize_model_name(x)
        if x:
            out.append(x)
    return out or DEFAULT_MODELS


def finite_points(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != 3:
        X = X.reshape(-1, 3)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X.astype(np.float32)


def unwrap_object_array(arr) -> np.ndarray:
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)

    if arr.dtype != object:
        return arr

    if arr.shape == ():
        try:
            arr = arr.item()
            arr = np.asarray(arr)
        except Exception:
            pass

    if isinstance(arr, np.ndarray) and arr.dtype == object and arr.size == 1:
        try:
            arr = arr.reshape(-1)[0]
            arr = np.asarray(arr)
        except Exception:
            pass

    if isinstance(arr, np.ndarray) and arr.dtype == object:
        try:
            arr = np.asarray(arr.tolist())
        except Exception:
            try:
                arr = np.concatenate([np.asarray(x).reshape(-1) for x in arr.reshape(-1)], axis=0)
            except Exception:
                arr = np.asarray(arr)

    return np.asarray(arr)


def load_any_array(path: Path, preferred_keys: Optional[List[str]] = None) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe archivo: {path}")

    preferred_keys = preferred_keys or [
        "arr_0", "pred", "prediction", "pred_labels",
        "labels", "gt", "y", "Y", "xyz", "points", "X"
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
        preferred_keys=["gt", "labels", "gt_labels", "y", "Y", "arr_0"]
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


def fix_length_match(
    X: np.ndarray,
    pred: np.ndarray,
    gt: Optional[np.ndarray] = None,
    strict: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    X = finite_points(X)
    pred = np.asarray(pred).reshape(-1).astype(np.int64)
    gt_arr = None if gt is None else np.asarray(gt).reshape(-1).astype(np.int64)

    lengths = [X.shape[0], pred.shape[0]]
    if gt_arr is not None:
        lengths.append(gt_arr.shape[0])

    min_len = min(lengths)
    max_len = max(lengths)

    if min_len != max_len:
        msg = f"Longitudes distintas: X={X.shape[0]}, pred={pred.shape[0]}, gt={None if gt_arr is None else gt_arr.shape[0]}"
        if strict:
            raise ValueError(msg)
        print(f"[WARN] {msg}. Recortando a {min_len}.")

    X = X[:min_len]
    pred = pred[:min_len]
    if gt_arr is not None:
        gt_arr = gt_arr[:min_len]

    return X, pred, gt_arr


# ============================================================
# -------------------- CARGA CASE -----------------------------
# ============================================================

def load_common_bundle(case_dir: Path) -> Dict[str, np.ndarray]:
    bundle_path = Path(case_dir) / "common" / "bundle_case_data.npz"
    if not bundle_path.exists():
        return {}

    data = np.load(bundle_path, allow_pickle=True)
    out = {}

    for k in data.keys():
        out[k] = np.asarray(unwrap_object_array(data[k]))

    return out


def load_model_case(case_dir: Path, model: str, require_pred_labels: bool = False) -> Dict:
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
            "model_dir": str(model_dir),
        }

    pred = load_pred_labels(pred_path)

    gt = None
    if gt_path.exists():
        gt = load_gt_labels(gt_path)

    xyz = None
    if xyz_path.exists():
        xyz = load_xyz(xyz_path)

    if xyz is not None:
        xyz, pred, gt = fix_length_match(xyz, pred, gt, strict=False)

    return {
        "model": model_key,
        "pretty": pretty_model_name(model_key),
        "available": True,
        "model_dir": str(model_dir),
        "pred": pred,
        "gt": gt,
        "xyz": xyz,
        "pred_path": str(pred_path),
        "gt_path": str(gt_path) if gt_path.exists() else "",
        "xyz_path": str(xyz_path) if xyz_path.exists() else "",
    }


# ============================================================
# -------------------- RAW MESH -------------------------------
# ============================================================

def find_raw_mesh_from_meta(case_dir: Path) -> Optional[Path]:
    meta = read_json(Path(case_dir) / "meta.json")
    raw = meta.get("raw_mesh_path", "")
    if raw:
        p = Path(raw)
        if p.exists():
            return p
    return None


def load_raw_mesh_vertices_faces(raw_mesh_path: Optional[Path]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if raw_mesh_path is None:
        return None, None

    if not HAS_TRIMESH:
        print("[WARN] trimesh no está disponible. No se cargará raw mesh.")
        return None, None

    raw_mesh_path = Path(raw_mesh_path)
    if not raw_mesh_path.exists():
        print(f"[WARN] Raw mesh no existe: {raw_mesh_path}")
        return None, None

    try:
        loaded = trimesh.load(str(raw_mesh_path), force="mesh", process=False)
    except Exception:
        try:
            loaded = trimesh.load(str(raw_mesh_path), force="mesh", process=True)
        except Exception as e:
            print(f"[WARN] No pude cargar raw mesh {raw_mesh_path}: {e}")
            return None, None

    if isinstance(loaded, trimesh.Scene):
        geos = list(loaded.dump().geometry.values())
        if not geos:
            return None, None
        loaded = trimesh.util.concatenate(geos)

    if not isinstance(loaded, trimesh.Trimesh):
        return None, None

    V = np.asarray(loaded.vertices, dtype=np.float32)
    F = np.asarray(loaded.faces, dtype=np.int64) if loaded.faces is not None else None

    if F is not None and F.size == 0:
        F = None

    V = finite_points(V)
    return V, F


# ============================================================
# -------------------- ALINEACIÓN -----------------------------
# ============================================================

def normalize_unit_sphere(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    X = finite_points(X)
    center = X.mean(axis=0, keepdims=True)
    Xc = X - center
    scale = np.linalg.norm(Xc, axis=1).max()
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return (Xc / scale).astype(np.float32), center.reshape(3).astype(np.float32), float(scale)


def robust_bounds(X: np.ndarray, q_low=0.01, q_high=0.99) -> Tuple[np.ndarray, np.ndarray]:
    X = finite_points(X)
    lo = np.quantile(X, q_low, axis=0)
    hi = np.quantile(X, q_high, axis=0)
    return lo.astype(np.float32), hi.astype(np.float32)


def bbox_center_scale(X: np.ndarray, robust=True) -> Tuple[np.ndarray, float]:
    X = finite_points(X)

    if robust:
        lo, hi = robust_bounds(X)
    else:
        lo, hi = X.min(axis=0), X.max(axis=0)

    center = 0.5 * (lo + hi)
    diag = np.linalg.norm(hi - lo)

    if not np.isfinite(diag) or diag <= 0:
        diag = 1.0

    return center.astype(np.float32), float(diag)


def align_source_to_target_bbox(
    X_source: np.ndarray,
    X_target: np.ndarray,
    robust: bool = True,
) -> Tuple[np.ndarray, Dict]:
    X_source = finite_points(X_source)
    X_target = finite_points(X_target)

    c_src, s_src = bbox_center_scale(X_source, robust=robust)
    c_tgt, s_tgt = bbox_center_scale(X_target, robust=robust)

    X_aligned = (X_source - c_src.reshape(1, 3)) / (s_src + 1e-9)
    X_aligned = X_aligned * s_tgt + c_tgt.reshape(1, 3)

    meta = {
        "mode": "robust_bbox" if robust else "bbox",
        "source_center": c_src.tolist(),
        "source_scale_diag": float(s_src),
        "target_center": c_tgt.tolist(),
        "target_scale_diag": float(s_tgt),
    }

    return X_aligned.astype(np.float32), meta


def align_raw_mesh(
    raw_vertices: Optional[np.ndarray],
    raw_faces: Optional[np.ndarray],
    target_xyz: np.ndarray,
    align_mode: str = "robust_bbox",
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict]:
    if raw_vertices is None:
        return None, raw_faces, {"mode": "missing_raw"}

    raw_vertices = finite_points(raw_vertices)
    target_xyz = finite_points(target_xyz)

    if align_mode == "none":
        return raw_vertices, raw_faces, {"mode": "none"}

    if align_mode == "unit_sphere":
        V_norm, c, s = normalize_unit_sphere(raw_vertices)
        return V_norm, raw_faces, {
            "mode": "unit_sphere",
            "center": c.tolist(),
            "scale": float(s),
        }

    if align_mode == "bbox":
        V_aligned, meta = align_source_to_target_bbox(raw_vertices, target_xyz, robust=False)
        return V_aligned, raw_faces, meta

    if align_mode == "robust_bbox":
        V_aligned, meta = align_source_to_target_bbox(raw_vertices, target_xyz, robust=True)
        return V_aligned, raw_faces, meta

    raise ValueError(f"align_mode no reconocido: {align_mode}")


# ============================================================
# -------------------- PROYECCIÓN A MALLA --------------------
# ============================================================

def labels_points_to_vertices(
    V: np.ndarray,
    X_points: np.ndarray,
    labels_points: np.ndarray,
) -> np.ndarray:
    """
    Proyecta etiquetas de puntos 8192 a vértices de malla con nearest neighbor.
    """
    if not HAS_SCIPY:
        raise RuntimeError("scipy no disponible. Se requiere scipy.spatial.cKDTree para proyectar a malla.")

    V = finite_points(V)
    X_points = finite_points(X_points)
    labels_points = np.asarray(labels_points).reshape(-1).astype(np.int64)

    X_points, labels_points, _ = fix_length_match(
        X_points,
        labels_points,
        None,
        strict=False,
    )

    tree = cKDTree(X_points)
    _, idx = tree.query(V, k=1, workers=-1)
    vertex_labels = labels_points[idx].astype(np.int64)

    return vertex_labels


def vertex_labels_to_face_labels(F: Optional[np.ndarray], vertex_labels: np.ndarray) -> Optional[np.ndarray]:
    """
    Convierte etiquetas por vértice a etiquetas por cara usando mayoría.
    """
    if F is None:
        return None

    F = np.asarray(F, dtype=np.int64)
    vertex_labels = np.asarray(vertex_labels).reshape(-1).astype(np.int64)

    face_labels = np.zeros((F.shape[0],), dtype=np.int64)

    for i in range(F.shape[0]):
        labs = vertex_labels[F[i]]
        vals, counts = np.unique(labs, return_counts=True)
        face_labels[i] = vals[np.argmax(counts)]

    return face_labels


def point_errors_to_vertices(
    V: np.ndarray,
    X_points: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray,
) -> np.ndarray:
    """
    Proyecta error correcto/incorrecto hacia vértices.
    0 = correcto
    1 = error
    """
    if not HAS_SCIPY:
        raise RuntimeError("scipy no disponible.")

    V = finite_points(V)
    X_points, pred, gt = fix_length_match(X_points, pred, gt, strict=False)

    err = (pred != gt).astype(np.int64)

    tree = cKDTree(X_points)
    _, idx = tree.query(V, k=1, workers=-1)
    return err[idx].astype(np.int64)


# ============================================================
# -------------------- COLORES MALLA -------------------------
# ============================================================

def labels_to_rgba(labels: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    colors = TAB20(labels % 20).astype(np.float32)

    bg = labels == 0
    colors[bg] = COLOR_BG

    colors[:, 3] = alpha
    colors[bg, 3] = min(alpha, 0.42)

    return colors


def d21_to_rgba(labels: np.ndarray, d21_class: int, alpha_mesh: float = 1.0) -> np.ndarray:
    labels = np.asarray(labels).reshape(-1).astype(np.int64)

    colors = np.zeros((labels.shape[0], 4), dtype=np.float32)
    colors[:] = np.array([0.80, 0.80, 0.80, 0.25], dtype=np.float32)

    fg = labels != 0
    colors[fg] = np.array([0.45, 0.45, 0.45, 0.45], dtype=np.float32)

    d21 = labels == int(d21_class)
    colors[d21] = COLOR_D21

    colors[:, 3] *= alpha_mesh
    colors[d21, 3] = 1.0

    return colors


def errors_to_rgba(error_flags: np.ndarray, alpha_mesh: float = 1.0) -> np.ndarray:
    error_flags = np.asarray(error_flags).reshape(-1).astype(np.int64)

    colors = np.zeros((error_flags.shape[0], 4), dtype=np.float32)
    colors[error_flags == 0] = COLOR_CORRECT
    colors[error_flags == 1] = COLOR_ERROR

    colors[:, 3] *= alpha_mesh
    colors[error_flags == 1, 3] = 1.0

    return colors


def raw_to_rgba(n: int, alpha_mesh: float = 1.0) -> np.ndarray:
    colors = np.tile(COLOR_RAW.reshape(1, 4), (int(n), 1))
    colors[:, 3] = alpha_mesh
    return colors.astype(np.float32)


# ============================================================
# -------------------- MÉTRICAS -------------------------------
# ============================================================

def compute_simple_metrics(pred: np.ndarray, gt: Optional[np.ndarray], d21_class: int = 8) -> Dict:
    if gt is None:
        return {}

    pred = np.asarray(pred).reshape(-1)
    gt = np.asarray(gt).reshape(-1)

    n = min(len(pred), len(gt))
    pred = pred[:n]
    gt = gt[:n]

    acc_all = float((pred == gt).mean()) if n > 0 else 0.0

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
        "acc_all": acc_all,
        "acc_no_bg": acc_no_bg,
        "d21_precision": float(prec),
        "d21_recall": float(rec),
        "d21_f1": float(f1),
        "d21_iou": float(iou),
        "n_points": int(n),
    }


# ============================================================
# -------------------- RENDER MATPLOTLIB HD ------------------
# ============================================================

def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)

    radius = 0.5 * max([x_range, y_range, z_range, 1e-6])

    ax.set_xlim3d([x_mid - radius, x_mid + radius])
    ax.set_ylim3d([y_mid - radius, y_mid + radius])
    ax.set_zlim3d([z_mid - radius, z_mid + radius])


def apply_view(ax, elev: float = 25, azim: float = -55, zoom: float = 1.0):
    ax.view_init(elev=elev, azim=azim)
    try:
        ax.dist = 10 / max(float(zoom), 1e-6)
    except Exception:
        pass
    ax.set_axis_off()
    set_axes_equal(ax)


def facecolors_from_vertexcolors(F, vertex_colors):
    if F is None:
        return None
    F = np.asarray(F, dtype=np.int64)
    vertex_colors = np.asarray(vertex_colors, dtype=np.float32)
    return vertex_colors[F].mean(axis=1)


def plot_mesh_panel(
    ax,
    V: Optional[np.ndarray],
    F: Optional[np.ndarray],
    vertex_colors: Optional[np.ndarray],
    title: str,
    elev: float,
    azim: float,
    zoom: float,
    mesh_alpha: float,
    X_overlay: Optional[np.ndarray] = None,
    labels_overlay: Optional[np.ndarray] = None,
    overlay_points: bool = False,
    overlay_point_size: float = 0.9,
    overlay_alpha: float = 0.22,
):
    if V is None:
        ax.text2D(0.5, 0.5, "Malla no disponible", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()
        return

    V = finite_points(V)

    if vertex_colors is None:
        vertex_colors = raw_to_rgba(V.shape[0], alpha_mesh=mesh_alpha)
    else:
        vertex_colors = np.asarray(vertex_colors, dtype=np.float32).copy()
        vertex_colors[:, 3] = np.clip(vertex_colors[:, 3] * mesh_alpha, 0.0, 1.0)

    if F is not None and len(F) > 0:
        F = np.asarray(F, dtype=np.int64)
        polys = V[F]
        facecolors = facecolors_from_vertexcolors(F, vertex_colors)
        coll = Poly3DCollection(
            polys,
            facecolors=facecolors,
            linewidths=0.02,
            edgecolors=(0.20, 0.20, 0.20, 0.04),
            antialiased=True,
        )
        ax.add_collection3d(coll)
        ax.auto_scale_xyz(V[:, 0], V[:, 1], V[:, 2])
    else:
        ax.scatter(
            V[:, 0], V[:, 1], V[:, 2],
            c=vertex_colors,
            s=0.35,
            linewidths=0,
            depthshade=False,
        )

    if overlay_points and X_overlay is not None:
        Xo = finite_points(X_overlay)
        if labels_overlay is not None:
            co = labels_to_rgba(labels_overlay, alpha=overlay_alpha)
        else:
            co = np.tile(COLOR_POINT_OVERLAY.reshape(1, 4), (Xo.shape[0], 1))
            co[:, 3] = overlay_alpha

        ax.scatter(
            Xo[:, 0], Xo[:, 1], Xo[:, 2],
            c=co,
            s=float(overlay_point_size),
            linewidths=0,
            depthshade=False,
        )

    ax.set_title(title, fontsize=10, pad=2)
    apply_view(ax, elev=elev, azim=azim, zoom=zoom)


def make_png_figure(
    case_dir: Path,
    out_png: Path,
    models: List[str],
    d21_class: int,
    align_mode: str,
    zoom: float,
    elev: float,
    azim: float,
    dpi: int,
    figsize_scale: float,
    mesh_alpha: float,
    overlay_points: bool,
    overlay_point_size: float,
    overlay_alpha: float,
    require_pred_labels: bool,
    title: str,
) -> Dict:
    case_dir = Path(case_dir)
    out_png = Path(out_png)
    ensure_dir(out_png.parent)

    common = load_common_bundle(case_dir)
    meta = read_json(case_dir / "meta.json")

    loaded_models = []
    for m in models:
        md = load_model_case(case_dir, m, require_pred_labels=require_pred_labels)
        if md.get("available"):
            loaded_models.append(md)
        else:
            print(f"[WARN] Modelo omitido: {m} | {md.get('reason')}")

    if not loaded_models:
        raise RuntimeError("No hay modelos disponibles con pred_labels.npy.")

    base_xyz = loaded_models[0].get("xyz")
    if base_xyz is None:
        if "xyz_8192" in common:
            base_xyz = finite_points(common["xyz_8192"])
        else:
            raise RuntimeError("No hay xyz_labels.npy ni common/bundle_case_data.npz con xyz_8192.")

    gt = loaded_models[0].get("gt")
    if gt is None and "y_8192" in common:
        gt = np.asarray(common["y_8192"]).reshape(-1).astype(np.int64)

    if gt is not None:
        base_xyz, dummy, gt = fix_length_match(
            base_xyz,
            np.zeros(base_xyz.shape[0], dtype=np.int64),
            gt,
            strict=False,
        )

    raw_mesh_path = find_raw_mesh_from_meta(case_dir)
    raw_V, raw_F = load_raw_mesh_vertices_faces(raw_mesh_path)

    raw_V_aligned, raw_F, align_meta = align_raw_mesh(
        raw_vertices=raw_V,
        raw_faces=raw_F,
        target_xyz=base_xyz,
        align_mode=align_mode,
    )

    if raw_V_aligned is None:
        raise RuntimeError("No se pudo cargar/alinear la malla raw.")

    # Proyección GT a malla
    gt_vertex_labels = None
    gt_vertex_colors = None
    gt_d21_colors = None

    if gt is not None:
        gt_vertex_labels = labels_points_to_vertices(raw_V_aligned, base_xyz, gt)
        gt_vertex_colors = labels_to_rgba(gt_vertex_labels, alpha=1.0)
        gt_d21_colors = d21_to_rgba(gt_vertex_labels, d21_class=d21_class, alpha_mesh=1.0)

    n_models = len(loaded_models)
    n_cols = 3
    n_rows = 1 + n_models

    fig_w = 5.8 * n_cols * float(figsize_scale)
    fig_h = 4.8 * n_rows * float(figsize_scale)

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=int(dpi))
    gs = GridSpec(n_rows, n_cols, figure=fig, wspace=0.02, hspace=0.08)

    sample_title = meta.get("sample_name", case_dir.name)
    fig.suptitle(
        title or f"Paper figure mesh-projected — {sample_title}",
        fontsize=16,
        y=0.995,
    )

    # Fila 0: raw, GT, GT d21
    ax_raw = fig.add_subplot(gs[0, 0], projection="3d")
    plot_mesh_panel(
        ax_raw,
        raw_V_aligned,
        raw_F,
        raw_to_rgba(raw_V_aligned.shape[0], alpha_mesh=1.0),
        title=f"Raw mesh alineada\n{align_mode}",
        elev=elev,
        azim=azim,
        zoom=zoom,
        mesh_alpha=1.0,
        X_overlay=base_xyz,
        labels_overlay=None,
        overlay_points=overlay_points,
        overlay_point_size=overlay_point_size,
        overlay_alpha=overlay_alpha,
    )

    ax_gt = fig.add_subplot(gs[0, 1], projection="3d")
    if gt_vertex_colors is not None:
        plot_mesh_panel(
            ax_gt,
            raw_V_aligned,
            raw_F,
            gt_vertex_colors,
            title="GT proyectado en malla",
            elev=elev,
            azim=azim,
            zoom=zoom,
            mesh_alpha=mesh_alpha,
            X_overlay=base_xyz,
            labels_overlay=gt,
            overlay_points=overlay_points,
            overlay_point_size=overlay_point_size,
            overlay_alpha=overlay_alpha,
        )
    else:
        plot_mesh_panel(
            ax_gt,
            raw_V_aligned,
            raw_F,
            raw_to_rgba(raw_V_aligned.shape[0], alpha_mesh=mesh_alpha),
            title="GT no disponible",
            elev=elev,
            azim=azim,
            zoom=zoom,
            mesh_alpha=mesh_alpha,
        )

    ax_gt_d21 = fig.add_subplot(gs[0, 2], projection="3d")
    if gt_d21_colors is not None:
        plot_mesh_panel(
            ax_gt_d21,
            raw_V_aligned,
            raw_F,
            gt_d21_colors,
            title=f"GT diente 21 proyectado\nclase interna={d21_class}",
            elev=elev,
            azim=azim,
            zoom=zoom,
            mesh_alpha=mesh_alpha,
            X_overlay=base_xyz,
            labels_overlay=gt,
            overlay_points=overlay_points,
            overlay_point_size=overlay_point_size,
            overlay_alpha=overlay_alpha,
        )
    else:
        plot_mesh_panel(
            ax_gt_d21,
            raw_V_aligned,
            raw_F,
            raw_to_rgba(raw_V_aligned.shape[0], alpha_mesh=mesh_alpha),
            title="GT d21 no disponible",
            elev=elev,
            azim=azim,
            zoom=zoom,
            mesh_alpha=mesh_alpha,
        )

    metrics_summary = {}

    for r, md in enumerate(loaded_models, start=1):
        model_name = md["model"]
        pretty = md.get("pretty", pretty_model_name(model_name))

        X = md.get("xyz")
        if X is None:
            X = base_xyz.copy()

        pred = md["pred"]
        gt_model = md.get("gt", gt)

        X, pred, gt_model = fix_length_match(X, pred, gt_model, strict=False)

        metrics_summary[model_name] = compute_simple_metrics(pred, gt_model, d21_class=d21_class)

        pred_vertex_labels = labels_points_to_vertices(raw_V_aligned, X, pred)
        pred_vertex_colors = labels_to_rgba(pred_vertex_labels, alpha=1.0)
        pred_d21_vertex_colors = d21_to_rgba(pred_vertex_labels, d21_class=d21_class, alpha_mesh=1.0)

        err_vertex_colors = None
        if gt_model is not None:
            vertex_errors = point_errors_to_vertices(raw_V_aligned, X, pred, gt_model)
            err_vertex_colors = errors_to_rgba(vertex_errors, alpha_mesh=1.0)

        mtxt = ""
        if metrics_summary[model_name]:
            mtxt = (
                f"\nacc_nb={metrics_summary[model_name]['acc_no_bg']:.3f}"
                f" | d21_f1={metrics_summary[model_name]['d21_f1']:.3f}"
            )

        # Predicción en malla
        ax_pred = fig.add_subplot(gs[r, 0], projection="3d")
        plot_mesh_panel(
            ax_pred,
            raw_V_aligned,
            raw_F,
            pred_vertex_colors,
            title=f"{pretty} — predicción en malla{mtxt}",
            elev=elev,
            azim=azim,
            zoom=zoom,
            mesh_alpha=mesh_alpha,
            X_overlay=X,
            labels_overlay=pred,
            overlay_points=overlay_points,
            overlay_point_size=overlay_point_size,
            overlay_alpha=overlay_alpha,
        )

        # Errores
        ax_err = fig.add_subplot(gs[r, 1], projection="3d")
        if err_vertex_colors is not None:
            plot_mesh_panel(
                ax_err,
                raw_V_aligned,
                raw_F,
                err_vertex_colors,
                title=f"{pretty} — errores en malla\nrojo = pred ≠ GT",
                elev=elev,
                azim=azim,
                zoom=zoom,
                mesh_alpha=mesh_alpha,
                X_overlay=X,
                labels_overlay=None,
                overlay_points=overlay_points,
                overlay_point_size=overlay_point_size,
                overlay_alpha=overlay_alpha,
            )
        else:
            plot_mesh_panel(
                ax_err,
                raw_V_aligned,
                raw_F,
                raw_to_rgba(raw_V_aligned.shape[0], alpha_mesh=mesh_alpha),
                title=f"{pretty} — GT no disponible",
                elev=elev,
                azim=azim,
                zoom=zoom,
                mesh_alpha=mesh_alpha,
            )

        # Diente 21 predicho
        ax_d21 = fig.add_subplot(gs[r, 2], projection="3d")
        plot_mesh_panel(
            ax_d21,
            raw_V_aligned,
            raw_F,
            pred_d21_vertex_colors,
            title=f"{pretty} — diente 21 predicho en malla",
            elev=elev,
            azim=azim,
            zoom=zoom,
            mesh_alpha=mesh_alpha,
            X_overlay=X,
            labels_overlay=pred,
            overlay_points=overlay_points,
            overlay_point_size=overlay_point_size,
            overlay_alpha=overlay_alpha,
        )

    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.04, dpi=int(dpi))
    plt.close(fig)

    summary = {
        "case_dir": str(case_dir),
        "out_png": str(out_png),
        "models": [m["model"] for m in loaded_models],
        "d21_class": int(d21_class),
        "align_mode": align_mode,
        "align_meta": align_meta,
        "raw_mesh_path": str(raw_mesh_path) if raw_mesh_path else "",
        "mesh_projection": "nearest_neighbor_points_to_vertices",
        "dpi": int(dpi),
        "figsize_scale": float(figsize_scale),
        "mesh_alpha": float(mesh_alpha),
        "overlay_points": bool(overlay_points),
        "metrics_summary": metrics_summary,
    }

    save_json(summary, out_png.with_suffix(".summary.json"))
    return summary


# ============================================================
# -------------------- PLOTLY HTML ---------------------------
# ============================================================

def rgba_to_plotly(colors: np.ndarray) -> List[str]:
    colors = np.asarray(colors)
    out = []
    for c in colors:
        r = int(np.clip(c[0] * 255, 0, 255))
        g = int(np.clip(c[1] * 255, 0, 255))
        b = int(np.clip(c[2] * 255, 0, 255))
        a = float(np.clip(c[3], 0, 1))
        out.append(f"rgba({r},{g},{b},{a:.3f})")
    return out


def add_plotly_mesh(fig, row, col, V, F, vertex_colors, name):
    if V is None:
        return
    V = finite_points(V)
    vertex_colors = np.asarray(vertex_colors, dtype=np.float32)

    if F is not None and len(F) > 0:
        F = np.asarray(F, dtype=np.int64)
        fig.add_trace(
            go.Mesh3d(
                x=V[:, 0],
                y=V[:, 1],
                z=V[:, 2],
                i=F[:, 0],
                j=F[:, 1],
                k=F[:, 2],
                vertexcolor=rgba_to_plotly(vertex_colors),
                opacity=1.0,
                name=name,
                showscale=False,
            ),
            row=row,
            col=col,
        )
    else:
        fig.add_trace(
            go.Scatter3d(
                x=V[:, 0],
                y=V[:, 1],
                z=V[:, 2],
                mode="markers",
                marker=dict(size=1.2, color=rgba_to_plotly(vertex_colors)),
                name=name,
                showlegend=False,
            ),
            row=row,
            col=col,
        )


def make_plotly_html(
    case_dir: Path,
    out_html: Path,
    models: List[str],
    d21_class: int,
    align_mode: str,
    require_pred_labels: bool,
    title: str,
):
    if not HAS_PLOTLY:
        print("[WARN] Plotly no está instalado. No se exportará HTML.")
        return None

    case_dir = Path(case_dir)
    out_html = Path(out_html)
    ensure_dir(out_html.parent)

    common = load_common_bundle(case_dir)
    meta = read_json(case_dir / "meta.json")

    loaded_models = []
    for m in models:
        md = load_model_case(case_dir, m, require_pred_labels=require_pred_labels)
        if md.get("available"):
            loaded_models.append(md)

    if not loaded_models:
        raise RuntimeError("No hay modelos disponibles para Plotly.")

    base_xyz = loaded_models[0].get("xyz")
    if base_xyz is None:
        if "xyz_8192" in common:
            base_xyz = finite_points(common["xyz_8192"])
        else:
            raise RuntimeError("No hay xyz para Plotly.")

    gt = loaded_models[0].get("gt")
    if gt is None and "y_8192" in common:
        gt = np.asarray(common["y_8192"]).reshape(-1).astype(np.int64)

    raw_mesh_path = find_raw_mesh_from_meta(case_dir)
    raw_V, raw_F = load_raw_mesh_vertices_faces(raw_mesh_path)
    raw_V_aligned, raw_F, align_meta = align_raw_mesh(raw_V, raw_F, base_xyz, align_mode=align_mode)

    if raw_V_aligned is None:
        raise RuntimeError("No se pudo cargar/alinear la malla raw para Plotly.")

    n_rows = 1 + len(loaded_models)
    n_cols = 3

    subplot_titles = ["Raw mesh", "GT en malla", f"GT d21={d21_class}"]
    for md in loaded_models:
        pretty = md.get("pretty", pretty_model_name(md["model"]))
        subplot_titles.extend([
            f"{pretty} pred",
            f"{pretty} errores",
            f"{pretty} d21",
        ])

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        specs=[[{"type": "scene"} for _ in range(n_cols)] for _ in range(n_rows)],
        subplot_titles=subplot_titles,
        horizontal_spacing=0.01,
        vertical_spacing=0.03,
    )

    add_plotly_mesh(
        fig,
        1,
        1,
        raw_V_aligned,
        raw_F,
        raw_to_rgba(raw_V_aligned.shape[0], alpha_mesh=1.0),
        "raw",
    )

    if gt is not None:
        gt_vertex_labels = labels_points_to_vertices(raw_V_aligned, base_xyz, gt)
        add_plotly_mesh(
            fig,
            1,
            2,
            raw_V_aligned,
            raw_F,
            labels_to_rgba(gt_vertex_labels, alpha=1.0),
            "gt",
        )
        add_plotly_mesh(
            fig,
            1,
            3,
            raw_V_aligned,
            raw_F,
            d21_to_rgba(gt_vertex_labels, d21_class=d21_class, alpha_mesh=1.0),
            "gt_d21",
        )

    for r, md in enumerate(loaded_models, start=2):
        X = md.get("xyz", base_xyz)
        pred = md["pred"]
        gt_model = md.get("gt", gt)

        X, pred, gt_model = fix_length_match(X, pred, gt_model, strict=False)

        pred_vertex_labels = labels_points_to_vertices(raw_V_aligned, X, pred)

        add_plotly_mesh(
            fig,
            r,
            1,
            raw_V_aligned,
            raw_F,
            labels_to_rgba(pred_vertex_labels, alpha=1.0),
            f"{md['model']}_pred",
        )

        if gt_model is not None:
            vertex_errors = point_errors_to_vertices(raw_V_aligned, X, pred, gt_model)
            add_plotly_mesh(
                fig,
                r,
                2,
                raw_V_aligned,
                raw_F,
                errors_to_rgba(vertex_errors, alpha_mesh=1.0),
                f"{md['model']}_errors",
            )

        add_plotly_mesh(
            fig,
            r,
            3,
            raw_V_aligned,
            raw_F,
            d21_to_rgba(pred_vertex_labels, d21_class=d21_class, alpha_mesh=1.0),
            f"{md['model']}_d21",
        )

    fig.update_layout(
        title=title or f"Paper figure mesh-projected interactive — {meta.get('sample_name', case_dir.name)}",
        height=max(850, 430 * n_rows),
        width=1650,
        margin=dict(l=5, r=5, t=80, b=5),
    )

    for k in fig.layout:
        if str(k).startswith("scene"):
            fig.layout[k].update(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode="data",
            )

    fig.write_html(str(out_html), include_plotlyjs="cdn")

    summary = {
        "out_html": str(out_html),
        "align_meta": align_meta,
        "models": [m["model"] for m in loaded_models],
    }
    save_json(summary, out_html.with_suffix(".summary.json"))
    return summary


# ============================================================
# -------------------- MAIN ----------------------------------
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--case_dir", required=True)
    parser.add_argument("--out_png", required=True)
    parser.add_argument("--out_html", default="")

    parser.add_argument("--models", default="pointnet")
    parser.add_argument("--d21_class", type=int, default=8)
    parser.add_argument("--align_mode", default="robust_bbox",
                        choices=["none", "unit_sphere", "bbox", "robust_bbox"])

    parser.add_argument("--zoom", type=float, default=1.55)
    parser.add_argument("--elev", type=float, default=25.0)
    parser.add_argument("--azim", type=float, default=-55.0)

    parser.add_argument("--dpi", type=int, default=450)
    parser.add_argument("--figsize_scale", type=float, default=1.15)
    parser.add_argument("--mesh_alpha", type=float, default=0.92)

    parser.add_argument("--overlay_points", action="store_true",
                        help="Superpone puntos 8192 sobre la malla.")
    parser.add_argument("--overlay_point_size", type=float, default=0.75)
    parser.add_argument("--overlay_alpha", type=float, default=0.20)

    parser.add_argument("--title", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--require_pred_labels", action="store_true")
    parser.add_argument("--no_plotly", action="store_true")

    args = parser.parse_args()

    case_dir = Path(args.case_dir).resolve()
    out_png = Path(args.out_png).resolve()

    if out_png.exists() and not args.force:
        raise FileExistsError(f"Ya existe {out_png}. Use --force para sobrescribir.")

    models = parse_models(args.models)

    print("[INFO] case_dir:", case_dir)
    print("[INFO] out_png:", out_png)
    print("[INFO] models:", models)
    print("[INFO] align_mode:", args.align_mode)
    print("[INFO] dpi:", args.dpi)

    summary = make_png_figure(
        case_dir=case_dir,
        out_png=out_png,
        models=models,
        d21_class=args.d21_class,
        align_mode=args.align_mode,
        zoom=args.zoom,
        elev=args.elev,
        azim=args.azim,
        dpi=args.dpi,
        figsize_scale=args.figsize_scale,
        mesh_alpha=args.mesh_alpha,
        overlay_points=args.overlay_points,
        overlay_point_size=args.overlay_point_size,
        overlay_alpha=args.overlay_alpha,
        require_pred_labels=args.require_pred_labels,
        title=args.title,
    )

    print(f"[OK] PNG HD guardado: {out_png}")
    print(f"[OK] Summary: {out_png.with_suffix('.summary.json')}")

    if not args.no_plotly:
        if args.out_html:
            out_html = Path(args.out_html).resolve()
        else:
            out_html = out_png.with_suffix(".html")

        make_plotly_html(
            case_dir=case_dir,
            out_html=out_html,
            models=models,
            d21_class=args.d21_class,
            align_mode=args.align_mode,
            require_pred_labels=args.require_pred_labels,
            title=args.title,
        )
        print(f"[OK] Plotly HTML guardado: {out_html}")

    print("\nListo ✅")


if __name__ == "__main__":
    main()