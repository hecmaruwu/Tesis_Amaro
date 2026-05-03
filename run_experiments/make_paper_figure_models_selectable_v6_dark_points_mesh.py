#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_paper_figure_models_selectable_v6_dark_points_mesh.py

Versión PRO:
- Fondo negro / oscuro estilo resumen.
- Malla raw translúcida como referencia, NO coloreada completa.
- Puntos 8192 encima con colores fuertes.
- Figura estática HD con menos espacio en blanco.
- Plotly HTML con malla transparente + puntos.
- Vista extra: d21 + dientes vecinos + errores d21.
- Diseñada para probar primero PointNet y luego extender a todos los modelos.

Entrada esperada:
case_dir/
  meta.json
  common/bundle_case_data.npz
  pointnet/
    pred_labels.npy
    gt_labels.npy
    xyz_labels.npy
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
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False


# ============================================================
# MODELOS
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
# COLORES OSCUROS / PAPER
# ============================================================

# Color de clase interna: 0 bg + 1..14 dientes
CLASS_COLORS = {
    0:  "#4A4A4A",  # fondo
    1:  "#3CB44B",  # d11
    2:  "#42D4F4",  # d12
    3:  "#4363D8",  # d13
    4:  "#911EB4",  # d14
    5:  "#F032E6",  # d15
    6:  "#E6194B",  # d16
    7:  "#F58231",  # d17
    8:  "#FF69B4",  # d21
    9:  "#FFD700",  # d22
    10: "#BFEF45",  # d23
    11: "#469990",  # d24
    12: "#00CED1",  # d25
    13: "#DCBEFF",  # d26
    14: "#A9A9A9",  # d27
}

COLOR_BG = "#303030"
COLOR_RAW_MESH = "#BDBDBD"
COLOR_ERROR = "#FF2020"
COLOR_OK_D21 = "#00D65F"
COLOR_D21 = "#FF4FB3"
COLOR_LEFT_NEIGH = "#2196F3"
COLOR_RIGHT_NEIGH = "#FF69B4"
COLOR_OTHER_TOOTH = "#777777"

D21_DEFAULT = 8
DEFAULT_NEIGHBORS = "d11:1,d22:9"


# ============================================================
# UTILS
# ============================================================

def ensure_dir(path: Path) -> Path:
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


def parse_neighbor_teeth(s: str) -> List[Tuple[str, int]]:
    """
    Ejemplo:
      "d11:1,d22:9"
    """
    out = []
    if not s:
        return []
    for part in s.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            continue
        name, val = part.split(":", 1)
        try:
            out.append((name.strip(), int(val)))
        except Exception:
            pass
    return out


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
            arr = np.asarray(arr.item())
        except Exception:
            pass

    if isinstance(arr, np.ndarray) and arr.dtype == object and arr.size == 1:
        try:
            arr = np.asarray(arr.reshape(-1)[0])
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

    preferred_keys = preferred_keys or ["arr_0", "pred", "gt", "xyz", "points", "X", "Y", "labels"]

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
    arr = load_any_array(path, ["pred", "prediction", "pred_labels", "labels", "y", "Y", "arr_0"])
    arr = np.asarray(arr).squeeze()
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr.astype(np.int64)


def load_gt_labels(path: Path) -> np.ndarray:
    arr = load_any_array(path, ["gt", "labels", "gt_labels", "y", "Y", "arr_0"])
    arr = np.asarray(arr).squeeze()
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr.astype(np.int64)


def load_xyz(path: Path) -> np.ndarray:
    arr = load_any_array(path, ["xyz", "points", "X", "arr_0"])
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
# CARGA CASE
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
        "model_dir": str(model_dir),
        "pred": pred,
        "gt": gt,
        "xyz": xyz,
        "pred_path": str(pred_path),
        "gt_path": str(gt_path) if gt_path.exists() else "",
        "xyz_path": str(xyz_path) if xyz_path.exists() else "",
    }


# ============================================================
# RAW MESH
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
        print("[WARN] trimesh no está disponible.")
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

    V = finite_points(np.asarray(loaded.vertices, dtype=np.float32))
    F = np.asarray(loaded.faces, dtype=np.int64) if loaded.faces is not None else None
    if F is not None and F.size == 0:
        F = None

    return V, F


# ============================================================
# ALINEACIÓN
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
# COLORES DE PUNTOS
# ============================================================

def hex_to_rgba01(hex_color: str, alpha: float = 1.0):
    hex_color = hex_color.strip().lstrip("#")
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return (r, g, b, alpha)


def labels_to_colors(labels: np.ndarray, alpha_bg=0.28, alpha_fg=1.0) -> List:
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    colors = []
    for y in labels:
        if int(y) == 0:
            colors.append(hex_to_rgba01(CLASS_COLORS.get(0, COLOR_BG), alpha_bg))
        else:
            colors.append(hex_to_rgba01(CLASS_COLORS.get(int(y), "#FFFFFF"), alpha_fg))
    return colors


def d21_focus_colors(labels: np.ndarray, d21_class: int, alpha_bg=0.10) -> List:
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    colors = []
    for y in labels:
        if int(y) == int(d21_class):
            colors.append(hex_to_rgba01(COLOR_D21, 1.0))
        elif int(y) == 0:
            colors.append(hex_to_rgba01("#555555", alpha_bg))
        else:
            colors.append(hex_to_rgba01("#7A7A7A", 0.35))
    return colors


def neighbor_focus_colors(labels: np.ndarray, d21_class: int, neighbors: List[Tuple[str, int]], alpha_bg=0.06) -> List:
    """
    d21 = verde si correcto se maneja aparte en error view.
    Vecino 1 = azul.
    Vecino 2 = rosa.
    Otros = gris tenue.
    """
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    neigh_ids = [v for _, v in neighbors]
    colors = []
    for y in labels:
        yy = int(y)
        if yy == int(d21_class):
            colors.append(hex_to_rgba01(COLOR_D21, 1.0))
        elif len(neigh_ids) >= 1 and yy == neigh_ids[0]:
            colors.append(hex_to_rgba01(COLOR_LEFT_NEIGH, 1.0))
        elif len(neigh_ids) >= 2 and yy == neigh_ids[1]:
            colors.append(hex_to_rgba01(COLOR_RIGHT_NEIGH, 1.0))
        elif yy == 0:
            colors.append(hex_to_rgba01("#555555", alpha_bg))
        else:
            colors.append(hex_to_rgba01("#777777", 0.10))
    return colors


def d21_tp_error_colors(pred: np.ndarray, gt: np.ndarray, d21_class: int, neighbors: List[Tuple[str, int]]) -> List:
    """
    Verde = TP d21.
    Rojo = FP/FN d21.
    Vecinos = azul/rosa.
    Fondo/otros = gris tenue.
    """
    pred = np.asarray(pred).reshape(-1).astype(np.int64)
    gt = np.asarray(gt).reshape(-1).astype(np.int64)
    n = min(len(pred), len(gt))
    pred = pred[:n]
    gt = gt[:n]

    neigh_ids = [v for _, v in neighbors]
    colors = []

    for p, g in zip(pred, gt):
        p = int(p)
        g = int(g)

        is_tp = (p == d21_class and g == d21_class)
        is_err = ((p == d21_class) != (g == d21_class))

        if is_tp:
            colors.append(hex_to_rgba01(COLOR_OK_D21, 1.0))
        elif is_err:
            colors.append(hex_to_rgba01(COLOR_ERROR, 1.0))
        elif len(neigh_ids) >= 1 and g == neigh_ids[0]:
            colors.append(hex_to_rgba01(COLOR_LEFT_NEIGH, 0.95))
        elif len(neigh_ids) >= 2 and g == neigh_ids[1]:
            colors.append(hex_to_rgba01(COLOR_RIGHT_NEIGH, 0.95))
        elif g == 0:
            colors.append(hex_to_rgba01("#555555", 0.08))
        else:
            colors.append(hex_to_rgba01("#777777", 0.12))

    return colors


def error_colors_all(pred: np.ndarray, gt: np.ndarray) -> List:
    pred = np.asarray(pred).reshape(-1).astype(np.int64)
    gt = np.asarray(gt).reshape(-1).astype(np.int64)
    n = min(len(pred), len(gt))
    pred = pred[:n]
    gt = gt[:n]
    colors = []
    for p, g in zip(pred, gt):
        if int(p) == int(g):
            colors.append(hex_to_rgba01("#777777", 0.10))
        else:
            colors.append(hex_to_rgba01(COLOR_ERROR, 1.0))
    return colors


# ============================================================
# MÉTRICAS
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
# RENDER ESTÁTICO OSCURO HD
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


def apply_dark_axis(ax, elev: float, azim: float, zoom: float):
    ax.view_init(elev=elev, azim=azim)
    try:
        ax.dist = 7.2 / max(float(zoom), 1e-6)
    except Exception:
        pass

    ax.set_facecolor("#000000")
    ax.set_axis_off()
    set_axes_equal(ax)


def add_mesh_static(
    ax,
    V: Optional[np.ndarray],
    F: Optional[np.ndarray],
    alpha: float = 0.22,
    edge_alpha: float = 0.025,
):
    if V is None:
        return

    V = finite_points(V)

    if F is not None and len(F) > 0:
        F = np.asarray(F, dtype=np.int64)
        polys = V[F]
        coll = Poly3DCollection(
            polys,
            facecolors=hex_to_rgba01(COLOR_RAW_MESH, alpha),
            edgecolors=(1.0, 1.0, 1.0, edge_alpha),
            linewidths=0.015,
            antialiased=True,
        )
        ax.add_collection3d(coll)
        ax.auto_scale_xyz(V[:, 0], V[:, 1], V[:, 2])
    else:
        ax.scatter(
            V[:, 0], V[:, 1], V[:, 2],
            c=[hex_to_rgba01(COLOR_RAW_MESH, alpha)],
            s=0.12,
            linewidths=0,
            depthshade=False,
        )


def add_points_static(
    ax,
    X: np.ndarray,
    colors: List,
    point_size: float = 5.0,
    depthshade: bool = False,
):
    X = finite_points(X)
    ax.scatter(
        X[:, 0], X[:, 1], X[:, 2],
        c=colors,
        s=float(point_size),
        linewidths=0,
        depthshade=depthshade,
    )


def plot_panel_static(
    ax,
    V: Optional[np.ndarray],
    F: Optional[np.ndarray],
    X: Optional[np.ndarray],
    colors: Optional[List],
    title: str,
    elev: float,
    azim: float,
    zoom: float,
    mesh_alpha: float,
    point_size: float,
    show_mesh: bool = True,
):
    if show_mesh:
        add_mesh_static(ax, V, F, alpha=mesh_alpha)

    if X is not None and colors is not None:
        add_points_static(ax, X, colors, point_size=point_size)

    ax.set_title(title, fontsize=13, color="white", pad=2, fontweight="bold")
    apply_dark_axis(ax, elev=elev, azim=azim, zoom=zoom)


def make_static_dark_figure(
    case_dir: Path,
    out_png: Path,
    models: List[str],
    d21_class: int,
    neighbors: List[Tuple[str, int]],
    align_mode: str,
    zoom: float,
    elev: float,
    azim: float,
    dpi: int,
    point_size: float,
    mesh_alpha: float,
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
        base_xyz, _, gt = fix_length_match(
            base_xyz,
            np.zeros(base_xyz.shape[0], dtype=np.int64),
            gt,
            strict=False,
        )

    raw_mesh_path = find_raw_mesh_from_meta(case_dir)
    raw_V, raw_F = load_raw_mesh_vertices_faces(raw_mesh_path)
    raw_V_aligned, raw_F, align_meta = align_raw_mesh(raw_V, raw_F, base_xyz, align_mode=align_mode)

    if raw_V_aligned is None:
        raise RuntimeError("No se pudo cargar/alinear la malla raw.")

    # Solo usaremos primer modelo para la vista detallada si se está probando PointNet.
    md0 = loaded_models[0]
    model_name = md0["model"]
    pretty = md0.get("pretty", pretty_model_name(model_name))
    X = md0.get("xyz", base_xyz)
    pred = md0["pred"]
    gt_model = md0.get("gt", gt)
    X, pred, gt_model = fix_length_match(X, pred, gt_model, strict=False)

    metrics = compute_simple_metrics(pred, gt_model, d21_class=d21_class)

    # Colores
    gt_colors = labels_to_colors(gt_model if gt_model is not None else gt, alpha_bg=0.12, alpha_fg=1.0)
    pred_colors = labels_to_colors(pred, alpha_bg=0.12, alpha_fg=1.0)

    err_all_colors = error_colors_all(pred, gt_model) if gt_model is not None else labels_to_colors(pred)
    d21_gt_colors = d21_focus_colors(gt_model, d21_class=d21_class, alpha_bg=0.06) if gt_model is not None else d21_focus_colors(pred, d21_class)
    d21_pred_colors = d21_focus_colors(pred, d21_class=d21_class, alpha_bg=0.06)
    neighbor_gt_colors = neighbor_focus_colors(gt_model, d21_class, neighbors, alpha_bg=0.04) if gt_model is not None else neighbor_focus_colors(pred, d21_class, neighbors)
    neighbor_pred_colors = neighbor_focus_colors(pred, d21_class, neighbors, alpha_bg=0.04)
    d21_error_colors = d21_tp_error_colors(pred, gt_model, d21_class, neighbors) if gt_model is not None else d21_pred_colors

    fig = plt.figure(figsize=(16.8, 20.5), dpi=int(dpi), facecolor="#000000")
    gs = GridSpec(
        4, 3,
        figure=fig,
        width_ratios=[1.0, 1.0, 0.72],
        height_ratios=[0.10, 1.0, 0.10, 1.0],
        wspace=0.03,
        hspace=0.05,
    )

    sample_name = meta.get("sample_name", case_dir.name)
    title_main = title or f"VISTA PRO: malla raw translúcida + nubes coloreadas — {sample_name}"

    fig.text(0.5, 0.985, title_main, color="white", ha="center", va="top",
             fontsize=22, fontweight="bold")
    fig.text(0.5, 0.955, f"modelo={pretty} | d21={d21_class} | vecinos={neighbors}",
             color="#CFCFCF", ha="center", va="top", fontsize=12)

    # Header 1
    ax_h1 = fig.add_subplot(gs[0, :])
    ax_h1.set_facecolor("#000000")
    ax_h1.axis("off")
    ax_h1.text(0.5, 0.5, "TODAS LAS CLASES", color="white", fontsize=20,
               fontweight="bold", ha="center", va="center")
    ax_h1.axhline(0.05, color="#777777", lw=1.3)
    ax_h1.axhline(0.95, color="#777777", lw=1.3)

    # Paneles todas las clases
    ax_gt = fig.add_subplot(gs[1, 0], projection="3d")
    plot_panel_static(
        ax_gt, raw_V_aligned, raw_F, X, gt_colors,
        "GT\n(todas las clases)",
        elev=elev, azim=azim, zoom=zoom,
        mesh_alpha=mesh_alpha, point_size=point_size,
    )

    ax_pred = fig.add_subplot(gs[1, 1], projection="3d")
    metric_txt = ""
    if metrics:
        metric_txt = f"\nacc_nb={metrics['acc_no_bg']:.3f} | d21_f1={metrics['d21_f1']:.3f}"
    plot_panel_static(
        ax_pred, raw_V_aligned, raw_F, X, pred_colors,
        f"{pretty}\n(predicción multiclase){metric_txt}",
        elev=elev, azim=azim, zoom=zoom,
        mesh_alpha=mesh_alpha, point_size=point_size,
    )

    ax_err = fig.add_subplot(gs[1, 2], projection="3d")
    plot_panel_static(
        ax_err, raw_V_aligned, raw_F, X, err_all_colors,
        "Errores\nrojo = pred ≠ GT",
        elev=elev, azim=azim, zoom=zoom,
        mesh_alpha=mesh_alpha, point_size=point_size,
    )

    # Header 2
    ax_h2 = fig.add_subplot(gs[2, :])
    ax_h2.set_facecolor("#000000")
    ax_h2.axis("off")
    ax_h2.text(0.5, 0.5, "FOCO DIENTE 21 + VECINOS", color="white", fontsize=20,
               fontweight="bold", ha="center", va="center")
    ax_h2.axhline(0.05, color="#777777", lw=1.3)
    ax_h2.axhline(0.95, color="#777777", lw=1.3)

    # Paneles d21 + vecinos
    ax_d21_gt = fig.add_subplot(gs[3, 0], projection="3d")
    plot_panel_static(
        ax_d21_gt, raw_V_aligned, raw_F, X, neighbor_gt_colors,
        "GT\nazul/rosa = vecinos | magenta = d21",
        elev=elev, azim=azim, zoom=zoom * 1.12,
        mesh_alpha=mesh_alpha, point_size=point_size,
    )

    ax_d21_pred = fig.add_subplot(gs[3, 1], projection="3d")
    plot_panel_static(
        ax_d21_pred, raw_V_aligned, raw_F, X, neighbor_pred_colors,
        f"{pretty}\nazul/rosa = vecinos | magenta = d21",
        elev=elev, azim=azim, zoom=zoom * 1.12,
        mesh_alpha=mesh_alpha, point_size=point_size,
    )

    ax_d21_err = fig.add_subplot(gs[3, 2], projection="3d")
    plot_panel_static(
        ax_d21_err, raw_V_aligned, raw_F, X, d21_error_colors,
        "Error d21\nverde=TP | rojo=FP/FN",
        elev=elev, azim=azim, zoom=zoom * 1.35,
        mesh_alpha=mesh_alpha * 0.75,
        point_size=point_size * 1.15,
    )

    # Leyenda manual
    legend_x = 0.74
    legend_y = 0.91
    line_h = 0.018

    fig.text(legend_x, legend_y, "Leyenda", color="white", fontsize=15, fontweight="bold", ha="left")
    legend_items = [
        ("fondo", CLASS_COLORS[0]),
        ("d21", COLOR_D21),
        ("vecino izq.", COLOR_LEFT_NEIGH),
        ("vecino der.", COLOR_RIGHT_NEIGH),
        ("error", COLOR_ERROR),
        ("TP d21", COLOR_OK_D21),
        ("malla raw", COLOR_RAW_MESH),
    ]
    for i, (lab, col) in enumerate(legend_items):
        y = legend_y - (i + 1) * line_h
        fig.text(legend_x, y, "■", color=col, fontsize=15, ha="left", va="center")
        fig.text(legend_x + 0.025, y, lab, color="#EAEAEA", fontsize=11, ha="left", va="center")

    fig.text(
        0.5, 0.018,
        "Malla raw translúcida + puntos 8192 coloreados. Rojo = discrepancia entre predicción y ground truth.",
        color="#DADADA",
        ha="center",
        fontsize=12,
        fontweight="bold",
        bbox=dict(facecolor="#111111", edgecolor="#AAAAAA", boxstyle="round,pad=0.45"),
    )

    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.02, dpi=int(dpi), facecolor="#000000")
    plt.close(fig)

    summary = {
        "case_dir": str(case_dir),
        "out_png": str(out_png),
        "models": [m["model"] for m in loaded_models],
        "active_model": model_name,
        "d21_class": int(d21_class),
        "neighbors": neighbors,
        "align_mode": align_mode,
        "align_meta": align_meta,
        "raw_mesh_path": str(raw_mesh_path) if raw_mesh_path else "",
        "metrics": metrics,
    }
    save_json(summary, out_png.with_suffix(".summary.json"))
    return summary


# ============================================================
# PLOTLY
# ============================================================

def rgba_tuple_to_plotly(c):
    r = int(np.clip(c[0] * 255, 0, 255))
    g = int(np.clip(c[1] * 255, 0, 255))
    b = int(np.clip(c[2] * 255, 0, 255))
    a = float(np.clip(c[3], 0, 1))
    return f"rgba({r},{g},{b},{a:.3f})"


def color_list_to_plotly(colors: List) -> List[str]:
    out = []
    for c in colors:
        if isinstance(c, str):
            out.append(c)
        else:
            out.append(rgba_tuple_to_plotly(c))
    return out


def add_plotly_mesh(fig, row, col, V, F, name, opacity=0.16):
    if V is None:
        return
    V = finite_points(V)
    if F is not None and len(F) > 0:
        F = np.asarray(F, dtype=np.int64)
        fig.add_trace(
            go.Mesh3d(
                x=V[:, 0], y=V[:, 1], z=V[:, 2],
                i=F[:, 0], j=F[:, 1], k=F[:, 2],
                color="lightgray",
                opacity=float(opacity),
                name=name,
                showscale=False,
                hoverinfo="skip",
            ),
            row=row, col=col,
        )
    else:
        fig.add_trace(
            go.Scatter3d(
                x=V[:, 0], y=V[:, 1], z=V[:, 2],
                mode="markers",
                marker=dict(size=1, color="rgba(200,200,200,0.12)"),
                name=name,
                showlegend=False,
            ),
            row=row, col=col,
        )


def add_plotly_points(fig, row, col, X, colors, name, size=2.2):
    X = finite_points(X)
    fig.add_trace(
        go.Scatter3d(
            x=X[:, 0], y=X[:, 1], z=X[:, 2],
            mode="markers",
            marker=dict(
                size=float(size),
                color=color_list_to_plotly(colors),
                opacity=1.0,
            ),
            name=name,
            showlegend=False,
        ),
        row=row, col=col,
    )


def make_plotly_html(
    case_dir: Path,
    out_html: Path,
    models: List[str],
    d21_class: int,
    neighbors: List[Tuple[str, int]],
    align_mode: str,
    mesh_opacity: float,
    point_size: float,
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

    md0 = loaded_models[0]
    pretty = md0.get("pretty", pretty_model_name(md0["model"]))
    X = md0.get("xyz", base_xyz)
    pred = md0["pred"]
    gt_model = md0.get("gt", gt)
    X, pred, gt_model = fix_length_match(X, pred, gt_model, strict=False)

    raw_mesh_path = find_raw_mesh_from_meta(case_dir)
    raw_V, raw_F = load_raw_mesh_vertices_faces(raw_mesh_path)
    raw_V_aligned, raw_F, align_meta = align_raw_mesh(raw_V, raw_F, X, align_mode=align_mode)

    if raw_V_aligned is None:
        raise RuntimeError("No se pudo cargar/alinear la malla raw para Plotly.")

    fig = make_subplots(
        rows=2,
        cols=3,
        specs=[[{"type": "scene"} for _ in range(3)] for _ in range(2)],
        subplot_titles=[
            "GT multiclase", f"{pretty} pred", "Errores multiclase",
            "GT d21 + vecinos", f"{pretty} d21 + vecinos", "Error d21: verde TP / rojo FP-FN",
        ],
        horizontal_spacing=0.01,
        vertical_spacing=0.03,
    )

    panels = [
        (1, 1, gt_colors if False else labels_to_colors(gt_model, alpha_bg=0.08, alpha_fg=1.0), "gt_all"),
        (1, 2, labels_to_colors(pred, alpha_bg=0.08, alpha_fg=1.0), "pred_all"),
        (1, 3, error_colors_all(pred, gt_model), "err_all"),
        (2, 1, neighbor_focus_colors(gt_model, d21_class, neighbors, alpha_bg=0.04), "gt_neighbors"),
        (2, 2, neighbor_focus_colors(pred, d21_class, neighbors, alpha_bg=0.04), "pred_neighbors"),
        (2, 3, d21_tp_error_colors(pred, gt_model, d21_class, neighbors), "d21_errors"),
    ]

    for row, col, colors, name in panels:
        add_plotly_mesh(fig, row, col, raw_V_aligned, raw_F, "raw_mesh", opacity=mesh_opacity)
        add_plotly_points(fig, row, col, X, colors, name, size=point_size)

    fig.update_layout(
        title=title or f"Vista interactiva PRO — {meta.get('sample_name', case_dir.name)} — {pretty}",
        height=1050,
        width=1700,
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
        margin=dict(l=5, r=5, t=80, b=5),
    )

    for k in fig.layout:
        if str(k).startswith("scene"):
            fig.layout[k].update(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode="data",
                bgcolor="black",
            )

    fig.write_html(str(out_html), include_plotlyjs="cdn")

    summary = {
        "out_html": str(out_html),
        "align_meta": align_meta,
        "models": [m["model"] for m in loaded_models],
        "active_model": md0["model"],
        "mesh_opacity": float(mesh_opacity),
        "point_size": float(point_size),
    }
    save_json(summary, out_html.with_suffix(".summary.json"))
    return summary


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--case_dir", required=True)
    parser.add_argument("--out_png", required=True)
    parser.add_argument("--out_html", default="")

    parser.add_argument("--models", default="pointnet")
    parser.add_argument("--d21_class", type=int, default=D21_DEFAULT)
    parser.add_argument("--neighbor_teeth", default=DEFAULT_NEIGHBORS)

    parser.add_argument("--align_mode", default="robust_bbox",
                        choices=["none", "unit_sphere", "bbox", "robust_bbox"])

    parser.add_argument("--zoom", type=float, default=2.35)
    parser.add_argument("--elev", type=float, default=28.0)
    parser.add_argument("--azim", type=float, default=-58.0)

    parser.add_argument("--dpi", type=int, default=420)
    parser.add_argument("--point_size", type=float, default=4.5)
    parser.add_argument("--mesh_alpha", type=float, default=0.18)

    parser.add_argument("--plotly_mesh_opacity", type=float, default=0.16)
    parser.add_argument("--plotly_point_size", type=float, default=2.6)

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
    neighbors = parse_neighbor_teeth(args.neighbor_teeth)

    print("[INFO] case_dir:", case_dir)
    print("[INFO] out_png:", out_png)
    print("[INFO] models:", models)
    print("[INFO] d21_class:", args.d21_class)
    print("[INFO] neighbors:", neighbors)
    print("[INFO] align_mode:", args.align_mode)

    make_static_dark_figure(
        case_dir=case_dir,
        out_png=out_png,
        models=models,
        d21_class=args.d21_class,
        neighbors=neighbors,
        align_mode=args.align_mode,
        zoom=args.zoom,
        elev=args.elev,
        azim=args.azim,
        dpi=args.dpi,
        point_size=args.point_size,
        mesh_alpha=args.mesh_alpha,
        require_pred_labels=args.require_pred_labels,
        title=args.title,
    )

    print(f"[OK] PNG oscuro HD guardado: {out_png}")
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
            neighbors=neighbors,
            align_mode=args.align_mode,
            mesh_opacity=args.plotly_mesh_opacity,
            point_size=args.plotly_point_size,
            require_pred_labels=args.require_pred_labels,
            title=args.title,
        )

        print(f"[OK] Plotly HTML guardado: {out_html}")

    print("\nListo ✅")


if __name__ == "__main__":
    main()

    