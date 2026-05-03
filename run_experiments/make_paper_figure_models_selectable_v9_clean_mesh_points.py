#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_paper_figure_models_selectable_v9_clean_mesh_points.py

Versión corregida:
- Fondo blanco.
- Malla translúcida visible, pero no invasiva.
- Puntos 8192 sobre malla.
- Vista general 2x3:
    GT multiclase
    Pred multiclase
    Errores multiclase
    GT d21 + vecinos
    Pred d21 + vecinos
    Error d21
- Vista individual:
    exporta PNG separado por panel.
- Plotly corregido:
    usa la misma nube y la misma malla alineada.
    NO colorea la malla por clase.
    NO proyecta etiquetas a la malla.
- Colores d21 + vecinos iguales en GT y Pred:
    d21 = naranja
    vecino izquierdo = azul
    vecino derecho = rosado
- Error d21:
    TP = verde
    FP/FN = rojo
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

# Multiclase
CLASS_COLORS = {
    0:  "#BDBDBD",
    1:  "#2CA02C",
    2:  "#17BECF",
    3:  "#1F77B4",
    4:  "#9467BD",
    5:  "#E377C2",
    6:  "#D62728",
    7:  "#FF7F0E",
    8:  "#FF8C00",   # d21 también naranja
    9:  "#FF69B4",   # vecino derecho rosado
    10: "#BFEF45",
    11: "#2CA58D",
    12: "#00BCD4",
    13: "#C5B0D5",
    14: "#7F7F7F",
}

# Foco d21 + vecinos
COLOR_MESH = "#AFAFAF"
COLOR_BG = "#D0D0D0"
COLOR_OTHER = "#BDBDBD"
COLOR_D21 = "#FF8C00"        # naranja
COLOR_LEFT_NEIGH = "#0070C0" # azul
COLOR_RIGHT_NEIGH = "#FF69B4"# rosado
COLOR_ERROR = "#FF0000"      # rojo
COLOR_TP = "#00B050"         # verde


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


def fix_length_match(X, pred, gt=None, strict=False):
    X = finite_points(X)
    pred = np.asarray(pred).reshape(-1).astype(np.int64)
    gt_arr = None if gt is None else np.asarray(gt).reshape(-1).astype(np.int64)

    lengths = [X.shape[0], pred.shape[0]]
    if gt_arr is not None:
        lengths.append(gt_arr.shape[0])

    mn, mx = min(lengths), max(lengths)

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
# ALIGNMENT
# ============================================================

def robust_bounds(X: np.ndarray, q_low=0.01, q_high=0.99):
    X = finite_points(X)
    lo = np.quantile(X, q_low, axis=0)
    hi = np.quantile(X, q_high, axis=0)
    return lo.astype(np.float32), hi.astype(np.float32)


def bbox_center_scale(X: np.ndarray, robust=True):
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


def normalize_unit_sphere(X: np.ndarray):
    X = finite_points(X)
    c = X.mean(axis=0, keepdims=True)
    Xc = X - c
    r = np.linalg.norm(Xc, axis=1).max()

    if not np.isfinite(r) or r <= 0:
        r = 1.0

    return (Xc / r).astype(np.float32), c.reshape(3).astype(np.float32), float(r)


def align_source_to_target_bbox(X_source, X_target, robust=True):
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


def align_raw_mesh(raw_vertices, raw_faces, target_xyz, align_mode="robust_bbox"):
    if raw_vertices is None:
        return None, raw_faces, {"mode": "missing_raw"}

    raw_vertices = finite_points(raw_vertices)
    target_xyz = finite_points(target_xyz)

    if align_mode == "none":
        return raw_vertices, raw_faces, {"mode": "none"}

    if align_mode == "unit_sphere":
        V, c, s = normalize_unit_sphere(raw_vertices)
        return V, raw_faces, {
            "mode": "unit_sphere",
            "center": c.tolist(),
            "scale": float(s),
        }

    if align_mode == "bbox":
        V, meta = align_source_to_target_bbox(raw_vertices, target_xyz, robust=False)
        return V, raw_faces, meta

    if align_mode == "robust_bbox":
        V, meta = align_source_to_target_bbox(raw_vertices, target_xyz, robust=True)
        return V, raw_faces, meta

    raise ValueError(f"align_mode no reconocido: {align_mode}")


# ============================================================
# COLORS
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


def colors_to_plotly(colors: List):
    out = []
    for c in colors:
        if isinstance(c, str):
            out.append(c)
        else:
            out.append(rgba_to_plotly(c))
    return out


def labels_to_colors(labels, alpha_bg=0.22, alpha_fg=1.0):
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    out = []

    for y in labels:
        y = int(y)
        if y == 0:
            out.append(hex_to_rgba01(CLASS_COLORS[0], alpha_bg))
        else:
            out.append(hex_to_rgba01(CLASS_COLORS.get(y, "#444444"), alpha_fg))

    return out


def error_colors_all(pred, gt, alpha_ok=0.10):
    pred = np.asarray(pred).reshape(-1).astype(np.int64)
    gt = np.asarray(gt).reshape(-1).astype(np.int64)

    n = min(len(pred), len(gt))
    pred = pred[:n]
    gt = gt[:n]

    out = []
    for p, g in zip(pred, gt):
        if int(p) == int(g):
            out.append(hex_to_rgba01("#BDBDBD", alpha_ok))
        else:
            out.append(hex_to_rgba01(COLOR_ERROR, 1.0))

    return out


def neighbor_focus_colors(labels, d21_class: int, neighbors: List[Tuple[str, int]], alpha_bg=0.035):
    """
    MISMO COLOR para GT y Pred:
    d21 = naranja
    vecino izquierdo = azul
    vecino derecho = rosado
    """
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    neigh_ids = [v for _, v in neighbors]

    out = []
    for y in labels:
        yy = int(y)

        if yy == int(d21_class):
            out.append(hex_to_rgba01(COLOR_D21, 1.0))
        elif len(neigh_ids) >= 1 and yy == neigh_ids[0]:
            out.append(hex_to_rgba01(COLOR_LEFT_NEIGH, 1.0))
        elif len(neigh_ids) >= 2 and yy == neigh_ids[1]:
            out.append(hex_to_rgba01(COLOR_RIGHT_NEIGH, 1.0))
        elif yy == 0:
            out.append(hex_to_rgba01(COLOR_BG, alpha_bg))
        else:
            out.append(hex_to_rgba01(COLOR_OTHER, 0.07))

    return out


def d21_tp_error_colors(pred, gt, d21_class: int, neighbors: List[Tuple[str, int]]):
    pred = np.asarray(pred).reshape(-1).astype(np.int64)
    gt = np.asarray(gt).reshape(-1).astype(np.int64)

    n = min(len(pred), len(gt))
    pred = pred[:n]
    gt = gt[:n]

    neigh_ids = [v for _, v in neighbors]
    out = []

    for p, g in zip(pred, gt):
        p = int(p)
        g = int(g)

        is_tp = (p == d21_class and g == d21_class)
        is_err = ((p == d21_class) != (g == d21_class))

        if is_tp:
            out.append(hex_to_rgba01(COLOR_TP, 1.0))
        elif is_err:
            out.append(hex_to_rgba01(COLOR_ERROR, 1.0))
        elif len(neigh_ids) >= 1 and g == neigh_ids[0]:
            out.append(hex_to_rgba01(COLOR_LEFT_NEIGH, 0.92))
        elif len(neigh_ids) >= 2 and g == neigh_ids[1]:
            out.append(hex_to_rgba01(COLOR_RIGHT_NEIGH, 0.92))
        elif g == 0:
            out.append(hex_to_rgba01(COLOR_BG, 0.030))
        else:
            out.append(hex_to_rgba01(COLOR_OTHER, 0.050))

    return out


def compute_simple_metrics(pred, gt, d21_class=8):
    if gt is None:
        return {}

    pred = np.asarray(pred).reshape(-1)
    gt = np.asarray(gt).reshape(-1)

    n = min(len(pred), len(gt))
    pred = pred[:n]
    gt = gt[:n]

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
        "acc_no_bg": float(acc_no_bg),
        "d21_precision": float(prec),
        "d21_recall": float(rec),
        "d21_f1": float(f1),
        "d21_iou": float(iou),
        "n_points": int(n),
    }


# ============================================================
# RENDER HELPERS
# ============================================================

def set_axes_equal_from_points(ax, X, zoom=1.55, pad=0.08):
    X = finite_points(X)

    lo = np.quantile(X, 0.005, axis=0)
    hi = np.quantile(X, 0.995, axis=0)

    center = 0.5 * (lo + hi)
    span = hi - lo
    radius = 0.5 * float(np.max(span)) * (1.0 + pad)
    radius = radius / max(float(zoom), 1e-6)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def apply_view(ax, X, elev=25, azim=-55, zoom=1.55):
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_facecolor("white")
    try:
        ax.dist = 7.5
    except Exception:
        pass
    set_axes_equal_from_points(ax, X, zoom=zoom)


def add_mesh(ax, V, F, alpha=0.20):
    if V is None:
        return

    V = finite_points(V)

    if F is not None and len(F) > 0:
        F = np.asarray(F, dtype=np.int64)
        poly = Poly3DCollection(
            V[F],
            facecolors=hex_to_rgba01(COLOR_MESH, alpha),
            edgecolors=(0.55, 0.55, 0.55, 0.018),
            linewidths=0.012,
            antialiased=True,
        )
        ax.add_collection3d(poly)
        ax.auto_scale_xyz(V[:, 0], V[:, 1], V[:, 2])
    else:
        ax.scatter(
            V[:, 0], V[:, 1], V[:, 2],
            s=0.1,
            c=[hex_to_rgba01(COLOR_MESH, alpha)],
            linewidths=0,
            depthshade=False,
        )


def add_points(ax, X, colors, point_size=5.8):
    X = finite_points(X)
    ax.scatter(
        X[:, 0], X[:, 1], X[:, 2],
        c=colors,
        s=float(point_size),
        linewidths=0,
        depthshade=False,
    )


def plot_panel(ax, V, F, X, colors, title, elev, azim, zoom, mesh_alpha, point_size):
    add_mesh(ax, V, F, alpha=mesh_alpha)
    add_points(ax, X, colors, point_size=point_size)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=2)
    apply_view(ax, X, elev=elev, azim=azim, zoom=zoom)


# ============================================================
# DATA PREP
# ============================================================

def prepare_case_data(case_dir, model_key, d21_class, neighbors, align_mode):
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
    V, F, align_meta = align_raw_mesh(V_raw, F_raw, X, align_mode=align_mode)

    metrics = compute_simple_metrics(pred, gt, d21_class=d21_class)

    colors = {
        "gt_all": labels_to_colors(gt, alpha_bg=0.22, alpha_fg=1.0),
        "pred_all": labels_to_colors(pred, alpha_bg=0.22, alpha_fg=1.0),
        "err_all": error_colors_all(pred, gt, alpha_ok=0.10),
        "gt_focus": neighbor_focus_colors(gt, d21_class, neighbors, alpha_bg=0.035),
        "pred_focus": neighbor_focus_colors(pred, d21_class, neighbors, alpha_bg=0.035),
        "d21_error": d21_tp_error_colors(pred, gt, d21_class, neighbors),
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
        "metrics": metrics,
        "colors": colors,
    }


# ============================================================
# STATIC SUMMARY PNG
# ============================================================

def make_summary_png(
    data,
    out_png,
    d21_class,
    neighbors,
    elev,
    azim,
    zoom,
    zoom_focus,
    mesh_alpha,
    point_size,
    dpi,
    title="",
):
    out_png = Path(out_png)
    ensure_dir(out_png.parent)

    X = data["X"]
    V = data["V"]
    F = data["F"]
    colors = data["colors"]
    pretty = data["pretty"]
    metrics = data["metrics"]

    fig = plt.figure(figsize=(15.5, 9.4), dpi=int(dpi), facecolor="white")
    gs = GridSpec(2, 3, figure=fig, wspace=0.015, hspace=0.10)

    metric_txt = ""
    if metrics:
        metric_txt = f"\nacc_nb={metrics['acc_no_bg']:.3f} | d21_f1={metrics['d21_f1']:.3f}"

    panel_specs = [
        ("GT\n(todas las clases)", colors["gt_all"], zoom, point_size),
        (f"{pretty}\n(predicción multiclase){metric_txt}", colors["pred_all"], zoom, point_size),
        ("Errores multiclase\nrojo = pred ≠ GT", colors["err_all"], zoom, point_size * 1.04),
        ("GT: d21 + vecinos\nazul/naranja/rosado", colors["gt_focus"], zoom_focus, point_size * 1.12),
        (f"{pretty}: d21 + vecinos\nazul/naranja/rosado", colors["pred_focus"], zoom_focus, point_size * 1.12),
        ("Error d21\nverde=TP | rojo=FP/FN", colors["d21_error"], zoom_focus, point_size * 1.18),
    ]

    for i, (ttl, cols, z, ps) in enumerate(panel_specs):
        ax = fig.add_subplot(gs[i // 3, i % 3], projection="3d")
        plot_panel(
            ax, V, F, X, cols, ttl,
            elev=elev, azim=azim, zoom=z,
            mesh_alpha=mesh_alpha, point_size=ps,
        )

    main_title = title or f"Vista paper — malla raw translúcida + nubes coloreadas"
    fig.suptitle(main_title, fontsize=18, fontweight="bold", y=0.995)

    fig.text(
        0.5, 0.962,
        f"modelo={pretty} | d21={d21_class} | vecinos={neighbors}",
        ha="center",
        va="top",
        fontsize=10.5,
        color="#333333",
    )

    legend_items = [
        ("vecino izq.", COLOR_LEFT_NEIGH),
        ("d21", COLOR_D21),
        ("vecino der.", COLOR_RIGHT_NEIGH),
        ("error", COLOR_ERROR),
        ("TP d21", COLOR_TP),
        ("malla raw", COLOR_MESH),
    ]

    x0 = 0.17
    y0 = 0.035
    dx = 0.125
    for i, (lab, col) in enumerate(legend_items):
        x = x0 + i * dx
        fig.text(x, y0, "■", color=col, fontsize=14, ha="left", va="center")
        fig.text(x + 0.017, y0, lab, color="black", fontsize=9.5, ha="left", va="center")

    fig.text(
        0.5,
        0.012,
        "Malla raw alineada y translúcida como referencia; puntos 8192 coloreados por clase/predicción/error.",
        ha="center",
        fontsize=9.5,
        color="#333333",
    )

    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.025, dpi=int(dpi), facecolor="white")
    plt.close(fig)

    return out_png


# ============================================================
# INDIVIDUAL PNGS
# ============================================================

def make_individual_pngs(
    data,
    out_dir,
    elev,
    azim,
    zoom,
    zoom_focus,
    mesh_alpha,
    point_size,
    dpi,
):
    out_dir = ensure_dir(Path(out_dir))

    X = data["X"]
    V = data["V"]
    F = data["F"]
    colors = data["colors"]
    pretty = data["pretty"]
    metrics = data["metrics"]

    metric_txt = ""
    if metrics:
        metric_txt = f"\nacc_nb={metrics['acc_no_bg']:.3f} | d21_f1={metrics['d21_f1']:.3f}"

    panels = [
        ("01_gt_multiclase", "GT\n(todas las clases)", colors["gt_all"], zoom),
        ("02_pred_multiclase", f"{pretty}\n(predicción multiclase){metric_txt}", colors["pred_all"], zoom),
        ("03_errores_multiclase", "Errores multiclase\nrojo = pred ≠ GT", colors["err_all"], zoom),
        ("04_gt_d21_vecinos", "GT: d21 + vecinos\nazul/naranja/rosado", colors["gt_focus"], zoom_focus),
        ("05_pred_d21_vecinos", f"{pretty}: d21 + vecinos\nazul/naranja/rosado", colors["pred_focus"], zoom_focus),
        ("06_error_d21", "Error d21\nverde=TP | rojo=FP/FN", colors["d21_error"], zoom_focus),
    ]

    saved = []
    for stem, ttl, cols, z in panels:
        fig = plt.figure(figsize=(7.2, 5.6), dpi=int(dpi), facecolor="white")
        ax = fig.add_subplot(111, projection="3d")
        plot_panel(
            ax, V, F, X, cols, ttl,
            elev=elev, azim=azim, zoom=z,
            mesh_alpha=mesh_alpha, point_size=point_size,
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
            x=V[:, 0], y=V[:, 1], z=V[:, 2],
            i=F[:, 0], j=F[:, 1], k=F[:, 2],
            color=COLOR_MESH,
            opacity=float(mesh_opacity),
            showscale=False,
            hoverinfo="skip",
            name="raw mesh",
        ),
        row=row, col=col,
    )


def add_plotly_points(fig, row, col, X, colors, name, point_size):
    X = finite_points(X)
    fig.add_trace(
        go.Scatter3d(
            x=X[:, 0], y=X[:, 1], z=X[:, 2],
            mode="markers",
            marker=dict(
                size=float(point_size),
                color=colors_to_plotly(colors),
                opacity=1.0,
            ),
            name=name,
            showlegend=False,
        ),
        row=row, col=col,
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


def make_plotly_html(
    data,
    out_html,
    mesh_opacity,
    point_size,
    plotly_zoom,
    title="",
):
    if not HAS_PLOTLY:
        print("[WARN] Plotly no está instalado. No se exportará HTML.")
        return None

    out_html = Path(out_html)
    ensure_dir(out_html.parent)

    X = data["X"]
    V = data["V"]
    F = data["F"]
    colors = data["colors"]
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
        horizontal_spacing=0.01,
        vertical_spacing=0.03,
    )

    panels = [
        (1, 1, colors["gt_all"], "gt_all"),
        (1, 2, colors["pred_all"], "pred_all"),
        (1, 3, colors["err_all"], "err_all"),
        (2, 1, colors["gt_focus"], "gt_focus"),
        (2, 2, colors["pred_focus"], "pred_focus"),
        (2, 3, colors["d21_error"], "d21_error"),
    ]

    for r, c, cols, name in panels:
        add_plotly_mesh(fig, r, c, V, F, mesh_opacity=mesh_opacity)
        add_plotly_points(fig, r, c, X, cols, name, point_size=point_size)

    fig.update_layout(
        title=title or f"Vista interactiva — {pretty}",
        height=980,
        width=1650,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
        margin=dict(l=5, r=5, t=75, b=5),
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

    ap.add_argument("--align_mode", default="robust_bbox",
                    choices=["none", "unit_sphere", "bbox", "robust_bbox"])

    ap.add_argument("--zoom", type=float, default=1.65)
    ap.add_argument("--zoom_focus", type=float, default=1.95)
    ap.add_argument("--elev", type=float, default=26.0)
    ap.add_argument("--azim", type=float, default=-55.0)

    ap.add_argument("--dpi", type=int, default=500)
    ap.add_argument("--point_size", type=float, default=5.8)
    ap.add_argument("--mesh_alpha", type=float, default=0.22)

    ap.add_argument("--plotly_mesh_opacity", type=float, default=0.16)
    ap.add_argument("--plotly_point_size", type=float, default=2.4)
    ap.add_argument("--plotly_zoom", type=float, default=1.35)

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

    print("[INFO] case_dir:", case_dir)
    print("[INFO] model:", model_key)
    print("[INFO] out_png:", out_png)
    print("[INFO] align_mode:", args.align_mode)

    data = prepare_case_data(
        case_dir=case_dir,
        model_key=model_key,
        d21_class=args.d21_class,
        neighbors=neighbors,
        align_mode=args.align_mode,
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
        title=args.title,
    )

    print(f"[OK] PNG resumen: {out_png}")

    individual_saved = []
    if not args.no_individual:
        if args.out_individual_dir:
            out_ind = Path(args.out_individual_dir).resolve()
        else:
            out_ind = out_png.parent / "individual_panels_v9"

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
        )
        print(f"[OK] Paneles individuales: {out_ind}")

    out_html = None
    if not args.no_plotly:
        if args.out_html:
            out_html = Path(args.out_html).resolve()
        else:
            out_html = out_png.with_suffix(".html")

        make_plotly_html(
            data=data,
            out_html=out_html,
            mesh_opacity=args.plotly_mesh_opacity,
            point_size=args.plotly_point_size,
            plotly_zoom=args.plotly_zoom,
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
        "raw_mesh_path": data.get("raw_path", ""),
        "align_meta": data.get("align_meta", {}),
        "metrics": data.get("metrics", {}),
        "colors_focus": {
            "d21": COLOR_D21,
            "left_neighbor": COLOR_LEFT_NEIGH,
            "right_neighbor": COLOR_RIGHT_NEIGH,
            "tp": COLOR_TP,
            "error": COLOR_ERROR,
        },
    }

    save_json(summary, out_png.with_suffix(".summary.json"))
    print(f"[OK] Summary JSON: {out_png.with_suffix('.summary.json')}")
    print("\nListo ✅")


if __name__ == "__main__":
    main()