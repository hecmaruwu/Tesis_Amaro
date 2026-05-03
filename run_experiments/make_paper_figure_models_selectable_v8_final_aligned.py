#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_paper_figure_models_selectable_v8_final_aligned.py

Versión final corregida:
- Fondo blanco.
- Malla raw visible y translúcida.
- Puntos 8192 dominan la predicción.
- Alineación robusta tipo v4/v5.
- Misma alineación para PNG y Plotly.
- Plotly sin proyección rara de etiquetas a la malla.
- d21 GT = magenta.
- d21 predicho = naranja.
- Vecino izquierdo = azul.
- Vecino derecho = rosa.
- TP d21 = verde.
- Error d21 FP/FN = rojo.
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

D21_DEFAULT = 8
DEFAULT_MODELS = ["pointnet"]
DEFAULT_NEIGHBORS = "d11:1,d22:9"

CLASS_COLORS = {
    0:  "#C8C8C8",
    1:  "#2CA02C",
    2:  "#17BECF",
    3:  "#1F77B4",
    4:  "#9467BD",
    5:  "#E377C2",
    6:  "#D62728",
    7:  "#FF7F0E",
    8:  "#FF4FA3",
    9:  "#FFBF00",
    10: "#BFEF45",
    11: "#2CA58D",
    12: "#00BCD4",
    13: "#C5B0D5",
    14: "#7F7F7F",
}

COLOR_MESH = "#9E9E9E"
COLOR_ERROR = "#FF0000"
COLOR_TP_D21 = "#00B050"
COLOR_D21_GT = "#FF4FA3"
COLOR_D21_PRED = "#FF8C00"
COLOR_LEFT_NEIGH = "#0070C0"
COLOR_RIGHT_NEIGH = "#FF69B4"
COLOR_OTHER = "#BDBDBD"


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
        msg = f"Longitudes distintas: X={X.shape[0]}, pred={pred.shape[0]}, gt={None if gt_arr is None else gt_arr.shape[0]}"
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
        return {"model": model_key, "available": False, "reason": f"No existe {pred_path}"}

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
# RAW MESH LOADING
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
        return V, raw_faces, {"mode": "unit_sphere", "center": c.tolist(), "scale": float(s)}

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


def labels_to_colors(labels, alpha_bg=0.18, alpha_fg=1.0):
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    out = []
    for y in labels:
        y = int(y)
        if y == 0:
            out.append(hex_to_rgba01(CLASS_COLORS[0], alpha_bg))
        else:
            out.append(hex_to_rgba01(CLASS_COLORS.get(y, "#444444"), alpha_fg))
    return out


def error_colors_all(pred, gt, alpha_ok=0.08):
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


def neighbor_focus_colors(
    labels,
    d21_class: int,
    neighbors: List[Tuple[str, int]],
    alpha_bg=0.035,
    d21_color=COLOR_D21_GT,
):
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    neigh_ids = [v for _, v in neighbors]
    out = []

    for y in labels:
        yy = int(y)

        if yy == int(d21_class):
            out.append(hex_to_rgba01(d21_color, 1.0))
        elif len(neigh_ids) >= 1 and yy == neigh_ids[0]:
            out.append(hex_to_rgba01(COLOR_LEFT_NEIGH, 1.0))
        elif len(neigh_ids) >= 2 and yy == neigh_ids[1]:
            out.append(hex_to_rgba01(COLOR_RIGHT_NEIGH, 1.0))
        elif yy == 0:
            out.append(hex_to_rgba01("#D0D0D0", alpha_bg))
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
            out.append(hex_to_rgba01(COLOR_TP_D21, 1.0))
        elif is_err:
            out.append(hex_to_rgba01(COLOR_ERROR, 1.0))
        elif len(neigh_ids) >= 1 and g == neigh_ids[0]:
            out.append(hex_to_rgba01(COLOR_LEFT_NEIGH, 0.90))
        elif len(neigh_ids) >= 2 and g == neigh_ids[1]:
            out.append(hex_to_rgba01(COLOR_RIGHT_NEIGH, 0.90))
        elif g == 0:
            out.append(hex_to_rgba01("#D0D0D0", 0.025))
        else:
            out.append(hex_to_rgba01(COLOR_OTHER, 0.045))

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
# STATIC FIGURE (PNG)
# ============================================================

def plot_mesh_matplotlib(ax, V, F, alpha=0.12):
    if V is None:
        return

    if F is not None:
        mesh = Poly3DCollection(V[F], alpha=alpha)
        mesh.set_facecolor(COLOR_MESH)
        mesh.set_edgecolor("none")
        ax.add_collection3d(mesh)
    else:
        ax.scatter(V[:,0], V[:,1], V[:,2], s=0.1, c="gray", alpha=0.05)


def plot_points(ax, X, colors, size=6.5):
    if X is None:
        return
    cols = np.array(colors)
    ax.scatter(X[:,0], X[:,1], X[:,2],
               c=cols, s=size, depthshade=False)


def auto_zoom(ax, X, zoom=2.0):
    lo, hi = X.min(0), X.max(0)
    center = (lo + hi) / 2
    scale = (hi - lo).max() / zoom

    ax.set_xlim(center[0]-scale, center[0]+scale)
    ax.set_ylim(center[1]-scale, center[1]+scale)
    ax.set_zlim(center[2]-scale, center[2]+scale)


def make_static_white_figure(
    case_dir,
    out_png,
    model_key,
    d21_class,
    neighbors,
    zoom,
    mesh_alpha
):
    case_dir = Path(case_dir)

    md = load_model_case(case_dir, model_key, require_pred_labels=True)

    X = md["xyz"]
    pred = md["pred"]
    gt = md["gt"]

    raw_path = find_raw_mesh_from_meta(case_dir)
    V, F = load_raw_mesh_vertices_faces(raw_path)

    V, F, _ = align_raw_mesh(V, F, X)

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 3)

    axes = [fig.add_subplot(gs[i,j], projection="3d") for i in range(2) for j in range(3)]

    titles = [
        "GT",
        "Pred",
        "Errores",
        "GT d21 + vecinos",
        "Pred d21 + vecinos",
        "Error d21"
    ]

    for ax, t in zip(axes, titles):
        ax.set_title(t)
        ax.axis("off")

    # 1
    plot_mesh_matplotlib(axes[0], V, F, alpha=mesh_alpha)
    plot_points(axes[0], X, labels_to_colors(gt))

    # 2
    plot_mesh_matplotlib(axes[1], V, F, alpha=mesh_alpha)
    plot_points(axes[1], X, labels_to_colors(pred))

    # 3
    plot_mesh_matplotlib(axes[2], V, F, alpha=mesh_alpha)
    plot_points(axes[2], X, error_colors_all(pred, gt))

    # 4
    plot_mesh_matplotlib(axes[3], V, F, alpha=mesh_alpha)
    plot_points(axes[3], X,
        neighbor_focus_colors(gt, d21_class, neighbors, d21_color=COLOR_D21_GT))

    # 5 (AQUÍ ESTÁ EL CAMBIO → naranja)
    plot_mesh_matplotlib(axes[4], V, F, alpha=mesh_alpha)
    plot_points(axes[4], X,
        neighbor_focus_colors(pred, d21_class, neighbors, d21_color=COLOR_D21_PRED))

    # 6
    plot_mesh_matplotlib(axes[5], V, F, alpha=mesh_alpha)
    plot_points(axes[5], X,
        d21_tp_error_colors(pred, gt, d21_class, neighbors))

    for ax in axes:
        auto_zoom(ax, X, zoom)

    plt.tight_layout()
    plt.savefig(out_png, dpi=500)
    plt.close()

def make_plotly_figure(
    case_dir,
    out_html,
    model_key,
    d21_class,
    neighbors,
    mesh_opacity=0.12,
    point_size=2.5
):
    if not HAS_PLOTLY:
        return

    case_dir = Path(case_dir)
    md = load_model_case(case_dir, model_key, require_pred_labels=True)

    X = md["xyz"]
    pred = md["pred"]
    gt = md["gt"]

    raw_path = find_raw_mesh_from_meta(case_dir)
    V, F = load_raw_mesh_vertices_faces(raw_path)
    V, F, _ = align_raw_mesh(V, F, X)

    fig = make_subplots(rows=2, cols=3,
                        specs=[[{"type":"scene"}]*3]*2)

    def add_mesh(scene):
        if V is None or F is None:
            return
        fig.add_trace(go.Mesh3d(
            x=V[:,0], y=V[:,1], z=V[:,2],
            i=F[:,0], j=F[:,1], k=F[:,2],
            color=COLOR_MESH,
            opacity=mesh_opacity
        ), row=scene[0], col=scene[1])

    def add_pts(scene, colors):
        fig.add_trace(go.Scatter3d(
            x=X[:,0], y=X[:,1], z=X[:,2],
            mode="markers",
            marker=dict(
                size=point_size,
                color=colors_to_plotly(colors)
            )
        ), row=scene[0], col=scene[1])

    # fila 1
    add_mesh((1,1)); add_pts((1,1), labels_to_colors(gt))
    add_mesh((1,2)); add_pts((1,2), labels_to_colors(pred))
    add_mesh((1,3)); add_pts((1,3), error_colors_all(pred, gt))

    # fila 2
    add_mesh((2,1)); add_pts((2,1),
        neighbor_focus_colors(gt, d21_class, neighbors, d21_color=COLOR_D21_GT))

    add_mesh((2,2)); add_pts((2,2),
        neighbor_focus_colors(pred, d21_class, neighbors, d21_color=COLOR_D21_PRED))

    add_mesh((2,3)); add_pts((2,3),
        d21_tp_error_colors(pred, gt, d21_class, neighbors))

    fig.update_layout(height=900, width=1200)
    fig.write_html(out_html)

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--case_dir", required=True)
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--out_html", required=True)

    ap.add_argument("--models", default="pointnet")
    ap.add_argument("--d21_class", type=int, default=8)
    ap.add_argument("--neighbor_teeth", default="d11:1,d22:9")

    ap.add_argument("--zoom", type=float, default=2.0)
    ap.add_argument("--mesh_alpha", type=float, default=0.12)

    args = ap.parse_args()

    models = parse_models(args.models)
    neighbors = parse_neighbor_teeth(args.neighbor_teeth)

    model_key = models[0]

    make_static_white_figure(
        args.case_dir,
        args.out_png,
        model_key,
        args.d21_class,
        neighbors,
        args.zoom,
        args.mesh_alpha
    )

    make_plotly_figure(
        args.case_dir,
        args.out_html,
        model_key,
        args.d21_class,
        neighbors
    )


if __name__ == "__main__":
    main()