#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_paper_figure_models_selectable_v4_fixed.py

Figura paper-level para comparar modelos de segmentación dental 3D.

Entrada esperada:
case_dir/
  meta.json
  common/
    bundle_case_data.npz
    01_raw_mesh.png
    02_sampled_200k_labeled.png
    03_final_8192_labeled.png
  pointnet/
    pred_labels.npy
    gt_labels.npy
    xyz_labels.npy
  pointnetpp/
    pred_labels.npy
    gt_labels.npy
    xyz_labels.npy
  dgcnn/
    pred_labels.npy
    gt_labels.npy
    xyz_labels.npy
  pointnettransformer/
    pred_labels.npy
    gt_labels.npy
    xyz_labels.npy

Características:
- Render PNG estático.
- Export Plotly HTML opcional.
- Carga robusta de predicciones dtype=object.
- Alineación raw mesh <-> nube normalizada:
    none
    robust_bbox
    unit_sphere
- Visualiza:
    raw mesh alineada
    ground truth
    predicción multiclase
    errores
    diente 21 vs resto
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap

try:
    import pyvista as pv
    HAS_PYVISTA = True
except Exception:
    HAS_PYVISTA = False

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
# -------------------- CONFIG VISUAL -------------------------
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

DEFAULT_MODELS = ["pointnet", "pointnetpp", "dgcnn", "pointnettransformer"]

# Colores tipo tab20 para clases 0..N
TAB20 = plt.get_cmap("tab20", 20)

# Fondo / encía en gris claro
BACKGROUND_COLOR = np.array([0.72, 0.72, 0.72, 1.0])


# ============================================================
# -------------------- UTILS BÁSICOS -------------------------
# ============================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def sanitize_model_name(name: str) -> str:
    name = str(name).strip().lower()
    return MODEL_DIR_ALIASES.get(name, name)


def pretty_model_name(name: str) -> str:
    key = sanitize_model_name(name)
    return MODEL_ALIASES.get(key, key)


def finite_points(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != 3:
        X = X.reshape(-1, 3)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X.astype(np.float32)


def normalize_unit_sphere(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Normaliza una nube/malla a esfera unitaria:
    X_norm = (X - center) / scale
    """
    X = finite_points(X)
    center = X.mean(axis=0, keepdims=True)
    Xc = X - center
    scale = np.linalg.norm(Xc, axis=1).max()
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return (Xc / scale).astype(np.float32), center.reshape(3).astype(np.float32), float(scale)


def robust_bounds(X: np.ndarray, q_low=0.01, q_high=0.99) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bounds robustos para evitar que outliers deformen la alineación.
    """
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
    """
    Alinea X_source al espacio de X_target usando centro y escala de bounding box.

    X_aligned = (X_source - c_src) / s_src * s_tgt + c_tgt
    """
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


# ============================================================
# -------------------- LOAD ARRAYS ROBUSTO -------------------
# ============================================================

def load_any_array(path: Path, preferred_keys: Optional[List[str]] = None) -> np.ndarray:
    """
    Carga .npy/.npz robustamente.

    Corrige:
    - allow_pickle=False fallando con dtype object.
    - arrays guardados como object.
    - npz con distintas claves.
    """
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

    arr = unwrap_object_array(arr)
    return np.asarray(arr)


def unwrap_object_array(arr) -> np.ndarray:
    """
    Convierte arrays dtype=object a arrays numéricos normales cuando sea posible.
    """
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)

    if arr.dtype != object:
        return arr

    # Caso escalar object: array(object)
    if arr.shape == ():
        try:
            arr = arr.item()
        except Exception:
            pass
        arr = np.asarray(arr)

    # Caso array con un único elemento object
    if isinstance(arr, np.ndarray) and arr.dtype == object and arr.size == 1:
        try:
            arr = arr.reshape(-1)[0]
            arr = np.asarray(arr)
        except Exception:
            pass

    # Caso lista de arrays
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        try:
            arr = np.asarray(arr.tolist())
        except Exception:
            try:
                arr = np.concatenate([np.asarray(x).reshape(-1) for x in arr.reshape(-1)], axis=0)
            except Exception:
                arr = np.asarray(arr)

    return np.asarray(arr)


def load_pred_labels(path: Path) -> np.ndarray:
    """
    Carga pred_labels.npy de forma robusta.
    Soluciona el error:
        NameError: load_pred_labels is not defined
    y también problemas dtype=object.
    """
    arr = load_any_array(
        path,
        preferred_keys=["pred", "prediction", "pred_labels", "labels", "y", "Y", "arr_0"]
    )
    arr = unwrap_object_array(arr)
    arr = np.asarray(arr).squeeze()

    if arr.ndim != 1:
        arr = arr.reshape(-1)

    return arr.astype(np.int64)


def load_gt_labels(path: Path) -> np.ndarray:
    arr = load_any_array(
        path,
        preferred_keys=["gt", "labels", "gt_labels", "y", "Y", "arr_0"]
    )
    arr = unwrap_object_array(arr)
    arr = np.asarray(arr).squeeze()

    if arr.ndim != 1:
        arr = arr.reshape(-1)

    return arr.astype(np.int64)


def load_xyz(path: Path) -> np.ndarray:
    arr = load_any_array(
        path,
        preferred_keys=["xyz", "points", "X", "arr_0"]
    )
    arr = unwrap_object_array(arr)
    arr = np.asarray(arr).squeeze()

    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]

    arr = finite_points(arr)
    return arr.astype(np.float32)


def fix_length_match(
    X: np.ndarray,
    pred: np.ndarray,
    gt: Optional[np.ndarray] = None,
    strict: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Asegura que X, pred y gt tengan el mismo largo.
    Si hay diferencias, recorta al mínimo común.
    """
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
# -------------------- LOAD CASE DATA ------------------------
# ============================================================

def load_common_bundle(case_dir: Path) -> Dict[str, np.ndarray]:
    """
    Carga common/bundle_case_data.npz si existe.
    """
    case_dir = Path(case_dir)
    bundle_path = case_dir / "common" / "bundle_case_data.npz"

    if not bundle_path.exists():
        return {}

    data = np.load(bundle_path, allow_pickle=True)
    out = {}

    for k in data.keys():
        arr = unwrap_object_array(data[k])
        out[k] = np.asarray(arr)

    return out


def load_model_case(case_dir: Path, model: str, require_pred_labels: bool = False) -> Dict:
    """
    Carga datos por modelo desde:
      case_dir/<model>/pred_labels.npy
      case_dir/<model>/gt_labels.npy
      case_dir/<model>/xyz_labels.npy
    """
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
            "model_dir": model_dir,
            "available": False,
            "reason": f"No existe {pred_path}",
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
        "model_dir": model_dir,
        "available": True,
        "pred": pred,
        "gt": gt,
        "xyz": xyz,
        "pred_path": pred_path,
        "gt_path": gt_path if gt_path.exists() else None,
        "xyz_path": xyz_path if xyz_path.exists() else None,
    }


# ============================================================
# -------------------- RAW MESH LOADING ----------------------
# ============================================================

def find_raw_mesh_from_meta(case_dir: Path) -> Optional[Path]:
    meta = read_json(Path(case_dir) / "meta.json")
    raw = meta.get("raw_mesh_path", "")
    if raw:
        p = Path(raw)
        if p.exists():
            return p
    return None


def trimesh_to_vertices_faces(mesh_tm) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    vertices = np.asarray(mesh_tm.vertices, dtype=np.float32)
    faces = None
    if hasattr(mesh_tm, "faces") and mesh_tm.faces is not None:
        faces = np.asarray(mesh_tm.faces, dtype=np.int64)
        if faces.size == 0:
            faces = None
    return vertices, faces


def load_raw_mesh_vertices_faces(raw_mesh_path: Optional[Path]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Carga raw mesh como vertices/faces usando trimesh.
    """
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

    V, F = trimesh_to_vertices_faces(loaded)
    V = finite_points(V)

    return V, F


def align_raw_mesh(
    raw_vertices: Optional[np.ndarray],
    raw_faces: Optional[np.ndarray],
    target_xyz: np.ndarray,
    align_mode: str = "robust_bbox",
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict]:
    """
    Alinea raw mesh al espacio de target_xyz.
    """
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
# -------------------- COLORS / PLOTS ------------------------
# ============================================================

def class_to_rgba(labels: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    colors = TAB20(labels % 20)

    # fondo clase 0 en gris
    mask_bg = labels == 0
    colors[mask_bg] = BACKGROUND_COLOR

    colors[:, 3] = alpha
    return colors


def error_colors(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
    Correcto: gris claro.
    Error: rojo.
    """
    pred = np.asarray(pred).reshape(-1)
    gt = np.asarray(gt).reshape(-1)

    ok = pred == gt
    colors = np.zeros((len(pred), 4), dtype=np.float32)
    colors[ok] = np.array([0.72, 0.72, 0.72, 0.65])
    colors[~ok] = np.array([1.00, 0.05, 0.02, 1.00])
    return colors


def d21_colors(labels: np.ndarray, d21_class: int) -> np.ndarray:
    """
    Diente 21: azul.
    Resto no-bg: gris medio.
    Fondo: gris claro transparente.
    """
    labels = np.asarray(labels).reshape(-1).astype(np.int64)

    colors = np.zeros((len(labels), 4), dtype=np.float32)
    colors[:] = np.array([0.80, 0.80, 0.80, 0.35])

    fg = labels != 0
    colors[fg] = np.array([0.45, 0.45, 0.45, 0.55])

    d21 = labels == int(d21_class)
    colors[d21] = np.array([0.00, 0.25, 1.00, 1.00])

    return colors


def set_axes_equal(ax):
    """
    Escala igual para 3D matplotlib.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range, 1e-6])

    ax.set_xlim3d([x_mid - plot_radius, x_mid + plot_radius])
    ax.set_ylim3d([y_mid - plot_radius, y_mid + plot_radius])
    ax.set_zlim3d([z_mid - plot_radius, z_mid + plot_radius])


def apply_view(ax, elev: float = 25, azim: float = -55, zoom: float = 1.0):
    ax.view_init(elev=elev, azim=azim)

    try:
        ax.dist = 10 / max(float(zoom), 1e-6)
    except Exception:
        pass

    ax.set_axis_off()
    set_axes_equal(ax)


def maybe_downsample(X: np.ndarray, labels_list: List[np.ndarray], max_points: int, seed: int = 123):
    X = finite_points(X)
    n = X.shape[0]

    if max_points is None or max_points <= 0 or n <= max_points:
        return X, labels_list

    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=int(max_points), replace=False)

    X2 = X[idx]
    labels2 = [np.asarray(y)[idx] if y is not None else None for y in labels_list]
    return X2, labels2


def plot_point_cloud(
    ax,
    X: np.ndarray,
    colors: np.ndarray,
    title: str,
    point_size: float = 4.0,
    elev: float = 25,
    azim: float = -55,
    zoom: float = 1.0,
):
    X = finite_points(X)
    colors = np.asarray(colors)

    ax.scatter(
        X[:, 0], X[:, 1], X[:, 2],
        c=colors,
        s=float(point_size),
        linewidths=0,
        depthshade=False,
    )

    ax.set_title(title, fontsize=10, pad=2)
    apply_view(ax, elev=elev, azim=azim, zoom=zoom)


def plot_mesh_wire_or_surface(
    ax,
    V: Optional[np.ndarray],
    F: Optional[np.ndarray],
    title: str,
    elev: float = 25,
    azim: float = -55,
    zoom: float = 1.0,
):
    if V is None:
        ax.text2D(0.5, 0.5, "Raw mesh no disponible", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()
        return

    V = finite_points(V)

    if F is not None and len(F) > 0:
        try:
            ax.plot_trisurf(
                V[:, 0], V[:, 1], V[:, 2],
                triangles=F,
                color=(0.78, 0.78, 0.78, 1.0),
                edgecolor="none",
                linewidth=0.0,
                alpha=1.0,
                shade=True,
            )
        except Exception:
            ax.scatter(
                V[:, 0], V[:, 1], V[:, 2],
                c=[(0.65, 0.65, 0.65, 0.55)],
                s=0.3,
                linewidths=0,
                depthshade=False,
            )
    else:
        ax.scatter(
            V[:, 0], V[:, 1], V[:, 2],
            c=[(0.65, 0.65, 0.65, 0.55)],
            s=0.3,
            linewidths=0,
            depthshade=False,
        )

    ax.set_title(title, fontsize=10, pad=2)
    apply_view(ax, elev=elev, azim=azim, zoom=zoom)


# ============================================================
# -------------------- MÉTRICAS SIMPLES ----------------------
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
# -------------------- FIGURA PNG ----------------------------
# ============================================================

def make_png_figure(
    case_dir: Path,
    out_png: Path,
    models: List[str],
    d21_class: int,
    align_mode: str,
    max_points: int,
    point_size: float,
    zoom: float,
    elev: float,
    azim: float,
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
        raise RuntimeError("No hay modelos con pred_labels.npy cargables.")

    # Elegir nube base:
    # prioridad: xyz del primer modelo; fallback: xyz_8192 del bundle.
    base_xyz = loaded_models[0].get("xyz")
    if base_xyz is None:
        if "xyz_8192" in common:
            base_xyz = finite_points(common["xyz_8192"])
        else:
            raise RuntimeError("No hay xyz_labels.npy ni common/bundle_case_data.npz con xyz_8192.")

    # Si algún modelo no tiene xyz, usar base.
    for md in loaded_models:
        if md.get("xyz") is None:
            md["xyz"] = base_xyz.copy()
            md["xyz_source"] = "fallback_base_xyz"
        else:
            md["xyz_source"] = str(md.get("xyz_path", ""))

    raw_mesh_path = find_raw_mesh_from_meta(case_dir)
    raw_V, raw_F = load_raw_mesh_vertices_faces(raw_mesh_path)

    raw_V_aligned, raw_F, align_meta = align_raw_mesh(
        raw_vertices=raw_V,
        raw_faces=raw_F,
        target_xyz=base_xyz,
        align_mode=align_mode,
    )

    # Ground truth:
    gt = loaded_models[0].get("gt")
    if gt is None and "y_8192" in common:
        gt = np.asarray(common["y_8192"]).reshape(-1).astype(np.int64)

    # Longitud para GT base
    if gt is not None:
        base_xyz_gt, _, gt_fixed = fix_length_match(base_xyz, np.zeros(base_xyz.shape[0], dtype=np.int64), gt, strict=False)
        base_xyz = base_xyz_gt
        gt = gt_fixed

    n_models = len(loaded_models)

    # Layout:
    # fila 0: raw mesh + ground truth
    # por cada modelo: pred, error, d21
    n_cols = 3
    n_rows = 1 + n_models

    fig = plt.figure(figsize=(5.2 * n_cols, 4.6 * n_rows), dpi=180)
    gs = GridSpec(n_rows, n_cols, figure=fig, wspace=0.02, hspace=0.12)

    fig.suptitle(
        title or f"Paper figure — {meta.get('sample_name', case_dir.name)}",
        fontsize=15,
        y=0.995,
    )

    ax_raw = fig.add_subplot(gs[0, 0], projection="3d")
    plot_mesh_wire_or_surface(
        ax_raw,
        raw_V_aligned,
        raw_F,
        title=f"Raw mesh alineada\n{align_mode}",
        elev=elev,
        azim=azim,
        zoom=zoom,
    )

    ax_gt = fig.add_subplot(gs[0, 1], projection="3d")
    if gt is not None:
        Xp, [gtp] = maybe_downsample(base_xyz, [gt], max_points=max_points)
        plot_point_cloud(
            ax_gt,
            Xp,
            class_to_rgba(gtp),
            title="Ground truth 8192",
            point_size=point_size,
            elev=elev,
            azim=azim,
            zoom=zoom,
        )
    else:
        ax_gt.text2D(0.5, 0.5, "GT no disponible", ha="center", va="center", transform=ax_gt.transAxes)
        ax_gt.set_axis_off()
        ax_gt.set_title("Ground truth", fontsize=10)

    ax_d21_gt = fig.add_subplot(gs[0, 2], projection="3d")
    if gt is not None:
        Xp, [gtp] = maybe_downsample(base_xyz, [gt], max_points=max_points)
        plot_point_cloud(
            ax_d21_gt,
            Xp,
            d21_colors(gtp, d21_class=d21_class),
            title=f"GT diente 21\nclase interna={d21_class}",
            point_size=point_size,
            elev=elev,
            azim=azim,
            zoom=zoom,
        )
    else:
        ax_d21_gt.text2D(0.5, 0.5, "GT no disponible", ha="center", va="center", transform=ax_d21_gt.transAxes)
        ax_d21_gt.set_axis_off()
        ax_d21_gt.set_title("GT diente 21", fontsize=10)

    metrics_summary = {}

    for r, md in enumerate(loaded_models, start=1):
        model_name = md["model"]
        pretty = md.get("pretty", pretty_model_name(model_name))

        X = md["xyz"]
        pred = md["pred"]
        gt_model = md.get("gt", gt)

        X, pred, gt_model = fix_length_match(X, pred, gt_model, strict=False)

        metrics_summary[model_name] = compute_simple_metrics(pred, gt_model, d21_class=d21_class)

        # Predicción multiclase
        ax_pred = fig.add_subplot(gs[r, 0], projection="3d")
        Xp, [predp] = maybe_downsample(X, [pred], max_points=max_points)
        mtxt = ""
        if metrics_summary[model_name]:
            mtxt = f"\nacc_nb={metrics_summary[model_name]['acc_no_bg']:.3f} | d21_f1={metrics_summary[model_name]['d21_f1']:.3f}"
        plot_point_cloud(
            ax_pred,
            Xp,
            class_to_rgba(predp),
            title=f"{pretty} — predicción{mtxt}",
            point_size=point_size,
            elev=elev,
            azim=azim,
            zoom=zoom,
        )

        # Errores
        ax_err = fig.add_subplot(gs[r, 1], projection="3d")
        if gt_model is not None:
            Xp, [predp, gtp] = maybe_downsample(X, [pred, gt_model], max_points=max_points)
            plot_point_cloud(
                ax_err,
                Xp,
                error_colors(predp, gtp),
                title=f"{pretty} — errores\nrojo = pred ≠ GT",
                point_size=point_size,
                elev=elev,
                azim=azim,
                zoom=zoom,
            )
        else:
            ax_err.text2D(0.5, 0.5, "GT no disponible", ha="center", va="center", transform=ax_err.transAxes)
            ax_err.set_axis_off()
            ax_err.set_title(f"{pretty} — errores", fontsize=10)

        # Diente 21 predicho
        ax_d21 = fig.add_subplot(gs[r, 2], projection="3d")
        Xp, [predp] = maybe_downsample(X, [pred], max_points=max_points)
        plot_point_cloud(
            ax_d21,
            Xp,
            d21_colors(predp, d21_class=d21_class),
            title=f"{pretty} — diente 21 predicho",
            point_size=point_size,
            elev=elev,
            azim=azim,
            zoom=zoom,
        )

    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    run_summary = {
        "case_dir": str(case_dir),
        "out_png": str(out_png),
        "models": [md["model"] for md in loaded_models],
        "d21_class": int(d21_class),
        "align_mode": align_mode,
        "align_meta": align_meta,
        "raw_mesh_path": str(raw_mesh_path) if raw_mesh_path else "",
        "metrics_summary": metrics_summary,
    }

    save_json(run_summary, out_png.with_suffix(".summary.json"))
    return run_summary


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


def add_plotly_points(fig, row, col, X, colors, name, marker_size=2.5):
    X = finite_points(X)
    fig.add_trace(
        go.Scatter3d(
            x=X[:, 0],
            y=X[:, 1],
            z=X[:, 2],
            mode="markers",
            marker=dict(
                size=marker_size,
                color=rgba_to_plotly(colors),
                opacity=1.0,
            ),
            name=name,
            showlegend=False,
        ),
        row=row,
        col=col,
    )


def add_plotly_mesh(fig, row, col, V, F, name="raw mesh"):
    if V is None:
        return

    V = finite_points(V)

    if F is not None and len(F) > 0:
        fig.add_trace(
            go.Mesh3d(
                x=V[:, 0],
                y=V[:, 1],
                z=V[:, 2],
                i=F[:, 0],
                j=F[:, 1],
                k=F[:, 2],
                color="lightgray",
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
                marker=dict(size=1.0, color="lightgray"),
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
    max_points: int,
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

    raw_mesh_path = find_raw_mesh_from_meta(case_dir)
    raw_V, raw_F = load_raw_mesh_vertices_faces(raw_mesh_path)
    raw_V_aligned, raw_F, align_meta = align_raw_mesh(raw_V, raw_F, base_xyz, align_mode=align_mode)

    gt = loaded_models[0].get("gt")
    if gt is None and "y_8192" in common:
        gt = np.asarray(common["y_8192"]).reshape(-1).astype(np.int64)

    n_rows = 1 + len(loaded_models)
    n_cols = 3

    subplot_titles = ["Raw mesh", "Ground truth", f"GT d21={d21_class}"]
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

    add_plotly_mesh(fig, 1, 1, raw_V_aligned, raw_F, name="raw mesh")

    if gt is not None:
        Xp, [gtp] = maybe_downsample(base_xyz, [gt], max_points=max_points)
        add_plotly_points(fig, 1, 2, Xp, class_to_rgba(gtp), "GT", marker_size=2.5)
        add_plotly_points(fig, 1, 3, Xp, d21_colors(gtp, d21_class), "GT d21", marker_size=2.5)

    for r, md in enumerate(loaded_models, start=2):
        X = md.get("xyz", base_xyz)
        pred = md["pred"]
        gt_model = md.get("gt", gt)

        X, pred, gt_model = fix_length_match(X, pred, gt_model, strict=False)

        Xp, [predp] = maybe_downsample(X, [pred], max_points=max_points)
        add_plotly_points(fig, r, 1, Xp, class_to_rgba(predp), f"{md['model']} pred", marker_size=2.5)

        if gt_model is not None:
            Xp, [predp, gtp] = maybe_downsample(X, [pred, gt_model], max_points=max_points)
            add_plotly_points(fig, r, 2, Xp, error_colors(predp, gtp), f"{md['model']} errors", marker_size=2.5)

        Xp, [predp] = maybe_downsample(X, [pred], max_points=max_points)
        add_plotly_points(fig, r, 3, Xp, d21_colors(predp, d21_class), f"{md['model']} d21", marker_size=2.5)

    fig.update_layout(
        title=title or f"Paper figure interactive — {meta.get('sample_name', case_dir.name)}",
        height=max(800, 360 * n_rows),
        width=1500,
        margin=dict(l=5, r=5, t=70, b=5),
    )

    # Ocultar ejes de todas las escenas
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

def parse_models(s: str) -> List[str]:
    if not s:
        return DEFAULT_MODELS
    out = []
    for x in s.replace(",", " ").split():
        x = sanitize_model_name(x)
        if x:
            out.append(x)
    return out or DEFAULT_MODELS


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--case_dir", required=True, help="Carpeta del caso, ej: row_005_6BWQC0CT_upper")
    parser.add_argument("--out_png", required=True, help="Ruta de salida PNG")
    parser.add_argument("--out_html", default="", help="Ruta de salida HTML Plotly opcional")

    parser.add_argument("--models", default="pointnet pointnetpp dgcnn pointnettransformer",
                        help="Modelos a visualizar. Ej: 'pointnet pointnetpp dgcnn transformer'")

    parser.add_argument("--d21_class", type=int, default=8)
    parser.add_argument("--align_mode", default="robust_bbox",
                        choices=["none", "unit_sphere", "bbox", "robust_bbox"])

    parser.add_argument("--max_points", type=int, default=8192)
    parser.add_argument("--point_size", type=float, default=4.0)
    parser.add_argument("--zoom", type=float, default=1.55)
    parser.add_argument("--elev", type=float, default=25.0)
    parser.add_argument("--azim", type=float, default=-55.0)

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

    summary_png = make_png_figure(
        case_dir=case_dir,
        out_png=out_png,
        models=models,
        d21_class=args.d21_class,
        align_mode=args.align_mode,
        max_points=args.max_points,
        point_size=args.point_size,
        zoom=args.zoom,
        elev=args.elev,
        azim=args.azim,
        require_pred_labels=args.require_pred_labels,
        title=args.title,
    )

    print(f"[OK] PNG guardado: {out_png}")
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
            max_points=args.max_points,
            require_pred_labels=args.require_pred_labels,
            title=args.title,
        )
        print(f"[OK] Plotly HTML guardado: {out_html}")

    print("\nListo ✅")


if __name__ == "__main__":
    main()
