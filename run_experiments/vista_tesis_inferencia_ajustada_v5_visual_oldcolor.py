#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vista_tesis_inferencia_ajustada_v5_visual_oldcolor.py

Versión híbrida para análisis cualitativo de tesis.

Objetivo:
- Mantener la lógica nueva de ejecución: casos, modelos, vistas, títulos,
  métricas, rutas de salida, HTML/PNG y detección de malla.
- Recuperar la lógica visual antigua: colores cálidos, beige limpio,
  fondo/malla separados por capas y sin oscurecer los dientes beige.

Estructura esperada por caso:
  BASE_DIR/
    best_row091_01MAVT6A_upper/
      common/bundle_case_data.npz
      common/raw_mesh.obj                 opcional
      pointnet/{xyz_labels.npy,gt_labels.npy,pred_labels.npy}
      pointnetpp/{...}
      dgcnn/{...}
      pointnettransformer/{...}
    median_row089_015RHV4X_upper/
    worst_row071_3EU06ZN9_upper/

También intenta encontrar automáticamente la malla original en:
  /home/htaucare/Tesis_Amaro/case_comparisons/metric_case_selection_v5_global_meshes/selected_global_meshes

Autor: generado para Tesis_Amaro.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import trimesh
    HAS_TRIMESH = True
except Exception:
    trimesh = None
    HAS_TRIMESH = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except Exception:
    go = None
    make_subplots = None
    HAS_PLOTLY = False


# ============================================================
# CONFIGURACIÓN VISUAL ANTIGUA / V17
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
DEFAULT_CASES = ["best", "median", "worst"]
DEFAULT_VIEWS = ["isometrica_ajustada", "oclusal_paper"]

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

# Paleta multiclase antigua usada en las visualizaciones v17.
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

COLOR_BG = "#000000"
COLOR_BG_LIGHT = "#787878"
COLOR_MESH = "#B4B4B4"

# Por defecto dejo el beige claro que no se ensucia tanto en Plotly.
# Si quieres el beige más clásico v17, usa: --other_teeth_color '#E8C39E'
COLOR_OTHER_TEETH_DEFAULT = "#F2D1A9"
COLOR_OTHER_TEETH_V17 = "#E8C39E"

COLOR_D21 = "#FF8C00"
COLOR_LEFT_NEIGH = "#0070C0"
COLOR_RIGHT_NEIGH = "#FF69B4"
COLOR_ERROR = "#FF0000"
COLOR_TP = "#00B050"

# Alphas antiguos: se aplican por traza, no dentro de un array RGBA mezclado.
ALPHA_BG_MAIN = 0.56
ALPHA_BG_FOCUS = 0.07
ALPHA_BG_ERROR = 0.08
ALPHA_OTHER_FOCUS = 0.92
ALPHA_OTHER_ERROR = 0.90
ALPHA_ERROR = 0.98
ALPHA_TP = 1.00
ALPHA_FOCAL = 1.00

D21_DEFAULT = 8
DEFAULT_NEIGHBORS = "d11:1,d22:9"

# Ruta conocida de mallas seleccionadas.
DEFAULT_SELECTED_MESH_ROOT = Path(
    "/home/htaucare/Tesis_Amaro/case_comparisons/metric_case_selection_v5_global_meshes/selected_global_meshes"
)


# ============================================================
# CÁMARAS GUARDADAS
# ============================================================

CAMERA_PRESETS: Dict[str, Dict[str, Any]] = {
    # Vista cuasi-isométrica ajustada manualmente desde el HTML.
    "isometrica_ajustada": {
        "up": {
            "x": -0.3290923907481538,
            "y": -0.3739973520385337,
            "z": 0.8670779544076933,
        },
        "center": {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
        },
        "eye": {
            "x": 0.6146153334439807,
            "y": 0.7761467722695654,
            "z": 0.5680481951961102,
        },
        "projection": {"type": "perspective"},
    },

    # Vista oclusal superior oblicua ajustada manualmente desde el panel GT multiclase.
    "oclusal_paper": {
        "up": {"x": 0, "y": 1, "z": 0},
        "center": {"x": 0, "y": 0, "z": 0},
        "eye": {
            "x": 0.1950066638410309,
            "y": 0.195006663841031,
            "z": 1.1978980778806192,
        },
        "projection": {"type": "perspective"},
    },

    # Vista extra por defecto: lateral derecha.
    "lateral_derecha_extra": {
        "up": {"x": 0, "y": 0, "z": 1},
        "center": {"x": 0, "y": 0, "z": 0},
        "eye": {
            "x": 1.0947373729242973,
            "y": 5.462293827280975e-17,
            "z": 0.07464118451756574,
        },
        "projection": {"type": "perspective"},
    },
}

VIEW_DISPLAY = {
    "isometrica_ajustada": "Vista cuasi-isométrica",
    "oclusal_paper": "Vista oclusal superior",
    "lateral_derecha_extra": "Vista lateral derecha",
    "extra": "Vista extra",
}

CASE_DISPLAY = {
    "best": "Mejor caso",
    "median": "Caso mediano",
    "worst": "Peor caso",
}


# ============================================================
# UTILIDADES GENERALES
# ============================================================

def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def normalize_model_name(model: str) -> str:
    m = model.lower().strip()
    if m not in MODEL_DIR_ALIASES:
        raise ValueError(f"Modelo no soportado: {model}. Usa: {list(MODEL_DIR_ALIASES)}")
    return "pointnettransformer" if m == "transformer" else m


def pretty_model(model: str) -> str:
    return MODEL_ALIASES.get(model.lower(), model)


def parse_neighbors(text: str) -> Dict[str, int]:
    out = {}
    if not text:
        return out
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Formato inválido en neighbor_teeth: {item}. Usa nombre:clase")
        name, val = item.split(":", 1)
        out[name.strip()] = int(val.strip())
    return out


def first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p is not None and p.exists():
            return p
    return None


def load_npy_flex(path: Path) -> np.ndarray:
    """Carga .npy/.npz con allow_pickle=True y normaliza arrays objeto simples."""
    if not path.exists():
        raise FileNotFoundError(path)
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.lib.npyio.NpzFile):
        keys = list(obj.keys())
        if len(keys) == 1:
            arr = obj[keys[0]]
        elif "X" in obj:
            arr = obj["X"]
        elif "Y" in obj:
            arr = obj["Y"]
        else:
            arr = obj[keys[0]]
    else:
        arr = obj

    arr = np.asarray(arr)
    if arr.dtype == object:
        if arr.shape == ():
            arr = np.asarray(arr.item())
        elif arr.size == 1:
            arr = np.asarray(arr.reshape(-1)[0])
    return np.asarray(arr)


def load_bundle(case_dir: Path) -> Dict[str, Any]:
    bundle_path = case_dir / "common" / "bundle_case_data.npz"
    if not bundle_path.exists():
        return {}
    data = np.load(bundle_path, allow_pickle=True)
    out = {}
    for k in data.keys():
        val = data[k]
        if isinstance(val, np.ndarray) and val.shape == ():
            try:
                val = val.item()
            except Exception:
                pass
        out[k] = val
    return out


def safe_string(x: Any) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    if isinstance(x, np.ndarray):
        if x.shape == ():
            return safe_string(x.item())
        if x.size == 1:
            return safe_string(x.reshape(-1)[0])
    return str(x)


def extract_case_key(case_dir: Path) -> str:
    name = case_dir.name.lower()
    if name.startswith("best"):
        return "best"
    if name.startswith("median") or "median" in name:
        return "median"
    if name.startswith("worst"):
        return "worst"
    return name


def extract_sample_id(case_dir: Path, bundle: Dict[str, Any]) -> str:
    if "sample_name" in bundle:
        s = safe_string(bundle["sample_name"])
        if s and s.lower() != "none":
            return s

    # Ej: best_row091_01MAVT6A_upper -> 01MAVT6A
    m = re.search(r"row\d+_([A-Za-z0-9]+)_upper", case_dir.name)
    if m:
        return m.group(1)

    # Ej: median_row089_015RHV4X_upper
    parts = case_dir.name.split("_")
    for p in parts:
        if re.match(r"^[A-Za-z0-9]{6,}$", p) and not p.lower().startswith("row"):
            return p
    return case_dir.name


def discover_case_dirs(base_dir: Path, requested: Sequence[str]) -> List[Path]:
    base_dir = Path(base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"No existe base_dir: {base_dir}")

    all_dirs = [p for p in sorted(base_dir.iterdir()) if p.is_dir()]
    out: List[Path] = []

    for key in requested:
        key_low = key.lower().strip()
        if key_low == "all":
            for candidate in ["best", "median", "worst"]:
                out.extend(discover_case_dirs(base_dir, [candidate]))
            continue

        if Path(key).exists():
            out.append(Path(key))
            continue

        matches = []
        if key_low == "best":
            matches = [p for p in all_dirs if p.name.lower().startswith("best")]
        elif key_low == "median":
            matches = [p for p in all_dirs if p.name.lower().startswith("median") or "median" in p.name.lower()]
        elif key_low == "worst":
            matches = [p for p in all_dirs if p.name.lower().startswith("worst")]
        else:
            matches = [p for p in all_dirs if key_low in p.name.lower()]

        if not matches:
            raise FileNotFoundError(f"No encontré case_dir para '{key}' dentro de {base_dir}")
        out.append(matches[0])

    # quitar duplicados preservando orden
    uniq = []
    seen = set()
    for p in out:
        if str(p) not in seen:
            uniq.append(p)
            seen.add(str(p))
    return uniq


# ============================================================
# MÉTRICAS
# ============================================================

def compute_confusion(gt: np.ndarray, pred: np.ndarray, num_classes: int) -> np.ndarray:
    gt = gt.astype(np.int64).reshape(-1)
    pred = pred.astype(np.int64).reshape(-1)
    valid = (gt >= 0) & (gt < num_classes) & (pred >= 0) & (pred < num_classes)
    idx = gt[valid] * num_classes + pred[valid]
    cm = np.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return cm


def binary_metrics(mask_gt: np.ndarray, mask_pred: np.ndarray) -> Dict[str, float]:
    mask_gt = mask_gt.astype(bool).reshape(-1)
    mask_pred = mask_pred.astype(bool).reshape(-1)
    tp = float(np.sum(mask_gt & mask_pred))
    fp = float(np.sum(~mask_gt & mask_pred))
    fn = float(np.sum(mask_gt & ~mask_pred))
    tn = float(np.sum(~mask_gt & ~mask_pred))
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2.0 * prec * rec / (prec + rec + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "iou": iou}


def macro_f1_iou_no_bg(gt: np.ndarray, pred: np.ndarray, bg_index: int = 0) -> Dict[str, float]:
    max_cls = int(max(np.max(gt), np.max(pred))) if gt.size and pred.size else 0
    num_classes = max_cls + 1
    cm = compute_confusion(gt, pred, num_classes=num_classes)
    f1s = []
    ious = []
    for c in range(num_classes):
        if c == bg_index:
            continue
        tp = float(cm[c, c])
        fp = float(cm[:, c].sum() - tp)
        fn = float(cm[c, :].sum() - tp)
        support = float(cm[c, :].sum())
        pred_support = float(cm[:, c].sum())
        if support <= 0 and pred_support <= 0:
            continue
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2.0 * prec * rec / (prec + rec + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        f1s.append(f1)
        ious.append(iou)
    return {
        "f1_macro_no_bg": float(np.mean(f1s)) if f1s else float("nan"),
        "iou_macro_no_bg": float(np.mean(ious)) if ious else float("nan"),
    }


def compute_metrics_for_title(gt: np.ndarray, pred: np.ndarray, d21_class: int, bg_index: int) -> Dict[str, float]:
    d21 = binary_metrics(gt == d21_class, pred == d21_class)
    macro = macro_f1_iou_no_bg(gt, pred, bg_index=bg_index)
    return {
        "d21_f1": d21["f1"],
        "d21_iou": d21["iou"],
        "f1_macro_no_bg": macro["f1_macro_no_bg"],
        "iou_macro_no_bg": macro["iou_macro_no_bg"],
    }


# ============================================================
# CARGA DE DATOS Y MALLAS
# ============================================================

def load_case_model_data(case_dir: Path, model: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve xyz, gt, pred para un caso/modelo."""
    model_norm = normalize_model_name(model)
    model_dir = case_dir / MODEL_DIR_ALIASES[model_norm]
    bundle = load_bundle(case_dir)

    xyz_path = first_existing([
        model_dir / "xyz_labels.npy",
        model_dir / "xyz.npy",
        model_dir / "points.npy",
        case_dir / "common" / "xyz_labels.npy",
    ])
    gt_path = first_existing([
        model_dir / "gt_labels.npy",
        model_dir / "y_true.npy",
        model_dir / "labels_gt.npy",
        case_dir / "common" / "gt_labels.npy",
    ])
    pred_path = first_existing([
        model_dir / "pred_labels.npy",
        model_dir / "y_pred.npy",
        model_dir / "labels_pred.npy",
    ])

    if xyz_path is not None:
        xyz = load_npy_flex(xyz_path)
    elif "xyz_8192" in bundle:
        xyz = np.asarray(bundle["xyz_8192"])
    elif "xyz" in bundle:
        xyz = np.asarray(bundle["xyz"])
    else:
        raise FileNotFoundError(f"No encontré xyz para {case_dir.name}/{model_norm}")

    if gt_path is not None:
        gt = load_npy_flex(gt_path)
    elif "y_8192" in bundle:
        gt = np.asarray(bundle["y_8192"])
    elif "labels_8192" in bundle:
        gt = np.asarray(bundle["labels_8192"])
    else:
        raise FileNotFoundError(f"No encontré gt_labels para {case_dir.name}/{model_norm}")

    if pred_path is None:
        raise FileNotFoundError(f"No encontré pred_labels para {case_dir.name}/{model_norm}: {model_dir}")
    pred = load_npy_flex(pred_path)

    xyz = np.asarray(xyz, dtype=np.float32)
    gt = np.asarray(gt).astype(np.int64).reshape(-1)
    pred = np.asarray(pred).astype(np.int64).reshape(-1)

    if xyz.ndim == 3 and xyz.shape[0] == 1:
        xyz = xyz[0]
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz inválido en {case_dir.name}/{model_norm}: shape={xyz.shape}")

    n = min(xyz.shape[0], gt.shape[0], pred.shape[0])
    if n <= 0:
        raise ValueError(f"Datos vacíos en {case_dir.name}/{model_norm}")
    if xyz.shape[0] != n or gt.shape[0] != n or pred.shape[0] != n:
        print(
            f"[WARN] Longitudes distintas en {case_dir.name}/{model_norm}. "
            f"Uso n={n}. xyz={xyz.shape[0]} gt={gt.shape[0]} pred={pred.shape[0]}",
            file=sys.stderr,
        )
        xyz = xyz[:n]
        gt = gt[:n]
        pred = pred[:n]

    xyz = np.nan_to_num(xyz, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return xyz, gt, pred


def find_mesh_for_case(case_dir: Path, selected_mesh_root: Path) -> Optional[Path]:
    """Busca raw_mesh.obj o la malla seleccionada original."""
    direct_candidates = [
        case_dir / "common" / "raw_mesh.obj",
        case_dir / "common" / "raw_mesh.ply",
        case_dir / "common" / "raw_mesh.stl",
        case_dir / "raw_mesh.obj",
        case_dir / "raw_mesh.ply",
        case_dir / "raw_mesh.stl",
    ]
    found = first_existing(direct_candidates)
    if found is not None:
        return found

    key = extract_case_key(case_dir)
    case_name = case_dir.name
    sample_id = extract_sample_id(case_dir, load_bundle(case_dir))

    root = Path(selected_mesh_root)
    if root.exists():
        candidate_dirs = []
        if key == "best":
            candidate_dirs.append(root / "best")
        elif key == "median":
            candidate_dirs.append(root / "closest_to_median")
            candidate_dirs.append(root / "median")
        elif key == "worst":
            candidate_dirs.append(root / "worst")
        candidate_dirs.append(root)

        patterns = [f"*{sample_id}*upper*.obj", f"*{sample_id}*.obj", "*.obj", "*.ply", "*.stl"]
        for d in candidate_dirs:
            if not d.exists():
                continue
            for pat in patterns:
                hits = sorted(d.glob(pat))
                if hits:
                    return hits[0]

    # Último recurso: búsqueda local dentro del caso.
    for ext in ("*.obj", "*.ply", "*.stl"):
        hits = sorted(case_dir.rglob(ext))
        if hits:
            return hits[0]

    return None


def load_mesh_vertices_faces(path: Optional[Path]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if path is None:
        return None, None
    if not HAS_TRIMESH:
        print("[WARN] trimesh no está disponible. Se omite malla.", file=sys.stderr)
        return None, None
    try:
        obj = trimesh.load(path, process=False)
        if isinstance(obj, trimesh.Scene):
            meshes = [g for g in obj.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if not meshes:
                return None, None
            mesh = trimesh.util.concatenate(meshes)
        elif isinstance(obj, trimesh.Trimesh):
            mesh = obj
        else:
            return None, None

        v = np.asarray(mesh.vertices, dtype=np.float32)
        f = np.asarray(mesh.faces, dtype=np.int64) if mesh.faces is not None else None
        if v.ndim != 2 or v.shape[1] != 3 or v.shape[0] == 0:
            return None, None
        if f is None or f.ndim != 2 or f.shape[1] != 3 or f.shape[0] == 0:
            return None, None
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        return v, f
    except Exception as e:
        print(f"[WARN] No pude cargar malla {path}: {e}", file=sys.stderr)
        return None, None


def unit_sphere_normalize(points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=np.float32).copy()
    if p.size == 0:
        return p
    p -= p.mean(axis=0, keepdims=True)
    r = np.linalg.norm(p, axis=1).max()
    if np.isfinite(r) and r > 0:
        p /= r
    return p


def bbox_align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Alineación simple por centro y escala de bounding box."""
    s = np.asarray(source, dtype=np.float32).copy()
    t = np.asarray(target, dtype=np.float32)
    if s.size == 0 or t.size == 0:
        return s
    smin, smax = np.min(s, axis=0), np.max(s, axis=0)
    tmin, tmax = np.min(t, axis=0), np.max(t, axis=0)
    sc = 0.5 * (smin + smax)
    tc = 0.5 * (tmin + tmax)
    ss = np.linalg.norm(smax - smin)
    ts = np.linalg.norm(tmax - tmin)
    scale = ts / ss if ss > 1e-8 else 1.0
    return (s - sc[None, :]) * scale + tc[None, :]


def align_mesh_vertices(vertices: Optional[np.ndarray], xyz: np.ndarray, mode: str) -> Optional[np.ndarray]:
    if vertices is None:
        return None
    mode = mode.lower().strip()
    if mode == "none":
        return vertices.astype(np.float32)
    if mode == "unit":
        return unit_sphere_normalize(vertices)
    if mode == "bbox":
        return bbox_align(vertices, xyz)
    if mode == "unit_bbox":
        v = unit_sphere_normalize(vertices)
        return bbox_align(v, xyz)
    raise ValueError(f"mesh_align no soportado: {mode}")


# ============================================================
# SUBSAMPLING PARA PLOT
# ============================================================

def choose_plot_indices(
    gt: np.ndarray,
    pred: np.ndarray,
    max_points: int,
    d21_class: int,
    neighbor_classes: Sequence[int],
    seed: int,
) -> np.ndarray:
    n = gt.shape[0]
    if max_points <= 0 or n <= max_points:
        return np.arange(n, dtype=np.int64)

    rng = np.random.default_rng(seed)
    important = (gt == d21_class) | (pred == d21_class) | (gt != pred)
    for c in neighbor_classes:
        important |= (gt == c) | (pred == c)

    idx_imp = np.where(important)[0]
    idx_other = np.where(~important)[0]

    if idx_imp.size >= max_points:
        return np.sort(rng.choice(idx_imp, size=max_points, replace=False))

    rem = max_points - idx_imp.size
    if idx_other.size > 0:
        idx_add = rng.choice(idx_other, size=min(rem, idx_other.size), replace=False)
        idx = np.concatenate([idx_imp, idx_add])
    else:
        idx = idx_imp
    return np.sort(idx.astype(np.int64))


# ============================================================
# TRAZAS PLOTLY
# ============================================================

def make_scene_name(row: int, col: int) -> str:
    idx = (row - 1) * 3 + col
    return "scene" if idx == 1 else f"scene{idx}"


def add_mesh_trace(
    fig: Any,
    vertices: Optional[np.ndarray],
    faces: Optional[np.ndarray],
    row: int,
    col: int,
    opacity: float,
    mesh_color: str,
    name: str = "malla raw",
    showlegend: bool = False,
) -> None:
    if vertices is None or faces is None or opacity <= 0:
        return
    if vertices.shape[0] == 0 or faces.shape[0] == 0:
        return
    fig.add_trace(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color=mesh_color,
            opacity=opacity,
            name=name,
            hoverinfo="skip",
            showscale=False,
            showlegend=showlegend,
            flatshading=False,
            lighting=dict(
                ambient=0.72,
                diffuse=0.70,
                fresnel=0.02,
                roughness=0.80,
                specular=0.05,
            ),
            lightposition=dict(x=100, y=150, z=200),
        ),
        row=row,
        col=col,
    )


def add_points_trace(
    fig: Any,
    xyz: np.ndarray,
    mask: np.ndarray,
    row: int,
    col: int,
    color: str,
    opacity: float,
    size: float,
    name: str,
    showlegend: bool = False,
) -> None:
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() == 0 or opacity <= 0 or size <= 0:
        return
    pts = xyz[mask]
    fig.add_trace(
        go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode="markers",
            name=name,
            showlegend=showlegend,
            hoverinfo="skip",
            marker=dict(
                size=size,
                color=color,
                opacity=opacity,
                line=dict(width=0),
            ),
        ),
        row=row,
        col=col,
    )


def add_multiclass_panel(
    fig: Any,
    xyz: np.ndarray,
    labels: np.ndarray,
    row: int,
    col: int,
    point_size: float,
    bg_opacity: float,
    tooth_opacity: float,
    showlegend: bool,
) -> None:
    # Primero background, luego dientes. Así no se tapa el color dental.
    labels = labels.reshape(-1)
    add_points_trace(
        fig, xyz, labels == 0, row, col,
        color=COLOR_BG,
        opacity=bg_opacity,
        size=point_size,
        name="background",
        showlegend=showlegend,
    )

    for c in sorted(int(x) for x in np.unique(labels) if int(x) != 0):
        add_points_trace(
            fig, xyz, labels == c, row, col,
            color=CLASS_COLORS.get(c, "#777777"),
            opacity=tooth_opacity,
            size=point_size,
            name=f"{c} - {CLASS_NAMES.get(c, f'd{c}')}",
            showlegend=showlegend,
        )


def add_multiclass_error_panel(
    fig: Any,
    xyz: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    row: int,
    col: int,
    point_size: float,
    other_teeth_color: str,
    showlegend: bool,
) -> None:
    err = gt != pred
    bg_correct = (gt == 0) & (~err)
    tooth_correct = (gt != 0) & (~err)

    # Base muy tenue.
    add_points_trace(
        fig, xyz, bg_correct, row, col,
        color=COLOR_BG_LIGHT,
        opacity=ALPHA_BG_ERROR,
        size=point_size,
        name="correct bg",
        showlegend=False,
    )
    # Dientes correctos beige, separados de la malla/fondo.
    add_points_trace(
        fig, xyz, tooth_correct, row, col,
        color=other_teeth_color,
        opacity=ALPHA_OTHER_ERROR,
        size=point_size,
        name="dientes correctos",
        showlegend=showlegend,
    )
    # Errores arriba.
    add_points_trace(
        fig, xyz, err, row, col,
        color=COLOR_ERROR,
        opacity=ALPHA_ERROR,
        size=point_size * 1.03,
        name="error",
        showlegend=showlegend,
    )


def add_focus_panel(
    fig: Any,
    xyz: np.ndarray,
    labels: np.ndarray,
    row: int,
    col: int,
    point_size: float,
    d21_class: int,
    left_neighbor: Optional[int],
    right_neighbor: Optional[int],
    other_teeth_color: str,
    showlegend: bool,
) -> None:
    labels = labels.reshape(-1)
    bg = labels == 0
    d21 = labels == d21_class
    left = labels == left_neighbor if left_neighbor is not None else np.zeros_like(labels, dtype=bool)
    right = labels == right_neighbor if right_neighbor is not None else np.zeros_like(labels, dtype=bool)
    focal = d21 | left | right
    other = (labels != 0) & (~focal)

    add_points_trace(fig, xyz, bg, row, col, COLOR_BG_LIGHT, ALPHA_BG_FOCUS, point_size, "background tenue", False)
    add_points_trace(fig, xyz, other, row, col, other_teeth_color, ALPHA_OTHER_FOCUS, point_size, "otros dientes", showlegend)
    add_points_trace(fig, xyz, left, row, col, COLOR_LEFT_NEIGH, ALPHA_FOCAL, point_size * 1.03, "vecino izq.", showlegend)
    add_points_trace(fig, xyz, d21, row, col, COLOR_D21, ALPHA_FOCAL, point_size * 1.05, "d21", showlegend)
    add_points_trace(fig, xyz, right, row, col, COLOR_RIGHT_NEIGH, ALPHA_FOCAL, point_size * 1.03, "vecino der.", showlegend)


def add_d21_error_panel(
    fig: Any,
    xyz: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    row: int,
    col: int,
    point_size: float,
    d21_class: int,
    left_neighbor: Optional[int],
    right_neighbor: Optional[int],
    other_teeth_color: str,
    showlegend: bool,
) -> None:
    # Base anatómica usando GT: fondo tenue + otros dientes + vecinos.
    bg = gt == 0
    left = gt == left_neighbor if left_neighbor is not None else np.zeros_like(gt, dtype=bool)
    right = gt == right_neighbor if right_neighbor is not None else np.zeros_like(gt, dtype=bool)
    d21_gt = gt == d21_class
    focal = d21_gt | left | right
    other = (gt != 0) & (~focal)

    gt_d21 = gt == d21_class
    pred_d21 = pred == d21_class
    tp = gt_d21 & pred_d21
    fp_fn = gt_d21 ^ pred_d21

    # Para que el verde/rojo mande, quitamos esos puntos de la base.
    overlay = tp | fp_fn
    add_points_trace(fig, xyz, bg & ~overlay, row, col, COLOR_BG_LIGHT, ALPHA_BG_FOCUS, point_size, "background tenue", False)
    add_points_trace(fig, xyz, other & ~overlay, row, col, other_teeth_color, ALPHA_OTHER_FOCUS, point_size, "otros dientes", showlegend)
    add_points_trace(fig, xyz, left & ~overlay, row, col, COLOR_LEFT_NEIGH, ALPHA_FOCAL, point_size * 1.03, "vecino izq.", showlegend)
    add_points_trace(fig, xyz, right & ~overlay, row, col, COLOR_RIGHT_NEIGH, ALPHA_FOCAL, point_size * 1.03, "vecino der.", showlegend)

    add_points_trace(fig, xyz, fp_fn, row, col, COLOR_ERROR, ALPHA_ERROR, point_size * 1.08, "error d21", showlegend)
    add_points_trace(fig, xyz, tp, row, col, COLOR_TP, ALPHA_TP, point_size * 1.08, "TP d21", showlegend)


# ============================================================
# FIGURA
# ============================================================

def compute_ranges(xyz: np.ndarray, vertices: Optional[np.ndarray], pad_frac: float = 0.04) -> Dict[str, List[float]]:
    arrs = [xyz]
    if vertices is not None and vertices.size > 0:
        arrs.append(vertices)
    pts = np.vstack(arrs)
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    span = mx - mn
    pad = np.maximum(span * pad_frac, 1e-3)
    return {
        "x": [float(mn[0] - pad[0]), float(mx[0] + pad[0])],
        "y": [float(mn[1] - pad[1]), float(mx[1] + pad[1])],
        "z": [float(mn[2] - pad[2]), float(mx[2] + pad[2])],
    }


def make_scene(ranges: Dict[str, List[float]], camera: Dict[str, Any], show_axes: bool = False) -> Dict[str, Any]:
    axis_cfg = dict(
        visible=show_axes,
        showgrid=False,
        zeroline=False,
        showbackground=False
    )
    return dict(
        xaxis=dict(**axis_cfg, range=ranges["x"]),
        yaxis=dict(**axis_cfg, range=ranges["y"]),
        zaxis=dict(**axis_cfg, range=ranges["z"]),
        aspectmode="data",
        bgcolor="white",
        camera=camera,
    )


def build_post_script(sync_camera: bool) -> str:
    base = """
const gd = document.getElementById('{plot_id}');
const sceneNames = ['scene', 'scene2', 'scene3', 'scene4', 'scene5', 'scene6'];

window.printCameras = function() {
  const cameras = {};
  for (const s of sceneNames) {
    if (gd._fullLayout[s]) cameras[s] = gd._fullLayout[s].camera;
  }
  console.log(JSON.stringify(cameras, null, 2));
  try { copy(JSON.stringify(cameras, null, 2)); } catch(e) {}
};

window.printCamera = function(sceneName='scene') {
  const cam = gd._fullLayout[sceneName].camera;
  console.log(JSON.stringify(cam, null, 2));
  try { copy(JSON.stringify(cam, null, 2)); } catch(e) {}
};
"""
    if not sync_camera:
        return base

    return base + """
let syncing = false;
gd.on('plotly_relayout', function(e) {
  if (syncing) return;
  let cam = null;
  for (const s of sceneNames) {
    const key = s + '.camera';
    if (e[key]) { cam = e[key]; break; }
  }
  if (cam) {
    const update = {};
    for (const s of sceneNames) update[s + '.camera'] = cam;
    syncing = true;
    Plotly.relayout(gd, update).then(() => { syncing = false; });
    console.log('[SYNC_CAMERA]');
    console.log(JSON.stringify(cam, null, 2));
  }
});
"""


def build_title(model: str, sample_id: str, case_key: str, view_key: str, metrics: Dict[str, float]) -> str:
    model_pretty = pretty_model(model)
    view_display = VIEW_DISPLAY.get(view_key, view_key)
    case_display = CASE_DISPLAY.get(case_key, case_key)
    title_line = f"{model_pretty} - ID: {sample_id} - {case_display} - {view_display}"
    metric_line = (
        f"F1 diente 21 = {metrics['d21_f1']:.3f} | "
        f"IoU diente 21 = {metrics['d21_iou']:.3f} | "
        f"F1 macro sin fondo = {metrics['f1_macro_no_bg']:.3f}"
    )
    return f"{title_line}<br><sup>{metric_line}</sup>"


def build_figure(
    xyz: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    mesh_vertices: Optional[np.ndarray],
    mesh_faces: Optional[np.ndarray],
    model: str,
    case_key: str,
    sample_id: str,
    view_key: str,
    camera: Dict[str, Any],
    d21_class: int,
    neighbors: Dict[str, int],
    width: int,
    height: int,
    point_size: float,
    bg_main_opacity: float,
    tooth_main_opacity: float,
    mesh_opacity: float,
    mesh_color: str,
    other_teeth_color: str,
    show_axes: bool,
    show_legend: bool,
) -> Any:
    if not HAS_PLOTLY:
        raise RuntimeError("Plotly no está instalado.")

    metrics = compute_metrics_for_title(gt, pred, d21_class=d21_class, bg_index=0)
    title = build_title(model, sample_id, case_key, view_key, metrics)

    subplot_titles = [
        "GT multiclase",
        f"{pretty_model(model)} predicción",
        "Errores multiclase",
        "GT d21 + vecinos",
        f"{pretty_model(model)} d21 + vecinos",
        "Error d21: verde TP / rojo FP-FN",
    ]

    fig = make_subplots(
        rows=2,
        cols=3,
        specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}],
               [{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
        horizontal_spacing=0.01,
        vertical_spacing=0.02,
        subplot_titles=subplot_titles,
    )

    left_neighbor = neighbors.get("d11", None)
    right_neighbor = neighbors.get("d22", None)

    # La malla se dibuja primero en todos los paneles para no oscurecer el beige.
    for row in [1, 2]:
        for col in [1, 2, 3]:
            add_mesh_trace(
                fig,
                mesh_vertices,
                mesh_faces,
                row=row,
                col=col,
                opacity=mesh_opacity,
                mesh_color=mesh_color,
                showlegend=False,
            )

    # Fila superior.
    add_multiclass_panel(
        fig, xyz, gt, 1, 1,
        point_size=point_size,
        bg_opacity=bg_main_opacity,
        tooth_opacity=tooth_main_opacity,
        showlegend=False,
    )
    add_multiclass_panel(
        fig, xyz, pred, 1, 2,
        point_size=point_size,
        bg_opacity=bg_main_opacity,
        tooth_opacity=tooth_main_opacity,
        showlegend=False,
    )
    add_multiclass_error_panel(
        fig, xyz, gt, pred, 1, 3,
        point_size=point_size,
        other_teeth_color=other_teeth_color,
        showlegend=False,
    )

    # Fila inferior.
    add_focus_panel(
        fig, xyz, gt, 2, 1,
        point_size=point_size,
        d21_class=d21_class,
        left_neighbor=left_neighbor,
        right_neighbor=right_neighbor,
        other_teeth_color=other_teeth_color,
        showlegend=False,
    )
    add_focus_panel(
        fig, xyz, pred, 2, 2,
        point_size=point_size,
        d21_class=d21_class,
        left_neighbor=left_neighbor,
        right_neighbor=right_neighbor,
        other_teeth_color=other_teeth_color,
        showlegend=False,
    )
    add_d21_error_panel(
        fig, xyz, gt, pred, 2, 3,
        point_size=point_size,
        d21_class=d21_class,
        left_neighbor=left_neighbor,
        right_neighbor=right_neighbor,
        other_teeth_color=other_teeth_color,
        showlegend=False,
    )

    ranges = compute_ranges(xyz, mesh_vertices)
    scene_cfg = make_scene(ranges, camera, show_axes=show_axes)
    fig.update_layout(
        scene=scene_cfg,
        scene2=scene_cfg,
        scene3=scene_cfg,
        scene4=scene_cfg,
        scene5=scene_cfg,
        scene6=scene_cfg,
        width=width,
        height=height,
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            y=0.985,
            font=dict(size=20),
        ),
        margin=dict(l=5, r=5, t=82, b=8),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=show_legend,
        font=dict(size=13, color="#2E4057"),
    )

    # Evita títulos de subplot demasiado grandes.
    fig.update_annotations(font=dict(size=13, color="#2E4057"))
    return fig


# ============================================================
# CÁMARAS PERSONALIZADAS
# ============================================================

def load_camera_from_json(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No existe camera_json: {p}")
    with p.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    # Acepta JSON directo de cámara o JSON con clave camera.
    if "camera" in obj:
        obj = obj["camera"]
    return obj


def resolve_views(args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    views: Dict[str, Dict[str, Any]] = {}
    for view in args.views:
        if view not in CAMERA_PRESETS:
            raise KeyError(f"Vista desconocida: {view}. Disponibles: {list(CAMERA_PRESETS)}")
        views[view] = CAMERA_PRESETS[view]

    if args.write_extra_view:
        cam = load_camera_from_json(args.extra_camera_json)
        if cam is None:
            cam = CAMERA_PRESETS["lateral_derecha_extra"]
        views[args.extra_view_name] = cam
        VIEW_DISPLAY[args.extra_view_name] = args.extra_view_display

    return views


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Visualizaciones cualitativas de inferencia con coloración antigua y vistas ajustadas."
    )
    ap.add_argument(
        "--base_dir",
        default="/home/htaucare/Tesis_Amaro/case_comparisons/QUALI_CASES_V17",
        help="Directorio base que contiene best/median/worst case dirs.",
    )
    ap.add_argument(
        "--out_dir",
        default=None,
        help="Directorio de salida. Si se omite, se crea dentro de base_dir.",
    )
    ap.add_argument(
        "--selected_mesh_root",
        default=str(DEFAULT_SELECTED_MESH_ROOT),
        help="Raíz alternativa donde buscar las mallas seleccionadas originales.",
    )
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    ap.add_argument("--views", nargs="+", default=DEFAULT_VIEWS)

    ap.add_argument("--d21_class", type=int, default=D21_DEFAULT)
    ap.add_argument("--neighbor_teeth", default=DEFAULT_NEIGHBORS)

    ap.add_argument("--write_html", action="store_true", help="Guardar HTML interactivo.")
    ap.add_argument("--write_png", action="store_true", help="Guardar PNG usando Kaleido si está disponible.")
    ap.add_argument("--force", action="store_true", help="Sobrescribir salidas existentes.")

    ap.add_argument("--width", type=int, default=1800)
    ap.add_argument("--height", type=int, default=1000)
    ap.add_argument("--point_size", type=float, default=4.2)
    ap.add_argument("--bg_main_opacity", type=float, default=ALPHA_BG_MAIN)
    ap.add_argument("--tooth_main_opacity", type=float, default=1.0)
    ap.add_argument("--mesh_opacity", type=float, default=0.05)
    ap.add_argument("--mesh_color", default=COLOR_MESH)
    ap.add_argument("--other_teeth_color", default=COLOR_OTHER_TEETH_DEFAULT)
    ap.add_argument(
        "--other_teeth_color_v17",
        action="store_true",
        help="Usar #E8C39E en vez de #F2D1A9 para otros dientes.",
    )
    ap.add_argument(
        "--mesh_align",
        choices=["none", "unit", "bbox", "unit_bbox"],
        default="unit",
        help="Alineación simple de malla respecto a la nube. Usa 'none' si ya viene alineada.",
    )
    ap.add_argument("--max_points_plot", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--show_axes", action="store_true")
    ap.add_argument("--show_legend", action="store_true")
    ap.add_argument("--sync_camera", action="store_true", help="Sincroniza las 6 escenas al mover el HTML.")

    ap.add_argument(
        "--write_extra_view",
        action="store_true",
        help="Además de --views, genera una vista extra. Por defecto usa lateral derecha.",
    )
    ap.add_argument(
        "--extra_camera_json",
        default=None,
        help="JSON de cámara para la vista extra. Si se omite, usa lateral_derecha_extra.",
    )
    ap.add_argument("--extra_view_name", default="lateral_derecha_extra")
    ap.add_argument("--extra_view_display", default="Vista lateral derecha")

    args = ap.parse_args()

    if not HAS_PLOTLY:
        raise RuntimeError("Falta plotly. Instala con: pip install plotly")

    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir) if args.out_dir else base_dir / "VISTA_TESIS_INFERENCIA_AJUSTADA_V5_VISUAL_OLDCOLOR"
    out_dir.mkdir(parents=True, exist_ok=True)

    other_teeth_color = COLOR_OTHER_TEETH_V17 if args.other_teeth_color_v17 else args.other_teeth_color

    models = [normalize_model_name(m) for m in args.models]
    neighbors = parse_neighbors(args.neighbor_teeth)
    neighbor_classes = list(neighbors.values())
    views = resolve_views(args)
    case_dirs = discover_case_dirs(base_dir, args.cases)

    manifest_rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []

    print("=" * 90)
    print("[INFO] Script: vista_tesis_inferencia_ajustada_v5_visual_oldcolor.py")
    print(f"[INFO] base_dir: {base_dir}")
    print(f"[INFO] out_dir:  {out_dir}")
    print(f"[INFO] models:   {models}")
    print(f"[INFO] cases:    {[p.name for p in case_dirs]}")
    print(f"[INFO] views:    {list(views.keys())}")
    print(f"[INFO] color otros dientes: {other_teeth_color}")
    print(f"[INFO] mesh_align: {args.mesh_align}")
    print("=" * 90)

    for case_dir in case_dirs:
        case_key = extract_case_key(case_dir)
        bundle = load_bundle(case_dir)
        sample_id = extract_sample_id(case_dir, bundle)
        print(f"\n[CASE] {case_dir.name} | key={case_key} | ID={sample_id}")

        mesh_path = find_mesh_for_case(case_dir, Path(args.selected_mesh_root))
        if mesh_path is None:
            print("  [WARN] No se encontró malla. Se generará solo nube de puntos.")
        else:
            print(f"  [MESH] {mesh_path}")
        raw_v, raw_f = load_mesh_vertices_faces(mesh_path)

        for model in models:
            print(f"  [MODEL] {model}")
            try:
                xyz, gt, pred = load_case_model_data(case_dir, model)
                idx = choose_plot_indices(
                    gt=gt,
                    pred=pred,
                    max_points=args.max_points_plot,
                    d21_class=args.d21_class,
                    neighbor_classes=neighbor_classes,
                    seed=args.seed,
                )
                xyz_plot = xyz[idx]
                gt_plot = gt[idx]
                pred_plot = pred[idx]

                mesh_v = align_mesh_vertices(raw_v, xyz_plot, mode=args.mesh_align) if raw_v is not None else None
                mesh_f = raw_f

                metrics = compute_metrics_for_title(gt, pred, d21_class=args.d21_class, bg_index=0)

                for view_key, camera in views.items():
                    stem = f"{model}_{case_key}_{sample_id}_{view_key}".replace(" ", "_")
                    html_path = out_dir / f"{stem}.html"
                    png_path = out_dir / f"{stem}.png"
                    summary_path = out_dir / f"{stem}.summary.json"

                    if (html_path.exists() or png_path.exists()) and not args.force:
                        print(f"    [SKIP] existe: {stem}  (usa --force para sobrescribir)")
                        continue

                    fig = build_figure(
                        xyz=xyz_plot,
                        gt=gt_plot,
                        pred=pred_plot,
                        mesh_vertices=mesh_v,
                        mesh_faces=mesh_f,
                        model=model,
                        case_key=case_key,
                        sample_id=sample_id,
                        view_key=view_key,
                        camera=camera,
                        d21_class=args.d21_class,
                        neighbors=neighbors,
                        width=args.width,
                        height=args.height,
                        point_size=args.point_size,
                        bg_main_opacity=args.bg_main_opacity,
                        tooth_main_opacity=args.tooth_main_opacity,
                        mesh_opacity=args.mesh_opacity,
                        mesh_color=args.mesh_color,
                        other_teeth_color=other_teeth_color,
                        show_axes=args.show_axes,
                        show_legend=args.show_legend,
                    )

                    wrote_html = False
                    wrote_png = False

                    if args.write_html or not args.write_png:
                        fig.write_html(
                            str(html_path),
                            include_plotlyjs="cdn",
                            post_script=build_post_script(sync_camera=args.sync_camera),
                            config={
                                "displaylogo": False,
                                "scrollZoom": True,
                                "toImageButtonOptions": {
                                    "format": "png",
                                    "filename": stem,
                                    "height": args.height,
                                    "width": args.width,
                                    "scale": 3,
                                },
                            },
                        )
                        wrote_html = True
                        print(f"    [OK] HTML: {html_path}")

                    if args.write_png:
                        try:
                            fig.write_image(str(png_path), width=args.width, height=args.height, scale=3)
                            wrote_png = True
                            print(f"    [OK] PNG:  {png_path}")
                        except Exception as e:
                            print(f"    [WARN] PNG no generado por Kaleido/Chrome: {e}")
                            print("           El HTML queda disponible; abre el HTML y usa 'Download plot as png'.")

                    summary = {
                        "case_dir": str(case_dir),
                        "case_key": case_key,
                        "sample_id": sample_id,
                        "model": model,
                        "view_key": view_key,
                        "camera": camera,
                        "mesh_path": str(mesh_path) if mesh_path else None,
                        "mesh_align": args.mesh_align,
                        "n_points_original": int(xyz.shape[0]),
                        "n_points_plot": int(xyz_plot.shape[0]),
                        "d21_class": args.d21_class,
                        "neighbors": neighbors,
                        "metrics": metrics,
                        "colors": {
                            "COLOR_BG": COLOR_BG,
                            "COLOR_BG_LIGHT": COLOR_BG_LIGHT,
                            "COLOR_MESH": args.mesh_color,
                            "COLOR_OTHER_TEETH": other_teeth_color,
                            "COLOR_D21": COLOR_D21,
                            "COLOR_LEFT_NEIGH": COLOR_LEFT_NEIGH,
                            "COLOR_RIGHT_NEIGH": COLOR_RIGHT_NEIGH,
                            "COLOR_ERROR": COLOR_ERROR,
                            "COLOR_TP": COLOR_TP,
                        },
                        "alpha": {
                            "bg_main_opacity": args.bg_main_opacity,
                            "tooth_main_opacity": args.tooth_main_opacity,
                            "mesh_opacity": args.mesh_opacity,
                            "ALPHA_BG_FOCUS": ALPHA_BG_FOCUS,
                            "ALPHA_OTHER_FOCUS": ALPHA_OTHER_FOCUS,
                            "ALPHA_BG_ERROR": ALPHA_BG_ERROR,
                            "ALPHA_OTHER_ERROR": ALPHA_OTHER_ERROR,
                        },
                        "html_path": str(html_path) if wrote_html else None,
                        "png_path": str(png_path) if wrote_png else None,
                    }
                    save_json(summary, summary_path)

                    manifest_rows.append({
                        "case_key": case_key,
                        "case_dir": str(case_dir),
                        "sample_id": sample_id,
                        "model": model,
                        "view": view_key,
                        "html": str(html_path) if wrote_html else "",
                        "png": str(png_path) if wrote_png else "",
                        "summary_json": str(summary_path),
                        "d21_f1": f"{metrics['d21_f1']:.8f}",
                        "d21_iou": f"{metrics['d21_iou']:.8f}",
                        "f1_macro_no_bg": f"{metrics['f1_macro_no_bg']:.8f}",
                    })

            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                print(f"    [ERROR] {case_dir.name}/{model}: {msg}", file=sys.stderr)
                errors.append({"case_dir": str(case_dir), "model": model, "error": msg})
                continue

    manifest_path = out_dir / "manifest_vista_tesis_inferencia_ajustada_v5_visual_oldcolor.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "case_key", "case_dir", "sample_id", "model", "view", "html", "png", "summary_json",
            "d21_f1", "d21_iou", "f1_macro_no_bg",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    run_meta = {
        "script": "vista_tesis_inferencia_ajustada_v5_visual_oldcolor.py",
        "base_dir": str(base_dir),
        "out_dir": str(out_dir),
        "models": models,
        "cases": [str(p) for p in case_dirs],
        "views": list(views.keys()),
        "camera_presets_used": views,
        "d21_class": args.d21_class,
        "neighbor_teeth": neighbors,
        "point_size": args.point_size,
        "bg_main_opacity": args.bg_main_opacity,
        "tooth_main_opacity": args.tooth_main_opacity,
        "mesh_opacity": args.mesh_opacity,
        "mesh_color": args.mesh_color,
        "other_teeth_color": other_teeth_color,
        "mesh_align": args.mesh_align,
        "max_points_plot": args.max_points_plot,
        "sync_camera": args.sync_camera,
        "write_html": args.write_html,
        "write_png": args.write_png,
        "manifest": str(manifest_path),
        "n_outputs": len(manifest_rows),
        "errors": errors,
    }
    save_json(run_meta, out_dir / "run_meta_vista_tesis_inferencia_ajustada_v5_visual_oldcolor.json")

    if errors:
        save_json(errors, out_dir / "errors_vista_tesis_inferencia_ajustada_v5_visual_oldcolor.json")

    print("\n" + "=" * 90)
    print("[DONE] Visualizaciones generadas.")
    print(f"[OK] Manifest: {manifest_path}")
    print(f"[OK] Run meta: {out_dir / 'run_meta_vista_tesis_inferencia_ajustada_v5_visual_oldcolor.json'}")
    if errors:
        print(f"[WARN] Hubo {len(errors)} errores. Revisa errors_vista_tesis_inferencia_ajustada_v5_visual_oldcolor.json")
    print("=" * 90)


if __name__ == "__main__":
    main()
