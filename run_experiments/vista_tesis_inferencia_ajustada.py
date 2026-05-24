#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
vista_tesis_inferencia_ajustada.py

Genera visualizaciones cualitativas de inferencia para tesis.

Objetivo:
- Usar los case_dirs ya construidos en QUALI_CASES_V17.
- Generar paneles 2x3 por modelo y por caso:
    1) GT multiclase
    2) Predicción multiclase
    3) Errores multiclase
    4) GT d21 + vecinos
    5) Predicción d21 + vecinos
    6) Error d21: verde=TP, rojo=FP/FN
- Aplicar cámaras fijas ajustadas manualmente:
    - isometrica_ajustada
    - oclusal_paper
    - extra_view por defecto: lateral_derecha_extra
- Guardar HTML interactivo.
- Opcionalmente guardar PNG si Kaleido/Chrome están disponibles.
- Cargar .npy de forma robusta con allow_pickle=True.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception as e:
    raise ImportError(
        "Falta plotly. Instala con: pip install plotly"
    ) from e

try:
    import trimesh
    HAS_TRIMESH = True
except Exception:
    HAS_TRIMESH = False


# ============================================================
# CÁMARAS FINALES AJUSTADAS
# ============================================================

CAMERA_PRESETS: Dict[str, Dict[str, Any]] = {
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
        "projection": {
            "type": "perspective",
        },
    },

    "oclusal_paper": {
        "up": {
            "x": 0.0,
            "y": 1.0,
            "z": 0.0,
        },
        "center": {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
        },
        "eye": {
            "x": 0.1950066638410309,
            "y": 0.195006663841031,
            "z": 1.1978980778806192,
        },
        "projection": {
            "type": "perspective",
        },
    },

    "lateral_derecha_extra": {
        "up": {
            "x": 0.0,
            "y": 0.0,
            "z": 1.0,
        },
        "center": {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
        },
        "eye": {
            "x": 1.0947373729242973,
            "y": 5.462293827280975e-17,
            "z": 0.07464118451756574,
        },
        "projection": {
            "type": "perspective",
        },
    },
}


# ============================================================
# CONFIGURACIÓN DE MODELOS
# ============================================================

DEFAULT_MODELS = [
    "pointnet",
    "pointnetpp",
    "dgcnn",
    "pointnettransformer",
]

MODEL_DISPLAY = {
    "pointnet": "PointNet",
    "pointnetpp": "PointNet++",
    "dgcnn": "DGCNN",
    "pointnettransformer": "Transformer",
}


# ============================================================
# COLORES
# ============================================================

CLASS_COLORS = {
    0: "#000000",
    1: "#2ca02c",
    2: "#98df8a",
    3: "#ff7f0e",
    4: "#ffbb78",
    5: "#1f77b4",
    6: "#aec7e8",
    7: "#17becf",
    8: "#ff8c00",
    9: "#e377c2",
    10: "#9467bd",
    11: "#c5b0d5",
    12: "#d62728",
    13: "#ff9896",
    14: "#7f7f7f",
    15: "#bcbd22",
    16: "#8c564b",
}

COLOR_BG = "#000000"
COLOR_BG_LIGHT = "rgba(120,120,120,0.22)"
COLOR_OTHER_TEETH = "#F2D1A9"
COLOR_D21 = "#FF8C00"
COLOR_NEIGHBOR_LEFT = "#0072B2"
COLOR_NEIGHBOR_RIGHT = "#E76DB6"
COLOR_ERROR = "#FF0000"
COLOR_TP = "#00B050"
COLOR_MESH = "rgba(180,180,180,0.10)"


# ============================================================
# UTILIDADES DE ARCHIVOS
# ============================================================

def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def find_first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def load_npy_robust(path: Path, preferred_keys: Optional[List[str]] = None) -> np.ndarray:
    """
    Carga .npy/.npz de forma robusta.

    Soporta:
    - npy normal
    - npy dtype=object
    - npy object scalar con dict
    - npz con claves conocidas
    """
    preferred_keys = preferred_keys or []

    if not path.exists():
        raise FileNotFoundError(f"No existe: {path}")

    if path.suffix.lower() == ".npz":
        data = np.load(path, allow_pickle=True)
        keys = list(data.keys())

        for key in preferred_keys:
            if key in data:
                return np.asarray(data[key])

        for key in ["xyz", "points", "coords", "X", "gt", "y", "Y", "pred", "labels"]:
            if key in data:
                return np.asarray(data[key])

        if len(keys) == 1:
            return np.asarray(data[keys[0]])

        raise ValueError(
            f"No pude decidir qué clave usar en {path}. "
            f"Claves disponibles: {keys}"
        )

    arr = np.load(path, allow_pickle=True)

    if isinstance(arr, np.ndarray) and arr.dtype == object:
        if arr.shape == ():
            obj = arr.item()

            if isinstance(obj, dict):
                for key in preferred_keys:
                    if key in obj:
                        return np.asarray(obj[key])

                for key in [
                    "xyz", "points", "coords", "X",
                    "gt", "y", "Y", "labels",
                    "pred", "prediction", "pred_labels",
                ]:
                    if key in obj:
                        return np.asarray(obj[key])

                raise ValueError(
                    f"{path} es un dict, pero no tiene claves esperadas. "
                    f"Claves disponibles: {list(obj.keys())}"
                )

            return np.asarray(obj)

        if arr.size == 1:
            return np.asarray(arr.reshape(-1)[0])

        try:
            arr2 = np.asarray(arr.tolist())
            if arr2.dtype != object:
                return arr2
        except Exception:
            pass

    return np.asarray(arr)


def load_case_model_data(case_dir: Path, model: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Carga:
    - xyz_labels.npy
    - gt_labels.npy
    - pred_labels.npy

    desde:
    case_dir/model/
    """
    model_dir = case_dir / model

    if not model_dir.exists():
        raise FileNotFoundError(f"No existe carpeta del modelo: {model_dir}")

    xyz_path = find_first_existing([
        model_dir / "xyz_labels.npy",
        model_dir / "xyz.npy",
        model_dir / "points.npy",
        model_dir / "coords.npy",
        model_dir / "xyz_labels.npz",
        model_dir / "xyz.npz",
        model_dir / "points.npz",
    ])

    gt_path = find_first_existing([
        model_dir / "gt_labels.npy",
        model_dir / "gt.npy",
        model_dir / "y_true.npy",
        model_dir / "labels.npy",
        model_dir / "gt_labels.npz",
        model_dir / "gt.npz",
    ])

    pred_path = find_first_existing([
        model_dir / "pred_labels.npy",
        model_dir / "pred.npy",
        model_dir / "y_pred.npy",
        model_dir / "prediction.npy",
        model_dir / "pred_labels.npz",
        model_dir / "pred.npz",
    ])

    if xyz_path is None:
        raise FileNotFoundError(f"No encontré xyz en {model_dir}")
    if gt_path is None:
        raise FileNotFoundError(f"No encontré gt_labels en {model_dir}")
    if pred_path is None:
        raise FileNotFoundError(f"No encontré pred_labels en {model_dir}")

    xyz = load_npy_robust(
        xyz_path,
        preferred_keys=["xyz", "points", "coords", "X"],
    )

    gt = load_npy_robust(
        gt_path,
        preferred_keys=["gt", "y", "Y", "labels", "gt_labels"],
    )

    pred = load_npy_robust(
        pred_path,
        preferred_keys=["pred", "prediction", "pred_labels", "y_pred"],
    )

    xyz = np.asarray(xyz)
    gt = np.asarray(gt)
    pred = np.asarray(pred)

    if xyz.ndim == 3 and xyz.shape[0] == 1:
        xyz = xyz[0]

    if gt.ndim > 1:
        gt = gt.reshape(-1)

    if pred.ndim > 1:
        pred = pred.reshape(-1)

    if xyz.ndim != 2:
        raise ValueError(
            f"xyz debe tener forma [N,3] o [N,>=3]. "
            f"Recibido {xyz.shape} en {xyz_path}"
        )

    if xyz.shape[1] < 3:
        raise ValueError(
            f"xyz debe tener al menos 3 columnas. "
            f"Recibido {xyz.shape} en {xyz_path}"
        )

    xyz = xyz[:, :3].astype(np.float32)
    gt = gt.astype(np.int32).reshape(-1)
    pred = pred.astype(np.int32).reshape(-1)

    if not (len(xyz) == len(gt) == len(pred)):
        raise ValueError(
            f"Largos incompatibles en {model_dir}: "
            f"len(xyz)={len(xyz)}, len(gt)={len(gt)}, len(pred)={len(pred)}"
        )

    return xyz, gt, pred


# ============================================================
# CARGA OPCIONAL DE MALLA
# ============================================================

def try_load_mesh_from_bundle(case_dir: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    """
    Busca una malla en:
    - common/bundle_case_data.npz
    - common/*.obj, *.ply, *.stl
    - case_dir/**/*.obj, *.ply, *.stl

    Devuelve:
    vertices, faces, source
    """
    common_dir = case_dir / "common"
    bundle = common_dir / "bundle_case_data.npz"

    if bundle.exists():
        try:
            data = np.load(bundle, allow_pickle=True)
            keys = list(data.keys())

            vertex_keys = [
                "mesh_vertices_aligned",
                "raw_vertices_aligned",
                "vertices_aligned",
                "mesh_vertices",
                "raw_vertices",
                "vertices",
                "V",
            ]
            face_keys = [
                "mesh_faces",
                "raw_faces",
                "faces",
                "F",
            ]

            v_key = next((k for k in vertex_keys if k in data), None)
            f_key = next((k for k in face_keys if k in data), None)

            if v_key is not None and f_key is not None:
                vertices = np.asarray(data[v_key], dtype=np.float32)
                faces = np.asarray(data[f_key], dtype=np.int64)

                if vertices.ndim == 2 and vertices.shape[1] >= 3 and faces.ndim == 2 and faces.shape[1] >= 3:
                    return vertices[:, :3], faces[:, :3], str(bundle)

        except Exception as e:
            print(f"[WARN] No pude cargar malla desde bundle {bundle}: {e}", file=sys.stderr)

    if HAS_TRIMESH:
        candidates = []

        if common_dir.exists():
            candidates.extend(sorted(common_dir.glob("*.obj")))
            candidates.extend(sorted(common_dir.glob("*.ply")))
            candidates.extend(sorted(common_dir.glob("*.stl")))

        candidates.extend(sorted(case_dir.glob("*.obj")))
        candidates.extend(sorted(case_dir.glob("*.ply")))
        candidates.extend(sorted(case_dir.glob("*.stl")))

        if not candidates:
            candidates.extend(sorted(case_dir.rglob("*.obj")))
            candidates.extend(sorted(case_dir.rglob("*.ply")))
            candidates.extend(sorted(case_dir.rglob("*.stl")))

        for p in candidates:
            try:
                obj = trimesh.load(p, process=False)

                if isinstance(obj, trimesh.Scene):
                    geoms = [
                        g for g in obj.geometry.values()
                        if isinstance(g, trimesh.Trimesh)
                    ]
                    if not geoms:
                        continue
                    mesh = trimesh.util.concatenate(geoms)

                elif isinstance(obj, trimesh.Trimesh):
                    mesh = obj

                else:
                    continue

                vertices = np.asarray(mesh.vertices, dtype=np.float32)
                faces = np.asarray(mesh.faces, dtype=np.int64)

                if vertices.ndim == 2 and vertices.shape[1] >= 3 and faces.ndim == 2 and faces.shape[1] >= 3:
                    return vertices[:, :3], faces[:, :3], str(p)

            except Exception:
                continue

    return None, None, None


def align_mesh_bbox_to_xyz(
    vertices: np.ndarray,
    xyz: np.ndarray,
) -> np.ndarray:
    """
    Alineación simple por bounding box.

    Esto no reemplaza ICP, pero sirve para que la malla tenue
    quede en el mismo rango visual de la nube si viene en otra escala.
    """
    v = np.asarray(vertices, dtype=np.float32)
    x = np.asarray(xyz, dtype=np.float32)

    if v.size == 0 or x.size == 0:
        return v

    v_min, v_max = np.nanmin(v, axis=0), np.nanmax(v, axis=0)
    x_min, x_max = np.nanmin(x, axis=0), np.nanmax(x, axis=0)

    v_center = 0.5 * (v_min + v_max)
    x_center = 0.5 * (x_min + x_max)

    v_span = np.maximum(v_max - v_min, 1e-8)
    x_span = np.maximum(x_max - x_min, 1e-8)

    scale = float(np.median(x_span / v_span))

    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    return (v - v_center[None, :]) * scale + x_center[None, :]


# ============================================================
# CASOS, MODELOS Y ARGUMENTOS
# ============================================================

def discover_case_dirs(base_dir: Path, models: List[str]) -> List[Path]:
    """
    Descubre carpetas de caso dentro de base_dir.
    Un case_dir válido contiene al menos una carpeta de modelo.
    """
    if not base_dir.exists():
        raise FileNotFoundError(f"No existe base_dir: {base_dir}")

    out = []

    for p in sorted(base_dir.iterdir()):
        if not p.is_dir():
            continue

        has_model = any((p / m).exists() for m in models)
        has_common = (p / "common").exists()

        if has_model or has_common:
            out.append(p)

    if not out:
        raise RuntimeError(
            f"No encontré case_dirs válidos dentro de {base_dir}. "
            f"Esperaba carpetas con modelos como {models}."
        )

    return out


def filter_case_dirs(case_dirs: List[Path], cases: List[str]) -> List[Path]:
    if not cases or cases == ["all"]:
        return case_dirs

    selected = []

    for c in case_dirs:
        name_low = c.name.lower()
        for key in cases:
            key_low = key.lower()
            if key_low in name_low:
                selected.append(c)
                break

    if not selected:
        raise RuntimeError(
            f"No se encontró ningún caso con filtro {cases}. "
            f"Disponibles: {[p.name for p in case_dirs]}"
        )

    return selected


def parse_neighbor_teeth(text: str) -> List[Tuple[str, int]]:
    """
    Formato:
    d11:1,d22:9
    """
    out = []

    text = (text or "").strip()

    if not text:
        return out

    for chunk in text.split(","):
        chunk = chunk.strip()

        if not chunk:
            continue

        if ":" not in chunk:
            raise ValueError(
                f"Formato incorrecto en neighbor_teeth: {chunk}. "
                f"Usa formato d11:1,d22:9"
            )

        name, value = chunk.split(":", 1)
        out.append((name.strip(), int(value.strip())))

    return out


def case_short_tag(case_dir: Path) -> str:
    name = case_dir.name

    if name.startswith("best"):
        return "best"

    if name.startswith("median"):
        return "median"

    if name.startswith("worst"):
        return "worst"

    if "best" in name:
        return "best"

    if "median" in name:
        return "median"

    if "mean" in name:
        return "mean"

    if "worst" in name:
        return "worst"

    return name


# ============================================================
# COLOREADO
# ============================================================

def colors_for_multiclass(labels: np.ndarray) -> List[str]:
    colors = []

    for y in labels.reshape(-1):
        y_int = int(y)
        colors.append(CLASS_COLORS.get(y_int, "#9E9E9E"))

    return colors


def colors_for_focus(
    labels: np.ndarray,
    d21_class: int,
    neighbors: List[Tuple[str, int]],
) -> List[str]:

    neighbor_ids = [v for _, v in neighbors]
    left_id = neighbor_ids[0] if len(neighbor_ids) >= 1 else None
    right_id = neighbor_ids[1] if len(neighbor_ids) >= 2 else None

    colors = []

    for y in labels.reshape(-1):
        y_int = int(y)

        if y_int == 0:
            colors.append(COLOR_BG_LIGHT)
        elif y_int == d21_class:
            colors.append(COLOR_D21)
        elif left_id is not None and y_int == left_id:
            colors.append(COLOR_NEIGHBOR_LEFT)
        elif right_id is not None and y_int == right_id:
            colors.append(COLOR_NEIGHBOR_RIGHT)
        else:
            colors.append(COLOR_OTHER_TEETH)

    return colors


def colors_for_multiclass_error(
    gt: np.ndarray,
    pred: np.ndarray,
) -> List[str]:
    colors = []

    for g, p in zip(gt.reshape(-1), pred.reshape(-1)):
        if int(g) != int(p):
            colors.append(COLOR_ERROR)
        else:
            colors.append("rgba(130,130,130,0.20)")

    return colors


def colors_for_d21_error(
    gt: np.ndarray,
    pred: np.ndarray,
    d21_class: int,
    neighbors: List[Tuple[str, int]],
) -> List[str]:

    neighbor_ids = [v for _, v in neighbors]
    left_id = neighbor_ids[0] if len(neighbor_ids) >= 1 else None
    right_id = neighbor_ids[1] if len(neighbor_ids) >= 2 else None

    colors = []

    for g, p in zip(gt.reshape(-1), pred.reshape(-1)):
        g = int(g)
        p = int(p)

        is_gt_d21 = g == d21_class
        is_pred_d21 = p == d21_class

        if is_gt_d21 and is_pred_d21:
            colors.append(COLOR_TP)
        elif is_gt_d21 != is_pred_d21:
            colors.append(COLOR_ERROR)
        elif left_id is not None and g == left_id:
            colors.append(COLOR_NEIGHBOR_LEFT)
        elif right_id is not None and g == right_id:
            colors.append(COLOR_NEIGHBOR_RIGHT)
        elif g != 0:
            colors.append(COLOR_OTHER_TEETH)
        else:
            colors.append(COLOR_BG_LIGHT)

    return colors


# ============================================================
# MÉTRICAS RÁPIDAS PARA TÍTULO
# ============================================================

def binary_f1_iou(gt: np.ndarray, pred: np.ndarray, cls: int) -> Tuple[float, float]:
    gt_pos = gt == cls
    pred_pos = pred == cls

    tp = float(np.logical_and(gt_pos, pred_pos).sum())
    fp = float(np.logical_and(~gt_pos, pred_pos).sum())
    fn = float(np.logical_and(gt_pos, ~pred_pos).sum())

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)

    return float(f1), float(iou)


def accuracy_no_bg(gt: np.ndarray, pred: np.ndarray, bg: int = 0) -> float:
    mask = gt != bg

    if mask.sum() == 0:
        return float("nan")

    return float((gt[mask] == pred[mask]).mean())


# ============================================================
# SUBMUESTREO PARA PLOT
# ============================================================

def sample_plot_points(
    xyz: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    max_points: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    n = len(xyz)

    if max_points <= 0 or n <= max_points:
        return xyz, gt, pred

    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)

    return xyz[idx], gt[idx], pred[idx]


# ============================================================
# RANGOS Y ESCENA
# ============================================================

def compute_ranges(
    xyz: np.ndarray,
    vertices: Optional[np.ndarray],
    pad_frac: float,
) -> Dict[str, List[float]]:

    arrays = [xyz]

    if vertices is not None and vertices.size > 0:
        arrays.append(vertices)

    pts = np.concatenate(arrays, axis=0)

    mins = np.nanmin(pts, axis=0)
    maxs = np.nanmax(pts, axis=0)

    span = maxs - mins
    span = np.maximum(span, 1e-6)

    pad = span * pad_frac

    return {
        "x": [float(mins[0] - pad[0]), float(maxs[0] + pad[0])],
        "y": [float(mins[1] - pad[1]), float(maxs[1] + pad[1])],
        "z": [float(mins[2] - pad[2]), float(maxs[2] + pad[2])],
    }


def make_scene(
    ranges: Dict[str, List[float]],
    camera: Dict[str, Any],
    show_axes: bool,
) -> Dict[str, Any]:

    axis_cfg = dict(
        visible=show_axes,
        showgrid=False,
        zeroline=False,
        showbackground=False,
    )

    return dict(
        xaxis=dict(**axis_cfg, range=ranges["x"]),
        yaxis=dict(**axis_cfg, range=ranges["y"]),
        zaxis=dict(**axis_cfg, range=ranges["z"]),
        aspectmode="data",
        bgcolor="white",
        camera=copy.deepcopy(camera),
    )


# ============================================================
# TRACES
# ============================================================

def mesh_trace(
    vertices: np.ndarray,
    faces: np.ndarray,
    name: str = "malla raw",
    opacity: float = 0.05,
    showlegend: bool = False,
) -> go.Mesh3d:

    v = np.asarray(vertices)
    f = np.asarray(faces)

    return go.Mesh3d(
        x=v[:, 0],
        y=v[:, 1],
        z=v[:, 2],
        i=f[:, 0],
        j=f[:, 1],
        k=f[:, 2],
        color="#B0B0B0",
        opacity=opacity,
        flatshading=False,
        name=name,
        showscale=False,
        showlegend=showlegend,
        hoverinfo="skip",
        lighting=dict(
            ambient=0.65,
            diffuse=0.65,
            fresnel=0.05,
            roughness=0.55,
            specular=0.08,
        ),
        lightposition=dict(x=100, y=200, z=300),
    )


def point_trace(
    xyz: np.ndarray,
    colors: List[str],
    name: str,
    size: float,
    opacity: float,
    showlegend: bool = False,
) -> go.Scatter3d:

    return go.Scatter3d(
        x=xyz[:, 0],
        y=xyz[:, 1],
        z=xyz[:, 2],
        mode="markers",
        marker=dict(
            size=size,
            color=colors,
            opacity=opacity,
            line=dict(width=0),
        ),
        name=name,
        showlegend=showlegend,
        hoverinfo="skip",
    )


def dummy_legend_trace(name: str, color: str) -> go.Scatter3d:
    return go.Scatter3d(
        x=[None],
        y=[None],
        z=[None],
        mode="markers",
        marker=dict(size=8, color=color),
        name=name,
        showlegend=True,
        hoverinfo="skip",
    )


# ============================================================
# FIGURA PRINCIPAL
# ============================================================

def build_panel_figure(
    xyz: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    model: str,
    case_dir: Path,
    view_name: str,
    camera: Dict[str, Any],
    d21_class: int,
    neighbors: List[Tuple[str, int]],
    vertices: Optional[np.ndarray],
    faces: Optional[np.ndarray],
    width: int,
    height: int,
    point_size: float,
    point_opacity: float,
    mesh_opacity: float,
    show_axes: bool,
    title_prefix: str,
) -> go.Figure:

    model_display = MODEL_DISPLAY.get(model, model)

    d21_f1, d21_iou = binary_f1_iou(gt, pred, d21_class)
    acc_nb = accuracy_no_bg(gt, pred, bg=0)

    subplot_titles = [
        "GT multiclase",
        f"{model_display} predicción",
        "Errores multiclase",
        "GT d21 + vecinos",
        f"{model_display} d21 + vecinos",
        "Error d21: verde TP / rojo FP-FN",
    ]

    fig = make_subplots(
        rows=2,
        cols=3,
        specs=[
            [{"type": "scene"}, {"type": "scene"}, {"type": "scene"}],
            [{"type": "scene"}, {"type": "scene"}, {"type": "scene"}],
        ],
        subplot_titles=subplot_titles,
        horizontal_spacing=0.015,
        vertical_spacing=0.03,
    )

    colors_gt = colors_for_multiclass(gt)
    colors_pred = colors_for_multiclass(pred)
    colors_err = colors_for_multiclass_error(gt, pred)
    colors_gt_focus = colors_for_focus(gt, d21_class, neighbors)
    colors_pred_focus = colors_for_focus(pred, d21_class, neighbors)
    colors_d21_err = colors_for_d21_error(gt, pred, d21_class, neighbors)

    panel_data = [
        ("GT", colors_gt),
        ("Pred", colors_pred),
        ("Error multiclase", colors_err),
        ("GT d21 + vecinos", colors_gt_focus),
        ("Pred d21 + vecinos", colors_pred_focus),
        ("Error d21", colors_d21_err),
    ]

    positions = [
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 1),
        (2, 2),
        (2, 3),
    ]

    for idx, ((trace_name, colors), (row, col)) in enumerate(zip(panel_data, positions)):
        if vertices is not None and faces is not None:
            fig.add_trace(
                mesh_trace(
                    vertices=vertices,
                    faces=faces,
                    opacity=mesh_opacity,
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

        fig.add_trace(
            point_trace(
                xyz=xyz,
                colors=colors,
                name=trace_name,
                size=point_size,
                opacity=point_opacity,
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    # Leyenda compacta
    legend_items = [
        ("background", COLOR_BG),
        ("otros dientes", COLOR_OTHER_TEETH),
        ("d21", COLOR_D21),
        ("vecino izq.", COLOR_NEIGHBOR_LEFT),
        ("vecino der.", COLOR_NEIGHBOR_RIGHT),
        ("error", COLOR_ERROR),
        ("TP d21", COLOR_TP),
        ("malla raw", "#B0B0B0"),
    ]

    for name, color in legend_items:
        fig.add_trace(dummy_legend_trace(name, color), row=1, col=1)

    ranges = compute_ranges(
        xyz=xyz,
        vertices=vertices,
        pad_frac=0.03,
    )

    scene_cfg = make_scene(
        ranges=ranges,
        camera=camera,
        show_axes=show_axes,
    )

    for scene_name in ["scene", "scene2", "scene3", "scene4", "scene5", "scene6"]:
        fig.update_layout(**{scene_name: copy.deepcopy(scene_cfg)})

    case_name = case_dir.name

    metric_text = (
        f"acc_nb={acc_nb:.3f}" if np.isfinite(acc_nb) else "acc_nb=nan"
    )

    title = (
        f"{title_prefix} — {model_display} — {case_name} — {view_name}"
        f"<br><sup>{metric_text} | d21_f1={d21_f1:.3f} | d21_iou={d21_iou:.3f}</sup>"
    )

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            font=dict(size=22),
        ),
        width=width,
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=0, r=0, t=95, b=10),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.02,
            xanchor="center",
            x=0.5,
            font=dict(size=13),
        ),
        font=dict(size=14),
    )

    fig.update_annotations(font=dict(size=16))

    return fig


def build_post_script(sync_camera: bool) -> str:
    if not sync_camera:
        return """
const gd = document.getElementById('{plot_id}');

window.printCameras = function() {
  const scenes = ["scene", "scene2", "scene3", "scene4", "scene5", "scene6"];
  const cameras = {};
  for (const s of scenes) {
    cameras[s] = gd._fullLayout[s]?.camera;
  }
  console.log(JSON.stringify(cameras, null, 2));
};

window.printMasterCamera = function() {
  console.log(JSON.stringify(gd._fullLayout.scene.camera, null, 2));
};
"""

    return """
const gd = document.getElementById('{plot_id}');
const scenes = ["scene", "scene2", "scene3", "scene4", "scene5", "scene6"];
let syncing = false;

gd.on('plotly_relayout', function(e) {
  if (syncing) return;

  let cam = null;
  for (const s of scenes) {
    const key = `${s}.camera`;
    if (e[key]) {
      cam = e[key];
      break;
    }
  }

  if (cam) {
    const update = {};
    for (const s of scenes) {
      update[`${s}.camera`] = cam;
    }

    syncing = true;
    Plotly.relayout(gd, update).then(() => {
      syncing = false;
    });

    console.log("[SYNC_CAMERA]");
    console.log(JSON.stringify(cam, null, 2));
  }
});

window.printCameras = function() {
  const cameras = {};
  for (const s of scenes) {
    cameras[s] = gd._fullLayout[s]?.camera;
  }
  console.log(JSON.stringify(cameras, null, 2));
};

window.printMasterCamera = function() {
  console.log(JSON.stringify(gd._fullLayout.scene.camera, null, 2));
};
"""


# ============================================================
# CÁMARA EXTRA PERSONALIZADA
# ============================================================

def load_extra_camera(args) -> Dict[str, Any]:
    if args.extra_camera_json:
        p = Path(args.extra_camera_json)

        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                return json.load(f)

        try:
            return json.loads(args.extra_camera_json)
        except Exception as e:
            raise ValueError(
                "--extra_camera_json debe ser una ruta existente o un JSON válido."
            ) from e

    return copy.deepcopy(CAMERA_PRESETS["lateral_derecha_extra"])


def build_views_to_run(args) -> Dict[str, Dict[str, Any]]:
    views = {}

    if args.views:
        for view_name in args.views:
            if view_name not in CAMERA_PRESETS:
                raise KeyError(
                    f"Vista '{view_name}' no existe. "
                    f"Disponibles: {list(CAMERA_PRESETS.keys())}"
                )
            views[view_name] = copy.deepcopy(CAMERA_PRESETS[view_name])
    else:
        views["isometrica_ajustada"] = copy.deepcopy(CAMERA_PRESETS["isometrica_ajustada"])
        views["oclusal_paper"] = copy.deepcopy(CAMERA_PRESETS["oclusal_paper"])

    if not args.disable_extra_view:
        views[args.extra_view_name] = load_extra_camera(args)

    return views


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--base_dir",
        default="/home/htaucare/Tesis_Amaro/case_comparisons/QUALI_CASES_V17",
        help="Carpeta base con los case_dirs: best, median, worst.",
    )

    ap.add_argument(
        "--out_dir",
        default=None,
        help=(
            "Carpeta de salida. Si no se entrega, se usa "
            "<base_dir>/VISTA_TESIS_INFERENCIA_AJUSTADA."
        ),
    )

    ap.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Modelos a graficar.",
    )

    ap.add_argument(
        "--cases",
        nargs="+",
        default=["all"],
        help='Casos a usar. Ej: "best median worst" o "all".',
    )

    ap.add_argument(
        "--views",
        nargs="+",
        default=None,
        help=(
            "Vistas base. Por defecto usa isometrica_ajustada y oclusal_paper. "
            f"Disponibles: {list(CAMERA_PRESETS.keys())}"
        ),
    )

    ap.add_argument(
        "--disable_extra_view",
        action="store_true",
        help="Desactiva la vista extra.",
    )

    ap.add_argument(
        "--extra_view_name",
        default="lateral_derecha_extra",
        help="Nombre de la vista extra.",
    )

    ap.add_argument(
        "--extra_camera_json",
        default=None,
        help=(
            "Ruta a JSON de cámara extra o JSON en texto. "
            "Si no se entrega, usa lateral_derecha_extra."
        ),
    )

    ap.add_argument("--d21_class", type=int, default=8)
    ap.add_argument("--neighbor_teeth", default="d11:1,d22:9")

    ap.add_argument("--width", type=int, default=2200)
    ap.add_argument("--height", type=int, default=1350)

    ap.add_argument("--point_size", type=float, default=4.2)
    ap.add_argument("--point_opacity", type=float, default=0.92)

    ap.add_argument("--mesh_opacity", type=float, default=0.05)
    ap.add_argument("--no_mesh", action="store_true")
    ap.add_argument("--no_mesh_bbox_align", action="store_true")

    ap.add_argument(
        "--max_points_plot",
        type=int,
        default=12000,
        help="Máximo de puntos a graficar. <=0 usa todos.",
    )

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--show_axes", action="store_true")

    ap.add_argument(
        "--write_html",
        action="store_true",
        help="Guardar HTML interactivo.",
    )

    ap.add_argument(
        "--write_png",
        action="store_true",
        help="Guardar PNG con Kaleido si está disponible.",
    )

    ap.add_argument(
        "--png_scale",
        type=float,
        default=2.0,
        help="Escala para exportar PNG con Kaleido.",
    )

    ap.add_argument(
        "--sync_camera",
        action="store_true",
        help="En HTML, sincroniza las seis escenas si se mueve una cámara.",
    )

    ap.add_argument(
        "--force",
        action="store_true",
        help="Sobrescribe outputs existentes.",
    )

    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    out_root = Path(args.out_dir) if args.out_dir else base_dir / "VISTA_TESIS_INFERENCIA_AJUSTADA"
    out_root.mkdir(parents=True, exist_ok=True)

    # Si el usuario no pasa ningún formato, guardamos HTML por defecto.
    if not args.write_html and not args.write_png:
        args.write_html = True

    models = [m.lower().strip() for m in args.models]
    neighbors = parse_neighbor_teeth(args.neighbor_teeth)
    views = build_views_to_run(args)

    case_dirs_all = discover_case_dirs(base_dir, models)
    case_dirs = filter_case_dirs(case_dirs_all, args.cases)

    manifest_rows = []

    print("=" * 90)
    print("[INFO] vista_tesis_inferencia_ajustada.py")
    print(f"[INFO] base_dir: {base_dir}")
    print(f"[INFO] out_root: {out_root}")
    print(f"[INFO] cases: {[p.name for p in case_dirs]}")
    print(f"[INFO] models: {models}")
    print(f"[INFO] views: {list(views.keys())}")
    print(f"[INFO] write_html: {args.write_html}")
    print(f"[INFO] write_png: {args.write_png}")
    print("=" * 90)

    for case_dir in case_dirs:
        case_tag = case_short_tag(case_dir)

        print(f"\n[CASE] {case_dir.name}")

        mesh_vertices = None
        mesh_faces = None
        mesh_source = None

        if not args.no_mesh:
            mesh_vertices, mesh_faces, mesh_source = try_load_mesh_from_bundle(case_dir)

            if mesh_vertices is not None and mesh_faces is not None:
                print(f"   [MESH] {mesh_source}")
            else:
                print("   [MESH] No se encontró malla. Se grafican solo puntos.")

        for model in models:
            model_dir = case_dir / model

            if not model_dir.exists():
                print(f"   [SKIP] {model}: no existe {model_dir}")
                continue

            print(f"   [MODEL] {model}")

            try:
                xyz, gt, pred = load_case_model_data(case_dir, model)
            except Exception as e:
                print(f"      [ERROR] No pude cargar datos de {model}: {e}", file=sys.stderr)
                continue

            xyz_plot, gt_plot, pred_plot = sample_plot_points(
                xyz=xyz,
                gt=gt,
                pred=pred,
                max_points=args.max_points_plot,
                seed=args.seed,
            )

            vertices_plot = None
            faces_plot = None

            if mesh_vertices is not None and mesh_faces is not None:
                vertices_plot = mesh_vertices
                faces_plot = mesh_faces

                if not args.no_mesh_bbox_align:
                    vertices_plot = align_mesh_bbox_to_xyz(
                        vertices=vertices_plot,
                        xyz=xyz_plot,
                    )

            d21_f1, d21_iou = binary_f1_iou(gt, pred, args.d21_class)
            acc_nb = accuracy_no_bg(gt, pred, bg=0)

            for view_name, camera in views.items():
                out_dir = out_root / case_dir.name / model
                out_dir.mkdir(parents=True, exist_ok=True)

                stem = f"vista_tesis_{case_tag}_{model}_{view_name}"

                html_path = out_dir / f"{stem}.html"
                png_path = out_dir / f"{stem}.png"
                summary_path = out_dir / f"{stem}.summary.json"

                if html_path.exists() and not args.force and args.write_html:
                    print(f"      [SKIP HTML existe] {html_path}")
                    html_done = True
                else:
                    html_done = False

                if png_path.exists() and not args.force and args.write_png:
                    print(f"      [SKIP PNG existe] {png_path}")
                    png_done = True
                else:
                    png_done = False

                fig = None

                if (args.write_html and not html_done) or (args.write_png and not png_done):
                    fig = build_panel_figure(
                        xyz=xyz_plot,
                        gt=gt_plot,
                        pred=pred_plot,
                        model=model,
                        case_dir=case_dir,
                        view_name=view_name,
                        camera=camera,
                        d21_class=args.d21_class,
                        neighbors=neighbors,
                        vertices=vertices_plot,
                        faces=faces_plot,
                        width=args.width,
                        height=args.height,
                        point_size=args.point_size,
                        point_opacity=args.point_opacity,
                        mesh_opacity=args.mesh_opacity,
                        show_axes=args.show_axes,
                        title_prefix="Vista interactiva filtrable",
                    )

                if args.write_html and not html_done and fig is not None:
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
                                "scale": args.png_scale,
                            },
                        },
                    )
                    print(f"      [OK HTML] {html_path}")

                if args.write_png and not png_done and fig is not None:
                    try:
                        fig.write_image(
                            str(png_path),
                            width=args.width,
                            height=args.height,
                            scale=args.png_scale,
                        )
                        print(f"      [OK PNG]  {png_path}")
                    except Exception as e:
                        print(
                            f"      [WARN PNG] No se pudo exportar PNG con Kaleido/Chrome: {e}",
                            file=sys.stderr,
                        )
                        print(
                            "      [INFO] El HTML igual queda disponible. "
                            "Puedes abrirlo y usar Download plot as png.",
                            file=sys.stderr,
                        )

                summary = {
                    "case_dir": str(case_dir),
                    "case_name": case_dir.name,
                    "case_tag": case_tag,
                    "model": model,
                    "model_display": MODEL_DISPLAY.get(model, model),
                    "view_name": view_name,
                    "camera": camera,
                    "n_points_original": int(len(xyz)),
                    "n_points_plot": int(len(xyz_plot)),
                    "d21_class": int(args.d21_class),
                    "neighbor_teeth": neighbors,
                    "metrics": {
                        "acc_no_bg": None if not np.isfinite(acc_nb) else float(acc_nb),
                        "d21_f1": float(d21_f1),
                        "d21_iou": float(d21_iou),
                    },
                    "mesh_source": mesh_source,
                    "outputs": {
                        "html": str(html_path) if args.write_html else None,
                        "png": str(png_path) if args.write_png else None,
                    },
                }

                save_json(summary, summary_path)

                manifest_rows.append({
                    "case_name": case_dir.name,
                    "case_tag": case_tag,
                    "model": model,
                    "view_name": view_name,
                    "html": str(html_path) if args.write_html else "",
                    "png": str(png_path) if args.write_png else "",
                    "summary": str(summary_path),
                    "acc_no_bg": "" if not np.isfinite(acc_nb) else f"{acc_nb:.6f}",
                    "d21_f1": f"{d21_f1:.6f}",
                    "d21_iou": f"{d21_iou:.6f}",
                    "mesh_source": mesh_source or "",
                })

    manifest_path = out_root / "manifest_vista_tesis_inferencia_ajustada.csv"

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "case_name",
            "case_tag",
            "model",
            "view_name",
            "html",
            "png",
            "summary",
            "acc_no_bg",
            "d21_f1",
            "d21_iou",
            "mesh_source",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow(row)

    run_meta = {
        "script": "vista_tesis_inferencia_ajustada.py",
        "base_dir": str(base_dir),
        "out_root": str(out_root),
        "models": models,
        "cases": [p.name for p in case_dirs],
        "views": views,
        "d21_class": args.d21_class,
        "neighbor_teeth": neighbors,
        "write_html": args.write_html,
        "write_png": args.write_png,
        "width": args.width,
        "height": args.height,
        "point_size": args.point_size,
        "point_opacity": args.point_opacity,
        "mesh_opacity": args.mesh_opacity,
        "max_points_plot": args.max_points_plot,
        "sync_camera": args.sync_camera,
        "manifest": str(manifest_path),
    }

    save_json(run_meta, out_root / "run_meta_vista_tesis_inferencia_ajustada.json")

    print("\n" + "=" * 90)
    print("[DONE] Visualizaciones generadas.")
    print(f"[OK] Manifest: {manifest_path}")
    print(f"[OK] Run meta: {out_root / 'run_meta_vista_tesis_inferencia_ajustada.json'}")
    print("=" * 90)


if __name__ == "__main__":
    main()