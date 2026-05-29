#!/usr/bin/env python3
# -*- coding: utf-8 -*-


vi
vista_tesis_inferencia_ajustada_v4_visual_v17.py

Visualizaciones cualitativas 2x3 para tesis dental 3D.

V4 mantiene estructura/salidas de V2 y corrige SOLO la capa visual:
- Carga robusta de .npy/.npz con allow_pickle=True.
- Búsqueda automática de malla en:
    1) meta.json
    2) common/raw_mesh.obj
    3) common/*.obj/*.ply/*.stl
    4) selected_mesh_root buscando por ID de paciente
    5) case_dir recursivo
- Título formal:
    Modelo - ID: XXXXX - Mejor/Mediano/Peor caso - Vista <nombre>
- Subtítulo con:
    F1 diente 21 | IoU diente 21 | F1 macro sin fondo
- Manifest CSV + summary JSON por figura.
- Cámaras ajustadas manualmente:
    oclusal_paper
    isometrica_ajustada
    lateral_derecha_extra opcional
- Capa visual estilo v17/paper:
    malla como fondo anatómico tenue; puntos importantes como capas opacas;
    beige de otros dientes no se mezcla con la malla ni con el contexto.

Estructura esperada de entrada:
QUALI_CASES_V17/
  best_row091_01MAVT6A_upper/
    common/bundle_case_data.npz   (opcional)
    common/raw_mesh.obj           (opcional)
    pointnet/xyz_labels.npy gt_labels.npy pred_labels.npy
    pointnetpp/...
    dgcnn/...
    pointnettransformer/...
  median_row089_015RHV4X_upper/
  worst_row071_3EU06ZN9_upper/
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception as e:
    raise ImportError("Falta plotly. Instala con: pip install plotly") from e

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
# MODELOS Y COLORES
# ============================================================

MODEL_DIR_ALIASES = {
    "pointnet": "pointnet",
    "pointnetpp": "pointnetpp",
    "pointnet++": "pointnetpp",
    "dgcnn": "dgcnn",
    "pointnettransformer": "pointnettransformer",
    "transformer": "pointnettransformer",
}

MODEL_DISPLAY = {
    "pointnet": "PointNet",
    "pointnetpp": "PointNet++",
    "dgcnn": "DGCNN",
    "pointnettransformer": "Transformer",
}

DEFAULT_MODELS = ["pointnet", "pointnetpp", "dgcnn", "pointnettransformer"]

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

# Colores de la visualización v17/paper.
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
COLOR_BG_LIGHT = "rgba(120,120,120,0.20)"
COLOR_OTHER_TEETH = "#E8C39E"
COLOR_D21 = "#FF8C00"
COLOR_NEIGHBOR_LEFT = "#0070C0"
COLOR_NEIGHBOR_RIGHT = "#FF69B4"
COLOR_ERROR = "#FF0000"
COLOR_TP = "#00B050"
COLOR_MESH_HEX = "#9E9E9E"


# ============================================================
# UTILIDADES GENERALES
# ============================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def sanitize_model_name(name: str) -> str:
    key = str(name).strip().lower()
    return MODEL_DIR_ALIASES.get(key, key)


def model_display_name(model: str) -> str:
    return MODEL_DISPLAY.get(sanitize_model_name(model), str(model))


def find_first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        p = Path(p)
        if p.exists():
            return p
    return None


def scalar_to_str(x: Any, default: str = "") -> str:
    if x is None:
        return default
    try:
        arr = np.asarray(x)
        if arr.shape == ():
            return str(arr.item())
        if arr.size >= 1:
            return str(arr.reshape(-1)[0])
    except Exception:
        pass
    try:
        return str(x)
    except Exception:
        return default


def clean_patient_id(s: str) -> str:
    s = str(s).strip()
    s = s.replace("_upper", "").replace("_lower", "")
    return s


def load_npz_dict(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        return {}
    data = np.load(path, allow_pickle=True)
    out = {}
    for k in data.keys():
        val = data[k]
        if isinstance(val, np.ndarray) and val.dtype == object:
            try:
                if val.shape == ():
                    val = val.item()
                elif val.size == 1:
                    val = val.reshape(-1)[0]
            except Exception:
                pass
        out[k] = np.asarray(val)
    return out


def load_common_bundle(case_dir: Path) -> Dict[str, np.ndarray]:
    return load_npz_dict(Path(case_dir) / "common" / "bundle_case_data.npz")


def infer_patient_id(case_dir: Path, common: Optional[Dict[str, np.ndarray]] = None) -> str:
    common = common or {}

    for k in ["sample_name", "patient_id", "case_id", "id"]:
        if k in common:
            sid = clean_patient_id(scalar_to_str(common[k], ""))
            if sid:
                return sid

    name = Path(case_dir).name

    # Patrones típicos: best_row091_01MAVT6A_upper, median_row089_015RHV4X_upper.
    m = re.search(r"row\d+_([A-Za-z0-9]+)", name)
    if m:
        return clean_patient_id(m.group(1))

    parts = name.split("_")
    for p in parts:
        if len(p) >= 6 and any(ch.isalpha() for ch in p) and any(ch.isdigit() for ch in p):
            return clean_patient_id(p)

    return name


def case_short_tag(case_dir: Path) -> str:
    name = Path(case_dir).name.lower()
    if "best" in name or "mejor" in name:
        return "best"
    if "median" in name or "mediana" in name or "mediano" in name:
        return "median"
    if "mean" in name or "promedio" in name:
        return "mean"
    if "worst" in name or "peor" in name:
        return "worst"
    return Path(case_dir).name


def case_role_label(case_dir: Path, override: str = "") -> str:
    key = (override or case_short_tag(case_dir)).strip().lower()
    if key in {"best", "mejor", "mejor_caso"}:
        return "Mejor caso"
    if key in {"median", "mediana", "mediano", "caso_mediano"}:
        return "Caso mediano"
    if key in {"mean", "promedio", "caso_promedio"}:
        return "Caso promedio"
    if key in {"worst", "peor", "peor_caso"}:
        return "Peor caso"
    return "Caso representativo"


def pretty_view_name(view_name: str) -> str:
    v = str(view_name).strip().lower().replace("-", "_")
    mapping = {
        "oclusal_paper": "oclusal superior",
        "oclusal_superior": "oclusal superior",
        "oclusal_oblicua": "oclusal oblicua",
        "isometrica_ajustada": "cuasi-isométrica",
        "isometrica": "isométrica",
        "lateral_derecha_extra": "lateral derecha",
        "lateral_derecha": "lateral derecha",
        "frontal": "frontal",
    }
    return mapping.get(v, v.replace("_", " "))


def build_main_title(model: str, patient_id: str, case_role: str, view_name: str) -> str:
    return (
        f"{model_display_name(model)} - "
        f"ID: {patient_id} - "
        f"{case_role} - "
        f"Vista {pretty_view_name(view_name)}"
    )


# ============================================================
# CARGA ROBUSTA DE ARRAYS
# ============================================================

def unwrap_object_array(arr: Any) -> np.ndarray:
    if not isinstance(arr, np.ndarray):
        return np.asarray(arr)
    if arr.dtype != object:
        return arr
    if arr.shape == ():
        try:
            return np.asarray(arr.item())
        except Exception:
            return arr
    if arr.size == 1:
        try:
            return np.asarray(arr.reshape(-1)[0])
        except Exception:
            return arr
    try:
        arr2 = np.asarray(arr.tolist())
        return arr2
    except Exception:
        return arr


def load_array_robust(path: Path, preferred_keys: Sequence[str]) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe: {path}")

    obj = np.load(path, allow_pickle=True)

    if isinstance(obj, np.lib.npyio.NpzFile):
        keys = list(obj.keys())
        for k in preferred_keys:
            if k in keys:
                return unwrap_object_array(obj[k])
        if len(keys) == 1:
            return unwrap_object_array(obj[keys[0]])
        raise ValueError(f"No pude elegir clave en {path}. Claves: {keys}")

    return unwrap_object_array(obj)


def finite_xyz(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    x = x.squeeze()
    if x.ndim != 2 or x.shape[1] < 3:
        raise ValueError(f"xyz debe tener forma [N,3] o [N,>=3]. Recibido: {x.shape}")
    x = x[:, :3].astype(np.float32)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def flatten_labels(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    y = y.squeeze()
    if y.ndim != 1:
        y = y.reshape(-1)
    return y.astype(np.int64)


def fix_lengths(xyz: np.ndarray, gt: np.ndarray, pred: np.ndarray, strict: bool = False):
    n = min(len(xyz), len(gt), len(pred))
    m = max(len(xyz), len(gt), len(pred))
    if n != m:
        msg = f"Longitudes distintas: xyz={len(xyz)}, gt={len(gt)}, pred={len(pred)}. Recortando a {n}."
        if strict:
            raise ValueError(msg)
        print(f"[WARN] {msg}", file=sys.stderr)
    return xyz[:n], gt[:n], pred[:n]


def load_case_model_data(case_dir: Path, model: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, str]]:
    case_dir = Path(case_dir)
    model = sanitize_model_name(model)
    model_dir = case_dir / model
    common = load_common_bundle(case_dir)

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

    if pred_path is None:
        raise FileNotFoundError(f"No encontré pred_labels en {model_dir}")

    if xyz_path is not None:
        xyz = finite_xyz(load_array_robust(xyz_path, ["xyz", "points", "coords", "X", "arr_0"]))
    elif "xyz_8192" in common:
        xyz = finite_xyz(common["xyz_8192"])
        xyz_path = case_dir / "common" / "bundle_case_data.npz::xyz_8192"
    elif "xyz" in common:
        xyz = finite_xyz(common["xyz"])
        xyz_path = case_dir / "common" / "bundle_case_data.npz::xyz"
    else:
        raise FileNotFoundError(f"No encontré xyz en {model_dir} ni en common/bundle_case_data.npz")

    if gt_path is not None:
        gt = flatten_labels(load_array_robust(gt_path, ["gt", "gt_labels", "y", "Y", "labels", "arr_0"]))
    elif "y_8192" in common:
        gt = flatten_labels(common["y_8192"])
        gt_path = case_dir / "common" / "bundle_case_data.npz::y_8192"
    elif "labels" in common:
        gt = flatten_labels(common["labels"])
        gt_path = case_dir / "common" / "bundle_case_data.npz::labels"
    else:
        raise FileNotFoundError(f"No encontré gt_labels en {model_dir} ni en common/bundle_case_data.npz")

    pred = flatten_labels(load_array_robust(pred_path, ["pred", "prediction", "pred_labels", "y_pred", "arr_0"]))

    xyz, gt, pred = fix_lengths(xyz, gt, pred, strict=False)

    sources = {
        "xyz": str(xyz_path),
        "gt": str(gt_path),
        "pred": str(pred_path),
    }
    return xyz, gt, pred, sources


# ============================================================
# CARGA DE MALLA
# ============================================================

def load_mesh_vertices_faces(path: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if path is None or not Path(path).exists() or not HAS_TRIMESH:
        return None, None
    try:
        obj = trimesh.load(path, process=False)
        if isinstance(obj, trimesh.Scene):
            geoms = [g for g in obj.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if not geoms:
                return None, None
            mesh = trimesh.util.concatenate(geoms)
        elif isinstance(obj, trimesh.Trimesh):
            mesh = obj
        else:
            return None, None
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] < 3 or faces.ndim != 2 or faces.shape[1] < 3:
            return None, None
        return vertices[:, :3], faces[:, :3]
    except Exception as e:
        print(f"[WARN] No pude cargar malla {path}: {e}", file=sys.stderr)
        return None, None


def candidate_mesh_paths(case_dir: Path,
                         selected_mesh_root: Optional[Path],
                         patient_id: str,
                         case_tag: str) -> List[Path]:
    case_dir = Path(case_dir)
    out: List[Path] = []

    meta = read_json(case_dir / "meta.json")
    for key in ["raw_mesh_path", "mesh_path", "raw_path"]:
        val = meta.get(key, "")
        if val:
            out.append(Path(val))

    out.extend([
        case_dir / "common" / "raw_mesh.obj",
        case_dir / "common" / "raw_mesh.ply",
        case_dir / "common" / "raw_mesh.stl",
        case_dir / "raw_mesh.obj",
        case_dir / "raw_mesh.ply",
        case_dir / "raw_mesh.stl",
    ])

    common_dir = case_dir / "common"
    if common_dir.exists():
        for ext in ["*.obj", "*.ply", "*.stl"]:
            out.extend(sorted(common_dir.glob(ext)))

    if selected_mesh_root and Path(selected_mesh_root).exists() and patient_id:
        root = Path(selected_mesh_root)
        matches = []
        for ext in ["*.obj", "*.ply", "*.stl"]:
            matches.extend(sorted(root.rglob(f"*{patient_id}*{ext[1:]}")))

        # Priorización por tipo de caso.
        def score(p: Path) -> int:
            s = str(p).lower()
            tag = case_tag.lower()
            score_val = 0
            if tag == "best" and "/best/" in s:
                score_val -= 20
            if tag == "median" and ("closest_to_median" in s or "median" in s):
                score_val -= 20
            if tag == "mean" and ("closest_to_mean" in s or "mean" in s):
                score_val -= 20
            if tag == "worst" and "/worst/" in s:
                score_val -= 20
            score_val += len(s)
            return score_val

        out.extend(sorted(matches, key=score))

    for ext in ["*.obj", "*.ply", "*.stl"]:
        out.extend(sorted(case_dir.glob(ext)))

    # Último recurso: recursivo dentro del caso, pero evitando carpetas de outputs.
    for ext in ["*.obj", "*.ply", "*.stl"]:
        for p in sorted(case_dir.rglob(ext)):
            low = str(p).lower()
            if "vista_tesis" in low or "v17_individual" in low:
                continue
            out.append(p)

    # Únicos preservando orden.
    seen = set()
    uniq = []
    for p in out:
        p = Path(p)
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq


def try_load_mesh(case_dir: Path,
                  selected_mesh_root: Optional[Path],
                  patient_id: str,
                  case_tag: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    if not HAS_TRIMESH:
        return None, None, ""

    # Primero: bundle con vertices/faces si existiera.
    bundle = case_dir / "common" / "bundle_case_data.npz"
    if bundle.exists():
        try:
            data = np.load(bundle, allow_pickle=True)
            vertex_keys = [
                "mesh_vertices_aligned", "raw_vertices_aligned", "vertices_aligned",
                "mesh_vertices", "raw_vertices", "vertices", "V",
            ]
            face_keys = ["mesh_faces", "raw_faces", "faces", "F"]
            vk = next((k for k in vertex_keys if k in data), None)
            fk = next((k for k in face_keys if k in data), None)
            if vk and fk:
                v = np.asarray(data[vk], dtype=np.float32)
                f = np.asarray(data[fk], dtype=np.int64)
                if v.ndim == 2 and v.shape[1] >= 3 and f.ndim == 2 and f.shape[1] >= 3:
                    return v[:, :3], f[:, :3], f"{bundle}::{vk},{fk}"
        except Exception as e:
            print(f"[WARN] No pude leer malla desde bundle {bundle}: {e}", file=sys.stderr)

    for p in candidate_mesh_paths(case_dir, selected_mesh_root, patient_id, case_tag):
        if not p.exists():
            continue
        v, f = load_mesh_vertices_faces(p)
        if v is not None and f is not None:
            return v, f, str(p.resolve())

    return None, None, ""


def align_mesh_bbox_to_xyz(vertices: np.ndarray, xyz: np.ndarray) -> np.ndarray:
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
# MÉTRICAS
# ============================================================

def binary_f1_iou(gt: np.ndarray, pred: np.ndarray, cls: int) -> Tuple[float, float, float, float]:
    gt = flatten_labels(gt)
    pred = flatten_labels(pred)
    n = min(len(gt), len(pred))
    gt = gt[:n]
    pred = pred[:n]

    gt_pos = gt == int(cls)
    pr_pos = pred == int(cls)

    tp = float(np.logical_and(gt_pos, pr_pos).sum())
    fp = float(np.logical_and(~gt_pos, pr_pos).sum())
    fn = float(np.logical_and(gt_pos, ~pr_pos).sum())

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-9)
    iou = tp / (tp + fp + fn + 1e-9)
    return float(f1), float(iou), float(precision), float(recall)


def f1_macro_no_bg(gt: np.ndarray, pred: np.ndarray, bg: int = 0) -> float:
    gt = flatten_labels(gt)
    pred = flatten_labels(pred)
    n = min(len(gt), len(pred))
    gt = gt[:n]
    pred = pred[:n]

    classes = sorted(set(gt.tolist()) | set(pred.tolist()))
    classes = [c for c in classes if int(c) != int(bg)]

    vals = []
    for cls in classes:
        gt_c = gt == int(cls)
        pr_c = pred == int(cls)
        tp = float(np.logical_and(gt_c, pr_c).sum())
        fp = float(np.logical_and(~gt_c, pr_c).sum())
        fn = float(np.logical_and(gt_c, ~pr_c).sum())
        if tp + fp + fn <= 0:
            continue
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        vals.append(float(f1))
    return float(np.mean(vals)) if vals else 0.0


def accuracy_no_bg(gt: np.ndarray, pred: np.ndarray, bg: int = 0) -> float:
    gt = flatten_labels(gt)
    pred = flatten_labels(pred)
    n = min(len(gt), len(pred))
    gt = gt[:n]
    pred = pred[:n]
    mask = gt != int(bg)
    if not np.any(mask):
        return float("nan")
    return float((gt[mask] == pred[mask]).mean())


def compute_metrics(gt: np.ndarray, pred: np.ndarray, d21_class: int) -> Dict[str, float]:
    d21_f1, d21_iou, d21_precision, d21_recall = binary_f1_iou(gt, pred, d21_class)
    return {
        "acc_no_bg": accuracy_no_bg(gt, pred, bg=0),
        "d21_f1": d21_f1,
        "d21_iou": d21_iou,
        "d21_precision": d21_precision,
        "d21_recall": d21_recall,
        "f1_macro_no_bg": f1_macro_no_bg(gt, pred, bg=0),
    }


def metric_subtitle(metrics: Dict[str, float]) -> str:
    return (
        f"F1 diente 21 = {metrics.get('d21_f1', 0.0):.3f} | "
        f"IoU diente 21 = {metrics.get('d21_iou', 0.0):.3f} | "
        f"F1 macro sin fondo = {metrics.get('f1_macro_no_bg', 0.0):.3f}"
    )


# ============================================================
# COLOREADO
# ============================================================

def hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.strip().lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.3f})"


def colors_for_multiclass(labels: np.ndarray,
                          bg_alpha: float = 0.56,
                          bg_mode: str = "black") -> List[str]:
    labels = flatten_labels(labels)
    out = []
    for y in labels:
        y_int = int(y)
        if y_int == 0:
            if bg_mode == "soft":
                out.append(f"rgba(40,40,40,{bg_alpha:.3f})")
            else:
                out.append(hex_to_rgba(CLASS_COLORS[0], bg_alpha))
        else:
            out.append(CLASS_COLORS.get(y_int, "#9E9E9E"))
    return out


def colors_for_focus(labels: np.ndarray,
                     d21_class: int,
                     neighbors: List[Tuple[str, int]]) -> List[str]:
    labels = flatten_labels(labels)
    neigh_ids = [int(v) for _, v in neighbors]
    left_id = neigh_ids[0] if len(neigh_ids) >= 1 else None
    right_id = neigh_ids[1] if len(neigh_ids) >= 2 else None
    out = []
    for y in labels:
        y = int(y)
        if y == 0:
            out.append(COLOR_BG_LIGHT)
        elif y == int(d21_class):
            out.append(COLOR_D21)
        elif left_id is not None and y == left_id:
            out.append(COLOR_NEIGHBOR_LEFT)
        elif right_id is not None and y == right_id:
            out.append(COLOR_NEIGHBOR_RIGHT)
        else:
            out.append(COLOR_OTHER_TEETH)
    return out


def colors_for_multiclass_error(gt: np.ndarray, pred: np.ndarray) -> List[str]:
    gt = flatten_labels(gt)
    pred = flatten_labels(pred)
    n = min(len(gt), len(pred))
    out = []
    for g, p in zip(gt[:n], pred[:n]):
        if int(g) != int(p):
            out.append(COLOR_ERROR)
        else:
            if int(g) == 0:
                out.append("rgba(80,80,80,0.09)")
            else:
                out.append(hex_to_rgba(COLOR_OTHER_TEETH, 0.88))
    return out


def colors_for_d21_error(gt: np.ndarray,
                         pred: np.ndarray,
                         d21_class: int,
                         neighbors: List[Tuple[str, int]]) -> List[str]:
    gt = flatten_labels(gt)
    pred = flatten_labels(pred)
    n = min(len(gt), len(pred))
    gt = gt[:n]
    pred = pred[:n]
    neigh_ids = [int(v) for _, v in neighbors]
    left_id = neigh_ids[0] if len(neigh_ids) >= 1 else None
    right_id = neigh_ids[1] if len(neigh_ids) >= 2 else None
    out = []
    for g, p in zip(gt, pred):
        g = int(g)
        p = int(p)
        is_gt = g == int(d21_class)
        is_pr = p == int(d21_class)
        if is_gt and is_pr:
            out.append(COLOR_TP)
        elif is_gt != is_pr:
            out.append(COLOR_ERROR)
        elif left_id is not None and g == left_id:
            out.append(COLOR_NEIGHBOR_LEFT)
        elif right_id is not None and g == right_id:
            out.append(COLOR_NEIGHBOR_RIGHT)
        elif g == 0:
            out.append(COLOR_BG_LIGHT)
        else:
            out.append(COLOR_OTHER_TEETH)
    return out


# ============================================================
# PLOT HELPERS
# ============================================================

def sample_plot_points(xyz: np.ndarray,
                       gt: np.ndarray,
                       pred: np.ndarray,
                       max_points: int,
                       seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(xyz)
    if max_points <= 0 or n <= max_points:
        return xyz, gt, pred
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=int(max_points), replace=False)
    return xyz[idx], gt[idx], pred[idx]


def compute_ranges(xyz: np.ndarray,
                   vertices: Optional[np.ndarray],
                   pad_frac: float) -> Dict[str, List[float]]:
    arrays = [np.asarray(xyz, dtype=np.float32)]
    if vertices is not None and np.asarray(vertices).size > 0:
        arrays.append(np.asarray(vertices, dtype=np.float32))
    pts = np.concatenate(arrays, axis=0)
    mins = np.nanmin(pts, axis=0)
    maxs = np.nanmax(pts, axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    pad = span * float(pad_frac)
    return {
        "x": [float(mins[0] - pad[0]), float(maxs[0] + pad[0])],
        "y": [float(mins[1] - pad[1]), float(maxs[1] + pad[1])],
        "z": [float(mins[2] - pad[2]), float(maxs[2] + pad[2])],
    }


def make_scene(ranges: Dict[str, List[float]],
               camera: Dict[str, Any],
               show_axes: bool) -> Dict[str, Any]:
    axis_cfg = dict(
        visible=bool(show_axes),
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


def mesh_trace(vertices: np.ndarray,
               faces: np.ndarray,
               opacity: float = 0.05,
               color: str = COLOR_MESH_HEX,
               showlegend: bool = False) -> go.Mesh3d:
    """
    Malla raw como fondo anatómico tenue.

    Importante para la visual final:
    - La malla se agrega ANTES de los puntos.
    - Los puntos relevantes se dibujan después, en trazas separadas y opacas.
    - Así la malla no oscurece los puntos beige ni los colores clínicos.
    """
    v = np.asarray(vertices, dtype=np.float32)
    f = np.asarray(faces, dtype=np.int64)
    return go.Mesh3d(
        x=v[:, 0],
        y=v[:, 1],
        z=v[:, 2],
        i=f[:, 0],
        j=f[:, 1],
        k=f[:, 2],
        color=color,
        opacity=float(opacity),
        flatshading=False,
        name="malla raw",
        showscale=False,
        showlegend=showlegend,
        hoverinfo="skip",
        lighting=dict(
            ambient=0.72,
            diffuse=0.58,
            fresnel=0.03,
            roughness=0.70,
            specular=0.06,
        ),
        lightposition=dict(x=100, y=200, z=300),
    )


def point_trace(xyz: np.ndarray,
                colors: List[str],
                name: str,
                size: float,
                opacity: float,
                showlegend: bool = False) -> go.Scatter3d:
    """Traza de compatibilidad: colores por punto en una sola capa."""
    xyz = np.asarray(xyz, dtype=np.float32)
    return go.Scatter3d(
        x=xyz[:, 0],
        y=xyz[:, 1],
        z=xyz[:, 2],
        mode="markers",
        marker=dict(
            size=float(size),
            color=colors,
            opacity=float(opacity),
            line=dict(width=0),
        ),
        name=name,
        showlegend=showlegend,
        hoverinfo="skip",
    )


def point_trace_subset(xyz: np.ndarray,
                       mask: np.ndarray,
                       color: str,
                       name: str,
                       size: float,
                       opacity: float,
                       showlegend: bool = False) -> Optional[go.Scatter3d]:
    """
    Traza por subconjunto.

    Esta es la corrección visual clave respecto a una sola traza con colores mixtos:
    permite que cada grupo tenga su propia opacidad. Por ejemplo, el background
    puede ser tenue, mientras que los puntos beige de 'otros dientes' quedan
    opacos y no se oscurecen por la malla.
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    n = min(len(xyz), len(mask))
    xyz = xyz[:n]
    mask = mask[:n]

    if not np.any(mask):
        return None

    pts = xyz[mask]
    return go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode="markers",
        marker=dict(
            size=float(size),
            color=color,
            opacity=float(opacity),
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


def add_trace(fig: go.Figure, trace: Optional[Any], row: int, col: int) -> None:
    if trace is not None:
        fig.add_trace(trace, row=row, col=col)


def add_mesh_background(fig: go.Figure,
                        row: int,
                        col: int,
                        vertices: Optional[np.ndarray],
                        faces: Optional[np.ndarray],
                        mesh_opacity: float,
                        mesh_color: str) -> None:
    if vertices is not None and faces is not None and mesh_opacity > 0:
        fig.add_trace(
            mesh_trace(vertices, faces, opacity=mesh_opacity, color=mesh_color),
            row=row,
            col=col,
        )


def _labels_np(labels: np.ndarray) -> np.ndarray:
    return flatten_labels(labels).astype(np.int64, copy=False)


def add_multiclass_panel(fig: go.Figure,
                         xyz: np.ndarray,
                         labels: np.ndarray,
                         row: int,
                         col: int,
                         vertices: Optional[np.ndarray],
                         faces: Optional[np.ndarray],
                         point_size: float,
                         point_opacity: float,
                         mesh_opacity: float,
                         mesh_color: str,
                         multiclass_bg_opacity: float,
                         multiclass_bg_mode: str) -> None:
    """
    Panel GT/pred multiclase.

    Visual v17/paper corregida:
    - malla primero, muy tenue;
    - background negro/soft por separado;
    - cada clase dental por separado, opaca.
    """
    y = _labels_np(labels)
    n = min(len(xyz), len(y))
    xyz = np.asarray(xyz, dtype=np.float32)[:n]
    y = y[:n]

    add_mesh_background(fig, row, col, vertices, faces, mesh_opacity, mesh_color)

    if multiclass_bg_mode == "soft":
        bg_color = "#282828"
    else:
        bg_color = COLOR_BG

    add_trace(
        fig,
        point_trace_subset(
            xyz, y == 0, bg_color, "background",
            size=point_size, opacity=multiclass_bg_opacity, showlegend=False,
        ),
        row,
        col,
    )

    for cls in sorted(int(c) for c in np.unique(y) if int(c) != 0):
        add_trace(
            fig,
            point_trace_subset(
                xyz,
                y == cls,
                CLASS_COLORS.get(cls, "#9E9E9E"),
                CLASS_NAMES.get(cls, f"class {cls}"),
                size=point_size,
                opacity=point_opacity,
                showlegend=False,
            ),
            row,
            col,
        )


def add_multiclass_error_panel(fig: go.Figure,
                               xyz: np.ndarray,
                               gt: np.ndarray,
                               pred: np.ndarray,
                               row: int,
                               col: int,
                               vertices: Optional[np.ndarray],
                               faces: Optional[np.ndarray],
                               point_size: float,
                               mesh_opacity: float,
                               mesh_color: str,
                               bg_context_opacity: float,
                               other_teeth_context_opacity: float,
                               error_opacity: float) -> None:
    """Panel de errores multiclase: rojo encima, contexto debajo."""
    g = _labels_np(gt)
    p = _labels_np(pred)
    n = min(len(xyz), len(g), len(p))
    xyz = np.asarray(xyz, dtype=np.float32)[:n]
    g = g[:n]
    p = p[:n]

    err = g != p
    ok = ~err

    add_mesh_background(fig, row, col, vertices, faces, mesh_opacity, mesh_color)

    add_trace(
        fig,
        point_trace_subset(
            xyz, ok & (g == 0), "#808080", "correcto background",
            size=point_size, opacity=bg_context_opacity, showlegend=False,
        ),
        row,
        col,
    )
    add_trace(
        fig,
        point_trace_subset(
            xyz, ok & (g != 0), COLOR_OTHER_TEETH, "correcto otros dientes",
            size=point_size, opacity=other_teeth_context_opacity, showlegend=False,
        ),
        row,
        col,
    )
    add_trace(
        fig,
        point_trace_subset(
            xyz, err, COLOR_ERROR, "error",
            size=point_size * 1.06, opacity=error_opacity, showlegend=False,
        ),
        row,
        col,
    )


def add_focus_panel(fig: go.Figure,
                    xyz: np.ndarray,
                    labels: np.ndarray,
                    row: int,
                    col: int,
                    vertices: Optional[np.ndarray],
                    faces: Optional[np.ndarray],
                    d21_class: int,
                    neighbors: List[Tuple[str, int]],
                    point_size: float,
                    mesh_opacity: float,
                    mesh_color: str,
                    bg_context_opacity: float,
                    other_teeth_opacity: float,
                    target_opacity: float) -> None:
    """
    Panel d21 + vecinos.

    Corrección principal del beige:
    el background/contexto se dibuja tenue, luego 'otros dientes' beige como
    capa opaca, y al final d21/vecinos. Así los puntos beige no quedan lavados
    por una opacidad global ni por la malla.
    """
    y = _labels_np(labels)
    n = min(len(xyz), len(y))
    xyz = np.asarray(xyz, dtype=np.float32)[:n]
    y = y[:n]

    neigh_ids = [int(v) for _, v in neighbors]
    left_id = neigh_ids[0] if len(neigh_ids) >= 1 else None
    right_id = neigh_ids[1] if len(neigh_ids) >= 2 else None

    add_mesh_background(fig, row, col, vertices, faces, mesh_opacity, mesh_color)

    add_trace(
        fig,
        point_trace_subset(
            xyz, y == 0, "#8C8C8C", "background contexto",
            size=point_size, opacity=bg_context_opacity, showlegend=False,
        ),
        row,
        col,
    )

    special = (y == int(d21_class))
    if left_id is not None:
        special |= (y == left_id)
    if right_id is not None:
        special |= (y == right_id)

    other_mask = (y != 0) & (~special)
    add_trace(
        fig,
        point_trace_subset(
            xyz, other_mask, COLOR_OTHER_TEETH, "otros dientes",
            size=point_size, opacity=other_teeth_opacity, showlegend=False,
        ),
        row,
        col,
    )

    if left_id is not None:
        add_trace(
            fig,
            point_trace_subset(
                xyz, y == left_id, COLOR_NEIGHBOR_LEFT, "vecino izq.",
                size=point_size * 1.04, opacity=target_opacity, showlegend=False,
            ),
            row,
            col,
        )

    add_trace(
        fig,
        point_trace_subset(
            xyz, y == int(d21_class), COLOR_D21, "d21",
            size=point_size * 1.05, opacity=target_opacity, showlegend=False,
        ),
        row,
        col,
    )

    if right_id is not None:
        add_trace(
            fig,
            point_trace_subset(
                xyz, y == right_id, COLOR_NEIGHBOR_RIGHT, "vecino der.",
                size=point_size * 1.04, opacity=target_opacity, showlegend=False,
            ),
            row,
            col,
        )


def add_d21_error_panel(fig: go.Figure,
                        xyz: np.ndarray,
                        gt: np.ndarray,
                        pred: np.ndarray,
                        row: int,
                        col: int,
                        vertices: Optional[np.ndarray],
                        faces: Optional[np.ndarray],
                        d21_class: int,
                        neighbors: List[Tuple[str, int]],
                        point_size: float,
                        mesh_opacity: float,
                        mesh_color: str,
                        bg_context_opacity: float,
                        other_teeth_opacity: float,
                        target_opacity: float,
                        error_opacity: float) -> None:
    """Panel error d21: verde TP, rojo FP/FN, contexto conservado."""
    g = _labels_np(gt)
    p = _labels_np(pred)
    n = min(len(xyz), len(g), len(p))
    xyz = np.asarray(xyz, dtype=np.float32)[:n]
    g = g[:n]
    p = p[:n]

    neigh_ids = [int(v) for _, v in neighbors]
    left_id = neigh_ids[0] if len(neigh_ids) >= 1 else None
    right_id = neigh_ids[1] if len(neigh_ids) >= 2 else None

    gt21 = g == int(d21_class)
    pr21 = p == int(d21_class)
    tp = gt21 & pr21
    err = gt21 != pr21

    add_mesh_background(fig, row, col, vertices, faces, mesh_opacity, mesh_color)

    add_trace(
        fig,
        point_trace_subset(
            xyz, g == 0, "#8C8C8C", "background contexto",
            size=point_size, opacity=bg_context_opacity, showlegend=False,
        ),
        row,
        col,
    )

    special = gt21.copy()
    if left_id is not None:
        special |= (g == left_id)
    if right_id is not None:
        special |= (g == right_id)

    other_mask = (g != 0) & (~special)
    add_trace(
        fig,
        point_trace_subset(
            xyz, other_mask, COLOR_OTHER_TEETH, "otros dientes",
            size=point_size, opacity=other_teeth_opacity, showlegend=False,
        ),
        row,
        col,
    )

    if left_id is not None:
        add_trace(
            fig,
            point_trace_subset(
                xyz, g == left_id, COLOR_NEIGHBOR_LEFT, "vecino izq.",
                size=point_size * 1.04, opacity=target_opacity, showlegend=False,
            ),
            row,
            col,
        )

    if right_id is not None:
        add_trace(
            fig,
            point_trace_subset(
                xyz, g == right_id, COLOR_NEIGHBOR_RIGHT, "vecino der.",
                size=point_size * 1.04, opacity=target_opacity, showlegend=False,
            ),
            row,
            col,
        )

    add_trace(
        fig,
        point_trace_subset(
            xyz, tp, COLOR_TP, "TP d21",
            size=point_size * 1.12, opacity=target_opacity, showlegend=False,
        ),
        row,
        col,
    )
    add_trace(
        fig,
        point_trace_subset(
            xyz, err, COLOR_ERROR, "error d21",
            size=point_size * 1.12, opacity=error_opacity, showlegend=False,
        ),
        row,
        col,
    )


def build_panel_figure(xyz: np.ndarray,
                       gt: np.ndarray,
                       pred: np.ndarray,
                       model: str,
                       patient_id: str,
                       case_role: str,
                       view_name: str,
                       camera: Dict[str, Any],
                       metrics: Dict[str, float],
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
                       multiclass_bg_opacity: float,
                       multiclass_bg_mode: str,
                       mesh_color: str = COLOR_MESH_HEX,
                       focus_bg_opacity: float = 0.075,
                       other_teeth_opacity: float = 1.0,
                       target_opacity: float = 1.0,
                       error_opacity: float = 1.0,
                       error_bg_context_opacity: float = 0.065,
                       error_other_teeth_opacity: float = 0.68) -> go.Figure:
    model_disp = model_display_name(model)

    subplot_titles = [
        "GT multiclase",
        f"{model_disp} predicción",
        "Errores multiclase",
        "GT d21 + vecinos",
        f"{model_disp} d21 + vecinos",
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
        vertical_spacing=0.030,
    )

    # Fila superior: multiclase y error multiclase.
    add_multiclass_panel(
        fig, xyz, gt, 1, 1, vertices, faces,
        point_size=point_size,
        point_opacity=point_opacity,
        mesh_opacity=mesh_opacity,
        mesh_color=mesh_color,
        multiclass_bg_opacity=multiclass_bg_opacity,
        multiclass_bg_mode=multiclass_bg_mode,
    )
    add_multiclass_panel(
        fig, xyz, pred, 1, 2, vertices, faces,
        point_size=point_size,
        point_opacity=point_opacity,
        mesh_opacity=mesh_opacity,
        mesh_color=mesh_color,
        multiclass_bg_opacity=multiclass_bg_opacity,
        multiclass_bg_mode=multiclass_bg_mode,
    )
    add_multiclass_error_panel(
        fig, xyz, gt, pred, 1, 3, vertices, faces,
        point_size=point_size,
        mesh_opacity=mesh_opacity,
        mesh_color=mesh_color,
        bg_context_opacity=error_bg_context_opacity,
        other_teeth_context_opacity=error_other_teeth_opacity,
        error_opacity=error_opacity,
    )

    # Fila inferior: d21 + vecinos. Aquí se evita la mezcla visual del beige.
    add_focus_panel(
        fig, xyz, gt, 2, 1, vertices, faces,
        d21_class=d21_class,
        neighbors=neighbors,
        point_size=point_size,
        mesh_opacity=mesh_opacity,
        mesh_color=mesh_color,
        bg_context_opacity=focus_bg_opacity,
        other_teeth_opacity=other_teeth_opacity,
        target_opacity=target_opacity,
    )
    add_focus_panel(
        fig, xyz, pred, 2, 2, vertices, faces,
        d21_class=d21_class,
        neighbors=neighbors,
        point_size=point_size,
        mesh_opacity=mesh_opacity,
        mesh_color=mesh_color,
        bg_context_opacity=focus_bg_opacity,
        other_teeth_opacity=other_teeth_opacity,
        target_opacity=target_opacity,
    )
    add_d21_error_panel(
        fig, xyz, gt, pred, 2, 3, vertices, faces,
        d21_class=d21_class,
        neighbors=neighbors,
        point_size=point_size,
        mesh_opacity=mesh_opacity,
        mesh_color=mesh_color,
        bg_context_opacity=focus_bg_opacity,
        other_teeth_opacity=other_teeth_opacity,
        target_opacity=target_opacity,
        error_opacity=error_opacity,
    )

    legend_items = [
        ("background", COLOR_BG),
        ("otros dientes", COLOR_OTHER_TEETH),
        ("d21", COLOR_D21),
        ("vecino izq.", COLOR_NEIGHBOR_LEFT),
        ("vecino der.", COLOR_NEIGHBOR_RIGHT),
        ("error", COLOR_ERROR),
        ("TP d21", COLOR_TP),
        ("malla raw", mesh_color),
    ]
    for name, color in legend_items:
        fig.add_trace(dummy_legend_trace(name, color), row=1, col=1)

    ranges = compute_ranges(xyz, vertices, pad_frac=0.03)
    scene_cfg = make_scene(ranges, camera, show_axes)
    for scene_name in ["scene", "scene2", "scene3", "scene4", "scene5", "scene6"]:
        fig.update_layout(**{scene_name: copy.deepcopy(scene_cfg)})

    title = build_main_title(model, patient_id, case_role, view_name)
    subtitle = metric_subtitle(metrics)

    fig.update_layout(
        title=dict(
            text=f"{title}<br><sup>{subtitle}</sup>",
            x=0.5,
            xanchor="center",
            font=dict(size=22, color="#2E4057"),
        ),
        width=int(width),
        height=int(height),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=0, r=0, t=100, b=10),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.020,
            xanchor="center",
            x=0.5,
            font=dict(size=13),
            bgcolor="rgba(255,255,255,0.75)",
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
  for (const s of scenes) cameras[s] = gd._fullLayout[s]?.camera;
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
    if (e[key]) { cam = e[key]; break; }
  }
  if (cam) {
    const update = {};
    for (const s of scenes) update[`${s}.camera`] = cam;
    syncing = true;
    Plotly.relayout(gd, update).then(() => { syncing = false; });
    console.log("[SYNC_CAMERA]");
    console.log(JSON.stringify(cam, null, 2));
  }
});
window.printCameras = function() {
  const cameras = {};
  for (const s of scenes) cameras[s] = gd._fullLayout[s]?.camera;
  console.log(JSON.stringify(cameras, null, 2));
};
window.printMasterCamera = function() {
  console.log(JSON.stringify(gd._fullLayout.scene.camera, null, 2));
};
"""


# ============================================================
# ARGUMENTOS Y MAIN
# ============================================================

def parse_neighbor_teeth(text: str) -> List[Tuple[str, int]]:
    text = (text or "").strip()
    if not text:
        return []
    out = []
    for chunk in text.replace(";", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"Formato incorrecto en neighbor_teeth: {chunk}. Usa d11:1,d22:9")
        name, value = chunk.split(":", 1)
        out.append((name.strip(), int(value.strip())))
    return out


def discover_case_dirs(base_dir: Path, models: List[str]) -> List[Path]:
    base_dir = Path(base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"No existe base_dir: {base_dir}")
    out = []
    for p in sorted(base_dir.iterdir()):
        if not p.is_dir():
            continue
        if p.name.startswith("VISTA_TESIS"):
            continue
        has_model = any((p / m).exists() for m in models)
        has_common = (p / "common").exists()
        if has_model or has_common:
            out.append(p)
    if not out:
        raise RuntimeError(f"No encontré case_dirs válidos dentro de {base_dir}")
    return out


def filter_case_dirs(case_dirs: List[Path], cases: List[str]) -> List[Path]:
    if not cases or cases == ["all"] or "all" in [c.lower() for c in cases]:
        return case_dirs
    selected = []
    for c in case_dirs:
        low = c.name.lower()
        for key in cases:
            if key.lower() in low:
                selected.append(c)
                break
    if not selected:
        raise RuntimeError(f"No encontré casos con filtro {cases}. Disponibles: {[p.name for p in case_dirs]}")
    return selected


def load_extra_camera(args: argparse.Namespace) -> Dict[str, Any]:
    if args.extra_camera_json:
        p = Path(args.extra_camera_json)
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                return json.load(f)
        try:
            return json.loads(args.extra_camera_json)
        except Exception as e:
            raise ValueError("--extra_camera_json debe ser ruta existente o JSON válido.") from e
    return copy.deepcopy(CAMERA_PRESETS["lateral_derecha_extra"])


def build_views_to_run(args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    views: Dict[str, Dict[str, Any]] = {}
    base_views = args.views or ["oclusal_paper", "isometrica_ajustada"]
    for v in base_views:
        if v not in CAMERA_PRESETS:
            raise KeyError(f"Vista '{v}' no existe. Disponibles: {list(CAMERA_PRESETS.keys())}")
        views[v] = copy.deepcopy(CAMERA_PRESETS[v])
    if args.include_extra_view and not args.disable_extra_view:
        views[args.extra_view_name] = load_extra_camera(args)
    return views


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base_dir",
        default="/home/htaucare/Tesis_Amaro/case_comparisons/QUALI_CASES_V17",
        help="Carpeta base con case_dirs best/median/worst.",
    )
    ap.add_argument(
        "--out_dir",
        default=None,
        help="Carpeta de salida. Default: <base_dir>/VISTA_TESIS_INFERENCIA_AJUSTADA (misma salida/base que la versión actual).",
    )
    ap.add_argument(
        "--selected_mesh_root",
        default="/home/htaucare/Tesis_Amaro/case_comparisons/metric_case_selection_v5_global_meshes/selected_global_meshes",
        help="Raíz donde están las mallas seleccionadas best/median/worst. Se usa si common/raw_mesh.obj no existe.",
    )
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Modelos a graficar.")
    ap.add_argument("--cases", nargs="+", default=["all"], help="Casos a usar: best median worst o all.")
    ap.add_argument(
        "--views",
        nargs="+",
        default=None,
        help="Vistas base. Default: oclusal_paper isometrica_ajustada.",
    )
    ap.add_argument("--include_extra_view", action="store_true", help="Agrega vista extra lateral_derecha_extra o cámara custom.")
    ap.add_argument("--disable_extra_view", action="store_true", help="Compatibilidad: desactiva vista extra si fue solicitada.")
    ap.add_argument("--extra_view_name", default="lateral_derecha_extra")
    ap.add_argument("--extra_camera_json", default=None, help="Ruta JSON o string JSON para cámara extra.")
    ap.add_argument("--d21_class", type=int, default=8)
    ap.add_argument("--neighbor_teeth", default="d11:1,d22:9")
    ap.add_argument("--case_role", default="", help="Override opcional: best/median/worst.")
    ap.add_argument("--patient_id", default="", help="Override opcional del ID mostrado en título.")
    ap.add_argument("--width", type=int, default=2200)
    ap.add_argument("--height", type=int, default=1350)
    ap.add_argument("--point_size", type=float, default=4.2)
    ap.add_argument("--point_opacity", type=float, default=0.92)
    ap.add_argument("--mesh_color", default=COLOR_MESH_HEX,
                    help="Color de la malla raw. Default: gris v17/paper.")
    ap.add_argument("--focus_bg_opacity", type=float, default=0.075,
                    help="Opacidad del background/contexto en la fila d21 + vecinos.")
    ap.add_argument("--other_teeth_opacity", type=float, default=1.0,
                    help="Opacidad de otros dientes/beige en la fila inferior.")
    ap.add_argument("--target_opacity", type=float, default=1.0,
                    help="Opacidad de d21 y vecinos.")
    ap.add_argument("--error_opacity", type=float, default=1.0,
                    help="Opacidad de errores rojos y TP d21.")
    ap.add_argument("--error_bg_context_opacity", type=float, default=0.065,
                    help="Opacidad del background correcto en el panel de errores multiclase.")
    ap.add_argument("--error_other_teeth_opacity", type=float, default=0.68,
                    help="Opacidad del contexto beige correcto en errores multiclase.")
    ap.add_argument("--mesh_opacity", type=float, default=0.05)
    ap.add_argument("--no_mesh", action="store_true")
    ap.add_argument("--no_mesh_bbox_align", action="store_true")
    ap.add_argument("--max_points_plot", type=int, default=12000, help="<=0 usa todos los puntos.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--show_axes", action="store_true")
    ap.add_argument("--write_html", action="store_true")
    ap.add_argument("--write_png", action="store_true")
    ap.add_argument("--png_scale", type=float, default=2.0)
    ap.add_argument("--sync_camera", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--multiclass_bg_opacity",
        type=float,
        default=0.56,
        help="Opacidad del background en paneles multiclase.",
    )
    ap.add_argument(
        "--multiclass_bg_mode",
        choices=["black", "soft"],
        default="black",
        help="black conserva la visual v17; soft deja el fondo más tenue.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    base_dir = Path(args.base_dir)
    out_root = Path(args.out_dir) if args.out_dir else base_dir / "VISTA_TESIS_INFERENCIA_AJUSTADA"
    ensure_dir(out_root)

    if not args.write_html and not args.write_png:
        args.write_html = True

    models = [sanitize_model_name(m) for m in args.models]
    neighbors = parse_neighbor_teeth(args.neighbor_teeth)
    views = build_views_to_run(args)
    selected_mesh_root = Path(args.selected_mesh_root) if args.selected_mesh_root else None

    case_dirs_all = discover_case_dirs(base_dir, models)
    case_dirs = filter_case_dirs(case_dirs_all, args.cases)

    manifest_rows: List[Dict[str, Any]] = []

    print("=" * 90)
    print("[INFO] vista_tesis_inferencia_ajustada_v4_visual_v17.py")
    print(f"[INFO] base_dir: {base_dir}")
    print(f"[INFO] out_root: {out_root}")
    print(f"[INFO] selected_mesh_root: {selected_mesh_root}")
    print(f"[INFO] cases: {[p.name for p in case_dirs]}")
    print(f"[INFO] models: {models}")
    print(f"[INFO] views: {list(views.keys())}")
    print(f"[INFO] write_html: {args.write_html}")
    print(f"[INFO] write_png: {args.write_png}")
    print("=" * 90)

    for case_dir in case_dirs:
        common = load_common_bundle(case_dir)
        case_tag = case_short_tag(case_dir)
        patient_id = clean_patient_id(args.patient_id) if args.patient_id else infer_patient_id(case_dir, common)
        role = case_role_label(case_dir, args.case_role)

        print(f"\n[CASE] {case_dir.name} | ID={patient_id} | {role}")

        mesh_vertices = None
        mesh_faces = None
        mesh_source = ""
        if not args.no_mesh:
            mesh_vertices, mesh_faces, mesh_source = try_load_mesh(case_dir, selected_mesh_root, patient_id, case_tag)
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
                xyz, gt, pred, data_sources = load_case_model_data(case_dir, model)
            except Exception as e:
                print(f"      [ERROR] No pude cargar datos de {model}: {e}", file=sys.stderr)
                continue

            metrics = compute_metrics(gt, pred, d21_class=args.d21_class)
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
                    vertices_plot = align_mesh_bbox_to_xyz(vertices_plot, xyz_plot)

            for view_name, camera in views.items():
                out_dir = out_root / case_dir.name / model
                ensure_dir(out_dir)
                stem = f"vista_tesis_{case_tag}_{model}_{view_name}"
                html_path = out_dir / f"{stem}.html"
                png_path = out_dir / f"{stem}.png"
                summary_path = out_dir / f"{stem}.summary.json"

                need_fig = False
                if args.write_html and (args.force or not html_path.exists()):
                    need_fig = True
                if args.write_png and (args.force or not png_path.exists()):
                    need_fig = True

                fig = None
                if need_fig:
                    fig = build_panel_figure(
                        xyz=xyz_plot,
                        gt=gt_plot,
                        pred=pred_plot,
                        model=model,
                        patient_id=patient_id,
                        case_role=role,
                        view_name=view_name,
                        camera=camera,
                        metrics=metrics,
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
                        multiclass_bg_opacity=args.multiclass_bg_opacity,
                        multiclass_bg_mode=args.multiclass_bg_mode,
                        mesh_color=args.mesh_color,
                        focus_bg_opacity=args.focus_bg_opacity,
                        other_teeth_opacity=args.other_teeth_opacity,
                        target_opacity=args.target_opacity,
                        error_opacity=args.error_opacity,
                        error_bg_context_opacity=args.error_bg_context_opacity,
                        error_other_teeth_opacity=args.error_other_teeth_opacity,
                    )

                if args.write_html:
                    if html_path.exists() and not args.force:
                        print(f"      [SKIP HTML existe] {html_path}")
                    elif fig is not None:
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
                                    "height": int(args.height),
                                    "width": int(args.width),
                                    "scale": float(args.png_scale),
                                },
                            },
                        )
                        print(f"      [OK HTML] {html_path}")

                if args.write_png:
                    if png_path.exists() and not args.force:
                        print(f"      [SKIP PNG existe] {png_path}")
                    elif fig is not None:
                        try:
                            fig.write_image(
                                str(png_path),
                                width=int(args.width),
                                height=int(args.height),
                                scale=float(args.png_scale),
                            )
                            print(f"      [OK PNG]  {png_path}")
                        except Exception as e:
                            print(
                                f"      [WARN PNG] No se pudo exportar PNG con Kaleido/Chrome: {e}",
                                file=sys.stderr,
                            )
                            print(
                                "      [INFO] El HTML queda disponible; puedes abrirlo y usar Download plot as png.",
                                file=sys.stderr,
                            )

                summary = {
                    "script": "vista_tesis_inferencia_ajustada.py",
                    "case_dir": str(case_dir),
                    "case_name": case_dir.name,
                    "case_tag": case_tag,
                    "case_role": role,
                    "patient_id": patient_id,
                    "model": model,
                    "model_display": model_display_name(model),
                    "view_name": view_name,
                    "view_display": pretty_view_name(view_name),
                    "title": build_main_title(model, patient_id, role, view_name),
                    "subtitle": metric_subtitle(metrics),
                    "camera": camera,
                    "n_points_original": int(len(xyz)),
                    "n_points_plot": int(len(xyz_plot)),
                    "d21_class": int(args.d21_class),
                    "neighbor_teeth": neighbors,
                    "metrics": {k: (None if isinstance(v, float) and not np.isfinite(v) else float(v)) for k, v in metrics.items()},
                    "mesh_source": mesh_source,
                    "mesh_loaded": bool(mesh_vertices is not None and mesh_faces is not None),
                    "mesh_bbox_align": not bool(args.no_mesh_bbox_align),
                    "visual_layer": "v17_layered_points_mesh_background",
                    "mesh_color": args.mesh_color,
                    "focus_bg_opacity": float(args.focus_bg_opacity),
                    "other_teeth_opacity": float(args.other_teeth_opacity),
                    "target_opacity": float(args.target_opacity),
                    "error_opacity": float(args.error_opacity),
                    "error_bg_context_opacity": float(args.error_bg_context_opacity),
                    "error_other_teeth_opacity": float(args.error_other_teeth_opacity),
                    "data_sources": data_sources,
                    "outputs": {
                        "html": str(html_path) if args.write_html else "",
                        "png": str(png_path) if args.write_png else "",
                        "summary": str(summary_path),
                    },
                }
                save_json(summary, summary_path)

                manifest_rows.append({
                    "case_name": case_dir.name,
                    "case_tag": case_tag,
                    "case_role": role,
                    "patient_id": patient_id,
                    "model": model,
                    "model_display": model_display_name(model),
                    "view_name": view_name,
                    "view_display": pretty_view_name(view_name),
                    "html": str(html_path) if args.write_html else "",
                    "png": str(png_path) if args.write_png else "",
                    "summary": str(summary_path),
                    "acc_no_bg": "" if not np.isfinite(metrics["acc_no_bg"]) else f"{metrics['acc_no_bg']:.6f}",
                    "d21_f1": f"{metrics['d21_f1']:.6f}",
                    "d21_iou": f"{metrics['d21_iou']:.6f}",
                    "f1_macro_no_bg": f"{metrics['f1_macro_no_bg']:.6f}",
                    "mesh_source": mesh_source,
                })

    manifest_path = out_root / "manifest_vista_tesis_inferencia_ajustada.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "case_name", "case_tag", "case_role", "patient_id",
            "model", "model_display", "view_name", "view_display",
            "html", "png", "summary",
            "acc_no_bg", "d21_f1", "d21_iou", "f1_macro_no_bg",
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
        "selected_mesh_root": str(selected_mesh_root) if selected_mesh_root else "",
        "models": models,
        "cases": [p.name for p in case_dirs],
        "views": list(views.keys()),
        "d21_class": int(args.d21_class),
        "neighbor_teeth": neighbors,
        "write_html": bool(args.write_html),
        "write_png": bool(args.write_png),
        "width": int(args.width),
        "height": int(args.height),
        "point_size": float(args.point_size),
        "point_opacity": float(args.point_opacity),
        "mesh_opacity": float(args.mesh_opacity),
        "mesh_color": args.mesh_color,
        "focus_bg_opacity": float(args.focus_bg_opacity),
        "other_teeth_opacity": float(args.other_teeth_opacity),
        "target_opacity": float(args.target_opacity),
        "error_opacity": float(args.error_opacity),
        "error_bg_context_opacity": float(args.error_bg_context_opacity),
        "error_other_teeth_opacity": float(args.error_other_teeth_opacity),
        "visual_layer": "v17_layered_points_mesh_background",
        "max_points_plot": int(args.max_points_plot),
        "sync_camera": bool(args.sync_camera),
        "multiclass_bg_opacity": float(args.multiclass_bg_opacity),
        "multiclass_bg_mode": args.multiclass_bg_mode,
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
er = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow(row)

    run_meta = {
        "script": "vista_tesis_inferencia_ajustada.py",
        "base_dir": str(base_dir),
        "out_root": str(out_root),
        "selected_mesh_root": str(selected_mesh_root) if selected_mesh_root else "",
        "models": models,
        "cases": [p.name for p in case_dirs],
        "views": list(views.keys()),
        "d21_class": int(args.d21_class),
        "neighbor_teeth": neighbors,
        "write_html": bool(args.write_html),
        "write_png": bool(args.write_png),
        "width": int(args.width),
        "height": int(args.height),
        "point_size": float(args.point_size),
        "point_opacity": float(args.point_opacity),
        "mesh_opacity": float(args.mesh_opacity),
        "mesh_color": args.mesh_color,
        "focus_bg_opacity": float(args.focus_bg_opacity),
        "other_teeth_opacity": float(args.other_teeth_opacity),
        "target_opacity": float(args.target_opacity),
        "error_opacity": float(args.error_opacity),
        "error_bg_context_opacity": float(args.error_bg_context_opacity),
        "error_other_teeth_opacity": float(args.error_other_teeth_opacity),
        "visual_layer": "v17_layered_points_mesh_background",
        "max_points_plot": int(args.max_points_plot),
        "sync_camera": bool(args.sync_camera),
        "multiclass_bg_opacity": float(args.multiclass_bg_opacity),
        "multiclass_bg_mode": args.multiclass_bg_mode,
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
