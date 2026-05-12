#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
augmentation_and_split_paperlike_v3_sampleo_FPS.py

Paper-like augmentation and split creation con soporte de múltiples estrategias
de submuestreo para segmentación dental 3D.

Este script toma un dataset merged_* ya trazable:

    merged_200000_safe_excl_wisdom_upper_only/
        X_train.npz
        Y_train.npz
        X_val.npz
        Y_val.npz
        X_test.npz
        Y_test.npz
        index_train.csv
        index_val.csv
        index_test.csv
        artifacts/
            label_map.json
            meta.json

y genera un fixed_split con N puntos por muestra:

    fixed_split/<N>/<experimento>/
        X_train.npz
        Y_train.npz
        X_val.npz
        Y_val.npz
        X_test.npz
        Y_test.npz
        artifacts/
            label_map.json
            class_weights.json
            meta.json

IMPORTANTE:
- Este script NO modifica la trazabilidad por sí mismo.
- Después de crear el fixed_split, se debe ejecutar:

    create_traceability_indices_for_fixed_split.py

para restaurar index_train.csv, index_val.csv e index_test.csv.

Estrategias de sampleo soportadas:

1) random
   - Equivalente al comportamiento tradicional del pipeline.
   - Soporta control de background mediante --cap_bg.

2) fps
   - Farthest Point Sampling global.
   - Soporta control de background mediante --cap_bg.
   - Si --cap_bg está activo, primero construye un pool balanceado bg/fg y luego aplica FPS
     sobre ese pool o selecciona por cuotas.

3) fps_curvature
   - Híbrido FPS + curvatura.
   - Intenta preservar cobertura geométrica global y reforzar bordes/surcos.
   - Soporta control de background mediante --cap_bg.
   - Usa una aproximación de curvatura local basada en distancia a vecinos kNN.

4) boundary_labels
   - Boundary-aware usando etiquetas.
   - Sobremuestrea puntos cercanos a cambios de clase, detectados con kNN en labels.
   - Soporta control de background mediante --cap_bg.
   - Es un modo supervisado: se recomienda usarlo como experimento/ablation.

Diseño metodológico:
- El split se hereda desde el dataset merged.
- El orden de las muestras train/val/test se mantiene.
- El sampleo cambia solo los puntos dentro de cada nube, no la identidad de la muestra.
- Por eso se puede restaurar trazabilidad desde el merged original.

Autor: Adaptado para Tesis_Amaro.
"""

import argparse
import json
import math
import time
import gc
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np


try:
    from tqdm import tqdm
    HAS_TQDM = True
except Exception:
    HAS_TQDM = False


# ============================================================
# IO / JSON / RNG
# ============================================================

def np_rng(seed: int) -> np.random.Generator:
    """
    Generador NumPy moderno y reproducible.
    """
    return np.random.default_rng(int(seed))


def json_dump(obj: Any, path: Path) -> None:
    """
    Guarda JSON con indentación.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def json_load(path: Path) -> Optional[Any]:
    """
    Carga JSON si existe.
    """
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> Path:
    """
    Crea directorio si no existe.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def iter_progress(items, desc: str):
    """
    Iterador con tqdm si está instalado.
    """
    if HAS_TQDM:
        return tqdm(items, desc=desc)
    return items


def as_jsonable_labelmap(id2idx: Dict[int, int], idx2id: Dict[int, int]) -> Dict[str, Dict[str, int]]:
    """
    Convierte mapas con claves int a JSON serializable.
    """
    return {
        "id2idx": {str(k): int(v) for k, v in id2idx.items()},
        "idx2id": {str(k): int(v) for k, v in idx2id.items()},
    }


# ============================================================
# SAFE RESUME / ATOMIC CACHE HELPERS
# ============================================================

def atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    """
    Guarda un .npy de forma atómica.

    Motivo:
    - np.save directo puede dejar un archivo corrupto si el proceso se cae.
    - Aquí se escribe primero a *.tmp y solo al final se reemplaza el destino.
    - Path.replace() es atómico dentro del mismo filesystem.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_name(path.name + ".tmp")

    with tmp.open("wb") as f:
        np.save(f, arr)
        f.flush()

    tmp.replace(path)


def atomic_json_dump(obj: Any, path: Path) -> None:
    """
    Guarda un JSON de forma atómica.
    Se usa como marcador de muestra terminada en el cache de resume.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_name(path.name + ".tmp")

    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.flush()

    tmp.replace(path)


def resume_sample_paths(cache_dir: Path, split_name: str, stage_name: str, sample_idx: int) -> Dict[str, Path]:
    """
    Construye rutas del cache por muestra.

    Ejemplos:
      _resume_cache/train/original/X_000001.npy
      _resume_cache/train/original/Y_000001.npy
      _resume_cache/train/original/DONE_000001.json

      _resume_cache/train/augmentation_aug001/X_000001.npy
      _resume_cache/train/augmentation_aug001/Y_000001.npy
      _resume_cache/train/augmentation_aug001/DONE_000001.json
    """
    root = Path(cache_dir) / str(split_name) / str(stage_name)
    sample_idx = int(sample_idx)

    return {
        "root": root,
        "x": root / f"X_{sample_idx:06d}.npy",
        "y": root / f"Y_{sample_idx:06d}.npy",
        "done": root / f"DONE_{sample_idx:06d}.json",
    }


def is_valid_cached_sample(
    x_path: Path,
    y_path: Path,
    done_path: Path,
    N: int,
) -> bool:
    """
    Verifica si una muestra cacheada está completa y tiene shape esperado.
    No basta con que existan X/Y: se exige DONE_*.json para evitar leer un guardado parcial.
    """
    x_path = Path(x_path)
    y_path = Path(y_path)
    done_path = Path(done_path)

    if not (x_path.exists() and y_path.exists() and done_path.exists()):
        return False

    try:
        x = np.load(x_path, mmap_mode="r")
        y = np.load(y_path, mmap_mode="r")

        if tuple(x.shape) != (int(N), 3):
            return False
        if tuple(y.shape) != (int(N),):
            return False
        if x.dtype not in (np.float32, np.float64):
            return False
        if y.dtype not in (np.int32, np.int64, np.int16, np.uint8, np.uint16, np.uint32):
            return False

        _ = json_load(done_path)
        return True

    except Exception:
        return False


def load_cached_sample(
    x_path: Path,
    y_path: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carga una muestra desde cache y fuerza dtype final del pipeline.
    """
    x = np.load(Path(x_path)).astype(np.float32, copy=False)
    y = np.load(Path(y_path)).astype(np.int32, copy=False)
    return x, y


def save_cached_sample(
    x_path: Path,
    y_path: Path,
    done_path: Path,
    x: np.ndarray,
    y: np.ndarray,
    meta: Dict[str, Any],
) -> None:
    """
    Guarda X/Y por muestra y luego escribe marcador DONE.
    Si el proceso cae antes de DONE, en el próximo resume esa muestra se recalcula.
    """
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)

    atomic_save_npy(Path(x_path), x)
    atomic_save_npy(Path(y_path), y)
    atomic_json_dump(meta, Path(done_path))


def resolve_resume_cache_dir(args: argparse.Namespace, out_dir: Path) -> Path:
    """
    Devuelve la carpeta de cache para safe-resume.
    Por defecto queda dentro de out_dir para que viaje junto al experimento.
    """
    custom = getattr(args, "resume_cache_dir", None)
    if custom is not None and str(custom).strip() != "":
        return Path(custom).resolve()
    return Path(out_dir).resolve() / "_resume_cache"


# ============================================================
# GEOMETRY HELPERS
# ============================================================

def sanitize_points(points: np.ndarray) -> np.ndarray:
    """
    Asegura formato float32, shape [N,3] y valores finitos.
    """
    points = np.asarray(points, dtype=np.float32)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points debe tener shape [N,3], recibido {points.shape}")

    if not np.isfinite(points).all():
        points = np.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)

    return np.ascontiguousarray(points, dtype=np.float32)


def sanitize_labels(labels: np.ndarray, n_points: int) -> np.ndarray:
    """
    Asegura labels int32 de largo n_points.
    """
    labels = np.asarray(labels).reshape(-1).astype(np.int32, copy=False)

    if labels.shape[0] != int(n_points):
        raise ValueError(
            f"labels no calzan con points: labels={labels.shape[0]} points={n_points}"
        )

    return np.ascontiguousarray(labels, dtype=np.int32)


def normalize_unit_sphere_np(points: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Normaliza una nube a esfera unitaria:
      - centra por media
      - divide por radio máximo

    Se incluye por seguridad, aunque el merged normalmente ya viene normalizado.
    """
    points = sanitize_points(points)
    c = points.mean(axis=0, keepdims=True)
    x = points - c
    r = np.linalg.norm(x, axis=1).max()
    if np.isfinite(r) and r > eps:
        x = x / r
    return x.astype(np.float32, copy=False)


def rotate_z(points: np.ndarray, max_deg: float = 15.0, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Rotación alrededor del eje Z.
    """
    if rng is None:
        rng = np_rng(42)

    theta = float(rng.uniform(-float(max_deg), float(max_deg))) * (np.pi / 180.0)
    c, s = np.cos(theta), np.sin(theta)

    R = np.array(
        [
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    return (points @ R).astype(np.float32, copy=False)


def jitter(
    points: np.ndarray,
    sigma: float = 0.005,
    clip: float = 0.02,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Ruido gaussiano por punto.
    """
    if rng is None:
        rng = np_rng(42)

    noise = rng.normal(0.0, float(sigma), size=points.shape).astype(np.float32)
    noise = np.clip(noise, -float(clip), float(clip))
    return (points + noise).astype(np.float32, copy=False)


def scale(
    points: np.ndarray,
    min_s: float = 0.95,
    max_s: float = 1.05,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Escalado isotrópico.
    """
    if rng is None:
        rng = np_rng(42)

    s = float(rng.uniform(float(min_s), float(max_s)))
    return (points * s).astype(np.float32, copy=False)


def dropout_points_with_labels(
    points: np.ndarray,
    labels: np.ndarray,
    drop_rate: float = 0.05,
    min_keep: int = 32,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dropout de puntos manteniendo labels alineados.
    """
    if rng is None:
        rng = np_rng(42)

    points = sanitize_points(points)
    labels = sanitize_labels(labels, points.shape[0])

    drop_rate = float(drop_rate)
    if drop_rate <= 0.0:
        return points, labels

    n = int(points.shape[0])
    if n <= int(min_keep):
        return points, labels

    mask = rng.random(n) > drop_rate

    if int(mask.sum()) < int(min_keep):
        keep_idx = rng.choice(np.arange(n), size=int(min_keep), replace=False)
        mask[:] = False
        mask[keep_idx] = True

    return points[mask].astype(np.float32, copy=False), labels[mask].astype(np.int32, copy=False)


def augment_points_labels(
    points: np.ndarray,
    labels: np.ndarray,
    rotate_deg: float = 15.0,
    jitter_sigma: float = 0.005,
    jitter_clip: float = 0.02,
    scale_min: float = 0.95,
    scale_max: float = 1.05,
    dropout_rate: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augmentation geométrico manteniendo labels.
    No fuerza N final; el resample final se hace luego con la estrategia elegida.
    """
    if rng is None:
        rng = np_rng(42)

    points = sanitize_points(points)
    labels = sanitize_labels(labels, points.shape[0])

    x = rotate_z(points, max_deg=rotate_deg, rng=rng)
    x = jitter(x, sigma=jitter_sigma, clip=jitter_clip, rng=rng)
    x = scale(x, min_s=scale_min, max_s=scale_max, rng=rng)
    x, y = dropout_points_with_labels(
        x,
        labels,
        drop_rate=dropout_rate,
        min_keep=max(32, min(1024, points.shape[0] // 8)),
        rng=rng,
    )

    return x.astype(np.float32, copy=False), y.astype(np.int32, copy=False)


# ============================================================
# LABEL MAP
# ============================================================

def build_global_label_map(Y_list: List[np.ndarray]) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Construye label map global sobre todos los splits.
    """
    vals = set()

    for Y in Y_list:
        yy = np.asarray(Y).reshape(-1)
        vals.update(np.unique(yy).astype(int).tolist())

    vals = sorted(int(v) for v in vals)

    id2idx = {int(v): int(i) for i, v in enumerate(vals)}
    idx2id = {int(i): int(v) for i, v in enumerate(vals)}

    return id2idx, idx2id


def apply_remap(Y: np.ndarray, id2idx: Dict[int, int]) -> np.ndarray:
    """
    Remapea etiquetas originales a índices internos 0..C-1.
    """
    Y = np.asarray(Y).astype(np.int32, copy=False)

    out = np.empty_like(Y, dtype=np.int32)

    # vectorización robusta sin np.vectorize para datasets grandes
    flat = Y.reshape(-1)
    out_flat = out.reshape(-1)

    for src, dst in id2idx.items():
        out_flat[flat == int(src)] = int(dst)

    # fallback defensivo: cualquier valor no mapeado va a 0
    mapped_values = set(int(k) for k in id2idx.keys())
    unknown_mask = np.ones(flat.shape[0], dtype=bool)
    for src in mapped_values:
        unknown_mask &= (flat != int(src))
    if unknown_mask.any():
        out_flat[unknown_mask] = 0

    return out.astype(np.int32, copy=False)


# ============================================================
# CLASS WEIGHTS
# ============================================================

def compute_class_weights_from_train(Y_train: np.ndarray, num_classes: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula pesos inversos simples desde train.
    """
    flat = np.asarray(Y_train).reshape(-1).astype(np.int64, copy=False)

    hist = np.bincount(flat, minlength=int(num_classes)).astype(np.float64)

    w = 1.0 / (hist + 1e-8)
    w = w * (float(num_classes) / (w.sum() + 1e-8))

    return w.astype(np.float32), hist.astype(np.int64)


def compute_bg_fraction(Y: np.ndarray, bg_id: int = 0) -> float:
    """
    Fracción de background.
    """
    Y = np.asarray(Y).reshape(-1)
    if Y.size == 0:
        return 0.0
    return float((Y == int(bg_id)).mean())


def split_eda(Y: np.ndarray, bg_id: int = 0) -> Dict[str, Any]:
    """
    Resumen simple por split.
    """
    Y = np.asarray(Y)
    flat = Y.reshape(-1)
    if flat.size == 0:
        return {
            "min": None,
            "max": None,
            "unique": 0,
            "bg_frac": 0.0,
        }

    return {
        "min": int(flat.min()),
        "max": int(flat.max()),
        "unique": int(np.unique(flat).size),
        "bg_frac": compute_bg_fraction(flat, bg_id=bg_id),
    }


# ============================================================
# DISTANCE / KNN HELPERS NUMPY
# ============================================================

def pairwise_dist_sq_chunked(
    query: np.ndarray,
    ref: np.ndarray,
    chunk_size: int = 4096,
) -> np.ndarray:
    """
    Calcula distancias cuadradas entre query [M,3] y ref [N,3] por chunks.
    Retorna [M,N].

    Se usa solo en piezas donde N no sea enorme o con chunks controlados.
    Para 200k x 200k no debe usarse completo.
    """
    query = sanitize_points(query)
    ref = sanitize_points(ref)

    M = int(query.shape[0])
    N = int(ref.shape[0])

    out = np.empty((M, N), dtype=np.float32)

    ref_norm = np.sum(ref * ref, axis=1, keepdims=True).T  # [1,N]

    for s in range(0, M, int(chunk_size)):
        e = min(M, s + int(chunk_size))
        q = query[s:e]
        q_norm = np.sum(q * q, axis=1, keepdims=True)      # [m,1]
        d = q_norm + ref_norm - 2.0 * (q @ ref.T)
        np.maximum(d, 0.0, out=d)
        out[s:e] = d.astype(np.float32, copy=False)

    return out


def knn_indices_numpy(
    points: np.ndarray,
    k: int = 16,
    query_indices: Optional[np.ndarray] = None,
    chunk_size: int = 2048,
) -> np.ndarray:
    """
    kNN simple en NumPy para subconjuntos.

    points: [N,3]
    query_indices:
      - None: queries = points completos
      - array: queries = points[query_indices]

    Retorna:
      idx_knn [M,k]
    """
    points = sanitize_points(points)
    N = int(points.shape[0])
    k = int(max(1, min(int(k), N)))

    if query_indices is None:
        query = points
    else:
        query_indices = np.asarray(query_indices, dtype=np.int64).reshape(-1)
        query = points[query_indices]

    M = int(query.shape[0])
    idx_out = np.empty((M, k), dtype=np.int64)

    ref_norm = np.sum(points * points, axis=1, keepdims=True).T

    for s in range(0, M, int(chunk_size)):
        e = min(M, s + int(chunk_size))
        q = query[s:e]
        q_norm = np.sum(q * q, axis=1, keepdims=True)
        d = q_norm + ref_norm - 2.0 * (q @ points.T)
        np.maximum(d, 0.0, out=d)

        # argpartition para eficiencia
        idx = np.argpartition(d, kth=k - 1, axis=1)[:, :k]
        idx_out[s:e] = idx.astype(np.int64, copy=False)

    return idx_out


# ============================================================
# BASIC INDEX HELPERS
# ============================================================

def unique_preserve_order(x: np.ndarray) -> np.ndarray:
    """
    Únicos preservando orden.
    """
    x = np.asarray(x, dtype=np.int64).reshape(-1)
    seen = set()
    out = []
    for v in x.tolist():
        if int(v) not in seen:
            seen.add(int(v))
            out.append(int(v))
    return np.asarray(out, dtype=np.int64)


def fill_to_n_from_pool(
    selected: np.ndarray,
    pool: np.ndarray,
    n_target: int,
    rng: np.random.Generator,
    replace_if_needed: bool = True,
) -> np.ndarray:
    """
    Completa selected hasta n_target usando pool.
    Evita duplicados cuando sea posible.
    """
    selected = np.asarray(selected, dtype=np.int64).reshape(-1)
    pool = np.asarray(pool, dtype=np.int64).reshape(-1)

    selected = unique_preserve_order(selected)

    if selected.shape[0] >= int(n_target):
        return selected[:int(n_target)]

    selected_set = set(int(v) for v in selected.tolist())
    remaining = np.asarray([int(v) for v in pool.tolist() if int(v) not in selected_set], dtype=np.int64)

    need = int(n_target) - int(selected.shape[0])

    if remaining.size >= need:
        extra = rng.choice(remaining, size=need, replace=False)
    else:
        if remaining.size > 0:
            extra1 = remaining
        else:
            extra1 = np.empty((0,), dtype=np.int64)

        still = need - extra1.size

        if still > 0:
            if pool.size == 0:
                # fallback extremo
                extra2 = rng.choice(selected, size=still, replace=True) if selected.size else np.zeros(still, dtype=np.int64)
            else:
                extra2 = rng.choice(pool, size=still, replace=bool(replace_if_needed))
            extra = np.concatenate([extra1, extra2], axis=0)
        else:
            extra = extra1

    out = np.concatenate([selected, extra.astype(np.int64)], axis=0)

    if out.shape[0] != int(n_target):
        if out.shape[0] > int(n_target):
            out = out[:int(n_target)]
        else:
            extra = rng.choice(pool if pool.size else out, size=int(n_target) - out.shape[0], replace=True)
            out = np.concatenate([out, extra], axis=0)

    return out.astype(np.int64, copy=False)


def shuffle_indices(idx: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Baraja índices.
    """
    idx = np.asarray(idx, dtype=np.int64).reshape(-1).copy()
    if idx.size > 1:
        rng.shuffle(idx)
    return idx


# ============================================================
# BACKGROUND CONTROL
# ============================================================

def resolve_cap_bg_frac(cap_bg: Optional[float], N: int) -> Optional[float]:
    """
    Interpreta --cap_bg.

    Reglas:
    - None o <0: sin control de background.
    - 0 <= cap_bg <= 1: fracción máxima de background.
    - cap_bg > 1: número absoluto máximo de puntos background, convertido a fracción.

    Ejemplos:
    --cap_bg 0.8   => máximo 80% de background
    --cap_bg 6553  => máximo 6553 puntos de background sobre N
    """
    if cap_bg is None:
        return None

    cap_bg = float(cap_bg)

    if cap_bg < 0:
        return None

    if cap_bg <= 1.0:
        return float(cap_bg)

    if int(N) <= 0:
        return None

    return float(min(1.0, cap_bg / float(N)))


def split_bg_fg_indices(labels: np.ndarray, bg_id: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Divide índices en:
    - all
    - bg
    - fg
    """
    labels = np.asarray(labels).reshape(-1).astype(np.int32, copy=False)

    idx_all = np.arange(labels.shape[0], dtype=np.int64)
    idx_bg = idx_all[labels == int(bg_id)]
    idx_fg = idx_all[labels != int(bg_id)]

    return idx_all, idx_bg, idx_fg


def compute_bg_fg_quota(
    labels: np.ndarray,
    N: int,
    cap_bg_frac: Optional[float],
    bg_id: int = 0,
) -> Tuple[int, int]:
    """
    Calcula cuántos puntos bg y fg tomar para cumplir cap_bg.

    Si no hay cap_bg:
      - no impone cuota; devuelve (-1, -1)

    Si hay cap_bg:
      - take_bg <= cap_bg_frac * N
      - take_fg = N - take_bg
    """
    labels = np.asarray(labels).reshape(-1).astype(np.int32, copy=False)
    N = int(N)

    if cap_bg_frac is None:
        return -1, -1

    _, idx_bg, idx_fg = split_bg_fg_indices(labels, bg_id=bg_id)

    max_bg = int(round(float(cap_bg_frac) * float(N)))
    max_bg = max(0, min(N, max_bg))

    # Si hay suficientes foreground, limitamos bg.
    # Si NO hay suficientes foreground, se rellena con bg necesariamente.
    take_fg = min(idx_fg.size, N - min(max_bg, idx_bg.size))
    take_bg = N - take_fg

    # Asegurar que no pidamos más bg de lo disponible salvo que luego se permita replace.
    if idx_bg.size > 0:
        take_bg = min(take_bg, N)
    else:
        take_bg = 0
        take_fg = N

    return int(take_bg), int(take_fg)


def select_with_background_control_random(
    labels: np.ndarray,
    N: int,
    cap_bg_frac: Optional[float],
    bg_id: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Selección aleatoria con control de background.
    Devuelve índices de largo exacto N.
    """
    labels = np.asarray(labels).reshape(-1).astype(np.int32, copy=False)
    N = int(N)

    idx_all, idx_bg, idx_fg = split_bg_fg_indices(labels, bg_id=bg_id)

    if idx_all.size == 0:
        raise ValueError("No hay puntos para samplear.")

    if cap_bg_frac is None:
        return rng.choice(idx_all, size=N, replace=(idx_all.size < N)).astype(np.int64)

    max_bg = int(round(float(cap_bg_frac) * float(N)))
    max_bg = max(0, min(N, max_bg))

    # Tomamos bg hasta max_bg, pero si no hay suficientes fg, permitimos más bg.
    desired_fg = N - max_bg

    take_fg = min(idx_fg.size, desired_fg)
    take_bg = N - take_fg

    if idx_bg.size == 0:
        # no hay bg, todo desde fg
        return rng.choice(idx_fg, size=N, replace=(idx_fg.size < N)).astype(np.int64)

    if idx_fg.size == 0:
        # no hay fg, todo bg
        return rng.choice(idx_bg, size=N, replace=(idx_bg.size < N)).astype(np.int64)

    sel_fg = rng.choice(idx_fg, size=take_fg, replace=(idx_fg.size < take_fg)).astype(np.int64)
    sel_bg = rng.choice(idx_bg, size=take_bg, replace=(idx_bg.size < take_bg)).astype(np.int64)

    sel = np.concatenate([sel_bg, sel_fg], axis=0)
    sel = shuffle_indices(sel, rng)

    if sel.shape[0] != N:
        sel = fill_to_n_from_pool(sel, idx_all, N, rng=rng, replace_if_needed=True)

    return sel.astype(np.int64, copy=False)


# ============================================================
# FPS
# ============================================================

def farthest_point_sampling_indices(
    points: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
    start_idx: Optional[int] = None,
) -> np.ndarray:
    """
    Farthest Point Sampling puro sobre points.

    points:
      [M,3]

    Retorna:
      índices locales dentro de points, shape [n_samples]

    Nota:
    - O(M * n_samples), puede ser costoso.
    - Para M=200k y n_samples=8192 puede tardar, pero es manejable si no se abusa.
    """
    points = sanitize_points(points)
    M = int(points.shape[0])
    n_samples = int(n_samples)

    if M == 0:
        raise ValueError("FPS recibió nube vacía.")

    if n_samples <= 0:
        return np.empty((0,), dtype=np.int64)

    if M <= n_samples:
        base = np.arange(M, dtype=np.int64)
        if M < n_samples:
            extra = rng.choice(base, size=n_samples - M, replace=True)
            base = np.concatenate([base, extra.astype(np.int64)], axis=0)
        return base.astype(np.int64, copy=False)

    selected = np.empty((n_samples,), dtype=np.int64)
    distances = np.full((M,), np.inf, dtype=np.float32)

    if start_idx is None:
        farthest = int(rng.integers(0, M))
    else:
        farthest = int(start_idx) % M

    for i in range(n_samples):
        selected[i] = farthest

        centroid = points[farthest]
        diff = points - centroid[None, :]
        dist = np.einsum("ij,ij->i", diff, diff).astype(np.float32, copy=False)

        distances = np.minimum(distances, dist)
        farthest = int(np.argmax(distances))

    return selected.astype(np.int64, copy=False)


def fps_on_global_indices(
    points: np.ndarray,
    candidate_indices: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Aplica FPS sobre un subconjunto dado por candidate_indices,
    y devuelve índices globales respecto a la nube original.
    """
    points = sanitize_points(points)
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64).reshape(-1)

    if candidate_indices.size == 0:
        raise ValueError("fps_on_global_indices recibió candidate_indices vacío.")

    local_points = points[candidate_indices]
    local_sel = farthest_point_sampling_indices(local_points, int(n_samples), rng=rng)
    global_sel = candidate_indices[local_sel]

    return global_sel.astype(np.int64, copy=False)


def sample_fps_indices(
    points: np.ndarray,
    labels: np.ndarray,
    N: int,
    cap_bg_frac: Optional[float],
    bg_id: int,
    rng: np.random.Generator,
    fps_pool_multiplier: float = 4.0,
) -> np.ndarray:
    """
    FPS con soporte de control de background.

    Modo sin cap_bg:
      - FPS global sobre todos los puntos.

    Modo con cap_bg:
      - Calcula cuota bg/fg.
      - Aplica FPS por separado en bg y fg.
      - Junta y baraja.
      - Si falta alguna clase por escasez, rellena desde pool general.

    Esto permite que FPS también respete el control de fondo.
    """
    points = sanitize_points(points)
    labels = sanitize_labels(labels, points.shape[0])
    N = int(N)

    idx_all, idx_bg, idx_fg = split_bg_fg_indices(labels, bg_id=bg_id)

    if cap_bg_frac is None:
        return fps_on_global_indices(points, idx_all, N, rng=rng)

    max_bg = int(round(float(cap_bg_frac) * float(N)))
    max_bg = max(0, min(N, max_bg))

    desired_fg = N - max_bg

    # Si existen ambas clases, respetar cuotas.
    if idx_bg.size > 0 and idx_fg.size > 0:
        take_fg = min(idx_fg.size, desired_fg)
        take_bg = N - take_fg

        # si bg no alcanza, compensar con fg
        if take_bg > idx_bg.size:
            missing_bg = take_bg - idx_bg.size
            take_bg = idx_bg.size
            take_fg = min(N - take_bg, idx_fg.size)
            # si aún falta, se completará con replace al final

        sel_parts = []

        if take_bg > 0:
            sel_bg = fps_on_global_indices(points, idx_bg, take_bg, rng=rng)
            sel_parts.append(sel_bg)

        if take_fg > 0:
            sel_fg = fps_on_global_indices(points, idx_fg, take_fg, rng=rng)
            sel_parts.append(sel_fg)

        if sel_parts:
            sel = np.concatenate(sel_parts, axis=0)
        else:
            sel = np.empty((0,), dtype=np.int64)

        sel = fill_to_n_from_pool(sel, idx_all, N, rng=rng, replace_if_needed=True)
        sel = shuffle_indices(sel, rng)

        return sel.astype(np.int64, copy=False)

    # Fallback si solo hay bg o solo fg.
    return fps_on_global_indices(points, idx_all, N, rng=rng)


# ============================================================
# CURVATURE APPROXIMATION
# ============================================================

def estimate_local_curvature_scores(
    points: np.ndarray,
    candidate_indices: Optional[np.ndarray] = None,
    k: int = 16,
    max_candidates: int = 50000,
    rng: Optional[np.random.Generator] = None,
    chunk_size: int = 1024,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estima una puntuación simple de curvatura/local irregularity.

    En vez de calcular PCA completa por punto, se usa una aproximación rápida:
      score_i = media de distancias cuadradas a vecinos kNN
              + varianza local de distancias

    Esto favorece regiones de cambio local, bordes y zonas menos planas.

    Retorna:
      used_indices: índices globales usados para estimar score
      scores: score por índice usado

    Nota:
    - Es una aproximación práctica para sampleo, no una curvatura diferencial exacta.
    - Para odontología suele funcionar como proxy de surcos/bordes.
    """
    if rng is None:
        rng = np_rng(42)

    points = sanitize_points(points)

    N_total = int(points.shape[0])

    if candidate_indices is None:
        candidate_indices = np.arange(N_total, dtype=np.int64)
    else:
        candidate_indices = np.asarray(candidate_indices, dtype=np.int64).reshape(-1)

    if candidate_indices.size == 0:
        return candidate_indices, np.empty((0,), dtype=np.float32)

    # Para evitar coste excesivo, si hay demasiados candidatos se submuestrea
    # el conjunto donde se estima la curvatura.
    if candidate_indices.size > int(max_candidates):
        used_indices = rng.choice(candidate_indices, size=int(max_candidates), replace=False).astype(np.int64)
    else:
        used_indices = candidate_indices.astype(np.int64, copy=False)

    # kNN de used_indices contra points completos
    kk = int(max(4, min(int(k), N_total)))

    idx_knn = knn_indices_numpy(
        points=points,
        k=kk,
        query_indices=used_indices,
        chunk_size=int(chunk_size),
    )

    q = points[used_indices]                         # [M,3]
    neigh = points[idx_knn]                           # [M,k,3]
    diff = neigh - q[:, None, :]
    d2 = np.sum(diff * diff, axis=2).astype(np.float32)

    # ignorar vecino propio si aparece con distancia 0, usando orden estadístico simple
    mean_d2 = d2.mean(axis=1)
    var_d2 = d2.var(axis=1)

    scores = mean_d2 + var_d2
    scores = scores.astype(np.float32, copy=False)

    # normalización robusta 0..1
    if scores.size > 0:
        lo = float(np.percentile(scores, 1))
        hi = float(np.percentile(scores, 99))
        if hi > lo:
            scores = np.clip((scores - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
        else:
            scores = np.zeros_like(scores, dtype=np.float32)

    return used_indices.astype(np.int64, copy=False), scores.astype(np.float32, copy=False)


def select_top_score_indices(
    indices: np.ndarray,
    scores: np.ndarray,
    n_select: int,
    rng: np.random.Generator,
    temperature: float = 0.25,
    deterministic_top: bool = False,
) -> np.ndarray:
    """
    Selecciona puntos de alta puntuación.

    deterministic_top=True:
      - toma top n_select.

    deterministic_top=False:
      - samplea ponderado por score para evitar tomar solo ruido extremo.
    """
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    n_select = int(n_select)

    if n_select <= 0:
        return np.empty((0,), dtype=np.int64)

    if indices.size == 0:
        return np.empty((0,), dtype=np.int64)

    if indices.size <= n_select:
        return indices.astype(np.int64, copy=False)

    if deterministic_top:
        order = np.argsort(scores)[::-1]
        return indices[order[:n_select]].astype(np.int64, copy=False)

    # ponderado por score suavizado
    s = scores.copy()
    s = np.maximum(s, 0.0)
    if float(s.sum()) <= 1e-12:
        return rng.choice(indices, size=n_select, replace=False).astype(np.int64)

    # temperatura baja hace más fuerte la preferencia por bordes
    temp = max(1e-6, float(temperature))
    w = np.power(s + 1e-6, 1.0 / temp)
    w = w / (w.sum() + 1e-12)

    return rng.choice(indices, size=n_select, replace=False, p=w).astype(np.int64)


def sample_fps_curvature_indices(
    points: np.ndarray,
    labels: np.ndarray,
    N: int,
    cap_bg_frac: Optional[float],
    bg_id: int,
    rng: np.random.Generator,
    curvature_ratio: float = 0.35,
    curvature_k: int = 16,
    curvature_max_candidates: int = 50000,
    curvature_temperature: float = 0.35,
    curvature_deterministic_top: bool = False,
) -> np.ndarray:
    """
    Sampleo híbrido FPS + Curvatura con control de background.

    Estrategia:
      - Selecciona una fracción por curvatura.
      - Completa el resto con FPS.
      - Si cap_bg está activo, aplica cuotas bg/fg también en este modo.

    Ejemplo:
      N=8192
      curvature_ratio=0.35
      -> ~2867 puntos favorecidos por curvatura
      -> ~5325 puntos por FPS/cobertura
    """
    points = sanitize_points(points)
    labels = sanitize_labels(labels, points.shape[0])
    N = int(N)

    idx_all, idx_bg, idx_fg = split_bg_fg_indices(labels, bg_id=bg_id)

    curvature_ratio = float(np.clip(float(curvature_ratio), 0.0, 0.95))
    n_curv_total = int(round(float(N) * curvature_ratio))
    n_fps_total = int(N) - int(n_curv_total)

    if cap_bg_frac is None:
        # Curvatura global
        used_idx, scores = estimate_local_curvature_scores(
            points=points,
            candidate_indices=idx_all,
            k=int(curvature_k),
            max_candidates=int(curvature_max_candidates),
            rng=rng,
        )

        sel_curv = select_top_score_indices(
            used_idx,
            scores,
            n_select=n_curv_total,
            rng=rng,
            temperature=float(curvature_temperature),
            deterministic_top=bool(curvature_deterministic_top),
        )

        # FPS para completar evitando duplicados
        remaining_pool = np.asarray(
            [int(i) for i in idx_all.tolist() if int(i) not in set(sel_curv.tolist())],
            dtype=np.int64,
        )

        if remaining_pool.size > 0 and n_fps_total > 0:
            sel_fps = fps_on_global_indices(points, remaining_pool, n_fps_total, rng=rng)
        else:
            sel_fps = np.empty((0,), dtype=np.int64)

        sel = np.concatenate([sel_curv, sel_fps], axis=0)
        sel = fill_to_n_from_pool(sel, idx_all, N, rng=rng, replace_if_needed=True)
        sel = shuffle_indices(sel, rng)
        return sel.astype(np.int64, copy=False)

    # Con control de background.
    max_bg = int(round(float(cap_bg_frac) * float(N)))
    max_bg = max(0, min(N, max_bg))
    target_fg = N - max_bg

    if idx_bg.size == 0 or idx_fg.size == 0:
        # Si no hay ambas clases, fallback global.
        return sample_fps_indices(
            points=points,
            labels=labels,
            N=N,
            cap_bg_frac=None,
            bg_id=bg_id,
            rng=rng,
        )

    # Cuotas totales por bg/fg
    take_fg = min(idx_fg.size, target_fg)
    take_bg = N - take_fg

    # Dividir cada cuota en curvatura + FPS
    n_curv_bg = int(round(take_bg * curvature_ratio))
    n_curv_fg = int(round(take_fg * curvature_ratio))

    n_fps_bg = take_bg - n_curv_bg
    n_fps_fg = take_fg - n_curv_fg

    selected_parts = []

    # BG curvature + FPS
    if take_bg > 0 and idx_bg.size > 0:
        used_bg, scores_bg = estimate_local_curvature_scores(
            points=points,
            candidate_indices=idx_bg,
            k=int(curvature_k),
            max_candidates=min(int(curvature_max_candidates), max(1000, idx_bg.size)),
            rng=rng,
        )

        sel_bg_curv = select_top_score_indices(
            used_bg,
            scores_bg,
            n_select=n_curv_bg,
            rng=rng,
            temperature=float(curvature_temperature),
            deterministic_top=bool(curvature_deterministic_top),
        )

        bg_used_set = set(int(v) for v in sel_bg_curv.tolist())
        bg_remaining = np.asarray([int(i) for i in idx_bg.tolist() if int(i) not in bg_used_set], dtype=np.int64)

        if n_fps_bg > 0 and bg_remaining.size > 0:
            sel_bg_fps = fps_on_global_indices(points, bg_remaining, n_fps_bg, rng=rng)
        else:
            sel_bg_fps = np.empty((0,), dtype=np.int64)

        sel_bg = np.concatenate([sel_bg_curv, sel_bg_fps], axis=0)
        sel_bg = fill_to_n_from_pool(sel_bg, idx_bg, take_bg, rng=rng, replace_if_needed=True)
        selected_parts.append(sel_bg)

    # FG curvature + FPS
    if take_fg > 0 and idx_fg.size > 0:
        used_fg, scores_fg = estimate_local_curvature_scores(
            points=points,
            candidate_indices=idx_fg,
            k=int(curvature_k),
            max_candidates=min(int(curvature_max_candidates), max(1000, idx_fg.size)),
            rng=rng,
        )

        sel_fg_curv = select_top_score_indices(
            used_fg,
            scores_fg,
            n_select=n_curv_fg,
            rng=rng,
            temperature=float(curvature_temperature),
            deterministic_top=bool(curvature_deterministic_top),
        )

        fg_used_set = set(int(v) for v in sel_fg_curv.tolist())
        fg_remaining = np.asarray([int(i) for i in idx_fg.tolist() if int(i) not in fg_used_set], dtype=np.int64)

        if n_fps_fg > 0 and fg_remaining.size > 0:
            sel_fg_fps = fps_on_global_indices(points, fg_remaining, n_fps_fg, rng=rng)
        else:
            sel_fg_fps = np.empty((0,), dtype=np.int64)

        sel_fg = np.concatenate([sel_fg_curv, sel_fg_fps], axis=0)
        sel_fg = fill_to_n_from_pool(sel_fg, idx_fg, take_fg, rng=rng, replace_if_needed=True)
        selected_parts.append(sel_fg)

    if selected_parts:
        sel = np.concatenate(selected_parts, axis=0)
    else:
        sel = np.empty((0,), dtype=np.int64)

    sel = fill_to_n_from_pool(sel, idx_all, N, rng=rng, replace_if_needed=True)
    sel = shuffle_indices(sel, rng)

    return sel.astype(np.int64, copy=False)


# ============================================================
# BOUNDARY-AWARE LABEL SAMPLING
# ============================================================

def detect_label_boundary_indices(
    points: np.ndarray,
    labels: np.ndarray,
    k: int = 16,
    max_candidates: int = 60000,
    rng: Optional[np.random.Generator] = None,
    chunk_size: int = 1024,
) -> np.ndarray:
    """
    Detecta puntos cercanos a bordes usando labels.

    Un punto se considera boundary si entre sus vecinos kNN existe al menos
    una etiqueta distinta a la suya.

    Esto captura:
    - límites diente-encía
    - límites interdentales
    - bordes de diente 21 contra vecinos

    Nota:
    - Usa labels, por tanto es una estrategia supervisada.
    """
    if rng is None:
        rng = np_rng(42)

    points = sanitize_points(points)
    labels = sanitize_labels(labels, points.shape[0])

    N_total = int(points.shape[0])
    idx_all = np.arange(N_total, dtype=np.int64)

    if N_total == 0:
        return np.empty((0,), dtype=np.int64)

    if N_total > int(max_candidates):
        query_indices = rng.choice(idx_all, size=int(max_candidates), replace=False).astype(np.int64)
    else:
        query_indices = idx_all

    kk = int(max(4, min(int(k), N_total)))

    idx_knn = knn_indices_numpy(
        points=points,
        k=kk,
        query_indices=query_indices,
        chunk_size=int(chunk_size),
    )

    q_labels = labels[query_indices]             # [M]
    neigh_labels = labels[idx_knn]               # [M,k]

    is_boundary = np.any(neigh_labels != q_labels[:, None], axis=1)

    boundary_idx = query_indices[is_boundary]

    return boundary_idx.astype(np.int64, copy=False)


def sample_boundary_labels_indices(
    points: np.ndarray,
    labels: np.ndarray,
    N: int,
    cap_bg_frac: Optional[float],
    bg_id: int,
    rng: np.random.Generator,
    boundary_ratio: float = 0.45,
    boundary_k: int = 16,
    boundary_max_candidates: int = 60000,
    boundary_fill_method: str = "fps",
) -> np.ndarray:
    """
    Boundary-aware sampling con control de background.

    Estrategia:
      - Detecta puntos boundary por kNN + cambio de label.
      - Toma una fracción boundary_ratio.
      - Completa con FPS o random.
      - Si cap_bg está activo, respeta cuota de background.

    boundary_fill_method:
      - fps
      - random
    """
    points = sanitize_points(points)
    labels = sanitize_labels(labels, points.shape[0])
    N = int(N)

    idx_all, idx_bg, idx_fg = split_bg_fg_indices(labels, bg_id=bg_id)

    boundary_ratio = float(np.clip(float(boundary_ratio), 0.0, 0.95))
    n_boundary_total = int(round(float(N) * boundary_ratio))

    boundary_idx = detect_label_boundary_indices(
        points=points,
        labels=labels,
        k=int(boundary_k),
        max_candidates=int(boundary_max_candidates),
        rng=rng,
    )

    if boundary_idx.size == 0:
        # fallback si no se detectan bordes
        return sample_fps_indices(
            points=points,
            labels=labels,
            N=N,
            cap_bg_frac=cap_bg_frac,
            bg_id=bg_id,
            rng=rng,
        )

    boundary_set = set(int(v) for v in boundary_idx.tolist())

    if cap_bg_frac is None:
        # Tomar boundary global
        n_boundary = min(n_boundary_total, boundary_idx.size)
        sel_boundary = rng.choice(boundary_idx, size=n_boundary, replace=(boundary_idx.size < n_boundary)).astype(np.int64)

        remaining_pool = np.asarray([int(i) for i in idx_all.tolist() if int(i) not in boundary_set], dtype=np.int64)
        n_rest = N - sel_boundary.shape[0]

        if n_rest > 0:
            if str(boundary_fill_method).lower() == "random":
                sel_rest = rng.choice(remaining_pool if remaining_pool.size else idx_all, size=n_rest, replace=((remaining_pool.size if remaining_pool.size else idx_all.size) < n_rest)).astype(np.int64)
            else:
                pool = remaining_pool if remaining_pool.size else idx_all
                sel_rest = fps_on_global_indices(points, pool, n_rest, rng=rng)
        else:
            sel_rest = np.empty((0,), dtype=np.int64)

        sel = np.concatenate([sel_boundary, sel_rest], axis=0)
        sel = fill_to_n_from_pool(sel, idx_all, N, rng=rng, replace_if_needed=True)
        sel = shuffle_indices(sel, rng)
        return sel.astype(np.int64, copy=False)

    # Con control de background.
    max_bg = int(round(float(cap_bg_frac) * float(N)))
    max_bg = max(0, min(N, max_bg))
    target_fg = N - max_bg

    if idx_bg.size == 0 or idx_fg.size == 0:
        return sample_boundary_labels_indices(
            points=points,
            labels=labels,
            N=N,
            cap_bg_frac=None,
            bg_id=bg_id,
            rng=rng,
            boundary_ratio=boundary_ratio,
            boundary_k=boundary_k,
            boundary_max_candidates=boundary_max_candidates,
            boundary_fill_method=boundary_fill_method,
        )

    boundary_bg = np.asarray([i for i in boundary_idx.tolist() if labels[int(i)] == int(bg_id)], dtype=np.int64)
    boundary_fg = np.asarray([i for i in boundary_idx.tolist() if labels[int(i)] != int(bg_id)], dtype=np.int64)

    take_fg = min(idx_fg.size, target_fg)
    take_bg = N - take_fg

    n_boundary_bg = int(round(take_bg * boundary_ratio))
    n_boundary_fg = int(round(take_fg * boundary_ratio))

    selected_parts = []

    # BG
    if take_bg > 0:
        if boundary_bg.size > 0 and n_boundary_bg > 0:
            sel_bg_boundary = rng.choice(
                boundary_bg,
                size=min(n_boundary_bg, take_bg),
                replace=(boundary_bg.size < min(n_boundary_bg, take_bg)),
            ).astype(np.int64)
        else:
            sel_bg_boundary = np.empty((0,), dtype=np.int64)

        bg_used = set(int(v) for v in sel_bg_boundary.tolist())
        bg_remaining = np.asarray([int(i) for i in idx_bg.tolist() if int(i) not in bg_used], dtype=np.int64)
        n_bg_rest = take_bg - sel_bg_boundary.shape[0]

        if n_bg_rest > 0:
            if str(boundary_fill_method).lower() == "random":
                pool = bg_remaining if bg_remaining.size else idx_bg
                sel_bg_rest = rng.choice(pool, size=n_bg_rest, replace=(pool.size < n_bg_rest)).astype(np.int64)
            else:
                pool = bg_remaining if bg_remaining.size else idx_bg
                sel_bg_rest = fps_on_global_indices(points, pool, n_bg_rest, rng=rng)
        else:
            sel_bg_rest = np.empty((0,), dtype=np.int64)

        sel_bg = np.concatenate([sel_bg_boundary, sel_bg_rest], axis=0)
        sel_bg = fill_to_n_from_pool(sel_bg, idx_bg, take_bg, rng=rng, replace_if_needed=True)
        selected_parts.append(sel_bg)

    # FG
    if take_fg > 0:
        if boundary_fg.size > 0 and n_boundary_fg > 0:
            sel_fg_boundary = rng.choice(
                boundary_fg,
                size=min(n_boundary_fg, take_fg),
                replace=(boundary_fg.size < min(n_boundary_fg, take_fg)),
            ).astype(np.int64)
        else:
            sel_fg_boundary = np.empty((0,), dtype=np.int64)

        fg_used = set(int(v) for v in sel_fg_boundary.tolist())
        fg_remaining = np.asarray([int(i) for i in idx_fg.tolist() if int(i) not in fg_used], dtype=np.int64)
        n_fg_rest = take_fg - sel_fg_boundary.shape[0]

        if n_fg_rest > 0:
            if str(boundary_fill_method).lower() == "random":
                pool = fg_remaining if fg_remaining.size else idx_fg
                sel_fg_rest = rng.choice(pool, size=n_fg_rest, replace=(pool.size < n_fg_rest)).astype(np.int64)
            else:
                pool = fg_remaining if fg_remaining.size else idx_fg
                sel_fg_rest = fps_on_global_indices(points, pool, n_fg_rest, rng=rng)
        else:
            sel_fg_rest = np.empty((0,), dtype=np.int64)

        sel_fg = np.concatenate([sel_fg_boundary, sel_fg_rest], axis=0)
        sel_fg = fill_to_n_from_pool(sel_fg, idx_fg, take_fg, rng=rng, replace_if_needed=True)
        selected_parts.append(sel_fg)

    if selected_parts:
        sel = np.concatenate(selected_parts, axis=0)
    else:
        sel = np.empty((0,), dtype=np.int64)

    sel = fill_to_n_from_pool(sel, idx_all, N, rng=rng, replace_if_needed=True)
    sel = shuffle_indices(sel, rng)

    return sel.astype(np.int64, copy=False)


# ============================================================
# UNIFIED SAMPLER
# ============================================================

def sample_indices_by_method(
    points: np.ndarray,
    labels: np.ndarray,
    N: int,
    method: str,
    cap_bg_frac: Optional[float],
    bg_id: int,
    rng: np.random.Generator,
    fps_pool_multiplier: float = 4.0,
    curvature_ratio: float = 0.35,
    curvature_k: int = 16,
    curvature_max_candidates: int = 50000,
    curvature_temperature: float = 0.35,
    curvature_deterministic_top: bool = False,
    boundary_ratio: float = 0.45,
    boundary_k: int = 16,
    boundary_max_candidates: int = 60000,
    boundary_fill_method: str = "fps",
) -> np.ndarray:
    """
    Interfaz única de sampleo.
    """
    method = str(method).lower().strip()

    points = sanitize_points(points)
    labels = sanitize_labels(labels, points.shape[0])
    N = int(N)

    if method in ("random", "baseline", "traditional", "tradicional"):
        return select_with_background_control_random(
            labels=labels,
            N=N,
            cap_bg_frac=cap_bg_frac,
            bg_id=bg_id,
            rng=rng,
        )

    if method in ("fps", "farthest", "farthest_point_sampling"):
        return sample_fps_indices(
            points=points,
            labels=labels,
            N=N,
            cap_bg_frac=cap_bg_frac,
            bg_id=bg_id,
            rng=rng,
            fps_pool_multiplier=float(fps_pool_multiplier),
        )

    if method in ("fps_curvature", "fps+curvature", "hybrid_fps_curvature", "curvature_fps"):
        return sample_fps_curvature_indices(
            points=points,
            labels=labels,
            N=N,
            cap_bg_frac=cap_bg_frac,
            bg_id=bg_id,
            rng=rng,
            curvature_ratio=float(curvature_ratio),
            curvature_k=int(curvature_k),
            curvature_max_candidates=int(curvature_max_candidates),
            curvature_temperature=float(curvature_temperature),
            curvature_deterministic_top=bool(curvature_deterministic_top),
        )

    if method in ("boundary_labels", "boundary", "label_boundary", "boundary_aware"):
        return sample_boundary_labels_indices(
            points=points,
            labels=labels,
            N=N,
            cap_bg_frac=cap_bg_frac,
            bg_id=bg_id,
            rng=rng,
            boundary_ratio=float(boundary_ratio),
            boundary_k=int(boundary_k),
            boundary_max_candidates=int(boundary_max_candidates),
            boundary_fill_method=str(boundary_fill_method),
        )

    raise ValueError(
        f"Método de sampleo no reconocido: {method}. "
        "Usa: random, fps, fps_curvature, boundary_labels."
    )


def sample_points_labels_by_method(
    points: np.ndarray,
    labels: np.ndarray,
    N: int,
    method: str,
    cap_bg_frac: Optional[float],
    bg_id: int,
    rng: np.random.Generator,
    fps_pool_multiplier: float = 4.0,
    curvature_ratio: float = 0.35,
    curvature_k: int = 16,
    curvature_max_candidates: int = 50000,
    curvature_temperature: float = 0.35,
    curvature_deterministic_top: bool = False,
    boundary_ratio: float = 0.45,
    boundary_k: int = 16,
    boundary_max_candidates: int = 60000,
    boundary_fill_method: str = "fps",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Devuelve points_sampled, labels_sampled usando el método indicado.
    """
    points = sanitize_points(points)
    labels = sanitize_labels(labels, points.shape[0])

    idx = sample_indices_by_method(
        points=points,
        labels=labels,
        N=int(N),
        method=str(method),
        cap_bg_frac=cap_bg_frac,
        bg_id=int(bg_id),
        rng=rng,
        fps_pool_multiplier=float(fps_pool_multiplier),
        curvature_ratio=float(curvature_ratio),
        curvature_k=int(curvature_k),
        curvature_max_candidates=int(curvature_max_candidates),
        curvature_temperature=float(curvature_temperature),
        curvature_deterministic_top=bool(curvature_deterministic_top),
        boundary_ratio=float(boundary_ratio),
        boundary_k=int(boundary_k),
        boundary_max_candidates=int(boundary_max_candidates),
        boundary_fill_method=str(boundary_fill_method),
    )

    if idx.shape[0] != int(N):
        idx = fill_to_n_from_pool(
            selected=idx,
            pool=np.arange(points.shape[0], dtype=np.int64),
            n_target=int(N),
            rng=rng,
            replace_if_needed=True,
        )

    return (
        points[idx].astype(np.float32, copy=False),
        labels[idx].astype(np.int32, copy=False),
    )


# ============================================================
# COVERAGE SAFETY
# ============================================================

def ensure_coverage_min_moves(
    splits: Dict[str, Tuple[np.ndarray, np.ndarray]],
    rng: np.random.Generator,
) -> Dict[str, int]:
    """
    Asegura que val y test tengan todas las clases presentes globalmente
    si existen muestras en train que contienen esas clases.

    Mantiene el comportamiento del script v3 original:
    mueve muestras mínimas desde train hacia val/test si faltan clases.

    Nota:
    - Esto puede cambiar tamaños de split si falta una clase.
    - En tu pipeline actual normalmente reportó 0 movimientos.
    """
    global_classes = np.unique(
        np.concatenate(
            [
                np.unique(splits["train"][1]),
                np.unique(splits["val"][1]),
                np.unique(splits["test"][1]),
            ]
        )
    )

    moves = {"val": 0, "test": 0}

    for target in ["val", "test"]:
        Xt, Yt = splits[target]
        missing = [int(c) for c in global_classes if int(c) not in set(np.unique(Yt).astype(int).tolist())]

        if not missing:
            continue

        for cls in missing:
            donor = "train"
            Xd, Yd = splits[donor]

            cand = np.where(np.any(Yd == int(cls), axis=1))[0]

            if cand.size == 0:
                continue

            pick = int(rng.choice(cand))

            Xt = np.concatenate([Xt, Xd[pick:pick + 1]], axis=0)
            Yt = np.concatenate([Yt, Yd[pick:pick + 1]], axis=0)

            Xd = np.delete(Xd, pick, axis=0)
            Yd = np.delete(Yd, pick, axis=0)

            splits[donor] = (Xd, Yd)
            splits[target] = (Xt, Yt)

            moves[target] += 1

    return moves


# ============================================================
# SPLIT PROCESSING
# ============================================================

def process_split(
    X: np.ndarray,
    Y: np.ndarray,
    split_name: str,
    N: int,
    method: str,
    cap_bg_frac: Optional[float],
    bg_id: int,
    seed: int,
    fps_pool_multiplier: float,
    curvature_ratio: float,
    curvature_k: int,
    curvature_max_candidates: int,
    curvature_temperature: float,
    curvature_deterministic_top: bool,
    boundary_ratio: float,
    boundary_k: int,
    boundary_max_candidates: int,
    boundary_fill_method: str,
    use_safe_resume: bool = False,
    resume_cache_dir: Optional[Path] = None,
    overwrite_resume_cache: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Procesa un split completo X/Y sampleando cada muestra a N puntos.

    Safe resume:
    - Si use_safe_resume=True, cada muestra se guarda en cache independiente.
    - Si el proceso cae, al relanzar con los mismos argumentos reutiliza muestras completas.
    - El orden final del split NO cambia: se reconstruye con stack en el mismo i original.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)

    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"{split_name}: X e Y tienen distinto número de muestras.")

    out_X = []
    out_Y = []

    cache_enabled = bool(use_safe_resume) and resume_cache_dir is not None
    cache_dir = Path(resume_cache_dir) if cache_enabled else None

    if cache_enabled:
        ensure_dir(cache_dir / str(split_name) / "original")
        print(f"[RESUME] {split_name}: cache original en {cache_dir / str(split_name) / 'original'}")

    for i in iter_progress(range(X.shape[0]), desc=f"Procesando {split_name} ({method})"):
        # Semilla por muestra para reproducibilidad estable.
        local_seed = int(seed) + 100000 * split_seed_offset(split_name) + int(i)

        paths = None
        if cache_enabled:
            paths = resume_sample_paths(
                cache_dir=cache_dir,
                split_name=split_name,
                stage_name="original",
                sample_idx=int(i),
            )

            if (not bool(overwrite_resume_cache)) and is_valid_cached_sample(
                paths["x"],
                paths["y"],
                paths["done"],
                N=int(N),
            ):
                xs, ys = load_cached_sample(paths["x"], paths["y"])
                out_X.append(xs)
                out_Y.append(ys)
                continue

        rng = np_rng(local_seed)

        pts = sanitize_points(X[i])
        lab = sanitize_labels(Y[i], pts.shape[0])

        xs, ys = sample_points_labels_by_method(
            points=pts,
            labels=lab,
            N=int(N),
            method=str(method),
            cap_bg_frac=cap_bg_frac,
            bg_id=int(bg_id),
            rng=rng,
            fps_pool_multiplier=float(fps_pool_multiplier),
            curvature_ratio=float(curvature_ratio),
            curvature_k=int(curvature_k),
            curvature_max_candidates=int(curvature_max_candidates),
            curvature_temperature=float(curvature_temperature),
            curvature_deterministic_top=bool(curvature_deterministic_top),
            boundary_ratio=float(boundary_ratio),
            boundary_k=int(boundary_k),
            boundary_max_candidates=int(boundary_max_candidates),
            boundary_fill_method=str(boundary_fill_method),
        )

        if cache_enabled and paths is not None:
            save_cached_sample(
                x_path=paths["x"],
                y_path=paths["y"],
                done_path=paths["done"],
                x=xs,
                y=ys,
                meta={
                    "split": str(split_name),
                    "stage": "original",
                    "sample_idx": int(i),
                    "seed": int(local_seed),
                    "N": int(N),
                    "method": str(method),
                    "x_shape": list(xs.shape),
                    "y_shape": list(ys.shape),
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            )

        out_X.append(xs)
        out_Y.append(ys)

    Xo = np.stack(out_X, axis=0).astype(np.float32, copy=False)
    Yo = np.stack(out_Y, axis=0).astype(np.int32, copy=False)

    return Xo, Yo

def split_seed_offset(split_name: str) -> int:
    """
    Offset determinístico por split.
    """
    split_name = str(split_name).lower().strip()
    if split_name == "train":
        return 1
    if split_name == "val":
        return 2
    if split_name == "test":
        return 3
    return 9


def augment_and_resample_train(
    X_train_full: np.ndarray,
    Y_train_full: np.ndarray,
    N: int,
    method: str,
    cap_bg_frac: Optional[float],
    bg_id: int,
    seed: int,
    augment_times: int,
    rotate_deg: float,
    jitter_sigma: float,
    jitter_clip: float,
    scale_min: float,
    scale_max: float,
    dropout_rate: float,
    fps_pool_multiplier: float,
    curvature_ratio: float,
    curvature_k: int,
    curvature_max_candidates: int,
    curvature_temperature: float,
    curvature_deterministic_top: bool,
    boundary_ratio: float,
    boundary_k: int,
    boundary_max_candidates: int,
    boundary_fill_method: str,
    use_safe_resume: bool = False,
    resume_cache_dir: Optional[Path] = None,
    overwrite_resume_cache: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera muestras aumentadas para train.

    Orden de salida:
      bloque augment 1 para todas las muestras
      bloque augment 2 para todas las muestras
      ...

    Esto calza con create_traceability_indices_for_fixed_split.py:
      originales primero, luego aug_id=1, aug_id=2, etc.

    Safe resume:
    - Cada muestra aumentada queda cacheada en:
        _resume_cache/train/augmentation_augXXX/X_IIIIII.npy
        _resume_cache/train/augmentation_augXXX/Y_IIIIII.npy
    - La reconstrucción final preserva exactamente el mismo orden lógico.
    """
    X_train_full = np.asarray(X_train_full)
    Y_train_full = np.asarray(Y_train_full)

    aug_X = []
    aug_Y = []

    augment_times = int(augment_times)

    if augment_times <= 0:
        return (
            np.empty((0, int(N), 3), dtype=np.float32),
            np.empty((0, int(N)), dtype=np.int32),
        )

    cache_enabled = bool(use_safe_resume) and resume_cache_dir is not None
    cache_dir = Path(resume_cache_dir) if cache_enabled else None

    if cache_enabled:
        ensure_dir(cache_dir / "train")
        print(f"[RESUME] train augmentation: cache en {cache_dir / 'train'}")

    total = X_train_full.shape[0] * augment_times

    iterator = range(total)
    if HAS_TQDM:
        iterator = tqdm(iterator, desc=f"AUG train ({method})")

    for t in iterator:
        aug_id = int(t // X_train_full.shape[0]) + 1
        i = int(t % X_train_full.shape[0])

        local_seed = int(seed) + 900000 + aug_id * 100000 + int(i)

        paths = None
        if cache_enabled:
            paths = resume_sample_paths(
                cache_dir=cache_dir,
                split_name="train",
                stage_name=f"augmentation_aug{aug_id:03d}",
                sample_idx=int(i),
            )

            if (not bool(overwrite_resume_cache)) and is_valid_cached_sample(
                paths["x"],
                paths["y"],
                paths["done"],
                N=int(N),
            ):
                xs, ys = load_cached_sample(paths["x"], paths["y"])
                aug_X.append(xs)
                aug_Y.append(ys)
                continue

        rng = np_rng(local_seed)

        pts = sanitize_points(X_train_full[i])
        lab = sanitize_labels(Y_train_full[i], pts.shape[0])

        pts_aug, lab_aug = augment_points_labels(
            points=pts,
            labels=lab,
            rotate_deg=float(rotate_deg),
            jitter_sigma=float(jitter_sigma),
            jitter_clip=float(jitter_clip),
            scale_min=float(scale_min),
            scale_max=float(scale_max),
            dropout_rate=float(dropout_rate),
            rng=rng,
        )

        xs, ys = sample_points_labels_by_method(
            points=pts_aug,
            labels=lab_aug,
            N=int(N),
            method=str(method),
            cap_bg_frac=cap_bg_frac,
            bg_id=int(bg_id),
            rng=rng,
            fps_pool_multiplier=float(fps_pool_multiplier),
            curvature_ratio=float(curvature_ratio),
            curvature_k=int(curvature_k),
            curvature_max_candidates=int(curvature_max_candidates),
            curvature_temperature=float(curvature_temperature),
            curvature_deterministic_top=bool(curvature_deterministic_top),
            boundary_ratio=float(boundary_ratio),
            boundary_k=int(boundary_k),
            boundary_max_candidates=int(boundary_max_candidates),
            boundary_fill_method=str(boundary_fill_method),
        )

        if cache_enabled and paths is not None:
            save_cached_sample(
                x_path=paths["x"],
                y_path=paths["y"],
                done_path=paths["done"],
                x=xs,
                y=ys,
                meta={
                    "split": "train",
                    "stage": f"augmentation_aug{aug_id:03d}",
                    "sample_idx": int(i),
                    "aug_id": int(aug_id),
                    "seed": int(local_seed),
                    "N": int(N),
                    "method": str(method),
                    "x_shape": list(xs.shape),
                    "y_shape": list(ys.shape),
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            )

        aug_X.append(xs)
        aug_Y.append(ys)

    Xa = np.stack(aug_X, axis=0).astype(np.float32, copy=False)
    Ya = np.stack(aug_Y, axis=0).astype(np.int32, copy=False)

    return Xa, Ya


# ============================================================
# ARGPARSE
# ============================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Paper-like fixed_split creation with random/FPS/FPS+curvature/"
            "boundary-aware label sampling and independent background control."
        )
    )

    # -------------------------
    # IO
    # -------------------------
    ap.add_argument(
        "--dataset_dir",
        required=True,
        help="Ruta al dataset merged_* con X_train/Y_train/X_val/Y_val/X_test/Y_test.",
    )
    ap.add_argument(
        "--out_dir",
        required=True,
        help="Directorio de salida fixed_split.",
    )

    # -------------------------
    # Sampling general
    # -------------------------
    ap.add_argument(
        "--N",
        type=int,
        default=8192,
        help="Número final de puntos por muestra.",
    )
    ap.add_argument(
        "--sampling_method",
        type=str,
        default="random",
        choices=["random", "fps", "fps_curvature", "boundary_labels"],
        help=(
            "Método de sampleo: random, fps, fps_curvature, boundary_labels."
        ),
    )
    ap.add_argument(
        "--cap_bg",
        type=float,
        default=None,
        help=(
            "Control de background. <=1 se interpreta como fracción; >1 como cantidad absoluta. "
            "Ej: 0.8 permite máximo 80%% de fondo."
        ),
    )
    ap.add_argument(
        "--bg_id",
        type=int,
        default=0,
        help="Clase interna del background.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla global.",
    )

    # -------------------------
    # FPS
    # -------------------------
    ap.add_argument(
        "--fps_pool_multiplier",
        type=float,
        default=4.0,
        help=(
            "Reservado para compatibilidad. En la versión actual FPS opera sobre el pool completo "
            "o sobre cuotas bg/fg completas."
        ),
    )

    # -------------------------
    # FPS + curvature
    # -------------------------
    ap.add_argument(
        "--curvature_ratio",
        type=float,
        default=0.35,
        help="Fracción de puntos favorecidos por curvatura en fps_curvature.",
    )
    ap.add_argument(
        "--curvature_k",
        type=int,
        default=16,
        help="kNN usado para estimar curvatura/local irregularity.",
    )
    ap.add_argument(
        "--curvature_max_candidates",
        type=int,
        default=50000,
        help="Máximo de candidatos donde estimar curvatura por muestra.",
    )
    ap.add_argument(
        "--curvature_temperature",
        type=float,
        default=0.35,
        help="Temperatura para sampleo ponderado por curvatura. Menor = más agresivo.",
    )
    ap.add_argument(
        "--curvature_deterministic_top",
        action="store_true",
        help="Si se activa, toma top curvatura en vez de sampleo ponderado.",
    )

    # -------------------------
    # Boundary-aware labels
    # -------------------------
    ap.add_argument(
        "--boundary_ratio",
        type=float,
        default=0.45,
        help="Fracción de puntos favorecidos por boundary_labels.",
    )
    ap.add_argument(
        "--boundary_k",
        type=int,
        default=16,
        help="kNN usado para detectar boundary por cambio de etiqueta.",
    )
    ap.add_argument(
        "--boundary_max_candidates",
        type=int,
        default=60000,
        help="Máximo de candidatos para detectar boundary por muestra.",
    )
    ap.add_argument(
        "--boundary_fill_method",
        type=str,
        default="fps",
        choices=["fps", "random"],
        help="Método para completar puntos no-boundary.",
    )

    # -------------------------
    # Augmentation
    # -------------------------
    ap.add_argument(
        "--use_augmentation",
        action="store_true",
        help="Activa augmentation geométrico en train.",
    )
    ap.add_argument(
        "--augment_times",
        type=int,
        default=2,
        help="Número de augmentaciones por muestra train.",
    )
    ap.add_argument(
        "--rotate_deg",
        type=float,
        default=15.0,
        help="Rotación máxima en grados alrededor de Z.",
    )
    ap.add_argument(
        "--jitter_sigma",
        type=float,
        default=0.005,
        help="Sigma del jitter.",
    )
    ap.add_argument(
        "--jitter_clip",
        type=float,
        default=0.02,
        help="Clip del jitter.",
    )
    ap.add_argument(
        "--scale_min",
        type=float,
        default=0.95,
        help="Escala mínima.",
    )
    ap.add_argument(
        "--scale_max",
        type=float,
        default=1.05,
        help="Escala máxima.",
    )
    ap.add_argument(
        "--dropout_rate",
        type=float,
        default=0.05,
        help="Dropout de puntos durante augmentation.",
    )

    # -------------------------
    # Coverage / metadata
    # -------------------------
    ap.add_argument(
        "--ensure_coverage",
        action="store_true",
        help="Mueve muestras mínimas desde train a val/test si faltan clases.",
    )
    ap.add_argument(
        "--copy_label_map_from_source",
        action="store_true",
        help="Si existe artifacts/label_map.json en dataset_dir, lo copia además de guardar el nuevo.",
    )
    ap.add_argument(
        "--save_uncompressed",
        action="store_true",
        help="Guarda .npz sin compresión. Por defecto usa np.savez_compressed.",
    )

    # -------------------------
    # Safe resume / checkpoint por muestra
    # -------------------------
    ap.add_argument(
        "--use_safe_resume",
        action="store_true",
        help=(
            "Activa guardado seguro por muestra en _resume_cache. "
            "Permite relanzar el script y continuar desde las muestras ya procesadas."
        ),
    )
    ap.add_argument(
        "--resume_cache_dir",
        type=str,
        default=None,
        help=(
            "Directorio opcional para cache de resume. "
            "Si no se entrega, usa <out_dir>/_resume_cache."
        ),
    )
    ap.add_argument(
        "--overwrite_resume_cache",
        action="store_true",
        help=(
            "Recalcula y sobrescribe muestras cacheadas aunque existan. "
            "Útil si cambiaste parámetros y quieres forzar regeneración."
        ),
    )
    ap.add_argument(
        "--cleanup_resume_cache",
        action="store_true",
        help=(
            "Elimina _resume_cache al terminar correctamente y después de guardar los .npz finales. "
            "Por defecto se conserva para auditoría/reanudación."
        ),
    )

    return ap.parse_args()


# ============================================================
# DATA LOADING
# ============================================================

def load_npz_array(path: Path, key: str) -> np.ndarray:
    """
    Carga un array desde .npz.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe: {path}")

    with np.load(path) as data:
        if key not in data:
            raise KeyError(f"{path} no contiene key='{key}'. Keys={list(data.keys())}")
        arr = data[key]

    return arr


def load_merged_dataset(dataset_dir: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Carga splits desde dataset merged_*.
    """
    dataset_dir = Path(dataset_dir)

    splits = {}

    for split in ["train", "val", "test"]:
        Xp = dataset_dir / f"X_{split}.npz"
        Yp = dataset_dir / f"Y_{split}.npz"

        X = load_npz_array(Xp, "X")
        Y = load_npz_array(Yp, "Y")

        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"{split}: número de muestras no calza: X={X.shape[0]} Y={Y.shape[0]}"
            )

        if X.shape[1] != Y.shape[1]:
            raise ValueError(
                f"{split}: número de puntos no calza: X={X.shape} Y={Y.shape}"
            )

        print(f"[OK] {split}: X={X.shape} Y={Y.shape}")

        splits[split] = (X, Y)

    return splits


def maybe_load_existing_label_map(dataset_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Intenta cargar label_map desde:
      dataset_dir/artifacts/label_map.json
      dataset_dir/label_map.json
    """
    dataset_dir = Path(dataset_dir)

    candidates = [
        dataset_dir / "artifacts" / "label_map.json",
        dataset_dir / "label_map.json",
    ]

    for p in candidates:
        if p.exists():
            try:
                return json_load(p)
            except Exception:
                continue

    return None


def build_or_reuse_label_map(
    splits: Dict[str, Tuple[np.ndarray, np.ndarray]],
    dataset_dir: Path,
) -> Tuple[Dict[int, int], Dict[int, int], Optional[Dict[str, Any]]]:
    """
    Construye label_map global.

    Aunque exista label_map previo, se reconstruye desde Y para asegurar consistencia.
    También retorna el label_map previo si existía para metadatos.
    """
    source_label_map = maybe_load_existing_label_map(dataset_dir)

    Y_list = [
        splits["train"][1],
        splits["val"][1],
        splits["test"][1],
    ]

    id2idx, idx2id = build_global_label_map(Y_list)

    print(f"[MAP] Global labels: {len(id2idx)} clases ({min(id2idx.values())}..{max(id2idx.values())})")
    print(f"[MAP] Original label ids: {sorted([int(k) for k in id2idx.keys()])}")

    return id2idx, idx2id, source_label_map


def remap_all_splits(
    splits: Dict[str, Tuple[np.ndarray, np.ndarray]],
    id2idx: Dict[int, int],
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Remapea todos los Y de los splits a etiquetas internas.
    """
    out = {}

    for split, (X, Y) in splits.items():
        Yr = apply_remap(Y, id2idx)
        out[split] = (X, Yr)
        print(
            f"[REMAP] {split}: min={int(Yr.min())} max={int(Yr.max())} "
            f"unique={int(np.unique(Yr).size)}"
        )

    return out


# ============================================================
# SAVING
# ============================================================

def save_npz(path: Path, key: str, arr: np.ndarray, compressed: bool = True) -> None:
    """
    Guarda .npz.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if compressed:
        np.savez_compressed(path, **{key: arr})
    else:
        np.savez(path, **{key: arr})


def save_split_arrays(
    out_dir: Path,
    splits: Dict[str, Tuple[np.ndarray, np.ndarray]],
    compressed: bool = True,
) -> None:
    """
    Guarda X/Y por split.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        X, Y = splits[split]

        save_npz(out_dir / f"X_{split}.npz", "X", X.astype(np.float32, copy=False), compressed=compressed)
        save_npz(out_dir / f"Y_{split}.npz", "Y", Y.astype(np.int32, copy=False), compressed=compressed)

        print(f"[SAVE] {split}: X={X.shape} Y={Y.shape}")


def save_artifacts(
    out_dir: Path,
    args: argparse.Namespace,
    id2idx: Dict[int, int],
    idx2id: Dict[int, int],
    source_label_map: Optional[Dict[str, Any]],
    splits: Dict[str, Tuple[np.ndarray, np.ndarray]],
    class_weights: np.ndarray,
    class_hist: np.ndarray,
    coverage_moves: Optional[Dict[str, int]],
    elapsed_sec: float,
) -> None:
    """
    Guarda artifacts y meta.
    """
    out_dir = Path(out_dir)
    artifacts = ensure_dir(out_dir / "artifacts")

    json_dump(as_jsonable_labelmap(id2idx, idx2id), artifacts / "label_map.json")

    if source_label_map is not None and bool(args.copy_label_map_from_source):
        json_dump(source_label_map, artifacts / "source_label_map.json")

    class_weights_obj = {
        "weights": [float(x) for x in class_weights.tolist()],
        "hist": [int(x) for x in class_hist.tolist()],
        "num_classes": int(len(class_weights)),
        "note": "weights inversos simples normalizados desde Y_train",
    }
    json_dump(class_weights_obj, artifacts / "class_weights.json")

    eda = {
        split: split_eda(splits[split][1], bg_id=int(args.bg_id))
        for split in ["train", "val", "test"]
    }

    meta = {
        "script_name": "augmentation_and_split_paperlike_v3_sampleo_FPS_SAFE_RESUME.py",
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "out_dir": str(Path(args.out_dir).resolve()),
        "N": int(args.N),
        "sampling_method": str(args.sampling_method),
        "cap_bg_raw": None if args.cap_bg is None else float(args.cap_bg),
        "cap_bg_frac_resolved": resolve_cap_bg_frac(args.cap_bg, int(args.N)),
        "bg_id": int(args.bg_id),
        "seed": int(args.seed),
        "use_augmentation": bool(args.use_augmentation),
        "augment_times": int(args.augment_times),
        "augmentation": {
            "rotate_deg": float(args.rotate_deg),
            "jitter_sigma": float(args.jitter_sigma),
            "jitter_clip": float(args.jitter_clip),
            "scale_min": float(args.scale_min),
            "scale_max": float(args.scale_max),
            "dropout_rate": float(args.dropout_rate),
        },
        "fps": {
            "fps_pool_multiplier": float(args.fps_pool_multiplier),
        },
        "curvature": {
            "curvature_ratio": float(args.curvature_ratio),
            "curvature_k": int(args.curvature_k),
            "curvature_max_candidates": int(args.curvature_max_candidates),
            "curvature_temperature": float(args.curvature_temperature),
            "curvature_deterministic_top": bool(args.curvature_deterministic_top),
        },
        "boundary_labels": {
            "boundary_ratio": float(args.boundary_ratio),
            "boundary_k": int(args.boundary_k),
            "boundary_max_candidates": int(args.boundary_max_candidates),
            "boundary_fill_method": str(args.boundary_fill_method),
            "supervised_sampling_note": (
                "boundary_labels usa etiquetas para detectar bordes; "
                "reportar como estrategia supervisada/ablation."
            ),
        },
        "ensure_coverage": bool(args.ensure_coverage),
        "coverage_moves": coverage_moves or {"val": 0, "test": 0},
        "splits": {
            split: {
                "X_shape": list(splits[split][0].shape),
                "Y_shape": list(splits[split][1].shape),
                "eda": eda[split],
            }
            for split in ["train", "val", "test"]
        },
        "num_classes": int(len(id2idx)),
        "label_ids_original": sorted([int(k) for k in id2idx.keys()]),
        "save_compressed": not bool(args.save_uncompressed),
        "safe_resume": {
            "use_safe_resume": bool(getattr(args, "use_safe_resume", False)),
            "resume_cache_dir": None if getattr(args, "resume_cache_dir", None) is None else str(args.resume_cache_dir),
            "overwrite_resume_cache": bool(getattr(args, "overwrite_resume_cache", False)),
            "cleanup_resume_cache": bool(getattr(args, "cleanup_resume_cache", False)),
            "note": "El cache por muestra no cambia los outputs finales; solo permite continuar si el proceso se interrumpe.",
        },
        "elapsed_sec": float(elapsed_sec),
        "traceability_note": (
            "Este script mantiene el orden de muestras heredado del merged, "
            "pero no escribe index_*.csv. Ejecutar create_traceability_indices_for_fixed_split.py "
            "para restaurar trazabilidad explícita."
        ),
    }

    json_dump(meta, artifacts / "meta.json")
    json_dump(meta, out_dir / "meta_sampling.json")


# ============================================================
# MAIN PROCESSING
# ============================================================

def build_fixed_split(args: argparse.Namespace) -> None:
    """
    Ejecuta todo el pipeline:
      - carga merged
      - remapea labels
      - samplea splits
      - augmentation train
      - ensure coverage opcional
      - guarda X/Y/artifacts
    """
    t0 = time.time()

    dataset_dir = Path(args.dataset_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    ensure_dir(out_dir)
    ensure_dir(out_dir / "artifacts")

    resume_cache_dir = resolve_resume_cache_dir(args, out_dir)
    if bool(getattr(args, "use_safe_resume", False)):
        ensure_dir(resume_cache_dir)
        print(f"[RESUME] Activado. Cache: {resume_cache_dir}")
        if bool(getattr(args, "overwrite_resume_cache", False)):
            print("[RESUME] overwrite_resume_cache=True: se recalcularán muestras aunque exista cache.")
    else:
        print("[RESUME] desactivado")

    print("=" * 80)
    print("[START] augmentation_and_split_paperlike_v3_sampleo_FPS_SAFE_RESUME.py")
    print("=" * 80)
    print(f"[INFO] dataset_dir     : {dataset_dir}")
    print(f"[INFO] out_dir         : {out_dir}")
    print(f"[INFO] N               : {args.N}")
    print(f"[INFO] sampling_method : {args.sampling_method}")
    print(f"[INFO] cap_bg          : {args.cap_bg}")
    print(f"[INFO] bg_id           : {args.bg_id}")
    print(f"[INFO] seed            : {args.seed}")
    print(f"[INFO] augmentation    : {args.use_augmentation} x{args.augment_times}")
    print("=" * 80)

    cap_bg_frac = resolve_cap_bg_frac(args.cap_bg, int(args.N))
    print(f"[INFO] cap_bg_frac_resolved: {cap_bg_frac}")

    # -------------------------
    # Load
    # -------------------------
    raw_splits = load_merged_dataset(dataset_dir)

    # -------------------------
    # Label map + remap
    # -------------------------
    id2idx, idx2id, source_label_map = build_or_reuse_label_map(raw_splits, dataset_dir)
    splits_remapped = remap_all_splits(raw_splits, id2idx)

    # Liberar referencia al raw_splits si no se necesita más
    del raw_splits
    gc.collect()

    # -------------------------
    # Process train/val/test originals
    # -------------------------
    processed_splits: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for split in ["train", "val", "test"]:
        X_full, Y_full = splits_remapped[split]

        Xs, Ys = process_split(
            X=X_full,
            Y=Y_full,
            split_name=split,
            N=int(args.N),
            method=str(args.sampling_method),
            cap_bg_frac=cap_bg_frac,
            bg_id=int(args.bg_id),
            seed=int(args.seed),
            fps_pool_multiplier=float(args.fps_pool_multiplier),
            curvature_ratio=float(args.curvature_ratio),
            curvature_k=int(args.curvature_k),
            curvature_max_candidates=int(args.curvature_max_candidates),
            curvature_temperature=float(args.curvature_temperature),
            curvature_deterministic_top=bool(args.curvature_deterministic_top),
            boundary_ratio=float(args.boundary_ratio),
            boundary_k=int(args.boundary_k),
            boundary_max_candidates=int(args.boundary_max_candidates),
            boundary_fill_method=str(args.boundary_fill_method),
            use_safe_resume=bool(getattr(args, "use_safe_resume", False)),
            resume_cache_dir=resume_cache_dir,
            overwrite_resume_cache=bool(getattr(args, "overwrite_resume_cache", False)),
        )

        processed_splits[split] = (Xs, Ys)

        print(
            f"[PROC] {split}: X={Xs.shape} Y={Ys.shape} "
            f"bg={compute_bg_fraction(Ys, bg_id=int(args.bg_id)):.4f}"
        )

    # -------------------------
    # Augmentation train
    # -------------------------
    if bool(args.use_augmentation) and int(args.augment_times) > 0:
        X_train_full, Y_train_full = splits_remapped["train"]

        Xa, Ya = augment_and_resample_train(
            X_train_full=X_train_full,
            Y_train_full=Y_train_full,
            N=int(args.N),
            method=str(args.sampling_method),
            cap_bg_frac=cap_bg_frac,
            bg_id=int(args.bg_id),
            seed=int(args.seed),
            augment_times=int(args.augment_times),
            rotate_deg=float(args.rotate_deg),
            jitter_sigma=float(args.jitter_sigma),
            jitter_clip=float(args.jitter_clip),
            scale_min=float(args.scale_min),
            scale_max=float(args.scale_max),
            dropout_rate=float(args.dropout_rate),
            fps_pool_multiplier=float(args.fps_pool_multiplier),
            curvature_ratio=float(args.curvature_ratio),
            curvature_k=int(args.curvature_k),
            curvature_max_candidates=int(args.curvature_max_candidates),
            curvature_temperature=float(args.curvature_temperature),
            curvature_deterministic_top=bool(args.curvature_deterministic_top),
            boundary_ratio=float(args.boundary_ratio),
            boundary_k=int(args.boundary_k),
            boundary_max_candidates=int(args.boundary_max_candidates),
            boundary_fill_method=str(args.boundary_fill_method),
            use_safe_resume=bool(getattr(args, "use_safe_resume", False)),
            resume_cache_dir=resume_cache_dir,
            overwrite_resume_cache=bool(getattr(args, "overwrite_resume_cache", False)),
        )

        Xtr, Ytr = processed_splits["train"]

        Xtr2 = np.concatenate([Xtr, Xa], axis=0).astype(np.float32, copy=False)
        Ytr2 = np.concatenate([Ytr, Ya], axis=0).astype(np.int32, copy=False)

        processed_splits["train"] = (Xtr2, Ytr2)

        print(f"[AUG] train: +{Xa.shape[0]} muestras")
        print(f"[AUG] train final: X={Xtr2.shape} Y={Ytr2.shape}")

        del Xa, Ya, Xtr, Ytr, Xtr2, Ytr2
        gc.collect()

    else:
        print("[AUG] desactivado")

    # Ya no necesitamos splits remapped completos
    del splits_remapped
    gc.collect()

    # -------------------------
    # Ensure coverage
    # -------------------------
    coverage_moves = {"val": 0, "test": 0}

    if bool(args.ensure_coverage):
        rng_cov = np_rng(int(args.seed) + 777)
        coverage_moves = ensure_coverage_min_moves(processed_splits, rng_cov)
        print(f"[COVER] Movimientos realizados: val={coverage_moves.get('val',0)} test={coverage_moves.get('test',0)}")
    else:
        print("[COVER] desactivado")

    # -------------------------
    # Class weights
    # -------------------------
    Y_train = processed_splits["train"][1]
    class_weights, class_hist = compute_class_weights_from_train(
        Y_train=Y_train,
        num_classes=int(len(id2idx)),
    )

    # -------------------------
    # EDA
    # -------------------------
    print("\n[EDA] Validación por split:")
    for split in ["train", "val", "test"]:
        eda = split_eda(processed_splits[split][1], bg_id=int(args.bg_id))
        print(
            f"  {split}: min={eda['min']} max={eda['max']} unique={eda['unique']} "
            f"bg={eda['bg_frac']*100:.2f}%"
        )

    # -------------------------
    # Save
    # -------------------------
    compressed = not bool(args.save_uncompressed)

    save_split_arrays(
        out_dir=out_dir,
        splits=processed_splits,
        compressed=compressed,
    )

    elapsed = time.time() - t0

    save_artifacts(
        out_dir=out_dir,
        args=args,
        id2idx=id2idx,
        idx2id=idx2id,
        source_label_map=source_label_map,
        splits=processed_splits,
        class_weights=class_weights,
        class_hist=class_hist,
        coverage_moves=coverage_moves,
        elapsed_sec=float(elapsed),
    )

    print("\n" + "=" * 80)
    print(f"[DONE] Splits listos en: {out_dir}")
    print(f"[META] Total clases: {len(id2idx)}")
    print(f"[TIME] elapsed_sec={elapsed:.2f}")
    print("[NEXT] Restaurar trazabilidad con:")
    print(
        "python scripts_last_version/create_traceability_indices_for_fixed_split.py "
        f"--source_dir {dataset_dir} "
        f"--target_dir {out_dir} "
        f"--augment_times {int(args.augment_times) if bool(args.use_augmentation) else 0} "
        "--overwrite"
    )
    print("=" * 80)


# ============================================================
# VALIDATIONS
# ============================================================

def validate_dataset_shapes(
    splits: Dict[str, Tuple[np.ndarray, np.ndarray]],
    expected_N: int,
) -> None:
    """
    Validaciones defensivas finales.
    """
    expected_N = int(expected_N)

    for split in ["train", "val", "test"]:
        X, Y = splits[split]

        if X.ndim != 3:
            raise ValueError(f"{split}: X debe ser [B,N,3], recibido {X.shape}")

        if Y.ndim != 2:
            raise ValueError(f"{split}: Y debe ser [B,N], recibido {Y.shape}")

        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"{split}: cantidad de muestras no coincide: "
                f"X={X.shape[0]} Y={Y.shape[0]}"
            )

        if X.shape[1] != Y.shape[1]:
            raise ValueError(
                f"{split}: cantidad de puntos no coincide: "
                f"X={X.shape} Y={Y.shape}"
            )

        if X.shape[1] != expected_N:
            raise ValueError(
                f"{split}: N incorrecto. Esperado={expected_N}, recibido={X.shape[1]}"
            )

        if X.shape[2] != 3:
            raise ValueError(
                f"{split}: dimensión XYZ inválida: {X.shape}"
            )

        if not np.isfinite(X).all():
            raise ValueError(f"{split}: X contiene NaN/Inf")

        if not np.isfinite(Y).all():
            raise ValueError(f"{split}: Y contiene NaN/Inf")

        print(
            f"[VALID] {split}: "
            f"B={X.shape[0]} N={X.shape[1]} "
            f"bg={compute_bg_fraction(Y):.4f}"
        )


def validate_class_presence(
    splits: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> None:
    """
    Reporta presencia de clases por split.
    """
    print("\n[CLASS PRESENCE]")

    global_classes = set()

    for split in ["train", "val", "test"]:
        _, Y = splits[split]
        cls = sorted(np.unique(Y).astype(int).tolist())

        global_classes.update(cls)

        print(f"  {split}: {cls}")

    global_classes = sorted(int(x) for x in global_classes)

    print(f"\n[GLOBAL CLASSES] {global_classes}")


def validate_bg_cap(
    splits: Dict[str, Tuple[np.ndarray, np.ndarray]],
    cap_bg_frac: Optional[float],
    bg_id: int = 0,
) -> None:
    """
    Verifica si el control de background se respetó aproximadamente.

    Nota:
    - boundary/fps pueden requerir ligeras desviaciones cuando no hay suficiente fg.
    """
    if cap_bg_frac is None:
        print("\n[BG CAP] desactivado")
        return

    print("\n[BG CAP CHECK]")

    for split in ["train", "val", "test"]:
        _, Y = splits[split]

        frac = compute_bg_fraction(Y, bg_id=bg_id)

        status = "OK"
        if frac > float(cap_bg_frac) + 0.02:
            status = "WARN"

        print(
            f"  {split}: "
            f"bg_frac={frac:.4f} "
            f"target<={float(cap_bg_frac):.4f} "
            f"[{status}]"
        )


def validate_sampling_distribution(
    splits: Dict[str, Tuple[np.ndarray, np.ndarray]],
    bg_id: int = 0,
) -> None:
    """
    Estadísticas simples de distribución.
    """
    print("\n[DISTRIBUTION CHECK]")

    for split in ["train", "val", "test"]:
        _, Y = splits[split]

        flat = Y.reshape(-1)

        hist = np.bincount(flat.astype(np.int64))

        print(f"\n  {split}")
        print(f"    total_points={flat.size:,}")
        print(f"    bg_frac={compute_bg_fraction(flat, bg_id=bg_id):.4f}")

        for cls_id, count in enumerate(hist.tolist()):
            frac = count / max(1, flat.size)

            print(
                f"    class={cls_id:02d} "
                f"count={count:>12,} "
                f"frac={frac:.6f}"
            )


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    # --------------------------------------------------------
    # Defensive argument checks
    # --------------------------------------------------------
    if int(args.N) <= 0:
        raise ValueError("--N debe ser > 0")

    if int(args.seed) < 0:
        raise ValueError("--seed debe ser >= 0")

    if int(args.bg_id) < 0:
        raise ValueError("--bg_id debe ser >= 0")

    if bool(args.use_augmentation):
        if int(args.augment_times) < 0:
            raise ValueError("--augment_times debe ser >= 0")

    if float(args.curvature_ratio) < 0 or float(args.curvature_ratio) > 1:
        raise ValueError("--curvature_ratio debe estar en [0,1]")

    if float(args.boundary_ratio) < 0 or float(args.boundary_ratio) > 1:
        raise ValueError("--boundary_ratio debe estar en [0,1]")

    dataset_dir = Path(args.dataset_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    if not dataset_dir.exists():
        raise FileNotFoundError(f"No existe dataset_dir: {dataset_dir}")

    # --------------------------------------------------------
    # Ejecutar pipeline
    # --------------------------------------------------------
    build_fixed_split(args)

    # --------------------------------------------------------
    # Validación final recargando desde disco
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print("[POST-SAVE VALIDATION]")
    print("=" * 80)

    splits_saved = {}

    for split in ["train", "val", "test"]:
        X = load_npz_array(out_dir / f"X_{split}.npz", "X")
        Y = load_npz_array(out_dir / f"Y_{split}.npz", "Y")

        splits_saved[split] = (X, Y)

    validate_dataset_shapes(
        splits=splits_saved,
        expected_N=int(args.N),
    )

    validate_class_presence(splits_saved)

    validate_bg_cap(
        splits=splits_saved,
        cap_bg_frac=resolve_cap_bg_frac(args.cap_bg, int(args.N)),
        bg_id=int(args.bg_id),
    )

    validate_sampling_distribution(
        splits=splits_saved,
        bg_id=int(args.bg_id),
    )

    print("\n" + "=" * 80)
    print("[SUCCESS]")
    print("=" * 80)
    print("Dataset fixed_split generado correctamente.")
    print(f"sampling_method = {args.sampling_method}")
    print(f"N = {args.N}")
    print(f"out_dir = {out_dir}")

    print("\nIMPORTANTE:")
    print(
        "Restaurar trazabilidad explícita ejecutando:"
    )

    aug_times = int(args.augment_times) if bool(args.use_augmentation) else 0

    print(
        f"python scripts_last_version/create_traceability_indices_for_fixed_split.py "
        f"--source_dir {dataset_dir} "
        f"--target_dir {out_dir} "
        f"--augment_times {aug_times} "
        "--overwrite"
    )

    print("=" * 80)


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    main()
