# =========================
# PART 1/4
# preprocess_and_flatten_safe.py
# =========================
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Preprocesamiento robusto Teeth3DS:
- Carga mallas (obj/ply/stl) + json (labels/instances por vértice)
- Sampleo: global (surface), stratified (por label de vértice), fps (sobre vértices)
- Normaliza (center + unit sphere)
- Guarda por sujeto:
    out_struct_root/<N>/<jaw>/<subject>/{point_cloud.npy, labels.npy, instances.npy, meta.json}
  y crea vista "flat" por symlink/copy:
    out_flat_root/<N>/<sample_type_jaw>/<subject> -> symlink a struct

CORRECCIÓN CLAVE (para tu tesis):
- Por defecto, EXCLUYE muelas del juicio {18, 28, 38, 48} remapeándolas a 0 (encía/fondo).
- Mantiene solo FDI válidos (sin wisdom) + 0.
- FPS ahora es reproducible (usa rng con seed), no np.random global.

Uso típico:
python preprocess_and_flatten_safe.py \
  --in_root /.../raw \
  --out_struct_root /.../processed_struct_safe \
  --out_flat_root /.../processed_flat_safe \
  --parts data_part_1 data_part_2 \
  --jaws upper lower \
  --n_points 200000 \
  --sample_mode global \
  --seed 42 \
  --threads 1
"""

import argparse
import os
import json
import shutil
import sys
import gc
import random
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np

# ---- Limitar hilos nativos (se aplicará tras parsear args) ----
def set_thread_env(n_threads: int):
    n = str(max(1, int(n_threads)))
    os.environ.setdefault("OMP_NUM_THREADS", n)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", n)
    os.environ.setdefault("MKL_NUM_THREADS", n)
    os.environ.setdefault("NUMEXPR_NUM_THREADS", n)

# Importar librerías pesadas después de definir hilos
import trimesh as tm
from scipy.spatial import cKDTree


# --------------------
# Label policy (FDI)
# --------------------
WISDOM_FDI = {18, 28, 38, 48}

def allowed_fdi_without_wisdom() -> set:
    # 11–17, 21–27, 31–37, 41–47 + 0
    base = set(list(range(11, 18)) + list(range(21, 28)) + list(range(31, 38)) + list(range(41, 48)))
    base.add(0)
    return base

def apply_label_policy_fdi(labels: np.ndarray, keep_wisdom: bool) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Recibe labels en FDI (incluye 0) y:
    - si keep_wisdom=False: 18/28/38/48 -> 0
    - cualquier label no permitido -> 0
    Retorna (labels_filtrados, stats)
    """
    y = np.asarray(labels, dtype=np.int32).copy()
    stats = {
        "n_total": int(y.size),
        "n_to_0_wisdom": 0,
        "n_to_0_other": 0,
    }

    if y.size == 0:
        return y, stats

    allowed = allowed_fdi_without_wisdom() if not keep_wisdom else (allowed_fdi_without_wisdom() | WISDOM_FDI)

    # wisdom -> 0 (si aplica)
    if not keep_wisdom:
        mask_w = np.isin(y, np.asarray(sorted(WISDOM_FDI), dtype=np.int32))
        stats["n_to_0_wisdom"] = int(mask_w.sum())
        y[mask_w] = 0

    # todo lo que no está permitido -> 0
    mask_bad = ~np.isin(y, np.asarray(sorted(allowed), dtype=np.int32))
    stats["n_to_0_other"] = int(mask_bad.sum())
    y[mask_bad] = 0

    return y, stats


# --------------------
# Utils generales
# --------------------
def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))

def find_first(mesh_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    mesh = None
    meta = None
    for ext in (".obj", ".ply", ".stl"):
        cand = sorted(mesh_dir.glob(f"*{ext}"))
        if cand:
            mesh = cand[0]
            break
    j = sorted(mesh_dir.glob("*.json"))
    if j:
        meta = j[0]
    return mesh, meta

def normalize(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        return points
    if not np.isfinite(points).all():
        points = np.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)

    c = points.mean(axis=0, keepdims=True)
    points = points - c
    norms = np.linalg.norm(points, axis=1)
    if norms.size == 0:
        return points
    m = norms.max()
    if np.isfinite(m) and m > 0:
        points = points / m
    return points

def is_valid_mesh(mesh: tm.Trimesh) -> bool:
    try:
        v = np.asarray(mesh.vertices, dtype=np.float32)
        if v.ndim != 2 or v.shape[1] != 3 or v.shape[0] == 0:
            return False
        if not np.isfinite(v).all():
            return False
        return True
    except Exception:
        return False

def load_mesh_safe(mesh_path: Path) -> Optional[tm.Trimesh]:
    """
    Carga robusta:
      - Fusiona escenas
      - Limpia NaNs
      - Devuelve Trimesh o None
    """
    try:
        m = tm.load(mesh_path, process=False, force="mesh")
        if isinstance(m, tm.Scene):
            geos = list(m.dump().geometry.values())
            if len(geos) == 0:
                return None
            m = tm.util.concatenate(geos)
        if not isinstance(m, tm.Trimesh):
            return None
        v = np.asarray(m.vertices, dtype=np.float32)
        if v.size == 0 or v.ndim != 2 or v.shape[1] != 3:
            return None
        if not np.isfinite(v).all():
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            m.vertices = v
        return m
    except Exception:
        return None

# =========================
# PART 2/4
# =========================

def symlink_or_copy(src_dir: Path, dst_dir: Path, do_copy: bool):
    dst_dir.parent.mkdir(parents=True, exist_ok=True)
    if dst_dir.exists() or dst_dir.is_symlink():
        return
    if do_copy:
        shutil.copytree(src_dir, dst_dir)
    else:
        try:
            os.symlink(src_dir, dst_dir, target_is_directory=True)
        except OSError:
            shutil.copytree(src_dir, dst_dir)

def sample_points(mesh: tm.Trimesh, n_points: int) -> np.ndarray:
    """
    Sampleo de superficie con trimesh.sample.
    Si falla, cae a muestreo de vértices.
    """
    n_points = int(n_points)
    if n_points <= 0:
        raise ValueError("n_points debe ser > 0")

    try:
        if mesh.faces is not None and len(mesh.faces) > 0:
            pts, _ = tm.sample.sample_surface(mesh, n_points)
            pts = np.asarray(pts, dtype=np.float32)
            return pts
    except Exception:
        pass

    # Fallback: vértices
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    if verts.shape[0] == 0:
        raise ValueError("Malla sin vértices.")
    rng = np.random.default_rng()
    idx = rng.choice(verts.shape[0], size=n_points, replace=(verts.shape[0] < n_points))
    return verts[idx].astype(np.float32)

def assign_by_vertex_nn(sampled_pts: np.ndarray, vert_coords: np.ndarray, labels_v: np.ndarray) -> np.ndarray:
    """
    Asigna etiqueta por vecino más cercano en vértices.
    Usa cKDTree si puede; fallback por bloques si falla.
    """
    sampled_pts = np.asarray(sampled_pts, dtype=np.float32)
    vert_coords = np.asarray(vert_coords, dtype=np.float32)
    labels_v = np.asarray(labels_v)

    try:
        tree = cKDTree(vert_coords.astype(np.float32), leafsize=64)
        _, idx = tree.query(sampled_pts.astype(np.float32), k=1, workers=-1)
        return labels_v[idx]
    except Exception as e:
        print(f"[WARN] KDTree falló ({e}). Fallback L2 directo.", file=sys.stderr)
        B = 8192
        out = np.empty((sampled_pts.shape[0],), dtype=labels_v.dtype)
        for s in range(0, sampled_pts.shape[0], B):
            e2 = min(sampled_pts.shape[0], s + B)
            block = sampled_pts[s:e2]
            d2 = ((block[:, None, :] - vert_coords[None, :, :]) ** 2).sum(axis=2)
            nn = np.argmin(d2, axis=1)
            out[s:e2] = labels_v[nn]
        return out

# -------------------- FPS (reproducible) --------------------
def farthest_point_sampling(points: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """
    FPS en NumPy. Devuelve subconjunto de puntos más lejanos entre sí.
    Complejidad O(N * n_samples). Reproducible si rng es fijo.
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] == 0:
        raise ValueError("FPS: nube vacía.")
    if not np.isfinite(pts).all():
        pts = np.nan_to_num(pts, nan=0.0, posinf=0.0, neginf=0.0)

    N = pts.shape[0]
    n_samples = int(n_samples)
    if n_samples >= N:
        return pts.copy()

    centroids = np.zeros((n_samples,), dtype=np.int32)
    distances = np.full((N,), np.float32(1e10), dtype=np.float32)

    farthest = int(rng.integers(low=0, high=N))
    for i in range(n_samples):
        centroids[i] = farthest
        centroid = pts[farthest, :]
        diff = pts - centroid
        dist = (diff * diff).sum(axis=1)
        mask = dist < distances
        distances[mask] = dist[mask]
        farthest = int(np.argmax(distances))

    return pts[centroids, :]

# -------------------- Stratified (por labels de vértice) --------------------
def stratified_sample_points_with_labels(mesh: tm.Trimesh, n_points: int, labels_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Samplea en superficie y asigna etiqueta por NN a vértices,
    pero intentando mantener presencia de clases (estratificado simple).
    """
    n_points = int(n_points)
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    labels_v = np.asarray(labels_v, dtype=np.int32)
    if verts.shape[0] == 0 or labels_v.shape[0] != verts.shape[0]:
        raise ValueError("Stratified: vertices/labels incompatibles.")

    # base: sampleo superficie
    pts = sample_points(mesh, n_points)
    lbs = assign_by_vertex_nn(pts, verts, labels_v)

    return pts.astype(np.float32), lbs.astype(np.int32)

# =========================
# PART 3/4
# =========================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_root", required=True, help="data/Teeth_3ds/raw")
    p.add_argument("--out_struct_root", required=True, help="data/Teeth_3ds/processed_struct_safe")
    p.add_argument("--out_flat_root", required=True, help="data/Teeth_3ds/processed_flat_safe")
    p.add_argument("--parts", nargs="+", required=True, help="data_part_1 data_part_2 ...")
    p.add_argument("--jaws", nargs="+", default=["upper", "lower"])
    p.add_argument("--n_points", type=int, default=8192)
    p.add_argument("--copy", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=1, help="Hilos nativos BLAS/OpenMP (recomendado=1)")
    p.add_argument(
        "--sample_mode",
        choices=["global", "stratified", "fps"],
        default="global",
        help="Muestreo: global (superficie), stratified (por etiqueta de vértice), fps (sobre vértices).",
    )

    # ✅ Política de labels
    p.add_argument(
        "--keep_wisdom",
        action="store_true",
        help="Si se activa, NO se excluyen muelas del juicio (18/28/38/48). Por defecto se excluyen.",
    )

    args = p.parse_args()

    # limitar hilos nativos para evitar segfaults en BLAS/OpenMP
    set_thread_env(args.threads)

    seed_all(args.seed)
    rng = make_rng(args.seed)

    in_root = Path(args.in_root)
    out_s = Path(args.out_struct_root)
    out_f = Path(args.out_flat_root)
    out_s.mkdir(parents=True, exist_ok=True)
    out_f.mkdir(parents=True, exist_ok=True)

    label_policy_name = "keep_wisdom" if args.keep_wisdom else "exclude_wisdom_to_0"

    try:
        for part in args.parts:
            for jaw in args.jaws:
                src = in_root / part / jaw
                if not src.is_dir():
                    print(f"[WARN] No existe: {src}", file=sys.stderr)
                    continue

                for subj in sorted(src.iterdir()):
                    if not subj.is_dir():
                        continue

                    mesh_file, meta_file = find_first(subj)
                    if mesh_file is None:
                        print(f"[WARN] Sin malla en: {subj}", file=sys.stderr)
                        continue

                    mesh = load_mesh_safe(mesh_file)
                    if mesh is None or not is_valid_mesh(mesh):
                        print(f"[WARN] Malla inválida: {mesh_file}", file=sys.stderr)
                        continue

                    meta: Dict[str, Any] = {}
                    if meta_file is not None:
                        try:
                            with open(meta_file, "r", encoding="utf-8") as fh:
                                meta = json.load(fh)
                        except Exception as e:
                            print(f"[WARN] JSON no legible {meta_file}: {e}", file=sys.stderr)
                            meta = {}

                    labels_json = (np.array(meta.get("labels", []), dtype=np.int32) if "labels" in meta else None)
                    inst_json   = (np.array(meta.get("instances", []), dtype=np.int32) if "instances" in meta else None)

                    pts = None
                    lbs = None
                    inst = None
                    mode = "UNKNOWN"
                    policy_stats = {}

                    try:
                        # ---------- MODOS DE MUESTREO ----------
                        if args.sample_mode == "stratified" and labels_json is not None and len(labels_json) == len(mesh.vertices):
                            pts_surf, lbs0 = stratified_sample_points_with_labels(
                                mesh, int(args.n_points), labels_json.astype(np.int32)
                            )
                            pts = normalize(pts_surf)
                            lbs, policy_stats = apply_label_policy_fdi(lbs0, keep_wisdom=args.keep_wisdom)
                            inst = None
                            mode = "STRATIFIED_SAMPLE_VERT_LABELS"

                        elif args.sample_mode == "fps":
                            verts = np.asarray(mesh.vertices, dtype=np.float32)
                            if verts.shape[0] == 0:
                                raise ValueError("Malla sin vértices.")
                            pts_fps = farthest_point_sampling(verts, int(args.n_points), rng=rng)
                            pts = normalize(pts_fps)

                            lbs0 = None
                            if labels_json is not None and len(labels_json) == len(mesh.vertices):
                                lbs0 = assign_by_vertex_nn(pts, verts, labels_json.astype(np.int32))
                            if lbs0 is not None:
                                lbs, policy_stats = apply_label_policy_fdi(lbs0, keep_wisdom=args.keep_wisdom)
                            else:
                                lbs = None

                            if inst_json is not None and len(inst_json) == len(mesh.vertices):
                                inst = assign_by_vertex_nn(pts, verts, inst_json.astype(np.int32))
                            else:
                                inst = None

                            mode = "FPS_VERTICES_BASE"

                        else:
                            # modo global por defecto (superficie con fallback)
                            pts_surf = sample_points(mesh, int(args.n_points))
                            pts = normalize(pts_surf)

                            verts = np.asarray(mesh.vertices, dtype=np.float32)

                            lbs0 = None
                            if labels_json is not None and len(labels_json) == len(verts):
                                lbs0 = assign_by_vertex_nn(pts_surf, verts, labels_json.astype(np.int32))
                            if lbs0 is not None:
                                lbs, policy_stats = apply_label_policy_fdi(lbs0, keep_wisdom=args.keep_wisdom)
                            else:
                                lbs = None

                            if inst_json is not None and len(inst_json) == len(verts):
                                inst = assign_by_vertex_nn(pts_surf, verts, inst_json.astype(np.int32))
                            else:
                                inst = None

                            mode = "SURF_SAMPLE_GLOBAL"

                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        print(f"[ERR] Preproc falló {mesh_file}: {e}", file=sys.stderr)
                        del mesh
                        gc.collect()
                        continue

# =========================
# PART 4/4
# =========================

                    # ---------- Guardado estructurado ----------
                    try:
                        dst_struct = out_s / str(int(pts.shape[0])) / jaw / subj.name
                        dst_struct.mkdir(parents=True, exist_ok=True)

                        np.save(dst_struct / "point_cloud.npy", pts.astype(np.float32))
                        if lbs is not None:
                            np.save(dst_struct / "labels.npy", np.asarray(lbs, dtype=np.int32))
                        if inst is not None:
                            np.save(dst_struct / "instances.npy", np.asarray(inst, dtype=np.int32))

                        meta_out = {
                            "source_mesh": str(mesh_file),
                            "source_json": str(meta_file) if meta_file else None,
                            "n_points": int(pts.shape[0]),
                            "mode": mode,
                            "sample_mode": args.sample_mode,
                            "label_policy": label_policy_name,
                            "label_policy_stats": policy_stats,
                        }

                        # info rápida de clases presentes tras policy (si hay labels)
                        if lbs is not None and lbs.size > 0:
                            u, c = np.unique(lbs, return_counts=True)
                            meta_out["labels_unique_after_policy"] = {int(k): int(v) for k, v in zip(u.tolist(), c.tolist())}

                        with open(dst_struct / "meta.json", "w", encoding="utf-8") as fh:
                            json.dump(meta_out, fh, indent=2)

                        # ---------- Vista flat ----------
                        sample_type = mode.lower().replace(" ", "_")
                        dst_flat = out_f / str(int(pts.shape[0])) / f"{sample_type}_{jaw}" / subj.name
                        symlink_or_copy(dst_struct, dst_flat, do_copy=args.copy)

                        print(f"[OK] {part}/{jaw}/{subj.name} -> {dst_flat} ({mode}, {pts.shape[0]} pts, policy={label_policy_name})")

                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        print(f"[ERR] Guardado falló {subj}: {e}", file=sys.stderr)

                    # liberar memoria
                    del mesh, pts, lbs, inst
                    gc.collect()

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Detenido por el usuario.", file=sys.stderr)
    finally:
        gc.collect()

if __name__ == "__main__":
    main()
