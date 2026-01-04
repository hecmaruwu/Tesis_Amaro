#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import json
import shutil
import sys
from pathlib import Path
import math
import warnings
import gc
import random
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

# -------------------- utils --------------------

def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

def find_first(mesh_dir: Path):
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
    # proteger NaNs
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
        # faces pueden no existir; eso no invalida el muestreo por vértices
        return True
    except Exception:
        return False

def load_mesh_safe(mesh_path: Path):
    """
    Carga robusta:
     - Fusiona escenas
     - Limpia NaNs
     - Devuelve Trimesh o None si no se puede usar
    """
    try:
        m = tm.load(mesh_path, process=False, force='mesh')  # evitar reparaciones costosas
        if isinstance(m, tm.Scene):
            # concatenar geometrías de la escena de forma segura
            geos = list(m.dump().geometry.values())
            if len(geos) == 0:
                return None
            m = tm.util.concatenate(geos)
        if not isinstance(m, tm.Trimesh):
            return None
        # filtrar NaNs en vertices si existieran
        v = np.asarray(m.vertices, dtype=np.float32)
        if v.size == 0 or v.ndim != 2 or v.shape[1] != 3:
            return None
        if not np.isfinite(v).all():
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            m.vertices = v
        # faces pueden ser None; está bien
        return m
    except Exception as e:
        print(f"[ERR] Carga fallida {mesh_path}: {e}", file=sys.stderr)
        return None

def sample_surface_safe(mesh: tm.Trimesh, n_points: int) -> np.ndarray:
    """
    Muestreo de superficie robusto:
     - Si hay caras, usa sample_surface; si falla o no hay caras, cae a vértices.
    """
    v = np.asarray(mesh.vertices, dtype=np.float32)
    f = np.asarray(mesh.faces) if (hasattr(mesh, "faces") and mesh.faces is not None) else None

    if f is not None and f.size > 0:
        try:
            # trimesh devuelve float64; convertimos a float32
            pts, _ = tm.sample.sample_surface(mesh, int(n_points))
            pts = np.asarray(pts, dtype=np.float32)
            if not np.isfinite(pts).all():
                pts = np.nan_to_num(pts, nan=0.0, posinf=0.0, neginf=0.0)
            return pts
        except Exception as e:
            print(f"[WARN] sample_surface falló ({e}). Caigo a muestreo por vértices.", file=sys.stderr)

    # Fallback: muestreo por vértices
    if v.size == 0:
        raise ValueError("Malla sin vértices.")
    replace = (v.shape[0] < n_points)
    idx = np.random.choice(v.shape[0], size=int(n_points), replace=replace)
    pts = v[idx].astype(np.float32, copy=False)
    if not np.isfinite(pts).all():
        pts = np.nan_to_num(pts, nan=0.0, posinf=0.0, neginf=0.0)
    return pts

def sample_points(mesh: tm.Trimesh, n_points: int) -> np.ndarray:
    """Muestreo uniforme en superficie con fallback robusto."""
    return sample_surface_safe(mesh, n_points)

def stratified_sample_points_with_labels(mesh, n_points, labels_by_vertex):
    """Muestreo estratificado por etiqueta de vértice (robusto a clases vacías)."""
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    if verts.size == 0:
        raise ValueError("Malla sin vértices.")
    labels_by_vertex = np.asarray(labels_by_vertex).reshape(-1)
    if verts.shape[0] != labels_by_vertex.shape[0]:
        raise ValueError("labels_by_vertex no coincide en longitud con vértices.")
    unique_labels = np.unique(labels_by_vertex)
    unique_labels = unique_labels[np.isfinite(unique_labels)]
    if unique_labels.size == 0:
        # no hay etiquetas válidas; caer a muestreo global
        return sample_surface_safe(mesh, n_points), None

    n_labels = len(unique_labels)
    pts_per_label = [n_points // n_labels] * n_labels
    for i in range(n_points - sum(pts_per_label)):
        pts_per_label[i % n_labels] += 1

    pts_list = []
    lbs_list = []
    for k, label in enumerate(unique_labels):
        inds = np.where(labels_by_vertex == label)[0]
        if inds.size == 0:
            continue
        replace = (inds.size < pts_per_label[k])
        sel_inds = np.random.choice(inds, size=pts_per_label[k], replace=replace)
        pts = verts[sel_inds]
        lbs = np.full(pts.shape[0], int(label), dtype=np.int32)
        pts_list.append(pts.astype(np.float32, copy=False))
        lbs_list.append(lbs)

    if not pts_list:
        return sample_surface_safe(mesh, n_points), None
    X = np.concatenate(pts_list, axis=0).astype(np.float32, copy=False)
    Y = np.concatenate(lbs_list, axis=0).astype(np.int32, copy=False)
    # en casos borde por redondeos, ajustar a n_points
    if X.shape[0] != n_points:
        replace = (X.shape[0] < n_points)
        idx = np.random.choice(X.shape[0], size=n_points, replace=replace)
        X, Y = X[idx], Y[idx]
    return X, Y

def assign_by_vertex_nn(sampled_pts: np.ndarray,
                        vert_coords: np.ndarray,
                        labels_v: np.ndarray) -> np.ndarray:
    """Asigna etiquetas a puntos muestreados usando NN (KDTree robusto)."""
    sampled_pts = np.asarray(sampled_pts, dtype=np.float32)
    vert_coords = np.asarray(vert_coords, dtype=np.float32)
    labels_v = np.asarray(labels_v).reshape(-1)

    if vert_coords.shape[0] == 0:
        raise ValueError("No hay vértices para NN.")
    # limpiar NaNs
    if not np.isfinite(vert_coords).all():
        vert_coords = np.nan_to_num(vert_coords, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.isfinite(sampled_pts).all():
        sampled_pts = np.nan_to_num(sampled_pts, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        tree = cKDTree(vert_coords.astype(np.float32), leafsize=64)
        _, idx = tree.query(sampled_pts.astype(np.float32), k=1, workers=-1)
        return labels_v[idx]
    except Exception as e:
        # Fallback simple (O(N*M)) si KDTree falla por BLAS/Hilos
        print(f"[WARN] KDTree falló ({e}). Fallback L2 directo.", file=sys.stderr)
        # Para no explotar memoria, hacerlo por bloques
        B = 8192
        out = np.empty((sampled_pts.shape[0],), dtype=labels_v.dtype)
        for s in range(0, sampled_pts.shape[0], B):
            e = min(sampled_pts.shape[0], s + B)
            block = sampled_pts[s:e]  # (b,3)
            # distancias al conjunto de vértices
            d2 = ((block[:, None, :] - vert_coords[None, :, :]) ** 2).sum(axis=2)  # (b, Nv)
            nn = np.argmin(d2, axis=1)
            out[s:e] = labels_v[nn]
        return out

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

# -------------------- FPS --------------------

def farthest_point_sampling(points: np.ndarray, n_samples: int) -> np.ndarray:
    """
    FPS en NumPy. Devuelve un subconjunto de puntos más lejanos entre sí.
    Complejidad O(N * n_samples). Asegurar float32 y limpiar NaNs.
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] == 0:
        raise ValueError("FPS: nube vacía.")
    if not np.isfinite(pts).all():
        pts = np.nan_to_num(pts, nan=0.0, posinf=0.0, neginf=0.0)

    N = pts.shape[0]
    if n_samples >= N:
        return pts.copy()

    centroids = np.zeros((n_samples,), dtype=np.int32)
    distances = np.full((N,), np.float32(1e10), dtype=np.float32)
    farthest = np.random.randint(0, N, dtype=np.int32)

    for i in range(n_samples):
        centroids[i] = farthest
        centroid = pts[farthest, :]
        # distancias euclídeas al último centro
        diff = pts - centroid  # (N,3)
        dist = (diff * diff).sum(axis=1)
        mask = dist < distances
        distances[mask] = dist[mask]
        farthest = int(np.argmax(distances))

    return pts[centroids, :]

# -------------------- main --------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_root", required=True, help="data/Teeth_3ds/raw")
    p.add_argument("--out_struct_root", required=True, help="data/Teeth_3ds/processed_struct")
    p.add_argument("--out_flat_root", required=True, help="data/Teeth_3ds/processed_flat")
    p.add_argument("--parts", nargs="+", required=True, help="data_part_1 data_part_2 ...")
    p.add_argument("--jaws", nargs="+", default=["upper", "lower"])
    p.add_argument("--n_points", type=int, default=8192)
    p.add_argument("--copy", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=1, help="Hilos nativos para BLAS/OpenMP (recomendado=1)")
    p.add_argument("--sample_mode", choices=["global", "stratified", "fps"], default="global",
                   help="Muestreo: global (superficie), estratificado (por etiqueta de vértice) o fps (Farthest Point Sampling sobre vértices).")
    args = p.parse_args()

    # limitar hilos nativos para evitar segfaults en BLAS/OpenMP
    set_thread_env(args.threads)

    seed_all(args.seed)

    in_root = Path(args.in_root)
    out_s = Path(args.out_struct_root)
    out_f = Path(args.out_flat_root)
    out_s.mkdir(parents=True, exist_ok=True)
    out_f.mkdir(parents=True, exist_ok=True)

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

                    meta = {}
                    if meta_file is not None:
                        try:
                            with open(meta_file, "r", encoding="utf-8") as fh:
                                meta = json.load(fh)
                        except Exception as e:
                            print(f"[WARN] JSON no legible {meta_file}: {e}", file=sys.stderr)

                    labels_json = (np.array(meta.get("labels", []), dtype=np.int32)
                                   if "labels" in meta else None)
                    inst_json = (np.array(meta.get("instances", []), dtype=np.int32)
                                 if "instances" in meta else None)

                    pts = None; lbs = None; inst = None; mode = "UNKNOWN"

                    try:
                        # ---------- MODOS DE MUESTREO ----------
                        if args.sample_mode == "stratified" and labels_json is not None and len(labels_json) == len(mesh.vertices):
                            pts_surf, lbs = stratified_sample_points_with_labels(
                                mesh, int(args.n_points), labels_json.astype(np.int32)
                            )
                            pts = normalize(pts_surf)
                            inst = None
                            mode = "STRATIFIED_SAMPLE_VERT_LABELS"

                        elif args.sample_mode == "fps":
                            verts = np.asarray(mesh.vertices, dtype=np.float32)
                            if verts.shape[0] == 0:
                                raise ValueError("Malla sin vértices.")
                            pts_fps = farthest_point_sampling(verts, int(args.n_points))
                            pts = normalize(pts_fps)

                            if labels_json is not None and len(labels_json) == len(mesh.vertices):
                                lbs = assign_by_vertex_nn(pts, verts, labels_json.astype(np.int32))
                            else:
                                lbs = None
                            if inst_json is not None and len(inst_json) == len(mesh.vertices):
                                inst = assign_by_vertex_nn(pts, verts, inst_json.astype(np.int32))
                            else:
                                inst = None
                            mode = "FPS_VERTICES_BASE"

                        else:  # modo global por defecto (superficie con fallback)
                            pts_surf = sample_points(mesh, int(args.n_points))
                            pts = normalize(pts_surf)

                            verts = np.asarray(mesh.vertices, dtype=np.float32)
                            if labels_json is not None and len(labels_json) == len(verts):
                                lbs = assign_by_vertex_nn(pts_surf, verts, labels_json.astype(np.int32))
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
                        # pasar al siguiente caso sin abortar
                        del mesh
                        gc.collect()
                        continue

                    # ---------- Guardado estructurado ----------
                    try:
                        dst_struct = out_s / str(pts.shape[0]) / jaw / subj.name
                        dst_struct.mkdir(parents=True, exist_ok=True)
                        np.save(dst_struct / "point_cloud.npy", pts)
                        if lbs is not None:
                            np.save(dst_struct / "labels.npy", lbs)
                        if inst is not None:
                            np.save(dst_struct / "instances.npy", inst)
                        with open(dst_struct / "meta.json", "w", encoding="utf-8") as fh:
                            json.dump({
                                "source_mesh": str(mesh_file),
                                "source_json": str(meta_file) if meta_file else None,
                                "n_points": int(pts.shape[0]),
                                "mode": mode,
                                "sample_mode": args.sample_mode
                            }, fh, indent=2)

                        # ---------- Vista flat ----------
                        sample_type = mode.lower().replace(" ", "_")
                        dst_flat = out_f / str(pts.shape[0]) / f"{sample_type}_{jaw}" / subj.name
                        symlink_or_copy(dst_struct, dst_flat, do_copy=args.copy)
                        print(f"[OK] {part}/{jaw}/{subj.name} -> {dst_flat} ({mode}, {pts.shape[0]} pts)")
                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        print(f"[ERR] Guardado falló {subj}: {e}", file=sys.stderr)

                    # liberar memoria de esta iteración
                    del mesh, pts, lbs, inst
                    gc.collect()

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Detenido por el usuario.", file=sys.stderr)
    finally:
        gc.collect()

if __name__ == "__main__":
    main()
