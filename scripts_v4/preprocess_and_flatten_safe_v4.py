#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Versión v4 FINAL:
--------------------------------------------------------------
Basada en preprocess_and_flatten_safe original, pero añade:
 - Cálculo de normales unitarias y curvatura por punto (λ3/Σλ).
 - Guardado adicional de point_cloud_feats.npy (N,7) = [x,y,z,nx,ny,nz,kappa].
 - Compatible con flujo posterior (merged, augmentation, train v11).
 - No altera point_cloud.npy (solo XYZ normalizado).
--------------------------------------------------------------
Autor: Adaptado para Tesis de H. Taucare
"""

import argparse, os, json, sys, gc, random, warnings
from pathlib import Path
import numpy as np
import trimesh as tm
from scipy.spatial import cKDTree


# ==============================================================
# === Utilidades generales =====================================
# ==============================================================

def set_thread_env(n_threads: int):
    """Configura número de hilos para BLAS/OpenMP (control de rendimiento)."""
    n = str(max(1, int(n_threads)))
    os.environ.setdefault("OMP_NUM_THREADS", n)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", n)
    os.environ.setdefault("MKL_NUM_THREADS", n)
    os.environ.setdefault("NUMEXPR_NUM_THREADS", n)

def seed_all(seed: int = 42):
    """Inicializa RNGs globales."""
    random.seed(seed)
    np.random.seed(seed)

def find_first(mesh_dir: Path):
    """Encuentra el primer archivo de malla (obj/ply/stl) y su JSON asociado."""
    mesh = None; meta = None
    for ext in (".obj", ".ply", ".stl"):
        cand = sorted(mesh_dir.glob(f"*{ext}"))
        if cand:
            mesh = cand[0]; break
    j = sorted(mesh_dir.glob("*.json"))
    if j:
        meta = j[0]
    return mesh, meta

def normalize(points: np.ndarray) -> np.ndarray:
    """Centra y escala nube a esfera unitaria (centro=0, radio=1)."""
    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        return points
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
    """Valida geometría de malla (coordenadas finitas y formato correcto)."""
    try:
        v = np.asarray(mesh.vertices, dtype=np.float32)
        if v.ndim != 2 or v.shape[1] != 3 or v.shape[0] == 0:
            return False
        if not np.isfinite(v).all():
            return False
        return True
    except Exception:
        return False

def load_mesh_safe(mesh_path: Path):
    """Carga malla 3D de forma robusta, tolerando escenas o errores parciales."""
    try:
        m = tm.load(mesh_path, process=False, force='mesh')
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
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        m.vertices = v
        return m
    except Exception as e:
        print(f"[ERR] Carga fallida {mesh_path}: {e}", file=sys.stderr)
        return None


# ==============================================================
# === Funciones geométricas ====================================
# ==============================================================

def compute_normals_curvature(X: np.ndarray, k: int = 20):
    """
    Calcula normales unitarias y curvatura aproximada (λ3 / Σλ).
    X: (N,3)
    Devuelve:
      normals (N,3), curvature (N,1)
    """
    tree = cKDTree(X)
    N = X.shape[0]
    normals = np.zeros((N, 3), np.float32)
    curv = np.zeros((N, 1), np.float32)
    for i in range(N):
        _, idx = tree.query(X[i], k=min(k, N))
        P = X[idx] - X[i]
        C = P.T @ P / max(1, P.shape[0])
        w, V = np.linalg.eigh(C)
        if w.sum() <= 1e-12:
            normals[i] = [0, 0, 1]; curv[i] = 0; continue
        n = V[:, 0]
        if n[2] < 0: n = -n
        normals[i] = n / (np.linalg.norm(n) + 1e-8)
        curv[i] = w[0] / (w.sum() + 1e-8)
    return normals, curv


# ==============================================================
# === Sampling (Superficie y FPS) ===============================
# ==============================================================

def sample_surface_safe(mesh: tm.Trimesh, n_points: int) -> np.ndarray:
    """Muestreo uniforme sobre superficie de la malla."""
    v = np.asarray(mesh.vertices, dtype=np.float32)
    f = np.asarray(mesh.faces) if (hasattr(mesh, "faces") and mesh.faces is not None) else None
    if f is not None and f.size > 0:
        try:
            pts, _ = tm.sample.sample_surface(mesh, int(n_points))
            return np.asarray(pts, np.float32)
        except Exception as e:
            print(f"[WARN] sample_surface falló ({e}). Caigo a vértices.", file=sys.stderr)
    replace = (v.shape[0] < n_points)
    idx = np.random.choice(v.shape[0], size=int(n_points), replace=replace)
    return v[idx].astype(np.float32)

def farthest_point_sampling(points: np.ndarray, n_samples: int) -> np.ndarray:
    """FPS: selecciona puntos maximizando la distancia mínima (nube más dispersa posible)."""
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] == 0:
        raise ValueError("FPS: nube vacía.")
    if n_samples >= pts.shape[0]:
        return pts.copy()
    N = pts.shape[0]
    centroids = np.zeros((n_samples,), dtype=np.int32)
    distances = np.full((N,), np.float32(1e10))
    farthest = np.random.randint(0, N)
    for i in range(n_samples):
        centroids[i] = farthest
        diff = pts - pts[farthest, :]
        dist = (diff * diff).sum(axis=1)
        distances = np.minimum(distances, dist)
        farthest = np.argmax(distances)
    return pts[centroids, :]


# ==============================================================
# === Main ======================================================
# ==============================================================

def main():
    p = argparse.ArgumentParser(description="Preprocesa mallas 3D dentales (v4) para generar nubes FPS con features geométricos.")
    p.add_argument("--in_root", required=True, help="Directorio raíz de datos RAW (por ej. data/Teeth_3ds/raw)")
    p.add_argument("--out_struct_root", required=True, help="Directorio de salida estructurado (data/Teeth_3ds/processed_struct)")
    p.add_argument("--out_flat_root", required=True, help="Directorio de salida plano (data/Teeth_3ds/processed_flat)")
    p.add_argument("--parts", nargs="+", required=True, help="Subcarpetas de entrada, ej: data_part_1 data_part_2 ... data_part_7")
    p.add_argument("--jaws", nargs="+", default=["upper", "lower"], help="Maxilares a procesar (por defecto upper y lower).")
    p.add_argument("--n_points", type=int, default=8192, help="Número de puntos por nube (default=8192).")
    p.add_argument("--seed", type=int, default=42, help="Semilla RNG para reproducibilidad.")
    p.add_argument("--threads", type=int, default=1, help="Hilos nativos para BLAS/OpenMP (recomendado=1).")
    p.add_argument("--sample_mode", choices=["global", "stratified", "fps"], default="global",
                   help="Modo de muestreo: global (superficie), stratified (por etiqueta de vértice) o fps (Farthest Point Sampling sobre vértices).")
    args = p.parse_args()

    set_thread_env(args.threads)
    seed_all(args.seed)

    in_root = Path(args.in_root)
    out_s = Path(args.out_struct_root)
    out_f = Path(args.out_flat_root)
    out_s.mkdir(parents=True, exist_ok=True)
    out_f.mkdir(parents=True, exist_ok=True)

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

                # --- Muestreo FPS y normalización ---
                verts = np.asarray(mesh.vertices, np.float32)
                pts = farthest_point_sampling(verts, args.n_points)
                pts = normalize(pts)

                # --- Cálculo de features geométricos ---
                normals, curv = compute_normals_curvature(pts, k=20)
                X_feat = np.concatenate([pts, normals, curv], axis=1).astype(np.float32)  # (N,7)

                # --- Guardado estructurado ---
                dst_struct = out_s / str(args.n_points) / f"fps_vertices_base_{jaw}" / subj.name
                dst_struct.mkdir(parents=True, exist_ok=True)
                np.save(dst_struct / "point_cloud.npy", pts)
                np.save(dst_struct / "point_cloud_feats.npy", X_feat)

                meta = {
                    "source_mesh": str(mesh_file),
                    "n_points": int(pts.shape[0]),
                    "mode": "FPS_VERTICES_BASE",
                    "sample_mode": args.sample_mode,
                    "features": ["x","y","z","nx","ny","nz","curv"]
                }
                with open(dst_struct / "meta.json","w",encoding="utf-8") as fh:
                    json.dump(meta, fh, indent=2)

                # --- Link simbólico plano ---
                dst_flat = out_f / str(args.n_points) / f"fps_vertices_base_{jaw}" / subj.name
                dst_flat.parent.mkdir(parents=True, exist_ok=True)
                if not dst_flat.exists():
                    os.symlink(dst_struct, dst_flat, target_is_directory=True)

                print(f"[OK] {part}/{jaw}/{subj.name} ({pts.shape[0]} pts con features)")
                del mesh, pts, X_feat; gc.collect()


if __name__ == "__main__":
    main()
