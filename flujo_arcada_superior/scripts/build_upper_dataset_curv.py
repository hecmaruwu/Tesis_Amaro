#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extrae nubes de puntos (8192) + normales + curvaturas desde todas las partes raw/.
Soporta Open3D o Trimesh automáticamente.
"""
import argparse, os, sys
from pathlib import Path
import numpy as np
import trimesh

# --- intentar importar open3d ---
try:
    import open3d as o3d
    HAS_O3D = True
    print("[INFO] Open3D detectado, usándolo para curvaturas más precisas.")
except Exception:
    HAS_O3D = False
    print("[WARN] Open3D no disponible. Usando Trimesh (modo compatible).")

# ---------------------------------------------------
def sample_with_trimesh(mesh, n_points=8192):
    pts, _ = trimesh.sample.sample_surface_even(mesh, n_points)
    normals = mesh.face_normals[_]
    if normals.shape[0] != n_points:
        normals = mesh.face_normals[np.random.choice(len(mesh.face_normals), n_points)]
    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
    curvature = np.zeros((n_points, 2), dtype=np.float32)
    mesh_curv = mesh.vertex_defects / (2 * np.pi)
    idx = np.random.choice(len(mesh_curv), n_points)
    curvature[:, 0] = mesh_curv[idx]         # curvatura media aprox.
    curvature[:, 1] = mesh_curv[idx] ** 2    # "curvatura gaussiana" proxy
    return np.concatenate([pts, normals, curvature], axis=1)

def sample_with_open3d(mesh, n_points=8192):
    pcd = mesh.sample_points_uniformly(number_of_points=n_points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30))
    pts = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    curv = np.asarray(mesh.vertex_curvature()) if hasattr(mesh, "vertex_curvature") else np.zeros((len(pts),2))
    if curv.shape[0] != n_points:
        curv = np.zeros((n_points,2))
    return np.concatenate([pts, normals, curv], axis=1)

# ---------------------------------------------------
def process_file(path: Path, n_points: int, out_dir: Path):
    try:
        mesh = trimesh.load_mesh(path, process=False)
        if mesh.is_empty:
            print(f"[SKIP] {path.name} vacío."); return
        data = sample_with_open3d(mesh, n_points) if HAS_O3D else sample_with_trimesh(mesh, n_points)
        data = data.astype(np.float32)
        np.savez_compressed(out_dir / f"{path.stem}.npz", X=data)
        print(f"[OK] {path.name} → {data.shape}")
    except Exception as e:
        print(f"[ERR] {path.name}: {e}", file=sys.stderr)

# ---------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True, help="Carpeta raíz raw/")
    ap.add_argument("--output_dir", required=True, help="Salida npz/")
    ap.add_argument("--num_points", type=int, default=8192)
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stls = [p for p in base_dir.rglob("*.stl")] + [p for p in base_dir.rglob("*.obj")]
    print(f"[INFO] Archivos STL/OBJ detectados: {len(stls)}")
    for p in stls:
        process_file(p, args.num_points, out_dir)

if __name__ == "__main__":
    main()
