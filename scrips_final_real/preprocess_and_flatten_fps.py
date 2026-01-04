#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocesamiento 'paper-like' con Farthest Point Sampling (FPS) y PCA alignment opcional.
Basado en el pipeline de Qi et al. (PointNet, CVPR 2017) y Akahori et al. (ToothSegNet, 2023).

Convierte mallas en nubes normalizadas (centro en origen, radio unitario),
submuestreadas a N puntos vía FPS, y opcionalmente alineadas por PCA.

Salida:
  processed_struct/<N>/<jaw>/<sample>/point_cloud.npy (+labels/instances)
  processed_flat/<N>/<jaw>/<sample>/ (copias o symlinks)
"""

import argparse, os, json, sys, gc, shutil, random, warnings
from pathlib import Path
import numpy as np
import trimesh as tm
from scipy.spatial import cKDTree

# ------------------------- Utilidades básicas -------------------------

def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

def normalize(points: np.ndarray) -> np.ndarray:
    """Centrado y escalado a esfera unitaria."""
    pts = np.asarray(points, dtype=np.float32)
    if pts.size == 0:
        return pts
    if not np.isfinite(pts).all():
        pts = np.nan_to_num(pts, nan=0.0)
    pts -= pts.mean(axis=0, keepdims=True)
    norm = np.linalg.norm(pts, axis=1)
    m = norm.max() if norm.size else 1.0
    if m > 0:
        pts /= m
    return pts

def pca_align(points: np.ndarray) -> np.ndarray:
    """Alinea la nube según los ejes principales (PCA)."""
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] < 3:
        return pts
    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    aligned = np.dot(pts, eigvecs)
    return aligned.astype(np.float32)

def load_mesh_safe(mesh_path: Path):
    """Carga robusta de malla .ply/.obj/.stl"""
    try:
        m = tm.load(mesh_path, process=False, force='mesh')
        if isinstance(m, tm.Scene):
            geos = list(m.dump().geometry.values())
            if not geos:
                return None
            m = tm.util.concatenate(geos)
        if not isinstance(m, tm.Trimesh):
            return None
        v = np.asarray(m.vertices, dtype=np.float32)
        if v.ndim != 2 or v.shape[1] != 3 or v.shape[0] == 0:
            return None
        if not np.isfinite(v).all():
            m.vertices = np.nan_to_num(v, nan=0.0)
        return m
    except Exception as e:
        print(f"[ERR] Carga fallida {mesh_path}: {e}", file=sys.stderr)
        return None

def farthest_point_sampling(points: np.ndarray, n_samples: int) -> np.ndarray:
    """FPS puro en NumPy (O(N * n_samples))."""
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] == 0:
        raise ValueError("FPS: nube vacía.")
    if not np.isfinite(pts).all():
        pts = np.nan_to_num(pts, nan=0.0)
    N = pts.shape[0]
    if n_samples >= N:
        return pts.copy()
    centroids = np.zeros((n_samples,), dtype=np.int32)
    dist = np.full((N,), 1e10, dtype=np.float32)
    farthest = np.random.randint(0, N)
    for i in range(n_samples):
        centroids[i] = farthest
        centroid = pts[farthest]
        diff = pts - centroid
        d = (diff * diff).sum(axis=1)
        dist = np.minimum(dist, d)
        farthest = int(np.argmax(dist))
    return pts[centroids]

def assign_by_vertex_nn(sampled_pts, verts, labels_v):
    """Asigna etiquetas de vértices a puntos muestreados vía NN."""
    tree = cKDTree(verts.astype(np.float32), leafsize=64)
    _, idx = tree.query(sampled_pts.astype(np.float32), k=1, workers=-1)
    return labels_v[idx]

# ------------------------- Guardado -------------------------

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

# ------------------------- Main -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", required=True)
    ap.add_argument("--out_struct_root", required=True)
    ap.add_argument("--out_flat_root", required=True)
    ap.add_argument("--parts", nargs="+", required=True)
    ap.add_argument("--jaws", nargs="+", default=["upper", "lower"])
    ap.add_argument("--n_points", type=int, default=8192)
    ap.add_argument("--pca_align", action="store_true",
                    help="Aplica alineación PCA tras normalización.")
    ap.add_argument("--copy", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_all(args.seed)
    in_root, out_s, out_f = Path(args.in_root), Path(args.out_struct_root), Path(args.out_flat_root)
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

                mesh_files = list(subj.glob("*.ply")) + list(subj.glob("*.stl")) + list(subj.glob("*.obj"))
                if not mesh_files:
                    continue
                mesh = load_mesh_safe(mesh_files[0])
                if mesh is None:
                    continue

                verts = np.asarray(mesh.vertices, dtype=np.float32)
                pts = farthest_point_sampling(verts, args.n_points)
                if args.pca_align:
                    pts = pca_align(pts)
                pts = normalize(pts)

                meta_path = subj / "meta.json"
                labels = None
                instances = None
                if meta_path.exists():
                    try:
                        meta = json.load(open(meta_path))
                        if "labels" in meta:
                            lv = np.array(meta["labels"], dtype=np.int32)
                            if len(lv) == len(verts):
                                labels = assign_by_vertex_nn(pts, verts, lv)
                        if "instances" in meta:
                            iv = np.array(meta["instances"], dtype=np.int32)
                            if len(iv) == len(verts):
                                instances = assign_by_vertex_nn(pts, verts, iv)
                    except Exception as e:
                        print(f"[WARN] meta.json inválido en {meta_path}: {e}", file=sys.stderr)

                dst_struct = out_s / str(args.n_points) / jaw / subj.name
                dst_struct.mkdir(parents=True, exist_ok=True)
                np.save(dst_struct / "point_cloud.npy", pts)
                if labels is not None:
                    np.save(dst_struct / "labels.npy", labels)
                if instances is not None:
                    np.save(dst_struct / "instances.npy", instances)

                json.dump({
                    "source_mesh": str(mesh_files[0]),
                    "n_points": args.n_points,
                    "mode": "FPS",
                    "pca_aligned": args.pca_align
                }, open(dst_struct / "meta.json", "w"), indent=2)

                dst_flat = out_f / str(args.n_points) / ("fps_pca" if args.pca_align else "fps") / jaw / subj.name
                symlink_or_copy(dst_struct, dst_flat, do_copy=args.copy)
                print(f"[OK] {part}/{jaw}/{subj.name} ({'FPS+PCA' if args.pca_align else 'FPS'})")

                del mesh, verts, pts, labels, instances
                gc.collect()

if __name__ == "__main__":
    main()
