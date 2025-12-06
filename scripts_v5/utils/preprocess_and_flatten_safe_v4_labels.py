#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocess v4 FINAL con labels:
--------------------------------------------------------------
 - FPS sobre vértices (8192 puntos)
 - XYZ normalizado
 - Normales + Curvatura (λ3/Σλ)
 - Guarda point_cloud.npy y point_cloud_feats.npy
 - LEE labels e instances DESDE meta.json del raw
 - Reasigna etiquetas POST-FPS vía NN
 - Guarda labels.npy / instances.npy
--------------------------------------------------------------
Compatible con merge v4, augmentation v4 y train v11.
"""

import argparse, os, json, sys, gc, random, warnings
from pathlib import Path
import numpy as np
import trimesh as tm
from scipy.spatial import cKDTree


# ==============================================================
# UTILIDADES
# ==============================================================

def set_thread_env(n_threads: int):
    n = str(max(1, int(n_threads)))
    os.environ.setdefault("OMP_NUM_THREADS", n)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", n)
    os.environ.setdefault("MKL_NUM_THREADS", n)
    os.environ.setdefault("NUMEXPR_NUM_THREADS", n)

def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

def find_first(mesh_dir: Path):
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
    pts = np.asarray(points, dtype=np.float32)
    pts = np.nan_to_num(pts)
    pts -= pts.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(pts, axis=1)
    m = norms.max() if norms.size > 0 else 1.0
    if m > 0:
        pts /= m
    return pts

def load_mesh_safe(mesh_path: Path):
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
        v = np.nan_to_num(v)
        m.vertices = v
        return m
    except:
        return None


# ==============================================================
# GEOMETRÍA
# ==============================================================

def compute_normals_curvature(X: np.ndarray, k: int = 20):
    tree = cKDTree(X)
    N = X.shape[0]
    normals = np.zeros((N,3), np.float32)
    curv    = np.zeros((N,1), np.float32)

    for i in range(N):
        _, idx = tree.query(X[i], k=min(k, N))
        P = X[idx] - X[i]
        C = (P.T @ P) / max(P.shape[0],1)
        w, V = np.linalg.eigh(C)
        if w.sum() <= 1e-12:
            normals[i] = [0,0,1]; curv[i] = 0; continue
        n = V[:,0]
        if n[2] < 0: n = -n
        normals[i] = n / (np.linalg.norm(n)+1e-8)
        curv[i] = w[0] / (w.sum()+1e-8)

    return normals, curv

def farthest_point_sampling(points: np.ndarray, n_samples: int):
    pts = np.asarray(points, dtype=np.float32)
    N = pts.shape[0]
    if n_samples >= N:
        return pts.copy()

    centroids = np.zeros((n_samples,), np.int32)
    dist = np.full((N,), 1e10, np.float32)
    far = np.random.randint(0, N)

    for i in range(n_samples):
        centroids[i] = far
        d = np.sum((pts - pts[far])**2, axis=1)
        dist = np.minimum(dist, d)
        far = np.argmax(dist)

    return pts[centroids], centroids


def assign_by_vertex_nn(sampled_pts, verts, labels_v):
    tree = cKDTree(verts.astype(np.float32))
    _, idx = tree.query(sampled_pts.astype(np.float32), k=1, workers=-1)
    return labels_v[idx]


# ==============================================================
# MAIN
# ==============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", required=True)
    ap.add_argument("--out_struct_root", required=True)
    ap.add_argument("--out_flat_root", required=True)
    ap.add_argument("--parts", nargs="+", required=True)
    ap.add_argument("--jaws", nargs="+", default=["upper","lower"])
    ap.add_argument("--n_points", type=int, default=8192)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_thread_env(args.threads)
    seed_all(args.seed)

    in_root = Path(args.in_root)
    out_s   = Path(args.out_struct_root)
    out_f   = Path(args.out_flat_root)
    out_s.mkdir(parents=True, exist_ok=True)
    out_f.mkdir(parents=True, exist_ok=True)

    for part in args.parts:
        for jaw in args.jaws:
            src = in_root / part / jaw
            if not src.is_dir():
                print(f"[WARN] No existe: {src}")
                continue

            for subj in sorted(src.iterdir()):
                if not subj.is_dir():
                    continue

                mesh_file, meta_file = find_first(subj)
                if mesh_file is None:
                    continue

                mesh = load_mesh_safe(mesh_file)
                if mesh is None:
                    continue

                verts = np.asarray(mesh.vertices, np.float32)

                # --- FPS + normalización ---
                pts, fps_idx = farthest_point_sampling(verts, args.n_points)
                pts = normalize(pts)

                # --- Features geométricos ---
                normals, curv = compute_normals_curvature(pts)
                X_feat = np.concatenate([pts, normals, curv], axis=1).astype(np.float32)

                # --- Etiquetas desde meta.json ---
                labels = None
                instances = None

                if meta_file is not None:
                    try:
                        meta_raw = json.load(open(meta_file))

                        if "labels" in meta_raw:
                            lv = np.asarray(meta_raw["labels"], np.int32)
                            if lv.shape[0] == verts.shape[0]:
                                labels = lv[fps_idx]
                        if "instances" in meta_raw:
                            iv = np.asarray(meta_raw["instances"], np.int32)
                            if iv.shape[0] == verts.shape[0]:
                                instances = iv[fps_idx]
                    except Exception as e:
                        print(f"[WARN] meta.json inválido: {e}")

                # --- Guardado STRUCT ---
                dst_struct = out_s / str(args.n_points) / f"fps_vertices_base_{jaw}" / subj.name
                dst_struct.mkdir(parents=True, exist_ok=True)

                np.save(dst_struct / "point_cloud.npy", pts)
                np.save(dst_struct / "point_cloud_feats.npy", X_feat)

                if labels is not None:
                    np.save(dst_struct / "labels.npy", labels)
                if instances is not None:
                    np.save(dst_struct / "instances.npy", instances)

                json.dump({
                    "source_mesh": str(mesh_file),
                    "n_points": args.n_points,
                    "features": ["x","y","z","nx","ny","nz","curv"],
                    "fps_idx": fps_idx.tolist()
                }, open(dst_struct / "meta.json","w"), indent=2)

                # --- Symlink FLAT ---
                dst_flat = out_f / str(args.n_points) / f"fps_vertices_base_{jaw}" / subj.name
                dst_flat.parent.mkdir(parents=True, exist_ok=True)
                if not dst_flat.exists():
                    os.symlink(dst_struct, dst_flat, target_is_directory=True)

                print(f"[OK] {part}/{jaw}/{subj.name} con feats + labels")

                del mesh, pts, X_feat, labels, instances
                gc.collect()


if __name__ == "__main__":
    main()
