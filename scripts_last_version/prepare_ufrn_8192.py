#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import json
import numpy as np
import trimesh as tm

def normalize(points):
    points = points.astype(np.float32)
    center = points.mean(axis=0, keepdims=True)
    points = points - center
    scale = np.linalg.norm(points, axis=1).max()
    points = points / (scale + 1e-8)
    return points, center.astype(np.float32), float(scale)

def sample_surface(mesh_path, n_points=8192, seed=42):
    np.random.seed(seed)
    mesh = tm.load(mesh_path, process=False, force="mesh")

    if isinstance(mesh, tm.Scene):
        mesh = tm.util.concatenate(tuple(mesh.geometry.values()))

    points, face_idx = tm.sample.sample_surface(mesh, n_points)
    points = points.astype(np.float32)

    points_norm, center, scale = normalize(points)

    return points_norm, points, face_idx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_points", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_norm, X_raw, face_idx = sample_surface(args.mesh, args.n_points, args.seed)

    np.save(out_dir / "point_cloud_8192_norm.npy", X_norm)
    np.save(out_dir / "point_cloud_8192_raw.npy", X_raw)
    np.save(out_dir / "face_idx_8192.npy", face_idx)

    meta = {
        "mesh": args.mesh,
        "n_points": args.n_points,
        "seed": args.seed,
        "note": "UFRN upper_full sampled to 8192 and normalized to unit sphere"
    }

    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("[OK] Guardado en:", out_dir)
    print("[OK] X_norm:", X_norm.shape)
    print("[OK] X_raw:", X_raw.shape)

if __name__ == "__main__":
    main()