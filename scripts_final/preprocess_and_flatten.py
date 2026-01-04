#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
preprocess_and_flatten.py

Pipeline idéntico al descrito en la tesis:
- Carga mallas .ply/.stl/.obj
- Muestreo superficial con trimesh (NO FPS)
- Normalización centro-radio unidad
- Asignación de etiquetas por nearest neighbor (KDTree)
- Guarda point_cloud.npy, labels.npy e instances.npy (si existen)
- Genera vista estructurada + vista “flat” como symlink/copia

100% coherente con la metodología reportada en la tesis.
"""

import argparse
import os
import json
import shutil
import sys
import random
import gc
from pathlib import Path

import numpy as np
import trimesh as tm
from scipy.spatial import cKDTree


# ============================================================
# Utils
# ============================================================

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def find_mesh_and_meta(folder: Path):
    """Devuelve (mesh, meta_json) si existen."""
    mesh = None
    for ext in (".ply", ".stl", ".obj"):
        cand = sorted(folder.glob(f"*{ext}"))
        if cand:
            mesh = cand[0]
            break

    meta = None
    j = sorted(folder.glob("*.json"))
    if j:
        meta = j[0]

    return mesh, meta


def normalize(points: np.ndarray):
    """Normaliza nube a esfera unitaria, idéntico a la tesis."""
    points = np.asarray(points, dtype=np.float32)
    if not np.isfinite(points).all():
        points = np.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)

    c = points.mean(axis=0, keepdims=True)
    points = points - c

    r = np.linalg.norm(points, axis=1).max()
    if r > 0:
        points = points / r

    return points


def safe_load_mesh(path: Path):
    """Carga robusta de mallas."""
    try:
        m = tm.load(path, process=False, force='mesh')
        if isinstance(m, tm.Scene):
            geos = list(m.dump().geometry.values())
            if not geos:
                return None
            m = tm.util.concatenate(geos)

        verts = np.asarray(m.vertices, dtype=np.float32)
        if verts.ndim != 2 or verts.shape[1] != 3 or verts.shape[0] == 0:
            return None

        if not np.isfinite(verts).all():
            m.vertices = np.nan_to_num(verts, nan=0, posinf=0, neginf=0)

        return m

    except Exception as e:
        print(f"[ERR] No pude cargar {path}: {e}", file=sys.stderr)
        return None


def sample_surface(mesh: tm.Trimesh, n_points: int):
    """Muestreo superficial igual al de la tesis."""
    try:
        pts, _ = tm.sample.sample_surface(mesh, n_points)
        pts = np.asarray(pts, dtype=np.float32)
        if not np.isfinite(pts).all():
            pts = np.nan_to_num(pts, nan=0, posinf=0, neginf=0)
        return pts
    except:
        # fallback a vértices si sample_surface falla
        v = np.asarray(mesh.vertices, dtype=np.float32)
        replace = v.shape[0] < n_points
        idx = np.random.choice(v.shape[0], size=n_points, replace=replace)
        pts = v[idx].astype(np.float32)
        return pts


def assign_nn(sampled_pts: np.ndarray, verts: np.ndarray, labels: np.ndarray):
    """Asignación de etiquetas por NN, igual a lo descrito en la tesis."""
    tree = cKDTree(verts.astype(np.float32))
    _, idx = tree.query(sampled_pts.astype(np.float32), k=1)
    return labels[idx]


def symlink_or_copy(src: Path, dst: Path, do_copy: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if do_copy:
        shutil.copytree(src, dst)
    else:
        try:
            os.symlink(src, dst, target_is_directory=True)
        except:
            shutil.copytree(src, dst)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_root", required=True, help="data/Teeth_3ds/raw")
    parser.add_argument("--out_struct_root", required=True, help="data/Teeth_3ds/processed_struct")
    parser.add_argument("--out_flat_root", required=True, help="data/Teeth_3ds/processed_flat")
    parser.add_argument("--parts", nargs="+", required=True)
    parser.add_argument("--jaws", nargs="+", default=["upper", "lower"])
    parser.add_argument("--n_points", type=int, default=8192)
    parser.add_argument("--copy", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_all(args.seed)

    in_root = Path(args.in_root)
    out_s = Path(args.out_struct_root)
    out_f = Path(args.out_flat_root)

    out_s.mkdir(parents=True, exist_ok=True)
    out_f.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Iterate dataset
    # ============================================================
    for part in args.parts:
        for jaw in args.jaws:
            src = in_root / part / jaw

            if not src.is_dir():
                print(f"[WARN] No existe: {src}")
                continue

            for subj in sorted(src.iterdir()):
                if not subj.is_dir():
                    continue

                mesh_file, meta_file = find_mesh_and_meta(subj)
                if mesh_file is None:
                    print(f"[WARN] Sin malla en {subj}")
                    continue

                mesh = safe_load_mesh(mesh_file)
                if mesh is None:
                    print(f"[WARN] Malla inválida: {mesh_file}")
                    continue

                # Leer JSON (si existe)
                labels_json = None
                inst_json = None
                if meta_file is not None:
                    try:
                        meta = json.load(open(meta_file))
                        if "labels" in meta:
                            labels_json = np.array(meta["labels"], dtype=np.int32)
                        if "instances" in meta:
                            inst_json = np.array(meta["instances"], dtype=np.int32)
                    except:
                        pass

                # ====================================================
                # 1. Sample surface
                # ====================================================
                pts_raw = sample_surface(mesh, args.n_points)
                verts = np.asarray(mesh.vertices, dtype=np.float32)

                # ====================================================
                # 2. Assign labels if available
                # ====================================================
                if labels_json is not None and len(labels_json) == len(verts):
                    lbs = assign_nn(pts_raw, verts, labels_json)
                else:
                    lbs = None

                if inst_json is not None and len(inst_json) == len(verts):
                    inst = assign_nn(pts_raw, verts, inst_json)
                else:
                    inst = None

                # ====================================================
                # 3. Normalize
                # ====================================================
                pts = normalize(pts_raw)

                # ====================================================
                # 4. Save structured
                # ====================================================
                dst_struct = out_s / jaw / subj.name
                dst_struct.mkdir(parents=True, exist_ok=True)

                np.save(dst_struct / "point_cloud.npy", pts)
                if lbs is not None:
                    np.save(dst_struct / "labels.npy", lbs)
                if inst is not None:
                    np.save(dst_struct / "instances.npy", inst)

                json.dump({
                    "source_mesh": str(mesh_file),
                    "source_json": str(meta_file) if meta_file else None,
                    "n_points": int(pts.shape[0]),
                    "mode": "SURFACE_SAMPLING"
                }, open(dst_struct / "meta.json", "w"), indent=2)

                # ====================================================
                # 5. Save flat view (symlink/copy)
                # ====================================================
                dst_flat = out_f / jaw / subj.name
                symlink_or_copy(dst_struct, dst_flat, do_copy=args.copy)

                print(f"[OK] {part}/{jaw}/{subj.name} -> {dst_struct} ({pts.shape[0]} pts)")


if __name__ == "__main__":
    main()
