#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Preprocesa mallas a nubes normalizadas y genera dos vistas:
- processed_struct/<jaw>/<case>/{point_cloud.npy, labels.npy?, instances.npy?, meta.json}
- processed_flat/<jaw>/<case> -> symlink o copia de la carpeta anterior

Soporta:
- Puntos + etiquetas directamente desde un JSON si existen y calzan (len(labels)==len(points)).
- Muestreo en superficie con Trimesh, y (si hay etiquetas por vértice en JSON) asignación por NN usando cKDTree.
"""

import argparse
import os
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import trimesh as tm
from scipy.spatial import cKDTree
import random


# -------------------- utils --------------------

def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def find_first(mesh_dir: Path):
    """Devuelve (mesh_file, meta_json_file) si existen, seleccionando de forma estable."""
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
    """Centra y escala a radio 1."""
    points = np.asarray(points, dtype=np.float32)
    c = points.mean(axis=0, keepdims=True)
    points = points - c
    m = np.linalg.norm(points, axis=1).max()
    if m > 0:
        points = points / m
    return points


def sample_points(mesh: tm.Trimesh, n_points: int) -> np.ndarray:
    """Muestrea en superficie si hay caras; si no, sobre vértices (con reemplazo si hace falta)."""
    if mesh.faces is not None and len(mesh.faces) > 0:
        pts, _ = tm.sample.sample_surface(mesh, n_points)
    else:
        v = np.asarray(mesh.vertices, dtype=np.float32)
        if v.size == 0:
            raise ValueError("Malla sin vértices.")
        idx = np.random.choice(len(v), size=n_points, replace=len(v) < n_points)
        pts = v[idx]
    return np.asarray(pts, dtype=np.float32)


def assign_by_vertex_nn(sampled_pts: np.ndarray,
                        vert_coords: np.ndarray,
                        labels_v: np.ndarray) -> np.ndarray:
    """Asigna a cada punto la etiqueta del vértice más cercano (1-NN)."""
    tree = cKDTree(vert_coords.astype(np.float32))
    _, idx = tree.query(sampled_pts.astype(np.float32), k=1)
    return labels_v[idx]


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


# -------------------- main --------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_root", required=True, help="data/3dteethseg/raw")
    p.add_argument("--out_struct_root", required=True, help="data/3dteethseg/processed_struct")
    p.add_argument("--out_flat_root", required=True, help="data/3dteethseg/processed_flat")
    p.add_argument("--parts", nargs="+", required=True, help="data_part_1 data_part_2 ...")
    p.add_argument("--jaws", nargs="+", default=["upper", "lower"])
    p.add_argument("--n_points", type=int, default=8192)
    p.add_argument("--copy", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

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

                # Cargar malla de forma robusta
                try:
                    mesh = tm.load(mesh_file, process=False, force='mesh')
                    if isinstance(mesh, tm.Scene):
                        mesh = mesh.dump().sum()
                except Exception as e:
                    print(f"[ERR] No pude cargar {mesh_file}: {e}", file=sys.stderr)
                    continue

                # Leer JSON si está disponible
                meta = {}
                if meta_file is not None:
                    try:
                        meta = json.load(open(meta_file, "r"))
                    except Exception as e:
                        print(f"[WARN] JSON no legible {meta_file}: {e}", file=sys.stderr)

                # ¿JSON trae point cloud propio?
                pts_json = None
                for key in ("points", "point_cloud", "xyz"):
                    if key in meta:
                        arr = np.array(meta[key], dtype=np.float32)
                        if arr.ndim == 2 and arr.shape[1] == 3:
                            pts_json = arr
                            break

                labels_json = (np.array(meta.get("labels", []), dtype=np.int32)
                               if "labels" in meta else None)
                inst_json = (np.array(meta.get("instances", []), dtype=np.int32)
                             if "instances" in meta else None)

                try:
                    if pts_json is not None and labels_json is not None and len(labels_json) == len(pts_json):
                        pts = normalize(pts_json.astype(np.float32))
                        lbs = labels_json.astype(np.int32)
                        inst = (inst_json.astype(np.int32)
                                if (inst_json is not None and len(inst_json) == len(pts_json)) else None)
                        mode = "JSON_POINTS_ALIGNED"
                    else:
                        pts_surf = sample_points(mesh, args.n_points)
                        if labels_json is not None and len(labels_json) == len(mesh.vertices):
                            lbs = assign_by_vertex_nn(
                                pts_surf,
                                np.asarray(mesh.vertices, dtype=np.float32),
                                labels_json.astype(np.int32)
                            )
                            if inst_json is not None and len(inst_json) == len(mesh.vertices):
                                inst = assign_by_vertex_nn(
                                    pts_surf,
                                    np.asarray(mesh.vertices, dtype=np.float32),
                                    inst_json.astype(np.int32)
                                )
                            else:
                                inst = None
                            pts = normalize(pts_surf)
                            mode = "SURF_SAMPLE_NN_FROM_VERT_LABELS"
                        else:
                            pts = normalize(pts_surf)
                            lbs = None
                            inst = None
                            mode = "SURF_SAMPLE_NO_LABELS"
                except Exception as e:
                    print(f"[ERR] Preproc falló {mesh_file}: {e}", file=sys.stderr)
                    continue

                # Guardado estructurado
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
                    "mode": mode
                }, open(dst_struct / "meta.json", "w"), indent=2)

                # Vista "flat" (symlink o copia)
                dst_flat = out_f / jaw / subj.name
                symlink_or_copy(dst_struct, dst_flat, do_copy=args.copy)
                print(f"[OK] {part}/{jaw}/{subj.name} -> {dst_struct} ({mode}, {pts.shape[0]} pts)")


if __name__ == "__main__":
    main()
