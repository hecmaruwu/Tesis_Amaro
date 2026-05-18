#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_ufrn_pseudolabels_all_patients.py

Genera pseudo-labels clínicas aproximadas del diente 21 removido
para TODOS los pacientes UFRN usando:

upper_full.stl
vs
upper_rec_21.stl

Pipeline:
1) sample surface de upper_full y upper_rec_21
2) distancia full -> rec_21
3) threshold por percentil
4) DBSCAN casero con scipy.cKDTree
5) quedarse con cluster principal
6) guardar pseudo-label del diente 21 removido

Evita sklearn.DBSCAN porque en este entorno existe conflicto NumPy/sklearn.
"""

import argparse
import json
from pathlib import Path
from collections import deque

import numpy as np
import trimesh
from scipy.spatial import cKDTree


# ============================================================
# Utils
# ============================================================

def load_mesh_safe(mesh_path: Path):
    mesh = trimesh.load(mesh_path, process=False, force="mesh")

    if isinstance(mesh, trimesh.Scene):
        geoms = list(mesh.geometry.values())
        if len(geoms) == 0:
            raise RuntimeError(f"Scene vacía: {mesh_path}")
        mesh = trimesh.util.concatenate(geoms)

    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"No se pudo cargar como Trimesh: {mesh_path}")

    if mesh.vertices is None or len(mesh.vertices) == 0:
        raise RuntimeError(f"Malla sin vértices: {mesh_path}")

    return mesh


def sample_mesh(mesh_path: Path, n_points: int, seed: int):
    np.random.seed(int(seed))

    mesh = load_mesh_safe(mesh_path)

    try:
        pts, _ = trimesh.sample.sample_surface(mesh, int(n_points))
    except Exception:
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        replace = verts.shape[0] < int(n_points)
        idx = np.random.choice(verts.shape[0], int(n_points), replace=replace)
        pts = verts[idx]

    pts = np.asarray(pts, dtype=np.float32)

    if not np.isfinite(pts).all():
        pts = np.nan_to_num(pts, nan=0.0, posinf=0.0, neginf=0.0)

    return pts


def compute_removed_region(full_pts, rec_pts, percentile):
    full_pts = np.asarray(full_pts, dtype=np.float32)
    rec_pts = np.asarray(rec_pts, dtype=np.float32)

    tree = cKDTree(rec_pts)
    dists, _ = tree.query(full_pts, k=1)

    thr = float(np.percentile(dists, float(percentile)))
    mask = dists >= thr

    removed_raw = np.asarray(full_pts[mask], dtype=np.float32)
    dists = np.asarray(dists, dtype=np.float32)

    return removed_raw, thr, dists


def save_ply_points(path: Path, pts: np.ndarray, color=(0, 255, 0)):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pts = np.asarray(pts, dtype=np.float32)

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for x, y, z in pts:
            f.write(f"{x} {y} {z} {color[0]} {color[1]} {color[2]}\n")


def run_dbscan(points, eps, min_samples, min_cluster_size):
    """
    DBSCAN simple usando scipy.cKDTree.
    Evita sklearn.DBSCAN por conflicto NumPy/sklearn en este entorno.

    labels:
      -1 = ruido
       0,1,2... = clusters
    """
    points = np.asarray(points, dtype=np.float64, order="C")

    if points.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float32), {
            "num_clusters": 0,
            "noise_points": 0,
            "clusters": [],
            "selected_label": None,
        }

    n = points.shape[0]

    labels = np.full(n, -1, dtype=np.int32)
    visited = np.zeros(n, dtype=bool)

    tree = cKDTree(points)
    neighbors = tree.query_ball_point(points, r=float(eps))

    cluster_id = 0

    for i in range(n):
        if visited[i]:
            continue

        visited[i] = True
        neigh_i = neighbors[i]

        if len(neigh_i) < int(min_samples):
            labels[i] = -1
            continue

        labels[i] = cluster_id
        queue = deque(neigh_i)

        while queue:
            p = queue.popleft()

            if not visited[p]:
                visited[p] = True
                neigh_p = neighbors[p]

                if len(neigh_p) >= int(min_samples):
                    for q in neigh_p:
                        if not visited[q]:
                            queue.append(q)

            if labels[p] == -1:
                labels[p] = cluster_id

        cluster_id += 1

    valid = labels >= 0
    unique, counts = np.unique(labels[valid], return_counts=True)

    clusters_info = []
    for lbl, cnt in zip(unique, counts):
        clusters_info.append({
            "label": int(lbl),
            "size": int(cnt)
        })

    clusters_info = sorted(
        clusters_info,
        key=lambda x: x["size"],
        reverse=True
    )

    if len(clusters_info) == 0:
        info = {
            "num_clusters": 0,
            "noise_points": int((labels == -1).sum()),
            "clusters": [],
            "selected_label": None,
        }
        return np.empty((0, 3), dtype=np.float32), info

    selected_label = None
    filtered = np.empty((0, 3), dtype=np.float32)

    for c in clusters_info:
        if c["size"] >= int(min_cluster_size):
            selected_label = c["label"]
            filtered = points[labels == selected_label].astype(np.float32)
            break

    info = {
        "num_clusters": int(len(clusters_info)),
        "noise_points": int((labels == -1).sum()),
        "clusters": clusters_info,
        "selected_label": None if selected_label is None else int(selected_label),
    }

    return filtered, info


def process_one_patient(
    paciente_dir: Path,
    out_root: Path,
    n_full: int,
    n_rec: int,
    percentile: float,
    eps: float,
    min_samples: int,
    min_cluster_size: int,
    seed: int,
):
    pid = paciente_dir.name
    stl_dir = paciente_dir / "stl"

    full_stl = stl_dir / "upper_full.stl"
    rec_stl = stl_dir / "upper_rec_21.stl"

    if not full_stl.exists():
        raise FileNotFoundError(full_stl)

    if not rec_stl.exists():
        raise FileNotFoundError(rec_stl)

    patient_out = out_root / pid
    patient_out.mkdir(parents=True, exist_ok=True)

    print("[INFO] Sampleando full...")
    full_pts = sample_mesh(
        full_stl,
        n_full,
        seed
    )

    print("[INFO] Sampleando rec_21...")
    rec_pts = sample_mesh(
        rec_stl,
        n_rec,
        seed + 1
    )

    print("[INFO] Calculando diferencia geométrica...")
    removed_raw, thr, dists = compute_removed_region(
        full_pts,
        rec_pts,
        percentile
    )

    print("[INFO] Ejecutando DBSCAN cKDTree...")
    removed_dbscan, dbscan_info = run_dbscan(
        removed_raw,
        eps=eps,
        min_samples=min_samples,
        min_cluster_size=min_cluster_size
    )

    np.save(
        patient_out / "full_sample_points.npy",
        np.asarray(full_pts, dtype=np.float32)
    )

    np.save(
        patient_out / "rec_sample_points.npy",
        np.asarray(rec_pts, dtype=np.float32)
    )

    np.save(
        patient_out / "dist_full_to_rec.npy",
        np.asarray(dists, dtype=np.float32)
    )

    np.save(
        patient_out / "raw_removed_candidates.npy",
        np.asarray(removed_raw, dtype=np.float32)
    )

    np.save(
        patient_out / "gt_removed_d21_dbscan.npy",
        np.asarray(removed_dbscan, dtype=np.float32)
    )

    save_ply_points(
        patient_out / "gt_removed_d21_dbscan.ply",
        removed_dbscan,
        color=(0, 255, 0)
    )

    summary = {
        "patient_id": pid,
        "full_stl": str(full_stl),
        "rec_stl": str(rec_stl),
        "n_full": int(n_full),
        "n_rec": int(n_rec),
        "percentile": float(percentile),
        "distance_threshold": float(thr),
        "eps": float(eps),
        "min_samples": int(min_samples),
        "min_cluster_size": int(min_cluster_size),
        "raw_removed_points": int(removed_raw.shape[0]),
        "dbscan_points": int(removed_dbscan.shape[0]),
        "dbscan_num_clusters": int(dbscan_info["num_clusters"]),
        "dbscan_noise_points": int(dbscan_info["noise_points"]),
        "dbscan_selected_label": dbscan_info["selected_label"],
        "dbscan_clusters": dbscan_info["clusters"],
    }

    with open(
        patient_out / "summary.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)

    print("[OK]", pid)
    print(json.dumps(summary, indent=2))

    return summary


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--ufrn_root", required=True)
    ap.add_argument("--out_root", required=True)

    ap.add_argument("--n_full", type=int, default=200000)
    ap.add_argument("--n_rec", type=int, default=200000)

    ap.add_argument("--percentile", type=float, default=97.5)
    ap.add_argument("--eps", type=float, default=1.25)
    ap.add_argument("--min_samples", type=int, default=20)
    ap.add_argument("--min_cluster_size", type=int, default=100)

    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    ufrn_root = Path(args.ufrn_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    pacientes = sorted(
        [
            p for p in ufrn_root.iterdir()
            if p.is_dir() and p.name.startswith("paciente_")
        ],
        key=lambda p: int(p.name.replace("paciente_", ""))
        if p.name.replace("paciente_", "").isdigit()
        else 10**9
    )

    print("=" * 80)
    print(f"[INFO] Pacientes encontrados: {len(pacientes)}")
    print("[INFO] ufrn_root:", ufrn_root)
    print("[INFO] out_root:", out_root)
    print("=" * 80)

    ok = []
    failed = []

    for paciente_dir in pacientes:
        pid = paciente_dir.name

        print("=" * 80)
        print(f"[INFO] Procesando {pid}")

        try:
            summary = process_one_patient(
                paciente_dir=paciente_dir,
                out_root=out_root,
                n_full=args.n_full,
                n_rec=args.n_rec,
                percentile=args.percentile,
                eps=args.eps,
                min_samples=args.min_samples,
                min_cluster_size=args.min_cluster_size,
                seed=args.seed,
            )

            ok.append(summary)

        except Exception as e:
            print(f"[ERROR] {pid}: {e}")
            failed.append({
                "patient_id": pid,
                "error": str(e)
            })

    global_summary = {
        "ufrn_root": str(ufrn_root),
        "out_root": str(out_root),
        "n_patients_found": int(len(pacientes)),
        "n_ok": int(len(ok)),
        "n_failed": int(len(failed)),
        "failed": failed,
        "ok_patients": [
            {
                "patient_id": s["patient_id"],
                "raw_removed_points": s["raw_removed_points"],
                "dbscan_points": s["dbscan_points"],
                "dbscan_num_clusters": s["dbscan_num_clusters"],
                "dbscan_selected_label": s["dbscan_selected_label"],
            }
            for s in ok
        ],
    }

    with open(
        out_root / "global_summary.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(global_summary, f, indent=2)

    print("=" * 80)
    print("[DONE]")
    print(json.dumps(global_summary, indent=2))


if __name__ == "__main__":
    main()