#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genera pseudolabels binarios para el diente 21 usando registro ICP
entre upper_full.stl y upper_rec_21.stl.

✔ Alinea automáticamente con ICP.
✔ Calcula distancias punto a punto (mm).
✔ Etiqueta 1 = puntos eliminados (zona diente 21).
✔ Genera X.npy, Y.npy y PNG de control.

Salida:
  data/UFRN/processed_pseudolabels_icp/8192/upper/paciente_XX/{X.npy, Y.npy}
"""

import numpy as np
import trimesh as tm
import open3d as o3d
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm


# === CONFIG ===
ufrn_root = Path("/home/htaucare/Tesis_Amaro/data/UFRN")
targets_root = ufrn_root / "targets_export"
out_root = ufrn_root / "processed_pseudolabels_icp" / "8192" / "upper"
out_root.mkdir(parents=True, exist_ok=True)
tau_mm = 0.7         # umbral de distancia (mm)
n_points = 8192      # puntos por malla


def sample_points(mesh, n=8192):
    pts, _ = tm.sample.sample_surface(mesh, n)
    return pts.astype(np.float32)


def mesh_to_pcd(pts: np.ndarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


def align_icp(source_pts, target_pts, voxel_size=0.3):
    """Alinea con ICP los puntos recortados (source) al full (target)."""
    src = mesh_to_pcd(source_pts)
    tgt = mesh_to_pcd(target_pts)
    threshold = voxel_size * 3
    icp = o3d.pipelines.registration.registration_icp(
        src, tgt, threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    src_aligned = np.asarray(src.transform(icp.transformation).points)
    return src_aligned, icp.transformation


def compute_distance_mask(full_pts, rec_pts, tau):
    """Devuelve máscara binaria donde distancia > tau."""
    rec_tree = o3d.geometry.KDTreeFlann(mesh_to_pcd(rec_pts))
    mask = np.zeros(len(full_pts), dtype=np.uint8)
    for i, p in enumerate(full_pts):
        _, _, d2 = rec_tree.search_knn_vector_3d(p, 1)
        if np.sqrt(d2[0]) > tau:
            mask[i] = 1
    return mask


def plot_and_save(pid, X, Y, out_dir):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[Y==0,0], X[Y==0,1], X[Y==0,2], s=1, c='lightgray', alpha=0.3)
    ax.scatter(X[Y==1,0], X[Y==1,1], X[Y==1,2], s=3, c='red', alpha=0.6)
    ax.set_title(f"{pid} — puntos 1 = {Y.sum()}")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    plt.tight_layout()
    plt.savefig(out_dir/f"{pid}_labels.png", dpi=300)
    plt.close(fig)


# === LOOP PRINCIPAL ===
patients = sorted([p for p in targets_root.iterdir() if (p/"stl"/"upper_full.stl").exists() and (p/"stl"/"upper_rec_21.stl").exists()])
print(f"[INFO] Pares upper_full / upper_rec_21 detectados: {len(patients)}")

ok = 0
for p in tqdm(patients, desc="[ICP-PSEUDOLABELS]"):
    try:
        f_full = p/"stl"/"upper_full.stl"
        f_rec  = p/"stl"/"upper_rec_21.stl"

        mesh_full = tm.load(f_full, process=False)
        mesh_rec  = tm.load(f_rec, process=False)
        pts_full = sample_points(mesh_full, n_points)
        pts_rec  = sample_points(mesh_rec, n_points)

        # Alineación ICP
        pts_rec_aligned, _ = align_icp(pts_rec, pts_full)

        # Distancia y máscara binaria
        Y = compute_distance_mask(pts_full, pts_rec_aligned, tau_mm)
        X = pts_full

        # Guardado
        pid = p.name.lower().replace(" ", "_")
        out_dir = out_root / pid
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir/"X.npy", X)
        np.save(out_dir/"Y.npy", Y)
        plot_and_save(pid, X, Y, out_root)
        ok += 1

    except Exception as e:
        print(f"[WARN] {p.name}: {e}")

print(f"\n✅ Procesados correctamente: {ok}/{len(patients)}")
print(f"Guardado en: {out_root}")
