#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genera pseudolabels del diente 21 usando los STL dentro de:
  /data/UFRN/targets_export/paciente_XX/stl/

Busca:
  - upper_full.stl
  - upper_rec_21.stl

Salida:
  /data/UFRN/processed_pseudolabels_targets/8192/upper/paciente_XX/{X.npy,Y.npy}
"""

from pathlib import Path
import numpy as np, json
import trimesh as tm
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

# --- Configuración ---
UFRN_ROOT = Path("/home/htaucare/Tesis_Amaro/data/UFRN")
SRC_DIR = UFRN_ROOT / "targets_export"
OUT_DIR = UFRN_ROOT / "processed_pseudolabels_targets" / "8192" / "upper"
N_POINTS = 8192
TAU_MM = 1.5

def normalize_points(pts):
    c = pts.mean(axis=0)
    pts -= c
    r = np.linalg.norm(pts, axis=1).max()
    return pts / r if r > 0 else pts

def fps(points, n_samples):
    """Farthest Point Sampling"""
    N = points.shape[0]
    centroids = np.zeros(n_samples, dtype=int)
    distances = np.ones(N) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(n_samples):
        centroids[i] = farthest
        centroid = points[farthest, :3]
        dist = np.sum((points - centroid) ** 2, -1)
        distances = np.minimum(distances, dist)
        farthest = np.argmax(distances)
    return points[centroids]

def align(mesh_a, mesh_b):
    """Alineamiento simple por centrado + escala global"""
    mesh_a.vertices -= mesh_a.vertices.mean(axis=0)
    mesh_b.vertices -= mesh_b.vertices.mean(axis=0)
    scale = np.linalg.norm(mesh_a.vertices, axis=1).max()
    mesh_a.vertices /= scale
    mesh_b.vertices /= scale
    return mesh_a, mesh_b

def sample_and_label(mesh_full, mesh_rec):
    """Crea X,Y desde full vs recortado"""
    pts_full, _ = tm.sample.sample_surface(mesh_full, N_POINTS)
    pts_full = normalize_points(pts_full)
    nbrs = NearestNeighbors(n_neighbors=1).fit(mesh_rec.vertices)
    dist, _ = nbrs.kneighbors(pts_full)
    mask = (dist.squeeze() * 1000) > TAU_MM
    return pts_full.astype(np.float32), mask.astype(np.uint8)

def build_pseudolabels():
    patients = sorted(SRC_DIR.glob("paciente_*"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    valid = 0
    for p in tqdm(patients, desc="[BUILD] pseudo-labels 21"):
        stl_dir = p / "stl"
        full = stl_dir / "upper_full.stl"
        rec = stl_dir / "upper_rec_21.stl"
        if not full.exists() or not rec.exists():
            continue

        try:
            mesh_full = tm.load(full, process=False)
            mesh_rec = tm.load(rec, process=False)
            mesh_full, mesh_rec = align(mesh_full, mesh_rec)
            X, Y = sample_and_label(mesh_full, mesh_rec)

            outp = OUT_DIR / p.name
            outp.mkdir(parents=True, exist_ok=True)
            np.save(outp / "X.npy", X)
            np.save(outp / "Y.npy", Y)
            (outp / "meta.json").write_text(json.dumps({
                "patient_id": p.name,
                "file_full": str(full),
                "file_rec": str(rec),
                "n_points": N_POINTS,
                "tau_mm": TAU_MM,
                "pos_ratio": float(Y.mean())
            }, indent=2))

            valid += 1
        except Exception as e:
            print(f"[WARN] {p.name}: error {e}")

    print(f"\n✅ Procesados correctamente: {valid}/{len(patients)}")
    print(f"Guardado en: {OUT_DIR}")

if __name__ == "__main__":
    build_pseudolabels()
