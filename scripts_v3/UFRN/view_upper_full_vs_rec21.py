#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualiza mallas upper_full.stl y upper_rec_21.stl de cada paciente
para verificar alineamiento antes de generar pseudolabels.

Salida:
  /home/htaucare/Tesis_Amaro/data/UFRN/visual_checks_icp/<paciente>.png
"""

import trimesh as tm
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm


# === CONFIGURACIÓN ===
root = Path("/home/htaucare/Tesis_Amaro/data/UFRN/targets_export")
out_dir = Path("/home/htaucare/Tesis_Amaro/data/UFRN/visual_checks_icp")
out_dir.mkdir(parents=True, exist_ok=True)


# === FUNCIÓN AUXILIAR ===
def sample_points(mesh, n=5000):
    pts, _ = tm.sample.sample_surface(mesh, n)
    return pts


# === LOOP POR PACIENTE ===
patients = sorted([p for p in root.iterdir() if p.is_dir() and (p/"stl").exists()])
print(f"[INFO] Pacientes detectados: {len(patients)}")

for p in tqdm(patients, desc="[VISUAL CHECK]"):
    f_full = p/"stl"/"upper_full.stl"
    f_rec  = p/"stl"/"upper_rec_21.stl"
    if not (f_full.exists() and f_rec.exists()):
        continue

    try:
        m_full = tm.load_mesh(f_full, process=False)
        m_rec  = tm.load_mesh(f_rec, process=False)
        pts_full = sample_points(m_full, n=8000)
        pts_rec  = sample_points(m_rec, n=8000)

        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pts_full[:,0], pts_full[:,1], pts_full[:,2], s=1, c='blue', alpha=0.4, label='Full')
        ax.scatter(pts_rec[:,0], pts_rec[:,1], pts_rec[:,2], s=1, c='red', alpha=0.4, label='Rec 21')
        ax.set_title(p.name)
        ax.legend()
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.view_init(elev=20, azim=45)

        plt.tight_layout()
        plt.savefig(out_dir/f"{p.name}_overlay.png", dpi=300)
        plt.close(fig)
    except Exception as e:
        print(f"[WARN] {p.name}: {e}")

print(f"\n✅ Visualizaciones guardadas en: {out_dir}")
