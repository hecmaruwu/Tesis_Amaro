#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualizador 3D automático (sin ventana)
Guarda vistas PNG de los resultados HDBSCAN (antes/después).
Usa Open3D OffscreenRenderer.
"""

import argparse
from pathlib import Path
import numpy as np
import open3d as o3d

# ----------------------------------------------------------
# Crear nube de puntos coloreada
# ----------------------------------------------------------
def make_pcd(points, mask, color_fg=(1, 0, 0), color_bg=(0.6, 0.6, 0.6)):
    colors = np.tile(color_bg, (len(points), 1))
    colors[mask > 0] = color_fg
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

# ----------------------------------------------------------
# Renderizado en modo offscreen (sin GUI)
# ----------------------------------------------------------
def render_to_png(pcd, out_path, width=1200, height=900):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    vis.add_geometry(pcd)
    vis.get_render_option().background_color = np.array([1, 1, 1])
    vis.get_render_option().point_size = 2.0
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(str(out_path))
    vis.destroy_window()

# ----------------------------------------------------------
# Ejecución principal
# ----------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True,
                    help="Directorio con X.npy originales (p. ej. processed_pseudolabels_icp/8192/upper)")
    ap.add_argument("--mask_dir", required=True,
                    help="Directorio con las máscaras *_raw.npy o *_clean.npy")
    ap.add_argument("--out_dir", required=True,
                    help="Carpeta donde se guardarán las imágenes PNG")
    ap.add_argument("--mode", choices=["raw", "clean"], default="clean",
                    help="Modo: 'raw' antes de HDBSCAN o 'clean' después")
    ap.add_argument("--sample", type=int, default=-1,
                    help="Número de pacientes a renderizar (-1 = todos)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    mask_dir = Path(args.mask_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    patients = sorted([p.name for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("paciente_")])
    if args.sample > 0:
        patients = patients[:args.sample]

    print(f"🖼 Renderizando {len(patients)} pacientes en modo {args.mode}...")
    for pid in patients:
        try:
            X = np.load(data_dir / pid / "X.npy").astype(np.float32)
            mask = np.load(mask_dir / f"{pid}_{args.mode}.npy")
            pcd = make_pcd(X, mask)
            out_img = out_dir / f"{pid}_{args.mode}.png"
            render_to_png(pcd, out_img)
            print(f"   ✓ {pid} → {out_img.name}")
        except Exception as e:
            print(f"   ⚠️ Error en {pid}: {e}")

    print(f"\n✅ Listo. Imágenes guardadas en: {out_dir}")

if __name__ == "__main__":
    main()
