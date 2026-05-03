#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
view_case_bundle_polyscope_v2.py

Visor Polyscope interactivo para un case_dir generado por build_best_model_comparison_cases_v1/v2.py.
Muestra raw mesh, GT 200k/8192 y predicciones/errores si existen pred_labels.npy.
"""

import argparse, json
from pathlib import Path
import numpy as np
import polyscope as ps
import trimesh


def load_json(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def label_colors_float(y):
    import matplotlib.cm as cm
    cmap = cm.get_cmap('tab20', 20)
    return cmap(np.asarray(y).reshape(-1).astype(np.int64) % 20)[:, :3].astype(np.float32)


def binary_colors_float(mask, pos=(0.95,0.05,0.45), neg=(0.70,0.70,0.70)):
    mask = np.asarray(mask).reshape(-1).astype(bool)
    out = np.zeros((mask.shape[0],3), dtype=np.float32)
    out[~mask] = np.asarray(neg, np.float32)
    out[mask] = np.asarray(pos, np.float32)
    return out


def load_raw_mesh(raw_path: Path):
    if not raw_path.exists(): return None, None
    try:
        loaded = trimesh.load(str(raw_path), force='mesh', process=False)
    except Exception:
        try: loaded = trimesh.load(str(raw_path), force='mesh', process=True)
        except Exception: return None, None
    if isinstance(loaded, trimesh.Scene):
        geos = list(loaded.dump().geometry.values())
        if not geos: return None, None
        mesh = trimesh.util.concatenate(geos)
    else:
        mesh = loaded
    if not isinstance(mesh, trimesh.Trimesh): return None, None
    return np.asarray(mesh.vertices, np.float32), np.asarray(mesh.faces, np.int32)


def register_cloud(name, xyz, labels, radius, enabled):
    pc = ps.register_point_cloud(name, xyz, enabled=enabled)
    pc.add_color_quantity('labels', label_colors_float(labels), enabled=True)
    pc.set_radius(radius)
    return pc


def register_binary_cloud(name, xyz, mask, radius, enabled):
    pc = ps.register_point_cloud(name, xyz, enabled=enabled)
    pc.add_color_quantity('binary', binary_colors_float(mask), enabled=True)
    pc.set_radius(radius)
    return pc


def register_error_cloud(name, xyz, mask, radius, enabled):
    mask = np.asarray(mask).reshape(-1).astype(bool)
    pts = xyz[mask] if mask.any() else xyz[:1]
    colors = np.tile(np.array([[1.0,0.0,0.0]], dtype=np.float32), (pts.shape[0],1))
    pc = ps.register_point_cloud(name, pts, enabled=enabled)
    pc.add_color_quantity('error_red', colors, enabled=True)
    pc.set_radius(radius)
    return pc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case_dir', required=True)
    ap.add_argument('--d21_class', type=int, default=8)
    args = ap.parse_args()

    case_dir = Path(args.case_dir).resolve()
    meta = load_json(case_dir/'meta.json')
    z = np.load(case_dir/'common'/'bundle_case_data.npz')
    xyz200k, y200k = z['xyz_200k'].astype(np.float32), z['y_200k'].astype(np.int64)
    xyz8192, y8192 = z['xyz_8192'].astype(np.float32), z['y_8192'].astype(np.int64)

    ps.init()
    ps.set_ground_plane_mode('none')

    raw_path_str = meta.get('raw_mesh_path','')
    if raw_path_str:
        V,F = load_raw_mesh(Path(raw_path_str))
        if V is not None and F is not None and len(F)>0:
            m = ps.register_surface_mesh('raw_mesh', V, F, enabled=True)
            m.set_transparency(0.25)

    register_cloud('cloud_200k_GT', xyz200k, y200k, 0.0012, False)
    register_cloud('cloud_8192_GT', xyz8192, y8192, 0.0040, True)

    for model in ['pointnet','pointnetpp','dgcnn','pointnettransformer']:
        pred_path = case_dir/model/'pred_labels.npy'
        if not pred_path.exists():
            continue
        pred = np.load(pred_path).reshape(-1).astype(np.int64)
        if pred.shape[0] != xyz8192.shape[0]:
            print(f'[WARN] {model}: pred_labels incompatible')
            continue
        register_cloud(f'{model}_pred_all', xyz8192, pred, 0.0042, False)
        register_binary_cloud(f'{model}_pred_d21', xyz8192, pred == int(args.d21_class), 0.0045, False)
        register_error_cloud(f'{model}_errors_all', xyz8192, pred != y8192, 0.0060, False)
        register_error_cloud(f'{model}_errors_d21', xyz8192, (pred == int(args.d21_class)) != (y8192 == int(args.d21_class)), 0.0065, False)

    print('='*100)
    print(f'CASE: {case_dir.name}')
    print(f"sample_name: {meta.get('sample_name','')}")
    print(f"jaw: {meta.get('jaw','')}")
    print('Activa/desactiva clouds desde el panel izquierdo de Polyscope.')
    print('='*100)
    ps.show()

if __name__ == '__main__':
    main()
