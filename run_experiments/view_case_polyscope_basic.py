#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np

try:
    import polyscope as ps
except ImportError:
    raise SystemExit("Falta polyscope. Instala con: pip install polyscope")

try:
    import trimesh
    HAS_TRIMESH = True
except Exception:
    HAS_TRIMESH = False


CLASS_COLORS = {
    0:  (0.02, 0.02, 0.02),  # bg
    1:  (0.00, 0.44, 0.75),  # d11 azul
    8:  (1.00, 0.55, 0.00),  # d21 naranja
    9:  (1.00, 0.35, 0.70),  # d22 rosado
}

COLOR_OTHER = (0.91, 0.76, 0.62)  # beige cálido
COLOR_ERROR = (1.00, 0.00, 0.00)
COLOR_TP = (0.00, 0.70, 0.25)


def load_arr(path):
    p = Path(path)
    arr = np.load(p, allow_pickle=True)
    if isinstance(arr, np.lib.npyio.NpzFile):
        key = "arr_0"
        for k in ["xyz", "points", "X", "pred", "gt", "labels", "Y"]:
            if k in arr.files:
                key = k
                break
        arr = arr[key]
    arr = np.asarray(arr).squeeze()
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def label_colors(labels):
    labels = labels.astype(int).reshape(-1)
    colors = np.zeros((len(labels), 3), dtype=float)
    for i, y in enumerate(labels):
        colors[i] = CLASS_COLORS.get(int(y), COLOR_OTHER)
    return colors


def focus_colors(labels, d21=8, neighbors=(1, 9)):
    labels = labels.astype(int).reshape(-1)
    colors = np.zeros((len(labels), 3), dtype=float)

    for i, y in enumerate(labels):
        if y == 0:
            colors[i] = (0.03, 0.03, 0.03)
        elif y == d21:
            colors[i] = CLASS_COLORS[8]
        elif y == neighbors[0]:
            colors[i] = CLASS_COLORS[1]
        elif len(neighbors) > 1 and y == neighbors[1]:
            colors[i] = CLASS_COLORS[9]
        else:
            colors[i] = COLOR_OTHER
    return colors


def error_colors(pred, gt):
    pred = pred.astype(int).reshape(-1)
    gt = gt.astype(int).reshape(-1)
    n = min(len(pred), len(gt))
    colors = np.zeros((n, 3), dtype=float)

    ok = pred[:n] == gt[:n]
    bg_ok = ok & (gt[:n] == 0)
    tooth_ok = ok & (gt[:n] != 0)
    err = ~ok

    colors[bg_ok] = (0.03, 0.03, 0.03)
    colors[tooth_ok] = COLOR_OTHER
    colors[err] = COLOR_ERROR
    return colors


def d21_error_colors(pred, gt, d21=8, neighbors=(1, 9)):
    pred = pred.astype(int).reshape(-1)
    gt = gt.astype(int).reshape(-1)
    n = min(len(pred), len(gt))
    colors = np.zeros((n, 3), dtype=float)

    for i, (p, g) in enumerate(zip(pred[:n], gt[:n])):
        if p == d21 and g == d21:
            colors[i] = COLOR_TP
        elif (p == d21) != (g == d21):
            colors[i] = COLOR_ERROR
        elif g == neighbors[0]:
            colors[i] = CLASS_COLORS[1]
        elif len(neighbors) > 1 and g == neighbors[1]:
            colors[i] = CLASS_COLORS[9]
        elif g == 0:
            colors[i] = (0.03, 0.03, 0.03)
        else:
            colors[i] = COLOR_OTHER
    return colors


def load_mesh(mesh_path):
    if not mesh_path or not HAS_TRIMESH:
        return None, None

    mesh_path = Path(mesh_path)
    if not mesh_path.exists():
        return None, None

    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.dump().geometry.values()))

    return np.asarray(mesh.vertices), np.asarray(mesh.faces)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xyz", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--mesh", default="")
    ap.add_argument("--d21", type=int, default=8)
    ap.add_argument("--neighbors", default="1,9")
    ap.add_argument("--point_radius", type=float, default=0.006)
    args = ap.parse_args()

    X = load_arr(args.xyz).astype(np.float32)
    gt = load_arr(args.gt).astype(np.int64).reshape(-1)
    pred = load_arr(args.pred).astype(np.int64).reshape(-1)

    n = min(len(X), len(gt), len(pred))
    X, gt, pred = X[:n], gt[:n], pred[:n]

    neighbors = tuple(int(x) for x in args.neighbors.split(",") if x.strip())

    ps.init()
    ps.set_ground_plane_mode("none")
    ps.set_up_dir("z_up")
    ps.set_navigation_style("free")

    pc = ps.register_point_cloud("cloud_8192", X, radius=args.point_radius)

    pc.add_color_quantity("01_GT_multiclase", label_colors(gt), enabled=True)
    pc.add_color_quantity("02_PRED_multiclase", label_colors(pred), enabled=False)
    pc.add_color_quantity("03_ERRORES_multiclase", error_colors(pred, gt), enabled=False)
    pc.add_color_quantity("04_GT_d21_vecinos", focus_colors(gt, args.d21, neighbors), enabled=False)
    pc.add_color_quantity("05_PRED_d21_vecinos", focus_colors(pred, args.d21, neighbors), enabled=False)
    pc.add_color_quantity("06_ERROR_d21", d21_error_colors(pred, gt, args.d21, neighbors), enabled=False)

    pc.add_scalar_quantity("gt_label", gt.astype(float), enabled=False)
    pc.add_scalar_quantity("pred_label", pred.astype(float), enabled=False)
    pc.add_scalar_quantity("is_error", (pred != gt).astype(float), enabled=False)

    V, F = load_mesh(args.mesh)
    if V is not None and F is not None:
        mesh = ps.register_surface_mesh("raw_mesh", V, F, transparency=0.75)
        mesh.set_enabled(True)

    print("\n[OK] Polyscope abierto.")
    print("Activa/desactiva quantities desde la interfaz:")
    print("  01_GT_multiclase")
    print("  02_PRED_multiclase")
    print("  03_ERRORES_multiclase")
    print("  04_GT_d21_vecinos")
    print("  05_PRED_d21_vecinos")
    print("  06_ERROR_d21\n")

    ps.show()


if __name__ == "__main__":
    main()