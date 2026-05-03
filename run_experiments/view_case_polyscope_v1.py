#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
import numpy as np
import trimesh
import polyscope as ps


CLASS_COLORS = {
    0:  (0.70, 0.70, 0.70),
    1:  (0.17, 0.63, 0.17),
    2:  (0.09, 0.75, 0.81),
    3:  (0.12, 0.47, 0.71),
    4:  (0.58, 0.40, 0.74),
    5:  (0.89, 0.47, 0.76),
    6:  (0.84, 0.15, 0.16),
    7:  (1.00, 0.50, 0.05),
    8:  (1.00, 0.31, 0.64),   # d21 GT
    9:  (1.00, 0.75, 0.00),
    10: (0.74, 0.87, 0.20),
    11: (0.17, 0.65, 0.55),
    12: (0.00, 0.74, 0.83),
    13: (0.77, 0.69, 0.84),
    14: (0.50, 0.50, 0.50),
}

COLOR_D21_GT = (1.00, 0.31, 0.64)
COLOR_D21_PRED = (1.00, 0.55, 0.00)
COLOR_LEFT = (0.00, 0.44, 0.75)
COLOR_RIGHT = (1.00, 0.41, 0.71)
COLOR_TP = (0.00, 0.70, 0.25)
COLOR_ERR = (1.00, 0.00, 0.00)


def read_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_npy(path):
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        try:
            arr = arr.item()
        except Exception:
            arr = np.asarray(arr.tolist())
    return np.asarray(arr)


def finite_points(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        x = x.reshape(-1, 3)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def robust_bounds(X, q_low=0.01, q_high=0.99):
    return np.quantile(X, q_low, axis=0), np.quantile(X, q_high, axis=0)


def bbox_center_scale(X, robust=True):
    X = finite_points(X)
    if robust:
        lo, hi = robust_bounds(X)
    else:
        lo, hi = X.min(axis=0), X.max(axis=0)
    c = 0.5 * (lo + hi)
    s = np.linalg.norm(hi - lo)
    if not np.isfinite(s) or s <= 0:
        s = 1.0
    return c.astype(np.float32), float(s)


def align_bbox(V, X, robust=True):
    c_src, s_src = bbox_center_scale(V, robust=robust)
    c_tgt, s_tgt = bbox_center_scale(X, robust=robust)
    return ((V - c_src) / (s_src + 1e-9) * s_tgt + c_tgt).astype(np.float32)


def find_raw_mesh(case_dir):
    meta = read_json(Path(case_dir) / "meta.json")
    for k in ["raw_mesh_path", "mesh_path", "raw_path"]:
        p = meta.get(k, "")
        if p and Path(p).exists():
            return Path(p)
    raise FileNotFoundError("No encontré raw_mesh_path en meta.json")


def load_mesh(path):
    m = trimesh.load(str(path), force="mesh", process=False)
    if isinstance(m, trimesh.Scene):
        geos = list(m.dump().geometry.values())
        m = trimesh.util.concatenate(geos)
    V = finite_points(np.asarray(m.vertices, dtype=np.float32))
    F = np.asarray(m.faces, dtype=np.int64)
    return V, F


def labels_to_rgb(labels):
    labels = np.asarray(labels).reshape(-1).astype(int)
    C = np.zeros((labels.shape[0], 3), dtype=np.float32)
    for i, y in enumerate(labels):
        C[i] = CLASS_COLORS.get(int(y), (0.2, 0.2, 0.2))
    return C


def focus_colors(labels, d21_class, neighbors, pred_view=False):
    labels = np.asarray(labels).reshape(-1).astype(int)
    neigh_ids = [v for _, v in neighbors]
    C = np.zeros((labels.shape[0], 3), dtype=np.float32)
    C[:] = (0.78, 0.78, 0.78)

    for i, y in enumerate(labels):
        if y == d21_class:
            C[i] = COLOR_D21_PRED if pred_view else COLOR_D21_GT
        elif len(neigh_ids) >= 1 and y == neigh_ids[0]:
            C[i] = COLOR_LEFT
        elif len(neigh_ids) >= 2 and y == neigh_ids[1]:
            C[i] = COLOR_RIGHT
    return C


def d21_error_colors(pred, gt, d21_class, neighbors):
    pred = np.asarray(pred).reshape(-1).astype(int)
    gt = np.asarray(gt).reshape(-1).astype(int)
    neigh_ids = [v for _, v in neighbors]
    C = np.zeros((gt.shape[0], 3), dtype=np.float32)
    C[:] = (0.82, 0.82, 0.82)

    for i, (p, g) in enumerate(zip(pred, gt)):
        is_tp = (p == d21_class and g == d21_class)
        is_err = ((p == d21_class) != (g == d21_class))

        if is_tp:
            C[i] = COLOR_TP
        elif is_err:
            C[i] = COLOR_ERR
        elif len(neigh_ids) >= 1 and g == neigh_ids[0]:
            C[i] = COLOR_LEFT
        elif len(neigh_ids) >= 2 and g == neigh_ids[1]:
            C[i] = COLOR_RIGHT
    return C


def all_error_colors(pred, gt):
    pred = np.asarray(pred).reshape(-1).astype(int)
    gt = np.asarray(gt).reshape(-1).astype(int)
    C = np.zeros((gt.shape[0], 3), dtype=np.float32)
    C[:] = (0.82, 0.82, 0.82)
    C[pred != gt] = COLOR_ERR
    return C


def parse_neighbors(s):
    out = []
    for part in s.replace(";", ",").split(","):
        if ":" not in part:
            continue
        name, val = part.split(":", 1)
        out.append((name.strip(), int(val)))
    return out


def register_cloud(name, X, colors, radius=0.004, enabled=True):
    pc = ps.register_point_cloud(name, X, radius=radius, enabled=enabled)
    pc.add_color_quantity("colors", colors, enabled=True)
    return pc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case_dir", required=True)
    ap.add_argument("--model", default="pointnet")
    ap.add_argument("--d21_class", type=int, default=8)
    ap.add_argument("--neighbor_teeth", default="d11:1,d22:9")
    ap.add_argument("--align_mode", default="robust_bbox", choices=["robust_bbox", "bbox", "none"])
    ap.add_argument("--point_radius", type=float, default=0.004)
    ap.add_argument("--mesh_transparency", type=float, default=0.45)
    args = ap.parse_args()

    case_dir = Path(args.case_dir)
    model_dir = case_dir / args.model
    neighbors = parse_neighbors(args.neighbor_teeth)

    X = finite_points(load_npy(model_dir / "xyz_labels.npy"))
    pred = load_npy(model_dir / "pred_labels.npy").reshape(-1).astype(int)
    gt = load_npy(model_dir / "gt_labels.npy").reshape(-1).astype(int)

    n = min(len(X), len(pred), len(gt))
    X, pred, gt = X[:n], pred[:n], gt[:n]

    raw_path = find_raw_mesh(case_dir)
    V, F = load_mesh(raw_path)

    if args.align_mode == "robust_bbox":
        V = align_bbox(V, X, robust=True)
    elif args.align_mode == "bbox":
        V = align_bbox(V, X, robust=False)

    ps.init()
    ps.set_program_name("Dental segmentation viewer")
    ps.set_ground_plane_mode("none")
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")

    mesh = ps.register_surface_mesh("raw mesh aligned", V, F, transparency=args.mesh_transparency)
    mesh.set_color((0.72, 0.72, 0.72))
    mesh.set_smooth_shade(True)
    mesh.set_enabled(True)

    register_cloud("01_GT_multiclase", X, labels_to_rgb(gt), args.point_radius, enabled=True)
    register_cloud("02_Pred_multiclase", X, labels_to_rgb(pred), args.point_radius, enabled=False)
    register_cloud("03_Errores_multiclase", X, all_error_colors(pred, gt), args.point_radius, enabled=False)

    register_cloud(
        "04_GT_d21_vecinos",
        X,
        focus_colors(gt, args.d21_class, neighbors, pred_view=False),
        args.point_radius * 1.15,
        enabled=False,
    )

    register_cloud(
        "05_Pred_d21_vecinos_d21_naranja",
        X,
        focus_colors(pred, args.d21_class, neighbors, pred_view=True),
        args.point_radius * 1.15,
        enabled=False,
    )

    register_cloud(
        "06_Error_d21_verde_TP_rojo_FP_FN",
        X,
        d21_error_colors(pred, gt, args.d21_class, neighbors),
        args.point_radius * 1.2,
        enabled=False,
    )

    print("\n[OK] Polyscope abierto.")
    print("Activa/desactiva capas en el panel izquierdo:")
    print("  01_GT_multiclase")
    print("  02_Pred_multiclase")
    print("  03_Errores_multiclase")
    print("  04_GT_d21_vecinos")
    print("  05_Pred_d21_vecinos_d21_naranja")
    print("  06_Error_d21_verde_TP_rojo_FP_FN\n")

    ps.show()


if __name__ == "__main__":
    main()