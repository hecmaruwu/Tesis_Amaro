#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import trimesh
import plotly.graph_objects as go
from scipy.spatial import cKDTree


def load_mesh_points(stl_path, n=120000):
    mesh = trimesh.load_mesh(stl_path, process=False)
    pts, _ = trimesh.sample.sample_surface(mesh, n)
    return pts.astype(np.float32)


def knn_label(points, gt_points, radius):
    tree = cKDTree(gt_points)
    d, _ = tree.query(points, k=1)
    return d <= radius


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--full_stl", required=True)
    ap.add_argument("--gt_npy", required=True)
    ap.add_argument("--pred_npy", required=True)

    ap.add_argument("--radius", type=float, default=1.25)

    ap.add_argument("--mesh_points", type=int, default=120000)

    ap.add_argument("--out_dir", required=True)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] cargando malla...")
    full_pts = load_mesh_points(args.full_stl, args.mesh_points)

    print("[INFO] cargando GT...")
    gt_pts = np.load(args.gt_npy).astype(np.float32)

    print("[INFO] cargando pred...")
    pred_pts = np.load(args.pred_npy).astype(np.float32)

    print("[INFO] calculando TP/FP/FN...")

    pred_is_gt = knn_label(pred_pts, gt_pts, args.radius)
    gt_is_pred = knn_label(gt_pts, pred_pts, args.radius)

    tp_pred = pred_pts[pred_is_gt]
    fp_pred = pred_pts[~pred_is_gt]

    fn_gt = gt_pts[~gt_is_pred]

    fig = go.Figure()

    # malla
    fig.add_trace(go.Scatter3d(
        x=full_pts[:,0],
        y=full_pts[:,1],
        z=full_pts[:,2],
        mode="markers",
        marker=dict(
            size=1,
            color="lightgray",
            opacity=0.08
        ),
        name="Arcada completa"
    ))

    # GT
    fig.add_trace(go.Scatter3d(
        x=gt_pts[:,0],
        y=gt_pts[:,1],
        z=gt_pts[:,2],
        mode="markers",
        marker=dict(
            size=3,
            color="green",
            opacity=0.9
        ),
        name="GT d21"
    ))

    # TP
    fig.add_trace(go.Scatter3d(
        x=tp_pred[:,0],
        y=tp_pred[:,1],
        z=tp_pred[:,2],
        mode="markers",
        marker=dict(
            size=4,
            color="blue",
            opacity=0.95
        ),
        name="TP"
    ))

    # FP
    if len(fp_pred) > 0:
        fig.add_trace(go.Scatter3d(
            x=fp_pred[:,0],
            y=fp_pred[:,1],
            z=fp_pred[:,2],
            mode="markers",
            marker=dict(
                size=5,
                color="red",
                opacity=0.95
            ),
            name="FP"
        ))

    # FN
    if len(fn_gt) > 0:
        fig.add_trace(go.Scatter3d(
            x=fn_gt[:,0],
            y=fn_gt[:,1],
            z=fn_gt[:,2],
            mode="markers",
            marker=dict(
                size=5,
                color="orange",
                opacity=0.95
            ),
            name="FN"
        ))

    fig.update_layout(
        title="UFRN Binary d21 Fine-tuning | TP azul | FP rojo | FN naranja",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data"
        ),
        template="plotly_white",
        height=900
    )

    html_path = out_dir / "visualization_binary_errors.html"
    fig.write_html(str(html_path))

    print("[OK] html:", html_path)

    print({
        "gt_points": int(len(gt_pts)),
        "pred_points": int(len(pred_pts)),
        "tp": int(len(tp_pred)),
        "fp": int(len(fp_pred)),
        "fn": int(len(fn_gt)),
    })


if __name__ == "__main__":
    main()