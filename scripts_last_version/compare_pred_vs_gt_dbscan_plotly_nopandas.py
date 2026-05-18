#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import plotly.graph_objects as go


def bbox(points):
    if points.shape[0] == 0:
        return None, None
    return points.min(axis=0), points.max(axis=0)


def bbox_iou_3d(a, b):
    amin, amax = bbox(a)
    bmin, bmax = bbox(b)
    if amin is None or bmin is None:
        return 0.0
    inter_min = np.maximum(amin, bmin)
    inter_max = np.minimum(amax, bmax)
    inter_dim = np.maximum(inter_max - inter_min, 0.0)
    inter_vol = float(np.prod(inter_dim))
    vol_a = float(np.prod(np.maximum(amax - amin, 0.0)))
    vol_b = float(np.prod(np.maximum(bmax - bmin, 0.0)))
    union = vol_a + vol_b - inter_vol
    return 0.0 if union <= 0 else inter_vol / union


def cloud_metrics(pred, gt):
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)

    if pred.shape[0] == 0 or gt.shape[0] == 0:
        return {
            "chamfer_mean": None,
            "hausdorff_approx": None,
            "centroid_distance": None,
            "bbox_iou_3d": 0.0,
            "pred_points": int(pred.shape[0]),
            "gt_points": int(gt.shape[0]),
        }

    tree_gt = cKDTree(gt)
    tree_pred = cKDTree(pred)

    d_pred_to_gt, _ = tree_gt.query(pred, k=1)
    d_gt_to_pred, _ = tree_pred.query(gt, k=1)

    return {
        "chamfer_mean": float(d_pred_to_gt.mean() + d_gt_to_pred.mean()),
        "hausdorff_approx": float(max(d_pred_to_gt.max(), d_gt_to_pred.max())),
        "centroid_distance": float(np.linalg.norm(pred.mean(axis=0) - gt.mean(axis=0))),
        "bbox_iou_3d": float(bbox_iou_3d(pred, gt)),
        "pred_points": int(pred.shape[0]),
        "gt_points": int(gt.shape[0]),
        "pred_to_gt_mean": float(d_pred_to_gt.mean()),
        "gt_to_pred_mean": float(d_gt_to_pred.mean()),
    }


def add_cloud(fig, pts, name, color, size=3, opacity=0.8):
    pts = np.asarray(pts, dtype=np.float32)
    if pts.shape[0] == 0:
        return
    fig.add_trace(go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode="markers",
        name=name,
        marker=dict(
            size=size,
            color=color,
            opacity=opacity
        )
    ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_npy", required=True)
    ap.add_argument("--pred_npy", required=True)
    ap.add_argument("--full_npy", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--title", default="Comparación UFRN: GT dentista vs predicción modelo")
    ap.add_argument("--max_full_points", type=int, default=50000)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt = np.asarray(np.load(args.gt_npy, allow_pickle=True), dtype=np.float32)
    pred = np.asarray(np.load(args.pred_npy, allow_pickle=True), dtype=np.float32)

    full = None
    if args.full_npy is not None and Path(args.full_npy).exists():
        full = np.asarray(np.load(args.full_npy, allow_pickle=True), dtype=np.float32)
        if full.shape[0] > args.max_full_points:
            rng = np.random.default_rng(42)
            idx = rng.choice(full.shape[0], args.max_full_points, replace=False)
            full = full[idx]

    metrics = cloud_metrics(pred, gt)

    with open(out_dir / "metrics_pred_vs_gt_dbscan.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    fig = go.Figure()

    if full is not None:
        add_cloud(fig, full, "Arcada completa", "lightgray", size=2, opacity=0.18)

    add_cloud(fig, gt, "GT dentista DBSCAN", "green", size=4, opacity=0.9)
    add_cloud(fig, pred, "Predicción modelo", "red", size=6, opacity=1.0)

    fig.update_layout(
        title=args.title,
        width=1100,
        height=850,
        scene=dict(
            aspectmode="data",
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
        ),
        legend=dict(itemsizing="constant")
    )

    html_path = out_dir / "comparison_pred_vs_gt_dbscan_plotly.html"
    fig.write_html(str(html_path))

    print("[OK] HTML guardado en:", html_path)
    print("[OK] Métricas guardadas en:", out_dir / "metrics_pred_vs_gt_dbscan.json")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()