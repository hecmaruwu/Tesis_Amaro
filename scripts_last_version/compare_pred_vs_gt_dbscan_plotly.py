#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import plotly.express as px


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
    if union <= 0:
        return 0.0

    return inter_vol / union


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


def make_dataframe(gt, pred, full=None, max_full_points=50000):
    dfs = []

    if full is not None and full.shape[0] > 0:
        if full.shape[0] > max_full_points:
            idx = np.random.choice(full.shape[0], max_full_points, replace=False)
            full = full[idx]

        df_full = pd.DataFrame(full, columns=["x", "y", "z"])
        df_full["grupo"] = "Arcada completa"
        df_full["color"] = "gris"
        dfs.append(df_full)

    df_gt = pd.DataFrame(gt, columns=["x", "y", "z"])
    df_gt["grupo"] = "GT dentista DBSCAN"
    df_gt["color"] = "verde"
    dfs.append(df_gt)

    df_pred = pd.DataFrame(pred, columns=["x", "y", "z"])
    df_pred["grupo"] = "Predicción modelo"
    df_pred["color"] = "rojo"
    dfs.append(df_pred)

    return pd.concat(dfs, ignore_index=True)


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

    gt = np.load(args.gt_npy, allow_pickle=True).astype(np.float32)
    pred = np.load(args.pred_npy, allow_pickle=True).astype(np.float32)

    full = None
    if args.full_npy is not None and Path(args.full_npy).exists():
        full = np.load(args.full_npy, allow_pickle=True).astype(np.float32)

    metrics = cloud_metrics(pred, gt)

    with open(out_dir / "metrics_pred_vs_gt_dbscan.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    df = make_dataframe(
        gt=gt,
        pred=pred,
        full=full,
        max_full_points=args.max_full_points
    )

    fig = px.scatter_3d(
        df,
        x="x",
        y="y",
        z="z",
        color="grupo",
        opacity=0.75,
        title=args.title,
        color_discrete_map={
            "Arcada completa": "lightgray",
            "GT dentista DBSCAN": "green",
            "Predicción modelo": "red",
        },
        height=850,
        width=1100,
    )

    fig.update_traces(marker=dict(size=3))
    fig.update_layout(
        scene=dict(
            aspectmode="data",
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
        )
    )

    html_path = out_dir / "comparison_pred_vs_gt_dbscan_plotly.html"
    fig.write_html(str(html_path))

    print("[OK] HTML guardado en:", html_path)
    print("[OK] Métricas guardadas en:", out_dir / "metrics_pred_vs_gt_dbscan.json")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()