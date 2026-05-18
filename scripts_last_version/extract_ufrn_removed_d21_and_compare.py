#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh as tm
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_mesh(path: Path):
    mesh = tm.load(path, process=False, force="mesh")
    if isinstance(mesh, tm.Scene):
        mesh = tm.util.concatenate(tuple(mesh.geometry.values()))
    return mesh


def sample_mesh(mesh, n_points: int, seed: int):
    np.random.seed(seed)
    pts, _ = tm.sample.sample_surface(mesh, n_points)
    return pts.astype(np.float32)


def save_ply_points(path: Path, pts: np.ndarray, color=(255, 0, 0)):
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = np.asarray(pts, dtype=np.float32)

    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for x, y, z in pts:
            f.write(f"{x} {y} {z} {color[0]} {color[1]} {color[2]}\n")


def bbox(points):
    points = np.asarray(points, dtype=np.float32)
    if len(points) == 0:
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


def cloud_metrics(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    if len(a) == 0 or len(b) == 0:
        return {
            "chamfer_mean": None,
            "hausdorff_approx": None,
            "centroid_distance": None,
            "bbox_iou_3d": 0.0,
        }

    tree_b = cKDTree(b)
    tree_a = cKDTree(a)

    d_ab, _ = tree_b.query(a, k=1)
    d_ba, _ = tree_a.query(b, k=1)

    chamfer = float(d_ab.mean() + d_ba.mean())
    hausdorff = float(max(d_ab.max(), d_ba.max()))
    centroid_dist = float(np.linalg.norm(a.mean(axis=0) - b.mean(axis=0)))

    return {
        "chamfer_mean": chamfer,
        "hausdorff_approx": hausdorff,
        "centroid_distance": centroid_dist,
        "bbox_iou_3d": float(bbox_iou_3d(a, b)),
    }


def plot_3views(full_pts, gt_removed, pred_pts, out_png):
    views = [
        ("Vista XY", 0, 1),
        ("Vista XZ", 0, 2),
        ("Vista YZ", 1, 2),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, (title, i, j) in zip(axes, views):
        ax.scatter(full_pts[:, i], full_pts[:, j], c="lightgray", s=1, alpha=0.12)

        if len(gt_removed) > 0:
            ax.scatter(gt_removed[:, i], gt_removed[:, j], c="green", s=8, alpha=0.85, label="GT dentista aprox.")

        if pred_pts is not None and len(pred_pts) > 0:
            ax.scatter(pred_pts[:, i], pred_pts[:, j], c="red", s=16, alpha=0.95, label="Pred modelo")

        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")

    fig.suptitle("UFRN | Verde = diente 21 removido por dentista aprox. | Rojo = predicción modelo")
    plt.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--upper_full_stl", required=True)
    ap.add_argument("--upper_rec_21_stl", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--pred_points_npy", default=None)

    ap.add_argument("--n_full", type=int, default=120000)
    ap.add_argument("--n_rec", type=int, default=120000)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--dist_threshold", type=float, default=None)
    ap.add_argument("--percentile", type=float, default=97.5)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    full_mesh = load_mesh(Path(args.upper_full_stl))
    rec_mesh = load_mesh(Path(args.upper_rec_21_stl))

    print("[INFO] Sampleando full...")
    full_pts = sample_mesh(full_mesh, args.n_full, args.seed)

    print("[INFO] Sampleando rec_21...")
    rec_pts = sample_mesh(rec_mesh, args.n_rec, args.seed + 1)

    print("[INFO] Calculando diferencia geométrica full vs rec_21...")
    tree_rec = cKDTree(rec_pts)
    dists, _ = tree_rec.query(full_pts, k=1)

    if args.dist_threshold is None:
        thr = float(np.percentile(dists, args.percentile))
    else:
        thr = float(args.dist_threshold)

    removed_mask = dists >= thr
    gt_removed = full_pts[removed_mask]

    pred_pts = None
    if args.pred_points_npy is not None and Path(args.pred_points_npy).exists():
        pred_pts = np.load(args.pred_points_npy, allow_pickle=True).astype(np.float32)

    np.save(out_dir / "full_sample_points.npy", full_pts.astype(np.float32))
    np.save(out_dir / "rec_sample_points.npy", rec_pts.astype(np.float32))
    np.save(out_dir / "dist_full_to_rec.npy", dists.astype(np.float32))
    np.save(out_dir / "gt_removed_d21_approx.npy", gt_removed.astype(np.float32))

    save_ply_points(out_dir / "gt_removed_d21_approx.ply", gt_removed, color=(0, 255, 0))

    if pred_pts is not None:
        save_ply_points(out_dir / "pred_model_d21.ply", pred_pts, color=(255, 0, 0))

    metrics = {
        "upper_full_stl": args.upper_full_stl,
        "upper_rec_21_stl": args.upper_rec_21_stl,
        "pred_points_npy": args.pred_points_npy,
        "n_full_sample": int(args.n_full),
        "n_rec_sample": int(args.n_rec),
        "distance_threshold": thr,
        "threshold_mode": "manual" if args.dist_threshold is not None else f"percentile_{args.percentile}",
        "gt_removed_points": int(gt_removed.shape[0]),
        "gt_removed_frac": float(gt_removed.shape[0] / max(full_pts.shape[0], 1)),
    }

    if pred_pts is not None:
        metrics["pred_points"] = int(pred_pts.shape[0])
        metrics["pred_vs_gt_removed"] = cloud_metrics(pred_pts, gt_removed)

    with open(out_dir / "metrics_removed_vs_prediction.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plot_3views(
        full_pts=full_pts,
        gt_removed=gt_removed,
        pred_pts=pred_pts,
        out_png=out_dir / "comparison_3views.png"
    )

    print("[OK] Guardado en:", out_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()