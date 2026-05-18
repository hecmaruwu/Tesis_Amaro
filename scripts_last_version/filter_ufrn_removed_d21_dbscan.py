#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from collections import deque

import numpy as np
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_ply_points(path, pts, color=(0, 255, 0)):
    pts = np.asarray(pts, dtype=np.float32)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

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


def plot_3views(full_pts, raw_removed, filtered_removed, out_png):
    views = [
        ("Vista XY", 0, 1),
        ("Vista XZ", 0, 2),
        ("Vista YZ", 1, 2),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, (title, a, b) in zip(axes, views):
        ax.scatter(full_pts[:, a], full_pts[:, b], c="lightgray", s=1, alpha=0.10)

        if raw_removed is not None and len(raw_removed) > 0:
            ax.scatter(
                raw_removed[:, a],
                raw_removed[:, b],
                c="orange",
                s=5,
                alpha=0.30,
                label="Candidatos antes DBSCAN"
            )

        if filtered_removed is not None and len(filtered_removed) > 0:
            ax.scatter(
                filtered_removed[:, a],
                filtered_removed[:, b],
                c="green",
                s=12,
                alpha=0.95,
                label="GT filtrada DBSCAN"
            )

        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")

    fig.suptitle("UFRN | Filtrado DBSCAN de diente 21 removido")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def dbscan_ckdtree(points, eps=1.25, min_samples=20):
    """
    DBSCAN simple usando scipy.cKDTree.
    Evita sklearn porque en este entorno está fallando con NumPy.

    labels:
      -1 = ruido
       0,1,2... = clusters
    """
    points = np.asarray(points, dtype=np.float64, order="C")
    n = points.shape[0]

    labels = np.full(n, -1, dtype=np.int32)
    visited = np.zeros(n, dtype=bool)

    tree = cKDTree(points)
    neighbors = tree.query_ball_point(points, r=float(eps))

    cluster_id = 0

    for i in range(n):
        if visited[i]:
            continue

        visited[i] = True
        neigh_i = neighbors[i]

        if len(neigh_i) < min_samples:
            labels[i] = -1
            continue

        labels[i] = cluster_id
        queue = deque(neigh_i)

        while queue:
            p = queue.popleft()

            if not visited[p]:
                visited[p] = True
                neigh_p = neighbors[p]

                if len(neigh_p) >= min_samples:
                    for q in neigh_p:
                        if not visited[q]:
                            queue.append(q)

            if labels[p] == -1:
                labels[p] = cluster_id

        cluster_id += 1

    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis_dir", required=True)
    ap.add_argument("--eps", type=float, default=1.25)
    ap.add_argument("--min_samples", type=int, default=20)
    ap.add_argument("--min_cluster_size", type=int, default=100)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    analysis_dir = Path(args.analysis_dir)
    out_dir = Path(args.out_dir) if args.out_dir else analysis_dir / f"dbscan_eps{args.eps}_min{args.min_samples}"
    out_dir.mkdir(parents=True, exist_ok=True)

    full_pts = np.load(
        analysis_dir / "full_sample_points.npy",
        allow_pickle=True
    )
    full_pts = np.asarray(full_pts, dtype=np.float32)

    raw_removed = np.load(
        analysis_dir / "gt_removed_d21_approx.npy",
        allow_pickle=True
    )
    raw_removed = np.asarray(raw_removed, dtype=np.float64, order="C")

    if raw_removed.shape[0] == 0:
        raise RuntimeError("gt_removed_d21_approx.npy está vacío.")

    print("[INFO] raw_removed:", raw_removed.shape)
    print("[INFO] Ejecutando DBSCAN cKDTree...")
    print("[INFO] eps:", args.eps)
    print("[INFO] min_samples:", args.min_samples)

    labels = dbscan_ckdtree(
        raw_removed,
        eps=args.eps,
        min_samples=args.min_samples
    )

    valid = labels >= 0
    unique, counts = np.unique(labels[valid], return_counts=True)

    clusters_info = []
    for lbl, cnt in zip(unique, counts):
        clusters_info.append({
            "label": int(lbl),
            "size": int(cnt)
        })

    if len(clusters_info) == 0:
        filtered = np.empty((0, 3), dtype=np.float32)
        selected_label = None
    else:
        clusters_info = sorted(clusters_info, key=lambda x: x["size"], reverse=True)
        selected_label = clusters_info[0]["label"]
        filtered = raw_removed[labels == selected_label].astype(np.float32)

        if filtered.shape[0] < args.min_cluster_size:
            filtered = np.empty((0, 3), dtype=np.float32)

    np.save(out_dir / "gt_removed_d21_dbscan.npy", filtered.astype(np.float32))
    np.save(out_dir / "dbscan_labels.npy", labels.astype(np.int32))

    save_ply_points(
        out_dir / "gt_removed_d21_dbscan.ply",
        filtered,
        color=(0, 255, 0)
    )

    plot_3views(
        full_pts=full_pts,
        raw_removed=raw_removed.astype(np.float32),
        filtered_removed=filtered,
        out_png=out_dir / "gt_removed_d21_dbscan_3views.png"
    )

    summary = {
        "analysis_dir": str(analysis_dir),
        "out_dir": str(out_dir),
        "eps": float(args.eps),
        "min_samples": int(args.min_samples),
        "min_cluster_size": int(args.min_cluster_size),
        "raw_removed_points": int(raw_removed.shape[0]),
        "noise_points": int((labels == -1).sum()),
        "num_clusters": int(len(clusters_info)),
        "clusters": clusters_info,
        "selected_label": None if selected_label is None else int(selected_label),
        "filtered_points": int(filtered.shape[0]),
        "filtered_frac_of_raw": float(filtered.shape[0] / max(raw_removed.shape[0], 1)),
    }

    with open(out_dir / "dbscan_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[OK] Guardado en:", out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()