#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
visualize_ufrn_binary_errors_plotly_aesthetic.py

Visualización estética/paper-like para evaluación binaria UFRN d21.

Muestra:
- Arcada completa como nube de referencia oscura.
- GT pseudo-label del diente 21.
- TP: predicción correcta.
- FP: falsos positivos.
- FN: falsos negativos.

IMPORTANTE:
Este script asume que pred_npy está en coordenadas RAW STL.
Use:
  pred_positive_points_raw.npy

NO use:
  pred_positive_points.npy
porque ese está en coordenadas normalizadas.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
import plotly.graph_objects as go
from scipy.spatial import cKDTree


# ============================================================
# Carga robusta
# ============================================================

def load_mesh_safe(stl_path: Path):
    mesh = trimesh.load_mesh(stl_path, process=False)

    if isinstance(mesh, trimesh.Scene):
        geoms = list(mesh.geometry.values())
        if len(geoms) == 0:
            raise RuntimeError(f"Scene vacía: {stl_path}")
        mesh = trimesh.util.concatenate(geoms)

    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"No se pudo cargar como Trimesh: {stl_path}")

    if mesh.vertices is None or len(mesh.vertices) == 0:
        raise RuntimeError(f"Malla sin vértices: {stl_path}")

    return mesh


def load_mesh_points(stl_path, n=120000, seed=42):
    stl_path = Path(stl_path)

    mesh = load_mesh_safe(stl_path)

    np.random.seed(int(seed))

    try:
        pts, _ = trimesh.sample.sample_surface(mesh, int(n))
    except Exception as e:
        print("[WARN] sample_surface falló. Usando vértices.", e)
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        replace = verts.shape[0] < int(n)
        idx = np.random.choice(verts.shape[0], int(n), replace=replace)
        pts = verts[idx]

    pts = np.asarray(pts, dtype=np.float32)

    if not np.isfinite(pts).all():
        pts = np.nan_to_num(pts, nan=0.0, posinf=0.0, neginf=0.0)

    return pts.astype(np.float32)


def load_points_npy(path):
    path = Path(path)

    pts = np.load(path, allow_pickle=True)
    pts = np.asarray(pts, dtype=np.float32)

    if pts.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    if pts.ndim == 1:
        if pts.shape[0] == 3:
            pts = pts.reshape(1, 3)
        else:
            raise RuntimeError(f"Archivo {path} no tiene forma Nx3. Shape: {pts.shape}")

    if pts.ndim > 2:
        pts = pts.reshape(-1, pts.shape[-1])

    if pts.shape[1] != 3:
        raise RuntimeError(f"Archivo {path} no tiene 3 columnas. Shape: {pts.shape}")

    if not np.isfinite(pts).all():
        pts = np.nan_to_num(pts, nan=0.0, posinf=0.0, neginf=0.0)

    return pts.astype(np.float32)


# ============================================================
# Utilidades robustas
# ============================================================

def safe_select(points, mask):
    selected = []

    for p, m in zip(points, mask):
        if bool(m):
            selected.append(p)

    if len(selected) == 0:
        return np.empty((0, 3), dtype=np.float32)

    return np.asarray(selected, dtype=np.float32)


def safe_invert_mask(mask):
    return [not bool(m) for m in mask]


def safe_count(mask):
    c = 0
    for m in mask:
        if bool(m):
            c += 1
    return int(c)


def knn_label(points, target_points, radius):
    """
    Retorna lista booleana:
    True si cada punto de `points` cae a distancia <= radius
    de algún punto en `target_points`.
    """
    points = np.asarray(points, dtype=np.float32)
    target_points = np.asarray(target_points, dtype=np.float32)

    if points.shape[0] == 0:
        return []

    if target_points.shape[0] == 0:
        return [False for _ in range(points.shape[0])]

    tree = cKDTree(target_points)
    d, _ = tree.query(points, k=1)

    mask = []
    for val in d:
        mask.append(float(val) <= float(radius))

    return mask


def approx_metrics(tp_pred, fp_pred, fn_gt, pred_pts, gt_pts):
    """
    Métricas aproximadas de cobertura espacial.
    No reemplazan las métricas punto-a-punto del dataset 8192,
    pero sirven para evaluar visualmente con nubes RAW.
    """
    tp = int(len(tp_pred))
    fp = int(len(fp_pred))
    fn = int(len(fn_gt))

    precision = tp / max(tp + fp, 1)

    gt_detected = max(int(len(gt_pts)) - fn, 0)
    recall = gt_detected / max(int(len(gt_pts)), 1)

    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    # IoU aproximada usando TP en espacio de predicción y FN en espacio GT.
    iou = tp / max(tp + fp + fn, 1)

    return {
        "gt_points": int(len(gt_pts)),
        "pred_points": int(len(pred_pts)),
        "tp_pred_points": int(tp),
        "fp_pred_points": int(fp),
        "fn_gt_points": int(fn),
        "precision_approx": float(precision),
        "recall_approx": float(recall),
        "f1_approx": float(f1),
        "iou_approx": float(iou),
    }


def centroid(points):
    points = np.asarray(points, dtype=np.float32)

    if points.shape[0] == 0:
        return None

    c = points.mean(axis=0)

    return [float(c[0]), float(c[1]), float(c[2])]


# ============================================================
# Plotly helpers
# ============================================================

def add_cloud_trace(
    fig,
    pts,
    name,
    color,
    size,
    opacity,
    symbol="circle",
    showlegend=True
):
    pts = np.asarray(pts, dtype=np.float32)

    if pts.shape[0] == 0:
        return

    fig.add_trace(
        go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode="markers",
            marker=dict(
                size=float(size),
                color=color,
                opacity=float(opacity),
                symbol=symbol
            ),
            name=name,
            showlegend=showlegend
        )
    )


def add_centroid_trace(fig, pts, name, color):
    c = centroid(pts)

    if c is None:
        return

    fig.add_trace(
        go.Scatter3d(
            x=[c[0]],
            y=[c[1]],
            z=[c[2]],
            mode="markers+text",
            marker=dict(
                size=8,
                color=color,
                opacity=1.0,
                symbol="diamond"
            ),
            text=[name],
            textposition="top center",
            name=name,
            showlegend=True
        )
    )


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--full_stl", required=True)
    ap.add_argument("--gt_npy", required=True)
    ap.add_argument("--pred_npy", required=True)

    ap.add_argument("--radius", type=float, default=1.25)
    ap.add_argument("--mesh_points", type=int, default=120000)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--out_dir", required=True)

    # Estética
    ap.add_argument("--mesh_size", type=float, default=1.2)
    ap.add_argument("--mesh_opacity", type=float, default=0.18)

    ap.add_argument("--gt_size", type=float, default=2.5)
    ap.add_argument("--gt_opacity", type=float, default=0.45)

    ap.add_argument("--tp_size", type=float, default=4.5)
    ap.add_argument("--fp_size", type=float, default=5.0)
    ap.add_argument("--fn_size", type=float, default=3.5)

    ap.add_argument("--tp_opacity", type=float, default=0.98)
    ap.add_argument("--fp_opacity", type=float, default=0.98)
    ap.add_argument("--fn_opacity", type=float, default=0.85)

    ap.add_argument("--height", type=int, default=950)
    ap.add_argument("--width", type=int, default=1450)

    ap.add_argument(
        "--keep_gt_in_background",
        action="store_true",
        help="Si se activa, no remueve la zona GT del fondo negro."
    )

    ap.add_argument(
        "--show_centroids",
        action="store_true",
        help="Muestra centroides de GT y predicción."
    )

    ap.add_argument(
        "--hide_axes",
        action="store_true",
        help="Oculta ejes y grillas para una figura más limpia."
    )

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Cargando malla completa...")
    full_pts = load_mesh_points(
        args.full_stl,
        n=args.mesh_points,
        seed=args.seed
    )

    print("[INFO] Cargando GT d21...")
    gt_pts = load_points_npy(args.gt_npy)

    print("[INFO] Cargando predicción...")
    pred_pts = load_points_npy(args.pred_npy)

    print("[INFO] full_pts:", full_pts.shape)
    print("[INFO] gt_pts:", gt_pts.shape)
    print("[INFO] pred_pts:", pred_pts.shape)

    print("[INFO] Calculando TP / FP / FN...")

    pred_is_gt = knn_label(
        pred_pts,
        gt_pts,
        args.radius
    )

    gt_is_pred = knn_label(
        gt_pts,
        pred_pts,
        args.radius
    )

    tp_pred = safe_select(
        pred_pts,
        pred_is_gt
    )

    fp_pred = safe_select(
        pred_pts,
        safe_invert_mask(pred_is_gt)
    )

    fn_gt = safe_select(
        gt_pts,
        safe_invert_mask(gt_is_pred)
    )

    # ------------------------------------------------------------
    # Fondo = arcada completa, idealmente sin la zona del d21
    # ------------------------------------------------------------

    if args.keep_gt_in_background:
        bg_pts = full_pts
    else:
        print("[INFO] Removiendo región GT del fondo negro...")
        full_is_gt = knn_label(
            full_pts,
            gt_pts,
            args.radius
        )

        bg_pts = safe_select(
            full_pts,
            safe_invert_mask(full_is_gt)
        )

    metrics = approx_metrics(
        tp_pred=tp_pred,
        fp_pred=fp_pred,
        fn_gt=fn_gt,
        pred_pts=pred_pts,
        gt_pts=gt_pts
    )

    metrics["radius"] = float(args.radius)
    metrics["gt_centroid"] = centroid(gt_pts)
    metrics["pred_centroid"] = centroid(pred_pts)
    metrics["tp_centroid"] = centroid(tp_pred)
    metrics["fp_centroid"] = centroid(fp_pred)
    metrics["fn_centroid"] = centroid(fn_gt)

    # ------------------------------------------------------------
    # Guardar arrays útiles
    # ------------------------------------------------------------

    np.save(out_dir / "tp_pred_points.npy", np.asarray(tp_pred, dtype=np.float32))
    np.save(out_dir / "fp_pred_points.npy", np.asarray(fp_pred, dtype=np.float32))
    np.save(out_dir / "fn_gt_points.npy", np.asarray(fn_gt, dtype=np.float32))
    np.save(out_dir / "background_points.npy", np.asarray(bg_pts, dtype=np.float32))

    with open(out_dir / "summary_visual_eval.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # ------------------------------------------------------------
    # Figura
    # ------------------------------------------------------------

    fig = go.Figure()

    # Fondo negro puro, pero con transparencia.
    # Se usa #000000 explícitamente.
    add_cloud_trace(
        fig,
        bg_pts,
        name="Arcada completa sin d21 aprox.",
        color="#000000",
        size=args.mesh_size,
        opacity=args.mesh_opacity
    )

    # GT como capa semi-transparente
    add_cloud_trace(
        fig,
        gt_pts,
        name="GT d21 pseudo-label",
        color="#008000",
        size=args.gt_size,
        opacity=args.gt_opacity
    )

    # FN primero, para ver zonas perdidas sobre GT
    add_cloud_trace(
        fig,
        fn_gt,
        name="FN: GT no detectado",
        color="#FFA500",
        size=args.fn_size,
        opacity=args.fn_opacity
    )

    # FP
    add_cloud_trace(
        fig,
        fp_pred,
        name="FP: predicción fuera del GT",
        color="#FF0000",
        size=args.fp_size,
        opacity=args.fp_opacity
    )

    # TP al final para que quede encima
    add_cloud_trace(
        fig,
        tp_pred,
        name="TP: predicción correcta",
        color="#0057FF",
        size=args.tp_size,
        opacity=args.tp_opacity
    )

    if args.show_centroids:
        add_centroid_trace(
            fig,
            gt_pts,
            "Centroide GT",
            "#008000"
        )

        add_centroid_trace(
            fig,
            pred_pts,
            "Centroide pred",
            "#FF0000"
        )

    title = (
        "UFRN d21 binary fine-tuning | "
        "Fondo negro = arcada sin d21 | "
        "TP azul | FP rojo | FN naranja"
        "<br>"
        f"Pred={metrics['pred_points']} | "
        f"GT={metrics['gt_points']} | "
        f"F1 aprox={metrics['f1_approx']:.4f} | "
        f"IoU aprox={metrics['iou_approx']:.4f}"
    )

    if args.hide_axes:
        axis_cfg = dict(
            visible=False,
            showbackground=False,
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            title=""
        )
    else:
        axis_cfg = dict(
            title="",
            showbackground=True,
            backgroundcolor="rgba(255,255,255,1)",
            gridcolor="rgba(180,180,180,0.22)",
            zerolinecolor="rgba(0,0,0,0.25)",
            showspikes=False
        )

    fig.update_layout(
        title=dict(
            text=title,
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
            font=dict(size=18)
        ),
        scene=dict(
            xaxis=axis_cfg,
            yaxis=axis_cfg,
            zaxis=axis_cfg,
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.65, y=1.75, z=0.95),
                center=dict(x=0.0, y=0.0, z=0.0)
            )
        ),
        template="plotly_white",
        height=int(args.height),
        width=int(args.width),
        legend=dict(
            x=0.78,
            y=0.96,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(0,0,0,0.15)",
            borderwidth=1,
            font=dict(size=13)
        ),
        margin=dict(l=0, r=0, t=70, b=0)
    )

    html_path = out_dir / "visualization_binary_errors_aesthetic.html"
    fig.write_html(str(html_path))

    print("[OK] HTML guardado en:", html_path)
    print("[OK] Métricas guardadas en:", out_dir / "summary_visual_eval.json")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()