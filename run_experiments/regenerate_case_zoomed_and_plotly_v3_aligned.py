#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
regenerate_case_zoomed_and_plotly_v3_aligned.py

V3:
- Mantiene la lógica de la v2/fix2.
- Corrige el problema clave: la malla raw está en coordenadas originales,
  mientras la nube 8192 está normalizada. Esta versión alinea la malla raw
  al espacio de la nube 8192 antes de generar HTML Plotly.

Genera:
common_zoomed/
  01_raw_mesh_zoom.png
  02_sampled_200k_labeled_zoom.png
  03_final_8192_labeled_zoom.png

plotly/
  raw_mesh.html
  gt_8192.html
  raw_plus_gt_8192.html

Si existe pred_labels.npy dentro de cada carpeta de modelo:
plotly/
  pointnet_pred_8192.html
  pointnet_pred_d21.html
  pointnet_errors_all.html
  pointnet_errors_d21.html
  ...
  pointnettransformer_pred_8192.html
  pointnettransformer_pred_d21.html
  pointnettransformer_errors_all.html
  pointnettransformer_errors_d21.html

IMPORTANTE:
- Las PNG estándar 04_inference_all.png, 05_inference_errors.png, 06_inference_d21.png
  no pueden transformarse mágicamente en 3D interactivo real.
- Para tener Plotly 3D por modelo se necesita pred_labels.npy.
- Si no existe pred_labels.npy, esta v3 deja registrado en meta que no pudo generar
  la vista interactiva 3D de ese modelo.

Uso:
python3 -u regenerate_case_zoomed_and_plotly_v3_aligned.py \
  --case_dir /home/htaucare/Tesis_Amaro/case_comparisons/best4_test_v2/row_005_6BWQC0CT_upper \
  --zoom 1.8 \
  --max_points_200k 80000 \
  --d21_class 8
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pyvista as pv
import trimesh

try:
    import plotly.graph_objects as go
except Exception:
    go = None


# ============================================================
# IO
# ============================================================

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ============================================================
# Mesh loading
# ============================================================

def trimesh_to_pyvista_poly(mesh_tm: trimesh.Trimesh) -> pv.PolyData:
    v = np.asarray(mesh_tm.vertices, dtype=np.float32)
    f = np.asarray(mesh_tm.faces, dtype=np.int64) if mesh_tm.faces is not None else None
    if f is None or f.size == 0:
        return pv.PolyData(v)
    faces = np.hstack([np.full((f.shape[0], 1), 3, dtype=np.int64), f]).ravel()
    poly = pv.PolyData(v, faces)
    try:
        poly = poly.compute_normals(point_normals=True, cell_normals=False, auto_orient_normals=True)
    except Exception:
        pass
    return poly


def load_raw_trimesh(raw_mesh_path: Path) -> Optional[trimesh.Trimesh]:
    if raw_mesh_path is None or not raw_mesh_path.exists():
        return None

    try:
        loaded = trimesh.load(str(raw_mesh_path), force="mesh", process=False)
    except Exception:
        try:
            loaded = trimesh.load(str(raw_mesh_path), force="mesh", process=True)
        except Exception:
            return None

    if isinstance(loaded, trimesh.Scene):
        geos = list(loaded.dump().geometry.values())
        if not geos:
            return None
        mesh_tm = trimesh.util.concatenate(geos)
    else:
        mesh_tm = loaded

    if not isinstance(mesh_tm, trimesh.Trimesh):
        return None

    return mesh_tm


def has_faces(mesh) -> bool:
    if mesh is None:
        return False
    if hasattr(mesh, "n_faces_strict"):
        try:
            return int(mesh.n_faces_strict) > 0
        except Exception:
            pass
    if hasattr(mesh, "n_cells"):
        try:
            return int(mesh.n_cells) > 0
        except Exception:
            pass
    return False


# ============================================================
# Alignment raw mesh -> normalized cloud
# ============================================================

def center_radius(points: np.ndarray) -> Tuple[np.ndarray, float]:
    pts = np.asarray(points, np.float64)
    c = pts.mean(axis=0)
    r = np.linalg.norm(pts - c[None, :], axis=1).max()
    r = float(max(r, 1e-12))
    return c, r


def align_trimesh_to_cloud(mesh_tm: trimesh.Trimesh, xyz_ref: np.ndarray) -> Tuple[trimesh.Trimesh, dict]:
    """
    Alineación simple y defendible para visualización:
    - centra la malla raw en su centroide
    - escala por radio máximo
    - la lleva al centro/radio de la nube 8192

    Esto NO altera el dataset ni las métricas. Solo sirve para overlay visual.
    """
    V = np.asarray(mesh_tm.vertices, np.float64)
    c_raw, r_raw = center_radius(V)
    c_ref, r_ref = center_radius(np.asarray(xyz_ref, np.float64))

    V_aligned = ((V - c_raw[None, :]) / r_raw) * r_ref + c_ref[None, :]

    mesh2 = mesh_tm.copy()
    mesh2.vertices = V_aligned

    info = {
        "raw_centroid": c_raw.tolist(),
        "raw_radius": float(r_raw),
        "ref_centroid": c_ref.tolist(),
        "ref_radius": float(r_ref),
        "alignment": "center_and_radius_to_xyz8192",
    }
    return mesh2, info


# ============================================================
# PyVista rendering
# ============================================================

def plotter_base(width: int, height: int):
    pv.global_theme.multi_samples = 0
    pl = pv.Plotter(off_screen=True, window_size=(int(width), int(height)))
    pl.set_background("white")
    try:
        pl.enable_lightkit()
    except Exception:
        pass
    try:
        pl.enable_shadows()
    except Exception:
        pass
    return pl


def bounds_diag(bounds) -> float:
    b = bounds
    dx = float(b[1] - b[0])
    dy = float(b[3] - b[2])
    dz = float(b[5] - b[4])
    return float(np.sqrt(dx * dx + dy * dy + dz * dz) + 1e-9)


def bounds_center(bounds):
    b = bounds
    return (0.5 * (b[0] + b[1]), 0.5 * (b[2] + b[3]), 0.5 * (b[4] + b[5]))


def compute_base_orientation_from_cloud(X: np.ndarray):
    dir_vec = np.array([1.0, 1.0, 0.65], dtype=np.float64)
    dir_vec /= (np.linalg.norm(dir_vec) + 1e-12)
    viewup = (0.0, 0.0, 1.0)
    return dir_vec, viewup


def apply_camera_from_bounds(pl: pv.Plotter, bounds, base_dir, viewup, dist_mult: float):
    c = np.array(bounds_center(bounds), dtype=np.float64)
    d = np.array(base_dir, dtype=np.float64)
    diag = bounds_diag(bounds)
    pos = (c + d * (float(dist_mult) * diag)).tolist()
    pl.camera.focal_point = c.tolist()
    pl.camera.position = pos
    pl.camera.up = viewup
    try:
        pl.camera.zoom(1.15)
    except Exception:
        pass
    try:
        pl.reset_camera_clipping_range()
    except Exception:
        pass


def label_colors_uint8(Y: np.ndarray) -> np.ndarray:
    import matplotlib.cm as cm
    cmap = cm.get_cmap("tab20", 20)
    Y = np.asarray(Y).astype(np.int64)
    rgb = cmap(Y % 20)[:, :3]
    return (rgb * 255).astype(np.uint8)


def render_raw_mesh_zoom(mesh: pv.PolyData, out_path: Path, title: str, base_dir, viewup,
                         width: int, height: int, zoom: float):
    pl = plotter_base(width, height)
    if has_faces(mesh):
        pl.add_mesh(
            mesh,
            color="lightgray",
            opacity=1.0,
            smooth_shading=True,
            lighting=True,
            show_edges=False,
            ambient=0.22,
            diffuse=0.85,
            specular=0.45,
            specular_power=35,
        )
    else:
        pl.add_points(mesh, color=(160, 160, 160), point_size=3, opacity=0.9)

    dist = max(0.7, 3.2 / float(zoom))
    apply_camera_from_bounds(pl, mesh.bounds, base_dir, viewup, dist_mult=dist)
    pl.add_text(title, position="upper_left", font_size=12, color="black")
    pl.show(screenshot=str(out_path), auto_close=True)


def render_cloud_zoom(X: np.ndarray, Y: np.ndarray, out_path: Path, title: str, base_dir, viewup,
                      width: int, height: int, zoom: float, point_size: float, max_points: int):
    X = np.asarray(X, np.float32)
    Y = np.asarray(Y).reshape(-1).astype(np.int64)

    if X.shape[0] > max_points:
        rng = np.random.default_rng(123)
        idx = rng.choice(X.shape[0], size=max_points, replace=False)
        Xp, Yp = X[idx], Y[idx]
    else:
        Xp, Yp = X, Y

    cloud = pv.PolyData(Xp)
    cloud["rgb"] = label_colors_uint8(Yp)

    pl = plotter_base(width, height)
    pl.add_points(
        cloud,
        scalars="rgb",
        rgb=True,
        render_points_as_spheres=True,
        point_size=point_size,
        opacity=1.0,
    )

    dist = max(0.7, 2.8 / float(zoom))
    apply_camera_from_bounds(pl, cloud.bounds, base_dir, viewup, dist_mult=dist)
    pl.add_text(title, position="upper_left", font_size=12, color="black")
    pl.show(screenshot=str(out_path), auto_close=True)


# ============================================================
# Plotly
# ============================================================

def label_name_from_id(v: int) -> str:
    v = int(v)
    special = {
        0: "0 — fondo/encía",
        1: "1 — d11",
        8: "8 — d21",
        9: "9 — d22",
    }
    return special.get(v, f"{v} — clase {v}")


def make_plotly_layout(title: str):
    return dict(
        title=title,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=45, b=0),
        legend=dict(title="Clase", itemsizing="constant", font=dict(size=11)),
        showlegend=True,
    )


def class_color(cls: int) -> str:
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    cmap = cm.get_cmap("tab20", 20)
    return mcolors.to_hex(cmap(int(cls) % 20))


def write_plotly_raw_mesh(mesh_tm: trimesh.Trimesh, out_html: Path, title: str):
    if go is None:
        return False
    V = np.asarray(mesh_tm.vertices)
    F = np.asarray(mesh_tm.faces)
    fig = go.Figure()
    fig.add_trace(go.Mesh3d(
        x=V[:, 0], y=V[:, 1], z=V[:, 2],
        i=F[:, 0], j=F[:, 1], k=F[:, 2],
        color="lightgray",
        opacity=1.0,
        name="raw mesh alineada",
        lighting=dict(ambient=0.35, diffuse=0.8, specular=0.35, roughness=0.35),
    ))
    fig.update_layout(**make_plotly_layout(title))
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    return True


def write_plotly_pointcloud(xyz, labels, out_html, title, point_size=3, max_points=8192):
    if go is None:
        return False

    xyz = np.asarray(xyz, np.float32)
    labels = np.asarray(labels).reshape(-1).astype(np.int64)

    if xyz.shape[0] > max_points:
        rng = np.random.default_rng(123)
        idx = rng.choice(xyz.shape[0], size=max_points, replace=False)
        xyz, labels = xyz[idx], labels[idx]

    fig = go.Figure()

    for cls in sorted(np.unique(labels).tolist()):
        mask = labels == int(cls)
        pts = xyz[mask]
        if pts.shape[0] == 0:
            continue
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode="markers",
            marker=dict(size=point_size, color=class_color(int(cls)), opacity=0.95),
            name=label_name_from_id(int(cls)),
            hovertemplate=(
                f"clase={label_name_from_id(int(cls))}<br>"
                "x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>"
            ),
        ))

    fig.update_layout(**make_plotly_layout(title))
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    return True


def write_plotly_raw_plus_points(mesh_tm, xyz, labels, out_html, title, point_size=3):
    if go is None:
        return False

    V = np.asarray(mesh_tm.vertices)
    F = np.asarray(mesh_tm.faces)
    xyz = np.asarray(xyz, np.float32)
    labels = np.asarray(labels).reshape(-1).astype(np.int64)

    fig = go.Figure()

    fig.add_trace(go.Mesh3d(
        x=V[:, 0], y=V[:, 1], z=V[:, 2],
        i=F[:, 0], j=F[:, 1], k=F[:, 2],
        color="lightgray",
        opacity=0.18,
        name="raw mesh alineada",
        showlegend=True,
    ))

    for cls in sorted(np.unique(labels).tolist()):
        mask = labels == int(cls)
        pts = xyz[mask]
        if pts.shape[0] == 0:
            continue
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode="markers",
            marker=dict(size=point_size, color=class_color(int(cls)), opacity=0.95),
            name=label_name_from_id(int(cls)),
            hovertemplate=(
                f"clase={label_name_from_id(int(cls))}<br>"
                "x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>"
            ),
        ))

    fig.update_layout(**make_plotly_layout(title))
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    return True


def write_plotly_errors(xyz, err_mask, out_html, title):
    if go is None:
        return False

    xyz = np.asarray(xyz, np.float32)
    err_mask = np.asarray(err_mask).reshape(-1).astype(bool)

    fig = go.Figure()

    ok = ~err_mask
    if ok.any():
        pts = xyz[ok]
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers",
            marker=dict(size=2, color="lightgray", opacity=0.18),
            name="correcto",
        ))

    if err_mask.any():
        pts = xyz[err_mask]
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers",
            marker=dict(size=5, color="red", opacity=1.0),
            name="error",
        ))

    fig.update_layout(**make_plotly_layout(title))
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    return True


def write_plotly_binary(xyz, mask, out_html, title):
    if go is None:
        return False

    xyz = np.asarray(xyz, np.float32)
    mask = np.asarray(mask).reshape(-1).astype(bool)

    fig = go.Figure()

    rest = ~mask
    if rest.any():
        pts = xyz[rest]
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers",
            marker=dict(size=2, color="lightgray", opacity=0.22),
            name="resto/fondo",
        ))

    if mask.any():
        pts = xyz[mask]
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers",
            marker=dict(size=5, color="red", opacity=1.0),
            name="d21",
        ))

    fig.update_layout(**make_plotly_layout(title))
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    return True


# ============================================================
# main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case_dir", required=True)
    ap.add_argument("--zoom", type=float, default=1.8)
    ap.add_argument("--width", type=int, default=1800)
    ap.add_argument("--height", type=int, default=1300)
    ap.add_argument("--max_points_200k", type=int, default=80000)
    ap.add_argument("--d21_class", type=int, default=8)
    args = ap.parse_args()

    case_dir = Path(args.case_dir).resolve()
    meta = load_json(case_dir / "meta.json")

    common_dir = case_dir / "common"
    bundle_path = common_dir / "bundle_case_data.npz"
    if not bundle_path.exists():
        raise FileNotFoundError(f"No existe bundle: {bundle_path}")

    z = np.load(bundle_path)
    xyz200k = z["xyz_200k"]
    y200k = z["y_200k"]
    xyz8192 = z["xyz_8192"]
    y8192 = z["y_8192"]

    sample = meta.get("sample_name", case_dir.name)
    jaw = meta.get("jaw", "upper")
    raw_mesh_path = Path(meta.get("raw_mesh_path", "")) if meta.get("raw_mesh_path", "") else None

    raw_tm_original = load_raw_trimesh(raw_mesh_path) if raw_mesh_path is not None else None
    raw_tm_aligned = None
    raw_pv_aligned = None
    alignment_info = {}

    if raw_tm_original is not None:
        raw_tm_aligned, alignment_info = align_trimesh_to_cloud(raw_tm_original, xyz8192)
        raw_pv_aligned = trimesh_to_pyvista_poly(raw_tm_aligned)

    base_dir, viewup = compute_base_orientation_from_cloud(xyz8192)

    zoom_dir = ensure_dir(case_dir / "common_zoomed")
    plotly_dir = ensure_dir(case_dir / "plotly")

    # PNG zoomed con malla alineada
    if raw_pv_aligned is not None:
        render_raw_mesh_zoom(
            raw_pv_aligned,
            zoom_dir / "01_raw_mesh_zoom.png",
            f"Raw mesh alineada — {sample} ({jaw})",
            base_dir,
            viewup,
            width=int(args.width),
            height=int(args.height),
            zoom=float(args.zoom),
        )

    render_cloud_zoom(
        xyz200k,
        y200k,
        zoom_dir / "02_sampled_200k_labeled_zoom.png",
        f"200k labeled — {sample} ({jaw})",
        base_dir,
        viewup,
        width=int(args.width),
        height=int(args.height),
        zoom=float(args.zoom),
        point_size=3.2,
        max_points=int(args.max_points_200k),
    )

    render_cloud_zoom(
        xyz8192,
        y8192,
        zoom_dir / "03_final_8192_labeled_zoom.png",
        f"8192 labeled — {sample} ({jaw})",
        base_dir,
        viewup,
        width=int(args.width),
        height=int(args.height),
        zoom=float(args.zoom),
        point_size=9.0,
        max_points=8192,
    )

    status = {}

    if raw_tm_aligned is not None:
        status["raw_mesh"] = write_plotly_raw_mesh(
            raw_tm_aligned,
            plotly_dir / "raw_mesh.html",
            f"Raw mesh alineada — {sample} ({jaw})",
        )
        status["raw_plus_gt_8192"] = write_plotly_raw_plus_points(
            raw_tm_aligned,
            xyz8192,
            y8192,
            plotly_dir / "raw_plus_gt_8192.html",
            f"Raw alineada + GT 8192 — {sample} ({jaw})",
        )

    status["gt_8192"] = write_plotly_pointcloud(
        xyz8192,
        y8192,
        plotly_dir / "gt_8192.html",
        f"GT 8192 — {sample} ({jaw})",
    )

    # Plotly por modelo si existe pred_labels.npy
    model_names = ["pointnet", "pointnetpp", "dgcnn", "pointnettransformer"]
    model_status = {}

    for model in model_names:
        model_dir = case_dir / model
        pred_path = model_dir / "pred_labels.npy"

        # Si no hay pred_labels, no hay 3D interactivo real para inference_all/errors/d21.
        if not pred_path.exists():
            model_status[model] = {
                "has_pred_labels": False,
                "message": "No existe pred_labels.npy; solo están disponibles las PNG estándar.",
            }
            continue

        pred = np.load(pred_path).reshape(-1).astype(np.int64)
        if pred.shape[0] != xyz8192.shape[0]:
            model_status[model] = {
                "has_pred_labels": True,
                "error": f"pred.shape={pred.shape} no coincide con xyz8192={xyz8192.shape}",
            }
            continue

        model_status[model] = {"has_pred_labels": True, "generated": {}}

        model_status[model]["generated"]["pred_8192"] = write_plotly_pointcloud(
            xyz8192,
            pred,
            plotly_dir / f"{model}_pred_8192.html",
            f"{model} — inference_all 3D — {sample} ({jaw})",
        )

        model_status[model]["generated"]["pred_d21"] = write_plotly_binary(
            xyz8192,
            pred == int(args.d21_class),
            plotly_dir / f"{model}_pred_d21.html",
            f"{model} — inference_d21 3D — {sample} ({jaw})",
        )

        model_status[model]["generated"]["errors_all"] = write_plotly_errors(
            xyz8192,
            pred != y8192,
            plotly_dir / f"{model}_errors_all.html",
            f"{model} — inference_errors 3D — {sample} ({jaw})",
        )

        model_status[model]["generated"]["errors_d21"] = write_plotly_errors(
            xyz8192,
            (pred == int(args.d21_class)) != (y8192 == int(args.d21_class)),
            plotly_dir / f"{model}_errors_d21.html",
            f"{model} — inference_d21_errors 3D — {sample} ({jaw})",
        )

        # También con raw alineado + pred, si hay malla.
        if raw_tm_aligned is not None:
            model_status[model]["generated"]["raw_plus_pred"] = write_plotly_raw_plus_points(
                raw_tm_aligned,
                xyz8192,
                pred,
                plotly_dir / f"{model}_raw_plus_pred_8192.html",
                f"{model} — Raw alineada + pred 8192 — {sample} ({jaw})",
            )

    out_meta = {
        "case_dir": str(case_dir),
        "sample_name": sample,
        "jaw": jaw,
        "zoom": float(args.zoom),
        "common_zoomed_dir": str(zoom_dir),
        "plotly_dir": str(plotly_dir),
        "plotly_available": bool(go is not None),
        "alignment_info": alignment_info,
        "common_status": status,
        "model_status": model_status,
        "note": (
            "Los HTML 3D de cada modelo requieren pred_labels.npy. "
            "Las PNG estándar no contienen coordenadas/labels predichos suficientes para reconstruir 3D interactivo."
        ),
    }
    save_json(out_meta, case_dir / "interactive_views_meta.json")

    print(f"[OK] PNGs zoomed: {zoom_dir}")
    print(f"[OK] Plotly HTML : {plotly_dir}")
    print("[INFO] raw_plus_gt_8192.html ahora usa malla raw alineada al espacio normalizado de 8192.")
    print("[INFO] Para inference_all/errors/d21 en Plotly por modelo se requiere pred_labels.npy.")


if __name__ == "__main__":
    main()
