#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
render_overlay_mesh_with_predictions_v1.py

Genera vistas "pro" sobreponiendo la malla raw con:
- predicción multiclase coloreada
- errores generales (pred != gt)
- predicción binaria d21 vs fondo
- errores d21 vs fondo

Modo de uso principal:
1) Caso simple con archivos sueltos:
python3 -u render_overlay_mesh_with_predictions_v1.py \
  --raw_mesh /ruta/caso_upper.obj \
  --xyz_npz /ruta/bundle_case_data.npz \
  --pred_labels /ruta/pred_labels.npy \
  --gt_labels /ruta/gt_labels.npy \
  --sample_name 6BWQC0CT \
  --jaw upper \
  --out_dir /ruta/salida \
  --d21_class 8

2) Si ya tiene bundle_case_data.npz con xyz_8192 / y_8192:
   - --xyz_npz debe apuntar al bundle_case_data.npz
   - --gt_labels puede omitirse si usa y_8192 del bundle

Qué hace:
- Colorea la malla raw por nearest-neighbor desde la nube 8192 predicha
- También dibuja la nube predicha sobre la malla
- Genera:
  01_overlay_multiclass.png
  02_overlay_errors_all.png
  03_overlay_d21_vs_bg.png
  04_overlay_d21_errors.png
  meta_overlay.json
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pyvista as pv
import trimesh
from scipy.spatial import cKDTree


# ============================================================
# utilidades
# ============================================================

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def sanitize(s: str) -> str:
    import re
    s = (s or "").strip().replace(" ", "_")
    s = re.sub(r"[^a-zA-Z0-9_\-\.]+", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def load_npy_or_npz(path: Path, preferred_key: Optional[str] = None):
    if path.suffix.lower() == ".npy":
        return np.load(path)
    elif path.suffix.lower() == ".npz":
        z = np.load(path)
        if preferred_key is not None and preferred_key in z:
            return z[preferred_key]
        # fallback: primer array
        keys = list(z.keys())
        if len(keys) == 0:
            raise ValueError(f"NPZ vacío: {path}")
        return z[keys[0]]
    else:
        raise ValueError(f"Formato no soportado: {path}")

def load_bundle_xyz_gt(bundle_npz: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    z = np.load(bundle_npz)
    xyz = None
    gt = None
    for k in ("xyz_8192", "X", "xyz"):
        if k in z:
            xyz = z[k]
            break
    for k in ("y_8192", "Y", "gt_labels"):
        if k in z:
            gt = z[k]
            break
    if xyz is None:
        raise ValueError(f"No se encontró xyz en {bundle_npz}")
    return np.asarray(xyz, np.float32), (np.asarray(gt).reshape(-1).astype(np.int64) if gt is not None else None)

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

def load_raw_mesh(raw_mesh_path: Path) -> pv.PolyData:
    try:
        loaded = trimesh.load(str(raw_mesh_path), force="mesh", process=False)
    except Exception:
        loaded = trimesh.load(str(raw_mesh_path), force="mesh", process=True)

    if isinstance(loaded, trimesh.Scene):
        geos = list(loaded.dump().geometry.values())
        if not geos:
            raise ValueError(f"Scene vacía: {raw_mesh_path}")
        mesh_tm = trimesh.util.concatenate(geos)
    else:
        mesh_tm = loaded

    if not isinstance(mesh_tm, trimesh.Trimesh):
        raise ValueError(f"No se pudo cargar como Trimesh: {raw_mesh_path}")

    return trimesh_to_pyvista_poly(mesh_tm)

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

def compute_base_orientation_from_cloud(X: np.ndarray):
    # orientación fija consistente con vistas previas
    dir_vec = np.array([1.0, 1.0, 0.65], dtype=np.float64)
    dir_vec /= (np.linalg.norm(dir_vec) + 1e-12)
    viewup = (0.0, 0.0, 1.0)
    return dir_vec, viewup

def bounds_diag(bounds) -> float:
    b = bounds
    dx = float(b[1] - b[0]); dy = float(b[3] - b[2]); dz = float(b[5] - b[4])
    return float(np.sqrt(dx * dx + dy * dy + dz * dz) + 1e-9)

def bounds_center(bounds):
    b = bounds
    return (0.5*(b[0] + b[1]), 0.5*(b[2] + b[3]), 0.5*(b[4] + b[5]))

def apply_camera_from_bounds(pl: pv.Plotter, bounds, base_dir, viewup, dist_mult: float = 3.0):
    c = np.array(bounds_center(bounds), dtype=np.float64)
    d = np.array(base_dir, dtype=np.float64)
    diag = bounds_diag(bounds)
    dist = float(dist_mult) * diag
    pos = (c + d * dist).tolist()
    pl.camera.focal_point = c.tolist()
    pl.camera.position = pos
    pl.camera.up = viewup
    try:
        pl.camera.SetClippingRange(max(1e-4, diag / 2000.0), max(10.0, diag * 50.0))
    except Exception:
        pass
    try:
        pl.reset_camera_clipping_range()
    except Exception:
        pass

def plotter_base():
    pv.global_theme.multi_samples = 0
    pl = pv.Plotter(off_screen=True, window_size=(1800, 1300))
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

# ============================================================
# colores y proyección de labels a la malla
# ============================================================

def make_tab20_colors(labels: np.ndarray) -> np.ndarray:
    import matplotlib.cm as cm
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    cmap = cm.get_cmap("tab20", 20)
    rgb = cmap(labels % 20)[:, :3]
    return (rgb * 255).astype(np.uint8)

def make_binary_colors(mask: np.ndarray, true_rgb=(220, 90, 170), false_rgb=(120, 120, 120)) -> np.ndarray:
    mask = np.asarray(mask).reshape(-1).astype(bool)
    out = np.zeros((mask.shape[0], 3), dtype=np.uint8)
    out[~mask] = np.array(false_rgb, dtype=np.uint8)
    out[mask] = np.array(true_rgb, dtype=np.uint8)
    return out

def project_point_labels_to_mesh_vertices(mesh: pv.PolyData, xyz: np.ndarray, labels: np.ndarray) -> np.ndarray:
    verts = np.asarray(mesh.points, dtype=np.float32)
    tree = cKDTree(np.asarray(xyz, dtype=np.float32), leafsize=64)
    _, idx = tree.query(verts, k=1, workers=-1)
    out = np.asarray(labels, dtype=np.int64).reshape(-1)[idx]
    return out

# ============================================================
# renders
# ============================================================

def add_mesh_colored_by_vertex_labels(pl: pv.Plotter, mesh: pv.PolyData, labels: np.ndarray, opacity=0.35):
    colors = make_tab20_colors(labels)
    mesh2 = mesh.copy()
    mesh2["rgb"] = colors
    if has_faces(mesh2):
        pl.add_mesh(
            mesh2,
            scalars="rgb",
            rgb=True,
            opacity=opacity,
            smooth_shading=True,
            lighting=True,
            show_edges=False,
            ambient=0.22,
            diffuse=0.85,
            specular=0.35,
            specular_power=28,
        )
    else:
        pl.add_points(mesh2, scalars="rgb", rgb=True, point_size=2.0, opacity=opacity)

def add_mesh_binary(pl: pv.Plotter, mesh: pv.PolyData, binary_mask: np.ndarray, opacity=0.32,
                    pos_rgb=(220, 90, 170), neg_rgb=(120, 120, 120)):
    colors = make_binary_colors(binary_mask, true_rgb=pos_rgb, false_rgb=neg_rgb)
    mesh2 = mesh.copy()
    mesh2["rgb"] = colors
    if has_faces(mesh2):
        pl.add_mesh(
            mesh2,
            scalars="rgb",
            rgb=True,
            opacity=opacity,
            smooth_shading=True,
            lighting=True,
            show_edges=False,
            ambient=0.22,
            diffuse=0.85,
            specular=0.35,
            specular_power=28,
        )
    else:
        pl.add_points(mesh2, scalars="rgb", rgb=True, point_size=2.0, opacity=opacity)

def add_point_overlay_multiclass(pl: pv.Plotter, xyz: np.ndarray, labels: np.ndarray, point_size=8.0):
    cloud = pv.PolyData(np.asarray(xyz, np.float32))
    cloud["rgb"] = make_tab20_colors(labels)
    pl.add_points(
        cloud,
        scalars="rgb",
        rgb=True,
        render_points_as_spheres=True,
        point_size=point_size,
        opacity=0.98,
    )

def add_point_overlay_binary(pl: pv.Plotter, xyz: np.ndarray, binary_mask: np.ndarray, point_size=9.5,
                             pos_rgb=(220, 90, 170), neg_rgb=(140, 140, 140)):
    cloud = pv.PolyData(np.asarray(xyz, np.float32))
    cloud["rgb"] = make_binary_colors(binary_mask, true_rgb=pos_rgb, false_rgb=neg_rgb)
    pl.add_points(
        cloud,
        scalars="rgb",
        rgb=True,
        render_points_as_spheres=True,
        point_size=point_size,
        opacity=0.98,
    )

def add_error_points(pl: pv.Plotter, xyz: np.ndarray, err_mask: np.ndarray,
                     point_size=11.0, err_rgb=(220, 30, 30), base_opacity=0.06):
    xyz = np.asarray(xyz, np.float32)
    err_mask = np.asarray(err_mask).reshape(-1).astype(bool)

    # fondo tenue con la nube completa
    cloud_all = pv.PolyData(xyz)
    cloud_all["rgb"] = np.tile(np.array([[170, 170, 170]], dtype=np.uint8), (xyz.shape[0], 1))
    pl.add_points(
        cloud_all,
        scalars="rgb",
        rgb=True,
        render_points_as_spheres=True,
        point_size=5.0,
        opacity=base_opacity,
    )

    if err_mask.any():
        cloud_err = pv.PolyData(xyz[err_mask])
        cloud_err["rgb"] = np.tile(np.array([[err_rgb[0], err_rgb[1], err_rgb[2]]], dtype=np.uint8), (err_mask.sum(), 1))
        pl.add_points(
            cloud_err,
            scalars="rgb",
            rgb=True,
            render_points_as_spheres=True,
            point_size=point_size,
            opacity=1.0,
        )

def render_scene(mesh: pv.PolyData, xyz: np.ndarray, out_path: Path, title: str, draw_fn):
    base_dir, viewup = compute_base_orientation_from_cloud(xyz)
    pl = plotter_base()
    draw_fn(pl)
    apply_camera_from_bounds(pl, mesh.bounds if mesh is not None else pv.PolyData(xyz).bounds, base_dir, viewup, dist_mult=3.1)
    pl.add_text(title, position="upper_left", font_size=13, color="black")
    pl.show(screenshot=str(out_path), auto_close=True)

# ============================================================
# main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_mesh", required=True, help="OBJ/STL/PLY de la malla raw")
    ap.add_argument("--xyz_npz", required=True, help="bundle_case_data.npz o NPY/NPZ con xyz de la nube final")
    ap.add_argument("--pred_labels", required=True, help="NPY/NPZ con labels predichas por punto")
    ap.add_argument("--gt_labels", default=None, help="NPY/NPZ con GT; si no se entrega y xyz_npz es bundle, intenta usar y_8192")
    ap.add_argument("--sample_name", default="sample")
    ap.add_argument("--jaw", default="upper")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--d21_class", type=int, default=8)
    args = ap.parse_args()

    raw_mesh_path = Path(args.raw_mesh).resolve()
    xyz_path = Path(args.xyz_npz).resolve()
    pred_path = Path(args.pred_labels).resolve()
    gt_path = Path(args.gt_labels).resolve() if args.gt_labels else None
    out_dir = ensure_dir(Path(args.out_dir).resolve())

    mesh = load_raw_mesh(raw_mesh_path)

    if xyz_path.suffix.lower() == ".npz":
        xyz, gt_from_bundle = load_bundle_xyz_gt(xyz_path)
    else:
        xyz = np.asarray(load_npy_or_npz(xyz_path), np.float32)
        gt_from_bundle = None

    pred = np.asarray(load_npy_or_npz(pred_path), np.int64).reshape(-1)
    gt = None
    if gt_path is not None:
        gt = np.asarray(load_npy_or_npz(gt_path), np.int64).reshape(-1)
    elif gt_from_bundle is not None:
        gt = gt_from_bundle

    if xyz.shape[0] != pred.shape[0]:
        raise ValueError(f"xyz ({xyz.shape[0]}) y pred ({pred.shape[0]}) no coinciden")

    if gt is not None and xyz.shape[0] != gt.shape[0]:
        raise ValueError(f"xyz ({xyz.shape[0]}) y gt ({gt.shape[0]}) no coinciden")

    # proyectar pred y binario a vértices de malla
    mesh_pred_labels = project_point_labels_to_mesh_vertices(mesh, xyz, pred)
    pred_d21 = (pred == int(args.d21_class))
    mesh_pred_d21 = project_point_labels_to_mesh_vertices(mesh, xyz, pred_d21.astype(np.int64)).astype(bool)

    title_base = f"{sanitize(args.sample_name)} ({args.jaw})"

    # 1) overlay multiclase
    out1 = out_dir / "01_overlay_multiclass.png"
    def draw1(pl):
        add_mesh_colored_by_vertex_labels(pl, mesh, mesh_pred_labels, opacity=0.34)
        add_point_overlay_multiclass(pl, xyz, pred, point_size=8.5)
    render_scene(mesh, xyz, out1, f"Overlay multiclase — {title_base}", draw1)

    # 2) errores generales
    out2 = out_dir / "02_overlay_errors_all.png"
    def draw2(pl):
        # malla gris base
        if has_faces(mesh):
            pl.add_mesh(
                mesh, color=(190, 190, 190), opacity=0.22,
                smooth_shading=True, lighting=True, show_edges=False,
                ambient=0.22, diffuse=0.85, specular=0.35, specular_power=28
            )
        else:
            pl.add_points(mesh, color=(180, 180, 180), point_size=2.0, opacity=0.22)
        if gt is not None:
            err_mask = (pred != gt)
        else:
            err_mask = np.zeros((xyz.shape[0],), dtype=bool)
        add_error_points(pl, xyz, err_mask, point_size=11.5, err_rgb=(220, 35, 35), base_opacity=0.08)
    render_scene(mesh, xyz, out2, f"Errores generales (Pred ≠ GT) — {title_base}", draw2)

    # 3) d21 vs fondo
    out3 = out_dir / "03_overlay_d21_vs_bg.png"
    def draw3(pl):
        add_mesh_binary(pl, mesh, mesh_pred_d21, opacity=0.32,
                        pos_rgb=(220, 90, 170), neg_rgb=(135, 135, 135))
        add_point_overlay_binary(pl, xyz, pred_d21, point_size=10.0,
                                 pos_rgb=(220, 90, 170), neg_rgb=(150, 150, 150))
    render_scene(mesh, xyz, out3, f"Overlay d21 vs fondo — {title_base}", draw3)

    # 4) errores d21
    out4 = out_dir / "04_overlay_d21_errors.png"
    def draw4(pl):
        if has_faces(mesh):
            pl.add_mesh(
                mesh, color=(190, 190, 190), opacity=0.22,
                smooth_shading=True, lighting=True, show_edges=False,
                ambient=0.22, diffuse=0.85, specular=0.35, specular_power=28
            )
        else:
            pl.add_points(mesh, color=(180, 180, 180), point_size=2.0, opacity=0.22)
        if gt is not None:
            gt_d21 = (gt == int(args.d21_class))
            err_d21 = (pred_d21 != gt_d21)
        else:
            err_d21 = np.zeros((xyz.shape[0],), dtype=bool)
        add_error_points(pl, xyz, err_d21, point_size=12.5, err_rgb=(220, 35, 35), base_opacity=0.08)
    render_scene(mesh, xyz, out4, f"Errores d21 vs fondo — {title_base}", draw4)

    meta = {
        "sample_name": args.sample_name,
        "jaw": args.jaw,
        "raw_mesh": str(raw_mesh_path),
        "xyz_source": str(xyz_path),
        "pred_labels": str(pred_path),
        "gt_labels": str(gt_path) if gt_path else ("bundle_case_data.npz:y_8192" if gt_from_bundle is not None else ""),
        "d21_class": int(args.d21_class),
        "n_points": int(xyz.shape[0]),
        "has_gt": bool(gt is not None),
        "outputs": {
            "overlay_multiclass": str(out1),
            "overlay_errors_all": str(out2),
            "overlay_d21_vs_bg": str(out3),
            "overlay_d21_errors": str(out4),
        },
    }
    save_json(meta, out_dir / "meta_overlay.json")
    print(f"[OK] guardado en: {out_dir}")

if __name__ == "__main__":
    main()
