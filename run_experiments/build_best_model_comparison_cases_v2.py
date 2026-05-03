#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_best_model_comparison_cases_v2.py

V2:
- Genera carpetas comparativas por caso para los 4 mejores modelos.
- Guarda:
    * raw mesh png
    * 200k labeled png
    * 8192 labeled png
    * xyz_8192.npy
    * gt_labels.npy
    * bundle_case_data.npz
- Copia las PNG estándar de inferencia de cada modelo:
    * inference_all
    * inference_errors
    * inference_d21
- Y NUEVO:
    * si encuentra pred_labels.npy por modelo, genera overlays sobre la malla raw:
      - overlay_multiclass
      - overlay_errors_all
      - overlay_d21_vs_bg
      - overlay_d21_errors

Busca predicciones por modelo en:
  <model_inference_dir>/predictions/test_row{row_i}_{sample_name}_pred.npy
  <model_inference_dir>/predictions/test_row{row_i}_{sample_name}_pred.npz
  <model_inference_dir>/predictions/{sample_name}_{jaw}_pred.npy
  <model_inference_dir>/predictions/{sample_name}_{jaw}_pred.npz
  <model_inference_dir>/predictions/row_{row_i}_pred.npy
  <model_inference_dir>/predictions/row_{row_i}_pred.npz
"""

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyvista as pv
import trimesh
from scipy.spatial import cKDTree


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if path is None or (not path.exists()):
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pick_field(fieldnames: List[str], *cands: str) -> Optional[str]:
    fields = {f.strip(): f for f in fieldnames}
    for c in cands:
        if c in fields:
            return fields[c]
    return None


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
        keys = list(z.keys())
        if len(keys) == 0:
            raise ValueError(f"NPZ vacío: {path}")
        return z[keys[0]]
    else:
        raise ValueError(f"Formato no soportado: {path}")


def read_index_csv(index_path: Path) -> List[Dict[str, str]]:
    return read_csv_rows(index_path)


def build_index_map(rows: List[Dict[str, str]]) -> Dict[int, Dict[str, str]]:
    if not rows:
        return {}
    fieldnames = list(rows[0].keys())
    row_key = pick_field(fieldnames, "row_i", "row", "i", "idx", "index")
    if row_key is None:
        raise ValueError("index_csv sin columna row_i/row/idx/index")
    out = {}
    for r in rows:
        out[int(r[row_key])] = dict(r)
    return out


def key_from_row(row: Dict[str, str]) -> Tuple[str, str]:
    return ((row.get("sample_name") or "").strip(), (row.get("jaw") or "").strip())


def build_sample_jaw_to_position(rows: List[Dict[str, str]]) -> Dict[Tuple[str, str], int]:
    out = {}
    for pos, r in enumerate(rows):
        out[key_from_row(r)] = pos
    return out


def read_manifest(inference_dir: Path) -> List[Dict[str, str]]:
    return read_csv_rows(inference_dir / "inference_manifest.csv")


def manifest_map_by_row(rows: List[Dict[str, str]]) -> Dict[int, Dict[str, str]]:
    out = {}
    for r in rows:
        try:
            out[int(r["row_i"])] = dict(r)
        except Exception:
            continue
    return out


def load_npz_pair(dir_path: Path, split: str):
    X = np.load(dir_path / f"X_{split}.npz")["X"]
    Y = np.load(dir_path / f"Y_{split}.npz")["Y"]
    return X, Y


def find_raw_mesh(raw_root: Path, sample_name: str, jaw: str) -> Optional[Path]:
    sample_name = (sample_name or "").strip()
    jaw = (jaw or "").strip().lower()
    cands = []
    for ext in ("obj", "stl", "ply"):
        cands.extend(raw_root.rglob(f"{sample_name}_{jaw}.{ext}"))
    if not cands:
        return None
    cands = sorted({p.resolve() for p in cands})
    return cands[0]


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


def load_raw_mesh(raw_mesh_path: Path) -> Optional[pv.PolyData]:
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


def bounds_diag(bounds) -> float:
    b = bounds
    dx = float(b[1] - b[0]); dy = float(b[3] - b[2]); dz = float(b[5] - b[4])
    return float(np.sqrt(dx * dx + dy * dy + dz * dz) + 1e-9)


def bounds_center(bounds):
    b = bounds
    return (0.5 * (b[0] + b[1]), 0.5 * (b[2] + b[3]), 0.5 * (b[4] + b[5]))


def compute_base_orientation_from_cloud(X: np.ndarray):
    dir_vec = np.array([1.0, 1.0, 0.65], dtype=np.float64)
    dir_vec /= (np.linalg.norm(dir_vec) + 1e-12)
    viewup = (0.0, 0.0, 1.0)
    return dir_vec, viewup


def apply_camera_from_bounds(pl: pv.Plotter, bounds, base_dir, viewup, dist_mult: float = 3.0):
    c = np.array(bounds_center(bounds), dtype=np.float64)
    d = np.array(base_dir, dtype=np.float64)
    diag = bounds_diag(bounds)
    dist = float(dist_mult) * float(diag)
    pos = (c + d * dist).tolist()
    pl.camera.focal_point = c.tolist()
    pl.camera.position = pos
    pl.camera.up = viewup
    try:
        pl.camera.SetClippingRange(max(1e-4, diag / 2000.0), max(10.0, diag * 50.0))
    except Exception:
        pass


def label_colors_uint8(Y: np.ndarray) -> np.ndarray:
    import matplotlib.cm as cm
    cmap = cm.get_cmap("tab20", 20)
    Y = Y.astype(np.int64)
    rgb = cmap(Y % 20)[:, :3]
    return (rgb * 255).astype(np.uint8)


def make_polydata_points(X: np.ndarray, Y: np.ndarray) -> pv.PolyData:
    cloud = pv.PolyData(X)
    cloud["rgb"] = label_colors_uint8(Y)
    return cloud


def render_raw_mesh(mesh: pv.PolyData, out_path: Path, title: str, base_dir, viewup):
    pl = plotter_base()
    if has_faces(mesh):
        pl.add_mesh(mesh, color="lightgray", opacity=1.0, smooth_shading=True,
                    lighting=True, show_edges=False, ambient=0.22, diffuse=0.85,
                    specular=0.45, specular_power=35)
    else:
        pl.add_points(mesh, color=(160, 160, 160), point_size=2, opacity=0.9)
    apply_camera_from_bounds(pl, mesh.bounds, base_dir, viewup, dist_mult=3.2)
    pl.add_text(title, position="upper_left", font_size=12, color="black")
    pl.show(screenshot=str(out_path), auto_close=True)


def render_labeled_cloud(X: np.ndarray, Y: np.ndarray, out_path: Path, title: str, base_dir, viewup,
                         max_points: int, point_size: float):
    if X.shape[0] > max_points:
        rng = np.random.default_rng(123)
        idx = rng.choice(X.shape[0], size=max_points, replace=False)
        Xp, Yp = X[idx], Y[idx]
    else:
        Xp, Yp = X, Y
    pl = plotter_base()
    cloud = make_polydata_points(Xp, Yp)
    pl.add_points(cloud, scalars="rgb", rgb=True, render_points_as_spheres=True,
                  point_size=point_size, opacity=1.0)
    apply_camera_from_bounds(pl, cloud.bounds, base_dir, viewup, dist_mult=2.8)
    pl.add_text(title, position="upper_left", font_size=12, color="black")
    pl.show(screenshot=str(out_path), auto_close=True)


def copy_inference_pngs(manifest_entry: Dict[str, str], inference_dir: Path, dst_dir: Path) -> Dict[str, bool]:
    out = {"all": False, "errors": False, "d21": False}
    mapping = [("png_all", "04_inference_all.png", "all"),
               ("png_errors", "05_inference_errors.png", "errors"),
               ("png_d21", "06_inference_d21.png", "d21")]
    for src_key, dst_name, out_key in mapping:
        rel = manifest_entry.get(src_key, "")
        if not rel:
            continue
        src = (inference_dir / rel).resolve()
        dst = dst_dir / dst_name
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            out[out_key] = True
    return out


def find_pred_file(inference_dir: Path, row_i: int, sample_name: str, jaw: str) -> Optional[Path]:
    pred_dir = inference_dir / "predictions"
    if not pred_dir.exists():
        return None
    cands = [
        pred_dir / f"test_row{row_i}_{sample_name}_pred.npy",
        pred_dir / f"test_row{row_i}_{sample_name}_pred.npz",
        pred_dir / f"{sample_name}_{jaw}_pred.npy",
        pred_dir / f"{sample_name}_{jaw}_pred.npz",
        pred_dir / f"row_{row_i}_pred.npy",
        pred_dir / f"row_{row_i}_pred.npz",
    ]
    for c in cands:
        if c.exists():
            return c.resolve()
    patterns = [
        f"*row{row_i}*{sample_name}*pred*.npy",
        f"*row{row_i}*{sample_name}*pred*.npz",
        f"*{sample_name}*{jaw}*pred*.npy",
        f"*{sample_name}*{jaw}*pred*.npz",
    ]
    matches = []
    for pat in patterns:
        matches.extend(pred_dir.glob(pat))
    if not matches:
        return None
    return sorted({m.resolve() for m in matches})[0]


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
    return np.asarray(labels, dtype=np.int64).reshape(-1)[idx]


def add_mesh_colored_by_vertex_labels(pl: pv.Plotter, mesh: pv.PolyData, labels: np.ndarray, opacity=0.34):
    colors = make_tab20_colors(labels)
    mesh2 = mesh.copy(); mesh2["rgb"] = colors
    if has_faces(mesh2):
        pl.add_mesh(mesh2, scalars="rgb", rgb=True, opacity=opacity,
                    smooth_shading=True, lighting=True, show_edges=False,
                    ambient=0.22, diffuse=0.85, specular=0.35, specular_power=28)
    else:
        pl.add_points(mesh2, scalars="rgb", rgb=True, point_size=2.0, opacity=opacity)


def add_mesh_binary(pl: pv.Plotter, mesh: pv.PolyData, binary_mask: np.ndarray, opacity=0.32,
                    pos_rgb=(220, 90, 170), neg_rgb=(120, 120, 120)):
    colors = make_binary_colors(binary_mask, true_rgb=pos_rgb, false_rgb=neg_rgb)
    mesh2 = mesh.copy(); mesh2["rgb"] = colors
    if has_faces(mesh2):
        pl.add_mesh(mesh2, scalars="rgb", rgb=True, opacity=opacity,
                    smooth_shading=True, lighting=True, show_edges=False,
                    ambient=0.22, diffuse=0.85, specular=0.35, specular_power=28)
    else:
        pl.add_points(mesh2, scalars="rgb", rgb=True, point_size=2.0, opacity=opacity)


def add_point_overlay_multiclass(pl: pv.Plotter, xyz: np.ndarray, labels: np.ndarray, point_size=8.5):
    cloud = pv.PolyData(np.asarray(xyz, np.float32))
    cloud["rgb"] = make_tab20_colors(labels)
    pl.add_points(cloud, scalars="rgb", rgb=True, render_points_as_spheres=True, point_size=point_size, opacity=0.98)


def add_point_overlay_binary(pl: pv.Plotter, xyz: np.ndarray, binary_mask: np.ndarray, point_size=10.0,
                             pos_rgb=(220, 90, 170), neg_rgb=(150, 150, 150)):
    cloud = pv.PolyData(np.asarray(xyz, np.float32))
    cloud["rgb"] = make_binary_colors(binary_mask, true_rgb=pos_rgb, false_rgb=neg_rgb)
    pl.add_points(cloud, scalars="rgb", rgb=True, render_points_as_spheres=True, point_size=point_size, opacity=0.98)


def add_error_points(pl: pv.Plotter, xyz: np.ndarray, err_mask: np.ndarray,
                     point_size=11.5, err_rgb=(220, 35, 35), base_opacity=0.08):
    xyz = np.asarray(xyz, np.float32)
    err_mask = np.asarray(err_mask).reshape(-1).astype(bool)
    cloud_all = pv.PolyData(xyz)
    cloud_all["rgb"] = np.tile(np.array([[170, 170, 170]], dtype=np.uint8), (xyz.shape[0], 1))
    pl.add_points(cloud_all, scalars="rgb", rgb=True, render_points_as_spheres=True, point_size=5.0, opacity=base_opacity)
    if err_mask.any():
        cloud_err = pv.PolyData(xyz[err_mask])
        cloud_err["rgb"] = np.tile(np.array([[err_rgb[0], err_rgb[1], err_rgb[2]]], dtype=np.uint8), (err_mask.sum(), 1))
        pl.add_points(cloud_err, scalars="rgb", rgb=True, render_points_as_spheres=True, point_size=point_size, opacity=1.0)


def render_overlay_scene(mesh: pv.PolyData, xyz: np.ndarray, out_path: Path, title: str, draw_fn):
    base_dir, viewup = compute_base_orientation_from_cloud(xyz)
    pl = plotter_base()
    draw_fn(pl)
    apply_camera_from_bounds(pl, mesh.bounds if mesh is not None else pv.PolyData(xyz).bounds, base_dir, viewup, dist_mult=3.1)
    pl.add_text(title, position="upper_left", font_size=13, color="black")
    pl.show(screenshot=str(out_path), auto_close=True)


def generate_overlays_for_model(mesh: pv.PolyData, xyz: np.ndarray, gt: np.ndarray, pred: np.ndarray,
                                model_dir: Path, sample_name: str, jaw: str, d21_class: int):
    title_base = f"{sample_name} ({jaw})"
    mesh_pred_labels = project_point_labels_to_mesh_vertices(mesh, xyz, pred)
    pred_d21 = (pred == int(d21_class))
    mesh_pred_d21 = project_point_labels_to_mesh_vertices(mesh, xyz, pred_d21.astype(np.int64)).astype(bool)

    out1 = model_dir / "07_overlay_multiclass.png"
    def draw1(pl):
        add_mesh_colored_by_vertex_labels(pl, mesh, mesh_pred_labels, opacity=0.34)
        add_point_overlay_multiclass(pl, xyz, pred, point_size=8.5)
    render_overlay_scene(mesh, xyz, out1, f"Overlay multiclase — {title_base}", draw1)

    out2 = model_dir / "08_overlay_errors_all.png"
    def draw2(pl):
        if has_faces(mesh):
            pl.add_mesh(mesh, color=(190, 190, 190), opacity=0.22,
                        smooth_shading=True, lighting=True, show_edges=False,
                        ambient=0.22, diffuse=0.85, specular=0.35, specular_power=28)
        else:
            pl.add_points(mesh, color=(180, 180, 180), point_size=2.0, opacity=0.22)
        err_mask = (pred != gt)
        add_error_points(pl, xyz, err_mask, point_size=11.5, err_rgb=(220, 35, 35), base_opacity=0.08)
    render_overlay_scene(mesh, xyz, out2, f"Errores generales (Pred ≠ GT) — {title_base}", draw2)

    out3 = model_dir / "09_overlay_d21_vs_bg.png"
    def draw3(pl):
        add_mesh_binary(pl, mesh, mesh_pred_d21, opacity=0.32,
                        pos_rgb=(220, 90, 170), neg_rgb=(135, 135, 135))
        add_point_overlay_binary(pl, xyz, pred_d21, point_size=10.0,
                                 pos_rgb=(220, 90, 170), neg_rgb=(150, 150, 150))
    render_overlay_scene(mesh, xyz, out3, f"Overlay d21 vs fondo — {title_base}", draw3)

    out4 = model_dir / "10_overlay_d21_errors.png"
    def draw4(pl):
        if has_faces(mesh):
            pl.add_mesh(mesh, color=(190, 190, 190), opacity=0.22,
                        smooth_shading=True, lighting=True, show_edges=False,
                        ambient=0.22, diffuse=0.85, specular=0.35, specular_power=28)
        else:
            pl.add_points(mesh, color=(180, 180, 180), point_size=2.0, opacity=0.22)
        gt_d21 = (gt == int(d21_class))
        err_d21 = (pred_d21 != gt_d21)
        add_error_points(pl, xyz, err_d21, point_size=12.5, err_rgb=(220, 35, 35), base_opacity=0.08)
    render_overlay_scene(mesh, xyz, out4, f"Errores d21 vs fondo — {title_base}", draw4)

    return {
        "overlay_multiclass": str(out1),
        "overlay_errors_all": str(out2),
        "overlay_d21_vs_bg": str(out3),
        "overlay_d21_errors": str(out4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", required=True)
    parser.add_argument("--merged_200k_dir", required=True)
    parser.add_argument("--final_8192_dir", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--index_csv", required=True)
    parser.add_argument("--pointnet_inference_dir", required=True)
    parser.add_argument("--pointnetpp_inference_dir", required=True)
    parser.add_argument("--dgcnn_inference_dir", required=True)
    parser.add_argument("--transformer_inference_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--mode", default="rows", choices=["rows", "manifest_intersection", "all_rows"])
    parser.add_argument("--rows", type=int, nargs="*", default=None)
    parser.add_argument("--max_points_200k", type=int, default=80000)
    parser.add_argument("--point_size_200k", type=float, default=3.0)
    parser.add_argument("--point_size_8192", type=float, default=7.0)
    parser.add_argument("--d21_class", type=int, default=8)
    args = parser.parse_args()

    raw_root = Path(args.raw_root).resolve()
    merged_dir = Path(args.merged_200k_dir).resolve()
    final_dir = Path(args.final_8192_dir).resolve()
    out_dir = ensure_dir(Path(args.out_dir).resolve())

    inf_dirs = {
        "pointnet": Path(args.pointnet_inference_dir).resolve(),
        "pointnetpp": Path(args.pointnetpp_inference_dir).resolve(),
        "dgcnn": Path(args.dgcnn_inference_dir).resolve(),
        "pointnettransformer": Path(args.transformer_inference_dir).resolve(),
    }

    X200k_all, Y200k_all = load_npz_pair(merged_dir, args.split)
    X8192_all, Y8192_all = load_npz_pair(final_dir, args.split)

    final_rows = read_index_csv(Path(args.index_csv))
    final_by_row = build_index_map(final_rows)
    merged_rows = read_index_csv(merged_dir / f"index_{args.split}.csv")
    merged_key_to_pos = build_sample_jaw_to_position(merged_rows)
    manifests = {name: read_manifest(p) for name, p in inf_dirs.items()}
    manifests_by_row = {name: manifest_map_by_row(rows) for name, rows in manifests.items()}

    if args.mode == "rows":
        if not args.rows:
            raise ValueError("--mode rows requiere --rows")
        rows_to_use = sorted(set(int(r) for r in args.rows))
    elif args.mode == "manifest_intersection":
        sets = [set(mp.keys()) for mp in manifests_by_row.values()]
        rows_to_use = sorted(set.intersection(*sets)) if sets else []
    else:
        rows_to_use = sorted(final_by_row.keys())

    print(f"[INFO] rows seleccionadas: {len(rows_to_use)}")

    for row_i in rows_to_use:
        if row_i not in final_by_row:
            print(f"[WARN] row {row_i} no existe en final index")
            continue

        final_meta = final_by_row[row_i]
        sample_name, jaw = key_from_row(final_meta)
        merged_key = (sample_name, jaw)
        if merged_key not in merged_key_to_pos:
            print(f"[WARN] row {row_i} no existe en merged index para {merged_key}")
            continue
        pos_merged = merged_key_to_pos[merged_key]

        case_root = ensure_dir(Path(args.out_dir).resolve() / f"row_{row_i:03d}_{sanitize(sample_name)}_{sanitize(jaw)}")
        common_dir = ensure_dir(case_root / "common")

        X200k = X200k_all[pos_merged]
        Y200k = Y200k_all[pos_merged]
        X8192 = X8192_all[row_i]
        Y8192 = Y8192_all[row_i]

        np.save(common_dir / "xyz_8192.npy", X8192.astype(np.float32))
        np.save(common_dir / "gt_labels.npy", Y8192.astype(np.int64))
        np.savez_compressed(common_dir / "bundle_case_data.npz",
                            xyz_200k=X200k.astype(np.float32), y_200k=Y200k.astype(np.int64),
                            xyz_8192=X8192.astype(np.float32), y_8192=Y8192.astype(np.int64))

        raw_mesh_path = find_raw_mesh(raw_root, sample_name, jaw)
        raw_mesh = load_raw_mesh(raw_mesh_path) if raw_mesh_path else None
        base_dir, viewup = compute_base_orientation_from_cloud(X8192)

        if raw_mesh is not None:
            render_raw_mesh(raw_mesh, common_dir / "01_raw_mesh.png", f"Raw mesh — {sample_name} ({jaw})", base_dir, viewup)

        render_labeled_cloud(X200k, Y200k, common_dir / "02_sampled_200k_labeled.png",
                             f"200k labeled — {sample_name} ({jaw})", base_dir, viewup,
                             max_points=int(args.max_points_200k), point_size=float(args.point_size_200k))
        render_labeled_cloud(X8192, Y8192, common_dir / "03_final_8192_labeled.png",
                             f"8192 labeled — {sample_name} ({jaw})", base_dir, viewup,
                             max_points=8192, point_size=float(args.point_size_8192))

        model_report = {}
        for model_name, inf_dir in inf_dirs.items():
            model_dir = ensure_dir(case_root / model_name)
            mani = manifests_by_row[model_name].get(row_i, {})
            copy_status = {"all": False, "errors": False, "d21": False}
            if mani:
                copy_status = copy_inference_pngs(mani, inf_dir, model_dir)

            pred_file = find_pred_file(inf_dir, row_i, sample_name, jaw)
            pred_status = {"found_pred_file": bool(pred_file), "pred_file": str(pred_file) if pred_file else ""}
            overlay_status = {"generated": False, "files": {}}

            if pred_file is not None and raw_mesh is not None:
                try:
                    pred = np.asarray(load_npy_or_npz(pred_file), np.int64).reshape(-1)
                    if pred.shape[0] != X8192.shape[0]:
                        raise ValueError(f"pred.shape[0]={pred.shape[0]} != {X8192.shape[0]}")
                    np.save(model_dir / "pred_labels.npy", pred.astype(np.int64))
                    files = generate_overlays_for_model(raw_mesh, X8192, Y8192, pred, model_dir, sample_name, jaw, int(args.d21_class))
                    overlay_status = {"generated": True, "files": files}
                except Exception as e:
                    overlay_status = {"generated": False, "error": str(e), "files": {}}

            model_report[model_name] = {
                "copy_status": copy_status,
                "pred_status": pred_status,
                "overlay_status": overlay_status,
            }

        meta = {
            "row_final": int(row_i),
            "row_merged_200k_pos": int(pos_merged),
            "sample_name": sample_name,
            "jaw": jaw,
            "final_index_meta": final_meta,
            "raw_mesh_path": str(raw_mesh_path) if raw_mesh_path else "",
            "d21_class": int(args.d21_class),
            "models": {k: str(v) for k, v in inf_dirs.items()},
            "model_report": model_report,
            "common_files": {
                "xyz_8192": str(common_dir / "xyz_8192.npy"),
                "gt_labels": str(common_dir / "gt_labels.npy"),
                "bundle_case_data": str(common_dir / "bundle_case_data.npz"),
            },
        }
        save_json(meta, case_root / "meta.json")
        print(f"[OK] {case_root}")

    print(f"\nListo ✅ salida en: {Path(args.out_dir).resolve()}")


if __name__ == "__main__":
    main()
