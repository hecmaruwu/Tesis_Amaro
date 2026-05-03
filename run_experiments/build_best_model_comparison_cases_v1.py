#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyvista as pv
import trimesh

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

def read_index_csv(index_path: Path) -> List[Dict[str, str]]:
    return read_csv_rows(index_path)

def build_index_map(rows: List[Dict[str, str]]) -> Dict[int, Dict[str, str]]:
    if not rows:
        return {}
    fieldnames = list(rows[0].keys())
    row_key = pick_field(fieldnames, "row_i", "row", "i", "idx", "index")
    if row_key is None:
        raise ValueError("index_csv sin columna de row")
    out = {}
    for r in rows:
        out[int(r[row_key])] = dict(r)
    return out

def key_from_row(row: Dict[str, str]) -> Tuple[str, str]:
    return ((row.get("sample_name") or "").strip(), (row.get("jaw") or "").strip())

def build_sample_jaw_to_row(rows: List[Dict[str, str]]) -> Dict[Tuple[str, str], int]:
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
    pl = pv.Plotter(off_screen=True, window_size=(1500, 1100))
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
    return float(np.sqrt(dx*dx + dy*dy + dz*dz) + 1e-9)

def bounds_center(bounds):
    b = bounds
    return (0.5*(b[0]+b[1]), 0.5*(b[2]+b[3]), 0.5*(b[4]+b[5]))

def compute_base_orientation_from_cloud(X: np.ndarray):
    dir_vec = np.array([1.0, 1.0, 0.65], dtype=np.float64)
    dir_vec /= (np.linalg.norm(dir_vec) + 1e-12)
    viewup = (0.0, 0.0, 1.0)
    return dir_vec, viewup

def apply_camera_from_bounds(pl: pv.Plotter, bounds, base_dir, viewup, dist_mult: float):
    c = np.array(bounds_center(bounds), dtype=np.float64)
    d = np.array(base_dir, dtype=np.float64)
    diag = bounds_diag(bounds)
    dist = float(dist_mult) * float(diag)
    pos = (c + d * dist).tolist()
    pl.camera.focal_point = c.tolist()
    pl.camera.position = pos
    pl.camera.up = viewup
    try:
        pl.camera.SetClippingRange(max(1e-4, diag/2000.0), max(10.0, diag*50.0))
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
        pl.add_points(mesh, color=(160,160,160), point_size=2, opacity=0.9)
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
    mapping = [
        ("png_all", "04_inference_all.png", "all"),
        ("png_errors", "05_inference_errors.png", "errors"),
        ("png_d21", "06_inference_d21.png", "d21"),
    ]
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", required=True)
    parser.add_argument("--merged_200k_dir", required=True)
    parser.add_argument("--final_8192_dir", required=True)
    parser.add_argument("--split", default="test", choices=["train","val","test"])
    parser.add_argument("--index_csv", required=True)
    parser.add_argument("--pointnet_inference_dir", required=True)
    parser.add_argument("--pointnetpp_inference_dir", required=True)
    parser.add_argument("--dgcnn_inference_dir", required=True)
    parser.add_argument("--transformer_inference_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--mode", default="rows", choices=["rows","manifest_intersection","all_rows"])
    parser.add_argument("--rows", type=int, nargs="*", default=None)
    parser.add_argument("--max_points_200k", type=int, default=80000)
    parser.add_argument("--point_size_200k", type=float, default=3.0)
    parser.add_argument("--point_size_8192", type=float, default=7.0)
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
    merged_key_to_row = build_sample_jaw_to_row(merged_rows)

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
        if merged_key not in merged_key_to_row:
            print(f"[WARN] row {row_i} no existe en merged index para {merged_key}")
            continue
        row_merged = merged_key_to_row[merged_key]

        case_root = ensure_dir(out_dir / f"row_{row_i:03d}_{sanitize(sample_name)}_{sanitize(jaw)}")
        common_dir = ensure_dir(case_root / "common")

        X200k = X200k_all[row_merged]
        Y200k = Y200k_all[row_merged]
        X8192 = X8192_all[row_i]
        Y8192 = Y8192_all[row_i]

        raw_mesh_path = find_raw_mesh(raw_root, sample_name, jaw)
        raw_mesh = load_raw_mesh(raw_mesh_path) if raw_mesh_path else None

        base_dir, viewup = compute_base_orientation_from_cloud(X8192)

        if raw_mesh is not None:
            render_raw_mesh(raw_mesh, common_dir / "01_raw_mesh.png",
                            f"Raw mesh — {sample_name} ({jaw})", base_dir, viewup)

        render_labeled_cloud(X200k, Y200k, common_dir / "02_sampled_200k_labeled.png",
                             f"200k labeled — {sample_name} ({jaw})",
                             base_dir, viewup,
                             max_points=int(args.max_points_200k),
                             point_size=float(args.point_size_200k))

        render_labeled_cloud(X8192, Y8192, common_dir / "03_final_8192_labeled.png",
                             f"8192 labeled — {sample_name} ({jaw})",
                             base_dir, viewup,
                             max_points=8192,
                             point_size=float(args.point_size_8192))

        np.savez_compressed(
            common_dir / "bundle_case_data.npz",
            xyz_200k=X200k.astype(np.float32),
            y_200k=Y200k.astype(np.int64),
            xyz_8192=X8192.astype(np.float32),
            y_8192=Y8192.astype(np.int64),
        )

        model_copy_status = {}
        for model_name, inf_dir in inf_dirs.items():
            model_dir = ensure_dir(case_root / model_name)
            mani = manifests_by_row[model_name].get(row_i, {})
            if mani:
                model_copy_status[model_name] = copy_inference_pngs(mani, inf_dir, model_dir)
            else:
                model_copy_status[model_name] = {"all": False, "errors": False, "d21": False}

        meta = {
            "row_final": int(row_i),
            "row_merged_200k": int(row_merged),
            "sample_name": sample_name,
            "jaw": jaw,
            "final_index_meta": final_meta,
            "raw_mesh_path": str(raw_mesh_path) if raw_mesh_path else "",
            "models": {k: str(v) for k, v in inf_dirs.items()},
            "model_copy_status": model_copy_status,
            "bundle_npz": str(common_dir / "bundle_case_data.npz"),
        }
        save_json(meta, case_root / "meta.json")
        print(f"[OK] {case_root}")

    print(f"\nListo ✅ salida en: {out_dir}")

if __name__ == "__main__":
    main()
