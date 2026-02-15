#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_pipeline_visualization_example.py

Pipeline visualization (PyVista, offscreen) — versión robusta y paper-friendly

Genera figuras del pipeline:
01: malla raw sólida (OBJ si tiene caras; si no, reconstruye superficie desde puntos 200k)
02: malla raw (wireframe) + overlay puntos 200k  (para que SE VEAN)
03: 200k labeled
04: 200k normalized labeled
05: 8192 labeled (vista base)
06: augment #1 (8192)
07: augment #2 (8192)
08: overlay orig vs aug1 + líneas de desplazamiento

FIX DEFINITIVO para "RAW no se ve / se ve gigante":
- La orientación (dirección de cámara) se aprende desde 8192 (05).
- La distancia SIEMPRE se recalcula por figura con el diag del bounding box de ESA figura.
  => así 01/02 no quedan metidos dentro aunque el raw sea mucho más grande.

FIX DEFINITIVO para "no se ven los 200k puntos / se ve liso":
- En 02 NO se usa malla sólida encima (tapa puntos). Se usa wireframe con baja opacidad.
- Por defecto se dibujan más puntos (max_points_200k=80000) y con opacidad alta.
- Puedes subir/bajar desde CLI.

Uso:
python /home/htaucare/Tesis_Amaro/scripts_last_version/make_pipeline_visualization_example.py \
  --raw_root /home/htaucare/Tesis_Amaro/data/Teeth_3ds/raw \
  --merged_200k_dir /home/htaucare/Tesis_Amaro/data/Teeth_3ds/merged_200000_safe_excl_wisdom_upper_only \
  --final_8192_dir /home/htaucare/Tesis_Amaro/data/Teeth_3ds/fixed_split/8192/upper_only_surf_global_excl_wisdom_seed42_aug2 \
  --split val --idx 0 \
  --out_dir /home/htaucare/Tesis_Amaro/figures/pipeline_example \
  --ext jpg \
  --max_points_200k 80000 \
  --overlay_point_opacity 0.85 \
  --overlay_point_size 3.0
"""

import csv
import argparse
from pathlib import Path
import numpy as np

import pyvista as pv
import trimesh


# =============================================================================
# PyVista compatibility helpers
# =============================================================================

def n_faces_compat(mesh) -> int:
    if mesh is None:
        return 0
    if hasattr(mesh, "n_faces_strict"):
        try:
            return int(mesh.n_faces_strict)
        except Exception:
            pass
    if hasattr(mesh, "n_cells"):
        try:
            return int(mesh.n_cells)
        except Exception:
            pass
    return 0


def has_faces(mesh) -> bool:
    return n_faces_compat(mesh) > 0


# =============================================================================
# Geometry utils
# =============================================================================

def normalize_unit_sphere(X: np.ndarray) -> np.ndarray:
    c = X.mean(axis=0, keepdims=True)
    Xc = X - c
    r = np.max(np.linalg.norm(Xc, axis=1))
    return Xc / (r + 1e-12)


def rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], dtype=np.float32)


def augment_8192(
    X: np.ndarray,
    Y: np.ndarray,
    alpha_deg: float = 15.0,
    smin: float = 0.95,
    smax: float = 1.05,
    sigma: float = 0.005,
    clip: float = 0.02,
    p_drop: float = 0.05,
    min_keep: int = 32,
    seed: int | None = None,
):
    rng = np.random.default_rng(seed)

    theta = np.deg2rad(rng.uniform(-alpha_deg, alpha_deg))
    R = rot_z(theta)
    s = rng.uniform(smin, smax)

    X2 = (X @ R.T) * s

    eps = rng.normal(0.0, sigma, size=X2.shape).astype(np.float32)
    eps = np.clip(eps, -clip, clip)
    X2 = X2 + eps

    keep = rng.random(X2.shape[0]) > p_drop
    if keep.sum() < min_keep:
        idx = rng.choice(X2.shape[0], size=min_keep, replace=False)
        keep[:] = False
        keep[idx] = True

    X2 = X2[keep]
    Y2 = Y[keep]

    meta = {
        "theta_deg": float(np.rad2deg(theta)),
        "scale": float(s),
        "sigma": float(sigma),
        "clip": float(clip),
        "p_drop": float(p_drop),
        "kept_points": int(X2.shape[0]),
    }
    return X2, Y2, meta


# =============================================================================
# Colors per label (tab20)
# =============================================================================

def _label_colors_uint8(Y: np.ndarray) -> np.ndarray:
    import matplotlib.cm as cm
    cmap = cm.get_cmap("tab20", 20)
    Y = Y.astype(np.int64)
    rgb = cmap(Y % 20)[:, :3]
    return (rgb * 255).astype(np.uint8)


def _make_polydata_points(X: np.ndarray, Y: np.ndarray) -> pv.PolyData:
    cloud = pv.PolyData(X)
    cloud["rgb"] = _label_colors_uint8(Y)
    return cloud


# =============================================================================
# Plotter + camera (robusto)
# =============================================================================

def _ensure_parent_dir(path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _plotter_base(bg: str = "white"):
    pv.global_theme.multi_samples = 0
    pl = pv.Plotter(off_screen=True, window_size=(1400, 1050))
    pl.set_background(bg)
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
    return float(np.sqrt(dx*dx + dy*dy + dz*dz) + 1e-9)


def bounds_center(bounds):
    b = bounds
    return (0.5*(b[0]+b[1]), 0.5*(b[2]+b[3]), 0.5*(b[4]+b[5]))


def compute_base_orientation_from_8192(X8192: np.ndarray):
    """
    Solo orientación estable (dirección + up). La distancia se recalcula por figura.
    (No depende de una cámara previa; es estable para todas las figuras)
    """
    dir_vec = np.array([1.0, 1.0, 0.65], dtype=np.float64)
    dir_vec /= (np.linalg.norm(dir_vec) + 1e-12)
    viewup = (0.0, 0.0, 1.0)
    return dir_vec, viewup


def apply_camera_from_bounds(pl: pv.Plotter, bounds, base_dir, viewup, dist_mult: float):
    """
    Pone la cámara mirando al centro de 'bounds', con dist = dist_mult * diag(bounds).
    Mantiene la orientación (base_dir, viewup) para todas las figuras.
    """
    c = np.array(bounds_center(bounds), dtype=np.float64)
    d = np.array(base_dir, dtype=np.float64)
    diag = bounds_diag(bounds)
    dist = float(dist_mult) * float(diag)

    pos = (c + d * dist).tolist()
    pl.camera.focal_point = c.tolist()
    pl.camera.position = pos
    pl.camera.up = viewup

    near = max(1e-4, diag / 2000.0)
    far = max(10.0, diag * 50.0)
    try:
        pl.camera.SetClippingRange(float(near), float(far))
    except Exception:
        pass

    try:
        pl.reset_camera_clipping_range()
    except Exception:
        pass


# =============================================================================
# RAW mesh loading + fallback surface reconstruction
# =============================================================================

def find_raw_upper_obj(raw_root: Path, sample_name: str) -> Path | None:
    for part in sorted(raw_root.glob("data_part_*")):
        cand = part / "upper" / sample_name / f"{sample_name}_upper.obj"
        if cand.exists():
            return cand
    return None


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


def load_raw_mesh_from_obj(obj_path: Path) -> pv.PolyData | None:
    try:
        loaded = trimesh.load(str(obj_path), force="mesh", process=False)
    except Exception:
        try:
            loaded = trimesh.load(str(obj_path), force="mesh", process=True)
        except Exception as e:
            print("⚠ trimesh.load falló:", repr(e))
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

    try:
        mesh_tm.remove_degenerate_faces()
        mesh_tm.remove_duplicate_faces()
    except Exception:
        pass

    return trimesh_to_pyvista_poly(mesh_tm)


def build_surface_from_points(X: np.ndarray, max_points: int = 60000, seed: int = 123):
    rng = np.random.default_rng(seed)
    if X.shape[0] > max_points:
        idx = rng.choice(X.shape[0], size=max_points, replace=False)
        Xs = X[idx]
    else:
        Xs = X

    cloud = pv.PolyData(Xs)

    # alpha adaptativo
    m = min(2000, Xs.shape[0])
    idx2 = rng.choice(Xs.shape[0], size=m, replace=False)
    sample = Xs[idx2]
    scale = float(np.percentile(np.linalg.norm(sample - sample.mean(0), axis=1), 90))
    alpha = max(0.01, 0.08 * scale)

    tet = cloud.delaunay_3d(alpha=alpha)
    surf = tet.extract_surface()
    try:
        surf = surf.compute_normals(point_normals=True, cell_normals=False, auto_orient_normals=True)
    except Exception:
        pass

    return surf, alpha, Xs.shape[0]


# =============================================================================
# Rendering
# =============================================================================

def render_raw_solid(mesh: pv.PolyData, out_path: str | Path, title: str,
                     base_dir, viewup, dist_mult_raw: float):
    out_path = str(out_path)
    _ensure_parent_dir(out_path)
    pl = _plotter_base("white")

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
            specular_power=35
        )
    else:
        pl.add_points(mesh, color=(160, 160, 160), point_size=2,
                      render_points_as_spheres=False, opacity=0.9)

    apply_camera_from_bounds(pl, mesh.bounds, base_dir, viewup, dist_mult=dist_mult_raw)

    if title:
        pl.add_text(title, position="upper_left", font_size=12, color="black")

    pl.show(screenshot=out_path, auto_close=True)


def render_raw_wire_plus_points(
    mesh: pv.PolyData,
    X: np.ndarray,
    out_path: str | Path,
    title: str,
    base_dir, viewup,
    dist_mult_raw: float,
    max_points: int = 80000,
    point_size: float = 3.0,
    point_opacity: float = 0.85,
    mesh_opacity: float = 0.35,
    mesh_line_width: float = 0.5,
):
    """
    (02) Overlay que sí se ve:
    - mesh en wireframe y semitransparente (no tapa puntos)
    - puntos con opacidad alta y tamaño decente
    """
    out_path = str(out_path)
    _ensure_parent_dir(out_path)

    rng = np.random.default_rng(123)
    if X.shape[0] > max_points:
        idx = rng.choice(X.shape[0], size=max_points, replace=False)
        Xp = X[idx]
    else:
        Xp = X

    pl = _plotter_base("white")

    if has_faces(mesh):
        pl.add_mesh(
            mesh,
            style="wireframe",
            color="lightgray",
            line_width=mesh_line_width,
            opacity=mesh_opacity
        )

    pl.add_points(
        pv.PolyData(Xp),
        color=(31, 119, 180),
        render_points_as_spheres=True,
        point_size=point_size,
        opacity=point_opacity
    )

    # cámara basada en bounds de la malla (más estable para overlay)
    apply_camera_from_bounds(pl, mesh.bounds, base_dir, viewup, dist_mult=dist_mult_raw)

    if title:
        pl.add_text(title, position="upper_left", font_size=12, color="black")

    pl.show(screenshot=out_path, auto_close=True)


def render_cloud_labeled(
    X: np.ndarray,
    Y: np.ndarray,
    out_path: str | Path,
    title: str,
    base_dir, viewup,
    dist_mult_cloud: float,
    max_points: int,
    point_size: float,
):
    out_path = str(out_path)
    _ensure_parent_dir(out_path)

    if X.shape[0] > max_points:
        rng = np.random.default_rng(123)
        idx = rng.choice(X.shape[0], size=max_points, replace=False)
        Xp, Yp = X[idx], Y[idx]
    else:
        Xp, Yp = X, Y

    pl = _plotter_base("white")
    cloud = _make_polydata_points(Xp, Yp)

    pl.add_points(
        cloud,
        scalars="rgb",
        rgb=True,
        render_points_as_spheres=True,
        point_size=point_size,
        opacity=1.0
    )

    apply_camera_from_bounds(pl, cloud.bounds, base_dir, viewup, dist_mult=dist_mult_cloud)

    if title:
        pl.add_text(title, position="upper_left", font_size=12, color="black")

    pl.show(screenshot=out_path, auto_close=True)


def render_overlay_displacement(
    X_orig: np.ndarray,
    X_aug: np.ndarray,
    out_path: str | Path,
    title: str,
    base_dir, viewup,
    dist_mult_cloud: float,
    n_lines: int = 300,
    max_points: int = 8192,
    point_size: float = 7.0,
    seed: int = 123,
):
    out_path = str(out_path)
    _ensure_parent_dir(out_path)

    rng = np.random.default_rng(seed)
    n_pair = min(X_orig.shape[0], X_aug.shape[0], max_points)
    Xo = X_orig[:n_pair]
    Xa = X_aug[:n_pair]

    n_lines = int(min(n_lines, n_pair))
    idx_lines = rng.choice(n_pair, size=n_lines, replace=False)

    pl = _plotter_base("white")
    pl.add_points(pv.PolyData(Xo), color=(180, 180, 180),
                  render_points_as_spheres=True, point_size=point_size, opacity=0.55)
    pl.add_points(pv.PolyData(Xa), color=(20, 20, 20),
                  render_points_as_spheres=True, point_size=point_size, opacity=0.85)

    pts = np.vstack([Xo[idx_lines], Xa[idx_lines]])
    lines = []
    for i in range(n_lines):
        lines.extend([2, i, i + n_lines])
    line_poly = pv.PolyData(pts)
    line_poly.lines = np.array(lines, dtype=np.int64)
    pl.add_mesh(line_poly, color=(60, 60, 60), line_width=1.0, opacity=0.75)

    bounds = pv.PolyData(Xo).bounds
    apply_camera_from_bounds(pl, bounds, base_dir, viewup, dist_mult=dist_mult_cloud)

    if title:
        pl.add_text(title, position="upper_left", font_size=12, color="black")

    pl.show(screenshot=out_path, auto_close=True)


# =============================================================================
# Index alignment (merged vs final)
# =============================================================================

def load_npz_pair(dir_path: str | Path, split: str):
    dir_path = Path(dir_path)
    X = np.load(dir_path / f"X_{split}.npz")["X"]
    Y = np.load(dir_path / f"Y_{split}.npz")["Y"]
    return X, Y


def read_index_csv(dir_path: str | Path, split: str):
    p = Path(dir_path) / f"index_{split}.csv"
    if not p.exists():
        return None
    with open(p, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _key_from_row(row: dict) -> tuple | None:
    s = (row.get("sample_name") or "").strip()
    j = (row.get("jaw") or "").strip()
    if s and j:
        return (s, j)
    p = (row.get("path") or "").strip()
    if p:
        return ("path", p)
    return None


def align_idx_by_indexcsv(merged_dir: str | Path, final_dir: str | Path, split: str, idx_final: int):
    merged_rows = read_index_csv(merged_dir, split)
    final_rows = read_index_csv(final_dir, split)
    if merged_rows is None or final_rows is None:
        return idx_final, idx_final
    if idx_final < 0 or idx_final >= len(final_rows):
        return idx_final, idx_final
    key = _key_from_row(final_rows[idx_final])
    if key is None:
        return idx_final, idx_final
    for j, r in enumerate(merged_rows):
        if _key_from_row(r) == key:
            return j, idx_final
    return idx_final, idx_final


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", required=True)
    parser.add_argument("--merged_200k_dir", required=True)
    parser.add_argument("--final_8192_dir", required=True)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--ext", default="jpg", choices=["jpg", "png"])

    # render 200k
    parser.add_argument("--max_points_200k", type=int, default=80000, help="Subsample SOLO para render (02/03/04).")
    parser.add_argument("--overlay_point_size", type=float, default=3.0)
    parser.add_argument("--overlay_point_opacity", type=float, default=0.85)
    parser.add_argument("--overlay_mesh_opacity", type=float, default=0.35)
    parser.add_argument("--overlay_mesh_line_width", type=float, default=0.5)

    # raw fallback
    parser.add_argument("--raw_recon_max_points", type=int, default=60000)

    # Cámara
    parser.add_argument("--dist_mult_cloud", type=float, default=2.8, help="Distancia (multiplicador) para 03-08")
    parser.add_argument("--dist_mult_raw", type=float, default=3.2, help="Distancia (multiplicador) para 01-02")

    # augment params
    parser.add_argument("--alpha_deg", type=float, default=15.0)
    parser.add_argument("--smin", type=float, default=0.95)
    parser.add_argument("--smax", type=float, default=1.05)
    parser.add_argument("--sigma", type=float, default=0.005)
    parser.add_argument("--clip", type=float, default=0.02)
    parser.add_argument("--p_drop", type=float, default=0.05)
    parser.add_argument("--seed_aug", type=int, default=42)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = args.ext.strip(".").lower()

    # Load datasets
    X200k_all, Y200k_all = load_npz_pair(args.merged_200k_dir, args.split)
    X8192_all, Y8192_all = load_npz_pair(args.final_8192_dir, args.split)

    # Align indices
    idx_merged, idx_final = align_idx_by_indexcsv(args.merged_200k_dir, args.final_8192_dir, args.split, args.idx)
    X200k, Y200k = X200k_all[idx_merged], Y200k_all[idx_merged]
    X8192, Y8192 = X8192_all[idx_final], Y8192_all[idx_final]

    merged_rows = read_index_csv(args.merged_200k_dir, args.split)
    if merged_rows is None:
        raise FileNotFoundError(f"No existe index_{args.split}.csv en {args.merged_200k_dir}")

    row = merged_rows[idx_merged]
    sample_name = (row.get("sample_name") or "").strip()
    jaw = (row.get("jaw") or "").strip()

    print(f"✔ align: idx_final={args.idx} -> idx_merged={idx_merged}, idx_final={idx_final}")
    print(f"✔ sample_name={sample_name} jaw={jaw}")

    # Base orientation from 8192
    base_dir, viewup = compute_base_orientation_from_8192(X8192)
    print("✔ base_dir:", base_dir, "viewup:", viewup)

    # Raw OBJ
    obj_path = find_raw_upper_obj(Path(args.raw_root), sample_name)
    if obj_path is None:
        raise FileNotFoundError(f"No se encontró OBJ raw upper para sample_name={sample_name}")
    print("✔ raw_obj:", obj_path)

    mesh = load_raw_mesh_from_obj(obj_path)
    if mesh is None or not has_faces(mesh):
        print("⚠ OBJ no entregó caras válidas. Reconstruyendo superficie desde puntos 200k (solo para visual)...")
        mesh, alpha, used = build_surface_from_points(X200k, max_points=args.raw_recon_max_points, seed=123)
        print(f"✔ superficie reconstruida: cells={mesh.n_cells}, points={mesh.n_points}, alpha={alpha:.4f}, used_points={used}")

    # 01
    render_raw_solid(
        mesh,
        out_dir / f"01_raw_mesh_solid.{ext}",
        title=f"Malla raw (superficie sólida) — {sample_name} (upper)",
        base_dir=base_dir, viewup=viewup,
        dist_mult_raw=args.dist_mult_raw
    )

    # 02 (FIX: wireframe + puntos visibles)
    render_raw_wire_plus_points(
        mesh,
        X200k,
        out_dir / f"02_raw_mesh_plus_200k.{ext}",
        title="Malla raw (wireframe) + muestreo superficial (200k) — overlay visible",
        base_dir=base_dir, viewup=viewup,
        dist_mult_raw=args.dist_mult_raw,
        max_points=args.max_points_200k,
        point_size=args.overlay_point_size,
        point_opacity=args.overlay_point_opacity,
        mesh_opacity=args.overlay_mesh_opacity,
        mesh_line_width=args.overlay_mesh_line_width
    )

    # 03
    render_cloud_labeled(
        X200k, Y200k,
        out_dir / f"03_sampled_200k_labeled.{ext}",
        title=f"Muestreo superficial global (N=200k) — split={args.split}, idx={args.idx}",
        base_dir=base_dir, viewup=viewup,
        dist_mult_cloud=args.dist_mult_cloud,
        max_points=args.max_points_200k,
        point_size=3.0
    )

    # 04
    X200k_norm = normalize_unit_sphere(X200k)
    render_cloud_labeled(
        X200k_norm, Y200k,
        out_dir / f"04_normalized_200k_labeled.{ext}",
        title="Normalización a esfera unitaria (N=200k)",
        base_dir=base_dir, viewup=viewup,
        dist_mult_cloud=args.dist_mult_cloud,
        max_points=args.max_points_200k,
        point_size=3.0
    )

    # 05
    render_cloud_labeled(
        X8192, Y8192,
        out_dir / f"05_subsampled_8192_labeled.{ext}",
        title="Submuestreo final (8192) + control de fondo/cobertura",
        base_dir=base_dir, viewup=viewup,
        dist_mult_cloud=args.dist_mult_cloud,
        max_points=8192,
        point_size=7.0
    )

    # 06-08
    Xa1, Ya1, meta1 = augment_8192(
        X8192, Y8192,
        alpha_deg=args.alpha_deg, smin=args.smin, smax=args.smax,
        sigma=args.sigma, clip=args.clip, p_drop=args.p_drop,
        seed=args.seed_aug + 1
    )
    Xa2, Ya2, meta2 = augment_8192(
        X8192, Y8192,
        alpha_deg=args.alpha_deg, smin=args.smin, smax=args.smax,
        sigma=args.sigma, clip=args.clip, p_drop=args.p_drop,
        seed=args.seed_aug + 2
    )

    render_cloud_labeled(
        Xa1, Ya1,
        out_dir / f"06_aug1_8192_labeled.{ext}",
        title=f"Aumentación #1 — θ={meta1['theta_deg']:.1f}°, s={meta1['scale']:.3f}, kept={meta1['kept_points']}",
        base_dir=base_dir, viewup=viewup,
        dist_mult_cloud=args.dist_mult_cloud,
        max_points=8192,
        point_size=7.0
    )

    render_cloud_labeled(
        Xa2, Ya2,
        out_dir / f"07_aug2_8192_labeled.{ext}",
        title=f"Aumentación #2 — θ={meta2['theta_deg']:.1f}°, s={meta2['scale']:.3f}, kept={meta2['kept_points']}",
        base_dir=base_dir, viewup=viewup,
        dist_mult_cloud=args.dist_mult_cloud,
        max_points=8192,
        point_size=7.0
    )

    render_overlay_displacement(
        X_orig=X8192,
        X_aug=Xa1,
        out_path=out_dir / f"08_overlay_orig_vs_aug1_lines.{ext}",
        title="Overlay: original (gris) vs aumentado (negro) + líneas de desplazamiento",
        base_dir=base_dir, viewup=viewup,
        dist_mult_cloud=args.dist_mult_cloud,
        n_lines=300,
        max_points=8192,
        point_size=7.0,
        seed=123
    )

    print("\nListo ✅ Figuras guardadas en:", str(out_dir.resolve()))
    for name in [
        f"01_raw_mesh_solid.{ext}",
        f"02_raw_mesh_plus_200k.{ext}",
        f"03_sampled_200k_labeled.{ext}",
        f"04_normalized_200k_labeled.{ext}",
        f"05_subsampled_8192_labeled.{ext}",
        f"06_aug1_8192_labeled.{ext}",
        f"07_aug2_8192_labeled.{ext}",
        f"08_overlay_orig_vs_aug1_lines.{ext}",
    ]:
        print(" -", name)


if __name__ == "__main__":
    main()
