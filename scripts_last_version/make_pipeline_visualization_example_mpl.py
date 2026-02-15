#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_pipeline_visualization_example_mpl.py

Pipeline visualization (Matplotlib Agg, 100% headless, NO PyVista/VTK)
Genera:
01: "malla raw sólida" (triangulada decimada si posible)
02: malla + overlay puntos 200k
03: 200k labeled
04: 200k normalized labeled
05: 8192 labeled
06: augment #1
07: augment #2
08: overlay orig vs aug1 + líneas

Uso:
python /home/htaucare/Tesis_Amaro/scripts_last_version/make_pipeline_visualization_example_mpl.py \
  --raw_obj /home/htaucare/Tesis_Amaro/data/Teeth_3ds/raw/data_part_7/upper/SN7PVUZ9/SN7PVUZ9_upper.obj \
  --merged_200k_dir /home/htaucare/Tesis_Amaro/data/Teeth_3ds/merged_200000_safe_excl_wisdom_upper_only \
  --final_8192_dir /home/htaucare/Tesis_Amaro/data/Teeth_3ds/fixed_split/8192/upper_only_surf_global_excl_wisdom_seed42_aug2 \
  --split val --idx 0 \
  --out_dir /home/htaucare/Tesis_Amaro/figures/pipeline_example_mpl \
  --ext jpg
"""

import argparse
import csv
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.cm as cm

import trimesh


# -----------------------------
# Utils: normalization + augment
# -----------------------------

def compute_center_radius(X: np.ndarray):
    c = X.mean(axis=0, keepdims=True)
    Xc = X - c
    r = np.max(np.linalg.norm(Xc, axis=1)) + 1e-12
    return c.reshape(3,), float(r)

def normalize_with_cr(X: np.ndarray, c: np.ndarray, r: float):
    return (X - c.reshape(1, 3)) / r

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


# -----------------------------
# Alignment: merged vs final idx
# -----------------------------

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
        return idx_final, idx_final, None
    if idx_final < 0 or idx_final >= len(final_rows):
        return idx_final, idx_final, None
    key = _key_from_row(final_rows[idx_final])
    if key is None:
        return idx_final, idx_final, None
    for j, r in enumerate(merged_rows):
        if _key_from_row(r) == key:
            return j, idx_final, final_rows[idx_final]
    return idx_final, idx_final, final_rows[idx_final]


# -----------------------------
# View: PCA frame from X8192
# -----------------------------

def pca_frame(X: np.ndarray):
    """
    Retorna matriz R (3x3) que rota puntos a frame PCA (componentes ordenadas).
    """
    Xc = X - X.mean(0, keepdims=True)
    C = (Xc.T @ Xc) / max(1, Xc.shape[0])
    w, V = np.linalg.eigh(C)     # asc
    V = V[:, np.argsort(w)[::-1]]  # desc

    # asegurar mano derecha (det=+1)
    if np.linalg.det(V) < 0:
        V[:, -1] *= -1
    return V  # columnas = ejes

def apply_frame(X: np.ndarray, V: np.ndarray):
    Xc = X - X.mean(0, keepdims=True)
    return Xc @ V  # rotación


# -----------------------------
# Rendering helpers (matplotlib)
# -----------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def set_equal_aspect(ax, X):
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    center = (mins + maxs) / 2
    span = (maxs - mins).max()
    r = span / 2
    ax.set_xlim(center[0] - r, center[0] + r)
    ax.set_ylim(center[1] - r, center[1] + r)
    ax.set_zlim(center[2] - r, center[2] + r)

def colors_tab20_uint8(Y: np.ndarray):
    cmap = cm.get_cmap("tab20", 20)
    rgb = cmap((Y.astype(np.int64) % 20))[:, :3]
    return (rgb * 255).astype(np.uint8)

def scatter_labeled(ax, X, Y, max_points=25000, s=2.0):
    rng = np.random.default_rng(123)
    if X.shape[0] > max_points:
        idx = rng.choice(X.shape[0], size=max_points, replace=False)
        Xp, Yp = X[idx], Y[idx]
    else:
        Xp, Yp = X, Y
    rgb = colors_tab20_uint8(Yp) / 255.0
    ax.scatter(Xp[:,0], Xp[:,1], Xp[:,2], c=rgb, s=s, depthshade=False, linewidths=0)

def scatter_gray(ax, X, max_points=60000, s=1.0, alpha=0.25, color=(0.2,0.2,0.2)):
    rng = np.random.default_rng(123)
    if X.shape[0] > max_points:
        idx = rng.choice(X.shape[0], size=max_points, replace=False)
        Xp = X[idx]
    else:
        Xp = X
    ax.scatter(Xp[:,0], Xp[:,1], Xp[:,2], c=[color], s=s, alpha=alpha, depthshade=False, linewidths=0)

def render_mesh_solid(ax, V, F, max_faces=60000, color=(0.85,0.85,0.85), edge_alpha=0.0):
    """
    V: (n,3) vertices
    F: (m,3) faces
    """
    rng = np.random.default_rng(123)
    if F.shape[0] > max_faces:
        idx = rng.choice(F.shape[0], size=max_faces, replace=False)
        Fp = F[idx]
    else:
        Fp = F

    tris = V[Fp]  # (m,3,3)
    poly = Poly3DCollection(tris, facecolor=color, edgecolor=(0,0,0,edge_alpha))
    poly.set_alpha(1.0)
    ax.add_collection3d(poly)

def fig3d(title, elev=18, azim=-65, figsize=(10,7), dpi=220):
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title, pad=18)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    return fig, ax

def save_fig(fig, out_path: Path):
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# -----------------------------
# Load OBJ robustly (trimesh)
# -----------------------------

def load_obj_as_trimesh(obj_path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(str(obj_path), force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        geos = list(loaded.dump().geometry.values())
        if not geos:
            raise RuntimeError("OBJ scene sin geometrías.")
        mesh = trimesh.util.concatenate(geos)
    else:
        mesh = loaded
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError("OBJ no se pudo convertir a Trimesh.")
    try:
        mesh.remove_degenerate_faces()
        mesh.remove_duplicate_faces()
    except Exception:
        pass
    return mesh


# -----------------------------
# MAIN
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_obj", required=True, help="Ruta directa al .obj raw (upper).")
    ap.add_argument("--merged_200k_dir", required=True)
    ap.add_argument("--final_8192_dir", required=True)
    ap.add_argument("--split", default="val", choices=["train","val","test"])
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ext", default="jpg", choices=["jpg","png"])

    ap.add_argument("--max_points_200k", type=int, default=25000)
    ap.add_argument("--max_faces_raw", type=int, default=60000)

    # view (paper-like)
    ap.add_argument("--elev", type=float, default=18.0)
    ap.add_argument("--azim", type=float, default=-65.0)

    args = ap.parse_args()
    out_dir = Path(args.out_dir); ensure_dir(out_dir)
    ext = args.ext

    # load datasets
    X200k_all, Y200k_all = load_npz_pair(args.merged_200k_dir, args.split)
    X8192_all, Y8192_all = load_npz_pair(args.final_8192_dir, args.split)

    idx_merged, idx_final, row_final = align_idx_by_indexcsv(args.merged_200k_dir, args.final_8192_dir, args.split, args.idx)

    X200k, Y200k = X200k_all[idx_merged], Y200k_all[idx_merged]
    X8192, Y8192 = X8192_all[idx_final], Y8192_all[idx_final]

    sample_name = None
    if row_final is not None:
        sample_name = (row_final.get("sample_name") or "").strip()
    if not sample_name:
        # fallback: leer del merged index
        rowsm = read_index_csv(args.merged_200k_dir, args.split)
        if rowsm:
            sample_name = (rowsm[idx_merged].get("sample_name") or "").strip()
    if not sample_name:
        sample_name = f"idx{args.idx}"

    print(f"✔ align: idx_final={args.idx} -> idx_merged={idx_merged}, idx_final={idx_final}")
    print(f"✔ sample_name={sample_name}")
    print(f"✔ raw_obj={args.raw_obj}")

    # normalización usando 200k (para que raw mesh y puntos queden en mismo frame)
    c200k, r200k = compute_center_radius(X200k)
    X200k_norm = normalize_with_cr(X200k, c200k, r200k)

    # frame PCA desde 8192 para orientar TODAS las figuras
    Vpca = pca_frame(X8192)
    X8192_view = apply_frame(X8192, Vpca)
    X200k_view = apply_frame(X200k_norm, Vpca)

    # cargar raw mesh + normalizar igual que 200k + rotar a frame PCA
    mesh_tm = load_obj_as_trimesh(Path(args.raw_obj))
    Vr = normalize_with_cr(mesh_tm.vertices.astype(np.float32), c200k, r200k)
    Vr = apply_frame(Vr, Vpca)
    Fr = mesh_tm.faces.astype(np.int64) if mesh_tm.faces is not None else None

    # 01 raw mesh solid
    fig, ax = fig3d(f"Malla raw (superficie sólida) — {sample_name} (upper)", elev=args.elev, azim=args.azim)
    if Fr is not None and Fr.size > 0:
        render_mesh_solid(ax, Vr, Fr, max_faces=args.max_faces_raw)
        set_equal_aspect(ax, Vr)
    else:
        scatter_gray(ax, Vr, max_points=80000, s=0.6, alpha=0.35)
        set_equal_aspect(ax, Vr)
    save_fig(fig, out_dir / f"01_raw_mesh_solid.{ext}")

    # 02 raw + 200k overlay
    fig, ax = fig3d("Malla raw (superficie) + muestreo superficial (200k)", elev=args.elev, azim=args.azim)
    if Fr is not None and Fr.size > 0:
        render_mesh_solid(ax, Vr, Fr, max_faces=args.max_faces_raw)
    scatter_gray(ax, X200k_view, max_points=args.max_points_200k, s=1.2, alpha=0.20, color=(0.12,0.47,0.71))
    set_equal_aspect(ax, Vr if Vr.shape[0] > 0 else X200k_view)
    save_fig(fig, out_dir / f"02_raw_mesh_plus_200k.{ext}")

    # 03 200k labeled
    fig, ax = fig3d(f"Muestreo superficial global (N=200k) — split={args.split}, idx={args.idx}", elev=args.elev, azim=args.azim)
    scatter_labeled(ax, X200k_view, Y200k, max_points=args.max_points_200k, s=2.2)
    set_equal_aspect(ax, X200k_view)
    save_fig(fig, out_dir / f"03_sampled_200k_labeled.{ext}")

    # 04 normalized (igual en este script; lo mantenemos como paso)
    fig, ax = fig3d("Normalización a esfera unitaria (N=200k)", elev=args.elev, azim=args.azim)
    scatter_labeled(ax, X200k_view, Y200k, max_points=args.max_points_200k, s=2.2)
    set_equal_aspect(ax, X200k_view)
    save_fig(fig, out_dir / f"04_normalized_200k_labeled.{ext}")

    # 05 8192 labeled
    fig, ax = fig3d("Submuestreo final (8192) + control de fondo/cobertura", elev=args.elev, azim=args.azim)
    scatter_labeled(ax, X8192_view, Y8192, max_points=8192, s=9.0)
    set_equal_aspect(ax, X8192_view)
    save_fig(fig, out_dir / f"05_subsampled_8192_labeled.{ext}")

    # 06-07 aug
    Xa1, Ya1, meta1 = augment_8192(X8192, Y8192, seed=43)
    Xa2, Ya2, meta2 = augment_8192(X8192, Y8192, seed=44)
    Xa1_view = apply_frame(Xa1, Vpca)
    Xa2_view = apply_frame(Xa2, Vpca)

    fig, ax = fig3d(f"Aumentación #1 — θ={meta1['theta_deg']:.1f}°, s={meta1['scale']:.3f}, kept={meta1['kept_points']}", elev=args.elev, azim=args.azim)
    scatter_labeled(ax, Xa1_view, Ya1, max_points=8192, s=9.0)
    set_equal_aspect(ax, Xa1_view)
    save_fig(fig, out_dir / f"06_aug1_8192_labeled.{ext}")

    fig, ax = fig3d(f"Aumentación #2 — θ={meta2['theta_deg']:.1f}°, s={meta2['scale']:.3f}, kept={meta2['kept_points']}", elev=args.elev, azim=args.azim)
    scatter_labeled(ax, Xa2_view, Ya2, max_points=8192, s=9.0)
    set_equal_aspect(ax, Xa2_view)
    save_fig(fig, out_dir / f"07_aug2_8192_labeled.{ext}")

    # 08 overlay + lines
    fig, ax = fig3d("Overlay: original (gris) vs aumentado (negro) + líneas de desplazamiento", elev=args.elev, azim=args.azim)
    # orig gray
    scatter_gray(ax, X8192_view, max_points=8192, s=10.0, alpha=0.35, color=(0.6,0.6,0.6))
    # aug black
    scatter_gray(ax, Xa1_view, max_points=8192, s=10.0, alpha=0.65, color=(0.1,0.1,0.1))

    rng = np.random.default_rng(123)
    n_pair = min(X8192_view.shape[0], Xa1_view.shape[0])
    n_lines = min(300, n_pair)
    idx = rng.choice(n_pair, size=n_lines, replace=False)
    for i in idx:
        p = X8192_view[i]
        q = Xa1_view[i]
        ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], linewidth=0.6, alpha=0.35, color=(0.15,0.15,0.15))

    set_equal_aspect(ax, X8192_view)
    save_fig(fig, out_dir / f"08_overlay_orig_vs_aug1_lines.{ext}")

    print("\nListo ✅ Figuras guardadas en:", str(out_dir.resolve()))


if __name__ == "__main__":
    main()
