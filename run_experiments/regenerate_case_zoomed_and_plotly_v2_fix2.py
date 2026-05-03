#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
regenerate_case_zoomed_and_plotly_v2.py

V2 — Post-procesa una carpeta de caso generada por build_best_model_comparison_cases_v1/v2.py:
1) Regenera PNGs con zoom para reducir espacio en blanco.
2) Crea HTML interactivos con Plotly.
3) Usa pred_labels.npy por modelo si existen.
"""

import argparse, json
from pathlib import Path
from typing import Optional
import numpy as np
import pyvista as pv
import trimesh

try:
    import plotly.graph_objects as go
except Exception:
    go = None

# Nota:
# Esta fix2 usa plotly.graph_objects con leyendas por clase porque en el entorno
# actual pandas/plotly.express falla al construir DataFrames con NumPy.
# El resultado sigue siendo HTML interactivo Plotly con leyendas.
px = None
pd = None


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_json(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


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


def load_raw_mesh_trimesh(raw_mesh_path: Path) -> Optional[trimesh.Trimesh]:
    if raw_mesh_path is None or not raw_mesh_path.exists():
        return None
    try:
        loaded = trimesh.load(str(raw_mesh_path), force='mesh', process=False)
    except Exception:
        try:
            loaded = trimesh.load(str(raw_mesh_path), force='mesh', process=True)
        except Exception:
            return None
    if isinstance(loaded, trimesh.Scene):
        geos = list(loaded.dump().geometry.values())
        if not geos:
            return None
        return trimesh.util.concatenate(geos)
    return loaded if isinstance(loaded, trimesh.Trimesh) else None


def has_faces(mesh) -> bool:
    if mesh is None:
        return False
    if hasattr(mesh, 'n_faces_strict'):
        try: return int(mesh.n_faces_strict) > 0
        except Exception: pass
    if hasattr(mesh, 'n_cells'):
        try: return int(mesh.n_cells) > 0
        except Exception: pass
    return False


def plotter_base(width: int, height: int):
    pv.global_theme.multi_samples = 0
    pl = pv.Plotter(off_screen=True, window_size=(int(width), int(height)))
    pl.set_background('white')
    try: pl.enable_lightkit()
    except Exception: pass
    try: pl.enable_shadows()
    except Exception: pass
    return pl


def bounds_diag(bounds) -> float:
    b = bounds
    return float(np.sqrt((b[1]-b[0])**2 + (b[3]-b[2])**2 + (b[5]-b[4])**2) + 1e-9)


def bounds_center(bounds):
    b = bounds
    return (0.5*(b[0]+b[1]), 0.5*(b[2]+b[3]), 0.5*(b[4]+b[5]))


def compute_base_orientation_from_cloud(X: np.ndarray):
    d = np.array([1.0, 1.0, 0.65], dtype=np.float64)
    d /= np.linalg.norm(d) + 1e-12
    return d, (0.0, 0.0, 1.0)


def apply_camera(pl: pv.Plotter, bounds, base_dir, viewup, dist_mult: float):
    c = np.array(bounds_center(bounds), dtype=np.float64)
    diag = bounds_diag(bounds)
    pos = (c + np.asarray(base_dir) * float(dist_mult) * diag).tolist()
    pl.camera.focal_point = c.tolist()
    pl.camera.position = pos
    pl.camera.up = viewup
    try: pl.camera.zoom(1.20)
    except Exception: pass
    try: pl.reset_camera_clipping_range()
    except Exception: pass


def label_colors_uint8(Y: np.ndarray) -> np.ndarray:
    import matplotlib.cm as cm
    cmap = cm.get_cmap('tab20', 20)
    return (cmap(np.asarray(Y).reshape(-1).astype(np.int64) % 20)[:, :3] * 255).astype(np.uint8)


def render_raw(mesh: pv.PolyData, out_path: Path, title: str, base_dir, viewup, width, height, zoom):
    pl = plotter_base(width, height)
    if has_faces(mesh):
        pl.add_mesh(mesh, color='lightgray', opacity=1.0, smooth_shading=True, lighting=True,
                    show_edges=False, ambient=0.22, diffuse=0.85, specular=0.45, specular_power=35)
    else:
        pl.add_points(mesh, color=(160,160,160), point_size=3, opacity=0.9)
    apply_camera(pl, mesh.bounds, base_dir, viewup, dist_mult=max(0.55, 3.0 / float(zoom)))
    pl.add_text(title, position='upper_left', font_size=12, color='black')
    pl.show(screenshot=str(out_path), auto_close=True)


def render_cloud(X, Y, out_path: Path, title: str, base_dir, viewup, width, height, zoom, point_size, max_points):
    X = np.asarray(X, np.float32); Y = np.asarray(Y).reshape(-1).astype(np.int64)
    if X.shape[0] > max_points:
        rng = np.random.default_rng(123)
        idx = rng.choice(X.shape[0], size=max_points, replace=False)
        Xp, Yp = X[idx], Y[idx]
    else:
        Xp, Yp = X, Y
    cloud = pv.PolyData(Xp); cloud['rgb'] = label_colors_uint8(Yp)
    pl = plotter_base(width, height)
    pl.add_points(cloud, scalars='rgb', rgb=True, render_points_as_spheres=True, point_size=point_size, opacity=1.0)
    apply_camera(pl, cloud.bounds, base_dir, viewup, dist_mult=max(0.55, 2.6 / float(zoom)))
    pl.add_text(title, position='upper_left', font_size=12, color='black')
    pl.show(screenshot=str(out_path), auto_close=True)


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
        name="raw mesh",
        lighting=dict(ambient=0.35, diffuse=0.8, specular=0.35, roughness=0.35),
    ))
    fig.update_layout(**make_plotly_layout(title))
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    return True


def write_plotly_pointcloud(xyz, labels, out_html, title, name="points", point_size=3, max_points=8192):
    """
    Fix2: HTML interactivo Plotly robusto, con una traza por clase.
    Evita pandas/plotly.express porque el entorno actual lanza error de DataFrame.
    """
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
    """
    Raw mesh gris + puntos por clase con leyenda.
    """
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
        opacity=0.22,
        name="raw mesh",
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
    """
    Correcto/error con leyenda. Correcto gris tenue, error rojo.
    """
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
    """
    d21 vs resto/fondo con leyenda explícita.
    """
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


# ---------------------------------------------------------------------
# Aliases de compatibilidad con el main original del script v1.
# Mantienen el resto del script igual, pero apuntan a las funciones v2.
# ---------------------------------------------------------------------
def write_raw_mesh_html(mesh_tm, out_html, title):
    return write_plotly_raw_mesh(mesh_tm, out_html, title)


def write_points_html(xyz, labels, out_html, title, name="points", point_size=3, max_points=8192):
    return write_plotly_pointcloud(xyz, labels, out_html, title, name=name, point_size=point_size, max_points=max_points)


def write_raw_plus_points_html(mesh_tm, xyz, labels, out_html, title, point_size=3):
    return write_plotly_raw_plus_points(mesh_tm, xyz, labels, out_html, title, point_size=point_size)


def write_errors_html(xyz, err_mask, out_html, title):
    return write_plotly_errors(xyz, err_mask, out_html, title)


def write_binary_html(xyz, mask, out_html, title):
    return write_plotly_binary(xyz, mask, out_html, title)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case_dir', required=True)
    ap.add_argument('--zoom', type=float, default=2.0)
    ap.add_argument('--width', type=int, default=1800)
    ap.add_argument('--height', type=int, default=1300)
    ap.add_argument('--max_points_200k', type=int, default=80000)
    ap.add_argument('--d21_class', type=int, default=8)
    args = ap.parse_args()

    case_dir = Path(args.case_dir).resolve()
    meta = load_json(case_dir / 'meta.json')
    z = np.load(case_dir / 'common' / 'bundle_case_data.npz')
    xyz200k, y200k = z['xyz_200k'], z['y_200k']
    xyz8192, y8192 = z['xyz_8192'], z['y_8192']
    sample = meta.get('sample_name', case_dir.name); jaw = meta.get('jaw', 'upper')
    raw_path = Path(meta.get('raw_mesh_path', '')) if meta.get('raw_mesh_path','') else None
    tm = load_raw_mesh_trimesh(raw_path) if raw_path else None
    pvmesh = trimesh_to_pyvista_poly(tm) if tm is not None else None
    base_dir, viewup = compute_base_orientation_from_cloud(xyz8192)

    zoom_dir = ensure_dir(case_dir / 'common_zoomed')
    plotly_dir = ensure_dir(case_dir / 'plotly')

    if pvmesh is not None:
        render_raw(pvmesh, zoom_dir/'01_raw_mesh_zoom.png', f'Raw mesh — {sample} ({jaw})', base_dir, viewup, args.width, args.height, args.zoom)
    render_cloud(xyz200k, y200k, zoom_dir/'02_sampled_200k_labeled_zoom.png', f'200k labeled — {sample} ({jaw})', base_dir, viewup, args.width, args.height, args.zoom, 3.2, args.max_points_200k)
    render_cloud(xyz8192, y8192, zoom_dir/'03_final_8192_labeled_zoom.png', f'8192 labeled — {sample} ({jaw})', base_dir, viewup, args.width, args.height, args.zoom, 9.0, 8192)

    status = {}
    if tm is not None:
        status['raw_mesh'] = write_raw_mesh_html(tm, plotly_dir/'raw_mesh.html', f'Raw mesh — {sample} ({jaw})')
        status['raw_plus_gt_8192'] = write_raw_plus_points_html(tm, xyz8192, y8192, plotly_dir/'raw_plus_gt_8192.html', f'Raw + GT 8192 — {sample} ({jaw})')
    status['gt_8192'] = write_points_html(xyz8192, y8192, plotly_dir/'gt_8192.html', f'GT 8192 — {sample} ({jaw})')

    for model in ['pointnet','pointnetpp','dgcnn','pointnettransformer']:
        pred_path = case_dir / model / 'pred_labels.npy'
        if not pred_path.exists():
            continue
        pred = np.load(pred_path).reshape(-1).astype(np.int64)
        if pred.shape[0] != xyz8192.shape[0]:
            print(f'[WARN] {model}: pred shape incompatible')
            continue
        write_points_html(xyz8192, pred, plotly_dir/f'{model}_pred_8192.html', f'{model} pred 8192 — {sample} ({jaw})')
        write_binary_html(xyz8192, pred == int(args.d21_class), plotly_dir/f'{model}_pred_d21.html', f'{model} pred d21 — {sample} ({jaw})')
        write_errors_html(xyz8192, pred != y8192, plotly_dir/f'{model}_errors_all.html', f'{model} errores generales — {sample} ({jaw})')
        write_errors_html(xyz8192, (pred == int(args.d21_class)) != (y8192 == int(args.d21_class)), plotly_dir/f'{model}_errors_d21.html', f'{model} errores d21 — {sample} ({jaw})')

    save_json({'case_dir': str(case_dir), 'zoom': args.zoom, 'common_zoomed_dir': str(zoom_dir), 'plotly_dir': str(plotly_dir), 'plotly_available': go is not None, 'status': status}, case_dir/'interactive_views_meta.json')
    print(f'[OK] PNGs zoomed: {zoom_dir}')
    print(f'[OK] Plotly HTML : {plotly_dir}')

if __name__ == '__main__':
    main()
