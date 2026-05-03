#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_paper_figure_models_selectable_v3_aligned_plotly.py

Figura tipo paper/tesis con selección dinámica de modelos.

Antes de usar la versión full-pro con renders homogéneos, conviene correr:
  export_predictions_from_best_models_v1.py

porque este script usa, si existen:
  case_dir/<modelo>/pred_labels.npy

Si no existen, usa fallback con PNGs existentes:
  04_inference_all.png
  05_inference_errors.png
  06_inference_d21.png

Ejemplo 1 modelo:
python3 -u make_paper_figure_models_selectable_v3_aligned_plotly.py \
  --case_dir /home/htaucare/Tesis_Amaro/case_comparisons/best4_test_v2/row_005_6BWQC0CT_upper \
  --out_png /home/htaucare/Tesis_Amaro/case_comparisons/best4_test_v2/row_005_6BWQC0CT_upper/paper_pointnetpp_only.png \
  --models pointnetpp \
  --d21_class 8 \
  --zoom 1.55 \
  --force

Ejemplo 4 modelos:
python3 -u make_paper_figure_models_selectable_v3_aligned_plotly.py \
  --case_dir /home/htaucare/Tesis_Amaro/case_comparisons/best4_test_v2/row_005_6BWQC0CT_upper \
  --out_png /home/htaucare/Tesis_Amaro/case_comparisons/best4_test_v2/row_005_6BWQC0CT_upper/paper_best4_v2.png \
  --models pointnet,pointnetpp,dgcnn,pointnettransformer \
  --d21_class 8 \
  --zoom 1.55 \
  --force
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import pyvista as pv
import trimesh

try:
    import plotly.graph_objects as go
except Exception:
    go = None


IDX2FDI = {
    0: "background",
    1: "d11",
    2: "d12",
    3: "d13",
    4: "d14",
    5: "d15",
    6: "d16",
    7: "d17",
    8: "d21",
    9: "d22",
    10: "d23",
    11: "d24",
    12: "d25",
    13: "d26",
    14: "d27",
}

CLASS_COLORS = {
    0:  (70, 70, 70),
    1:  (76, 175, 80),
    2:  (104, 196, 222),
    3:  (70, 155, 205),
    4:  (116, 102, 180),
    5:  (181, 95, 170),
    6:  (224, 82, 140),
    7:  (238, 110, 110),
    8:  (226, 126, 108),
    9:  (238, 151, 78),
    10: (244, 198, 75),
    11: (205, 223, 88),
    12: (118, 203, 118),
    13: (70, 190, 195),
    14: (190, 205, 215),
}

ALL_MODEL_INFO = {
    "gt": ("GT", (235, 235, 235)),
    "pointnet": ("PointNet", (80, 150, 255)),
    "pointnetpp": ("PointNet++", (80, 210, 100)),
    "dgcnn": ("DGCNN", (255, 150, 45)),
    "pointnettransformer": ("Point Transformer", (255, 190, 40)),
}

MODEL_ALIASES = {
    "pointnet": "pointnet",
    "pn": "pointnet",
    "pointnetclassic": "pointnet",
    "pointnet_classic": "pointnet",
    "pointnetpp": "pointnetpp",
    "pointnet++": "pointnetpp",
    "pnpp": "pointnetpp",
    "dgcnn": "dgcnn",
    "pointnettransformer": "pointnettransformer",
    "transformer": "pointnettransformer",
    "pnt": "pointnettransformer",
}


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: Path):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def parse_models(models_arg: str) -> List[str]:
    raw = [x.strip().lower() for x in str(models_arg).split(",") if x.strip()]
    if not raw:
        raw = ["pointnet", "pointnetpp", "dgcnn", "pointnettransformer"]
    out = []
    for m in raw:
        key = MODEL_ALIASES.get(m, m)
        if key not in ("pointnet", "pointnetpp", "dgcnn", "pointnettransformer"):
            raise ValueError(f"Modelo no reconocido: {m}. Use pointnet, pointnetpp, dgcnn, pointnettransformer")
        if key not in out:
            out.append(key)
    return out


def find_font(size: int, bold: bool = False):
    candidates = []
    if bold:
        candidates += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        ]
    candidates += [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for c in candidates:
        if Path(c).exists():
            return ImageFont.truetype(c, size=size)
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font):
    bb = draw.textbbox((0, 0), text, font=font)
    return bb[2] - bb[0], bb[3] - bb[1]


def draw_centered_text(draw, box, text, font, fill):
    x0, y0, x1, y1 = box
    tw, th = text_size(draw, text, font)
    x = x0 + (x1 - x0 - tw) / 2
    y = y0 + (y1 - y0 - th) / 2
    draw.text((x, y), text, font=font, fill=fill)


def safe_open_image(path: Optional[Path], size: Tuple[int, int], bg=(0, 0, 0)) -> Image.Image:
    if path is not None and Path(path).exists():
        im = Image.open(path).convert("RGB")
        return im.resize(size, Image.LANCZOS)
    im = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(im)
    font = find_font(18, bold=False)
    msg = "Missing render"
    if path is not None:
        msg += f"\n{Path(path).name}"
    d.text((18, 18), msg, fill=(255, 80, 80), font=font)
    return im


def paste_fit(canvas: Image.Image, img: Image.Image, box):
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    iw, ih = img.size
    scale = min(bw / iw, bh / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    im = img.resize((nw, nh), Image.LANCZOS)
    x = x0 + (bw - nw) // 2
    y = y0 + (bh - nh) // 2
    canvas.paste(im, (x, y))


# ---------------- Mesh alignment ----------------

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
        return trimesh.util.concatenate(geos)
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    return None


def center_radius(points: np.ndarray) -> Tuple[np.ndarray, float]:
    pts = np.asarray(points, np.float64)
    c = pts.mean(axis=0)
    r = np.linalg.norm(pts - c[None, :], axis=1).max()
    return c, float(max(r, 1e-12))


def robust_center_extent(points: np.ndarray, q_low: float = 2.0, q_high: float = 98.0) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, np.float64)
    lo = np.percentile(pts, float(q_low), axis=0)
    hi = np.percentile(pts, float(q_high), axis=0)
    center = 0.5 * (lo + hi)
    extent = np.maximum(hi - lo, 1e-12)
    return center, extent


def align_trimesh_to_cloud(
    mesh_tm: trimesh.Trimesh,
    xyz_ref: np.ndarray,
    mode: str = "robust_bbox",
    q_low: float = 2.0,
    q_high: float = 98.0,
) -> trimesh.Trimesh:
    """
    Alinea la malla raw al espacio de la nube 8192.

    mode='robust_bbox' usa percentiles, no centroide/radio global.
    Esto es mejor para OBJ dentales porque la malla raw incluye zócalo/base
    y esa base puede desplazar el centroide respecto a la nube final.
    """
    V = np.asarray(mesh_tm.vertices, np.float64)
    X = np.asarray(xyz_ref, np.float64)

    mode = str(mode).lower().strip()

    if mode == "none":
        V_aligned = V.copy()

    elif mode == "centroid_radius":
        c_raw, r_raw = center_radius(V)
        c_ref, r_ref = center_radius(X)
        V_aligned = ((V - c_raw[None, :]) / r_raw) * r_ref + c_ref[None, :]

    elif mode == "robust_bbox_axis":
        c_raw, e_raw = robust_center_extent(V, q_low=q_low, q_high=q_high)
        c_ref, e_ref = robust_center_extent(X, q_low=q_low, q_high=q_high)
        V_aligned = ((V - c_raw[None, :]) / e_raw[None, :]) * e_ref[None, :] + c_ref[None, :]

    else:
        # robust_bbox: escala uniforme con mediana de razones por eje
        c_raw, e_raw = robust_center_extent(V, q_low=q_low, q_high=q_high)
        c_ref, e_ref = robust_center_extent(X, q_low=q_low, q_high=q_high)
        scale = float(np.median(e_ref / np.maximum(e_raw, 1e-12)))
        V_aligned = (V - c_raw[None, :]) * scale + c_ref[None, :]

    mesh2 = mesh_tm.copy()
    mesh2.vertices = V_aligned
    return mesh2


def trimesh_to_pyvista(mesh_tm: trimesh.Trimesh) -> pv.PolyData:
    V = np.asarray(mesh_tm.vertices, dtype=np.float32)
    F = np.asarray(mesh_tm.faces, dtype=np.int64)
    faces = np.hstack([np.full((F.shape[0], 1), 3, dtype=np.int64), F]).ravel()
    poly = pv.PolyData(V, faces)
    try:
        poly = poly.compute_normals(point_normals=True, cell_normals=False, auto_orient_normals=True)
    except Exception:
        pass
    return poly


# ---------------- PyVista render ----------------

def label_colors(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    out = np.zeros((labels.shape[0], 3), dtype=np.uint8)
    for cls, rgb in CLASS_COLORS.items():
        out[labels == cls] = np.array(rgb, dtype=np.uint8)
    return out


def binary_d21_colors(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask).reshape(-1).astype(bool)
    out = np.zeros((mask.shape[0], 3), dtype=np.uint8)
    out[~mask] = np.array((95, 95, 95), dtype=np.uint8)
    out[mask] = np.array((236, 105, 170), dtype=np.uint8)
    return out


def make_plotter(w: int, h: int, bg=(0, 0, 0)):
    pv.global_theme.multi_samples = 0
    pl = pv.Plotter(off_screen=True, window_size=(int(w), int(h)))
    pl.set_background(bg)
    try:
        pl.enable_lightkit()
    except Exception:
        pass
    return pl


def bounds_center(bounds):
    b = bounds
    return np.array([0.5 * (b[0] + b[1]), 0.5 * (b[2] + b[3]), 0.5 * (b[4] + b[5])], dtype=np.float64)


def bounds_diag(bounds):
    b = bounds
    return float(np.sqrt((b[1]-b[0])**2 + (b[3]-b[2])**2 + (b[5]-b[4])**2) + 1e-9)


def camera_dir():
    d = np.array([1.0, 1.0, 0.72], dtype=np.float64)
    d /= np.linalg.norm(d)
    return d


def apply_camera(pl: pv.Plotter, bounds, zoom: float):
    c = bounds_center(bounds)
    d = camera_dir()
    diag = bounds_diag(bounds)
    dist = max(0.45, 2.65 / float(zoom)) * diag
    pl.camera.position = (c + d * dist).tolist()
    pl.camera.focal_point = c.tolist()
    pl.camera.up = (0, 0, 1)
    try:
        pl.camera.zoom(1.10)
    except Exception:
        pass
    try:
        pl.reset_camera_clipping_range()
    except Exception:
        pass


def add_base_mesh(pl, mesh_pv: Optional[pv.PolyData], opacity=0.24, color=(145, 145, 145)):
    if mesh_pv is None:
        return
    pl.add_mesh(
        mesh_pv,
        color=color,
        opacity=float(opacity),
        smooth_shading=True,
        lighting=True,
        show_edges=False,
        ambient=0.30,
        diffuse=0.85,
        specular=0.28,
        specular_power=25,
    )


def render_all_classes(mesh_pv, xyz, labels, out_path, w, h, zoom, mesh_opacity=0.25, point_size=9.0):
    pl = make_plotter(w, h)
    add_base_mesh(pl, mesh_pv, opacity=mesh_opacity)
    cloud = pv.PolyData(np.asarray(xyz, np.float32))
    cloud["rgb"] = label_colors(labels)
    pl.add_points(cloud, scalars="rgb", rgb=True, render_points_as_spheres=True, point_size=float(point_size), opacity=0.98)
    bounds = mesh_pv.bounds if mesh_pv is not None else cloud.bounds
    apply_camera(pl, bounds, zoom)
    pl.show(screenshot=str(out_path), auto_close=True)


def render_errors_all(mesh_pv, xyz, pred, gt, out_path, w, h, zoom):
    err = np.asarray(pred).reshape(-1) != np.asarray(gt).reshape(-1)
    pl = make_plotter(w, h)
    add_base_mesh(pl, mesh_pv, opacity=0.34, color=(120, 120, 120))
    cloud_all = pv.PolyData(np.asarray(xyz, np.float32))
    cloud_all["rgb"] = np.tile(np.array([[105, 105, 105]], dtype=np.uint8), (xyz.shape[0], 1))
    pl.add_points(cloud_all, scalars="rgb", rgb=True, render_points_as_spheres=True, point_size=4.0, opacity=0.10)
    if err.any():
        cloud_e = pv.PolyData(np.asarray(xyz[err], np.float32))
        cloud_e["rgb"] = np.tile(np.array([[255, 35, 25]], dtype=np.uint8), (int(err.sum()), 1))
        pl.add_points(cloud_e, scalars="rgb", rgb=True, render_points_as_spheres=True, point_size=10.5, opacity=1.0)
    bounds = mesh_pv.bounds if mesh_pv is not None else cloud_all.bounds
    apply_camera(pl, bounds, zoom)
    pl.show(screenshot=str(out_path), auto_close=True)


def render_d21_binary(mesh_pv, xyz, labels, d21_class, out_path, w, h, zoom):
    mask = np.asarray(labels).reshape(-1).astype(np.int64) == int(d21_class)
    pl = make_plotter(w, h)
    add_base_mesh(pl, mesh_pv, opacity=0.34, color=(120, 120, 120))
    cloud = pv.PolyData(np.asarray(xyz, np.float32))
    cloud["rgb"] = binary_d21_colors(mask)
    pl.add_points(cloud, scalars="rgb", rgb=True, render_points_as_spheres=True, point_size=8.5, opacity=0.94)
    bounds = mesh_pv.bounds if mesh_pv is not None else cloud.bounds
    apply_camera(pl, bounds, zoom)
    pl.show(screenshot=str(out_path), auto_close=True)


def render_d21_errors(mesh_pv, xyz, pred, gt, d21_class, out_path, w, h, zoom):
    pred_d21 = np.asarray(pred).reshape(-1).astype(np.int64) == int(d21_class)
    gt_d21 = np.asarray(gt).reshape(-1).astype(np.int64) == int(d21_class)
    err = pred_d21 != gt_d21
    pl = make_plotter(w, h)
    add_base_mesh(pl, mesh_pv, opacity=0.36, color=(120, 120, 120))
    if gt_d21.any():
        cloud_gt = pv.PolyData(np.asarray(xyz[gt_d21], np.float32))
        cloud_gt["rgb"] = np.tile(np.array([[236, 105, 170]], dtype=np.uint8), (int(gt_d21.sum()), 1))
        pl.add_points(cloud_gt, scalars="rgb", rgb=True, render_points_as_spheres=True, point_size=8.5, opacity=0.55)
    if err.any():
        cloud_e = pv.PolyData(np.asarray(xyz[err], np.float32))
        cloud_e["rgb"] = np.tile(np.array([[255, 35, 25]], dtype=np.uint8), (int(err.sum()), 1))
        pl.add_points(cloud_e, scalars="rgb", rgb=True, render_points_as_spheres=True, point_size=11.5, opacity=1.0)
    bounds = mesh_pv.bounds if mesh_pv is not None else pv.PolyData(np.asarray(xyz, np.float32)).bounds
    apply_camera(pl, bounds, zoom)
    pl.show(screenshot=str(out_path), auto_close=True)


def make_legend_panel(width: int, height: int, mode: str = "all") -> Image.Image:
    im = Image.new("RGB", (width, height), (0, 0, 0))
    d = ImageDraw.Draw(im)
    font_title = find_font(22, bold=True)
    font = find_font(18, bold=False)
    font_small = find_font(16, bold=False)
    red = (255, 60, 55)
    white = (230, 230, 230)
    if mode == "all":
        d.text((22, 24), "Leyenda (clases)", fill=white, font=font_title)
        y = 72
        box = 22
        for cls in range(0, 15):
            rgb = CLASS_COLORS[cls]
            d.rectangle((24, y, 24 + box, y + box), fill=rgb)
            label = f"{cls} - {IDX2FDI.get(cls, 'clase')}"
            d.text((60, y - 1), label, fill=white, font=font)
            y += 31
        y += 16
        d.text((22, y), "Errores", fill=red, font=font_title)
        y += 34
        d.text((22, y), "Rojo = predicción", fill=white, font=font_small)
        y += 24
        d.text((22, y), "incorrecta (Pred ≠ GT)", fill=white, font=font_small)
    else:
        y = max(80, height // 4)
        d.text((22, y), "Leyenda (d21 vs fondo)", fill=white, font=font_title)
        y += 58
        d.rectangle((24, y, 48, y + 24), fill=(95, 95, 95))
        d.text((60, y - 1), "0 - fondo", fill=white, font=font)
        y += 42
        d.rectangle((24, y, 48, y + 24), fill=(236, 105, 170))
        d.text((60, y - 1), "8 - d21", fill=white, font=font)
        y += 66
        d.text((22, y), "Errores d21", fill=red, font=font_title)
        y += 34
        d.text((22, y), "Rojo = predicción", fill=white, font=font_small)
        y += 24
        d.text((22, y), "incorrecta (Pred ≠ GT)", fill=white, font=font_small)
    return im


def make_formula_bar(width: int, height: int) -> Image.Image:
    im = Image.new("RGB", (width, height), (0, 0, 0))
    d = ImageDraw.Draw(im)
    font = find_font(20, bold=False)
    font_b = find_font(20, bold=True)
    margin = max(60, int(width * 0.06))
    y0 = 12
    y1 = height - 12
    d.rounded_rectangle((margin, y0, width - margin, y1), radius=8, outline=(110, 110, 110), width=2)
    parts = [
        ("Malla raw semi-transparente", (235, 235, 235), font_b),
        (" + ", (235, 235, 235), font),
        ("Predicciones coloreadas", (235, 235, 235), font_b),
        (" + ", (235, 235, 235), font),
        ("Errores en rojo", (255, 70, 70), font_b),
        (" = ", (235, 235, 235), font),
        ("visualización para análisis detallado", (235, 235, 235), font_b),
    ]
    widths = [d.textbbox((0, 0), txt, font=fnt)[2] for txt, _, fnt in parts]
    total = sum(widths)
    x = max(margin + 20, (width - total) // 2)
    y = 28
    for (txt, col, fnt), ww in zip(parts, widths):
        d.text((x, y), txt, fill=col, font=fnt)
        x += ww
    return im


def auto_layout(n_selected_models: int, args):
    n_cols = 1 + n_selected_models
    if args.panel_w > 0 and args.panel_h > 0 and args.legend_w > 0:
        return args.panel_w, args.panel_h, args.legend_w
    if n_cols <= 2:
        return 460, 320, 280
    if n_cols == 3:
        return 380, 270, 270
    if n_cols == 4:
        return 330, 235, 260
    return 300, 210, 260


def compose_figure(render_paths, case_meta, selected_models, out_png, panel_w, panel_h, legend_w):
    model_cols = [("gt",) + ALL_MODEL_INFO["gt"]] + [(m,) + ALL_MODEL_INFO[m] for m in selected_models]
    n_model_cols = len(model_cols)
    title_h = 92
    block_title_h = 54
    col_title_h = 70
    row_gap = 18
    sep_h = 3
    formula_h = 70
    margin_x = 6
    block_h = block_title_h + col_title_h + panel_h + row_gap + 42 + panel_h
    fig_w = margin_x * 2 + n_model_cols * panel_w + legend_w
    fig_h = title_h + block_h + sep_h + block_h + formula_h
    canvas = Image.new("RGB", (fig_w, fig_h), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    font_title = find_font(30, bold=True)
    font_model = find_font(22, bold=True)
    font_small = find_font(19, bold=False)
    line_color = (80, 80, 80)
    sample = case_meta.get("sample_name", "")
    jaw = case_meta.get("jaw", "")
    row_i = case_meta.get("row_final", "")
    draw_centered_text(draw, (0, 8, fig_w, 38), f"test row={row_i} | sample={sample} | jaw={jaw}", font_small, (235, 235, 235))
    draw_centered_text(draw, (0, 42, fig_w, 88), "VISTA PRO: MALLA RAW COLOREADA CON INFERENCIA", font_title, (235, 235, 235))
    draw.line((0, title_h - 2, fig_w, title_h - 2), fill=line_color, width=3)

    def draw_block(y0: int, mode: str):
        if mode == "all":
            block_title = "TODAS LAS CLASES"
            top_key = "all"
            err_key = "err_all"
            row1_label = "todas las clases"
            row2_label = "Error (todas las clases)"
            legend = make_legend_panel(legend_w, block_h, "all")
        else:
            block_title = "ENFOQUE DIENTE 21 (CLASE 8) VS FONDO"
            top_key = "d21"
            err_key = "err_d21"
            row1_label = "d21 vs fondo"
            row2_label = "Error (diente 21 vs fondo)"
            legend = make_legend_panel(legend_w, block_h, "d21")
        draw_centered_text(draw, (0, y0, fig_w, y0 + block_title_h), block_title, font_title, (235, 235, 235))
        draw.line((0, y0 + block_title_h - 1, fig_w, y0 + block_title_h - 1), fill=line_color, width=2)
        y_titles = y0 + block_title_h
        y_row1 = y_titles + col_title_h
        y_row2_title = y_row1 + panel_h + row_gap
        y_row2 = y_row2_title + 42
        for ci, (key, name, color) in enumerate(model_cols):
            x0 = margin_x + ci * panel_w
            x1 = x0 + panel_w
            if key == "gt":
                draw_centered_text(draw, (x0, y_titles + 5, x1, y_titles + 35), "GT", font_model, color)
                sub = "(todas las clases)" if mode == "all" else "(diente 21 vs fondo)"
            else:
                draw_centered_text(draw, (x0, y_titles + 0, x1, y_titles + 32), name, font_model, color)
                sub = f"({row1_label})"
            draw_centered_text(draw, (x0, y_titles + 32, x1, y_titles + 65), sub, font_small, (235, 235, 235))
            draw_centered_text(draw, (x0, y_row2_title, x1, y_row2_title + 36), row2_label, font_small, (230, 230, 230))
            im_top = safe_open_image(render_paths.get(key, {}).get(top_key), (panel_w, panel_h))
            im_err = safe_open_image(render_paths.get(key, {}).get(err_key), (panel_w, panel_h))
            paste_fit(canvas, im_top, (x0, y_row1, x1, y_row1 + panel_h))
            paste_fit(canvas, im_err, (x0, y_row2, x1, y_row2 + panel_h))
        lx0 = margin_x + n_model_cols * panel_w
        canvas.paste(legend, (lx0, y0))
        draw.line((lx0, y0, lx0, y0 + block_h), fill=line_color, width=2)
        draw.line((0, y0 + block_h - 2, fig_w, y0 + block_h - 2), fill=line_color, width=2)

    block1_y = title_h
    draw_block(block1_y, "all")
    sep_y = block1_y + block_h
    draw.line((0, sep_y, fig_w, sep_y), fill=line_color, width=3)
    block2_y = sep_y + sep_h
    draw_block(block2_y, "d21")
    formula = make_formula_bar(fig_w, formula_h)
    canvas.paste(formula, (0, fig_h - formula_h))
    ensure_dir(out_png.parent)
    canvas.save(out_png)
    return out_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case_dir", required=True)
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--models", default="pointnet,pointnetpp,dgcnn,pointnettransformer")
    ap.add_argument("--d21_class", type=int, default=8)
    ap.add_argument("--zoom", type=float, default=1.55)
    ap.add_argument("--panel_w", type=int, default=-1)
    ap.add_argument("--panel_h", type=int, default=-1)
    ap.add_argument("--legend_w", type=int, default=-1)
    ap.add_argument("--render_w", type=int, default=720)
    ap.add_argument("--render_h", type=int, default=520)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--require_pred_labels", action="store_true")
    ap.add_argument("--align_mode", default="robust_bbox", choices=["robust_bbox", "robust_bbox_axis", "centroid_radius", "none"])
    ap.add_argument("--align_q_low", type=float, default=2.0)
    ap.add_argument("--align_q_high", type=float, default=98.0)
    ap.add_argument("--no_plotly", action="store_true", help="No exporta HTML Plotly interactivo.")
    args = ap.parse_args()
    case_dir = Path(args.case_dir).resolve()
    out_png = Path(args.out_png).resolve()
    selected_models = parse_models(args.models)
    panel_w, panel_h, legend_w = auto_layout(len(selected_models), args)
    tmp_dir = ensure_dir(case_dir / "paper_tmp_renders_v2")
    meta_path = case_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No existe meta.json en {case_dir}")
    meta = load_json(meta_path)
    bundle_path = case_dir / "common" / "bundle_case_data.npz"
    if not bundle_path.exists():
        raise FileNotFoundError(f"No existe bundle_case_data.npz: {bundle_path}")
    z = np.load(bundle_path)
    xyz = z["xyz_8192"].astype(np.float32)
    gt = z["y_8192"].astype(np.int64)
    raw_path = Path(meta.get("raw_mesh_path", "")) if meta.get("raw_mesh_path", "") else None
    mesh_pv = None
    tm_aligned = None
    if raw_path is not None and raw_path.exists():
        tm = load_raw_trimesh(raw_path)
        if tm is not None:
            tm_aligned = align_trimesh_to_cloud(
                tm, xyz,
                mode=args.align_mode,
                q_low=args.align_q_low,
                q_high=args.align_q_high,
            )
            mesh_pv = trimesh_to_pyvista(tm_aligned)
    render_paths: Dict[str, Dict[str, Path]] = {}
    gt_all = tmp_dir / "gt_all.png"
    gt_d21 = tmp_dir / "gt_d21.png"
    gt_err_all = tmp_dir / "gt_err_all.png"
    gt_err_d21 = tmp_dir / "gt_err_d21.png"
    if args.force or not gt_all.exists():
        render_all_classes(mesh_pv, xyz, gt, gt_all, args.render_w, args.render_h, args.zoom, mesh_opacity=0.28)
    if args.force or not gt_d21.exists():
        render_d21_binary(mesh_pv, xyz, gt, args.d21_class, gt_d21, args.render_w, args.render_h, args.zoom)
    if args.force or not gt_err_all.exists():
        render_errors_all(mesh_pv, xyz, gt, gt, gt_err_all, args.render_w, args.render_h, args.zoom)
    if args.force or not gt_err_d21.exists():
        render_d21_errors(mesh_pv, xyz, gt, gt, args.d21_class, gt_err_d21, args.render_w, args.render_h, args.zoom)
    render_paths["gt"] = {"all": gt_all, "err_all": gt_err_all, "d21": gt_d21, "err_d21": gt_err_d21}
    used_pred = {}
    for model_key in selected_models:
        model_dir = case_dir / model_key
        pred_path = model_dir / "pred_labels.npy"
        used_pred[model_key] = bool(pred_path.exists())
        if args.require_pred_labels and not pred_path.exists():
            raise FileNotFoundError(
                f"Falta {pred_path}. Primero corra export_predictions_from_best_models_v1.py y luego rebuild/copie pred_labels.npy al case_dir."
            )
        if not pred_path.exists():
            render_paths[model_key] = {
                "all": model_dir / "04_inference_all.png",
                "err_all": model_dir / "05_inference_errors.png",
                "d21": model_dir / "06_inference_d21.png",
                "err_d21": model_dir / "06_inference_d21.png",
            }
            continue
        pred = load_pred_labels(pred_path)
        if pred.shape[0] != xyz.shape[0]:
            if args.require_pred_labels:
                raise ValueError(f"{pred_path} tiene shape {pred.shape}, esperado {(xyz.shape[0],)}")
            render_paths[model_key] = {
                "all": model_dir / "04_inference_all.png",
                "err_all": model_dir / "05_inference_errors.png",
                "d21": model_dir / "06_inference_d21.png",
                "err_d21": model_dir / "06_inference_d21.png",
            }
            continue
        p_all = tmp_dir / f"{model_key}_all.png"
        p_err_all = tmp_dir / f"{model_key}_err_all.png"
        p_d21 = tmp_dir / f"{model_key}_d21.png"
        p_err_d21 = tmp_dir / f"{model_key}_err_d21.png"
        if args.force or not p_all.exists():
            render_all_classes(mesh_pv, xyz, pred, p_all, args.render_w, args.render_h, args.zoom, mesh_opacity=0.28)
        if args.force or not p_err_all.exists():
            render_errors_all(mesh_pv, xyz, pred, gt, p_err_all, args.render_w, args.render_h, args.zoom)
        if args.force or not p_d21.exists():
            render_d21_binary(mesh_pv, xyz, pred, args.d21_class, p_d21, args.render_w, args.render_h, args.zoom)
        if args.force or not p_err_d21.exists():
            render_d21_errors(mesh_pv, xyz, pred, gt, args.d21_class, p_err_d21, args.render_w, args.render_h, args.zoom)
        render_paths[model_key] = {"all": p_all, "err_all": p_err_all, "d21": p_d21, "err_d21": p_err_d21}
    out = compose_figure(render_paths, meta, selected_models, out_png, int(panel_w), int(panel_h), int(legend_w))
    plotly_dir = ""
    plotly_status = {}
    if not args.no_plotly:
        plotly_dir_obj, plotly_status = export_plotly_views(case_dir, selected_models, tm_aligned, xyz, gt, args.d21_class)
        plotly_dir = str(plotly_dir_obj)

    summary = {
        "case_dir": str(case_dir),
        "out_png": str(out),
        "tmp_dir": str(tmp_dir),
        "selected_models": selected_models,
        "layout": {
            "panel_w": int(panel_w), "panel_h": int(panel_h), "legend_w": int(legend_w),
            "render_w": int(args.render_w), "render_h": int(args.render_h), "zoom": float(args.zoom),
        },
        "used_pred_labels": used_pred,
        "plotly_dir": plotly_dir,
        "plotly_status": plotly_status,
        "alignment": {"mode": args.align_mode, "q_low": args.align_q_low, "q_high": args.align_q_high},
        "note": "Si used_pred_labels=false, esa columna usó PNG fallback y no render 3D homogéneo.",
    }
    save_json(summary, case_dir / "paper_figure_v2_summary.json")
    print(f"[OK] Figura final guardada en: {out}")
    print(f"[OK] Renders temporales en: {tmp_dir}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
