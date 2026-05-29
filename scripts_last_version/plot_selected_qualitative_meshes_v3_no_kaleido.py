#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_selected_qualitative_meshes_v3_no_kaleido.py

Versión 3 sin Kaleido ni Chrome.

Genera visualizaciones cualitativas de:
- Mejor caso
- Caso intermedio: mediana o promedio
- Peor caso

Salidas:
1) HTML interactivo con Plotly.
2) PNG estático con Matplotlib, sin usar Kaleido.

La cámara de cada vista se toma desde el mejor caso y se aplica igual
a las tres mallas de esa vista.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import trimesh

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ============================================================
# RUTAS DE CASOS
# ============================================================

BEST_PATH = Path(
    "/home/htaucare/Tesis_Amaro/case_comparisons/metric_case_selection_v5_global_meshes/"
    "selected_global_meshes/best/best__row91__01MAVT6A_upper.obj"
)

MEAN_PATH = Path(
    "/home/htaucare/Tesis_Amaro/case_comparisons/metric_case_selection_v5_global_meshes/"
    "selected_global_meshes/closest_to_mean/closest_to_mean__row46__WQOGLZY4_upper.obj"
)

MEDIAN_PATH = Path(
    "/home/htaucare/Tesis_Amaro/case_comparisons/metric_case_selection_v5_global_meshes/"
    "selected_global_meshes/closest_to_median/closest_to_median__row89__015RHV4X_upper.obj"
)

WORST_PATH = Path(
    "/home/htaucare/Tesis_Amaro/case_comparisons/metric_case_selection_v5_global_meshes/"
    "selected_global_meshes/worst/worst__row71__3EU06ZN9_upper.obj"
)


# ============================================================
# CÁMARAS MAESTRAS
# Cada cámara fue tomada desde el mejor caso y se aplicará a:
# scene, scene2 y scene3.
# ============================================================

CAMERA_VIEWS = {
    "oclusal_superior": dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0, y=0, z=5.23),
        projection=dict(type="perspective"),
    ),

    "lateral_derecha": dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(
            x=1.0947373729242973,
            y=5.462293827280975e-17,
            z=0.07464118451756574,
        ),
        projection=dict(type="perspective"),
    ),

    "frontal": dict(
        up=dict(x=0, y=0, z=1),
        center=dict(
            x=0.009610785552464467,
            y=-0.010284915074534523,
            z=0.1508454210913731,
        ),
        eye=dict(
            x=0.009610785552464534,
            y=1.833944900934298,
            z=0.2254520048546252,
        ),
        projection=dict(type="perspective"),
    ),

    "isometrica": dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(
            x=0.790344773730721,
            y=0.790344773730721,
            z=0.54336203196897,
        ),
        projection=dict(type="perspective"),
    ),
}


# ============================================================
# COLORES
# ============================================================

COLOR_PRESETS = {
    "neutral": {
        "Mejor caso": "#DCE6EF",
        "Caso intermedio": "#E8E6DF",
        "Peor caso": "#D8D8D8",
    },
    "status": {
        "Mejor caso": "#DDEFD8",       # verde suave
        "Caso intermedio": "#F7D7A1",  # mango suave
        "Peor caso": "#F2B6B6",        # rosado suave
    },
    "beige": {
        "Mejor caso": "#E8E0D0",
        "Caso intermedio": "#E8E0D0",
        "Peor caso": "#E8E0D0",
    },
}


# ============================================================
# UTILIDADES GENERALES
# ============================================================

def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_mesh_any(path: Path) -> trimesh.Trimesh:
    if not path.exists():
        raise FileNotFoundError(f"No existe la malla: {path}")

    obj = trimesh.load(path, process=False)

    if isinstance(obj, trimesh.Scene):
        geoms = []
        for geom in obj.geometry.values():
            if isinstance(geom, trimesh.Trimesh):
                geoms.append(geom)

        if not geoms:
            raise ValueError(f"No se encontraron geometrías Trimesh en: {path}")

        mesh = trimesh.util.concatenate(geoms)

    elif isinstance(obj, trimesh.Trimesh):
        mesh = obj

    else:
        raise TypeError(f"Tipo no soportado en {path}: {type(obj)}")

    if mesh.vertices is None or len(mesh.vertices) == 0:
        raise ValueError(f"Malla sin vértices: {path}")

    if mesh.faces is None or len(mesh.faces) == 0:
        raise ValueError(f"Malla sin caras: {path}")

    return mesh


def selected_case_paths(intermediate: str) -> List[Tuple[str, Path]]:
    intermediate = intermediate.lower().strip()

    if intermediate == "median":
        return [
            ("Mejor caso", BEST_PATH),
            ("Caso intermedio (mediana)", MEDIAN_PATH),
            ("Peor caso", WORST_PATH),
        ]

    if intermediate == "mean":
        return [
            ("Mejor caso", BEST_PATH),
            ("Caso intermedio (promedio)", MEAN_PATH),
            ("Peor caso", WORST_PATH),
        ]

    raise ValueError("--intermediate debe ser 'median' o 'mean'.")


def get_color(label: str, color_mode: str) -> str:
    colors = COLOR_PRESETS[color_mode]
    low = label.lower()

    if "mejor" in low:
        return colors["Mejor caso"]

    if "intermedio" in low:
        return colors["Caso intermedio"]

    if "peor" in low:
        return colors["Peor caso"]

    return "#E8E0D0"


def hex_to_rgb01(hex_color: str) -> np.ndarray:
    hex_color = hex_color.strip().lstrip("#")
    return np.array(
        [
            int(hex_color[0:2], 16),
            int(hex_color[2:4], 16),
            int(hex_color[4:6], 16),
        ],
        dtype=np.float32,
    ) / 255.0


def parse_manual_ranges(text: str | None) -> Dict[str, List[float]] | None:
    """
    Formato:
    --manual_ranges "x=-35,35;y=-35,35;z=-20,25"
    """
    if not text:
        return None

    out = {}
    for chunk in text.split(";"):
        name, vals = chunk.split("=")
        a, b = vals.split(",")
        out[name.strip().lower()] = [float(a), float(b)]

    for key in ["x", "y", "z"]:
        if key not in out:
            raise ValueError("manual_ranges debe incluir x, y, z.")

    return out


def parse_zoom_by_view(text: str | None) -> Dict[str, float]:
    """
    Formato:
    --zoom_by_view "lateral_derecha=0.85,frontal=1.0"
    """
    if not text:
        return {}

    out = {}
    for item in text.split(","):
        k, v = item.split("=")
        out[k.strip()] = float(v)
    return out


def compute_common_ranges(
    meshes: List[trimesh.Trimesh],
    pad: float,
    manual_ranges: Dict[str, List[float]] | None,
) -> Dict[str, List[float]]:

    if manual_ranges is not None:
        return manual_ranges

    mins = np.vstack([m.bounds[0] for m in meshes]).min(axis=0)
    maxs = np.vstack([m.bounds[1] for m in meshes]).max(axis=0)

    return {
        "x": [float(mins[0] - pad), float(maxs[0] + pad)],
        "y": [float(mins[1] - pad), float(maxs[1] + pad)],
        "z": [float(mins[2] - pad), float(maxs[2] + pad)],
    }


def apply_zoom_to_camera(camera: Dict[str, Any], zoom: float) -> Dict[str, Any]:
    """
    zoom < 1.0 acerca la cámara.
    zoom > 1.0 aleja la cámara.
    """
    cam = copy.deepcopy(camera)
    eye = cam.get("eye", {})
    center = cam.get("center", dict(x=0, y=0, z=0))

    for axis in ["x", "y", "z"]:
        eye_val = float(eye.get(axis, 0.0))
        center_val = float(center.get(axis, 0.0))
        eye[axis] = center_val + (eye_val - center_val) * zoom

    cam["eye"] = eye
    return cam


def apply_zoom_to_ranges(
    ranges: Dict[str, List[float]],
    zoom: float,
) -> Dict[str, List[float]]:
    """
    Para Matplotlib: zoom < 1.0 reduce rangos y acerca visualmente.
    """
    if zoom == 1.0:
        return copy.deepcopy(ranges)

    out = {}
    for axis in ["x", "y", "z"]:
        a, b = ranges[axis]
        c = 0.5 * (a + b)
        half = 0.5 * (b - a) * zoom
        out[axis] = [c - half, c + half]

    return out


def load_camera_json(path: Path | None) -> Dict[str, Dict[str, Any]] | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"No existe camera_json: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# PLOTLY HTML
# ============================================================

def mesh_to_plotly_trace(
    mesh: trimesh.Trimesh,
    label: str,
    color: str,
    opacity: float,
    flatshading: bool,
) -> go.Mesh3d:

    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.faces)

    return go.Mesh3d(
        x=v[:, 0],
        y=v[:, 1],
        z=v[:, 2],
        i=f[:, 0],
        j=f[:, 1],
        k=f[:, 2],
        color=color,
        opacity=opacity,
        flatshading=flatshading,
        name=label,
        showscale=False,
        hoverinfo="skip",
        lighting=dict(
            ambient=0.52,
            diffuse=0.75,
            fresnel=0.08,
            roughness=0.55,
            specular=0.18,
        ),
        lightposition=dict(x=100, y=200, z=300),
    )


def make_plotly_scene(
    ranges: Dict[str, List[float]],
    camera: Dict[str, Any],
    show_axes: bool,
) -> Dict[str, Any]:

    axis_cfg = dict(
        visible=show_axes,
        showbackground=False,
        showgrid=False,
        zeroline=False,
    )

    return dict(
        xaxis=dict(**axis_cfg, range=ranges["x"]),
        yaxis=dict(**axis_cfg, range=ranges["y"]),
        zaxis=dict(**axis_cfg, range=ranges["z"]),
        aspectmode="data",
        bgcolor="white",
        camera=camera,
    )


def build_post_script() -> str:
    return """
const gd = document.getElementById('{plot_id}');
let syncing = false;

gd.on('plotly_relayout', function(e) {
    if (syncing) return;

    const cam = e['scene.camera'] || e['scene2.camera'] || e['scene3.camera'];

    if (cam) {
        syncing = true;
        Plotly.relayout(gd, {
            'scene.camera': cam,
            'scene2.camera': cam,
            'scene3.camera': cam
        }).then(() => {
            syncing = false;
        });

        console.log('CAMERA_ACTUAL_SYNC:');
        console.log(JSON.stringify(cam, null, 2));
    }
});

window.printCameras = function() {
    console.log(JSON.stringify({
        scene: gd._fullLayout.scene.camera,
        scene2: gd._fullLayout.scene2.camera,
        scene3: gd._fullLayout.scene3.camera
    }, null, 2));
};

window.printMasterCamera = function() {
    console.log(JSON.stringify(gd._fullLayout.scene.camera, null, 2));
};
"""


def build_plotly_figure(
    cases: List[Tuple[str, Path, trimesh.Trimesh]],
    view_name: str,
    camera: Dict[str, Any],
    ranges: Dict[str, List[float]],
    color_mode: str,
    opacity: float,
    flatshading: bool,
    width: int,
    height: int,
    show_axes: bool,
    show_subplot_titles: bool,
    show_side_labels: bool,
) -> go.Figure:

    subplot_titles = [c[0] for c in cases] if show_subplot_titles else ["", "", ""]

    fig = make_subplots(
        rows=3,
        cols=1,
        specs=[
            [{"type": "scene"}],
            [{"type": "scene"}],
            [{"type": "scene"}],
        ],
        vertical_spacing=0.02,
        subplot_titles=subplot_titles,
    )

    for row_i, (label, path, mesh) in enumerate(cases, start=1):
        color = get_color(label, color_mode)

        fig.add_trace(
            mesh_to_plotly_trace(
                mesh=mesh,
                label=label,
                color=color,
                opacity=opacity,
                flatshading=flatshading,
            ),
            row=row_i,
            col=1,
        )

    scene_cfg = make_plotly_scene(ranges, camera, show_axes)

    fig.update_layout(
        scene=copy.deepcopy(scene_cfg),
        scene2=copy.deepcopy(scene_cfg),
        scene3=copy.deepcopy(scene_cfg),
        width=width,
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        margin=dict(l=5, r=5, t=65, b=5),
        title=dict(
            text=f"Casos representativos del análisis cualitativo — vista: {view_name}",
            x=0.5,
            xanchor="center",
            font=dict(size=18),
        ),
        font=dict(size=13),
    )

    if show_side_labels:
        extra = []
        y_positions = [0.86, 0.53, 0.20]
        for (label, path, _), y in zip(cases, y_positions):
            extra.append(
                dict(
                    text=f"{label}<br><sup>{path.stem}</sup>",
                    x=0.005,
                    y=y,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    align="left",
                    font=dict(size=11),
                )
            )
        fig.update_layout(annotations=list(fig.layout.annotations) + extra)

    return fig


# ============================================================
# MATPLOTLIB PNG SIN KALEIDO
# ============================================================

def camera_to_elev_azim(camera: Dict[str, Any]) -> Tuple[float, float]:
    """
    Convierte la cámara de Plotly a elevación y azimut aproximados para Matplotlib.
    Se usa la misma orientación para las tres mallas.
    """
    eye = camera.get("eye", dict(x=1, y=1, z=1))
    center = camera.get("center", dict(x=0, y=0, z=0))

    dx = float(eye.get("x", 0.0)) - float(center.get("x", 0.0))
    dy = float(eye.get("y", 0.0)) - float(center.get("y", 0.0))
    dz = float(eye.get("z", 0.0)) - float(center.get("z", 0.0))

    rxy = max(np.sqrt(dx * dx + dy * dy), 1e-8)
    elev = np.degrees(np.arctan2(dz, rxy))
    azim = np.degrees(np.arctan2(dy, dx))

    return float(elev), float(azim)


def sample_faces_for_static(
    mesh: trimesh.Trimesh,
    max_faces: int,
    rng: np.random.Generator,
) -> np.ndarray:
    faces = np.asarray(mesh.faces)

    if max_faces <= 0:
        return faces

    if faces.shape[0] <= max_faces:
        return faces

    idx = rng.choice(faces.shape[0], size=max_faces, replace=False)
    return faces[idx]


def shaded_facecolors(
    vertices: np.ndarray,
    faces: np.ndarray,
    base_hex: str,
) -> np.ndarray:
    tris = vertices[faces]

    n = np.cross(
        tris[:, 1, :] - tris[:, 0, :],
        tris[:, 2, :] - tris[:, 0, :],
    )

    norm = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / (norm + 1e-8)

    light = np.array([0.35, 0.45, 0.82], dtype=np.float32)
    light = light / (np.linalg.norm(light) + 1e-8)

    # abs para evitar caras casi negras si las normales vienen invertidas
    intensity = 0.38 + 0.62 * np.abs(n @ light)
    base = hex_to_rgb01(base_hex)

    rgb = np.clip(base[None, :] * intensity[:, None], 0.0, 1.0)
    rgba = np.concatenate([rgb, np.ones((rgb.shape[0], 1), dtype=np.float32)], axis=1)

    return rgba


def render_static_matplotlib(
    cases: List[Tuple[str, Path, trimesh.Trimesh]],
    view_name: str,
    camera: Dict[str, Any],
    ranges: Dict[str, List[float]],
    color_mode: str,
    out_png: Path,
    width_px: int,
    height_px: int,
    dpi: int,
    show_titles: bool,
    max_faces_static: int,
    seed: int,
) -> None:

    rng = np.random.default_rng(seed)
    elev, azim = camera_to_elev_azim(camera)

    fig_w = width_px / dpi
    fig_h = height_px / dpi

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor="white")

    if show_titles:
        fig.suptitle(
            f"Casos representativos del análisis cualitativo — vista: {view_name}",
            fontsize=12,
            y=0.985,
        )

    for idx, (label, path, mesh) in enumerate(cases, start=1):
        ax = fig.add_subplot(3, 1, idx, projection="3d")
        ax.set_facecolor("white")

        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = sample_faces_for_static(mesh, max_faces=max_faces_static, rng=rng)

        color = get_color(label, color_mode)
        face_colors = shaded_facecolors(vertices, faces, color)
        tris = vertices[faces]

        poly = Poly3DCollection(
            tris,
            facecolors=face_colors,
            edgecolors="none",
            linewidths=0.0,
            alpha=1.0,
        )

        ax.add_collection3d(poly)

        ax.set_xlim(ranges["x"])
        ax.set_ylim(ranges["y"])
        ax.set_zlim(ranges["z"])

        try:
            ax.set_box_aspect((
                ranges["x"][1] - ranges["x"][0],
                ranges["y"][1] - ranges["y"][0],
                ranges["z"][1] - ranges["z"][0],
            ))
        except Exception:
            pass

        try:
            ax.set_proj_type("persp")
        except Exception:
            pass

        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()

        if show_titles:
            ax.set_title(label, fontsize=10, pad=2)

    fig.subplots_adjust(
        left=0.01,
        right=0.99,
        top=0.96 if show_titles else 0.99,
        bottom=0.01,
        hspace=0.02,
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, facecolor="white")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--out_dir",
        default="/home/htaucare/Tesis_Amaro/case_comparisons/qualitative_plotly_mesh_views_v3_no_kaleido",
    )

    ap.add_argument(
        "--intermediate",
        choices=["median", "mean"],
        default="median",
    )

    ap.add_argument(
        "--views",
        nargs="+",
        default=["isometrica", "frontal", "lateral_derecha", "oclusal_superior"],
    )

    ap.add_argument(
        "--camera_json",
        default=None,
        help="JSON externo con cámaras. Si se entrega, reemplaza CAMERA_VIEWS.",
    )

    ap.add_argument(
        "--zoom",
        type=float,
        default=1.0,
        help="Zoom global. Menor a 1 acerca; mayor a 1 aleja.",
    )

    ap.add_argument(
        "--zoom_by_view",
        default=None,
        help='Ej: "lateral_derecha=0.85,frontal=1.0,isometrica=1.0,oclusal_superior=1.0"',
    )

    ap.add_argument(
        "--manual_ranges",
        default=None,
        help='Ej: "x=-35,35;y=-35,35;z=-20,25"',
    )

    ap.add_argument("--range_pad", type=float, default=2.0)

    ap.add_argument(
        "--color_mode",
        choices=["neutral", "status", "beige"],
        default="neutral",
    )

    ap.add_argument("--opacity", type=float, default=1.0)
    ap.add_argument("--flatshading", action="store_true")

    ap.add_argument("--width", type=int, default=1400)
    ap.add_argument("--height", type=int, default=2300)

    ap.add_argument("--write_html", action="store_true")
    ap.add_argument("--write_png_mpl", action="store_true")

    ap.add_argument("--show_axes", action="store_true")
    ap.add_argument("--hide_subplot_titles", action="store_true")
    ap.add_argument("--show_side_labels", action="store_true")

    ap.add_argument(
        "--static_dpi",
        type=int,
        default=250,
        help="DPI para PNG de Matplotlib.",
    )

    ap.add_argument(
        "--max_faces_static",
        type=int,
        default=150000,
        help="Máximo de caras por malla en PNG estático. Use 0 para todas.",
    )

    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    views = args.views
    if len(views) == 1 and views[0].lower() == "all":
        views = ["isometrica", "frontal", "lateral_derecha", "oclusal_superior"]

    cameras = load_camera_json(Path(args.camera_json)) if args.camera_json else CAMERA_VIEWS
    zoom_by_view = parse_zoom_by_view(args.zoom_by_view)
    manual_ranges = parse_manual_ranges(args.manual_ranges)

    raw_cases = selected_case_paths(args.intermediate)

    cases = []
    meshes = []

    for label, path in raw_cases:
        mesh = load_mesh_any(path)
        cases.append((label, path, mesh))
        meshes.append(mesh)

    base_ranges = compute_common_ranges(
        meshes=meshes,
        pad=args.range_pad,
        manual_ranges=manual_ranges,
    )

    run_meta = {
        "script": "plot_selected_qualitative_meshes_v3_no_kaleido.py",
        "intermediate": args.intermediate,
        "views": views,
        "out_dir": str(out_dir),
        "color_mode": args.color_mode,
        "zoom": args.zoom,
        "zoom_by_view": zoom_by_view,
        "base_ranges": base_ranges,
        "write_html": args.write_html,
        "write_png_mpl": args.write_png_mpl,
        "width": args.width,
        "height": args.height,
        "static_dpi": args.static_dpi,
        "max_faces_static": args.max_faces_static,
        "cases": [
            {"label": label, "path": str(path), "stem": path.stem}
            for label, path, _ in cases
        ],
        "cameras_used": {},
        "ranges_used_for_static": {},
    }

    for view_name in views:
        if view_name not in cameras:
            raise KeyError(
                f"Vista '{view_name}' no existe. Disponibles: {list(cameras.keys())}"
            )

        view_zoom = zoom_by_view.get(view_name, args.zoom)

        camera = apply_zoom_to_camera(
            camera=copy.deepcopy(cameras[view_name]),
            zoom=view_zoom,
        )

        # Para PNG Matplotlib el zoom se refleja mejor reduciendo rangos.
        static_ranges = apply_zoom_to_ranges(base_ranges, view_zoom)

        run_meta["cameras_used"][view_name] = camera
        run_meta["ranges_used_for_static"][view_name] = static_ranges

        stem = f"qualitative_cases_{args.intermediate}_{args.color_mode}_{view_name}"

        if args.write_html:
            fig = build_plotly_figure(
                cases=cases,
                view_name=view_name,
                camera=camera,
                ranges=base_ranges,
                color_mode=args.color_mode,
                opacity=args.opacity,
                flatshading=args.flatshading,
                width=args.width,
                height=args.height,
                show_axes=args.show_axes,
                show_subplot_titles=not args.hide_subplot_titles,
                show_side_labels=args.show_side_labels,
            )

            html_path = out_dir / f"{stem}.html"
            fig.write_html(
                str(html_path),
                include_plotlyjs="cdn",
                post_script=build_post_script(),
                config={
                    "displaylogo": False,
                    "scrollZoom": True,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": stem,
                        "height": args.height,
                        "width": args.width,
                        "scale": 3,
                    },
                },
            )
            print(f"[OK] HTML: {html_path}")

        if args.write_png_mpl:
            png_path = out_dir / f"{stem}_mpl.png"

            render_static_matplotlib(
                cases=cases,
                view_name=view_name,
                camera=camera,
                ranges=static_ranges,
                color_mode=args.color_mode,
                out_png=png_path,
                width_px=args.width,
                height_px=args.height,
                dpi=args.static_dpi,
                show_titles=not args.hide_subplot_titles,
                max_faces_static=args.max_faces_static,
                seed=args.seed,
            )

            print(f"[OK] PNG Matplotlib: {png_path}")

    save_json(run_meta, out_dir / "run_meta_qualitative_plotly_v3_no_kaleido.json")
    save_json(cameras, out_dir / "camera_presets_base_v3.json")

    print(f"[OK] Metadata: {out_dir / 'run_meta_qualitative_plotly_v3_no_kaleido.json'}")
    print("[DONE] Finalizado sin Kaleido.")


if __name__ == "__main__":
    main()