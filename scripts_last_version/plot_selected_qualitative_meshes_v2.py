#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_selected_qualitative_meshes_v2.py

Visualización cualitativa de casos representativos:
- Mejor caso
- Caso intermedio: mediana o promedio
- Peor caso

Genera una figura vertical con 3 mallas por vista:
1) mejor
2) intermedio
3) peor

Características:
- Usa la cámara del mejor caso como cámara maestra para las tres mallas.
- Permite modificar cámaras por diccionario interno o JSON externo.
- Permite zoom global o zoom por vista.
- Permite colores neutros o colores por caso.
- Exporta HTML interactivo y PNG estático.
- Sincroniza cámaras dentro del HTML para copiar coordenadas finales.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import trimesh
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
# Estas cámaras corresponden a la cámara ajustada sobre el mejor caso.
# Cada cámara se aplica igual a scene, scene2 y scene3.
# ============================================================

CAMERA_VIEWS = {
    "oclusal_superior": dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0, y=0, z=1.0500316697772248),
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
# UTILIDADES
# ============================================================

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
        raise TypeError(f"Tipo de objeto no soportado en {path}: {type(obj)}")

    if mesh.vertices is None or len(mesh.vertices) == 0:
        raise ValueError(f"La malla no contiene vértices válidos: {path}")

    if mesh.faces is None or len(mesh.faces) == 0:
        raise ValueError(f"La malla no contiene caras válidas: {path}")

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


def mesh_to_trace(
    mesh: trimesh.Trimesh,
    label: str,
    color: str,
    opacity: float,
    flatshading: bool,
) -> go.Mesh3d:
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)

    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
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


def compute_common_ranges(
    meshes: List[trimesh.Trimesh],
    pad: float,
    manual_ranges: Dict[str, List[float]] | None = None,
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


def parse_manual_ranges(text: str | None) -> Dict[str, List[float]] | None:
    """
    Formato:
    --manual_ranges "x=-35,35;y=-35,35;z=-20,25"
    """
    if not text:
        return None

    out = {}
    chunks = text.split(";")
    for ch in chunks:
        name, vals = ch.split("=")
        a, b = vals.split(",")
        name = name.strip().lower()
        out[name] = [float(a), float(b)]

    for key in ["x", "y", "z"]:
        if key not in out:
            raise ValueError("manual_ranges debe incluir x, y, z.")

    return out


def parse_zoom_by_view(text: str | None) -> Dict[str, float]:
    """
    Formato:
    --zoom_by_view "lateral_derecha=0.9,frontal=1.0,isometrica=1.0,oclusal_superior=1.0"
    """
    if not text:
        return {}

    out = {}
    for item in text.split(","):
        k, v = item.split("=")
        out[k.strip()] = float(v)
    return out


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
        ctr_val = float(center.get(axis, 0.0))
        eye[axis] = ctr_val + (eye_val - ctr_val) * zoom

    cam["eye"] = eye
    return cam


def load_camera_json(path: Path | None) -> Dict[str, Dict[str, Any]] | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"No existe camera_json: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def make_scene_config(
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
    """
    Sincroniza las tres escenas dentro del HTML.
    Al mover una cámara, replica esa cámara en las otras dos.
    Además imprime la cámara en consola.
    """
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


# ============================================================
# FIGURA
# ============================================================

def build_figure(
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

        trace = mesh_to_trace(
            mesh=mesh,
            label=label,
            color=color,
            opacity=opacity,
            flatshading=flatshading,
        )
        fig.add_trace(trace, row=row_i, col=1)

    scene_cfg = make_scene_config(ranges, camera, show_axes=show_axes)

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
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--out_dir",
        default="/home/htaucare/Tesis_Amaro/case_comparisons/qualitative_plotly_mesh_views_v2",
        help="Carpeta de salida.",
    )

    ap.add_argument(
        "--intermediate",
        choices=["median", "mean"],
        default="median",
        help="Caso intermedio: median o mean.",
    )

    ap.add_argument(
        "--views",
        nargs="+",
        default=["isometrica", "frontal", "lateral_derecha", "oclusal_superior"],
        help="Vistas a exportar. Use: all, isometrica, frontal, lateral_derecha, oclusal_superior.",
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
        help='Zoom por vista. Ej: "lateral_derecha=0.85,frontal=1.0".',
    )

    ap.add_argument(
        "--manual_ranges",
        default=None,
        help='Rangos manuales. Ej: "x=-35,35;y=-35,35;z=-20,25".',
    )

    ap.add_argument(
        "--range_pad",
        type=float,
        default=2.0,
        help="Padding para rangos automáticos.",
    )

    ap.add_argument(
        "--color_mode",
        choices=["neutral", "status", "beige"],
        default="neutral",
        help="Modo de color: neutral, status o beige.",
    )

    ap.add_argument("--opacity", type=float, default=1.0)
    ap.add_argument("--flatshading", action="store_true")

    ap.add_argument("--width", type=int, default=1400)
    ap.add_argument("--height", type=int, default=2300)
    ap.add_argument("--scale", type=int, default=3)

    ap.add_argument("--write_html", action="store_true")
    ap.add_argument("--write_png", action="store_true")

    ap.add_argument("--show_axes", action="store_true")
    ap.add_argument("--hide_subplot_titles", action="store_true")
    ap.add_argument("--show_side_labels", action="store_true")

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

    ranges = compute_common_ranges(
        meshes=meshes,
        pad=args.range_pad,
        manual_ranges=manual_ranges,
    )

    run_meta = {
        "intermediate": args.intermediate,
        "views": views,
        "out_dir": str(out_dir),
        "color_mode": args.color_mode,
        "opacity": args.opacity,
        "flatshading": args.flatshading,
        "width": args.width,
        "height": args.height,
        "scale": args.scale,
        "zoom": args.zoom,
        "zoom_by_view": zoom_by_view,
        "ranges": ranges,
        "cases": [
            {"label": label, "path": str(path), "stem": path.stem}
            for label, path, _ in cases
        ],
        "cameras_used": {},
    }

    for view_name in views:
        if view_name not in cameras:
            raise KeyError(
                f"Vista '{view_name}' no existe en cámaras. "
                f"Disponibles: {list(cameras.keys())}"
            )

        camera = copy.deepcopy(cameras[view_name])

        view_zoom = zoom_by_view.get(view_name, args.zoom)
        camera = apply_zoom_to_camera(camera, view_zoom)

        run_meta["cameras_used"][view_name] = camera

        fig = build_figure(
            cases=cases,
            view_name=view_name,
            camera=camera,
            ranges=ranges,
            color_mode=args.color_mode,
            opacity=args.opacity,
            flatshading=args.flatshading,
            width=args.width,
            height=args.height,
            show_axes=args.show_axes,
            show_subplot_titles=not args.hide_subplot_titles,
            show_side_labels=args.show_side_labels,
        )

        stem = f"qualitative_cases_{args.intermediate}_{args.color_mode}_{view_name}"

        if args.write_html:
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
                        "scale": args.scale,
                    },
                },
            )
            print(f"[OK] HTML: {html_path}")

        if args.write_png:
            png_path = out_dir / f"{stem}.png"
            try:
                fig.write_image(
                    str(png_path),
                    width=args.width,
                    height=args.height,
                    scale=args.scale,
                )
                print(f"[OK] PNG:  {png_path}")
            except Exception as e:
                print(f"[WARN] No se pudo exportar PNG: {png_path}")
                print(f"       Error: {e}")
                print("       Instale kaleido con: pip install -U kaleido")

    save_json(run_meta, out_dir / "run_meta_qualitative_plotly_v2.json")
    save_json(cameras, out_dir / "camera_presets_base.json")
    print(f"[OK] Metadata: {out_dir / 'run_meta_qualitative_plotly_v2.json'}")
    print("[DONE] Finalizado.")


if __name__ == "__main__":
    main()