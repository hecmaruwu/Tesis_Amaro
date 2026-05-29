#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_selected_qualitative_meshes_v5_clean_titles.py

Versión final corregida para visualización cualitativa de mallas dentales.

Objetivo:
- Generar HTML interactivos con Plotly para el análisis cualitativo.
- Mostrar tres casos representativos:
    1. Mejor caso
    2. Caso intermedio, usando mediana o promedio
    3. Peor caso
- Organizar los tres casos verticalmente en una figura por vista.
- Usar nombres de vistas limpios para tesis:
    * isometrica          -> isométrica
    * frontal             -> frontal
    * lateral_derecha     -> lateral derecha
    * oclusal_superior    -> oclusal superior
- Mantener colores por estado:
    * Mejor caso        = verde suave
    * Caso intermedio   = mango suave
    * Peor caso         = rosado suave
- No usar Kaleido.
- No usar Chrome.
- No usar PNG de Matplotlib.
- Generar solo HTML interactivo.

Nota:
- La vista oclusal superior mantiene cámaras distintas por fila para mejorar
  el encuadre de cada caso:
    scene  = mejor caso
    scene2 = caso intermedio
    scene3 = peor caso
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
# VISTAS
# ============================================================

ALL_VIEWS = [
    "isometrica",
    "frontal",
    "lateral_derecha",
    "oclusal_superior",
]

MAIN_VIEWS = [
    "oclusal_superior",
    "isometrica",
]

ANNEX_VIEWS = [
    "frontal",
    "lateral_derecha",
]

VIEW_DISPLAY_NAMES = {
    "isometrica": "isométrica",
    "frontal": "frontal",
    "lateral_derecha": "lateral derecha",
    "oclusal_superior": "oclusal superior",
}


def view_display_name(view_name: str, ascii_titles: bool = False) -> str:
    """
    Devuelve el nombre visible de la vista.

    ascii_titles=True evita tildes en los títulos del HTML. Esto puede ser útil
    si el navegador o la captura de pantalla no renderiza bien caracteres UTF-8.
    """
    if ascii_titles:
        ascii_map = {
            "isometrica": "isometrica",
            "frontal": "frontal",
            "lateral_derecha": "lateral derecha",
            "oclusal_superior": "oclusal superior",
        }
        return ascii_map.get(view_name, view_name)

    return VIEW_DISPLAY_NAMES.get(view_name, view_name)


def expand_views(raw_views: List[str]) -> List[str]:
    """
    Permite usar:
    --views all
    --views main
    --views principales
    --views annex
    --views anexos
    o una lista explícita de vistas.
    """
    if len(raw_views) == 1:
        key = raw_views[0].lower().strip()

        if key == "all":
            return ALL_VIEWS

        if key in {"main", "principal", "principales"}:
            return MAIN_VIEWS

        if key in {"annex", "anexo", "anexos"}:
            return ANNEX_VIEWS

    return raw_views


# ============================================================
# CÁMARAS
# ============================================================

CAMERA_VIEWS = {
    "oclusal_superior": dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0, y=0, z=1.1660883697387594),
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


OCLUSAL_SUPERIOR_PER_SCENE_CAMERAS = {
    "scene": dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0, y=0, z=1.1660883697387594),
        projection=dict(type="perspective"),
    ),
    "scene2": dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0, y=0, z=1.3352015910395147),
        projection=dict(type="perspective"),
    ),
    "scene3": dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0, y=0, z=1.7262774418516171),
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
        "Mejor caso": "#DDEFD8",
        "Caso intermedio": "#F7D7A1",
        "Peor caso": "#F2B6B6",
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


def parse_manual_ranges(text: str | None) -> Dict[str, List[float]] | None:
    """
    Formato:
    --manual_ranges "x=-35,35;y=-35,35;z=-20,25"
    """
    if not text:
        return None

    out: Dict[str, List[float]] = {}

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
    --zoom_by_view "lateral_derecha=0.85,frontal=0.90"
    """
    if not text:
        return {}

    out: Dict[str, float] = {}

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


def get_scene_cameras_for_view(
    view_name: str,
    base_camera: Dict[str, Any],
    zoom: float,
) -> List[Dict[str, Any]]:

    if view_name != "oclusal_superior":
        cam = apply_zoom_to_camera(copy.deepcopy(base_camera), zoom)
        return [copy.deepcopy(cam), copy.deepcopy(cam), copy.deepcopy(cam)]

    cam_best = apply_zoom_to_camera(
        copy.deepcopy(OCLUSAL_SUPERIOR_PER_SCENE_CAMERAS["scene"]),
        zoom,
    )
    cam_intermediate = apply_zoom_to_camera(
        copy.deepcopy(OCLUSAL_SUPERIOR_PER_SCENE_CAMERAS["scene2"]),
        zoom,
    )
    cam_worst = apply_zoom_to_camera(
        copy.deepcopy(OCLUSAL_SUPERIOR_PER_SCENE_CAMERAS["scene3"]),
        zoom,
    )

    return [
        copy.deepcopy(cam_best),
        copy.deepcopy(cam_intermediate),
        copy.deepcopy(cam_worst),
    ]


def load_camera_json(path: Path | None) -> Dict[str, Dict[str, Any]] | None:
    if path is None:
        return None

    if not path.exists():
        raise FileNotFoundError(f"No existe camera_json: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# PLOTLY
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


def build_post_script(sync_cameras: bool) -> str:
    """
    Por defecto, no sincroniza cámaras, porque la vista oclusal superior
    usa una cámara distinta por fila.
    """
    if not sync_cameras:
        return """
const gd = document.getElementById('{plot_id}');

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
    scene_cameras: List[Dict[str, Any]],
    ranges: Dict[str, List[float]],
    color_mode: str,
    opacity: float,
    flatshading: bool,
    width: int,
    height: int,
    show_axes: bool,
    show_subplot_titles: bool,
    show_side_labels: bool,
    hide_main_title: bool,
    ascii_titles: bool,
) -> go.Figure:

    view_label = view_display_name(view_name, ascii_titles=ascii_titles)

    subplot_titles = [c[0] for c in cases] if show_subplot_titles else ["", "", ""]

    fig = make_subplots(
        rows=3,
        cols=1,
        specs=[
            [{"type": "scene"}],
            [{"type": "scene"}],
            [{"type": "scene"}],
        ],
        vertical_spacing=0.025,
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

    scene_cfg_1 = make_plotly_scene(ranges, scene_cameras[0], show_axes)
    scene_cfg_2 = make_plotly_scene(ranges, scene_cameras[1], show_axes)
    scene_cfg_3 = make_plotly_scene(ranges, scene_cameras[2], show_axes)

    if ascii_titles:
        main_title = f"Casos representativos del analisis cualitativo - vista: {view_label}"
    else:
        main_title = f"Casos representativos del análisis cualitativo — vista: {view_label}"

    fig.update_layout(
        scene=copy.deepcopy(scene_cfg_1),
        scene2=copy.deepcopy(scene_cfg_2),
        scene3=copy.deepcopy(scene_cfg_3),
        width=width,
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        margin=dict(l=5, r=5, t=70 if not hide_main_title else 35, b=5),
        title=None if hide_main_title else dict(
            text=main_title,
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
        default="/home/htaucare/Tesis_Amaro/case_comparisons/qualitative_plotly_mesh_views_v5_clean_titles",
    )

    ap.add_argument(
        "--intermediate",
        choices=["median", "mean"],
        default="median",
    )

    ap.add_argument(
        "--views",
        nargs="+",
        default=["all"],
        help=(
            "Use 'all', 'main', 'principales', 'annex', 'anexos' o una lista explícita: "
            "isometrica frontal lateral_derecha oclusal_superior."
        ),
    )

    ap.add_argument(
        "--camera_json",
        default=None,
        help=(
            "JSON externo con cámaras. Si se entrega, reemplaza CAMERA_VIEWS "
            "para las vistas normales. La excepción oclusal_superior se mantiene."
        ),
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
        help='Ej: "frontal=0.85,lateral_derecha=0.90"',
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
        default="status",
    )

    ap.add_argument("--opacity", type=float, default=1.0)
    ap.add_argument("--flatshading", action="store_true")

    ap.add_argument("--width", type=int, default=1400)
    ap.add_argument("--height", type=int, default=2300)

    ap.add_argument("--write_html", action="store_true")

    ap.add_argument("--show_axes", action="store_true")
    ap.add_argument("--hide_subplot_titles", action="store_true")
    ap.add_argument("--hide_main_title", action="store_true")
    ap.add_argument("--show_side_labels", action="store_true")

    ap.add_argument(
        "--ascii_titles",
        action="store_true",
        help=(
            "Usa títulos sin tildes. Útil si el navegador/captura no renderiza bien "
            "la palabra 'análisis'."
        ),
    )

    ap.add_argument(
        "--sync_cameras",
        action="store_true",
        help=(
            "Sincroniza las cámaras al mover el HTML. No se recomienda para la figura "
            "oclusal superior, porque esa vista usa una cámara distinta por fila."
        ),
    )

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    views = expand_views(args.views)

    cameras = load_camera_json(Path(args.camera_json)) if args.camera_json else CAMERA_VIEWS
    zoom_by_view = parse_zoom_by_view(args.zoom_by_view)
    manual_ranges = parse_manual_ranges(args.manual_ranges)

    raw_cases = selected_case_paths(args.intermediate)

    cases: List[Tuple[str, Path, trimesh.Trimesh]] = []
    meshes: List[trimesh.Trimesh] = []

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
        "script": "plot_selected_qualitative_meshes_v5_clean_titles.py",
        "intermediate": args.intermediate,
        "views": views,
        "view_display_names": {
            v: view_display_name(v, ascii_titles=args.ascii_titles)
            for v in views
        },
        "out_dir": str(out_dir),
        "color_mode": args.color_mode,
        "zoom": args.zoom,
        "zoom_by_view": zoom_by_view,
        "base_ranges": base_ranges,
        "write_html": args.write_html,
        "width": args.width,
        "height": args.height,
        "sync_cameras": args.sync_cameras,
        "ascii_titles": args.ascii_titles,
        "hide_main_title": args.hide_main_title,
        "hide_subplot_titles": args.hide_subplot_titles,
        "cases": [
            {"label": label, "path": str(path), "stem": path.stem}
            for label, path, _ in cases
        ],
        "cameras_used": {},
    }

    for view_name in views:
        if view_name not in cameras:
            raise KeyError(
                f"Vista '{view_name}' no existe. Disponibles: {list(cameras.keys())}"
            )

        view_zoom = zoom_by_view.get(view_name, args.zoom)

        scene_cameras = get_scene_cameras_for_view(
            view_name=view_name,
            base_camera=copy.deepcopy(cameras[view_name]),
            zoom=view_zoom,
        )

        run_meta["cameras_used"][view_name] = {
            "scene": scene_cameras[0],
            "scene2": scene_cameras[1],
            "scene3": scene_cameras[2],
        }

        stem = f"qualitative_cases_{args.intermediate}_{args.color_mode}_{view_name}"

        if args.write_html:
            fig = build_plotly_figure(
                cases=cases,
                view_name=view_name,
                scene_cameras=scene_cameras,
                ranges=base_ranges,
                color_mode=args.color_mode,
                opacity=args.opacity,
                flatshading=args.flatshading,
                width=args.width,
                height=args.height,
                show_axes=args.show_axes,
                show_subplot_titles=not args.hide_subplot_titles,
                show_side_labels=args.show_side_labels,
                hide_main_title=args.hide_main_title,
                ascii_titles=args.ascii_titles,
            )

            html_path = out_dir / f"{stem}.html"

            fig.write_html(
                str(html_path),
                include_plotlyjs="cdn",
                post_script=build_post_script(sync_cameras=args.sync_cameras),
                config={
                    "displaylogo": False,
                    "scrollZoom": True,
                    "displayModeBar": True,
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

    save_json(
        run_meta,
        out_dir / "run_meta_qualitative_plotly_v5_clean_titles.json",
    )

    save_json(
        CAMERA_VIEWS,
        out_dir / "camera_presets_base_v5.json",
    )

    print(f"[OK] Metadata: {out_dir / 'run_meta_qualitative_plotly_v5_clean_titles.json'}")
    print(f"[OK] Cámaras base: {out_dir / 'camera_presets_base_v5.json'}")
    print("[DONE] Finalizado sin Kaleido, sin Chrome y sin PNG Matplotlib.")


if __name__ == "__main__":
    main()