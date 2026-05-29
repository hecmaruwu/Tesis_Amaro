from pathlib import Path
import numpy as np
import trimesh
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =========================================================
# RUTAS
# =========================================================
BEST_PATH = Path("/home/htaucare/Tesis_Amaro/case_comparisons/metric_case_selection_v5_global_meshes/selected_global_meshes/best/best__row91__01MAVT6A_upper.obj")

MEAN_PATH = Path("/home/htaucare/Tesis_Amaro/case_comparisons/metric_case_selection_v5_global_meshes/selected_global_meshes/closest_to_mean/closest_to_mean__row46__WQOGLZY4_upper.obj")

MEDIAN_PATH = Path("/home/htaucare/Tesis_Amaro/case_comparisons/metric_case_selection_v5_global_meshes/selected_global_meshes/closest_to_median/closest_to_median__row89__015RHV4X_upper.obj")

WORST_PATH = Path("/home/htaucare/Tesis_Amaro/case_comparisons/metric_case_selection_v5_global_meshes/selected_global_meshes/worst/worst__row71__3EU06ZN9_upper.obj")


# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================
# Use "median" o "mean"
INTERMEDIATE_MODE = "median"

# Carpeta de salida
OUT_DIR = Path("/home/htaucare/Tesis_Amaro/case_comparisons/qualitative_plotly_mesh_views")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Si quiere fijar manualmente rangos idénticos en todos los gráficos:
# MANUAL_RANGES = {
#     "x": [-35, 35],
#     "y": [-35, 35],
#     "z": [-20, 20],
# }
MANUAL_RANGES = None

# Padding automático para rangos si no usa MANUAL_RANGES
AXIS_PAD = 2.0

# Colores
CASE_COLORS = {
    "Mejor caso": "lightsteelblue",
    "Caso intermedio": "lightgray",
    "Peor caso": "gainsboro",
}

# Exportar HTML y, si tiene kaleido instalado, también PNG
EXPORT_PNG = False


# =========================================================
# VISTAS / CÁMARAS
# Puede ajustar estas coordenadas y se aplicarán IGUAL
# a las 3 mallas del lienzo.
# =========================================================
CAMERA_VIEWS = {
    "isometrica": dict(
        eye=dict(x=1.6, y=1.6, z=1.1),
        center=dict(x=0, y=0, z=0),
        up=dict(x=0, y=0, z=1),
    ),
    "frontal": dict(
        eye=dict(x=0.0, y=2.2, z=0.15),
        center=dict(x=0, y=0, z=0),
        up=dict(x=0, y=0, z=1),
    ),
    "lateral_derecha": dict(
        eye=dict(x=2.2, y=0.0, z=0.15),
        center=dict(x=0, y=0, z=0),
        up=dict(x=0, y=0, z=1),
    ),
    "oclusal_superior": dict(
        eye=dict(x=0.0, y=0.0, z=2.6),
        center=dict(x=0, y=0, z=0),
        up=dict(x=0, y=1, z=0),
    ),
}

# Orden de exportación
VIEWS_TO_EXPORT = [
    "isometrica",
    "frontal",
    "lateral_derecha",
    "oclusal_superior",
]


# =========================================================
# FUNCIONES
# =========================================================
def load_mesh_any(path: Path) -> trimesh.Trimesh:
    """
    Carga un OBJ. Si viene como Scene, concatena todas las geometrías.
    """
    obj = trimesh.load(path, process=False)

    if isinstance(obj, trimesh.Scene):
        geoms = []
        for g in obj.geometry.values():
            if isinstance(g, trimesh.Trimesh):
                geoms.append(g)
        if len(geoms) == 0:
            raise ValueError(f"No se encontraron geometrías tipo Trimesh en: {path}")
        mesh = trimesh.util.concatenate(geoms)
    elif isinstance(obj, trimesh.Trimesh):
        mesh = obj
    else:
        raise ValueError(f"Tipo no soportado al cargar {path}: {type(obj)}")

    if mesh.vertices is None or len(mesh.vertices) == 0:
        raise ValueError(f"La malla no contiene vértices válidos: {path}")
    if mesh.faces is None or len(mesh.faces) == 0:
        raise ValueError(f"La malla no contiene caras válidas: {path}")

    return mesh


def get_case_paths():
    if INTERMEDIATE_MODE.lower() == "median":
        inter_path = MEDIAN_PATH
        inter_label = "Caso intermedio (mediana)"
    elif INTERMEDIATE_MODE.lower() == "mean":
        inter_path = MEAN_PATH
        inter_label = "Caso intermedio (promedio)"
    else:
        raise ValueError("INTERMEDIATE_MODE debe ser 'median' o 'mean'.")

    return [
        ("Mejor caso", BEST_PATH),
        (inter_label, inter_path),
        ("Peor caso", WORST_PATH),
    ]


def compute_global_ranges(meshes, pad=2.0):
    if MANUAL_RANGES is not None:
        return MANUAL_RANGES

    mins = np.vstack([m.bounds[0] for m in meshes]).min(axis=0)
    maxs = np.vstack([m.bounds[1] for m in meshes]).max(axis=0)

    return {
        "x": [float(mins[0] - pad), float(maxs[0] + pad)],
        "y": [float(mins[1] - pad), float(maxs[1] + pad)],
        "z": [float(mins[2] - pad), float(maxs[2] + pad)],
    }


def mesh_to_trace(mesh: trimesh.Trimesh, color: str, name: str):
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
        opacity=1.0,
        flatshading=False,
        showscale=False,
        name=name,
        lighting=dict(
            ambient=0.5,
            diffuse=0.7,
            fresnel=0.1,
            roughness=0.5,
            specular=0.2,
        ),
        lightposition=dict(x=100, y=200, z=300),
        hoverinfo="skip",
    )


def make_scene_dict(ranges, camera):
    return dict(
        xaxis=dict(
            title="x",
            range=ranges["x"],
            showbackground=False,
            showgrid=False,
            zeroline=False,
            visible=False,
        ),
        yaxis=dict(
            title="y",
            range=ranges["y"],
            showbackground=False,
            showgrid=False,
            zeroline=False,
            visible=False,
        ),
        zaxis=dict(
            title="z",
            range=ranges["z"],
            showbackground=False,
            showgrid=False,
            zeroline=False,
            visible=False,
        ),
        aspectmode="data",
        camera=camera,
        bgcolor="white",
    )


def build_stacked_figure(selected_cases, camera, ranges, view_name="isometrica"):
    fig = make_subplots(
        rows=3,
        cols=1,
        specs=[[{"type": "scene"}], [{"type": "scene"}], [{"type": "scene"}]],
        vertical_spacing=0.03,
        subplot_titles=[label for label, _, _ in selected_cases],
    )

    for row_i, (label, path, mesh) in enumerate(selected_cases, start=1):
        base_label = "Caso intermedio" if "intermedio" in label.lower() else label
        color = CASE_COLORS.get(base_label, "lightsteelblue")
        trace = mesh_to_trace(mesh, color=color, name=label)
        fig.add_trace(trace, row=row_i, col=1)

    scene_cfg = make_scene_dict(ranges, camera)

    fig.update_layout(
        scene=scene_cfg,
        scene2=scene_cfg.copy(),
        scene3=scene_cfg.copy(),
        height=1800,
        width=1100,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=10, r=10, t=70, b=10),
        showlegend=False,
        title=dict(
            text=f"Casos representativos del análisis cualitativo — vista: {view_name}",
            x=0.5,
            xanchor="center",
        ),
    )

    # Etiquetas laterales más informativas
    annotations_extra = []
    for idx, (label, path, _) in enumerate(selected_cases, start=1):
        annotations_extra.append(
            dict(
                text=f"{label}<br><sup>{path.stem}</sup>",
                x=0.01,
                y=1.0 - (idx - 1) / 3.0 - 0.12,
                xref="paper",
                yref="paper",
                showarrow=False,
                align="left",
                font=dict(size=12),
            )
        )

    fig.update_layout(annotations=list(fig.layout.annotations) + annotations_extra)

    return fig


# =========================================================
# MAIN
# =========================================================
def main():
    case_paths = get_case_paths()

    selected_cases = []
    meshes = []

    for label, path in case_paths:
        mesh = load_mesh_any(path)
        selected_cases.append((label, path, mesh))
        meshes.append(mesh)

    ranges = compute_global_ranges(meshes, pad=AXIS_PAD)

    print("\n[INFO] Casos seleccionados:")
    for label, path, _ in selected_cases:
        print(f" - {label}: {path}")

    print("\n[INFO] Rangos espaciales compartidos:")
    print(ranges)

    for view_name in VIEWS_TO_EXPORT:
        camera = CAMERA_VIEWS[view_name]
        fig = build_stacked_figure(
            selected_cases=selected_cases,
            camera=camera,
            ranges=ranges,
            view_name=view_name,
        )

        suffix = INTERMEDIATE_MODE.lower()
        html_path = OUT_DIR / f"qualitative_cases_{suffix}_{view_name}.html"
        fig.write_html(str(html_path), include_plotlyjs="cdn")
        print(f"[OK] HTML guardado en: {html_path}")

        if EXPORT_PNG:
            png_path = OUT_DIR / f"qualitative_cases_{suffix}_{view_name}.png"
            try:
                fig.write_image(str(png_path), scale=2)
                print(f"[OK] PNG guardado en: {png_path}")
            except Exception as e:
                print(f"[WARN] No se pudo exportar PNG ({png_path}): {e}")

    print("\n[DONE] Finalizado.")


if __name__ == "__main__":
    main()