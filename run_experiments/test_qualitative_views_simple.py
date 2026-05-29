#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_qualitative_views_simple.py

Objetivo:
- Probar 4 cámaras distintas para una misma visualización cualitativa.
- Usar un mismo caso y un mismo modelo.
- Generar HTMLs separados para comparar cuál vista comunica mejor.

Uso esperado:
1) Ajustar SCRIPT_BASE con la ruta real del script v17 que ya usas.
2) Ajustar CASE_DIR y MODEL.
3) Ejecutar y revisar los 4 HTML resultantes.

Importante:
- Este script NO reimplementa tu visualizador.
- Solo llama varias veces al script base con configuraciones distintas.
"""

import subprocess
from pathlib import Path

# ============================================================
# AJUSTA ESTAS RUTAS
# ============================================================

SCRIPT_BASE = Path("/home/htaucare/Tesis_Amaro/run_experiments/make_paper_figure_models_selectable_v17_dental_icp_paper_final.py")

CASE_DIR = Path("/home/htaucare/Tesis_Amaro/case_comparisons/QUALI_CASES_V17/median_row089_015RHV4X_upper")

MODEL = "dgcnn"   # o pointnetpp, pointnet, pointnettransformer

OUT_BASE = CASE_DIR / "camera_test_views"

OUT_BASE.mkdir(parents=True, exist_ok=True)

# ============================================================
# ARGUMENTOS COMUNES
# ============================================================

COMMON_ARGS = [
    "python", str(SCRIPT_BASE),
    "--case_dir", str(CASE_DIR),
    "--models", MODEL,
    "--d21_class", "8",
    "--neighbor_teeth", "d11:1,d22:9",
    "--align_mode", "dental_pca_icp",
    "--mesh_top_q", "0.45",
    "--icp_iter", "45",
    "--icp_max_corr", "0.08",
    "--icp_trim_q", "0.85",
    "--icp_focus", "foreground",
    "--show_classes", "all",
    "--point_size", "8.8",
    "--plotly_point_size", "4.2",
    "--mesh_alpha", "0.07",
    "--plotly_mesh_opacity", "0.05",
    "--force",
]

# ============================================================
# VISTAS DE PRUEBA
# OJO:
# Esto asume que el script base soporta cámara por JSON externo o alguna opción similar.
# Si no la soporta, abajo te doy la alternativa.
# ============================================================

CAMERAS = {
    "isometrica": {
        "eye": {"x": 1.35, "y": 1.35, "z": 0.75},
        "center": {"x": 0.0, "y": 0.0, "z": 0.0},
        "up": {"x": 0.0, "y": 0.0, "z": 1.0},
    },
    "oclusal_oblicua": {
        "eye": {"x": 0.20, "y": 0.20, "z": 2.30},
        "center": {"x": 0.0, "y": 0.0, "z": 0.0},
        "up": {"x": 0.0, "y": 1.0, "z": 0.0},
    },
    "frontal_oblicua": {
        "eye": {"x": 0.0, "y": 1.95, "z": 0.35},
        "center": {"x": 0.0, "y": 0.0, "z": 0.10},
        "up": {"x": 0.0, "y": 0.0, "z": 1.0},
    },
    "lateral_derecha": {
        "eye": {"x": 1.95, "y": 0.0, "z": 0.20},
        "center": {"x": 0.0, "y": 0.0, "z": 0.0},
        "up": {"x": 0.0, "y": 0.0, "z": 1.0},
    },
}

def save_camera_json(view_name, cam_dict):
    import json
    path = OUT_BASE / f"camera_{view_name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cam_dict, f, indent=2)
    return path

def run_view(view_name, cam):
    out_png = OUT_BASE / f"{MODEL}_{view_name}.png"
    out_html = OUT_BASE / f"{MODEL}_{view_name}.html"
    out_individual_dir = OUT_BASE / f"{MODEL}_{view_name}_individual"

    cam_json = save_camera_json(view_name, cam)

    cmd = COMMON_ARGS + [
        "--out_png", str(out_png),
        "--out_html", str(out_html),
        "--out_individual_dir", str(out_individual_dir),
        "--camera_json", str(cam_json),   # solo si tu script lo soporta
    ]

    print("\n==============================")
    print(f"[RUN] Vista: {view_name}")
    print(" ".join(cmd))
    print("==============================\n")

    subprocess.run(cmd, check=True)

def main():
    for view_name, cam in CAMERAS.items():
        run_view(view_name, cam)

    print("\n[DONE] Se generaron las 4 vistas de prueba en:")
    print(OUT_BASE)

if __name__ == "__main__":
    main()