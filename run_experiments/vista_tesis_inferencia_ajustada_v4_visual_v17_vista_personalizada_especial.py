#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
vista_tesis_inferencia_ajustada_v4_visual_v17_vista_personalizada_especial.py

Versión especial para la tesis:
- Reutiliza el script base:
  vista_tesis_inferencia_ajustada_v4_visual_v17.py
- NO modifica las cámaras globales del script base.
- Aplica cámaras personalizadas SOLO al worst case / Peor caso.
- Está pensada para regenerar el peor caso, por ejemplo:
  --cases worst --models dgcnn pointnettransformer --views oclusal_paper isometrica_ajustada

Motivo:
En el worst case, las cámaras generales del script base no daban un zoom/ángulo
adecuado. Estas cámaras fueron calibradas manualmente desde el HTML Plotly.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict


# ============================================================
# CÁMARAS PERSONALIZADAS SOLO PARA WORST CASE
# ============================================================

WORST_CASE_CAMERA_PRESETS: Dict[str, Dict[str, Any]] = {
    "oclusal_paper": {
        "up": {
            "x": -0.02548971620697424,
            "y": 0.9873363543028241,
            "z": -0.15657968527141758,
        },
        "center": {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
        },
        "eye": {
            "x": 0.29942526767062777,
            "y": 0.29942526767062905,
            "z": 1.839326644262434,
        },
        "projection": {
            "type": "perspective",
        },
    },
    "isometrica_ajustada": {
        "up": {
            "x": -0.32909239074815366,
            "y": -0.37399735203853346,
            "z": 0.8670779544076934,
        },
        "center": {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
        },
        "eye": {
            "x": 0.777566816878286,
            "y": 0.9819246971309226,
            "z": 0.7186534453949692,
        },
        "projection": {
            "type": "perspective",
        },
    },
}


def _load_base_module():
    """
    Carga el script base desde la misma carpeta run_experiments/.
    Esto permite mantener intacto el script original y aplicar solo un parche
    visual para el worst case.
    """
    this_file = Path(__file__).resolve()
    base_path = this_file.with_name("vista_tesis_inferencia_ajustada_v4_visual_v17.py")

    if not base_path.exists():
        raise FileNotFoundError(
            f"No encontré el script base esperado:\n{base_path}\n\n"
            "Guarda este archivo en la misma carpeta que "
            "vista_tesis_inferencia_ajustada_v4_visual_v17.py."
        )

    spec = importlib.util.spec_from_file_location("vista_tesis_base_v17", str(base_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"No pude cargar el script base desde: {base_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["vista_tesis_base_v17"] = module
    spec.loader.exec_module(module)
    return module


def _is_worst_case(case_role: str) -> bool:
    """
    Detecta si el caso corresponde al peor caso.
    El script base usa etiquetas como 'Peor caso'.
    """
    role = str(case_role or "").strip().lower()
    return ("peor" in role) or ("worst" in role)


def _patch_build_panel_figure(base_module):
    """
    Parchea build_panel_figure para que, si el caso es worst/peor,
    reemplace la cámara recibida por la cámara personalizada correspondiente.
    """
    original_build_panel_figure = base_module.build_panel_figure

    def build_panel_figure_worst_camera(*args, **kwargs):
        # Firma original:
        # build_panel_figure(
        #   xyz, gt, pred, model, patient_id, case_role, view_name, camera,
        #   metrics, d21_class, ...
        # )
        case_role = kwargs.get("case_role", None)
        view_name = kwargs.get("view_name", None)

        if case_role is None and len(args) >= 6:
            case_role = args[5]
        if view_name is None and len(args) >= 7:
            view_name = args[6]

        view_key = str(view_name or "").strip().lower()

        if _is_worst_case(str(case_role)) and view_key in WORST_CASE_CAMERA_PRESETS:
            custom_camera = copy.deepcopy(WORST_CASE_CAMERA_PRESETS[view_key])

            if "camera" in kwargs:
                kwargs["camera"] = custom_camera
            elif len(args) >= 8:
                args = list(args)
                args[7] = custom_camera
                args = tuple(args)

        return original_build_panel_figure(*args, **kwargs)

    base_module.build_panel_figure = build_panel_figure_worst_camera


def _patch_save_json_for_camera_source(base_module):
    """
    Parche opcional: cuando el script guarda un summary JSON de un worst case,
    deja registrado que se usó cámara personalizada.

    Esto no altera las figuras; solo mejora la trazabilidad del output.
    """
    original_save_json = base_module.save_json

    def save_json_with_camera_source(obj, path):
        try:
            if isinstance(obj, dict):
                case_role = str(obj.get("case_role", "")).strip().lower()
                view_name = str(obj.get("view_name", "")).strip().lower()
                if (("peor" in case_role) or ("worst" in case_role)) and view_name in WORST_CASE_CAMERA_PRESETS:
                    obj = dict(obj)
                    obj["camera_source"] = "worst_case_personalizada"
                    obj["camera"] = copy.deepcopy(WORST_CASE_CAMERA_PRESETS[view_name])
        except Exception:
            pass

        return original_save_json(obj, path)

    base_module.save_json = save_json_with_camera_source


def main() -> None:
    base_module = _load_base_module()

    _patch_build_panel_figure(base_module)
    _patch_save_json_for_camera_source(base_module)

    print("=" * 90)
    print("[INFO] Script especial: vista personalizada SOLO para worst case")
    print("[INFO] Script base: vista_tesis_inferencia_ajustada_v4_visual_v17.py")
    print("[INFO] Cámaras personalizadas activas para: oclusal_paper, isometrica_ajustada")
    print("=" * 90)

    base_module.main()


if __name__ == "__main__":
    main()
