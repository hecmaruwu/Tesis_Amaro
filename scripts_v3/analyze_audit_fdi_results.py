#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_audit_fdi_results.py
----------------------------------------------------------
Analiza el archivo audit_fdi_detailed.json generado por
audit_fdi_teeth_dataset.py para:

  1. Detectar el sistema de numeración dental (FDI o remapeado)
  2. Mostrar etiquetas únicas detectadas por arcada
  3. Identificar pacientes con todas las piezas en ambas arcadas
  4. Guardar los resultados filtrados (JSON + CSV)

Ejemplo:
  python scripts_v3/analyze_audit_fdi_results.py \
      --input_json logs/audit_fdi_v1/audit_fdi_detailed.json \
      --out_dir logs/audit_fdi_analysis_v1
----------------------------------------------------------
Autor: ChatGPT (GPT-5)
"""

import json
import csv
import argparse
from pathlib import Path
import numpy as np

# ==============================================================
# === CONFIGURACIONES DE LISTAS ================================
# ==============================================================

FDI_UPPER = list(range(11, 19)) + list(range(21, 29))  # 11–18, 21–28
FDI_LOWER = list(range(31, 39)) + list(range(41, 49))  # 31–38, 41–48
REMAP_UPPER = list(range(1, 17))                       # 1–16
REMAP_LOWER = list(range(17, 33))                      # 17–32


# ==============================================================
# === FUNCIONES ================================================
# ==============================================================

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def detect_label_system(data: dict) -> str:
    """Detecta si las etiquetas están en formato FDI o remapeado."""
    all_labels = set()
    for pid, info in data.items():
        all_labels.update(info.get("upper_labels", []))
        all_labels.update(info.get("lower_labels", []))
    if not all_labels:
        return "UNKNOWN"
    max_lbl = max(all_labels)
    return "FDI (11–48)" if max_lbl > 32 else "Remapeado (1–32)"


def get_expected_sets(label_system: str):
    """Retorna los sets esperados según el sistema."""
    if "FDI" in label_system:
        return set(FDI_UPPER), set(FDI_LOWER)
    elif "Remapeado" in label_system:
        return set(REMAP_UPPER), set(REMAP_LOWER)
    else:
        raise ValueError("Sistema de etiquetas desconocido.")


def summarize_labels(data: dict):
    """Obtiene las etiquetas únicas detectadas por arcada."""
    upper_labels = set()
    lower_labels = set()
    for pid, info in data.items():
        upper_labels.update(info.get("upper_labels", []))
        lower_labels.update(info.get("lower_labels", []))
    return sorted(upper_labels), sorted(lower_labels)


def find_complete_patients(data: dict, expected_upper, expected_lower):
    """Devuelve lista de pacientes con todas las piezas en ambas arcadas."""
    complete = []
    for pid, info in data.items():
        upper = set(info.get("upper_labels", []))
        lower = set(info.get("lower_labels", []))
        if expected_upper.issubset(upper) and expected_lower.issubset(lower):
            complete.append(pid)
    return complete


def save_json(obj, path: Path):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_csv(patients, out_csv: Path):
    ensure_dir(out_csv.parent)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["patient_id"])
        for pid in patients:
            w.writerow([pid])


# ==============================================================
# === MAIN =====================================================
# ==============================================================

def main():
    parser = argparse.ArgumentParser(description="Análisis del archivo audit_fdi_detailed.json")
    parser.add_argument("--input_json", required=True, help="Ruta al archivo audit_fdi_detailed.json")
    parser.add_argument("--out_dir", required=True, help="Carpeta de salida para reportes")
    args = parser.parse_args()

    input_json = Path(args.input_json)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # 1. Cargar JSON
    print(f"\n[INFO] Cargando archivo: {input_json}")
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[OK] Pacientes cargados: {len(data)}")

    # 2. Detectar sistema de etiquetas
    label_system = detect_label_system(data)
    expected_upper, expected_lower = get_expected_sets(label_system)
    print(f"[INFO] Sistema de numeración detectado: {label_system}")

    # 3. Resumen de etiquetas
    upper_labels, lower_labels = summarize_labels(data)
    print("\nEtiquetas detectadas:")
    print(f" - Arcada superior (upper): {upper_labels}")
    print(f" - Arcada inferior (lower): {lower_labels}")

    # 4. Buscar pacientes completos
    complete_patients = find_complete_patients(data, expected_upper, expected_lower)
    n_complete = len(complete_patients)
    total = len(data)
    pct_complete = (n_complete / max(total, 1)) * 100

    print("\n🦷  PACIENTES COMPLETOS")
    print("──────────────────────────────────────────")
    print(f"Total de pacientes analizados : {total}")
    print(f"Pacientes con ambas arcadas completas : {n_complete}  ({pct_complete:.2f}%)")

    if n_complete > 0:
        print("Ejemplos:", complete_patients[:10])
    else:
        print("⚠️  Ningún paciente tiene ambas arcadas con todas las piezas esperadas.")

    # 5. Guardar resultados
    save_json({"complete_patients": complete_patients}, out_dir / "complete_patients.json")
    save_csv(complete_patients, out_dir / "complete_patients.csv")

    print(f"\n✅ Resultados guardados en: {out_dir}\n")


if __name__ == "__main__":
    main()
