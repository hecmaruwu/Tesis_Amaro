#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_pairs_and_teeth.py
----------------------------------------------------------
Auditoría completa de pacientes con arcadas (upper/lower)
y verificación del número total de dientes presentes.

Genera tres reportes:
  1. JSON estructurado con info por paciente
  2. CSV resumen general (contadores y estadísticas)
  3. CSV detallado con flags por paciente
Y además muestra un resumen porcentual claro en consola.

Ejemplo:
  python scripts_v3/audit_pairs_and_teeth.py \
      --dataset_dir data/Teeth_3ds/processed_struct/8192 \
      --out_dir logs/audit_teeth \
      --expected_teeth 25
----------------------------------------------------------
Autor: ChatGPT (GPT-5)
"""

import os
import json
import csv
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

# ==============================================================
# === Utilidades ===============================================
# ==============================================================

def load_labels(path: Path) -> np.ndarray:
    """Carga un archivo labels.npy, devolviendo np.array vacío si no existe."""
    if not path.exists():
        return np.array([], dtype=np.int64)
    try:
        return np.load(path)
    except Exception as e:
        print(f"[WARN] Error cargando {path}: {e}")
        return np.array([], dtype=np.int64)


def summarize_arcada(arcada_dir: Path, expected_teeth: int) -> dict:
    """Devuelve estadísticas básicas de una arcada individual (upper/lower)."""
    labels_path = arcada_dir / "labels.npy"
    y = load_labels(labels_path)

    if y.size > 0:
        unique = np.unique(y)
        unique_list = unique.tolist()
    else:
        unique = np.array([], dtype=np.int64)
        unique_list = []

    return {
        "exists": arcada_dir.exists(),
        "labels_path": str(labels_path) if labels_path.exists() else None,
        "num_points": int(y.size),
        "unique_labels": unique_list,
        "num_teeth_present": int(len(unique[unique > 0])) if y.size > 0 else 0,
        "has_all_teeth": bool(set(range(1, expected_teeth + 1)).issubset(unique)),
    }


def ensure_dir(p: Path):
    """Crea directorio si no existe."""
    p.mkdir(parents=True, exist_ok=True)


# ==============================================================
# === Auditoría principal ======================================
# ==============================================================

def audit_dataset(root_dir: Path, expected_teeth: int):
    """Recorre todo el dataset y genera un dict por paciente."""
    upper_dir = root_dir / "upper"
    lower_dir = root_dir / "lower"

    if not (upper_dir.exists() or lower_dir.exists()):
        raise FileNotFoundError(f"No se encuentran subcarpetas upper/ y lower en {root_dir}")

    # Pacientes detectados
    upper_patients = {p.name for p in upper_dir.iterdir() if p.is_dir()}
    lower_patients = {p.name for p in lower_dir.iterdir() if p.is_dir()}
    all_patients = sorted(upper_patients | lower_patients)

    report = {}
    stats = {
        "total_patients": len(all_patients),
        "only_upper": 0,
        "only_lower": 0,
        "both_arcadas": 0,
        "complete_both": 0,  # ambas arcadas con todos los dientes
    }

    for pid in tqdm(all_patients, desc="Auditoría de pacientes"):
        up_dir = upper_dir / pid
        lo_dir = lower_dir / pid

        upper_info = summarize_arcada(up_dir, expected_teeth)
        lower_info = summarize_arcada(lo_dir, expected_teeth)

        has_upper = upper_info["exists"]
        has_lower = lower_info["exists"]

        both = has_upper and has_lower
        complete_upper = upper_info["has_all_teeth"]
        complete_lower = lower_info["has_all_teeth"]
        complete_both = both and complete_upper and complete_lower

        # actualiza contadores globales
        if both:
            stats["both_arcadas"] += 1
            if complete_both:
                stats["complete_both"] += 1
        elif has_upper:
            stats["only_upper"] += 1
        elif has_lower:
            stats["only_lower"] += 1

        report[pid] = {
            "upper": upper_info,
            "lower": lower_info,
            "has_upper": has_upper,
            "has_lower": has_lower,
            "both_arcadas": both,
            "complete_upper": complete_upper,
            "complete_lower": complete_lower,
            "complete_both": complete_both,
        }

    return report, stats


# ==============================================================
# === Guardado =================================================
# ==============================================================

def save_json(obj, path: Path):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_csv_summary(stats: dict, out_csv: Path):
    ensure_dir(out_csv.parent)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in stats.items():
            w.writerow([k, v])


def save_csv_detailed(report: dict, out_csv: Path):
    ensure_dir(out_csv.parent)
    header = [
        "patient_id", "has_upper", "has_lower", "both_arcadas",
        "num_teeth_upper", "num_teeth_lower",
        "complete_upper", "complete_lower", "complete_both"
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for pid, info in sorted(report.items()):
            u = info["upper"]
            l = info["lower"]
            w.writerow([
                pid,
                info["has_upper"], info["has_lower"], info["both_arcadas"],
                u["num_teeth_present"], l["num_teeth_present"],
                info["complete_upper"], info["complete_lower"], info["complete_both"]
            ])


# ==============================================================
# === MAIN =====================================================
# ==============================================================

def main():
    parser = argparse.ArgumentParser(description="Auditoría de arcadas y dientes presentes por paciente")
    parser.add_argument("--dataset_dir", required=True, help="Carpeta raíz con upper/ y lower/")
    parser.add_argument("--out_dir", required=True, help="Carpeta de salida para reportes")
    parser.add_argument("--expected_teeth", type=int, default=25,
                        help="Número esperado de dientes por arcada (por defecto=25)")
    args = parser.parse_args()

    root_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    print(f"\n[INFO] Analizando dataset en: {root_dir}")
    print(f"[INFO] Dientes esperados por arcada: {args.expected_teeth}\n")

    report, stats = audit_dataset(root_dir, args.expected_teeth)

    # Guardar resultados
    save_json(report, out_dir / "audit_pairs_and_teeth.json")
    save_csv_summary(stats, out_dir / "audit_summary.csv")
    save_csv_detailed(report, out_dir / "audit_detailed.csv")

    # ==========================================================
    # === Impresión de resumen porcentual ======================
    # ==========================================================

    total = max(stats["total_patients"], 1)
    pct = lambda x: (x / total) * 100

    print("\n🦷  RESULTADOS GENERALES DE AUDITORÍA")
    print("──────────────────────────────────────────")
    print(f"Total de pacientes analizados        : {stats['total_patients']}")
    print(f"Solo arcada superior (upper)         : {stats['only_upper']}  ({pct(stats['only_upper']):.2f}%)")
    print(f"Solo arcada inferior (lower)         : {stats['only_lower']}  ({pct(stats['only_lower']):.2f}%)")
    print(f"Ambas arcadas presentes              : {stats['both_arcadas']}  ({pct(stats['both_arcadas']):.2f}%)")
    print(f"Ambas arcadas con todos los dientes  : {stats['complete_both']}  ({pct(stats['complete_both']):.2f}%)")

    print("\n✅ Archivos generados:")
    print(f"  - JSON detallado : {out_dir}/audit_pairs_and_teeth.json")
    print(f"  - CSV resumen    : {out_dir}/audit_summary.csv")
    print(f"  - CSV detallado  : {out_dir}/audit_detailed.csv\n")


if __name__ == "__main__":
    main()
