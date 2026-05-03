#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compute_metrics_summary.py

Script para resumir métricas obtenidas en múltiples corridas experimentales
por distintas semillas.

Sirve para calcular, de forma automática:

    - media
    - desviación estándar
    - mínimo
    - máximo
    - número de corridas válidas

sobre métricas como:

    loss
    acc_all
    acc_no_bg
    f1_macro
    iou_macro
    d21_acc
    d21_f1
    d21_iou
    d21_bin_acc_all
    pred_bg_frac
    d11_acc, d11_f1, d11_iou
    d22_acc, d22_f1, d22_iou
    etc.

El script soporta dos fuentes principales:

1) Archivos finales de test:
   - test_metrics.json
   - test_metrics_filtered.json

   Estos son los más importantes para reportar resultados finales en tesis/paper.

2) Archivo metrics_epoch.csv:
   - permite resumir métricas de entrenamiento/validación por época.
   - se puede usar la última época o la mejor época según val d21_f1.

Estructura esperada de carpetas:

    BASE_DIR/
      splitseed1_trainseed42/
        metrics_epoch.csv
        test_metrics.json
        test_metrics_filtered.json

      splitseed7_trainseed42/
        metrics_epoch.csv
        test_metrics.json
        test_metrics_filtered.json

      splitseed21_trainseed42/
        ...

Ejemplo para PointNet Classic:

    BASE_DIR =
    /home/htaucare/Tesis_Amaro/outputs/pointnet_classic/stability_dataset_seeds_v2/exp03_best_config

Ejemplo de uso recomendado para resultados finales filtrados:

    python scripts_last_version/compute_metrics_summary.py \\
      --base_dir /home/htaucare/Tesis_Amaro/outputs/pointnet_classic/stability_dataset_seeds_v2/exp03_best_config \\
      --mode test_filtered_json \\
      --out_csv summary_test_filtered.csv

Ejemplo de uso para resultados finales sin filtrar:

    python scripts_last_version/compute_metrics_summary.py \\
      --base_dir /home/htaucare/Tesis_Amaro/outputs/pointnet_classic/stability_dataset_seeds_v2/exp03_best_config \\
      --mode test_json \\
      --out_csv summary_test.csv

Ejemplo usando metrics_epoch.csv y seleccionando la mejor época según val d21_f1:

    python scripts_last_version/compute_metrics_summary.py \\
      --base_dir /home/htaucare/Tesis_Amaro/outputs/pointnet_classic/stability_dataset_seeds_v2/exp03_best_config \\
      --mode best_val_d21_f1 \\
      --out_csv summary_best_epoch.csv

Ejemplo usando la última época disponible:

    python scripts_last_version/compute_metrics_summary.py \\
      --base_dir /home/htaucare/Tesis_Amaro/outputs/pointnet_classic/stability_dataset_seeds_v2/exp03_best_config \\
      --mode last_epoch \\
      --out_csv summary_last_epoch.csv

Nota metodológica:
    Para la tabla principal de tesis se recomienda usar test_metrics.json
    o test_metrics_filtered.json. El archivo metrics_epoch.csv es más útil
    para análisis de entrenamiento, convergencia y validación.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def summarize_numeric(df: pd.DataFrame, group_cols):
    """
    Calcula estadísticos descriptivos para todas las columnas numéricas
    de un DataFrame.

    Parámetros
    ----------
    df : pd.DataFrame
        Tabla con métricas de múltiples corridas.
    group_cols : list[str]
        Columnas por las cuales agrupar antes de resumir.
        Ejemplos:
            ["source"] para test_metrics.json
            ["split"] para metrics_epoch.csv

    Retorna
    -------
    pd.DataFrame
        Tabla larga con columnas:
            group_cols + metric + mean + std + min + max + n
    """

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    rows = []

    for keys, g in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)

        for col in numeric_cols:
            vals = g[col].dropna().to_numpy(dtype=float)

            if len(vals) == 0:
                continue

            row = {c: k for c, k in zip(group_cols, keys)}
            row.update({
                "metric": col,
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "min": float(vals.min()),
                "max": float(vals.max()),
                "n": int(len(vals)),
            })

            rows.append(row)

    return pd.DataFrame(rows)


def read_test_json(run_dir: Path, filename: str):
    """
    Lee un archivo test_metrics.json o test_metrics_filtered.json
    y conserva solo valores numéricos.

    Parámetros
    ----------
    run_dir : Path
        Carpeta de una corrida individual.
    filename : str
        Nombre del archivo JSON.

    Retorna
    -------
    dict | None
        Diccionario con métricas numéricas o None si el archivo no existe.
    """

    f = run_dir / filename

    if not f.exists():
        print(f"[WARN] Missing: {f}")
        return None

    with open(f, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    row = {
        "run": run_dir.name,
        "source": filename,
    }

    for k, v in data.items():
        if isinstance(v, (int, float)):
            row[k] = v

    return row


def read_metrics_epoch(run_dir: Path, mode: str):
    """
    Lee metrics_epoch.csv y selecciona las filas relevantes según el modo.

    Modos soportados
    ----------------
    last_epoch:
        Selecciona la última época disponible para cada split.

    best_val_d21_f1:
        Busca la época con mayor d21_f1 en split=val y luego selecciona
        todas las filas de esa época, normalmente train y val.

    Parámetros
    ----------
    run_dir : Path
        Carpeta de una corrida individual.
    mode : str
        "last_epoch" o "best_val_d21_f1"

    Retorna
    -------
    pd.DataFrame | None
        DataFrame filtrado o None si no existe el CSV.
    """

    f = run_dir / "metrics_epoch.csv"

    if not f.exists():
        print(f"[WARN] Missing: {f}")
        return None

    df = pd.read_csv(f)
    df["run"] = run_dir.name

    if "epoch" not in df.columns:
        print(f"[WARN] No column 'epoch' in {f}")
        return None

    if "split" not in df.columns:
        print(f"[WARN] No column 'split' in {f}")
        return None

    if mode == "last_epoch":
        # Última fila disponible para cada split en cada run.
        df_sel = (
            df.sort_values(["split", "epoch"])
              .groupby(["run", "split"], as_index=False)
              .tail(1)
        )

        return df_sel

    if mode == "best_val_d21_f1":
        if "d21_f1" not in df.columns:
            print(f"[WARN] No column 'd21_f1' in {f}")
            return None

        val = df[df["split"] == "val"].copy()

        if val.empty:
            print(f"[WARN] No validation rows in {f}")
            return None

        best_epoch = int(val.loc[val["d21_f1"].idxmax(), "epoch"])

        df_sel = df[df["epoch"] == best_epoch].copy()
        df_sel["best_epoch_selected"] = best_epoch

        return df_sel

    raise ValueError(f"Modo no soportado para metrics_epoch.csv: {mode}")


def main():
    parser = argparse.ArgumentParser(
        description="Resume métricas de múltiples corridas experimentales."
    )

    parser.add_argument(
        "--base_dir",
        required=True,
        help="Carpeta que contiene las corridas, por ejemplo splitseed*_trainseed42."
    )

    parser.add_argument(
        "--pattern",
        default="splitseed*_trainseed42",
        help="Patrón de carpetas de corridas dentro de base_dir."
    )

    parser.add_argument(
        "--mode",
        choices=[
            "test_json",
            "test_filtered_json",
            "last_epoch",
            "best_val_d21_f1",
        ],
        default="test_filtered_json",
        help=(
            "Fuente/modo de resumen. "
            "test_json usa test_metrics.json; "
            "test_filtered_json usa test_metrics_filtered.json; "
            "last_epoch usa la última época de metrics_epoch.csv; "
            "best_val_d21_f1 usa la época con mayor val d21_f1."
        ),
    )

    parser.add_argument(
        "--out_csv",
        default="summary_metrics.csv",
        help="Nombre del CSV de salida, guardado dentro de base_dir."
    )

    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    run_dirs = sorted(base_dir.glob(args.pattern))

    print(f"[INFO] Base dir: {base_dir}")
    print(f"[INFO] Pattern: {args.pattern}")
    print(f"[INFO] Mode: {args.mode}")
    print(f"[INFO] Found runs: {len(run_dirs)}")

    if not run_dirs:
        raise RuntimeError(
            f"No se encontraron carpetas con patrón {args.pattern} en {base_dir}"
        )

    all_rows = []

    for run_dir in run_dirs:
        if args.mode == "test_json":
            row = read_test_json(run_dir, "test_metrics.json")
            if row is not None:
                all_rows.append(row)

        elif args.mode == "test_filtered_json":
            row = read_test_json(run_dir, "test_metrics_filtered.json")
            if row is not None:
                all_rows.append(row)

        elif args.mode in ["last_epoch", "best_val_d21_f1"]:
            df_sel = read_metrics_epoch(run_dir, args.mode)
            if df_sel is not None:
                all_rows.append(df_sel)

        else:
            raise ValueError(f"Modo no reconocido: {args.mode}")

    if not all_rows:
        raise RuntimeError("No se encontraron datos válidos para resumir.")

    if args.mode in ["test_json", "test_filtered_json"]:
        df_all = pd.DataFrame(all_rows)
        summary = summarize_numeric(df_all, ["source"])

    else:
        df_all = pd.concat(all_rows, ignore_index=True)
        summary = summarize_numeric(df_all, ["split"])

    out_path = base_dir / args.out_csv
    summary.to_csv(out_path, index=False)

    print("\n===== SUMMARY =====\n")
    print(summary.to_string(index=False))

    print(f"\n[OK] Summary guardado en: {out_path}")


if __name__ == "__main__":
    main()