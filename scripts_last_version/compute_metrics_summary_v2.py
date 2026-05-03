#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compute_metrics_summary_v2.py

Resumen robusto de métricas para múltiples corridas experimentales.

Esta versión corrige el problema observado cuando algunos JSON contienen
campos no escalares, por ejemplo:

    "ignored_rows": [3, 20, 45]

En esos casos, el script ignora automáticamente listas, diccionarios, strings
no numéricos u otros objetos no escalares, y conserva solo métricas numéricas.

Sirve para calcular, por múltiples semillas:

    - media
    - desviación estándar
    - mínimo
    - máximo
    - número de corridas válidas

Fuentes soportadas:

1) test_metrics.json
2) test_metrics_filtered.json
3) metrics_epoch.csv usando:
   - última época
   - mejor época según val d21_f1

Ejemplos de uso
---------------

Test filtrado, recomendado para tabla principal de tesis/paper:

    python scripts_last_version/compute_metrics_summary_v2.py \
      --base_dir /home/htaucare/Tesis_Amaro/outputs/pointnetpp/stability_real_splits_v1/exp04_best_config \
      --pattern "splitseed*_trainseed42" \
      --mode test_filtered_json \
      --out_csv summary_pointnetpp_test_filtered.csv

Test sin filtrar:

    python scripts_last_version/compute_metrics_summary_v2.py \
      --base_dir /ruta/al/experimento \
      --pattern "splitseed*_trainseed42" \
      --mode test_json \
      --out_csv summary_test.csv

Train/val en la mejor época según val d21_f1:

    python scripts_last_version/compute_metrics_summary_v2.py \
      --base_dir /ruta/al/experimento \
      --pattern "splitseed*_trainseed42" \
      --mode best_val_d21_f1 \
      --out_csv summary_best_epoch_train_val.csv

Train/val en la última época:

    python scripts_last_version/compute_metrics_summary_v2.py \
      --base_dir /ruta/al/experimento \
      --pattern "splitseed*_trainseed42" \
      --mode last_epoch \
      --out_csv summary_last_epoch_train_val.csv

Notas metodológicas
-------------------

- Para resultados finales, usar preferentemente test_metrics_filtered.json.
- Para análisis de aprendizaje/convergencia, usar metrics_epoch.csv.
- metrics_epoch.csv normalmente NO está filtrado.
- test_metrics_filtered.json sí corresponde al test filtrado.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


def is_numeric_scalar(value: Any) -> bool:
    """
    Retorna True solo si value es un escalar numérico válido.

    Acepta:
        - int
        - float
        - np.integer
        - np.floating

    Rechaza:
        - bool
        - list
        - dict
        - tuple
        - ndarray
        - strings
        - NaN / inf
    """
    if isinstance(value, bool):
        return False

    if isinstance(value, (int, float, np.integer, np.floating)):
        value_f = float(value)
        return math.isfinite(value_f)

    return False


def to_float(value: Any) -> float:
    """
    Convierte un escalar numérico a float.
    """
    return float(value)


def summarize_numeric(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """
    Calcula estadísticos descriptivos para todas las columnas numéricas
    de un DataFrame agrupado por group_cols.

    Retorna una tabla larga con:
        group_cols + metric + mean + std + min + max + n
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Evitar resumir columnas auxiliares que no son métricas reales.
    excluded = {"seed_from_name"}
    numeric_cols = [c for c in numeric_cols if c not in excluded]

    rows = []

    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        for col in numeric_cols:
            vals = g[col].dropna().to_numpy(dtype=float)

            vals = vals[np.isfinite(vals)]
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


def extract_seed_from_run_name(run_name: str) -> Optional[int]:
    """
    Intenta extraer seed desde nombres tipo:
        splitseed42_trainseed42
        data_seed7_trainseed42

    Retorna None si no encuentra patrón.
    """
    import re

    patterns = [
        r"splitseed(\d+)",
        r"data_seed(\d+)",
        r"seed(\d+)",
    ]

    for pat in patterns:
        m = re.search(pat, run_name)
        if m:
            return int(m.group(1))

    return None


def read_test_json(run_dir: Path, filename: str, verbose: bool = True) -> Optional[Dict[str, Any]]:
    """
    Lee test_metrics.json o test_metrics_filtered.json y conserva únicamente
    valores numéricos escalares.

    Ignora campos como:
        ignored_rows: list
        cualquier dict/list/tuple/ndarray/string

    Retorna un dict plano listo para pandas.
    """
    f = run_dir / filename

    if not f.exists():
        print(f"[WARN] Missing: {f}")
        return None

    with open(f, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    row: Dict[str, Any] = {
        "run": run_dir.name,
        "source": filename,
    }

    seed = extract_seed_from_run_name(run_dir.name)
    if seed is not None:
        row["seed_from_name"] = seed

    ignored_keys = []

    for k, v in data.items():
        if is_numeric_scalar(v):
            row[k] = to_float(v)
        else:
            ignored_keys.append(k)

    if verbose and ignored_keys:
        print(f"[INFO] {run_dir.name}: ignored non-scalar fields from {filename}: {ignored_keys}")

    return row


def read_metrics_epoch(run_dir: Path, mode: str) -> Optional[pd.DataFrame]:
    """
    Lee metrics_epoch.csv y selecciona las filas relevantes según el modo.

    Modos:
        last_epoch:
            última época disponible por split.

        best_val_d21_f1:
            mejor época según d21_f1 en split=val, y extrae train/val
            correspondientes a esa misma época.
    """
    f = run_dir / "metrics_epoch.csv"

    if not f.exists():
        print(f"[WARN] Missing: {f}")
        return None

    try:
        df = pd.read_csv(f)
    except Exception as e:
        print(f"[WARN] Could not read {f}: {e}")
        return None

    df["run"] = run_dir.name

    seed = extract_seed_from_run_name(run_dir.name)
    if seed is not None:
        df["seed_from_name"] = seed

    if "epoch" not in df.columns:
        print(f"[WARN] No column 'epoch' in {f}")
        return None

    if "split" not in df.columns:
        print(f"[WARN] No column 'split' in {f}")
        return None

    if mode == "last_epoch":
        df_sel = (
            df.sort_values(["run", "split", "epoch"])
              .groupby(["run", "split"], as_index=False)
              .tail(1)
              .copy()
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

        val = val[np.isfinite(pd.to_numeric(val["d21_f1"], errors="coerce"))]
        if val.empty:
            print(f"[WARN] Validation d21_f1 has no finite values in {f}")
            return None

        best_epoch = int(val.loc[val["d21_f1"].idxmax(), "epoch"])

        df_sel = df[df["epoch"] == best_epoch].copy()
        df_sel["best_epoch_selected"] = best_epoch

        return df_sel

    raise ValueError(f"Modo no soportado para metrics_epoch.csv: {mode}")


def sort_run_dirs(run_dirs: Iterable[Path]) -> List[Path]:
    """
    Ordena carpetas por seed si es posible, si no por nombre.
    """
    def key_fn(p: Path):
        seed = extract_seed_from_run_name(p.name)
        return (0, seed) if seed is not None else (1, p.name)

    return sorted(run_dirs, key=key_fn)


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
        help="Nombre del CSV de resumen, guardado dentro de base_dir."
    )

    parser.add_argument(
        "--out_raw_csv",
        default=None,
        help=(
            "Nombre opcional del CSV con las filas crudas usadas para el resumen. "
            "Si no se especifica, se genera automáticamente como raw_<out_csv>."
        ),
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="No imprimir campos no escalares ignorados."
    )

    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    run_dirs = sort_run_dirs(base_dir.glob(args.pattern))

    print(f"[INFO] Base dir: {base_dir}")
    print(f"[INFO] Pattern: {args.pattern}")
    print(f"[INFO] Mode: {args.mode}")
    print(f"[INFO] Found runs: {len(run_dirs)}")

    if not run_dirs:
        raise RuntimeError(
            f"No se encontraron carpetas con patrón {args.pattern} en {base_dir}"
        )

    all_rows: List[Any] = []

    for run_dir in run_dirs:
        if args.mode == "test_json":
            row = read_test_json(run_dir, "test_metrics.json", verbose=not args.quiet)
            if row is not None:
                all_rows.append(row)

        elif args.mode == "test_filtered_json":
            row = read_test_json(run_dir, "test_metrics_filtered.json", verbose=not args.quiet)
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
        # from_records evita varios problemas con dicts heterogéneos.
        df_all = pd.DataFrame.from_records(all_rows)
        summary = summarize_numeric(df_all, ["source"])
    else:
        df_all = pd.concat(all_rows, ignore_index=True)
        summary = summarize_numeric(df_all, ["split"])

    out_path = base_dir / args.out_csv

    if args.out_raw_csv is None:
        out_raw_name = "raw_" + args.out_csv
    else:
        out_raw_name = args.out_raw_csv

    raw_path = base_dir / out_raw_name

    df_all.to_csv(raw_path, index=False)
    summary.to_csv(out_path, index=False)

    print("\n===== RAW DATA USED =====\n")
    print(df_all.to_string(index=False))

    print("\n===== SUMMARY =====\n")
    print(summary.to_string(index=False))

    print(f"\n[OK] Raw data guardado en: {raw_path}")
    print(f"[OK] Summary guardado en: {out_path}")


if __name__ == "__main__":
    main()
