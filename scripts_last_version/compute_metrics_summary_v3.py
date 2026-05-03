#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compute_metrics_summary_v3.py

Resumen robusto de métricas para múltiples corridas experimentales.

Versión v3:
- Ignora completamente campos no escalares en JSON, por ejemplo:
    ignored_rows: [3, 20, 45]
- Convierte explícitamente a float solo valores numéricos finitos.
- Conserva columnas auxiliares tipo string:
    run, source
- Evita el bug de pandas:
    TypeError: Cannot convert numpy.ndarray to numpy.ndarray
- Guarda:
    1) CSV raw con las corridas usadas
    2) CSV summary con mean/std/min/max/n

Modos:
- test_filtered_json  -> lee test_metrics_filtered.json
- test_json           -> lee test_metrics.json
- best_val_d21_f1     -> lee metrics_epoch.csv y toma la mejor época según val d21_f1
- last_epoch          -> lee metrics_epoch.csv y toma la última época por split

Uso recomendado para tabla principal:
    --mode test_filtered_json
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


AUX_STRING_COLUMNS = {"run", "source", "split"}


def is_finite_numeric_scalar(x: Any) -> bool:
    """True solo para escalares numéricos finitos. Rechaza bool, listas, dicts, arrays y strings."""
    if isinstance(x, bool):
        return False

    if isinstance(x, (int, float, np.integer, np.floating)):
        try:
            return math.isfinite(float(x))
        except Exception:
            return False

    return False


def safe_float(x: Any) -> Optional[float]:
    """Convierte a float si es escalar numérico finito; si no, retorna None."""
    if is_finite_numeric_scalar(x):
        return float(x)
    return None


def extract_seed_from_run_name(run_name: str) -> Optional[int]:
    """Extrae seed desde nombres como splitseed42_trainseed42 o data_seed7_trainseed42."""
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


def sort_run_dirs(run_dirs):
    """Ordena carpetas por seed si se puede; si no, por nombre."""
    def key_fn(p: Path):
        seed = extract_seed_from_run_name(p.name)
        return (0, seed) if seed is not None else (1, p.name)
    return sorted(run_dirs, key=key_fn)


def clean_row_dict(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Limpia un diccionario antes de pasarlo a pandas.

    Conserva:
    - strings auxiliares: run, source, split
    - números escalares finitos convertidos a float

    Ignora:
    - list
    - dict
    - tuple
    - ndarray
    - objetos raros
    - strings no auxiliares
    - NaN / inf
    """
    clean: Dict[str, Any] = {}

    for k, v in row.items():
        k = str(k)

        if k in AUX_STRING_COLUMNS and isinstance(v, str):
            clean[k] = v
            continue

        fv = safe_float(v)
        if fv is not None:
            clean[k] = fv

    return clean


def read_test_json(run_dir: Path, filename: str, verbose: bool = True) -> Optional[Dict[str, Any]]:
    """
    Lee test_metrics.json o test_metrics_filtered.json.
    Conserva solo métricas escalares numéricas y columnas auxiliares.
    """
    f = run_dir / filename

    if not f.exists():
        print(f"[WARN] Missing: {f}")
        return None

    with open(f, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    raw_row: Dict[str, Any] = {
        "run": run_dir.name,
        "source": filename,
    }

    seed = extract_seed_from_run_name(run_dir.name)
    if seed is not None:
        raw_row["seed_from_name"] = seed

    ignored = []

    for k, v in data.items():
        fv = safe_float(v)
        if fv is not None:
            raw_row[k] = fv
        else:
            ignored.append(k)

    if verbose and ignored:
        print(f"[INFO] {run_dir.name}: ignored non-scalar fields from {filename}: {ignored}")

    return clean_row_dict(raw_row)


def read_metrics_epoch(run_dir: Path, mode: str) -> Optional[pd.DataFrame]:
    """
    Lee metrics_epoch.csv y selecciona filas según el modo:
    - last_epoch: última época por split
    - best_val_d21_f1: época con mayor d21_f1 en validación
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

    if "epoch" not in df.columns:
        print(f"[WARN] No column 'epoch' in {f}")
        return None

    if "split" not in df.columns:
        print(f"[WARN] No column 'split' in {f}")
        return None

    df["run"] = run_dir.name

    seed = extract_seed_from_run_name(run_dir.name)
    if seed is not None:
        df["seed_from_name"] = seed

    # Convertir columnas numéricas posibles de manera segura.
    for col in df.columns:
        if col not in {"run", "split"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if mode == "last_epoch":
        return (
            df.sort_values(["run", "split", "epoch"])
              .groupby(["run", "split"], as_index=False)
              .tail(1)
              .copy()
        )

    if mode == "best_val_d21_f1":
        if "d21_f1" not in df.columns:
            print(f"[WARN] No column 'd21_f1' in {f}")
            return None

        val = df[df["split"] == "val"].copy()
        val = val[np.isfinite(val["d21_f1"])]

        if val.empty:
            print(f"[WARN] No valid val d21_f1 rows in {f}")
            return None

        best_epoch = int(val.loc[val["d21_f1"].idxmax(), "epoch"])
        df_sel = df[df["epoch"] == best_epoch].copy()
        df_sel["best_epoch_selected"] = float(best_epoch)
        return df_sel

    raise ValueError(f"Modo no soportado: {mode}")


def summarize_numeric(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """
    Resume todas las columnas numéricas por group_cols.
    Devuelve tabla larga: group + metric + mean/std/min/max/n.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Columnas auxiliares que no conviene resumir como métricas.
    exclude = {"seed_from_name"}
    numeric_cols = [c for c in numeric_cols if c not in exclude]

    rows = []

    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        group_info = {c: k for c, k in zip(group_cols, keys)}

        for col in numeric_cols:
            vals = pd.to_numeric(g[col], errors="coerce").dropna().to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]

            if len(vals) == 0:
                continue

            rows.append({
                **group_info,
                "metric": col,
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "min": float(vals.min()),
                "max": float(vals.max()),
                "n": int(len(vals)),
            })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Resume métricas de múltiples corridas experimentales."
    )

    parser.add_argument(
        "--base_dir",
        required=True,
        help="Carpeta que contiene las corridas."
    )

    parser.add_argument(
        "--pattern",
        default="splitseed*_trainseed42",
        help="Patrón de carpetas de corridas dentro de base_dir."
    )

    parser.add_argument(
        "--mode",
        choices=["test_json", "test_filtered_json", "last_epoch", "best_val_d21_f1"],
        default="test_filtered_json",
        help="Fuente/modo de resumen."
    )

    parser.add_argument(
        "--out_csv",
        default="summary_metrics.csv",
        help="CSV de resumen, guardado dentro de base_dir."
    )

    parser.add_argument(
        "--out_raw_csv",
        default=None,
        help="CSV con datos crudos usados. Si se omite, será raw_<out_csv>."
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="No imprimir campos ignorados."
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

        elif args.mode in {"last_epoch", "best_val_d21_f1"}:
            df_sel = read_metrics_epoch(run_dir, args.mode)
            if df_sel is not None:
                all_rows.append(df_sel)

        else:
            raise ValueError(f"Modo no reconocido: {args.mode}")

    if not all_rows:
        raise RuntimeError("No se encontraron datos válidos para resumir.")

    if args.mode in {"test_json", "test_filtered_json"}:
        # Construcción manual ultra segura para evitar errores raros de pandas.
        clean_rows = [clean_row_dict(r) for r in all_rows]
        df_all = pd.DataFrame(clean_rows)
        group_cols = ["source"]
    else:
        df_all = pd.concat(all_rows, ignore_index=True)
        group_cols = ["split"]

    if df_all.empty:
        raise RuntimeError("El DataFrame final quedó vacío después de limpiar datos.")

    summary = summarize_numeric(df_all, group_cols)

    out_path = base_dir / args.out_csv

    if args.out_raw_csv is None:
        raw_path = base_dir / ("raw_" + args.out_csv)
    else:
        raw_path = base_dir / args.out_raw_csv

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
