#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compute_metrics_summary_v4.py

Versión robusta SIN pandas para resumir test_metrics.json y
test_metrics_filtered.json.

Motivo:
    En algunos entornos pandas puede fallar al construir DataFrames con:
    TypeError: Cannot convert numpy.ndarray to numpy.ndarray

Esta versión evita pandas completamente para los modos:
    - test_json
    - test_filtered_json

Para metrics_epoch.csv también implementa lectura simple con csv estándar.

Salidas:
    - raw_<out_csv>: métricas por corrida/seed
    - <out_csv>: resumen mean/std/min/max/n

Ejemplo:

python scripts_last_version/compute_metrics_summary_v4.py \
  --base_dir /home/htaucare/Tesis_Amaro/outputs/pointnetpp/stability_real_splits_v1/exp04_best_config \
  --pattern "splitseed*_trainseed42" \
  --mode test_filtered_json \
  --out_csv summary_pointnetpp_test_filtered.csv
"""

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def is_number(x: Any) -> bool:
    if isinstance(x, bool):
        return False
    if isinstance(x, (int, float)):
        return math.isfinite(float(x))
    return False


def extract_seed(name: str) -> Optional[int]:
    for pat in [r"splitseed(\d+)", r"data_seed(\d+)", r"seed(\d+)"]:
        m = re.search(pat, name)
        if m:
            return int(m.group(1))
    return None


def sort_key(path: Path):
    seed = extract_seed(path.name)
    return (0, seed) if seed is not None else (1, path.name)


def read_json_metrics(run_dir: Path, filename: str, quiet: bool = False) -> Optional[Dict[str, Any]]:
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

    seed = extract_seed(run_dir.name)
    if seed is not None:
        row["seed_from_name"] = seed

    ignored = []
    for k, v in data.items():
        if is_number(v):
            row[str(k)] = float(v)
        else:
            ignored.append(str(k))

    if ignored and not quiet:
        print(f"[INFO] {run_dir.name}: ignored non-scalar fields from {filename}: {ignored}")

    return row


def read_metrics_epoch(run_dir: Path, mode: str) -> List[Dict[str, Any]]:
    f = run_dir / "metrics_epoch.csv"
    if not f.exists():
        print(f"[WARN] Missing: {f}")
        return []

    with open(f, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    if not rows:
        return []

    for r in rows:
        r["run"] = run_dir.name
        seed = extract_seed(run_dir.name)
        if seed is not None:
            r["seed_from_name"] = seed

    if "epoch" not in rows[0] or "split" not in rows[0]:
        print(f"[WARN] metrics_epoch.csv sin epoch/split: {f}")
        return []

    def to_float_safe(x):
        try:
            y = float(x)
            return y if math.isfinite(y) else None
        except Exception:
            return None

    if mode == "last_epoch":
        best_by_split: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            ep = to_float_safe(r.get("epoch"))
            sp = r.get("split")
            if ep is None or sp is None:
                continue
            if sp not in best_by_split or ep > float(best_by_split[sp]["epoch"]):
                best_by_split[sp] = r
        return list(best_by_split.values())

    if mode == "best_val_d21_f1":
        val_rows = [r for r in rows if r.get("split") == "val"]
        valid = []
        for r in val_rows:
            d21 = to_float_safe(r.get("d21_f1"))
            ep = to_float_safe(r.get("epoch"))
            if d21 is not None and ep is not None:
                valid.append((d21, ep, r))

        if not valid:
            print(f"[WARN] No valid val d21_f1 in {f}")
            return []

        _, best_epoch, _ = max(valid, key=lambda t: t[0])

        selected = []
        for r in rows:
            ep = to_float_safe(r.get("epoch"))
            if ep is not None and int(ep) == int(best_epoch):
                r = dict(r)
                r["best_epoch_selected"] = int(best_epoch)
                selected.append(r)

        return selected

    raise ValueError(f"Modo no soportado para metrics_epoch.csv: {mode}")


def normalize_rows(rows: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Convierte cada fila a dict plano:
    - run/source/split quedan como string.
    - valores numéricos se convierten a float.
    - lo demás se ignora.
    """
    aux = {"run", "source", "split"}
    cleaned = []
    columns = set()

    for row in rows:
        clean: Dict[str, Any] = {}
        for k, v in row.items():
            k = str(k)

            if k in aux:
                clean[k] = str(v)
                columns.add(k)
                continue

            if is_number(v):
                clean[k] = float(v)
                columns.add(k)
                continue

            # Intentar convertir strings numéricos del CSV.
            if isinstance(v, str):
                try:
                    fv = float(v)
                    if math.isfinite(fv):
                        clean[k] = fv
                        columns.add(k)
                except Exception:
                    pass

        cleaned.append(clean)

    # Orden estable: auxiliares primero, luego métricas alfabéticas.
    ordered = []
    for c in ["run", "source", "split", "seed_from_name"]:
        if c in columns:
            ordered.append(c)

    rest = sorted(c for c in columns if c not in set(ordered))
    ordered.extend(rest)

    return ordered, cleaned


def write_csv(path: Path, columns: List[str], rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c, "") for c in columns})


def summarize(rows: List[Dict[str, Any]], group_col: str) -> List[Dict[str, Any]]:
    aux = {"run", "source", "split", "seed_from_name"}
    metric_names = sorted(
        k for r in rows for k, v in r.items()
        if k not in aux and isinstance(v, (int, float)) and math.isfinite(float(v))
    )

    groups = sorted(set(str(r.get(group_col, "")) for r in rows))
    summary_rows = []

    for group in groups:
        group_rows = [r for r in rows if str(r.get(group_col, "")) == group]

        for metric in metric_names:
            vals = []
            for r in group_rows:
                v = r.get(metric)
                if isinstance(v, (int, float)) and math.isfinite(float(v)):
                    vals.append(float(v))

            if not vals:
                continue

            n = len(vals)
            mean = sum(vals) / n
            if n > 1:
                var = sum((x - mean) ** 2 for x in vals) / (n - 1)
                std = math.sqrt(var)
            else:
                std = 0.0

            summary_rows.append({
                group_col: group,
                "metric": metric,
                "mean": mean,
                "std": std,
                "min": min(vals),
                "max": max(vals),
                "n": n,
            })

    return summary_rows


def print_table(rows: List[Dict[str, Any]], max_rows: int = 200) -> None:
    if not rows:
        print("(empty)")
        return

    cols = list(rows[0].keys())
    print(",".join(cols))
    for r in rows[:max_rows]:
        print(",".join(str(r.get(c, "")) for c in cols))
    if len(rows) > max_rows:
        print(f"... ({len(rows) - max_rows} filas más)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--pattern", default="splitseed*_trainseed42")
    ap.add_argument(
        "--mode",
        choices=["test_json", "test_filtered_json", "last_epoch", "best_val_d21_f1"],
        default="test_filtered_json",
    )
    ap.add_argument("--out_csv", default="summary_metrics.csv")
    ap.add_argument("--out_raw_csv", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    base = Path(args.base_dir)
    run_dirs = sorted(base.glob(args.pattern), key=sort_key)

    print(f"[INFO] Base dir: {base}")
    print(f"[INFO] Pattern: {args.pattern}")
    print(f"[INFO] Mode: {args.mode}")
    print(f"[INFO] Found runs: {len(run_dirs)}")

    if not run_dirs:
        raise RuntimeError(f"No se encontraron carpetas con patrón {args.pattern} en {base}")

    rows: List[Dict[str, Any]] = []

    for d in run_dirs:
        if args.mode == "test_filtered_json":
            r = read_json_metrics(d, "test_metrics_filtered.json", quiet=args.quiet)
            if r:
                rows.append(r)
        elif args.mode == "test_json":
            r = read_json_metrics(d, "test_metrics.json", quiet=args.quiet)
            if r:
                rows.append(r)
        else:
            rows.extend(read_metrics_epoch(d, args.mode))

    if not rows:
        raise RuntimeError("No se encontraron datos válidos.")

    raw_cols, raw_rows = normalize_rows(rows)

    if args.mode in {"test_json", "test_filtered_json"}:
        group_col = "source"
    else:
        group_col = "split"

    summary_rows = summarize(raw_rows, group_col)
    summary_cols = [group_col, "metric", "mean", "std", "min", "max", "n"]

    out_path = base / args.out_csv
    raw_name = args.out_raw_csv if args.out_raw_csv else "raw_" + args.out_csv
    raw_path = base / raw_name

    write_csv(raw_path, raw_cols, raw_rows)
    write_csv(out_path, summary_cols, summary_rows)

    print("\n===== RAW DATA USED =====\n")
    print_table(raw_rows)

    print("\n===== SUMMARY =====\n")
    print_table(summary_rows)

    print(f"\n[OK] Raw data guardado en: {raw_path}")
    print(f"[OK] Summary guardado en: {out_path}")


if __name__ == "__main__":
    main()
