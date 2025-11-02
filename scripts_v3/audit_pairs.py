#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_pairs.py  (v3 compatible con flujo paper-like)

Audita el dataset de nubes procesadas (upper/lower) y genera reportes:
 - pairs_report.json
 - pairs_summary.csv
 - pairs_missing.csv

Compatible con los módulos nuevos:
  - build_pointcloud_dataset_v3.py
  - build_merged_pointcloud_dataset_v3.py
"""

import argparse, json, csv, re
from pathlib import Path
from typing import Dict, List


# ===========================================================
# --------- Funciones utilitarias de escaneo local ----------
# ===========================================================

def scan_cases(dataset_dir: Path, id_regex: str = None,
               split_char: str = None, split_index: int = 0) -> Dict[str, Dict[str, List[str]]]:
    """
    Busca casos en dataset_dir agrupados por PID y jaw (upper/lower).
    Retorna: {pid: {"upper": [...], "lower": [...]}}

    - Usa id_regex si se entrega (ej: "^([A-Z0-9]+)_")
    - O bien usa split_char/split_index (ej: "_" y 0)
    """
    dataset_dir = Path(dataset_dir)
    pattern = re.compile(id_regex) if id_regex else None
    cases = {}

    for sub in dataset_dir.glob("*/*"):  # ej: processed_flat/upper/XXXXXX
        if not sub.is_dir():
            continue
        jaw = sub.parent.name.lower()  # "upper" o "lower"
        if jaw not in ["upper", "lower"]:
            continue
        for f in sub.glob("*.npz"):
            fname = f.stem
            if pattern:
                m = pattern.match(fname)
                if not m:
                    continue
                pid = m.group(1)
            elif split_char:
                parts = fname.split(split_char)
                if len(parts) > split_index:
                    pid = parts[split_index]
                else:
                    continue
            else:
                pid = fname
            cases.setdefault(pid, {"upper": [], "lower": []})
            cases[pid][jaw].append(str(f))
    return cases


def filter_paired(cases: Dict[str, Dict[str, List[str]]]):
    """
    Dado un diccionario de casos, separa aquellos con pares upper/lower.
    """
    paired = {}
    upper_only, lower_only = {}, {}
    for pid, v in cases.items():
        has_u = len(v.get("upper", [])) > 0
        has_l = len(v.get("lower", [])) > 0
        if has_u and has_l:
            paired[pid] = v
        elif has_u:
            upper_only[pid] = v
        elif has_l:
            lower_only[pid] = v

    rep = {
        "total_cases": len(cases),
        "total_paired_cases": len(paired),
        "num_upper_only_pids": len(upper_only),
        "num_lower_only_pids": len(lower_only),
        "examples_upper_only": [{"pid": k, "case_id": Path(v["upper"][0]).stem} for k, v in list(upper_only.items())[:5]],
        "examples_lower_only": [{"pid": k, "case_id": Path(v["lower"][0]).stem} for k, v in list(lower_only.items())[:5]],
    }
    return paired, rep


# ===========================================================
# ---------------------- MAIN LOGIC --------------------------
# ===========================================================

def write_csv(path: Path, rows, header):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser(description="Auditoría de pares upper/lower en dataset procesado")
    ap.add_argument("--dataset_dir", required=True,
                    help="Ruta raíz de processed_flat o similar")
    ap.add_argument("--id_regex", default=None,
                    help="Regex con grupo capturado para pid, ej: '^([A-Za-z0-9]+)_'")
    ap.add_argument("--split_char", default=None,
                    help="Separador alternativo para extraer pid")
    ap.add_argument("--split_index", type=int, default=0,
                    help="Índice dentro del split_char")
    ap.add_argument("--out_json", default="logs/pairs_report.json")
    ap.add_argument("--out_csv_summary", default="logs/pairs_summary.csv")
    ap.add_argument("--out_csv_missing", default="logs/pairs_missing.csv")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta {dataset_dir}")

    cases = scan_cases(dataset_dir, args.id_regex, args.split_char, args.split_index)
    paired, rep = filter_paired(cases)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"[OK] JSON resumen: {args.out_json}")

    summary_rows = [{
        "total_cases": rep["total_cases"],
        "total_paired_cases": rep["total_paired_cases"],
        "num_upper_only_pids": rep["num_upper_only_pids"],
        "num_lower_only_pids": rep["num_lower_only_pids"],
    }]
    write_csv(Path(args.out_csv_summary), summary_rows,
              header=["total_cases", "total_paired_cases",
                      "num_upper_only_pids", "num_lower_only_pids"])
    print(f"[OK] CSV summary: {args.out_csv_summary}")

    missing_rows = []
    for ex in rep.get("examples_upper_only", []):
        missing_rows.append({"pid": ex["pid"], "jaw_missing": "lower", "case_id_example": ex["case_id"]})
    for ex in rep.get("examples_lower_only", []):
        missing_rows.append({"pid": ex["pid"], "jaw_missing": "upper", "case_id_example": ex["case_id"]})
    write_csv(Path(args.out_csv_missing), missing_rows,
              header=["pid", "jaw_missing", "case_id_example"])
    print(f"[OK] CSV faltantes: {args.out_csv_missing}")

    print("\nResumen global:")
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
