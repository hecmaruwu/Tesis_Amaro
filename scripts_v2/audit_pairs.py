#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse, json, csv
from pathlib import Path
from typing import Optional
from build_pointcloud_dataset import scan_cases, filter_paired  # Reutilizamos funciones helper


def write_csv(path: Path, rows, header):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True,
                    help="/home/usuario/Tesis_Amaro/data/Teeth_3ds/processed_flat")
    ap.add_argument("--id_regex", default=None,
                    help="Regex con grupo capturado para pid, ej: '^([A-Za-z0-9]+)_'")
    ap.add_argument("--split_char", default=None,
                    help="Separador para pid")
    ap.add_argument("--split_index", type=int, default=0,
                    help="Índice para usar en split_char")
    ap.add_argument("--out_json", default="logs/pairs_report.json")
    ap.add_argument("--out_csv_summary", default="logs/pairs_summary.csv")
    ap.add_argument("--out_csv_missing", default="logs/pairs_missing.csv")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    cases = scan_cases(dataset_dir, args.id_regex, args.split_char, args.split_index)
    paired, rep = filter_paired(cases)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print("[OK] JSON resumen:", args.out_json)

    summary_rows = [{
        "total_cases": rep["total_cases"],
        "total_paired_cases": rep["total_paired_cases"],
        "num_upper_only_pids": rep["num_upper_only_pids"],
        "num_lower_only_pids": rep["num_lower_only_pids"],
    }]
    write_csv(Path(args.out_csv_summary), summary_rows,
              header=["total_cases", "total_paired_cases", "num_upper_only_pids", "num_lower_only_pids"])
    print("[OK] CSV summary:", args.out_csv_summary)

    missing_rows = []
    for ex in rep.get("examples_upper_only", []):
        missing_rows.append({"pid": ex["pid"], "jaw_missing": "lower", "case_id_example": ex["case_id"]})
    for ex in rep.get("examples_lower_only", []):
        missing_rows.append({"pid": ex["pid"], "jaw_missing": "upper", "case_id_example": ex["case_id"]})
    write_csv(Path(args.out_csv_missing), missing_rows, header=["pid", "jaw_missing", "case_id_example"])
    print("[OK] CSV faltantes:", args.out_csv_missing)

    print("\nResumen:")
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()

    