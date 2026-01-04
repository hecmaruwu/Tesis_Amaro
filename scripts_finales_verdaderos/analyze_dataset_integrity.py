#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Auditoría de integridad de dataset dental 3D en processed_struct/<n_points>.
Escanea: <dataset_dir>/<jaw>/se_id_id>/{point_cloud.npy, labels.npy}
Produce Excel con hojas: summary, by_patient, missing_files, pairing.
"""

import argparse, re, json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

JAW_NAMES = ("upper", "lower")
REQUIRED_FILES = ("point_cloud.npy", "labels.npy")

def normalize_pid(case_id: str,
                  id_regex: Optional[str],
                  split_char: Optional[str],
                  split_index: int) -> str:
    if id_regex:
        m = re.match(id_regex, case_id)
        if m and m.groups():
            return m.group(1)
    if split_char is not None:
        parts = case_id.split(split_char)
        if 0 <= split_index < len(parts):
            return parts[split_index]
    return case_id

def scan_processed_struct(dataset_dir: Path,
                          id_regex: Optional[str],
                          split_char: Optional[str],
                          split_index: int):
    rows = []
    if not dataset_dir.exists():
        raise SystemExit(f"[ERROR] No existe dataset_dir: {dataset_dir}")

    resolution = dataset_dir.name
    for jaw in JAW_NAMES:
        jpath = dataset_dir / jaw
        if not jpath.exists():
            continue
        for case_dir in sorted([p for p in jpath.iterdir() if p.is_dir()]):
            case_id = case_dir.name
            pid = normalize_pid(case_id, id_regex, split_char, split_index)
            pc = case_dir / "point_cloud.npy"
            lb = case_dir / "labels.npy"
            has_pc = pc.exists()
            has_lb = lb.exists()
            reason = []
            if not has_pc:
                reason.append("missing point_cloud.npy")
            if not has_lb:
                reason.append("missing labels.npy")
            rows.append({
                "resolution": resolution,
                "jaw": jaw,
                "case_id": case_id,
                "pid": pid,
                "dir_path": str(case_dir),
                "has_point_cloud": has_pc,
                "has_labels": has_lb,
                "ok_arcada": bool(has_pc and has_lb),
                "reason": "; ".join(reason)
            })
    return rows

def build_tables(rows: List[Dict]):
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("[ERROR] No se encontraron casos en la estructura indicada.")

    df_missing = df.loc[~df["ok_arcada"]].copy().sort_values(["resolution", "pid", "jaw", "case_id"])
    grp = df.groupby(["resolution", "pid"])

    has_upper = grp.apply(lambda g: bool((g["jaw"] == "upper").any())).rename("has_upper")
    has_lower = grp.apply(lambda g: bool((g["jaw"] == "lower").any())).rename("has_lower")
    upper_ok = grp.apply(lambda g: bool(((g["jaw"]=="upper") & g["ok_arcada"]).any())).rename("upper_ok")
    lower_ok = grp.apply(lambda g: bool(((g["jaw"]=="lower") & g["ok_arcada"]).any())).rename("lower_ok")
    total_upper_cases = grp.apply(lambda g: int((g["jaw"]=="upper").sum())).rename("total_upper_cases")
    total_lower_cases = grp.apply(lambda g: int((g["jaw"]=="lower").sum())).rename("total_lower_cases")

    by_patient = pd.concat([has_upper, has_lower, upper_ok, lower_ok, total_upper_cases, total_lower_cases], axis=1).reset_index()

    def _status(row):
        if row["upper_ok"] and row["lower_ok"]:
            return "paired"
        if row["upper_ok"] and not row["lower_ok"]:
            return "lower_missing"
        if row["lower_ok"] and not row["upper_ok"]:
            return "upper_missing"
        if row["has_upper"] and not row["has_lower"]:
            return "lower_missing"
        if row["has_lower"] and not row["has_upper"]:
            return "upper_missing"
        return "incomplete"
    by_patient["status"] = by_patient.apply(_status, axis=1)

    def pick_ok_case_ids(g, jaw):
        return sorted(g.loc[(g["jaw"]==jaw) & (g["ok_arcada"]), "case_id"].unique().tolist())
    pairing_rows = []
    for (res, pid), g in df.groupby(["resolution","pid"]):
        ups = pick_ok_case_ids(g, "upper")
        lows = pick_ok_case_ids(g, "lower")
        pairing_rows.append({
            "resolution": res,
            "pid": pid,
            "upper_case_ids_ok": ", ".join(ups),
            "lower_case_ids_ok": ", ".join(lows),
            "paired": bool(ups and lows)
        })
    pairing = pd.DataFrame(pairing_rows).sort_values(["resolution","pid"])

    def summarize_res(resolution: Optional[str]):
        if resolution is None:
            bp = by_patient
        else:
            bp = by_patient[by_patient["resolution"]==resolution]
        total_pids = int(bp["pid"].nunique())
        paired = int((bp["status"]=="paired").sum())
        upper_only = int((bp["status"]=="lower_missing").sum())
        lower_only = int((bp["status"]=="upper_missing").sum())
        incomplete = int((bp["status"]=="incomplete").sum())
        return {
            "resolution": resolution or "ALL",
            "total_pids": total_pids,
            "paired_pids": paired,
            "upper_only_pids": upper_only,
            "lower_only_pids": lower_only,
            "incomplete_pids": incomplete,
        }

    summary_rows = []
    for res in sorted(df["resolution"].unique()):
        summary_rows.append(summarize_res(res))
    summary_rows.append(summarize_res(None))
    summary = pd.DataFrame(summary_rows)

    return df, by_patient, df_missing, pairing, summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True,
                    help="Ruta a processed_struct/<n_points> (ej. /home/usuario/Tesis_Amaro/data/Teeth_3ds/processed_struct/8192)")
    ap.add_argument("--out_xlsx", default="audit_dataset.xlsx",
                    help="Nombre de archivo Excel de salida")
    ap.add_argument("--id_regex", default=None,
                    help="Regex con grupo capturado para PID, ej: '^([A-Za-z0-9]+)_'")
    ap.add_argument("--split_char", default=None,
                    help="Separador para extraer PID desde case_id")
    ap.add_argument("--split_index", type=int, default=0,
                    help="Índice para usar si usas split_char")
    ap.add_argument("--split_dir", default=None,
                    help="Directorio de split para comparar PIDs (opcional)")

    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    split_dir = Path(args.split_dir) if args.split_dir else None

    print(f"[SCAN] Escaneando {dataset_dir}")
    rows = scan_processed_struct(dataset_dir, args.id_regex, args.split_char, args.split_index)
    print(f"[INFO] Encontradas {len(rows)} arcadas")

    df_raw, by_patient, df_missing, pairing, summary = build_tables(rows)

    out_xlsx = Path(args.out_xlsx)
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        summary.to_excel(writer, index=False, sheet_name="summary")
        by_patient.sort_values(["resolution","pid"]).to_excel(writer, index=False, sheet_name="by_patient")
        df_missing.to_excel(writer, index=False, sheet_name="missing_files")
        pairing.to_excel(writer, index=False, sheet_name="pairing")
        df_raw.sort_values(["resolution","jaw","pid","case_id"]).to_excel(writer, index=False, sheet_name="raw_rows")

    print("\n===== RESUMEN POR RESOLUCIÓN =====")
    print(summary.to_string(index=False))
    print(f"\n[OK] Excel generado en: {out_xlsx}")

if __name__ == "__main__":
    main()
