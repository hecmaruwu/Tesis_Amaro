#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Auditoría de dataset 3D Teeth Segmentation (estructura processed_struct).

Escanea:
  <dataset_dir>/<RES>/upper/<CASE_ID>/{point_cloud.npy, labels.npy}
  <dataset_dir>/<RES>/lower/<CASE_ID>/{point_cloud.npy, labels.npy}

Produce un Excel con varias hojas:
  - summary: conteos por resolución y globales (paired / upper-only / lower-only).
  - by_patient: una fila por PID y resolución, con estado y banderas de archivos.
  - missing_files: filas por arcada (upper/lower) que tiene faltantes, con 'reason'.
  - pairing: emparejamientos upper/lower por PID y resolución (y case_id involucrados).
  - split_vs_struct (opcional): si --split_dir se da, compara PIDs “pareables” vs PIDs en split.

Permite normalizar PID desde case_id con --id_regex ó --split_char/--split_index.

Requisitos: pandas, openpyxl (para Excel).
"""

import argparse, re, json
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
import pandas as pd

JAW_NAMES = ("upper", "lower")
REQUIRED_FILES = ("point_cloud.npy", "labels.npy")


def normalize_pid(case_id: str,
                  id_regex: Optional[str],
                  split_char: Optional[str],
                  split_index: int) -> str:
    """Normaliza PID a partir de case_id usando regex o split."""
    if id_regex:
        m = re.match(id_regex, case_id)
        if m and m.groups():
            return m.group(1)
    if split_char is not None:
        parts = case_id.split(split_char)
        if 0 <= split_index < len(parts):
            return parts[split_index]
    # Por defecto, usa el case_id crudo como PID
    return case_id


def scan_processed_struct(dataset_dir: Path,
                          id_regex: Optional[str],
                          split_char: Optional[str],
                          split_index: int):
    """
    Devuelve una lista de registros con:
    - resolution, jaw, case_id, pid, dir_path, has_point_cloud, has_labels, reason
    """
    rows = []
    if not dataset_dir.exists():
        raise SystemExit(f"[ERROR] No existe dataset_dir: {dataset_dir}")

    # Cada subcarpeta de primer nivel que sea "resolución" (e.g., 8192/4096/15230)
    for res_dir in sorted([p for p in dataset_dir.iterdir() if p.is_dir()]):
        resolution = res_dir.name
        for jaw in JAW_NAMES:
            jpath = res_dir / jaw
            if not jpath.exists():
                continue
            for case_dir in sorted([p for p in jpath.iterdir() if p.is_dir()]):
                case_id = case_dir.name
                pid = normalize_pid(case_id, id_regex, split_char, split_index)
                pc = case_dir / "point_cloud.npy"
                lb = case_dir / "labels.npy"
                has_pc = pc.exists()
                has_lb = lb.exists()
                reason_parts = []
                if not has_pc: reason_parts.append("missing point_cloud.npy")
                if not has_lb: reason_parts.append("missing labels.npy")
                reason = "; ".join(reason_parts) if reason_parts else ""
                rows.append({
                    "resolution": resolution,
                    "jaw": jaw,
                    "case_id": case_id,
                    "pid": pid,
                    "dir_path": str(case_dir),
                    "has_point_cloud": has_pc,
                    "has_labels": has_lb,
                    "ok_arcada": bool(has_pc and has_lb),
                    "reason": reason
                })
    return rows


def build_tables(rows: List[Dict]):
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("[ERROR] No se encontraron casos en la estructura indicada.")

    # ---- missing_files (una fila por arcada con problemas) ----
    df_missing = df.loc[~df["ok_arcada"]].copy()
    df_missing = df_missing.sort_values(["resolution", "pid", "jaw", "case_id"])

    # ---- by_patient: por PID+resolución ----
    # Marcas de presencia
    grp = df.groupby(["resolution", "pid"])
    def agg_has(series): return series.any()
    def agg_sum(series): return series.sum()

    # Flags por jaw
    has_upper = grp.apply(lambda g: bool((g["jaw"] == "upper").any())).rename("has_upper")
    has_lower = grp.apply(lambda g: bool((g["jaw"] == "lower").any())).rename("has_lower")

    upper_ok = grp.apply(lambda g: bool(((g["jaw"]=="upper") & g["ok_arcada"]).any())).rename("upper_ok")
    lower_ok = grp.apply(lambda g: bool(((g["jaw"]=="lower") & g["ok_arcada"]).any())).rename("lower_ok")

    total_upper_cases = grp.apply(lambda g: int((g["jaw"]=="upper").sum())).rename("total_upper_cases")
    total_lower_cases = grp.apply(lambda g: int((g["jaw"]=="lower").sum())).rename("total_lower_cases")

    by_patient = pd.concat([has_upper, has_lower, upper_ok, lower_ok,
                            total_upper_cases, total_lower_cases], axis=1).reset_index()

    # Estado
    def _status(row):
        if row["upper_ok"] and row["lower_ok"]:
            return "paired"
        if row["upper_ok"] and not row["lower_ok"]:
            return "lower_missing"
        if row["lower_ok"] and not row["upper_ok"]:
            return "upper_missing"
        # ninguno ok (pero quizá estén presentes sin archivos completos)
        if row["has_upper"] and not row["has_lower"]:
            return "lower_missing"
        if row["has_lower"] and not row["has_upper"]:
            return "upper_missing"
        return "incomplete"
    by_patient["status"] = by_patient.apply(_status, axis=1)

    # ---- pairing: case_ids usados para "ok_arcada" por PID+res ----
    # Tomamos el/los case_id que cumplen ok por cada jaw
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

    # ---- summary por resolución + global ----
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


def try_read_split_meta(split_dir: Path) -> Optional[Dict]:
    """Intenta leer meta.json de un split para extraer PIDs del split."""
    meta = split_dir / "meta.json"
    if not meta.exists():
        return None
    try:
        j = json.loads(meta.read_text(encoding="utf-8"))
    except Exception:
        return None

    # Intenta varias convenciones comunes
    candidate_keys = [
        "train_ids","val_ids","test_ids",
        "pids_train","pids_val","pids_test",
        "train_pids","val_pids","test_pids",
    ]
    found = {}
    for k in candidate_keys:
        if k in j and isinstance(j[k], list):
            found[k] = [str(x) for x in j[k]]
    if not found:
        # Nada claro en el meta; devolvemos todo el meta para inspección
        found["_raw_meta"] = j
    return found


def build_split_vs_struct_sheet(split_dir: Path,
                                by_patient: pd.DataFrame,
                                id_regex: Optional[str],
                                split_char: Optional[str],
                                split_index: int) -> Optional[pd.DataFrame]:
    """
    Compara PIDs pareables (status=='paired') en processed_struct vs PIDs del split.
    Requiere meta.json con listas de IDs. Si no están, devuelve None.
    """
    meta = try_read_split_meta(split_dir)
    if not meta:
        return None

    # PIDs pareables por resolución (y global)
    paired_by_res = by_patient[by_patient["status"]=="paired"] \
        .groupby("resolution")["pid"].apply(lambda s: sorted(set(s))).to_dict()
    paired_all = sorted(set(by_patient.loc[by_patient["status"]=="paired", "pid"].tolist()))

    # Construimos filas con diferencias por cada key encontrada en meta
    rows = []
    for part_key, pid_list in meta.items():
        if part_key == "_raw_meta":  # mostrarlo como referencia, no comparar
            continue
        split_pids = sorted(set(str(x) for x in pid_list))
        # comparación global
        missing_in_split = sorted(set(paired_all) - set(split_pids))
        extra_in_split   = sorted(set(split_pids) - set(paired_all))
        rows.append({
            "split_part": part_key,
            "scope": "ALL",
            "paired_in_struct": len(paired_all),
            "pids_in_split": len(split_pids),
            "missing_in_split": ", ".join(missing_in_split[:50]) + (" ..." if len(missing_in_split)>50 else ""),
            "extra_in_split": ", ".join(extra_in_split[:50]) + (" ..." if len(extra_in_split)>50 else "")
        })
        # comparación por resolución
        for res, plist in paired_by_res.items():
            missing = sorted(set(plist) - set(split_pids))
            extra   = sorted(set(split_pids) - set(plist))
            rows.append({
                "split_part": part_key,
                "scope": res,
                "paired_in_struct": len(plist),
                "pids_in_split": len(split_pids),
                "missing_in_split": ", ".join(missing[:50]) + (" ..." if len(missing)>50 else ""),
                "extra_in_split": ", ".join(extra[:50]) + (" ..." if len(extra)>50 else "")
            })

    df = pd.DataFrame(rows)
    if "_raw_meta" in meta:
        # Anexamos una fila con metadatos crudos para referencia
        df_meta = pd.DataFrame([{"split_part":"_raw_meta","scope":"meta.json","paired_in_struct":"-",
                                 "pids_in_split":"-", "missing_in_split":"-", "extra_in_split":"-",
                                 "_raw_meta_json": json.dumps(meta["_raw_meta"], ensure_ascii=False)}])
        df = pd.concat([df, df_meta], ignore_index=True)
    return df.sort_values(["split_part","scope"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True,
                    help="Ruta a processed_struct (e.g., data/3dteethseg/processed_struct)")
    ap.add_argument("--out_xlsx", default="audit_dataset.xlsx",
                    help="Nombre de salida Excel")
    # Normalización de PID
    ap.add_argument("--id_regex", default=None,
                    help="Regex con un grupo capturado para PID, ej: '^([A-Za-z0-9]+)_'")
    ap.add_argument("--split_char", default=None,
                    help="Separador alternativo para extraer PID desde case_id")
    ap.add_argument("--split_index", type=int, default=0,
                    help="Índice del split_char para tomar como PID")
    # Comparación contra split (opcional)
    ap.add_argument("--split_dir", default=None,
                    help="Carpeta de un split (con meta.json) para comparar PIDs (opcional)")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    split_dir = Path(args.split_dir) if args.split_dir else None

    print(f"[SCAN] {dataset_dir}")
    rows = scan_processed_struct(dataset_dir, args.id_regex, args.split_char, args.split_index)
    print(f"[INFO] Encontradas {len(rows)} arcadas (rows upper/lower x case_id x resolución)")

    df_raw, by_patient, df_missing, pairing, summary = build_tables(rows)

    # Guardar Excel
    out_xlsx = Path(args.out_xlsx)
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
        summary.to_excel(xw, index=False, sheet_name="summary")
        by_patient.sort_values(["resolution","pid"]).to_excel(xw, index=False, sheet_name="by_patient")
        df_missing.to_excel(xw, index=False, sheet_name="missing_files")
        pairing.to_excel(xw, index=False, sheet_name="pairing")
        df_raw.sort_values(["resolution","jaw","pid","case_id"]).to_excel(xw, index=False, sheet_name="raw_rows")

        if split_dir is not None:
            cmp_df = build_split_vs_struct_sheet(split_dir, by_patient,
                                                 args.id_regex, args.split_char, args.split_index)
            if cmp_df is not None:
                cmp_df.to_excel(xw, index=False, sheet_name="split_vs_struct")

    # Imprime resumen útil en consola
    print("\n===== RESUMEN POR RESOLUCIÓN =====")
    print(summary.to_string(index=False))
    print(f"\n[OK] Excel escrito en: {out_xlsx}")


if __name__ == "__main__":
    main()
