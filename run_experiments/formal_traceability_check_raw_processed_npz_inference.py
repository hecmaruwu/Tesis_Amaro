#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
formal_traceability_check_raw_processed_npz_inference.py

Chequeo formal de trazabilidad para un split NPZ de Teeth_3ds.

Objetivo:
probar, de forma documentable, la cadena:

row_i del NPZ
→ index_{split}.csv
→ processed_struct_safe/.../<sample_name>
→ raw/.../<sample_name>_<jaw>.obj|stl|ply
→ inference_manifest.csv (opcional)

Qué valida:
1) Alineación exacta NPZ ↔ index_csv
2) Existencia del path procesado registrado en index_csv
3) Existencia y unicidad del archivo RAW esperado por sample_name + jaw
4) Coincidencia index_csv ↔ inference_manifest (si se entrega)
5) Reporte formal en JSON + CSV + TXT

Uso típico:
python3 formal_traceability_check_raw_processed_npz_inference.py \
  --data_dir /home/htaucare/Tesis_Amaro/data/Teeth_3ds/fixed_split/8192/upper_only_surf_global_excl_wisdom_seed42_aug2 \
  --split test \
  --index_csv /home/htaucare/Tesis_Amaro/data/Teeth_3ds/fixed_split/8192/upper_only_surf_global_excl_wisdom_seed42_aug2/index_test.csv \
  --raw_root /home/htaucare/Tesis_Amaro/data/Teeth_3ds/raw \
  --inference_dir /home/htaucare/Tesis_Amaro/outputs/pointnettransformer/grid_pro_runner_v2_gpu0/exp02_bs4_lr1e4_dm256_dep4_h8_ff512_tok2048/inference \
  --out_dir /home/htaucare/Tesis_Amaro/audits/formal_traceability_transformer \
  --rows 0 5 10 25

Modo estricto:
agregue --strict para hacer que falle si encuentra inconsistencias.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np


# ============================================================
# Utils
# ============================================================

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if path is None or (not path.exists()):
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pick_field(fieldnames: List[str], *cands: str) -> Optional[str]:
    fields = {f.strip(): f for f in fieldnames}
    for c in cands:
        if c in fields:
            return fields[c]
    return None


# ============================================================
# CSV readers
# ============================================================

def read_index_csv(index_path: Path) -> Dict[int, Dict[str, str]]:
    rows = read_csv_rows(index_path)
    if not rows:
        return {}

    fieldnames = list(rows[0].keys())
    row_key = pick_field(fieldnames, "row_i", "row", "i", "idx", "index")
    if row_key is None:
        raise ValueError(f"No se encontró columna de índice en {index_path}")

    k_name = pick_field(fieldnames, "sample_name", "sample", "name", "patient")
    k_jaw = pick_field(fieldnames, "jaw", "arch")
    k_path = pick_field(fieldnames, "path", "file_path")

    mp: Dict[int, Dict[str, str]] = {}
    for r in rows:
        try:
            ri = int(r[row_key])
        except Exception:
            continue
        mp[ri] = {
            "row_i": str(ri),
            "sample_name": r.get(k_name, "") if k_name else "",
            "jaw": r.get(k_jaw, "") if k_jaw else "",
            "path": r.get(k_path, "") if k_path else "",
        }
    return mp


def read_inference_manifest(manifest_path: Optional[Path]) -> Dict[int, Dict[str, str]]:
    if manifest_path is None or (not manifest_path.exists()):
        return {}

    rows = read_csv_rows(manifest_path)
    if not rows:
        return {}

    fieldnames = list(rows[0].keys())
    row_key = pick_field(fieldnames, "row_i", "row", "i", "idx", "index")
    if row_key is None:
        return {}

    mp: Dict[int, Dict[str, str]] = {}
    for r in rows:
        try:
            ri = int(r[row_key])
        except Exception:
            continue
        mp[ri] = dict(r)
    return mp


# ============================================================
# Formal checks
# ============================================================

def check_npz_shapes(data_dir: Path, split: str) -> Dict[str, Any]:
    Xp = data_dir / f"X_{split}.npz"
    Yp = data_dir / f"Y_{split}.npz"
    if not Xp.exists() or not Yp.exists():
        raise FileNotFoundError(f"No existen {Xp} o {Yp}")

    X = np.load(Xp)["X"]
    Y = np.load(Yp)["Y"]

    return {
        "X_path": str(Xp),
        "Y_path": str(Yp),
        "X_shape": list(X.shape),
        "Y_shape": list(Y.shape),
        "n_rows": int(X.shape[0]),
        "ok_same_rows": bool(int(X.shape[0]) == int(Y.shape[0])),
        "ok_points_shape": bool(X.ndim == 3 and Y.ndim == 2 and int(X.shape[1]) == int(Y.shape[1])),
    }


def contiguous_row_check(index_map: Dict[int, Dict[str, str]], n_rows: int) -> Dict[str, Any]:
    keys = sorted(index_map.keys())
    expected = list(range(n_rows))
    ok = (keys == expected)
    missing = sorted(set(expected) - set(keys))
    extra = sorted(set(keys) - set(expected))
    return {
        "ok_exact_0_to_n_minus_1": bool(ok),
        "n_index_rows": int(len(keys)),
        "n_expected_rows": int(n_rows),
        "missing_rows": missing,
        "extra_rows": extra,
    }


def resolve_processed_path(raw_path: str) -> Tuple[Optional[Path], str]:
    raw_path = (raw_path or "").strip()
    if not raw_path:
        return None, "missing"
    p = Path(raw_path)
    if p.exists():
        return p.resolve(), "exists"
    return None, "not_found"


def find_raw_candidates(raw_root: Path, sample_name: str, jaw: str) -> List[Path]:
    """
    Busca raw tipo:
      .../<sample_name>_<jaw>.obj
      .../<sample_name>_<jaw>.stl
      .../<sample_name>_<jaw>.ply

    También acepta carpeta del sample y luego filtra por basename.
    """
    sample_name = (sample_name or "").strip()
    jaw = (jaw or "").strip().lower()

    if not sample_name or not jaw:
        return []

    patterns = [
        f"{sample_name}_{jaw}.obj",
        f"{sample_name}_{jaw}.stl",
        f"{sample_name}_{jaw}.ply",
    ]

    matches: List[Path] = []
    for pat in patterns:
        matches.extend(raw_root.rglob(pat))

    # quitar duplicados
    uniq = []
    seen = set()
    for m in matches:
        mr = str(m.resolve())
        if mr not in seen:
            uniq.append(m.resolve())
            seen.add(mr)
    return uniq


def compare_index_vs_manifest(index_meta: Dict[str, str], mani_meta: Dict[str, str]) -> Dict[str, Any]:
    if not mani_meta:
        return {"has_manifest_entry": False}

    out = {"has_manifest_entry": True}
    for k in ("sample_name", "jaw", "path"):
        a = (index_meta.get(k, "") or "").strip()
        b = (mani_meta.get(k, "") or "").strip()
        out[f"match_{k}"] = (a == b) if (a or b) else True
        out[f"index_{k}"] = a
        out[f"manifest_{k}"] = b
    return out


def formal_row_check(
    row_i: int,
    index_map: Dict[int, Dict[str, str]],
    manifest_map: Dict[int, Dict[str, str]],
    raw_root: Path,
) -> Dict[str, Any]:
    rec: Dict[str, Any] = {"row_i": int(row_i)}

    idx_meta = index_map.get(int(row_i), {})
    rec["index_meta"] = idx_meta
    rec["has_index_entry"] = bool(idx_meta)

    sample_name = idx_meta.get("sample_name", "")
    jaw = idx_meta.get("jaw", "")
    processed_path_raw = idx_meta.get("path", "")

    proc_path, proc_mode = resolve_processed_path(processed_path_raw)
    rec["processed_path_resolved"] = str(proc_path) if proc_path is not None else ""
    rec["processed_path_resolution_mode"] = proc_mode
    rec["processed_exists"] = bool(proc_path is not None and proc_path.exists())

    raw_candidates = find_raw_candidates(raw_root, sample_name=sample_name, jaw=jaw)
    rec["raw_candidates"] = [str(p) for p in raw_candidates]
    rec["raw_candidate_count"] = int(len(raw_candidates))
    rec["raw_unique_match"] = bool(len(raw_candidates) == 1)
    rec["raw_selected"] = str(raw_candidates[0]) if len(raw_candidates) >= 1 else ""

    mani_meta = manifest_map.get(int(row_i), {})
    rec["manifest_meta"] = mani_meta
    rec["manifest_check"] = compare_index_vs_manifest(idx_meta, mani_meta)

    # Veredictos formales
    rec["verdict_npz_to_index"] = bool(idx_meta)
    rec["verdict_index_to_processed"] = bool(rec["processed_exists"])
    rec["verdict_processed_to_raw"] = bool(len(raw_candidates) == 1)
    rec["verdict_index_to_manifest"] = bool(rec["manifest_check"].get("has_manifest_entry", False))
    rec["verdict_index_manifest_consistent"] = bool(
        rec["manifest_check"].get("has_manifest_entry", False)
        and rec["manifest_check"].get("match_sample_name", True)
        and rec["manifest_check"].get("match_jaw", True)
        and rec["manifest_check"].get("match_path", True)
    ) if mani_meta else False

    rec["verdict_full_traceability_formal"] = bool(
        rec["verdict_npz_to_index"]
        and rec["verdict_index_to_processed"]
        and rec["verdict_processed_to_raw"]
        and (
            (not manifest_map) or
            (rec["verdict_index_to_manifest"] and rec["verdict_index_manifest_consistent"])
        )
    )

    return rec


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Chequeo formal NPZ -> index -> processed -> raw -> inference")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--index_csv", type=str, required=True)
    p.add_argument("--raw_root", type=str, required=True)
    p.add_argument("--inference_dir", type=str, default=None)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--rows", type=int, nargs="+", required=True)
    p.add_argument("--strict", action="store_true")
    return p.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    data_dir = Path(args.data_dir).resolve()
    index_csv = Path(args.index_csv).resolve()
    raw_root = Path(args.raw_root).resolve()
    out_dir = ensure_dir(Path(args.out_dir).resolve())

    inference_dir = Path(args.inference_dir).resolve() if args.inference_dir else None
    manifest_path = (inference_dir / "inference_manifest.csv") if inference_dir and (inference_dir / "inference_manifest.csv").exists() else None

    npz_info = check_npz_shapes(data_dir, args.split)
    index_map = read_index_csv(index_csv)
    manifest_map = read_inference_manifest(manifest_path)

    align = contiguous_row_check(index_map, int(npz_info["n_rows"]))

    audited_rows = []
    for ri in args.rows:
        if ri < 0 or ri >= int(npz_info["n_rows"]):
            raise ValueError(f"row fuera de rango: {ri}")
        audited_rows.append(formal_row_check(ri, index_map, manifest_map, raw_root))

    summary = {
        "data_dir": str(data_dir),
        "split": str(args.split),
        "index_csv": str(index_csv),
        "raw_root": str(raw_root),
        "inference_manifest": str(manifest_path) if manifest_path else "",
        "npz_info": npz_info,
        "index_alignment_check": align,
    }

    report = {
        "summary": summary,
        "audited_rows": audited_rows,
    }

    save_json(report, out_dir / "formal_traceability_report.json")

    # CSV resumido
    table_rows = []
    for rec in audited_rows:
        idx = rec.get("index_meta", {})
        table_rows.append({
            "row_i": rec["row_i"],
            "sample_name": idx.get("sample_name", ""),
            "jaw": idx.get("jaw", ""),
            "processed_path": idx.get("path", ""),
            "processed_exists": rec["processed_exists"],
            "raw_candidate_count": rec["raw_candidate_count"],
            "raw_unique_match": rec["raw_unique_match"],
            "raw_selected": rec["raw_selected"],
            "manifest_has_entry": rec["manifest_check"].get("has_manifest_entry", False),
            "manifest_match_sample_name": rec["manifest_check"].get("match_sample_name", False),
            "manifest_match_jaw": rec["manifest_check"].get("match_jaw", False),
            "manifest_match_path": rec["manifest_check"].get("match_path", False),
            "verdict_full_traceability_formal": rec["verdict_full_traceability_formal"],
        })

    csv_path = out_dir / "formal_traceability_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table_rows[0].keys()) if table_rows else ["row_i"])
        w.writeheader()
        if table_rows:
            w.writerows(table_rows)

    # TXT legible
    lines = []
    lines.append("FORMAL TRACEABILITY CHECK")
    lines.append("=" * 90)
    lines.append(f"data_dir: {data_dir}")
    lines.append(f"split: {args.split}")
    lines.append(f"index_csv: {index_csv}")
    lines.append(f"raw_root: {raw_root}")
    lines.append(f"inference_manifest: {manifest_path if manifest_path else ''}")
    lines.append("")
    lines.append("NPZ INFO")
    lines.append(json.dumps(npz_info, ensure_ascii=False))
    lines.append("")
    lines.append("INDEX ALIGNMENT CHECK")
    lines.append(json.dumps(align, ensure_ascii=False))
    lines.append("")
    lines.append("ROW CHECKS")
    for rec in audited_rows:
        idx = rec.get("index_meta", {})
        lines.append(
            f"row={rec['row_i']} | sample={idx.get('sample_name','')} | jaw={idx.get('jaw','')} | "
            f"processed_exists={rec['processed_exists']} | raw_candidate_count={rec['raw_candidate_count']} | "
            f"raw_unique_match={rec['raw_unique_match']} | manifest_has_entry={rec['manifest_check'].get('has_manifest_entry', False)} | "
            f"full_traceability_formal={rec['verdict_full_traceability_formal']}"
        )

    txt_path = out_dir / "formal_traceability_summary.txt"
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    if args.strict:
        problems = []
        if not npz_info["ok_same_rows"]:
            problems.append("X/Y no tienen mismo número de filas")
        if not npz_info["ok_points_shape"]:
            problems.append("Shapes X/Y incompatibles")
        if not align["ok_exact_0_to_n_minus_1"]:
            problems.append("index_csv no coincide exactamente con 0..N-1")
        for rec in audited_rows:
            if not rec["verdict_full_traceability_formal"]:
                problems.append(f"row {rec['row_i']}: trazabilidad formal incompleta")
        if problems:
            raise RuntimeError("STRICT MODE FAILED:\n- " + "\n- ".join(problems))

    print(f"[OK] JSON: {out_dir / 'formal_traceability_report.json'}")
    print(f"[OK] CSV : {csv_path}")
    print(f"[OK] TXT : {txt_path}")


if __name__ == "__main__":
    main()
