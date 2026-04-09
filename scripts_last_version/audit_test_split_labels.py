#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
audit_test_split_labels.py

Auditoría de etiquetas por sample para datasets NPZ de segmentación dental 3D.

Objetivo:
- Detectar samples degenerados o sospechosos antes de comparar modelos en la tesis.
- Revisar train / val / test de forma sistemática.
- Identificar problemas como:
  ✅ sample con solo background
  ✅ sample sin d21
  ✅ sample con muy pocas clases
  ✅ sample con proporción extrema de background
  ✅ distribución de clases por sample
  ✅ trazabilidad con index_{split}.csv si existe

Entradas esperadas:
  data_dir/
    X_train.npz, Y_train.npz
    X_val.npz,   Y_val.npz
    X_test.npz,  Y_test.npz
    index_train.csv / index_val.csv / index_test.csv   (opcionales)

Salidas:
  out_dir/
    audit_summary.json
    audit_train.csv
    audit_val.csv
    audit_test.csv
    samples_only_bg_train.csv
    samples_only_bg_val.csv
    samples_only_bg_test.csv
    samples_no_d21_train.csv
    samples_no_d21_val.csv
    samples_no_d21_test.csv
    samples_few_classes_train.csv
    samples_few_classes_val.csv
    samples_few_classes_test.csv

Ejemplo:
python3 audit_test_split_labels.py \
  --data_dir /home/htaucare/Tesis_Amaro/data/Teeth_3ds/fixed_split/8192/upper_only_surf_global_excl_wisdom_seed42_aug2 \
  --out_dir  /home/htaucare/Tesis_Amaro/outputs/audits/test_split_audit \
  --bg_class 0 \
  --d21_internal 8 \
  --few_classes_threshold 2 \
  --high_bg_threshold 0.95
"""

import csv
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np


# ============================================================
# IO HELPERS
# ============================================================

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_csv(rows: List[Dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ============================================================
# INDEX CSV HELPERS
# ============================================================

def discover_index_csv(data_dir: Path, split: str) -> Optional[Path]:
    split = str(split).strip().lower()
    fname = f"index_{split}.csv"

    p1 = data_dir / fname
    if p1.exists():
        return p1

    cur = data_dir.resolve()
    for _ in range(12):
        cand = cur / fname
        if cand.exists():
            return cand
        if cur.name.lower() == "teeth_3ds":
            break
        if cur.parent == cur:
            break
        cur = cur.parent

    cur = data_dir.resolve()
    teeth_root = None
    for _ in range(20):
        if cur.name.lower() == "teeth_3ds":
            teeth_root = cur
            break
        if cur.parent == cur:
            break
        cur = cur.parent

    if teeth_root is None:
        return None

    cands = []
    for m in teeth_root.glob("merged_*"):
        if m.is_dir():
            p = m / fname
            if p.exists():
                cands.append(p)

    if not cands:
        return None

    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def read_index_csv(p: Optional[Path]) -> Optional[Dict[int, Dict[str, str]]]:
    if p is None or not Path(p).exists():
        return None

    with open(p, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return None

    fieldnames = list(rows[0].keys())
    idx_col = None
    for c in ("row_i", "idx", "index", "row", "i"):
        if c in fieldnames:
            idx_col = c
            break

    out: Dict[int, Dict[str, str]] = {}
    for i, row in enumerate(rows):
        rid = i
        if idx_col is not None:
            try:
                rid = int(float(str(row.get(idx_col, "")).strip()))
            except Exception:
                rid = i
        out[int(rid)] = {k: ("" if row.get(k) is None else str(row.get(k))) for k in row.keys()}
    return out


def get_trace_fields(info: Optional[Dict[str, str]]) -> Dict[str, str]:
    if not info:
        return {
            "sample_name": "",
            "jaw": "",
            "path": "",
            "idx_global": "",
        }
    return {
        "sample_name": info.get("sample_name", ""),
        "jaw": info.get("jaw", ""),
        "path": info.get("path", ""),
        "idx_global": info.get("idx_global", ""),
    }


# ============================================================
# AUDIT CORE
# ============================================================

def summarize_sample(
    y_row: np.ndarray,
    row_i: int,
    bg_class: int,
    d21_internal: int,
    few_classes_threshold: int,
    high_bg_threshold: float,
    trace_info: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    vals, counts = np.unique(y_row, return_counts=True)
    total = int(y_row.shape[0])

    class_counts = dict(zip(vals.tolist(), counts.tolist()))
    n_classes_present = int(len(vals))
    bg_count = int(class_counts.get(int(bg_class), 0))
    bg_frac = float(bg_count / total) if total > 0 else 0.0
    has_d21 = bool(int(d21_internal) in class_counts)

    only_bg = bool(n_classes_present == 1 and int(vals[0]) == int(bg_class))
    few_classes = bool(n_classes_present <= int(few_classes_threshold))
    high_bg = bool(bg_frac >= float(high_bg_threshold))

    row = {
        "row_i": int(row_i),
        "n_points": int(total),
        "n_classes_present": int(n_classes_present),
        "classes_present": json.dumps(vals.tolist(), ensure_ascii=False),
        "class_counts": json.dumps(class_counts, ensure_ascii=False),
        "bg_count": int(bg_count),
        "bg_frac": float(bg_frac),
        "has_d21": int(has_d21),
        "only_bg": int(only_bg),
        "few_classes": int(few_classes),
        "high_bg": int(high_bg),
    }
    row.update(get_trace_fields(trace_info))
    return row


def audit_split(
    data_dir: Path,
    split: str,
    bg_class: int,
    d21_internal: int,
    few_classes_threshold: int,
    high_bg_threshold: float,
    forced_index_csv: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    yp = data_dir / f"Y_{split}.npz"
    if not yp.exists():
        raise FileNotFoundError(f"No existe {yp}")

    Y = np.load(yp)["Y"]

    if forced_index_csv is not None:
        idx_map = read_index_csv(forced_index_csv)
    else:
        idx_path = discover_index_csv(data_dir, split)
        idx_map = read_index_csv(idx_path)

    rows: List[Dict[str, Any]] = []

    only_bg_rows = []
    no_d21_rows = []
    few_classes_rows = []
    high_bg_rows = []

    for i in range(len(Y)):
        info = idx_map.get(i) if idx_map is not None and i in idx_map else None
        row = summarize_sample(
            y_row=Y[i],
            row_i=i,
            bg_class=bg_class,
            d21_internal=d21_internal,
            few_classes_threshold=few_classes_threshold,
            high_bg_threshold=high_bg_threshold,
            trace_info=info,
        )
        rows.append(row)

        if row["only_bg"] == 1:
            only_bg_rows.append(i)
        if row["has_d21"] == 0:
            no_d21_rows.append(i)
        if row["few_classes"] == 1:
            few_classes_rows.append(i)
        if row["high_bg"] == 1:
            high_bg_rows.append(i)

    summary = {
        "split": split,
        "n_samples": int(len(Y)),
        "n_points_per_sample": int(Y.shape[1]) if Y.ndim >= 2 else None,
        "only_bg_rows": only_bg_rows,
        "n_only_bg": int(len(only_bg_rows)),
        "no_d21_rows": no_d21_rows,
        "n_no_d21": int(len(no_d21_rows)),
        "few_classes_rows": few_classes_rows,
        "n_few_classes": int(len(few_classes_rows)),
        "high_bg_rows": high_bg_rows,
        "n_high_bg": int(len(high_bg_rows)),
    }

    return rows, summary


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--bg_class", type=int, default=0)
    ap.add_argument("--d21_internal", type=int, default=8)
    ap.add_argument("--few_classes_threshold", type=int, default=2)
    ap.add_argument("--high_bg_threshold", type=float, default=0.95)

    ap.add_argument("--index_train_csv", type=str, default=None)
    ap.add_argument("--index_val_csv", type=str, default=None)
    ap.add_argument("--index_test_csv", type=str, default=None)

    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir = ensure_dir(Path(args.out_dir).resolve())

    split_forced_index = {
        "train": Path(args.index_train_csv).resolve() if args.index_train_csv else None,
        "val": Path(args.index_val_csv).resolve() if args.index_val_csv else None,
        "test": Path(args.index_test_csv).resolve() if args.index_test_csv else None,
    }

    global_summary = {
        "data_dir": str(data_dir),
        "out_dir": str(out_dir),
        "bg_class": int(args.bg_class),
        "d21_internal": int(args.d21_internal),
        "few_classes_threshold": int(args.few_classes_threshold),
        "high_bg_threshold": float(args.high_bg_threshold),
        "splits": {},
    }

    for split in ("train", "val", "test"):
        rows, summary = audit_split(
            data_dir=data_dir,
            split=split,
            bg_class=int(args.bg_class),
            d21_internal=int(args.d21_internal),
            few_classes_threshold=int(args.few_classes_threshold),
            high_bg_threshold=float(args.high_bg_threshold),
            forced_index_csv=split_forced_index[split],
        )

        global_summary["splits"][split] = summary

        # CSV completo del split
        write_csv(rows, out_dir / f"audit_{split}.csv")

        # subsets importantes
        write_csv([r for r in rows if r["only_bg"] == 1], out_dir / f"samples_only_bg_{split}.csv")
        write_csv([r for r in rows if r["has_d21"] == 0], out_dir / f"samples_no_d21_{split}.csv")
        write_csv([r for r in rows if r["few_classes"] == 1], out_dir / f"samples_few_classes_{split}.csv")
        write_csv([r for r in rows if r["high_bg"] == 1], out_dir / f"samples_high_bg_{split}.csv")

    save_json(global_summary, out_dir / "audit_summary.json")

    print("\n================ AUDIT SUMMARY ================\n")
    for split in ("train", "val", "test"):
        s = global_summary["splits"][split]
        print(f"[{split}]")
        print(f"  n_samples       : {s['n_samples']}")
        print(f"  n_only_bg       : {s['n_only_bg']}")
        print(f"  n_no_d21        : {s['n_no_d21']}")
        print(f"  n_few_classes   : {s['n_few_classes']}")
        print(f"  n_high_bg       : {s['n_high_bg']}")
        print(f"  only_bg_rows    : {s['only_bg_rows'][:20]}")
        print(f"  no_d21_rows     : {s['no_d21_rows'][:20]}")
        print(f"  few_classes_rows: {s['few_classes_rows'][:20]}")
        print(f"  high_bg_rows    : {s['high_bg_rows'][:20]}")
        print()

    print(f"Archivos guardados en: {out_dir}")


if __name__ == "__main__":
    main()