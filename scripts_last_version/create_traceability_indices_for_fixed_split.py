#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
create_traceability_indices_for_fixed_split.py

Crea index_train.csv, index_val.csv e index_test.csv para un fixed_split
derivado desde un merged_* que ya tiene trazabilidad.

No usa pandas. Solo csv + numpy.
"""

import argparse
import csv
import sys
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np


def npz_len(path: Path, key: str) -> int:
    if not path.exists():
        raise FileNotFoundError(f"No existe: {path}")
    with np.load(path) as data:
        return int(data[key].shape[0])


def read_csv_rows(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"No existe: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [dict(r) for r in reader]
        fieldnames = reader.fieldnames or []

    if not rows:
        raise ValueError(f"CSV vacío: {path}")

    # quitar columnas basura
    clean_fields = [c for c in fieldnames if not c.startswith("Unnamed")]
    clean_rows = []
    for r in rows:
        clean_rows.append({k: r.get(k, "") for k in clean_fields})

    return clean_rows, clean_fields


def backup(path: Path):
    if path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bkp = path.with_suffix(path.suffix + f".bak_{stamp}")
        shutil.copy2(path, bkp)
        print(f"[BACKUP] {path.name} -> {bkp.name}")


def write_csv_rows(path: Path, rows, fieldnames, overwrite: bool):
    if path.exists() and not overwrite:
        raise FileExistsError(f"Ya existe {path}. Usa --overwrite.")

    if path.exists() and overwrite:
        backup(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def expand_train_rows(base_rows, base_fields, target_len: int, augment_times):
    base_len = len(base_rows)

    if augment_times is None:
        if target_len % base_len != 0:
            raise ValueError(
                f"No puedo inferir augment_times: target_len={target_len}, base_len={base_len}"
            )
        augment_times = target_len // base_len - 1

    augment_times = int(augment_times)
    expected = base_len * (1 + augment_times)

    if expected != target_len:
        raise ValueError(
            "No calza train:\n"
            f"  base_len={base_len}\n"
            f"  augment_times={augment_times}\n"
            f"  expected={expected}\n"
            f"  target_len={target_len}"
        )

    extra_fields = [
        "row_i",
        "base_row",
        "aug_id",
        "augmented",
        "trace_split",
        "trace_row_in_source_split",
    ]

    final_fields = []
    for c in extra_fields + base_fields:
        if c not in final_fields:
            final_fields.append(c)

    out = []
    row_i = 0

    # OJO: el orden del augmentation_and_split_v3 es:
    # primero todos los originales, después augment 1, después augment 2.
    for aug_id in range(1 + augment_times):
        for base_row, r in enumerate(base_rows):
            nr = {k: "" for k in final_fields}
            nr.update(r)

            nr["row_i"] = str(row_i)
            nr["base_row"] = str(base_row)
            nr["aug_id"] = str(aug_id)
            nr["augmented"] = "0" if aug_id == 0 else "1"
            nr["trace_split"] = "train"
            nr["trace_row_in_source_split"] = str(base_row)

            out.append(nr)
            row_i += 1

    return out, final_fields


def make_eval_rows(base_rows, base_fields, target_len: int, split: str):
    base_len = len(base_rows)

    if base_len != target_len:
        raise ValueError(
            f"No calza {split}:\n"
            f"  source_index_len={base_len}\n"
            f"  target_len={target_len}\n"
            "Esto indica cambio de split u orden."
        )

    extra_fields = [
        "row_i",
        "base_row",
        "aug_id",
        "augmented",
        "trace_split",
        "trace_row_in_source_split",
    ]

    final_fields = []
    for c in extra_fields + base_fields:
        if c not in final_fields:
            final_fields.append(c)

    out = []
    for i, r in enumerate(base_rows):
        nr = {k: "" for k in final_fields}
        nr.update(r)

        nr["row_i"] = str(i)
        nr["base_row"] = str(i)
        nr["aug_id"] = "0"
        nr["augmented"] = "0"
        nr["trace_split"] = split
        nr["trace_row_in_source_split"] = str(i)

        out.append(nr)

    return out, final_fields


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_dir", required=True)
    ap.add_argument("--target_dir", required=True)
    ap.add_argument("--augment_times", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    source_dir = Path(args.source_dir).resolve()
    target_dir = Path(args.target_dir).resolve()

    print("[INFO] source_dir:", source_dir)
    print("[INFO] target_dir:", target_dir)

    source_rows = {}
    source_fields = {}

    for split in ["train", "val", "test"]:
        rows, fields = read_csv_rows(source_dir / f"index_{split}.csv")
        source_rows[split] = rows
        source_fields[split] = fields
        print(f"[SOURCE] {split}: index={len(rows)}")

    target_len = {}
    for split in ["train", "val", "test"]:
        x_len = npz_len(target_dir / f"X_{split}.npz", "X")
        y_len = npz_len(target_dir / f"Y_{split}.npz", "Y")

        if x_len != y_len:
            raise ValueError(f"{split}: X={x_len} distinto de Y={y_len}")

        target_len[split] = x_len
        print(f"[TARGET] {split}: X/Y={x_len}")

    train_rows, train_fields = expand_train_rows(
        source_rows["train"],
        source_fields["train"],
        target_len["train"],
        args.augment_times,
    )

    val_rows, val_fields = make_eval_rows(
        source_rows["val"],
        source_fields["val"],
        target_len["val"],
        "val",
    )

    test_rows, test_fields = make_eval_rows(
        source_rows["test"],
        source_fields["test"],
        target_len["test"],
        "test",
    )

    write_csv_rows(target_dir / "index_train.csv", train_rows, train_fields, args.overwrite)
    write_csv_rows(target_dir / "index_val.csv", val_rows, val_fields, args.overwrite)
    write_csv_rows(target_dir / "index_test.csv", test_rows, test_fields, args.overwrite)

    print("\n[DONE]")
    print(f"train: index={len(train_rows)} | X={target_len['train']} | OK={len(train_rows)==target_len['train']}")
    print(f"val:   index={len(val_rows)} | X={target_len['val']} | OK={len(val_rows)==target_len['val']}")
    print(f"test:  index={len(test_rows)} | X={target_len['test']} | OK={len(test_rows)==target_len['test']}")

    print("\n[HEAD index_test.csv]")
    for r in test_rows[:5]:
        print({k: r.get(k, "") for k in ["row_i", "idx", "sample_name", "jaw", "path", "aug_id"]})


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)