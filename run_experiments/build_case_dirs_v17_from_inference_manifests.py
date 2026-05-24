#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


DEFAULT_MODEL_INFERENCE_DIRS = {
    "pointnet": "/home/htaucare/Tesis_Amaro/outputs/pointnet_classic/grid_pro_runner_v1/exp03_bs16_lr3e4_do05_bg003_amp/inference",
    "pointnetpp": "/home/htaucare/Tesis_Amaro/outputs/pointnetpp/grid_pro_runner_v2/exp04_bs8_lr3e4_r010_020_040_ns32/inference",
    "dgcnn": "/home/htaucare/Tesis_Amaro/outputs/dgcnn/grid_pro_runner_v2_gpu1/exp06_bs8_lr2e4_k20_emb768_bg003/inference",
    "pointnettransformer": "/home/htaucare/Tesis_Amaro/outputs/pointnettransformer/grid_pro_runner_v2_gpu0/exp13_bs4_lr2e4_dm256_dep4_h8_ff512_do005/inference",
}

SELECTED_CASES = {
    "best": {
        "row_i": 91,
        "sample_name": "01MAVT6A",
        "jaw": "upper",
    },
    "median": {
        "row_i": 89,
        "sample_name": "015RHV4X",
        "jaw": "upper",
    },
    "worst": {
        "row_i": 71,
        "sample_name": "3EU06ZN9",
        "jaw": "upper",
    },
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_csv_dict(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_int(x) -> Optional[int]:
    try:
        return int(float(str(x).strip()))
    except Exception:
        return None


def resolve_manifest_path(value: str, inference_dir: Path) -> Path:
    if value is None:
        raise FileNotFoundError("Ruta vacía en manifest.")

    value = str(value).strip()
    if not value or value.lower() in {"nan", "none", "null"}:
        raise FileNotFoundError("Ruta vacía en manifest.")

    p = Path(value)

    candidates = []

    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend([
            inference_dir / p,
            inference_dir / "predictions" / p.name,
            inference_dir.parent / p,
            inference_dir.parent / "inference" / p,
            inference_dir.parent / "inference" / "predictions" / p.name,
        ])

    for cand in candidates:
        if cand.exists():
            return cand.resolve()

    raise FileNotFoundError(
        "No pude resolver ruta del manifest.\n"
        f"value={value}\n"
        f"inference_dir={inference_dir}\n"
        f"candidates:\n" + "\n".join(str(c) for c in candidates)
    )


def find_row_in_manifest(rows: List[dict], row_i: int, sample_name: str) -> dict:
    candidates = []

    for r in rows:
        ri = safe_int(r.get("row_i", ""))
        if ri == int(row_i):
            candidates.append(r)

    if candidates:
        if len(candidates) == 1:
            return candidates[0]

        sample_low = sample_name.lower()
        for r in candidates:
            text = " ".join(str(r.get(k, "")) for k in ["sample_name", "tag", "path"])
            if sample_low in text.lower():
                return r

        return candidates[0]

    sample_low = sample_name.lower()
    for r in rows:
        text = " ".join(str(r.get(k, "")) for k in ["sample_name", "tag", "path"])
        if sample_low in text.lower():
            return r

    raise RuntimeError(f"No encontré row_i={row_i} ni sample_name={sample_name} en manifest.")


def load_npy(path: Path) -> np.ndarray:
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.lib.npyio.NpzFile):
        key = list(arr.keys())[0]
        arr = arr[key]
    arr = np.asarray(arr)
    if arr.dtype == object:
        if arr.shape == ():
            arr = np.asarray(arr.item())
        elif arr.size == 1:
            arr = np.asarray(arr.reshape(-1)[0])
    return np.asarray(arr)


def copy_as_npy(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    arr = load_npy(src)
    np.save(dst, arr)


def find_raw_mesh(raw_root: Path, sample_name: str, jaw: str) -> Optional[Path]:
    raw_root = Path(raw_root)
    exts = ["obj", "ply", "stl"]

    patterns = []
    for ext in exts:
        patterns.extend([
            f"**/{jaw}/{sample_name}/{sample_name}_{jaw}.{ext}",
            f"**/{sample_name}/{sample_name}_{jaw}.{ext}",
            f"**/{sample_name}_{jaw}.{ext}",
        ])

    for pat in patterns:
        hits = sorted(raw_root.glob(pat))
        if hits:
            return hits[0].resolve()

    return None


def build_one_case(
    case_name: str,
    case_info: dict,
    out_root: Path,
    raw_root: Path,
    model_inference_dirs: Dict[str, str],
) -> Path:

    row_i = int(case_info["row_i"])
    sample_name = str(case_info["sample_name"])
    jaw = str(case_info.get("jaw", "upper"))

    case_dir = out_root / f"{case_name}_row{row_i:03d}_{sample_name}_{jaw}"
    ensure_dir(case_dir)

    case_records = []
    first_xyz = None
    first_gt = None

    for model_key, inf_dir_str in model_inference_dirs.items():
        inf_dir = Path(inf_dir_str)
        manifest_path = inf_dir / "inference_manifest.csv"

        if not manifest_path.exists():
            raise FileNotFoundError(f"No existe manifest para {model_key}: {manifest_path}")

        rows = read_csv_dict(manifest_path)
        row = find_row_in_manifest(rows, row_i=row_i, sample_name=sample_name)

        xyz_src = resolve_manifest_path(row.get("xyz_npy", ""), inf_dir)
        gt_src = resolve_manifest_path(row.get("gt_npy", ""), inf_dir)
        pred_src = resolve_manifest_path(row.get("pred_npy", ""), inf_dir)

        model_dir = ensure_dir(case_dir / model_key)

        copy_as_npy(xyz_src, model_dir / "xyz_labels.npy")
        copy_as_npy(gt_src, model_dir / "gt_labels.npy")
        copy_as_npy(pred_src, model_dir / "pred_labels.npy")

        if first_xyz is None:
            first_xyz = load_npy(xyz_src)
        if first_gt is None:
            first_gt = load_npy(gt_src)

        case_records.append({
            "model": model_key,
            "manifest": str(manifest_path),
            "row_i": row.get("row_i", ""),
            "sample_name": row.get("sample_name", ""),
            "tag": row.get("tag", ""),
            "xyz_src": str(xyz_src),
            "gt_src": str(gt_src),
            "pred_src": str(pred_src),
            "model_dir": str(model_dir),
        })

        print(f"[OK] {case_name} | {model_key} | row={row_i} | {sample_name}")

    common_dir = ensure_dir(case_dir / "common")
    if first_xyz is not None and first_gt is not None:
        np.savez_compressed(
            common_dir / "bundle_case_data.npz",
            xyz_8192=np.asarray(first_xyz),
            y_8192=np.asarray(first_gt),
            row_i=np.asarray(row_i),
            sample_name=np.asarray(sample_name),
            jaw=np.asarray(jaw),
        )

    raw_mesh = find_raw_mesh(raw_root=raw_root, sample_name=sample_name, jaw=jaw)

    meta = {
        "case_name": case_name,
        "row_i": row_i,
        "sample_name": sample_name,
        "jaw": jaw,
        "raw_mesh_path": str(raw_mesh) if raw_mesh else "",
        "source": "built_from_standardized_inference_manifest",
        "records": case_records,
    }

    save_json(meta, case_dir / "meta.json")

    if raw_mesh:
        print(f"[OK] raw mesh: {raw_mesh}")
    else:
        print(f"[WARN] No encontré raw mesh para {sample_name}_{jaw}. El script v17 correrá sin malla raw.")

    print(f"[DONE] case_dir construido: {case_dir}")
    return case_dir


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--out_root",
        default="/home/htaucare/Tesis_Amaro/case_comparisons/QUALI_CASES_V17",
    )

    ap.add_argument(
        "--raw_root",
        default="/home/htaucare/Tesis_Amaro/data/Teeth_3ds/raw",
    )

    ap.add_argument(
        "--cases",
        nargs="+",
        default=["best", "median", "worst"],
        choices=["best", "median", "worst"],
    )

    args = ap.parse_args()

    out_root = ensure_dir(Path(args.out_root))
    raw_root = Path(args.raw_root)

    made = []

    for case_name in args.cases:
        made_case = build_one_case(
            case_name=case_name,
            case_info=SELECTED_CASES[case_name],
            out_root=out_root,
            raw_root=raw_root,
            model_inference_dirs=DEFAULT_MODEL_INFERENCE_DIRS,
        )
        made.append(str(made_case))

    save_json(
        {
            "out_root": str(out_root),
            "raw_root": str(raw_root),
            "cases_built": made,
            "selected_cases": SELECTED_CASES,
            "model_inference_dirs": DEFAULT_MODEL_INFERENCE_DIRS,
        },
        out_root / "build_case_dirs_summary.json",
    )

    print("\n==============================")
    print("[DONE] Case dirs construidos:")
    for p in made:
        print("  ", p)
    print("==============================")


if __name__ == "__main__":
    main()
