#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_npz_array(path: Path, key: str):
    return np.load(path)[key]


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


def read_index_csv(index_path: Path) -> Dict[int, Dict[str, str]]:
    rows = read_csv_rows(index_path)
    if not rows:
        return {}
    fieldnames = list(rows[0].keys())

    row_key = pick_field(fieldnames, "row_i", "row", "i", "idx", "index")
    if row_key is None:
        return {}

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


def read_inference_manifest(manifest_path: Path) -> Dict[int, Dict[str, str]]:
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


def discover_index_csv(data_dir: Path, split: str) -> Optional[Path]:
    fname = f"index_{split}.csv"
    p = data_dir / fname
    if p.exists():
        return p
    cur = data_dir
    for _ in range(10):
        p = cur / fname
        if p.exists():
            return p
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


def sanitize(s: str, maxlen: int = 100) -> str:
    import re
    s = (s or "").strip().replace(" ", "_")
    s = re.sub(r"[^a-zA-Z0-9_\-\.]+", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:maxlen]


def resolve_original_path(raw_path: str, original_root: Optional[Path]) -> Tuple[Optional[Path], str]:
    raw_path = (raw_path or "").strip()
    if not raw_path:
        return None, "missing"

    p = Path(raw_path)
    if p.exists():
        return p.resolve(), "absolute_or_existing"

    if original_root is not None:
        cand = (original_root / raw_path).resolve()
        if cand.exists():
            return cand, "joined_with_original_root"

        try:
            base = Path(raw_path).name
            matches = list(original_root.rglob(base))
            if len(matches) == 1:
                return matches[0].resolve(), "basename_unique_match"
            elif len(matches) > 1:
                return matches[0].resolve(), f"basename_multiple_matches:{len(matches)}"
        except Exception:
            pass

    return None, "not_found"


def cloud_summary(xyz: np.ndarray, y: np.ndarray, bg_index: int = 0) -> Dict[str, Any]:
    xyz = np.asarray(xyz, dtype=np.float32)
    y = np.asarray(y).reshape(-1)

    xyz_min = xyz.min(axis=0).tolist()
    xyz_max = xyz.max(axis=0).tolist()
    centroid = xyz.mean(axis=0).tolist()
    unique, counts = np.unique(y, return_counts=True)
    hist = {int(k): int(v) for k, v in zip(unique.tolist(), counts.tolist())}
    bg_frac = float(hist.get(int(bg_index), 0) / max(1, len(y)))
    only_bg = (len(hist) == 1 and int(bg_index) in hist)

    return {
        "num_points": int(xyz.shape[0]),
        "bbox_min": [float(v) for v in xyz_min],
        "bbox_max": [float(v) for v in xyz_max],
        "centroid": [float(v) for v in centroid],
        "class_hist": hist,
        "bg_frac": float(bg_frac),
        "only_bg": bool(only_bg),
    }


def class_colors(C: int):
    cmap = plt.colormaps.get_cmap("tab20")
    C = max(int(C), 2)
    return [cmap(i / max(C - 1, 1)) for i in range(C)]


def plot_cloud_labels(xyz: np.ndarray, y: np.ndarray, out_png: Path, title: str = ""):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    xyz = np.asarray(xyz, dtype=np.float32)
    y = np.asarray(y).reshape(-1).astype(np.int32)
    C = int(np.max(y)) + 1 if y.size > 0 else 2
    cols = class_colors(C)
    c = np.array([cols[int(k)] for k in y], dtype=np.float32)

    fig = plt.figure(figsize=(7, 6), dpi=220)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=c, s=1.0, linewidths=0, depthshade=False)
    ax.set_axis_off()
    ax.view_init(elev=20, azim=45)
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


def plot_cloud_focus_class(xyz: np.ndarray, y: np.ndarray, focus_class: int, out_png: Path, title: str = ""):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    xyz = np.asarray(xyz, dtype=np.float32)
    y = np.asarray(y).reshape(-1).astype(np.int32)

    c = np.zeros((xyz.shape[0], 4), dtype=np.float32)
    c[:, :] = (0.75, 0.75, 0.75, 0.35)
    c[y == int(focus_class), :] = (0.85, 0.10, 0.10, 1.0)

    fig = plt.figure(figsize=(7, 6), dpi=220)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=c, s=1.2, linewidths=0, depthshade=False)
    ax.set_axis_off()
    ax.view_init(elev=20, azim=45)
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


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
        "missing_rows": missing[:50],
        "extra_rows": extra[:50],
    }


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


def audit_rows(
    X: np.ndarray,
    Y: np.ndarray,
    split: str,
    index_map: Dict[int, Dict[str, str]],
    manifest_map: Dict[int, Dict[str, str]],
    rows: List[int],
    out_dir: Path,
    original_root: Optional[Path],
    d21_class: int,
    bg_index: int,
) -> List[Dict[str, Any]]:
    rows_out = []
    vis_dir = ensure_dir(out_dir / "visuals")

    for ri in rows:
        rec: Dict[str, Any] = {"row_i": int(ri), "split": str(split)}

        xyz = X[int(ri)]
        y = Y[int(ri)]
        rec["npz_summary"] = cloud_summary(xyz, y, bg_index=bg_index)

        idx_meta = index_map.get(int(ri), {})
        rec["index_meta"] = idx_meta
        rec["has_index_entry"] = bool(idx_meta)

        orig_path, resolve_mode = resolve_original_path(idx_meta.get("path", ""), original_root)
        rec["original_path_resolved"] = str(orig_path) if orig_path is not None else ""
        rec["original_path_resolution_mode"] = resolve_mode
        rec["original_exists"] = bool(orig_path is not None and orig_path.exists())

        mani_meta = manifest_map.get(int(ri), {})
        rec["manifest_meta"] = mani_meta
        rec["manifest_check"] = compare_index_vs_manifest(idx_meta, mani_meta)

        rec["verdict_index_link_ok"] = bool(idx_meta)
        rec["verdict_original_link_ok"] = bool(rec["original_exists"])
        rec["verdict_manifest_link_ok"] = bool(rec["manifest_check"].get("has_manifest_entry", False))
        rec["verdict_index_vs_manifest_ok"] = bool(
            rec["manifest_check"].get("has_manifest_entry", False)
            and rec["manifest_check"].get("match_sample_name", True)
            and rec["manifest_check"].get("match_jaw", True)
            and rec["manifest_check"].get("match_path", True)
        ) if mani_meta else False

        sample_name = idx_meta.get("sample_name", f"row{ri}")
        jaw = idx_meta.get("jaw", "")
        tag = f"{split}_row{ri}_{sanitize(sample_name)}"
        if jaw:
            tag += f"_{sanitize(jaw)}"

        png_labels = vis_dir / f"{tag}_labels.png"
        png_d21 = vis_dir / f"{tag}_d21.png"

        title = f"{split} row={ri}"
        if sample_name:
            title += f" | sample={sample_name}"
        if jaw:
            title += f" | jaw={jaw}"

        plot_cloud_labels(xyz, y, png_labels, title=title)
        plot_cloud_focus_class(xyz, y, focus_class=int(d21_class), out_png=png_d21, title=title + f" | d21={d21_class}")

        rec["audit_png_labels"] = str(png_labels)
        rec["audit_png_d21"] = str(png_d21)

        rows_out.append(rec)

    return rows_out


def parse_args():
    p = argparse.ArgumentParser(description="Auditoría de trazabilidad NPZ -> index_csv -> original -> inference")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--index_csv", type=str, default=None)
    p.add_argument("--inference_dir", type=str, default=None)
    p.add_argument("--original_root", type=str, default=None)
    p.add_argument("--out_dir", type=str, required=True)

    p.add_argument("--rows", type=int, nargs="*", default=None)
    p.add_argument("--sample_n", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--d21_class", type=int, default=8)
    p.add_argument("--bg_index", type=int, default=0)
    p.add_argument("--strict", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir = ensure_dir(Path(args.out_dir).resolve())

    Xp = data_dir / f"X_{args.split}.npz"
    Yp = data_dir / f"Y_{args.split}.npz"

    if not Xp.exists() or not Yp.exists():
        raise FileNotFoundError(f"No existen {Xp} o {Yp}")

    X = load_npz_array(Xp, "X")
    Y = load_npz_array(Yp, "Y")

    n_rows = int(X.shape[0])
    assert int(Y.shape[0]) == n_rows, "Mismatch X/Y"

    index_csv = Path(args.index_csv).resolve() if args.index_csv else discover_index_csv(data_dir, args.split)
    if index_csv is None or (not index_csv.exists()):
        raise FileNotFoundError("No se encontró index_csv. Use --index_csv explícitamente.")

    inference_dir = Path(args.inference_dir).resolve() if args.inference_dir else None
    manifest_path = (inference_dir / "inference_manifest.csv") if inference_dir and (inference_dir / "inference_manifest.csv").exists() else None
    original_root = Path(args.original_root).resolve() if args.original_root else None

    index_map = read_index_csv(index_csv)
    manifest_map = read_inference_manifest(manifest_path) if manifest_path else {}

    global_summary: Dict[str, Any] = {
        "data_dir": str(data_dir),
        "split": str(args.split),
        "X_shape": list(X.shape),
        "Y_shape": list(Y.shape),
        "index_csv": str(index_csv),
        "inference_dir": str(inference_dir) if inference_dir else "",
        "inference_manifest": str(manifest_path) if manifest_path else "",
        "original_root": str(original_root) if original_root else "",
        "d21_class": int(args.d21_class),
        "bg_index": int(args.bg_index),
    }
    global_summary["index_alignment_check"] = contiguous_row_check(index_map, n_rows)

    if args.rows:
        rows = [int(r) for r in args.rows]
    else:
        rng = random.Random(int(args.seed))
        k = int(args.sample_n) if int(args.sample_n) > 0 else min(10, n_rows)
        rows = sorted(rng.sample(list(range(n_rows)), k=min(k, n_rows)))

    bad_rows = [r for r in rows if r < 0 or r >= n_rows]
    if bad_rows:
        raise ValueError(f"rows fuera de rango: {bad_rows[:20]}")

    audited = audit_rows(
        X=X,
        Y=Y,
        split=str(args.split),
        index_map=index_map,
        manifest_map=manifest_map,
        rows=rows,
        out_dir=out_dir,
        original_root=original_root,
        d21_class=int(args.d21_class),
        bg_index=int(args.bg_index),
    )

    table_rows = []
    for rec in audited:
        idx_meta = rec.get("index_meta", {})
        table_rows.append({
            "row_i": rec["row_i"],
            "sample_name": idx_meta.get("sample_name", ""),
            "jaw": idx_meta.get("jaw", ""),
            "path": idx_meta.get("path", ""),
            "original_exists": rec.get("original_exists", False),
            "index_link_ok": rec.get("verdict_index_link_ok", False),
            "manifest_link_ok": rec.get("verdict_manifest_link_ok", False),
            "index_vs_manifest_ok": rec.get("verdict_index_vs_manifest_ok", False),
            "only_bg": rec["npz_summary"]["only_bg"],
            "bg_frac": rec["npz_summary"]["bg_frac"],
            "num_points": rec["npz_summary"]["num_points"],
            "audit_png_labels": rec.get("audit_png_labels", ""),
            "audit_png_d21": rec.get("audit_png_d21", ""),
        })

    report = {"summary": global_summary, "audited_rows": audited}

    if args.strict:
        problems = []
        align = global_summary["index_alignment_check"]
        if not align["ok_exact_0_to_n_minus_1"]:
            problems.append("index_csv no coincide exactamente con 0..N-1")
        for rec in audited:
            if not rec.get("verdict_index_link_ok", False):
                problems.append(f"row {rec['row_i']}: sin index entry")
            if not rec.get("original_exists", False):
                problems.append(f"row {rec['row_i']}: original no resuelto/encontrado")
        if problems:
            save_json(report, out_dir / "traceability_report.json")
            with (out_dir / "traceability_table.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(table_rows[0].keys()) if table_rows else ["row_i"])
                writer.writeheader()
                if table_rows:
                    writer.writerows(table_rows)
            raise RuntimeError("STRICT MODE FAILED:\n- " + "\n- ".join(problems))

    save_json(report, out_dir / "traceability_report.json")

    with (out_dir / "traceability_table.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(table_rows[0].keys()) if table_rows else ["row_i"])
        writer.writeheader()
        if table_rows:
            writer.writerows(table_rows)

    txt_lines = []
    txt_lines.append("TRACEABILITY AUDIT SUMMARY")
    txt_lines.append("=" * 80)
    txt_lines.append(f"data_dir: {data_dir}")
    txt_lines.append(f"split: {args.split}")
    txt_lines.append(f"X shape: {X.shape}")
    txt_lines.append(f"Y shape: {Y.shape}")
    txt_lines.append(f"index_csv: {index_csv}")
    txt_lines.append(f"inference_manifest: {manifest_path if manifest_path else ''}")
    txt_lines.append(f"original_root: {original_root if original_root else ''}")
    txt_lines.append("")
    txt_lines.append("INDEX ALIGNMENT CHECK")
    txt_lines.append(str(global_summary["index_alignment_check"]))
    txt_lines.append("")
    txt_lines.append("AUDITED ROWS")
    for row in table_rows:
        txt_lines.append(
            f"row={row['row_i']} | sample={row['sample_name']} | jaw={row['jaw']} | "
            f"orig_exists={row['original_exists']} | idx_ok={row['index_link_ok']} | "
            f"manifest_ok={row['manifest_link_ok']} | idx_vs_manifest_ok={row['index_vs_manifest_ok']} | "
            f"only_bg={row['only_bg']} | bg_frac={row['bg_frac']:.4f}"
        )

    (out_dir / "traceability_summary.txt").write_text("\n".join(txt_lines), encoding="utf-8")

    print(f"[OK] Reporte JSON: {out_dir / 'traceability_report.json'}")
    print(f"[OK] Tabla CSV   : {out_dir / 'traceability_table.csv'}")
    print(f"[OK] Resumen TXT : {out_dir / 'traceability_summary.txt'}")
    print(f"[OK] Visuales    : {out_dir / 'visuals'}")


if __name__ == "__main__":
    main()
