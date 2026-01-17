#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_npz(path: Path, key: str):
    obj = np.load(path)
    if key not in obj:
        raise KeyError(f"Key '{key}' no existe en {path}. Keys={list(obj.keys())}")
    return obj[key]


def load_label_map(artifacts_dir: Path):
    p = artifacts_dir / "label_map.json"
    if not p.exists():
        return None, None
    data = json.load(open(p, "r", encoding="utf-8"))
    id2idx = data.get("id2idx", None)
    idx2id = data.get("idx2id", None)
    if isinstance(id2idx, dict):
        id2idx = {str(k): int(v) for k, v in id2idx.items()}
    else:
        id2idx = None
    if isinstance(idx2id, dict):
        idx2id = {str(k): int(v) for k, v in idx2id.items()}
    else:
        idx2id = None
    return id2idx, idx2id


def hist_from_Y(Y: np.ndarray, num_classes=None):
    y = Y.reshape(-1).astype(np.int64)
    if num_classes is None:
        num_classes = int(y.max()) + 1
    h = np.bincount(y, minlength=num_classes)
    return h


def per_sample_presence(Y: np.ndarray, cls: int):
    # Y: [B,N]
    return np.any(Y == cls, axis=1)


def scan_processed_struct_upper(processed_struct_upper: Path, tooth_id: int = 21):
    """
    Cuenta:
      - #pacientes (folders)
      - #pacientes con tooth_id presente en labels.npy
      - distribución de labels originales (conteo de puntos total)
    """
    processed_struct_upper = Path(processed_struct_upper)
    pids = sorted([p for p in processed_struct_upper.iterdir() if p.is_dir()])
    total_pids = len(pids)

    has_tooth = 0
    global_labels = []

    for pid_dir in pids:
        y_path = pid_dir / "labels.npy"
        x_path = pid_dir / "point_cloud.npy"
        if (not y_path.exists()) or (not x_path.exists()):
            continue
        y = np.load(y_path).astype(np.int32).reshape(-1)
        global_labels.append(y)

        if np.any(y == int(tooth_id)):
            has_tooth += 1

    if global_labels:
        all_y = np.concatenate(global_labels, axis=0)
        uniq, cnt = np.unique(all_y, return_counts=True)
        dist = {int(u): int(c) for u, c in zip(uniq, cnt)}
    else:
        dist = {}

    return {
        "total_pid_dirs": total_pids,
        "pids_with_tooth": int(has_tooth),
        "pids_without_tooth": int(max(0, total_pids - has_tooth)),
        "label_dist_original_counts": dist,
    }


def save_hist_plot(hist, out_png: Path, title: str):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    xs = np.arange(len(hist))
    plt.figure(figsize=(10, 4))
    plt.bar(xs, hist)
    plt.title(title)
    plt.xlabel("class id")
    plt.ylabel("#points")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_struct_upper", required=True,
                    help=".../processed_struct_safe/200000/upper")
    ap.add_argument("--merged_dir", required=True,
                    help=".../merged_200000_safe_excl_wisdom_upper_only (tiene X_train/Y_train etc 200k)")
    ap.add_argument("--final_dir", required=True,
                    help=".../fixed_split/8192/..._aug2 (tiene X_train/Y_train etc 8192)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tooth_id", type=int, default=21)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------- A) “21 por paciente” desde processed_struct (ORIGINAL IDS) --------
    rep = scan_processed_struct_upper(Path(args.processed_struct_upper), tooth_id=args.tooth_id)
    json.dump(rep, open(out_dir / "report_processed_struct_upper.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    # -------- B) Histograma “ANTES” (merged 200k, YA REMAPEADO 0..C-1) --------
    merged_dir = Path(args.merged_dir)
    Ym_tr = load_npz(merged_dir / "Y_train.npz", "Y")  # [B,200000]
    Ym_va = load_npz(merged_dir / "Y_val.npz", "Y")
    Ym_te = load_npz(merged_dir / "Y_test.npz", "Y")

    num_classes_merged = int(max(Ym_tr.max(), Ym_va.max(), Ym_te.max())) + 1
    hm = hist_from_Y(np.concatenate([Ym_tr, Ym_va, Ym_te], axis=0), num_classes=num_classes_merged)
    bg_pct_merged = float(hm[0] / (hm.sum() + 1e-12) * 100.0)

    # presencia de clase del 21: OJO, en merged ya es “idx interno”
    # lo resolvemos usando label_map del merged (si existe)
    id2idx_m, _ = load_label_map(merged_dir / "artifacts")
    if id2idx_m is not None and str(args.tooth_id) in id2idx_m:
        d21_internal_merged = int(id2idx_m[str(args.tooth_id)])
    else:
        d21_internal_merged = None

    if d21_internal_merged is not None:
        present_m_tr = per_sample_presence(Ym_tr, d21_internal_merged).sum()
        present_m_va = per_sample_presence(Ym_va, d21_internal_merged).sum()
        present_m_te = per_sample_presence(Ym_te, d21_internal_merged).sum()
    else:
        present_m_tr = present_m_va = present_m_te = -1

    save_hist_plot(hm, out_dir / "hist_merged_200k_all.png",
                   f"Merged 200k (all splits). bg={bg_pct_merged:.2f}% | C={len(hm)}")

    # -------- C) Histograma “DESPUÉS” (final 8192, YA REMAPEADO 0..C-1) --------
    final_dir = Path(args.final_dir)
    Yf_tr = load_npz(final_dir / "Y_train.npz", "Y")  # [B,8192]
    Yf_va = load_npz(final_dir / "Y_val.npz", "Y")
    Yf_te = load_npz(final_dir / "Y_test.npz", "Y")

    num_classes_final = int(max(Yf_tr.max(), Yf_va.max(), Yf_te.max())) + 1
    hf = hist_from_Y(np.concatenate([Yf_tr, Yf_va, Yf_te], axis=0), num_classes=num_classes_final)
    bg_pct_final = float(hf[0] / (hf.sum() + 1e-12) * 100.0)

    id2idx_f, _ = load_label_map(final_dir / "artifacts")
    if id2idx_f is not None and str(args.tooth_id) in id2idx_f:
        d21_internal_final = int(id2idx_f[str(args.tooth_id)])
    else:
        d21_internal_final = None

    if d21_internal_final is not None:
        present_f_tr = per_sample_presence(Yf_tr, d21_internal_final).sum()
        present_f_va = per_sample_presence(Yf_va, d21_internal_final).sum()
        present_f_te = per_sample_presence(Yf_te, d21_internal_final).sum()
    else:
        present_f_tr = present_f_va = present_f_te = -1

    save_hist_plot(hf, out_dir / "hist_final_8192_all.png",
                   f"Final 8192 (all splits). bg={bg_pct_final:.2f}% | C={len(hf)}")

    # -------- D) Reporte resumido --------
    summary = {
        "tooth_id_requested": int(args.tooth_id),

        "processed_struct_upper": rep,

        "merged_200k": {
            "C_internal": int(num_classes_merged),
            "bg_pct_all_points": float(bg_pct_merged),
            "d21_internal": (int(d21_internal_merged) if d21_internal_merged is not None else None),
            "samples_total": int(Ym_tr.shape[0] + Ym_va.shape[0] + Ym_te.shape[0]),
            "samples_with_d21_train": int(present_m_tr),
            "samples_with_d21_val": int(present_m_va),
            "samples_with_d21_test": int(present_m_te),
        },

        "final_8192": {
            "C_internal": int(num_classes_final),
            "bg_pct_all_points": float(bg_pct_final),
            "d21_internal": (int(d21_internal_final) if d21_internal_final is not None else None),
            "train_shape": list(Yf_tr.shape),
            "val_shape": list(Yf_va.shape),
            "test_shape": list(Yf_te.shape),
            "samples_with_d21_train": int(present_f_tr),
            "samples_with_d21_val": int(present_f_va),
            "samples_with_d21_test": int(present_f_te),
        },
    }
    json.dump(summary, open(out_dir / "summary.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print("[OK] EDA listo.")
    print(" -", out_dir / "summary.json")
    print(" -", out_dir / "hist_merged_200k_all.png")
    print(" -", out_dir / "hist_final_8192_all.png")
    print(" -", out_dir / "report_processed_struct_upper.json")


if __name__ == "__main__":
    main()
