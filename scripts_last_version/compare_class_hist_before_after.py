#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_all_Y(base: Path):
    Ys = []
    for s in ["train", "val", "test"]:
        p = base / f"Y_{s}.npz"
        if not p.exists():
            raise FileNotFoundError(str(p))
        Ys.append(np.load(p)["Y"])
    return Ys


def load_label_map(base: Path):
    lm = base / "artifacts" / "label_map.json"
    if not lm.exists():
        return None
    data = json.load(open(lm, "r", encoding="utf-8"))
    id2idx = data.get("id2idx", None)
    idx2id = data.get("idx2id", None)
    if isinstance(id2idx, dict):
        id2idx = {str(k): int(v) for k, v in id2idx.items()}
    if isinstance(idx2id, dict):
        idx2id = {str(k): int(v) for k, v in idx2id.items()}
    return id2idx, idx2id


def counts_from_Ylist(Y_list):
    flat = np.concatenate([Y.reshape(-1) for Y in Y_list], axis=0)
    u, c = np.unique(flat, return_counts=True)
    return flat.size, u, c


def _filter_nonzero(u, c):
    m = (u != 0)
    return u[m], c[m]


def plot_hist_counts(u, c, title, out_png: Path, logy=False):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 4), dpi=180)
    plt.bar(u, c)
    plt.xlabel("class id (internal)")
    plt.ylabel("#points")
    plt.title(title + (" (log y)" if logy else ""))
    if logy:
        plt.yscale("log")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_hist_percent(u, pct, title, out_png: Path, ylabel="%"):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 4), dpi=180)
    plt.bar(u, pct)
    plt.xlabel("class id (internal)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def presence_by_sample(Y: np.ndarray, num_classes: int):
    # Y: [B,N]
    B = Y.shape[0]
    pres = np.zeros((num_classes,), dtype=np.int64)
    for k in range(num_classes):
        pres[k] = int(np.any(Y == k, axis=1).sum())
    return pres, B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before_dir", required=True)
    ap.add_argument("--after_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    before = Path(args.before_dir)
    after  = Path(args.after_dir)
    out    = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- BEFORE ---
    Yb_list = load_all_Y(before)
    tot_b, u_b, c_b = counts_from_Ylist(Yb_list)
    bg_b = 100.0 * c_b[u_b == 0][0] / tot_b if np.any(u_b == 0) else 0.0

    # --- AFTER ---
    Ya_list = load_all_Y(after)
    tot_a, u_a, c_a = counts_from_Ylist(Ya_list)
    bg_a = 100.0 * c_a[u_a == 0][0] / tot_a if np.any(u_a == 0) else 0.0

    # num_classes (para presence)
    C = int(max(u_a.max(), u_b.max())) + 1

    # ---------- PRINT TABLES (NONZERO ONLY, %TOTAL and %FG) ----------
    def print_block(tag, tot, u, c, bg_pct):
        u_nz, c_nz = _filter_nonzero(u, c)
        fg = int(c_nz.sum())
        fg_pct = 100.0 * fg / tot if tot > 0 else 0.0

        print(f"[{tag}] tot_points={tot} unique={len(u)} bg={bg_pct:.2f}% | fg={fg_pct:.2f}% | C~{C}")
        if u_nz.size == 0:
            print("  (no nonzero classes found)")
            return

        print("  (showing classes != 0; %TOTAL uses all points incl bg; %FG is within nonzero only)")
        for ui, ci in zip(u_nz, c_nz):
            pct_total = 100.0 * float(ci) / float(tot)
            pct_fg = 100.0 * float(ci) / float(fg) if fg > 0 else 0.0
            print(f"  class {int(ui):>3d}: {int(ci):>12d} | %TOTAL={pct_total:6.2f}% | %FG={pct_fg:6.2f}%")

    print_block("BEFORE", tot_b, u_b, c_b, bg_b)
    print("")
    print_block("AFTER",  tot_a, u_a, c_a, bg_a)

    # ---------- PLOTS (NONZERO ONLY) ----------
    u_b_nz, c_b_nz = _filter_nonzero(u_b, c_b)
    u_a_nz, c_a_nz = _filter_nonzero(u_a, c_a)

    # counts hist
    plot_hist_counts(u_b_nz, c_b_nz,
        f"BEFORE (classes != 0). bg={bg_b:.2f}% | C~{C}",
        out / "hist_before_nonzero_counts_linear.png", logy=False)
    plot_hist_counts(u_b_nz, c_b_nz,
        f"BEFORE (classes != 0). bg={bg_b:.2f}% | C~{C}",
        out / "hist_before_nonzero_counts_log.png", logy=True)

    plot_hist_counts(u_a_nz, c_a_nz,
        f"AFTER (classes != 0). bg={bg_a:.2f}% | C~{C}",
        out / "hist_after_nonzero_counts_linear.png", logy=False)
    plot_hist_counts(u_a_nz, c_a_nz,
        f"AFTER (classes != 0). bg={bg_a:.2f}% | C~{C}",
        out / "hist_after_nonzero_counts_log.png", logy=True)

    # %TOTAL (nonzero classes, but denominator is total incl bg)
    if u_b_nz.size > 0:
        pct_b_total = (c_b_nz.astype(np.float64) / float(tot_b)) * 100.0
        plot_hist_percent(
            u_b_nz, pct_b_total,
            f"BEFORE (classes != 0): % of TOTAL points (incl bg). bg={bg_b:.2f}%",
            out / "hist_before_nonzero_percent_total.png",
            ylabel="% of TOTAL"
        )
    if u_a_nz.size > 0:
        pct_a_total = (c_a_nz.astype(np.float64) / float(tot_a)) * 100.0
        plot_hist_percent(
            u_a_nz, pct_a_total,
            f"AFTER (classes != 0): % of TOTAL points (incl bg). bg={bg_a:.2f}%",
            out / "hist_after_nonzero_percent_total.png",
            ylabel="% of TOTAL"
        )

    # %FG (nonzero classes only; sums to 100%)
    fg_b = float(c_b_nz.sum()) if c_b_nz.size > 0 else 0.0
    fg_a = float(c_a_nz.sum()) if c_a_nz.size > 0 else 0.0

    if u_b_nz.size > 0 and fg_b > 0:
        pct_b_fg = (c_b_nz.astype(np.float64) / fg_b) * 100.0
        plot_hist_percent(
            u_b_nz, pct_b_fg,
            "BEFORE (classes != 0): % within FG only (sum=100%)",
            out / "hist_before_nonzero_percent_fg.png",
            ylabel="% of FG (nonzero)"
        )

    if u_a_nz.size > 0 and fg_a > 0:
        pct_a_fg = (c_a_nz.astype(np.float64) / fg_a) * 100.0
        plot_hist_percent(
            u_a_nz, pct_a_fg,
            "AFTER (classes != 0): % within FG only (sum=100%)",
            out / "hist_after_nonzero_percent_fg.png",
            ylabel="% of FG (nonzero)"
        )

    # ---------- presence counts (after) ----------
    lm = load_label_map(after)
    d21_internal = None
    if lm is not None:
        id2idx, idx2id = lm
        d21_internal = id2idx.get("21", None)

    for split in ["train", "val", "test"]:
        Y = np.load(after / f"Y_{split}.npz")["Y"]
        pres, B = presence_by_sample(Y, C)
        if d21_internal is not None and 0 <= int(d21_internal) < C:
            print(f"\n[AFTER {split}] B={B} | samples with d21(internal={d21_internal}) = {pres[int(d21_internal)]}")
        else:
            print(f"\n[AFTER {split}] B={B} | d21 not found in label_map.json")

    # ---------- “en cuántos pacientes está el 21” sin contaminar por augment ----------
    lm_b = load_label_map(before)
    d21_b = None
    if lm_b is not None:
        d21_b = lm_b[0].get("21", None)

    if d21_b is not None:
        for split in ["train", "val", "test"]:
            Y = np.load(before / f"Y_{split}.npz")["Y"]
            pres, B = presence_by_sample(Y, C)
            print(f"[BEFORE {split}] B={B} | samples with d21(internal={d21_b}) = {pres[int(d21_b)]}")

    print(f"\n[DONE] outputs en: {out}")


if __name__ == "__main__":
    main()
