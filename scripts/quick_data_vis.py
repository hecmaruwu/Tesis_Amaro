#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, random, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

LABEL_COLORS = {
    0: 'red', 11: 'blue', 12: 'green', 13: 'orange', 14: 'purple', 15: 'cyan',
    16: 'magenta', 17: 'yellow', 18: 'brown', 21: 'lime', 22: 'navy', 23: 'teal',
    24: 'violet', 25: 'salmon', 26: 'gold', 27: 'lightblue', 28: 'coral',
    31: 'olive', 32: 'silver', 33: 'gray', 34: 'black', 35: 'darkred',
    36: 'darkgreen', 37: 'darkblue', 38: 'darkviolet', 41: 'peru',
    42: 'chocolate', 43: 'mediumvioletred', 44: 'lightskyblue', 45: 'lightpink',
    46: 'plum', 47: 'khaki', 48: 'powderblue',
}

def get_random_person(folder: Path):
    persons = [d for d in folder.iterdir() if d.is_dir()]
    if not persons:
        return None
    return random.choice(persons)

def load_point_cloud_and_labels(person_path: Path):
    pc = person_path/"point_cloud.npy"
    lb = person_path/"labels.npy"
    if pc.exists() and lb.exists():
        points = np.load(pc)
        labels = np.load(lb)
        if points.ndim == 2 and points.shape[1] == 3 and labels.ndim == 1:
            return points.astype(np.float32), labels
    return None, None

def visualize(ax, pts, labels, title):
    cols = np.array([LABEL_COLORS.get(int(l), "gray") for l in labels])
    ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=cols, s=1)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc_root", required=True, help="processed_struct/<P>/")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    root = Path(args.proc_root)
    upper = root / "upper"
    lower = root / "lower"
    assert upper.is_dir() and lower.is_dir(), f"No existen {upper} o {lower}"

    up_case = get_random_person(upper)
    lo_case = get_random_person(lower)
    print("[PICK]", up_case, lo_case)

    up_pts, up_lbl = load_point_cloud_and_labels(up_case)
    lo_pts, lo_lbl = load_point_cloud_and_labels(lo_case)

    out_dir = root / "quick_vis"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16,8), dpi=args.dpi)
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    if up_pts is not None:
        visualize(ax1, up_pts, up_lbl, f'Upper: {up_case.name}')
    if lo_pts is not None:
        visualize(ax2, lo_pts, lo_lbl, f'Lower: {lo_case.name}')
    fig.tight_layout()
    path = out_dir / "upper_lower_random.png"
    fig.savefig(path)
    plt.close(fig)
    print("[OK] Figura:", path)

if __name__ == "__main__":
    main()
