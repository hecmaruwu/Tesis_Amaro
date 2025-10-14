#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Selecciona puntos de un diente por ID desde los NPZ (usa Y del split).
Exporta PLY en results/inference/<split>/case_<idx>_tooth<id>.ply
"""

import os, json, argparse
from pathlib import Path
import numpy as np

def save_ply_xyz(path: Path, xyz: np.ndarray):
    xyz = np.asarray(xyz, dtype=np.float32)
    N = xyz.shape[0]
    with open(path, "wb") as f:
        header = (
            "ply\nformat ascii 1.0\n"
            f"element vertex {N}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "end_header\n"
        )
        f.write(header.encode("ascii"))
        for p in xyz:
            f.write(f"{p[0]} {p[1]} {p[2]}\n".encode("ascii"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", required=True, help="Carpeta con X_*.npz, Y_*.npz, meta.json")
    ap.add_argument("--split", choices=["train","val","test"], default="test")
    ap.add_argument("--index", type=int, default=0, help="Índice dentro del split")
    ap.add_argument("--tooth_id", type=int, default=21)
    ap.add_argument("--out_dir", default="results/inference")
    args = ap.parse_args()

    root = Path(args.npz_dir)
    X = np.load(root / f"X_{args.split}.npz")["X"]     # [N, P, 3]
    Y = np.load(root / f"Y_{args.split}.npz")["Y"]     # [N, P]
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))

    # Recupera metadata del caso concreto
    if args.split == "train":
        cases = meta["cases_train"]
    elif args.split == "val":
        cases = meta["cases_val"]
    else:
        cases = meta["cases_test"]

    idx = args.index
    if not (0 <= idx < X.shape[0]):
        raise SystemExit(f"Index fuera de rango (0..{X.shape[0]-1})")

    pts = X[idx]          # [P,3]
    lbs = Y[idx]          # [P]
    mask = (lbs == args.tooth_id)
    sel = pts[mask]

    out_dir = Path(args.out_dir) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    case = cases[idx]
    name = f"{case['pid']}_{case['jaw']}_{case['case_id']}_tooth{args.tooth_id}.ply"
    path = out_dir / name
    if sel.shape[0] == 0:
        print(f"[WARN] No se encontraron puntos con etiqueta {args.tooth_id} en idx={idx}. Exporto nube vacía.")
    save_ply_xyz(path, sel)
    print("[OK] Exportado:", path)

if __name__ == "__main__":
    main()
