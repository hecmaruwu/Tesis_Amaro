#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import hashlib

def canon_hash(x: np.ndarray, decimals: int = 4, take: int = 512) -> str:
    """
    Hash canónico tolerante:
    - redondea coords a 'decimals'
    - ordena puntos lexicográficamente
    - opcionalmente toma los primeros 'take' para acelerar
    """
    x = np.asarray(x, dtype=np.float32)
    xr = np.round(x, decimals=decimals)

    # ordenar por (x,y,z)
    order = np.lexsort((xr[:,2], xr[:,1], xr[:,0]))
    xr = xr[order]

    if take > 0 and xr.shape[0] > take:
        xr = xr[:take]

    return hashlib.sha1(xr.tobytes()).hexdigest()

def load_X(npz_path: Path) -> np.ndarray:
    arr = np.load(npz_path)["X"]
    return arr.astype(np.float32, copy=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--decimals", type=int, default=4)
    ap.add_argument("--take", type=int, default=512)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    splits = ["train", "val", "test"]

    H = {}
    for sp in splits:
        X = load_X(data_dir / f"X_{sp}.npz")
        hs = [canon_hash(X[i], decimals=args.decimals, take=args.take) for i in range(X.shape[0])]
        H[sp] = set(hs)
        print(f"[{sp}] hashes únicos (canónicos)={len(H[sp])} / B={X.shape[0]}")

    print("\n[CRUCES CANÓNICOS]")
    print(f"train ∩ val  = {len(H['train'] & H['val'])}")
    print(f"train ∩ test = {len(H['train'] & H['test'])}")
    print(f"val   ∩ test = {len(H['val'] & H['test'])}")

if __name__ == "__main__":
    main()
