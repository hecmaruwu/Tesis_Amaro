#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import hashlib

def hash_sample_bytes(x: np.ndarray) -> str:
    # x: [N,3] float32
    # hash exacto del buffer
    return hashlib.sha1(x.tobytes()).hexdigest()

def load_X(npz_path: Path) -> np.ndarray:
    arr = np.load(npz_path)["X"]
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    return arr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = todos; si no, limita muestras por split")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)

    splits = ["train", "val", "test"]
    hashes = {}  # split -> list[str]
    counts = {}  # split -> dict[hash]->count
    idxs = {}    # split -> dict[hash]->list[i]

    for sp in splits:
        Xp = data_dir / f"X_{sp}.npz"
        X = load_X(Xp)
        B = X.shape[0]
        L = B if args.limit <= 0 else min(B, args.limit)

        counts_sp = {}
        idxs_sp = {}
        hs = []
        for i in range(L):
            h = hash_sample_bytes(X[i])
            hs.append(h)
            counts_sp[h] = counts_sp.get(h, 0) + 1
            idxs_sp.setdefault(h, []).append(i)

        hashes[sp] = hs
        counts[sp] = counts_sp
        idxs[sp] = idxs_sp

        dup_in_split = sum(1 for h,c in counts_sp.items() if c > 1)
        max_rep = max(counts_sp.values()) if counts_sp else 0
        print(f"[{sp}] B={B} (usado={L}) | hashes únicos={len(counts_sp)} | duplicados internos={dup_in_split} | max_repeticiones={max_rep}")

    # Cruces entre splits
    set_tr = set(hashes["train"])
    set_va = set(hashes["val"])
    set_te = set(hashes["test"])

    inter_tr_va = set_tr & set_va
    inter_tr_te = set_tr & set_te
    inter_va_te = set_va & set_te

    print("\n[CRUCES]")
    print(f"train ∩ val  = {len(inter_tr_va)}")
    print(f"train ∩ test = {len(inter_tr_te)}")
    print(f"val   ∩ test = {len(inter_va_te)}")

    # Mostrar algunos ejemplos (indices)
    def show_examples(title, inter, a, b, k=10):
        if not inter:
            return
        print(f"\n{title} (mostrando hasta {k})")
        for j, h in enumerate(list(inter)[:k], start=1):
            ia = idxs[a].get(h, [])
            ib = idxs[b].get(h, [])
            print(f"  #{j:02d} hash={h[:12]}... | {a} idx={ia[:5]} | {b} idx={ib[:5]}")

    show_examples("EJEMPLOS train∩val", inter_tr_va, "train", "val")
    show_examples("EJEMPLOS train∩test", inter_tr_te, "train", "test")
    show_examples("EJEMPLOS val∩test", inter_va_te, "val", "test")

if __name__ == "__main__":
    main()
