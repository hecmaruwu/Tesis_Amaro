#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Suaviza etiquetas Y por muestra usando votación mayoritaria k-NN.
Lee X_{train,val,test}.npy/npz y Y_{...}.npy/npz (X:(S,P,3), Y:(S,P)).
Guarda SIEMPRE .npz con claves "X"/"Y" en --out_path.
"""
import os, argparse, shutil
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

SPLITS = ("train","val","test")

def _load_array(path: Path, key_candidates=("X","Y","arr_0")):
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".npz":
        with np.load(path) as z:
            for k in key_candidates:
                if k in z: return z[k]
            return z[list(z.keys())[0]]
    raise ValueError(f"Extensión no soportada: {path}")

def _save_npz(path: Path, key: str, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{key: arr})

def row_mode_bincount(A: np.ndarray) -> np.ndarray:
    """Modo por fila con bincount (A: (P,k), ints >=0)."""
    P = A.shape[0]
    out = np.empty(P, dtype=A.dtype)
    for i in range(P):
        out[i] = np.argmax(np.bincount(A[i]))
    return out

def smooth_one_sample(X: np.ndarray, Y: np.ndarray, k: int, iters: int) -> np.ndarray:
    Y_cur = Y.copy()
    for _ in range(max(1, iters)):
        tree = cKDTree(X.astype(np.float32))
        kk = min(k, X.shape[0])
        _, idx = tree.query(X.astype(np.float32), k=kk)
        if kk == 1: break
        neigh_labels = Y_cur[idx]          # (P,kk)
        Y_cur = row_mode_bincount(neigh_labels)
    return Y_cur

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True, help="Carpeta con X_*.npy/npz y Y_*.npy/npz")
    ap.add_argument("--out_path",  required=True, help="Carpeta destino para split suavizado")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--iters", type=int, default=1)
    args = ap.parse_args()

    IN  = Path(args.data_path)
    OUT = Path(args.out_path)
    OUT.mkdir(parents=True, exist_ok=True)

    # Copia artifacts/ si existe
    if (IN/"artifacts").exists():
        shutil.copytree(IN/"artifacts", OUT/"artifacts", dirs_exist_ok=True)

    for s in SPLITS:
        xnp = next((IN/f"X_{s}{ext}" for ext in (".npz",".npy") if (IN/f"X_{s}{ext}").exists()), None)
        ynp = next((IN/f"Y_{s}{ext}" for ext in (".npz",".npy") if (IN/f"Y_{s}{ext}").exists()), None)
        if xnp is None or ynp is None:
            print(f"[WARN] split {s} no encontrado completo (X/Y). Salto.")
            continue

        X = _load_array(xnp, key_candidates=("X","arr_0"))  # (S,P,3)
        Y = _load_array(ynp, key_candidates=("Y","arr_0"))  # (S,P)
        assert X.ndim==3 and X.shape[:2]==Y.shape[:2], f"Shape mismatch en {s}: X{X.shape} Y{Y.shape}"
        S, P, _ = X.shape

        Y_out = np.empty_like(Y, dtype=np.int32)
        print(f"[{s}] suavizando {S} muestras | P={P} | k={args.k} | iters={args.iters}")
        for i in range(S):
            Y_out[i] = smooth_one_sample(X[i], Y[i], k=args.k, iters=args.iters)
            if (i+1) % 50 == 0 or i == S-1:
                print(f"  - {i+1}/{S}")

        # Guardamos siempre como .npz con claves estándar
        _save_npz(OUT/f"X_{s}.npz", "X", X.astype(np.float32))
        _save_npz(OUT/f"Y_{s}.npz", "Y", Y_out.astype(np.int32))

    print(f"[OK] Split suavizado listo en: {OUT}")

if __name__ == "__main__":
    main()
