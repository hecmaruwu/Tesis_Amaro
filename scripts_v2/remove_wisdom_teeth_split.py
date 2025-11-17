#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crea una versión del split sin terceros molares (18, 28, 38, 48)
a partir de /fixed_split/8192/stratified/
"""

import numpy as np
from pathlib import Path

INPUT = Path("/home/htaucare/Tesis_Amaro/data/Teeth_3ds/fixed_split/8192/stratified")
OUTPUT = Path("/home/htaucare/Tesis_Amaro/data/Teeth_3ds/fixed_split/8192/stratified_sin_molares")
OUTPUT.mkdir(parents=True, exist_ok=True)

# IDs de molares (terceros) según FDI
molares_fdi = {18, 28, 38, 48}

def procesar_split(split):
    X = np.load(INPUT / f"X_{split}.npz")["X"]
    Y = np.load(INPUT / f"Y_{split}.npz")["Y"]

    # elimina puntos que pertenecen a muelas del juicio
    mask = ~np.isin(Y, list(molares_fdi))
    X_new, Y_new = [], []
    for i in range(len(X)):
        Xi, Yi = X[i][mask[i]], Y[i][mask[i]]
        X_new.append(Xi)
        Y_new.append(Yi)

    np.savez_compressed(OUTPUT / f"X_{split}.npz", X=np.array(X_new, dtype=object))
    np.savez_compressed(OUTPUT / f"Y_{split}.npz", Y=np.array(Y_new, dtype=object))
    print(f"[OK] Guardado {split}: {len(X_new)} muestras")

for s in ["train", "val", "test"]:
    procesar_split(s)
print(f"\n✅ Dataset sin molares creado en {OUTPUT}")
