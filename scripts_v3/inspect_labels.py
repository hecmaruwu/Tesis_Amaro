#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_labels.py
--------------------------------------------------------------
Inspecciona las etiquetas de segmentación (.npz) y muestra:
 - IDs únicos presentes en cada split
 - Número total de puntos por clase
 - Estimación visual para identificar el incisivo central (FDI 21 remapeado)
--------------------------------------------------------------
Autor: ChatGPT (GPT-5)
"""

import numpy as np
from pathlib import Path
from collections import Counter

def inspect_split(name: str, y_path: Path):
    data = np.load(y_path)["Y"]
    flat = data.flatten()
    cnt = Counter(flat)
    total = len(flat)
    print(f"\n🧩 Split: {name}")
    print(f"Total de puntos: {total:,}")
    print("IDs únicos:", sorted(cnt.keys()))
    print("Frecuencias:")
    for k in sorted(cnt.keys()):
        pct = cnt[k] / total * 100
        print(f"  ID {k:02d}: {cnt[k]:>10,} puntos ({pct:5.2f}%)")

def main():
    base = Path("../data/Teeth_3ds/fixed_split/8192/fps_aug_paperlike_v3")
    for split in ["train", "val", "test"]:
        y_path = base / f"Y_{split}.npz"
        if y_path.exists():
            inspect_split(split, y_path)
        else:
            print(f"[WARN] No se encontró {y_path}")

if __name__ == "__main__":
    main()
