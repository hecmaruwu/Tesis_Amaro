#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_labels_balance.py
---------------------------------
Revisa si las etiquetas (Y.npy) del dataset binario están balanceadas.
Detecta cuántos pacientes tienen solo ceros o solo unos.

Uso:
  python check_labels_balance.py --root /ruta/a/tu/dataset
Ejemplo:
  python check_labels_balance.py --root /home/htaucare/Tesis_Amaro/data/UFRN/processed_pseudolabels_icp/8192/upper
"""

import os
import numpy as np
import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Ruta raíz del dataset con subcarpetas paciente_*")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"No existe la ruta: {root}")

    pos = neg = mixed = 0
    total_points = []

    print(f"\n🔎 Analizando etiquetas en: {root}\n")
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not d.name.startswith("paciente_"):
            continue
        y_path = d / "Y.npy"
        if not y_path.exists():
            print(f"[WARN] {d.name} no contiene Y.npy")
            continue

        Y = np.load(y_path)
        Y = np.clip(Y, 0, 1)
        total_points.append(len(Y))

        if Y.sum() == 0:
            neg += 1
        elif Y.sum() == len(Y):
            pos += 1
        else:
            mixed += 1

    total = pos + neg + mixed
    if total == 0:
        print("⚠️ No se encontraron pacientes válidos.")
        return

    print("===== RESUMEN =====")
    print(f"Pacientes totales:       {total}")
    print(f"Solo Y=1 (positivos):    {pos}  ({pos/total*100:.1f}%)")
    print(f"Solo Y=0 (negativos):    {neg}  ({neg/total*100:.1f}%)")
    print(f"Mixtos (0 y 1 mezclados): {mixed}  ({mixed/total*100:.1f}%)")
    print(f"Número promedio de puntos: {np.mean(total_points):.0f}\n")

    if pos == total:
        print("⚠️ Todas las etiquetas son 1 → dataset trivial positivo.")
    elif neg == total:
        print("⚠️ Todas las etiquetas son 0 → dataset trivial negativo.")
    elif mixed == 0:
        print("⚠️ No hay pacientes con etiquetas mixtas (solo 0 o 1). Revisa tus pseudolabels.")
    else:
        print("✅ Dataset con clases mixtas detectado (bien balanceado o parcialmente balanceado).")

if __name__ == "__main__":
    main()
