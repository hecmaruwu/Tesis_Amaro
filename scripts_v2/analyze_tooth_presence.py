#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analiza la presencia de dientes por clase en los splits del dataset.
Calcula:
 - Cuántos pacientes tienen cada diente.
 - Cuántos tienen específicamente el diente 21.
 - Cuántos tienen al menos una muela del juicio.
 - Cuántos NO tienen ninguna muela del juicio.
"""

import numpy as np
from pathlib import Path

# Ruta al split (ajusta si estás usando otro)
split_dir = Path("/home/htaucare/Tesis_Amaro/data/Teeth_3ds/fixed_split/8192/stratified")

# Cargar etiquetas (usa train, val o test según necesites)
Y = np.load(split_dir / "Y_train.npz")["Y"]  # (N, 8192)
print(f"[INFO] Nubes cargadas: {Y.shape[0]} | Puntos por nube: {Y.shape[1]}")

# IDs de dientes (excluye encía 0)
tooth_ids = list(range(1, 33))
wisdom_ids = [8, 16, 24, 32]   # 18, 28, 38, 48 (índices relativos)
d21_id = 9                     # 21 corresponde a id=9 si 1→11

presence_per_class = {}
for tid in tooth_ids:
    count = sum(np.any(Y[i] == tid) for i in range(Y.shape[0]))
    presence_per_class[tid] = count

# Muela del juicio presencia por paciente
has_wisdom = [any(np.any(Y[i] == wid) for wid in wisdom_ids) for i in range(Y.shape[0])]
no_wisdom = [not hw for hw in has_wisdom]

total_patients = Y.shape[0]
patients_with_21 = presence_per_class.get(d21_id, 0)
patients_with_wisdom = sum(has_wisdom)
patients_without_wisdom = sum(no_wisdom)

print("\n=== RESUMEN ===")
print(f"Total de pacientes        : {total_patients}")
print(f"Con diente 21             : {patients_with_21}")
print(f"Con alguna muela del juicio (18/28/38/48): {patients_with_wisdom}")
print(f"Sin muelas del juicio     : {patients_without_wisdom}")

print("\n=== Presencia por clase (conteo de pacientes que tienen cada diente) ===")
for tid, cnt in presence_per_class.items():
    print(f"Diente {10 + tid:02d}: {cnt}/{total_patients}")
