#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualiza y guarda la nube de puntos con pseudolabels del diente 21.

Colores:
 - Gris claro: puntos negativos (fondo)
 - Rojo: puntos positivos (diente 21)

Salida: guarda la figura como PNG dentro de /data/UFRN/viz/
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- Configuración ---
pid = "paciente_1"  # 🔹 Cambia este valor para otro paciente
root = Path("/home/htaucare/Tesis_Amaro/data/UFRN/processed_pseudolabels_targets/8192/upper")
out_dir = Path("/home/htaucare/Tesis_Amaro/data/UFRN/viz")
out_dir.mkdir(parents=True, exist_ok=True)

# --- Cargar datos ---
x_path = root / pid / "X.npy"
y_path = root / pid / "Y.npy"

if not x_path.exists() or not y_path.exists():
    raise FileNotFoundError(f"No se encontraron archivos para {pid}")

x = np.load(x_path)
y = np.load(y_path)

# --- Graficar ---
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(x[y == 0, 0], x[y == 0, 1], x[y == 0, 2],
           s=1, c="lightgray", alpha=0.3, label="Fondo")
ax.scatter(x[y == 1, 0], x[y == 1, 1], x[y == 1, 2],
           s=3, c="red", label="Diente 21")

ax.set_title(f"{pid} — puntos etiquetados = {y.sum()}")
ax.legend(loc="upper right")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.view_init(elev=20, azim=35)

# --- Guardar ---
out_path = out_dir / f"{pid}_pseudolabels.png"
plt.tight_layout()
plt.savefig(out_path, dpi=300)
plt.close(fig)

print(f"✅ Visualización guardada en: {out_path}")
