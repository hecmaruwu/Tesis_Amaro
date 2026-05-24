#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT_DIR = Path("/home/htaucare/Tesis_Amaro/case_comparisons/QUALI_CASES_V17/VISTA_TESIS_INFERENCIA_AJUSTADA")
OUT_DIR.mkdir(parents=True, exist_ok=True)

legend_items = [
    ("Background / encía", "#000000"),
    ("Otros dientes", "#F2D1A9"),
    ("Diente 21", "#FF8C00"),
    ("Vecino izquierdo", "#0072B2"),
    ("Vecino derecho", "#E76DB6"),
    ("Error multiclase o FP/FN", "#FF0000"),
    ("Verdadero positivo d21", "#00B050"),
    ("Malla cruda de referencia", "#B0B0B0"),
]

handles = [
    Patch(facecolor=color, edgecolor="black", linewidth=0.4, label=label)
    for label, color in legend_items
]

fig, ax = plt.subplots(figsize=(13, 2.1), dpi=300)
ax.axis("off")

legend = ax.legend(
    handles=handles,
    loc="center",
    ncol=4,
    frameon=True,
    fontsize=11,
    title="Leyenda de visualización cualitativa",
    title_fontsize=13,
    columnspacing=1.8,
    handlelength=1.8,
    handleheight=1.0,
    borderpad=0.9,
)

legend.get_frame().set_edgecolor("black")
legend.get_frame().set_linewidth(0.8)
legend.get_frame().set_facecolor("white")

out_png = OUT_DIR / "leyenda_inferencia_tesis.png"
out_pdf = OUT_DIR / "leyenda_inferencia_tesis.pdf"

fig.savefig(out_png, bbox_inches="tight", pad_inches=0.15)
fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.15)

plt.close(fig)

print(f"[OK] PNG: {out_png}")
print(f"[OK] PDF: {out_pdf}")