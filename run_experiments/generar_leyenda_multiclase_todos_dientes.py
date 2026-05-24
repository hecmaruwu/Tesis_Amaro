#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
genera_leyenda_multiclase_todos_dientes.py

Genera una imagen de leyenda multiclase para todos los dientes
usando EXACTAMENTE la misma lógica de colores que tus scripts
de visualización que hacen:

    cmap = plt.colormaps.get_cmap("tab20")
    cols = [cmap(i / max(C - 1, 1)) for i in range(C)]

Esto permite complementar las figuras multiclase de la tesis
con una leyenda limpia, completa y exportable.

Salidas:
- PNG horizontal
- PDF horizontal
- SVG horizontal
- PNG vertical
- PDF vertical
- CSV con índice, etiqueta y color HEX
"""

from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import to_hex


# ============================================================
# CONFIG
# ============================================================

OUT_DIR = Path(
    "/home/htaucare/Tesis_Amaro/case_comparisons/QUALI_CASES_V17/"
    "VISTA_TESIS_INFERENCIA_AJUSTADA"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

# En tu proyecto final upper-only sin muelas del juicio:
# 0 = background
# 1..14 = dientes
NUM_CLASSES = 15

# Etiquetas internas -> nombre dental
CLASS_LABELS = {
    0: "background",
    1: "d11",
    2: "d12",
    3: "d13",
    4: "d14",
    5: "d15",
    6: "d16",
    7: "d17",
    8: "d21",
    9: "d22",
    10: "d23",
    11: "d24",
    12: "d25",
    13: "d26",
    14: "d27",
}


# ============================================================
# MISMA LÓGICA DE COLORES QUE TU CÓDIGO
# ============================================================

def _class_colors(C: int):
    cmap = plt.colormaps.get_cmap("tab20")
    C = max(int(C), 2)
    cols = [cmap(i / max(C - 1, 1)) for i in range(C)]
    return cols


# ============================================================
# HELPERS
# ============================================================

def build_handles_and_rows():
    cols = _class_colors(NUM_CLASSES)

    handles = []
    rows = []

    for idx in range(NUM_CLASSES):
        rgba = cols[idx]
        hex_color = to_hex(rgba, keep_alpha=False)

        label_name = CLASS_LABELS.get(idx, f"class_{idx}")
        display_label = f"{idx} - {label_name}"

        handles.append(
            Patch(
                facecolor=rgba,
                edgecolor="black",
                linewidth=0.6,
                label=display_label
            )
        )

        rows.append({
            "class_idx": idx,
            "label": label_name,
            "display_label": display_label,
            "hex_color": hex_color
        })

    return handles, rows


def save_csv(rows, out_csv: Path):
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["class_idx", "label", "display_label", "hex_color"]
        )
        writer.writeheader()
        writer.writerows(rows)


def make_horizontal_legend(handles, out_base: Path):
    fig, ax = plt.subplots(figsize=(16, 3.6), dpi=300)
    ax.axis("off")

    legend = ax.legend(
        handles=handles,
        loc="center",
        ncol=5,
        frameon=True,
        fontsize=11,
        title="Leyenda multiclase — todas las clases dentales",
        title_fontsize=13,
        columnspacing=1.5,
        handlelength=1.6,
        handleheight=1.1,
        borderpad=0.9,
        labelspacing=1.0,
    )

    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_linewidth(0.8)

    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", pad_inches=0.15)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.15)
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def make_vertical_legend(handles, out_base: Path):
    fig, ax = plt.subplots(figsize=(6.5, 8.5), dpi=300)
    ax.axis("off")

    legend = ax.legend(
        handles=handles,
        loc="center",
        ncol=1,
        frameon=True,
        fontsize=11,
        title="Leyenda multiclase — todas las clases dentales",
        title_fontsize=13,
        columnspacing=1.2,
        handlelength=1.6,
        handleheight=1.1,
        borderpad=0.9,
        labelspacing=0.9,
    )

    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_linewidth(0.8)

    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", pad_inches=0.15)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def main():
    handles, rows = build_handles_and_rows()

    out_csv = OUT_DIR / "leyenda_multiclase_todos_dientes_colores.csv"
    save_csv(rows, out_csv)

    out_horizontal = OUT_DIR / "leyenda_multiclase_todos_dientes_horizontal"
    out_vertical = OUT_DIR / "leyenda_multiclase_todos_dientes_vertical"

    make_horizontal_legend(handles, out_horizontal)
    make_vertical_legend(handles, out_vertical)

    print("=" * 80)
    print("[OK] Leyendas multiclase generadas.")
    print(f"[OK] CSV colores: {out_csv}")
    print(f"[OK] PNG horizontal: {out_horizontal.with_suffix('.png')}")
    print(f"[OK] PDF horizontal: {out_horizontal.with_suffix('.pdf')}")
    print(f"[OK] SVG horizontal: {out_horizontal.with_suffix('.svg')}")
    print(f"[OK] PNG vertical: {out_vertical.with_suffix('.png')}")
    print(f"[OK] PDF vertical: {out_vertical.with_suffix('.pdf')}")
    print("=" * 80)


if __name__ == "__main__":
    main()