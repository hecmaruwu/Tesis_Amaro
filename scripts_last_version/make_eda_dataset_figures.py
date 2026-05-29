#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_eda_dataset_figures.py

Script para generar figuras y tablas del Análisis Exploratorio de Datos (EDA)
del dataset dental 3D procesado SIN aumentación.

Este script está pensado para el dataset upper-only de la tesis, donde las
clases están remapeadas internamente:

    0  = fondo / encía
    1  = diente 11
    2  = diente 12
    3  = diente 13
    4  = diente 14
    5  = diente 15
    6  = diente 16
    7  = diente 17
    8  = diente 21
    9  = diente 22
    10 = diente 23
    11 = diente 24
    12 = diente 25
    13 = diente 26
    14 = diente 27

Salidas principales:
    - eda_summary.json
    - eda_per_sample.csv
    - eda_presence_by_tooth.csv
    - eda_teeth_count_grouped.csv
    - eda_symmetric_pairs.csv
    - fig_presence_by_tooth_pct.png/pdf
    - fig_teeth_count_grouped_pct.png/pdf
    - fig_symmetric_pairs_pct.png/pdf
    - table_degenerate_samples_eda.tex

El script también imprime en consola un resumen útil para redactar el EDA.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Mapeo interno -> notación FDI
# ============================================================

INTERNAL_TO_FDI = {
    0: 0,
    1: 11,
    2: 12,
    3: 13,
    4: 14,
    5: 15,
    6: 16,
    7: 17,
    8: 21,
    9: 22,
    10: 23,
    11: 24,
    12: 25,
    13: 26,
    14: 27,
}

FDI_ORDER = [11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27]

SYMMETRIC_PAIRS = [
    (11, 21),
    (12, 22),
    (13, 23),
    (14, 24),
    (15, 25),
    (16, 26),
    (17, 27),
]

TEETH_GROUP_ORDER = [
    "0 dientes",
    "1-8 dientes",
    "9-10 dientes",
    "11-12 dientes",
    "13-14 dientes",
]


# ============================================================
# Utilidades de guardado
# ============================================================

def save_json(obj, path: Path):
    """Guarda un objeto Python como JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_csv(rows, path: Path):
    """Guarda una lista de diccionarios como CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Agrupación de cantidad de dientes
# ============================================================

def tooth_bin(n_teeth: int) -> str:
    """
    Agrupa la cantidad de dientes presentes en rangos más interpretables.

    0 dientes corresponde a muestras only-background.
    """
    if n_teeth == 0:
        return "0 dientes"
    if 1 <= n_teeth <= 8:
        return "1-8 dientes"
    if 9 <= n_teeth <= 10:
        return "9-10 dientes"
    if 11 <= n_teeth <= 12:
        return "11-12 dientes"
    if 13 <= n_teeth <= 14:
        return "13-14 dientes"
    return "Otro"


# ============================================================
# Estilo de figuras
# ============================================================

def set_plot_style():
    """
    Define un estilo sobrio para figuras de tesis.
    """
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 350,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
    })


def add_bar_labels(ax, values, suffix="%", fontsize=9):
    """
    Agrega etiquetas sobre cada barra.
    """
    for patch, value in zip(ax.patches, values):
        height = patch.get_height()
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            height + max(values) * 0.015,
            f"{value:.1f}{suffix}",
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def nice_ylim(values, min_top=10, max_top=100, padding=1.18):
    """
    Calcula un límite superior del eje Y evitando demasiado espacio vacío.
    """
    if not values:
        return min_top

    top = max(values) * padding
    top = max(top, min_top)
    top = min(top, max_top)
    return top


def save_figure(fig, out_dir: Path, stem: str):
    """
    Guarda cada figura en PNG y PDF.
    """
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Figuras
# ============================================================

def plot_presence_by_tooth(presence_rows, out_dir: Path):
    """
    Figura 1:
    Presencia porcentual de cada pieza dental superior en el dataset.
    """
    xs = [str(r["fdi"]) for r in presence_rows]
    ys = [r["pct_samples_present"] for r in presence_rows]

    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    bars = ax.bar(xs, ys, color="#2f6f9f", edgecolor="black", linewidth=0.4)

    ax.set_title("Presencia por pieza dental en arcadas superiores")
    ax.set_xlabel("Pieza dental (FDI)")
    ax.set_ylabel("Muestras con pieza presente (%)")
    ax.set_ylim(0, nice_ylim(ys, min_top=75, max_top=100, padding=1.08))
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)

    add_bar_labels(ax, ys, suffix="%", fontsize=8)

    fig.tight_layout()
    save_figure(fig, out_dir, "fig_presence_by_tooth_pct")


def plot_teeth_count_grouped(grouped_rows, out_dir: Path):
    """
    Figura 2:
    Distribución agrupada de cantidad de dientes presentes por muestra.
    """
    xs = [r["teeth_group"] for r in grouped_rows]
    ys = [r["pct_samples"] for r in grouped_rows]

    colors = []
    for x in xs:
        if x == "0 dientes":
            colors.append("#8c8c8c")  # gris para casos only-background
        else:
            colors.append("#2f6f9f")

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.bar(xs, ys, color=colors, edgecolor="black", linewidth=0.4)

    ax.set_title("Distribución agrupada de piezas dentales por muestra")
    ax.set_xlabel("Cantidad de piezas dentales presentes")
    ax.set_ylabel("Muestras (%)")
    ax.set_ylim(0, nice_ylim(ys, min_top=20, max_top=80, padding=1.18))
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)

    add_bar_labels(ax, ys, suffix="%", fontsize=9)

    plt.xticks(rotation=18, ha="right")
    fig.tight_layout()
    save_figure(fig, out_dir, "fig_teeth_count_grouped_pct")


def plot_symmetric_pairs(pair_rows, out_dir: Path):
    """
    Figura 3:
    Presencia porcentual de pares dentales simétricos.
    """
    xs = [r["pair"] for r in pair_rows]
    ys = [r["pct_samples_present"] for r in pair_rows]

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.bar(xs, ys, color="#2f6f9f", edgecolor="black", linewidth=0.4)

    ax.set_title("Presencia de pares dentales simétricos")
    ax.set_xlabel("Par dental simétrico")
    ax.set_ylabel("Muestras con ambos dientes presentes (%)")
    ax.set_ylim(0, nice_ylim(ys, min_top=70, max_top=100, padding=1.08))
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)

    add_bar_labels(ax, ys, suffix="%", fontsize=9)

    fig.tight_layout()
    save_figure(fig, out_dir, "fig_symmetric_pairs_pct")


# ============================================================
# Tabla LaTeX
# ============================================================

def write_latex_table(split_summary, total_samples, total_only_bg, total_has_d21, out_dir: Path):
    """
    Genera una tabla LaTeX con:
    - muestras por split
    - muestras only-background
    - muestras con diente 21
    """
    pct_only_bg_total = 100 * total_only_bg / total_samples
    pct_d21_total = 100 * total_has_d21 / total_samples

    latex = []
    latex.append(r"\begin{table}[H]")
    latex.append(r"\centering")
    latex.append(r"\caption{Auditoría de muestras degeneradas y presencia del diente 21 en el dataset procesado sin aumentación.}")
    latex.append(r"\label{tab:degenerate_samples_eda}")
    latex.append(r"\begin{tabular}{lrrrrr}")
    latex.append(r"\toprule")
    latex.append(
        r"\textbf{Split} & \textbf{Muestras} & \textbf{Only-background} & "
        r"\textbf{\% Only-bg} & \textbf{Con diente 21} & \textbf{\% diente 21} \\"
    )
    latex.append(r"\midrule")

    names = {
        "train": "Entrenamiento",
        "val": "Validación",
        "test": "Prueba",
    }

    for split in ["train", "val", "test"]:
        if split in split_summary:
            s = split_summary[split]
            latex.append(
                f"{names[split]} & {s['n_samples']} & {s['n_only_bg']} & "
                f"{s['pct_only_bg']:.2f} & {s['n_has_d21']} & {s['pct_has_d21']:.2f} \\\\"
            )

    latex.append(r"\midrule")
    latex.append(
        f"Total & {total_samples} & {total_only_bg} & "
        f"{pct_only_bg_total:.2f} & {total_has_d21} & {pct_d21_total:.2f} \\\\"
    )
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    (out_dir / "table_degenerate_samples_eda.tex").write_text(
        "\n".join(latex),
        encoding="utf-8",
    )


# ============================================================
# Programa principal
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Carpeta con X_train/Y_train, X_val/Y_val y X_test/Y_test.")
    parser.add_argument("--out_dir", required=True, help="Carpeta donde se guardarán figuras, tablas y reportes.")
    parser.add_argument("--bg_class", type=int, default=0, help="Clase interna correspondiente al fondo/encía.")
    parser.add_argument("--d21_internal", type=int, default=8, help="Clase interna correspondiente al diente 21.")
    args = parser.parse_args()

    set_plot_style()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Contadores globales.
    per_sample_rows = []
    split_summary = {}

    presence_counter = {fdi: 0 for fdi in FDI_ORDER}
    pair_counter = {f"{a}-{b}": 0 for a, b in SYMMETRIC_PAIRS}
    grouped_teeth_counter = {group: 0 for group in TEETH_GROUP_ORDER}

    total_samples = 0
    total_has_bg = 0
    total_no_bg = 0
    total_only_bg = 0
    total_has_d21 = 0

    # ========================================================
    # Recorrido de los splits
    # ========================================================

    for split in ["train", "val", "test"]:
        yp = data_dir / f"Y_{split}.npz"

        if not yp.exists():
            print(f"[WARN] No existe {yp}")
            continue

        Y = np.load(yp)["Y"]
        n_samples, n_points = Y.shape

        split_has_bg = 0
        split_no_bg = 0
        split_only_bg = 0
        split_has_d21 = 0
        split_teeth_counts = []

        for row_i, y in enumerate(Y):
            internal_present = set(int(v) for v in np.unique(y))

            has_bg = args.bg_class in internal_present
            only_bg = internal_present == {args.bg_class}
            has_d21 = args.d21_internal in internal_present

            # Convertir clases internas a FDI, excluyendo fondo.
            fdi_present = sorted(
                INTERNAL_TO_FDI[c]
                for c in internal_present
                if c in INTERNAL_TO_FDI and c != args.bg_class
            )

            n_teeth = len(fdi_present)
            group = tooth_bin(n_teeth)

            # Actualizar contadores globales.
            total_samples += 1
            total_has_bg += int(has_bg)
            total_no_bg += int(not has_bg)
            total_only_bg += int(only_bg)
            total_has_d21 += int(has_d21)

            # Actualizar contadores por split.
            split_has_bg += int(has_bg)
            split_no_bg += int(not has_bg)
            split_only_bg += int(only_bg)
            split_has_d21 += int(has_d21)
            split_teeth_counts.append(n_teeth)

            # Actualizar histograma agrupado.
            grouped_teeth_counter[group] += 1

            # Actualizar presencia por diente.
            for fdi in fdi_present:
                if fdi in presence_counter:
                    presence_counter[fdi] += 1

            # Actualizar presencia de pares simétricos.
            fdi_set = set(fdi_present)
            for a, b in SYMMETRIC_PAIRS:
                if a in fdi_set and b in fdi_set:
                    pair_counter[f"{a}-{b}"] += 1

            # Registro por muestra para auditoría fina.
            per_sample_rows.append({
                "split": split,
                "row_i": int(row_i),
                "n_points": int(n_points),
                "has_bg": int(has_bg),
                "only_bg": int(only_bg),
                "has_d21_internal_8": int(has_d21),
                "n_teeth_present": int(n_teeth),
                "teeth_group": group,
                "fdi_present": json.dumps(fdi_present),
            })

        split_summary[split] = {
            "n_samples": int(n_samples),
            "n_points_per_sample": int(n_points),
            "n_has_bg": int(split_has_bg),
            "n_no_bg": int(split_no_bg),
            "pct_has_bg": round(100 * split_has_bg / n_samples, 2),
            "n_only_bg": int(split_only_bg),
            "pct_only_bg": round(100 * split_only_bg / n_samples, 2),
            "n_has_d21": int(split_has_d21),
            "pct_has_d21": round(100 * split_has_d21 / n_samples, 2),
            "n_no_d21": int(n_samples - split_has_d21),
            "pct_no_d21": round(100 * (n_samples - split_has_d21) / n_samples, 2),
            "mean_teeth_present": round(float(np.mean(split_teeth_counts)), 3),
            "median_teeth_present": round(float(np.median(split_teeth_counts)), 3),
            "min_teeth_present": int(np.min(split_teeth_counts)),
            "max_teeth_present": int(np.max(split_teeth_counts)),
        }

    # ========================================================
    # Tablas resumen
    # ========================================================

    presence_rows = []
    for fdi in FDI_ORDER:
        n = presence_counter[fdi]
        presence_rows.append({
            "fdi": int(fdi),
            "n_samples_present": int(n),
            "pct_samples_present": round(100 * n / total_samples, 2),
        })

    grouped_rows = []
    for group in TEETH_GROUP_ORDER:
        n = grouped_teeth_counter[group]
        grouped_rows.append({
            "teeth_group": group,
            "n_samples": int(n),
            "pct_samples": round(100 * n / total_samples, 2),
        })

    pair_rows = []
    for a, b in SYMMETRIC_PAIRS:
        pair = f"{a}-{b}"
        n = pair_counter[pair]
        pair_rows.append({
            "pair": pair,
            "n_samples_present": int(n),
            "pct_samples_present": round(100 * n / total_samples, 2),
        })

    global_summary = {
        "data_dir": str(data_dir),
        "out_dir": str(out_dir),
        "total_samples": int(total_samples),
        "internal_to_fdi": INTERNAL_TO_FDI,
        "bg_class": int(args.bg_class),
        "d21_internal": int(args.d21_internal),
        "d21_fdi": 21,
        "n_has_bg": int(total_has_bg),
        "pct_has_bg": round(100 * total_has_bg / total_samples, 2),
        "n_no_bg": int(total_no_bg),
        "pct_no_bg": round(100 * total_no_bg / total_samples, 2),
        "n_only_bg": int(total_only_bg),
        "pct_only_bg": round(100 * total_only_bg / total_samples, 2),
        "n_has_d21": int(total_has_d21),
        "pct_has_d21": round(100 * total_has_d21 / total_samples, 2),
        "n_no_d21": int(total_samples - total_has_d21),
        "pct_no_d21": round(100 * (total_samples - total_has_d21) / total_samples, 2),
        "split_summary": split_summary,
        "presence_by_tooth": presence_rows,
        "teeth_count_grouped": grouped_rows,
        "symmetric_pairs": pair_rows,
    }

    # Guardar reportes.
    save_json(global_summary, out_dir / "eda_summary.json")
    write_csv(per_sample_rows, out_dir / "eda_per_sample.csv")
    write_csv(presence_rows, out_dir / "eda_presence_by_tooth.csv")
    write_csv(grouped_rows, out_dir / "eda_teeth_count_grouped.csv")
    write_csv(pair_rows, out_dir / "eda_symmetric_pairs.csv")

    # Guardar figuras.
    plot_presence_by_tooth(presence_rows, out_dir)
    plot_teeth_count_grouped(grouped_rows, out_dir)
    plot_symmetric_pairs(pair_rows, out_dir)

    # Guardar tabla LaTeX.
    write_latex_table(
        split_summary=split_summary,
        total_samples=total_samples,
        total_only_bg=total_only_bg,
        total_has_d21=total_has_d21,
        out_dir=out_dir,
    )

    # ========================================================
    # Resumen en consola
    # ========================================================

    print("\n================ EDA SUMMARY ================\n")
    print(f"Data dir: {data_dir}")
    print(f"Out dir : {out_dir}")
    print(f"Total muestras: {total_samples}")
    print(f"Todas las muestras tienen fondo: {'SI' if total_no_bg == 0 else 'NO'}")
    print(f"Con fondo: {total_has_bg}/{total_samples} ({global_summary['pct_has_bg']}%)")
    print(f"Only-background: {total_only_bg}/{total_samples} ({global_summary['pct_only_bg']}%)")
    print(f"Con diente 21: {total_has_d21}/{total_samples} ({global_summary['pct_has_d21']}%)")
    print(f"Sin diente 21: {total_samples - total_has_d21}/{total_samples} ({global_summary['pct_no_d21']}%)\n")

    for split, s in split_summary.items():
        print(f"[{split}]")
        print(f"  muestras        : {s['n_samples']}")
        print(f"  con fondo       : {s['n_has_bg']} ({s['pct_has_bg']}%)")
        print(f"  only-background : {s['n_only_bg']} ({s['pct_only_bg']}%)")
        print(f"  con diente 21   : {s['n_has_d21']} ({s['pct_has_d21']}%)")
        print(f"  sin diente 21   : {s['n_no_d21']} ({s['pct_no_d21']}%)")
        print(f"  dientes min/max : {s['min_teeth_present']} / {s['max_teeth_present']}")
        print()

    print("[OK] Guardado:")
    for fname in [
        "eda_summary.json",
        "eda_per_sample.csv",
        "eda_presence_by_tooth.csv",
        "eda_teeth_count_grouped.csv",
        "eda_symmetric_pairs.csv",
        "fig_presence_by_tooth_pct.png",
        "fig_presence_by_tooth_pct.pdf",
        "fig_teeth_count_grouped_pct.png",
        "fig_teeth_count_grouped_pct.pdf",
        "fig_symmetric_pairs_pct.png",
        "fig_symmetric_pairs_pct.pdf",
        "table_degenerate_samples_eda.tex",
    ]:
        print(f"  {out_dir / fname}")


if __name__ == "__main__":
    main()