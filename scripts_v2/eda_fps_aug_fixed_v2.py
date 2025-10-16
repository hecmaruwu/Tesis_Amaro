#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EDA de dataset 3D Teeth – fps_aug_fixed_v2

Analiza:
- Distribución de etiquetas (train/val/test)
- Cobertura de clases
- Porcentaje de fondo
- Histogramas de frecuencia
"""

import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json

def load_split(data_dir: Path, split: str):
    X = np.load(data_dir / f"X_{split}.npz")["X"]
    Y = np.load(data_dir / f"Y_{split}.npz")["Y"]
    return X, Y

def compute_stats(Y):
    vals, counts = np.unique(Y, return_counts=True)
    freq = dict(zip(map(int, vals), map(int, counts)))
    total = int(Y.size)
    bg_ratio = freq.get(0, 0) / total * 100
    return {
        "unique": len(vals),
        "min": int(vals.min()),
        "max": int(vals.max()),
        "bg_ratio": round(bg_ratio, 2),
        "freq": freq,
        "total_points": total
    }

def plot_histogram(freq, split, out_dir, num_classes=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    classes = list(freq.keys())
    counts = [freq[k] for k in classes]
    plt.figure(figsize=(10,5))
    plt.bar(classes, counts, color="steelblue")
    plt.title(f"Distribución de clases – {split}")
    plt.xlabel("Etiqueta de clase")
    plt.ylabel("Número de puntos")
    if num_classes is not None:
        plt.xticks(range(num_classes))
    plt.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_dir / f"hist_{split}.png", dpi=300)
    plt.close()

def compare_splits(stats_dict, out_dir):
    all_classes = sorted(set().union(*[set(s["freq"].keys()) for s in stats_dict.values()]))
    fig, ax = plt.subplots(figsize=(12,6))
    width = 0.25
    for i, (split, st) in enumerate(stats_dict.items()):
        freq = st["freq"]
        vals = [freq.get(c, 0) for c in all_classes]
        ax.bar([x + i*width for x in range(len(all_classes))], vals, width, label=split)
    ax.set_xticks([x + width for x in range(len(all_classes))])
    ax.set_xticklabels(all_classes, rotation=90)
    ax.set_title("Comparación de frecuencia de clases entre splits")
    ax.set_xlabel("Etiqueta")
    ax.set_ylabel("Número de puntos")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "comparacion_splits.png", dpi=300)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Carpeta con X_train.npz, Y_train.npz, etc.")
    ap.add_argument("--out_dir", required=True, help="Carpeta para guardar resultados y gráficos")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = {}
    print(f"[INFO] Analizando dataset en: {data_dir}")

    for split in ["train", "val", "test"]:
        X, Y = load_split(data_dir, split)
        s = compute_stats(Y)
        stats[split] = s
        print(f"[{split.upper()}] min={s['min']} max={s['max']} unique={s['unique']} "
              f"bg={s['bg_ratio']}% puntos={s['total_points']:,}")
        plot_histogram(s["freq"], split, out_dir)

    # Comparación visual de splits
    compare_splits(stats, out_dir)

    # Detectar clases faltantes por split
    all_classes = sorted(set().union(*[set(s["freq"].keys()) for s in stats.values()]))
    coverage = {split: sorted(set(s["freq"].keys())) for split, s in stats.items()}
    missing = {split: sorted(set(all_classes) - set(coverage[split])) for split in stats}
    print("\n[CLASES FALTANTES POR SPLIT]:")
    for k, v in missing.items():
        print(f"  {k}: {v}")

    # Guardar resumen JSON
    summary = {
        "stats": stats,
        "all_classes": all_classes,
        "missing_classes": missing
    }
    json.dump(summary, open(out_dir / "eda_summary.json", "w"), indent=2)
    print(f"[DONE] Resultados guardados en {out_dir}")

if __name__ == "__main__":
    main()
