#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auditor de integridad del dataset segmentado (train/val/test .npz)

Revisa:
- Clases presentes en cada split (train, val, test)
- Clases ausentes o con frecuencia muy baja
- Proporción de fondo (label 0)
- Distribución de puntos totales y balance general
- Diferencias con el label_map.json
- Genera JSON resumen y gráfico de distribución de clases

Uso:
  python scripts_v2/check_split_quality.py \
      --data_dir /home/htaucare/Tesis_Amaro/data/Teeth_3ds/fixed_split/8192/fps_aug_fixed_v6
"""

import numpy as np, json, argparse, os
from pathlib import Path
import matplotlib.pyplot as plt

def load_npz_safe(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {path}")
    with np.load(path, allow_pickle=False) as z:
        for k in ("Y", "arr_0"):
            if k in z: return z[k]
        raise ValueError(f"No se encontró clave Y en {path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Carpeta con X_*.npz y Y_*.npz")
    ap.add_argument("--out_dir", default=None, help="Carpeta de salida (por defecto = data_dir/audit)")
    ap.add_argument("--save_plot", action="store_true", help="Guarda gráfico de distribución de clases")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir or data_dir/"audit")
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train", "val", "test"]
    all_classes = set()
    stats = {}

    print(f"\n[CHECK] Analizando dataset en {data_dir}")

    # intentar cargar el label_map
    label_map_path = data_dir / "artifacts/label_map.json"
    if label_map_path.exists():
        with open(label_map_path, "r", encoding="utf-8") as f:
            label_map = json.load(f)
        id2idx = {int(k): int(v) for k, v in label_map.get("id2idx", {}).items()}
        idx2id = {int(k): int(v) for k, v in label_map.get("idx2id", {}).items()}
        print(f"[MAP] Cargado label_map.json con {len(idx2id)} clases.")
    else:
        print("[WARN] No se encontró artifacts/label_map.json.")
        id2idx, idx2id = {}, {}

    for split in splits:
        y_path = data_dir / f"Y_{split}.npz"
        if not y_path.exists():
            print(f"[WARN] No se encontró {y_path}")
            continue

        Y = load_npz_safe(y_path)
        vals, counts = np.unique(Y, return_counts=True)
        total_pts = int(Y.size)
        bg_mask = (Y == 0)
        bg_ratio = float(bg_mask.sum() / total_pts) * 100

        stats[split] = {
            "unique_labels": vals.tolist(),
            "num_classes": int(len(vals)),
            "min_label": int(vals.min()),
            "max_label": int(vals.max()),
            "bg_percentage": round(bg_ratio, 3),
            "points_total": total_pts,
            "counts": {int(v): int(c) for v, c in zip(vals, counts)}
        }

        all_classes.update(vals.tolist())

        print(f"\n[{split.upper()}]")
        print(f"  clases: {len(vals)}  min={vals.min()}  max={vals.max()}")
        print(f"  fondo: {bg_ratio:.2f}%  puntos: {total_pts:,}")

    # === Auditoría global ===
    all_classes = sorted(all_classes)
    all_min, all_max = min(all_classes), max(all_classes)
    full_range = list(range(all_min, all_max + 1))
    missing = [c for c in full_range if c not in all_classes]

    print("\n=== RESUMEN GLOBAL ===")
    print(f"  Clases totales presentes: {len(all_classes)} ({all_classes[:10]}...)")
    if missing:
        print(f"  ⚠️  Faltan etiquetas en algún split: {missing}")
    else:
        print("  ✅ Todas las clases consecutivas están presentes en al menos un split.")

    # === Gráfico de distribución global ===
    if args.save_plot:
        all_counts = {}
        for split, st in stats.items():
            for c, n in st["counts"].items():
                all_counts.setdefault(c, 0)
                all_counts[c] += n
        keys = sorted(all_counts.keys())
        values = [all_counts[k] for k in keys]
        plt.figure(figsize=(12,4))
        plt.bar(keys, values, color="#008B8B")
        plt.xlabel("Clase")
        plt.ylabel("Cantidad de puntos (suma total)")
        plt.title("Distribución total de puntos por clase")
        plt.tight_layout()
        plt.savefig(out_dir/"class_distribution.png", dpi=300)
        plt.close()
        print(f"[PLOT] Guardado gráfico en {out_dir/'class_distribution.png'}")

    # === Guardar JSON con resumen completo ===
    summary = {
        "data_dir": str(data_dir),
        "splits": stats,
        "all_classes": all_classes,
        "missing_labels": missing,
        "label_map": {
            "num_classes_map": len(idx2id),
            "idx2id": idx2id
        }
    }

    with open(out_dir/"audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[JSON] Guardado resumen en {out_dir/'audit_summary.json'}")

    print("\n✅ Auditoría completa.")

if __name__ == "__main__":
    main()
