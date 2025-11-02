#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grid search de hiperparámetros para HDBSCAN post-procesamiento.
Evalúa distintas combinaciones de (thr, min_cluster_size, min_samples)
y guarda métricas antes/después por cada combinación.

Genera resultados bajo:
  /home/htaucare/Tesis_Amaro/data/UFRN/hdbscan_sweep_results_v19g_tuned
"""

import os, json, argparse, itertools
from pathlib import Path
import numpy as np
import subprocess

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_script", required=True,
                    help="Ruta a postprocess_hdbscan_seg_v2.py")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--model_ckpt", required=True)
    ap.add_argument("--out_root", required=True)
    args = ap.parse_args()

    # combinaciones
    thrs = [0.60, 0.65, 0.70]
    min_clusters = [10, 20, 40, 60]
    min_samples = [3, 5, 10]

    combos = list(itertools.product(thrs, min_clusters, min_samples))
    print(f"Total combinaciones: {len(combos)}")

    for i, (thr, mcs, ms) in enumerate(combos, 1):
        tag = f"thr{thr:.2f}_c{mcs}_s{ms}"
        out_dir = Path(args.out_root) / tag
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "python", args.base_script,
            "--data_dir", args.data_dir,
            "--model_ckpt", args.model_ckpt,
            "--thr", str(thr),
            "--min_cluster_size", str(mcs),
            "--min_samples", str(ms),
            "--out_dir", str(out_dir)
        ]
        print(f"\n[{i}/{len(combos)}] Ejecutando {tag} ...")
        subprocess.run(cmd, check=True)

    print("\n✅ Búsqueda terminada. Resultados guardados en:")
    print(args.out_root)

if __name__ == "__main__":
    main()
