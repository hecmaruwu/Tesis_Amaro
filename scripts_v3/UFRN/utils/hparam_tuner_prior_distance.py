#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hparam_tuner_prior_distance.py (versión corregida)
--------------------------------------------------
Filtra resultados nulos y guarda gráficos PNG solo con combinaciones válidas.
"""

import os, json, subprocess, itertools
from pathlib import Path
import matplotlib.pyplot as plt

# =====================================================
# CONFIGURACIÓN BASE
# =====================================================
BASE_SCRIPT = "/home/htaucare/Tesis_Amaro/scripts_v3/UFRN/models/train_pointnet2_binary_seg_v19g_prior_distance.py"
DATA_DIR = "/home/htaucare/Tesis_Amaro/data/UFRN/processed_pseudolabels_icp/8192/upper"
OUT_ROOT = "/home/htaucare/Tesis_Amaro/data/UFRN/tuning_prior_v19g"
os.makedirs(OUT_ROOT, exist_ok=True)

# =====================================================
# ESPACIO DE BÚSQUEDA
# =====================================================
lrs = [5e-5, 8e-5, 1e-4]
dropouts = [0.5, 0.6]
pos_weights = [10, 15, 20]
batch_size = 6
epochs = 200
seed = 77

# =====================================================
# FUNCIÓN DE EJECUCIÓN
# =====================================================
def run_experiment(lr, dropout, pos_weight):
    tag = f"lr{lr}_do{dropout}_pw{pos_weight}"
    out_dir = Path(OUT_ROOT) / tag
    cmd = [
        "python", BASE_SCRIPT,
        "--data_dir", DATA_DIR,
        "--out_root", str(OUT_ROOT),
        "--epochs", str(epochs),
        "--bs", str(batch_size),
        "--lr", str(lr),
        "--pos_weight", str(pos_weight),
        "--dropout", str(dropout),
        "--seed", str(seed)
    ]

    print(f"\n🚀 Ejecutando experimento: {tag}")
    result = subprocess.run(cmd, text=True, capture_output=True)
    log_path = out_dir / "train_log.txt"
    os.makedirs(out_dir, exist_ok=True)
    with open(log_path, "w") as f:
        f.write(result.stdout + "\n" + result.stderr)

    # Extraer métricas del log final
    acc, f1, rec, iou = None, None, None, None
    for line in result.stdout.splitlines():
        if "✅ TEST" in line:
            try:
                parts = line.split()
                acc = float(parts[2].split("=")[1])
                f1 = float(parts[3].split("=")[1])
                rec = float(parts[4].split("=")[1])
                iou = float(parts[5].split("=")[1])
            except Exception as e:
                print(f"[WARN] Error parseando métricas para {tag}: {e}")
    return dict(lr=lr, dropout=dropout, pos_weight=pos_weight,
                acc=acc, f1=f1, rec=rec, iou=iou)

# =====================================================
# LOOP DE ENTRENAMIENTOS
# =====================================================
results = []
for lr, dropout, pw in itertools.product(lrs, dropouts, pos_weights):
    res = run_experiment(lr, dropout, pw)
    results.append(res)
    with open(Path(OUT_ROOT)/"results_partial.json", "w") as f:
        json.dump(results, f, indent=2)

# =====================================================
# GUARDAR RESULTADOS FINALES
# =====================================================
out_json = Path(OUT_ROOT) / "results_final.json"
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)

# =====================================================
# VISUALIZACIÓN SEGURA
# =====================================================
valid_results = [r for r in results if r["f1"] is not None and r["iou"] is not None]
if len(valid_results) == 0:
    print("⚠️ Ningún resultado válido encontrado. Revisa logs en tuning_prior_v19g.")
    exit()

f1_vals = [r["f1"] for r in valid_results]
iou_vals = [r["iou"] for r in valid_results]
labels = [f'lr={r["lr"]},do={r["dropout"]},pw={r["pos_weight"]}' for r in valid_results]

plt.figure(figsize=(10, 6))
plt.barh(labels, f1_vals, color='royalblue')
plt.xlabel("F1-score")
plt.title("Comparación de combinaciones de hiperparámetros (F1-score)")
plt.tight_layout()
plt.savefig(Path(OUT_ROOT)/"f1_comparison.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
plt.barh(labels, iou_vals, color='seagreen')
plt.xlabel("IoU")
plt.title("Comparación de combinaciones de hiperparámetros (IoU)")
plt.tight_layout()
plt.savefig(Path(OUT_ROOT)/"iou_comparison.png", dpi=300)
plt.close()

# =====================================================
# MOSTRAR MEJOR RESULTADO
# =====================================================
best_f1 = max(valid_results, key=lambda x: x["f1"])
best_iou = max(valid_results, key=lambda x: x["iou"])

print("\n🏁 Resultados finales válidos:")
print(json.dumps(valid_results, indent=2))
print("\n⭐ Mejor F1:", best_f1)
print("⭐ Mejor IoU:", best_iou)
print(f"\n✅ Gráficos guardados en: {OUT_ROOT}")
