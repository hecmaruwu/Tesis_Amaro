#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auditoría completa del entrenamiento y de los splits:
1. Graficar curvas de entrenamiento (loss, acc, f1, iou)
2. Revisar fugas de datos (pacientes repetidos entre splits)
3. Detectar señales de overfitting / leakage
"""

import json, re
from pathlib import Path
import matplotlib.pyplot as plt

# === CONFIGURACIÓN ===
RUN_DIR = Path("/home/htaucare/Tesis_Amaro/results/pointnet_8192_ep400_v3/pointnet_lr0.001_bs8_drop0.5_2025-11-04_12-35-33")
DATA_SPLIT_DIR = Path("/home/htaucare/Tesis_Amaro/data/Teeth_3ds/fixed_split/8192/stratified")

history_path = RUN_DIR / "history.json"
meta_path = DATA_SPLIT_DIR / "meta.json"

# === PARTE 1: CURVAS ===
if history_path.exists():
    with open(history_path, "r", encoding="utf-8") as f:
        hist = json.load(f)

    out_dir = RUN_DIR / "audit_plots"
    out_dir.mkdir(exist_ok=True)

    metrics = ["loss", "acc", "f1", "iou"]
    for m in metrics:
        plt.figure(figsize=(7,4))
        for split in ["train", "val", "test"]:
            key = f"{split}_{m}"
            if key in hist and len(hist[key]) > 0:
                plt.plot(hist[key], label=split)
        plt.xlabel("Época")
        plt.ylabel(m.upper())
        plt.title(f"PointNet — {m.upper()}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"curve_{m}.png", dpi=200)
        plt.close()
    print(f"[OK] Curvas de entrenamiento guardadas en: {out_dir}")
else:
    print(f"[WARN] No se encontró {history_path}")

# === PARTE 2: CHEQUEO DE DUPLICADOS ENTRE SPLITS ===
def extract_pids(meta_dict):
    out = {}
    for k,v in meta_dict.items():
        if isinstance(v, list):
            pids = []
            for c in v:
                if isinstance(c, dict) and "pid" in c:
                    pids.append(c["pid"])
                elif isinstance(c, str):
                    # busca un número dentro del nombre (paciente_XX)
                    m = re.search(r"paciente[_\-]?(\d+)", c.lower())
                    if m: pids.append(f"paciente_{m.group(1)}")
            out[k] = sorted(set(pids))
    return out

if meta_path.exists():
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    splits_pids = extract_pids(meta)

    inter_train_val = set(splits_pids.get("cases_train", [])) & set(splits_pids.get("cases_val", []))
    inter_train_test = set(splits_pids.get("cases_train", [])) & set(splits_pids.get("cases_test", []))
    inter_val_test   = set(splits_pids.get("cases_val", []))   & set(splits_pids.get("cases_test", []))

    print("\n=== CHEQUEO DE LEAKAGE ENTRE SPLITS ===")
    print(f"Total train: {len(splits_pids.get('cases_train', []))}")
    print(f"Total val  : {len(splits_pids.get('cases_val', []))}")
    print(f"Total test : {len(splits_pids.get('cases_test', []))}")
    print(f"\n[train ∩ val]  → {len(inter_train_val)} duplicados")
    print(f"[train ∩ test] → {len(inter_train_test)} duplicados")
    print(f"[val ∩ test]   → {len(inter_val_test)} duplicados")

    if inter_train_val or inter_train_test or inter_val_test:
        print("\n⚠️  POSIBLE LEAKAGE DETECTADO:")
        for s,dup in [("train-val", inter_train_val), ("train-test", inter_train_test), ("val-test", inter_val_test)]:
            if dup:
                print(f"  - {s}: {', '.join(sorted(list(dup)))[:200]}")
    else:
        print("✅ No se detectaron pacientes repetidos entre splits.")
else:
    print(f"[WARN] No se encontró meta.json en {meta_path}")

# === PARTE 3: DIAGNÓSTICO BÁSICO ===
if history_path.exists():
    f1_tr = hist.get("train_f1", [])
    f1_val = hist.get("val_f1", [])
    if f1_tr and f1_val:
        last_gap = abs(f1_tr[-1] - f1_val[-1])
        print("\n=== ANÁLISIS DE CONVERGENCIA ===")
        print(f"Último F1 (train): {f1_tr[-1]:.4f}")
        print(f"Último F1 (val)  : {f1_val[-1]:.4f}")
        print(f"Diferencia final : {last_gap:.4f}")
        if last_gap > 0.15:
            print("⚠️  Posible sobreajuste o datos no bien estratificados.")
        elif last_gap < 0.05:
            print("✅ Buen equilibrio entre train y val (sin sobreajuste aparente).")
        else:
            print("ℹ️  Gap moderado: revisar curvas para confirmar.")
