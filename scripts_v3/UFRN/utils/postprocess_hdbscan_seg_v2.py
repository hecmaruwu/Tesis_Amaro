#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-procesamiento con HDBSCAN (v2 corregido)
---------------------------------------------
1. Carga modelo PointNet++ segmentador binario.
2. Ejecuta inferencia para cada paciente.
3. Guarda:
   - Máscara original (raw)
   - Máscara filtrada con HDBSCAN (clean)
4. Calcula métricas antes y después.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import hdbscan
from sklearn.metrics import f1_score, recall_score, accuracy_score, jaccard_score

# ============================================================
# 🔧 Clases y utilidades necesarias
# ============================================================
class UFRNBinaryDataset:
    def __init__(self, root):
        self.root = Path(root)
        self.ids = sorted([d.name for d in self.root.iterdir() if d.is_dir() and d.name.startswith("paciente_")])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]
        X = np.load(self.root / pid / "X.npy").astype(np.float32)
        Y = np.load(self.root / pid / "Y.npy").astype(np.int64).reshape(-1)
        Y = np.clip(Y, 0, 1)
        return X, Y, pid


class PointNet2BinarySeg(torch.nn.Module):
    """Versión resumida del modelo para inferencia"""
    def __init__(self, k=2):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(3, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, k)
        )

    def forward(self, x):
        # Espera entrada (B,N,3)
        return self.fc(x)  # (B,N,2)


def metrics_global(pred, y_true):
    """Calcula métricas binarias globales"""
    acc = accuracy_score(y_true, pred)
    f1 = f1_score(y_true, pred, zero_division=0)
    rec = recall_score(y_true, pred, zero_division=0)
    iou = jaccard_score(y_true, pred, zero_division=0)
    return dict(acc=acc, f1=f1, recall=rec, iou=iou)

# ============================================================
# 🧩 Función principal
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Ruta al dataset de pacientes (X.npy, Y.npy)")
    ap.add_argument("--model_ckpt", required=True, help="Ruta al modelo entrenado (.pt)")
    ap.add_argument("--thr", type=float, default=0.65, help="Umbral para convertir probas en binario")
    ap.add_argument("--min_cluster_size", type=int, default=30, help="Tamaño mínimo de clúster en HDBSCAN")
    ap.add_argument("--min_samples", type=int, default=5, help="Número mínimo de muestras por clúster")
    ap.add_argument("--out_dir", required=True, help="Carpeta de salida")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks_raw").mkdir(parents=True, exist_ok=True)

    # ------------------------------
    # Cargar modelo y dataset
    # ------------------------------
    print(f"\n🚀 Cargando modelo desde {args.model_ckpt}")
    model = PointNet2BinarySeg().to(device)
    model.load_state_dict(torch.load(args.model_ckpt, map_location=device))
    model.eval()

    ds = UFRNBinaryDataset(args.data_dir)
    print(f"📦 Pacientes detectados: {len(ds)}\n")

    all_metrics_raw, all_metrics_clean = [], []

    # ------------------------------
    # Procesar cada paciente
    # ------------------------------
    for X, Y, pid in tqdm(ds, desc="Post HDBSCAN"):
        try:
            X_t = torch.tensor(X[None, ...], dtype=torch.float32, device=device)
            with torch.no_grad():
                logits = model(X_t).cpu().squeeze(0)  # (N,2)

            p = F.softmax(logits, dim=-1)[:, 1].numpy()
            mask_raw = (p >= args.thr).astype(np.uint8)

            # ------------------------------
            # Guardar máscara cruda
            # ------------------------------
            np.save(out_dir / "masks_raw" / f"{pid}_raw.npy", mask_raw)

            # ------------------------------
            # Aplicar HDBSCAN sobre puntos positivos
            # ------------------------------
            pts_pos = X[mask_raw == 1]
            if len(pts_pos) > 0:
                clusterer = hdbscan.HDBSCAN(min_cluster_size=args.min_cluster_size,
                                            min_samples=args.min_samples)
                labels = clusterer.fit_predict(pts_pos)
                mask_clean = np.zeros_like(mask_raw)
                if len(labels) > 0:
                    # Detectar clúster más grande y marcarlo
                    main_cluster = np.argmax(np.bincount(labels[labels >= 0]))
                    keep_idx = np.where(mask_raw == 1)[0][labels == main_cluster]
                    mask_clean[keep_idx] = 1
            else:
                mask_clean = mask_raw

            np.save(out_dir / "masks" / f"{pid}_clean.npy", mask_clean)

            # ------------------------------
            # Métricas
            # ------------------------------
            m_raw = metrics_global(mask_raw, Y)
            m_clean = metrics_global(mask_clean, Y)
            all_metrics_raw.append(list(m_raw.values()))
            all_metrics_clean.append(list(m_clean.values()))

        except Exception as e:
            print(f"⚠️ Error en {pid}: {e}")

    # ------------------------------
    # Promedio global de métricas
    # ------------------------------
    all_metrics_raw = np.array(all_metrics_raw)
    all_metrics_clean = np.array(all_metrics_clean)
    mean_raw = all_metrics_raw.mean(0)
    mean_clean = all_metrics_clean.mean(0)

    print("\n====== MÉTRICAS GLOBALES ======")
    print(f"ANTES  (thr={args.thr:.2f})  acc={mean_raw[0]:.3f} f1={mean_raw[1]:.3f} rec={mean_raw[2]:.3f} iou={mean_raw[3]:.3f}")
    print(f"DESPUÉS(HDBSCAN)          acc={mean_clean[0]:.3f} f1={mean_clean[1]:.3f} rec={mean_clean[2]:.3f} iou={mean_clean[3]:.3f}")
    print(f"[DONE] Salidas guardadas en: {out_dir}")

if __name__ == "__main__":
    main()
