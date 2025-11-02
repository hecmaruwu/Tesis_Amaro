#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-procesamiento con HDBSCAN para segmentación binaria por puntos
-------------------------------------------------------------------
- Carga el modelo entrenado (PointNet++ o DGCNN).
- Genera logits / predicciones por paciente.
- Aplica HDBSCAN sobre los puntos predichos como 1.
- Conserva solo el cluster mayor y recalcula métricas globales.

Requiere:
    pip install hdbscan
"""

import os, torch, numpy as np
from pathlib import Path
import hdbscan
import torch.nn.functional as F
from tqdm import tqdm
import argparse
from train_pointnet2_binary_seg_v19g_tuned import PointNet2BinarySeg, UFRNBinaryDataset, collate, metrics_global

# =====================================================
def keep_largest_cluster(points, preds):
    """
    Mantiene solo el cluster mayoritario entre los puntos predichos como 1.
    """
    mask_pos = preds == 1
    if mask_pos.sum() == 0:
        return preds  # no hay positivos
    
    pts_pos = points[mask_pos]
    if len(pts_pos) < 5:
        return preds  # muy pocos puntos

    # Clustering
    clusterer = hdbscan.HDBSCAN(min_cluster_size=100, min_samples=10)
    labels = clusterer.fit_predict(pts_pos)

    if (labels >= 0).sum() == 0:
        return preds  # no se formó ningún cluster

    # Identificar cluster mayor
    largest = np.bincount(labels[labels >= 0]).argmax()
    mask_keep = np.zeros_like(preds, dtype=bool)
    mask_keep[np.where(mask_pos)[0][labels == largest]] = True

    # Nuevo vector de predicción limpio
    preds_clean = np.zeros_like(preds)
    preds_clean[mask_keep] = 1
    return preds_clean

# =====================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Carpeta con paciente_xx")
    ap.add_argument("--model_ckpt", required=True, help="Ruta a best.pt")
    ap.add_argument("--thr", type=float, default=0.65)
    ap.add_argument("--out_dir", default="post_hdbscan_results")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    print(f"🔍 Evaluando modelo en {args.data_dir} con HDBSCAN post-proceso (thr={args.thr:.2f})")

    # Dataset y modelo
    ds = UFRNBinaryDataset(args.data_dir, augment=False)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate)
    model = PointNet2BinarySeg().to(device)
    model.load_state_dict(torch.load(args.model_ckpt, map_location=device))
    model.eval()

    all_logits, all_y, all_y_clean = [], [], []

    with torch.no_grad():
        for X, Y, pid in tqdm(dl):
            X, Y = X.to(device), Y.to(device)
            logits = model(X)               # (1, N, 2)
            probs = F.softmax(logits, dim=-1)[..., 1].squeeze(0).cpu().numpy()
            pred = (probs >= args.thr).astype(np.int64)
            pts = X.squeeze(0).cpu().numpy()

            pred_clean = keep_largest_cluster(pts, pred)
            all_logits.append(torch.tensor(pred_clean))
            all_y.append(Y.squeeze(0).cpu())
            all_y_clean.append(torch.tensor(pred))

    # Métricas antes y después del post-proceso
    y_pred = torch.cat(all_y_clean).unsqueeze(-1)
    y_pred_hdb = torch.cat(all_logits).unsqueeze(-1)
    y_true = torch.cat(all_y).unsqueeze(-1)

    m_before = metrics_global(
        torch.cat([y_pred, 1 - y_pred], dim=-1).float(), y_true, thr=0.5)
    m_after = metrics_global(
        torch.cat([y_pred_hdb, 1 - y_pred_hdb], dim=-1).float(), y_true, thr=0.5)

    print("\n📊 MÉTRICAS GLOBALES")
    print(f"Antes del HDBSCAN: acc={m_before['acc']:.3f} f1={m_before['f1']:.3f} rec={m_before['recall']:.3f} iou={m_before['iou']:.3f}")
    print(f"Después del HDBSCAN: acc={m_after['acc']:.3f} f1={m_after['f1']:.3f} rec={m_after['recall']:.3f} iou={m_after['iou']:.3f}")

    np.save(out/"preds_hdbscan.npy", y_pred_hdb.numpy())
    print(f"\n✅ Guardado post-proceso → {out}/preds_hdbscan.npy")

if __name__ == "__main__":
    main()
