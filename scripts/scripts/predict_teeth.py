#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Intenta importar tus capas/modelos, por si fueron usadas al guardar
try:
    from models.DentalPointNet import DentalPointNet, TNet  # adapta si tu módulo tiene otro nombre/clase
    CUSTOM_OBJECTS = {'TNet': TNet, 'DentalPointNet': DentalPointNet}
except Exception:
    CUSTOM_OBJECTS = {}  # si no existen, igualmente podemos cargar redes estándar

from infer_utils import (
    load_npz_splits, make_dataset, load_label_mapping,
    convert_labels_to_indices, convert_indices_to_labels,
    evaluate_model_in_batches, predict_one_sample,
    plot_history, plot_true_vs_pred_3d, DEFAULT_LABEL_COLORS
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True,
                    help="Carpeta del SavedModel (ej: runs/.../final_model)")
    ap.add_argument("--data_path", required=True,
                    help="Carpeta con X_*.npz / Y_*.npz")
    ap.add_argument("--out_dir", default="./runs/infer_out",
                    help="Carpeta donde guardar gráficas y métricas")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--sample_idx", type=int, default=0,
                    help="Índice de muestra del set de test para visualizar")
    ap.add_argument("--meta_json", default=None,
                    help="Ruta a meta.json con mapping de etiquetas (opcional)")
    ap.add_argument("--show_plots", action="store_true",
                    help="Mostrar plots en pantalla además de guardarlos")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Cargar datos
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte), info = load_npz_splits(args.data_path)
    print(f"[DATA] Test set: {Xte.shape}, num_classes≈{info['num_classes']}")

    # 2) Mapping de etiquetas (opcional)
    l2i, i2l = load_label_mapping(args.meta_json)
    Yte_idx = convert_labels_to_indices(Yte, l2i)  # si no hay mapping, es identidad

    # 3) Cargar modelo SavedModel
    print(f"[MODEL] Cargando modelo desde: {model_dir}")
    model = keras.models.load_model(model_dir, custom_objects=CUSTOM_OBJECTS)
    print("[MODEL] Cargado OK")

    # 4) Si existe history.json junto al modelo, graficar loss/acc
    #    (por convención lo dejamos en el run_dir; ajusta si lo guardas en otro sitio)
    run_dir = model_dir.parent  # asumiendo .../run_single/final_model
    hist_json = run_dir/"history.json"
    if hist_json.exists():
        hist = json.loads(hist_json.read_text(encoding="utf-8"))
        plot_history(hist, out_dir=out_dir, show=args.show_plots)
        print(f"[PLOT] Guardadas curvas en: {out_dir}")
    else:
        print("[PLOT] history.json no encontrado; me salto las curvas loss/acc.")

    # 5) Evaluación por lotes en test
    test_loss, test_acc = evaluate_model_in_batches(model, Xte, Yte_idx, batch_size=args.batch_size)
    (out_dir/"test_metrics.json").write_text(
        json.dumps({"test_loss": float(test_loss), "test_accuracy": float(test_acc)}, indent=2),
        encoding="utf-8"
    )
    print(f"[EVAL] test_loss={test_loss:.4f}  test_accuracy={test_acc:.4f}")

    # 6) Visualizar una muestra
    sidx = int(np.clip(args.sample_idx, 0, len(Xte)-1))
    pc   = Xte[sidx]               # (P, 3)
    yt   = Yte[sidx]               # etiquetas originales
    yt_idx = convert_labels_to_indices(yt, l2i)
    yp_idx = predict_one_sample(model, pc)                # índices predichos
    yp     = convert_indices_to_labels(yp_idx, i2l)       # etiquetas originales

    fig_path = out_dir / f"sample_{sidx:04d}_true_vs_pred.png"
    plot_true_vs_pred_3d(pc, yt, yp, i2l, out_path=fig_path, show=args.show_plots)
    print(f"[VIS] Plot (true vs pred) guardado en: {fig_path}")

    print("[DONE] Inferencia y evaluación completadas.")


if __name__ == "__main__":
    main()
