#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inferencia y visualización para modelos de segmentación (PointNet / PointNet++).
- Carga modelo (best/final) desde run_dir/run_single
- Evalúa sobre X_test.npz / Y_test.npz
- Genera: figuras GT vs Pred, mapas de error, matriz de confusión y métricas por clase

Ejemplo:
  python -m scripts.infer_vis_pointcloud \
    --run_dir runs_grid/pointnetpp_8192_pairs_s42 \
    --data_dir /ruta/a/data/3dteethseg/splits/8192_seed42_pairs \
    --which_model best \
    --num_samples_vis 8 --vis_subsample 10000 --save_svg
"""
import argparse
from pathlib import Path
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras

# -------------------- utils --------------------
def normalize_cloud_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mean = x.mean(axis=0, keepdims=True)
    x = x - mean
    r = np.linalg.norm(x, axis=1, keepdims=True).max(axis=0, keepdims=True)
    return x / (r + 1e-6)

def load_best_or_final(run_dir: Path, which_model: str | None):
    rs = run_dir / "run_single"
    if not rs.exists():
        raise FileNotFoundError(f"No se encontró carpeta run_single bajo: {run_dir}")
    if which_model is None:
        if (rs/"checkpoints/best").exists():
            which_model = "best"
        elif (rs/"final_model").exists():
            which_model = "final"
        else:
            raise FileNotFoundError(f"No existe best ni final en {rs}")
    mdl_path = rs / ("checkpoints/best" if which_model == "best" else "final_model")
    print(f"[LOAD] keras.models.load_model({mdl_path}, compile=False)")
    model = keras.models.load_model(mdl_path, compile=False)
    return model, rs

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def maybe_subsample(points: np.ndarray, *arrays, k: int = 0):
    if k and points.shape[0] > k:
        idx = np.random.choice(points.shape[0], size=k, replace=False)
        return (points[idx],) + tuple(a[idx] for a in arrays)
    return (points,) + arrays

def scatter3(ax, P, c, s=4, alpha=0.9):
    ax.scatter(P[:,0], P[:,1], P[:,2], c=c, s=s, alpha=alpha, depthshade=False)
    ax.set_axis_off()

def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    # y_true, y_pred: (N*P,)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm

def per_class_metrics_from_cm(cm: np.ndarray):
    tp = np.diag(cm).astype(np.float64)
    pred = cm.sum(axis=0).astype(np.float64)  # TP+FP
    gt   = cm.sum(axis=1).astype(np.float64)  # TP+FN
    prec = np.divide(tp, pred, out=np.zeros_like(tp), where=pred>0)
    rec  = np.divide(tp, gt,   out=np.zeros_like(tp), where=gt>0)
    iou  = np.divide(tp, (pred + gt - tp), out=np.zeros_like(tp), where=(pred + gt - tp)>0)
    return prec, rec, iou

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Carpeta padre que contiene run_single/")
    ap.add_argument("--data_dir", required=True, help="Carpeta con X_*.npz / Y_*.npz")
    ap.add_argument("--which_model", default=None, choices=[None, "best", "final"])
    ap.add_argument("--batch_size", type=int, default=16)

    # Visual
    ap.add_argument("--num_samples_vis", type=int, default=8)
    ap.add_argument("--dpi", type=int, default=180)
    ap.add_argument("--point_size", type=float, default=6.0)
    ap.add_argument("--alpha", type=float, default=0.95)
    ap.add_argument("--vis_subsample", type=int, default=0,
                    help="Si >0, muestrea esa cantidad de puntos por nube para dibujar.")
    ap.add_argument("--elev", type=float, default=20.0)
    ap.add_argument("--azim", type=float, default=-60.0)
    ap.add_argument("--save_svg", action="store_true")

    # Opcional: resaltar una etiqueta en predicción
    ap.add_argument("--highlight_label", type=int, default=None)
    ap.add_argument("--highlight_color", type=str, default="lime")
    ap.add_argument("--alpha_highlight", type=float, default=1.0)
    ap.add_argument("--size_highlight", type=float, default=2.5)
    ap.add_argument("--outline_highlight", action="store_true")

    # Matriz de confusión y métricas
    ap.add_argument("--make_confusion", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    data_dir = Path(args.data_dir)

    # Cargar modelo
    model, run_single = load_best_or_final(run_dir, args.which_model)

    # Cargar datos
    Xte = np.load(data_dir / "X_test.npz")["X"]  # (N,P,3)
    Yte = np.load(data_dir / "Y_test.npz")["Y"]  # (N,P)
    N, P, _ = Xte.shape
    ncls = int(max(Yte.max(), 0) + 1)
    print(f"[DATA] Test: {Xte.shape}, clases≈{ncls}")

    # Normalizar como en train
    Xte_norm = np.empty_like(Xte, dtype=np.float32)
    for i in range(N):
        Xte_norm[i] = normalize_cloud_np(Xte[i])

    # Inferencia por lotes
    probs = model.predict(Xte_norm, batch_size=args.batch_size, verbose=1)
    ypred = probs.argmax(axis=-1)  # (N,P)
    ncls_pred = probs.shape[-1]
    if ncls_pred != ncls:
        print(f"[WARN] ncls_pred={ncls_pred} difiere de ncls_gt={ncls}. Uso ncls={ncls_pred} para gráficos.")
        ncls = ncls_pred

    # Salidas
    out_dir = ensure_dir(run_single / "vis_infer")
    out_err = ensure_dir(out_dir / "errors")
    out_met = ensure_dir(out_dir / "metrics")

    # Colormap discreto
    cmap = plt.cm.get_cmap("tab20", ncls)

    # Visualizar muestras
    k = min(args.num_samples_vis, N)
    idxs = np.linspace(0, N - 1, k, dtype=int)

    for j, i in enumerate(idxs, 1):
        pts_raw = Xte[i]
        gt = Yte[i]
        pr = ypred[i]
        pts_draw, gt_draw, pr_draw = maybe_subsample(pts_raw, gt, pr, k=args.vis_subsample)

        fig = plt.figure(figsize=(12, 6), dpi=args.dpi)
        ax1 = fig.add_subplot(121, projection="3d"); ax1.view_init(args.elev, args.azim)
        ax2 = fig.add_subplot(122, projection="3d"); ax2.view_init(args.elev, args.azim)

        scatter3(ax1, pts_draw, c=cmap(gt_draw % cmap.N), s=args.point_size, alpha=args.alpha)
        ax1.set_title("Ground Truth")

        scatter3(ax2, pts_draw, c=cmap(pr_draw % cmap.N), s=args.point_size, alpha=args.alpha)
        ax2.set_title("Predicción")

        if args.highlight_label is not None:
            m = (pr_draw == int(args.highlight_label))
            if m.any():
                ax2.scatter(
                    pts_draw[m, 0], pts_draw[m, 1], pts_draw[m, 2],
                    c=args.highlight_color, s=args.point_size * args.size_highlight,
                    alpha=args.alpha_highlight, depthshade=False,
                    edgecolors="k" if args.outline_highlight else None
                )

        fig.tight_layout()
        base = f"{j:03d}"
        png_path = out_dir / f"{base}.png"
        fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
        if args.save_svg:
            fig.savefig(out_dir / f"{base}.svg", bbox_inches="tight")
        plt.close(fig)

        # Mapa de errores
        err_mask = (gt != pr)
        if err_mask.any():
            pts_err, err_draw = maybe_subsample(pts_raw, err_mask.astype(bool), k=args.vis_subsample)
            fig2 = plt.figure(figsize=(6, 6), dpi=args.dpi)
            axe = fig2.add_subplot(111, projection="3d"); axe.view_init(args.elev, args.azim)
            colors = np.where(err_draw, "crimson", "lightgray")
            scatter3(axe, pts_err, c=colors, s=args.point_size, alpha=1.0)
            axe.set_title("Errores (rojo)")
            fig2.tight_layout()
            fig2.savefig(out_err / f"{base}_errors.png", dpi=args.dpi, bbox_inches="tight")
            if args.save_svg:
                fig2.savefig(out_err / f"{base}_errors.svg", bbox_inches="tight")
            plt.close(fig2)

        print(f"[OK] {png_path}")

    # Matriz de confusión + métricas por clase
    if args.make_confusion:
        y_true_flat = Yte.reshape(-1)
        y_pred_flat = ypred.reshape(-1)
        cm = confusion_matrix_np(y_true_flat, y_pred_flat, num_classes=ncls)

        # Guardar cm cruda
        np.save(out_met / "confusion_matrix.npy", cm)

        # Plot heatmap
        fig = plt.figure(figsize=(8, 7), dpi=args.dpi)
        ax = fig.add_subplot(111)
        im = ax.imshow(cm, interpolation="nearest", cmap="viridis")
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("Matriz de confusión (todos los puntos de test)")
        ax.set_xlabel("Predicción"); ax.set_ylabel("Ground Truth")
        # ticks (si C grande, deja cada 2/5 para no saturar)
        step = 1
        if ncls > 40: step = 5
        elif ncls > 25: step = 2
        ticks = np.arange(0, ncls, step)
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks], rotation=90)
        ax.set_yticklabels([str(t) for t in ticks])
        fig.tight_layout()
        fig.savefig(out_met / "confusion_matrix.png", bbox_inches="tight")
        if args.save_svg:
            fig.savefig(out_met / "confusion_matrix.svg", bbox_inches="tight")
        plt.close(fig)

        # Métricas por clase
        prec, rec, iou = per_class_metrics_from_cm(cm)
        macro = {
            "precision_macro": float(np.mean(prec)) if len(prec) else 0.0,
            "recall_macro": float(np.mean(rec)) if len(rec) else 0.0,
            "miou_macro": float(np.mean(iou)) if len(iou) else 0.0,
            "accuracy": float(np.trace(cm) / max(1, cm.sum())),
        }
        with open(out_met / "macro_metrics.json", "w", encoding="utf-8") as f:
            json.dump(macro, f, indent=2)

        # CSV por clase
        import csv
        with open(out_met / "per_class_metrics.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["class", "precision", "recall", "iou", "support(=GT)"])
            gt_counts = cm.sum(axis=1)
            for c in range(ncls):
                w.writerow([c, f"{prec[c]:.6f}", f"{rec[c]:.6f}", f"{iou[c]:.6f}", int(gt_counts[c])])

        print(f"[METRICS] guardadas en: {out_met}")

    print(f"✅ Listo. Revisa: {out_dir}")

if __name__ == "__main__":
    main()
