#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# ====== Métricas base ======
class _ConfusionMatrixMetric(tf.keras.metrics.Metric):
    def __init__(self, num_classes: int, name="cm_metric", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = int(num_classes)
        self.cm = self.add_weight(name="cm", shape=(self.num_classes, self.num_classes),
                                  initializer="zeros", dtype=tf.float32)
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(tf.cast(y_true, tf.int32), [-1])
        y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        y_pred = tf.reshape(y_pred, [-1])
        cm = tf.math.confusion_matrix(y_true, y_pred,
                                      num_classes=self.num_classes, dtype=tf.float32)
        self.cm.assign_add(cm)
    def reset_state(self):
        self.cm.assign(tf.zeros_like(self.cm))
    def reset_states(self):
        self.reset_state()

class PrecisionMacro(_ConfusionMatrixMetric):
    def result(self):
        tp = tf.linalg.tensor_diag_part(self.cm)
        pred = tf.reduce_sum(self.cm, axis=0)
        prec = tf.math.divide_no_nan(tp, pred)
        return tf.reduce_mean(prec)

class RecallMacro(_ConfusionMatrixMetric):
    def result(self):
        tp = tf.linalg.tensor_diag_part(self.cm)
        gt = tf.reduce_sum(self.cm, axis=1)
        rec = tf.math.divide_no_nan(tp, gt)
        return tf.reduce_mean(rec)

class F1Macro(_ConfusionMatrixMetric):
    def result(self):
        tp = tf.linalg.tensor_diag_part(self.cm)
        gt = tf.reduce_sum(self.cm, axis=1)
        pred = tf.reduce_sum(self.cm, axis=0)
        prec = tf.math.divide_no_nan(tp, pred)
        rec  = tf.math.divide_no_nan(tp, gt)
        f1 = tf.math.divide_no_nan(2.0 * prec * rec, prec + rec)
        return tf.reduce_mean(f1)

class SparseMeanIoU(_ConfusionMatrixMetric):
    def result(self):
        tp = tf.linalg.tensor_diag_part(self.cm)
        gt = tf.reduce_sum(self.cm, axis=1)
        pred = tf.reduce_sum(self.cm, axis=0)
        union = gt + pred - tp
        iou = tf.math.divide_no_nan(tp, union)
        return tf.reduce_mean(iou)

CUSTOM_OBJECTS = {
    "PrecisionMacro": PrecisionMacro,
    "RecallMacro": RecallMacro,
    "F1Macro": F1Macro,
    "SparseMeanIoU": SparseMeanIoU,
}

# ====== Utilidades ======
def load_npz_splits(data_dir: Path):
    Xte = np.load(data_dir/"X_test.npz")["X"]
    Yte = np.load(data_dir/"Y_test.npz")["Y"]
    meta = {}
    mpath = data_dir/"meta.json"
    if mpath.exists():
        meta = json.loads(mpath.read_text())
    return Xte, Yte, meta

def batched_predict(model, X, batch_size=8):
    preds = []
    for i in range(0, X.shape[0], batch_size):
        xb = X[i:i+batch_size]
        pb = model.predict(xb, verbose=0)
        preds.append(pb)
    return np.concatenate(preds, axis=0)

def fig_to_png_bytes(fig):
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=fig.dpi, bbox_inches='tight', pad_inches=0.02)
    buf.seek(0)
    return buf.read()

def _save_vis_triplet(out_dir, pts, y_true, y_pred, idx, dpi=300):
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_idx = np.arange(pts.shape[0])
    if pts.shape[0] > 5000:
        vis_idx = np.random.default_rng(0).choice(pts.shape[0], 5000, replace=False)
    pv = pts[vis_idx]; tg = y_true[vis_idx]; pd = y_pred[vis_idx]
    for kind, lbl in zip([tg, pd], ["gt", "pred"]):
        fig = plt.figure(figsize=(8,6), dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(pv[:,0], pv[:,1], pv[:,2], c=kind, s=1, cmap='tab20')
        plt.colorbar(sc, ax=ax, shrink=0.6)
        ax.set_title(f'{lbl.upper()} sample {idx}')
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        fig.tight_layout()
        (out_dir / f'{lbl}_{idx}.png').write_bytes(fig_to_png_bytes(fig))
        plt.close(fig)

# ====== Métricas detalladas ======
def evaluate_detailed(y_true, y_pred, num_classes, class_names):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for i in range(len(y_true)):
        yt = y_true[i].reshape(-1)
        yp = y_pred[i].reshape(-1)
        for a, b in zip(yt, yp):
            cm[a, b] += 1

    tp = np.diag(cm)
    gt = cm.sum(axis=1)
    pred = cm.sum(axis=0)
    iou = np.divide(tp, gt + pred - tp, out=np.zeros_like(tp, dtype=float), where=(gt+pred-tp)!=0)
    acc = np.divide(tp, gt, out=np.zeros_like(tp, dtype=float), where=gt!=0)
    prec = np.divide(tp, pred, out=np.zeros_like(tp, dtype=float), where=pred!=0)
    rec = np.divide(tp, gt, out=np.zeros_like(tp, dtype=float), where=gt!=0)
    f1 = np.divide(2*prec*rec, prec+rec, out=np.zeros_like(prec, dtype=float), where=(prec+rec)!=0)

    mean_acc = np.nanmean(acc)
    mean_iou = np.nanmean(iou)
    macro_f1 = np.nanmean(f1)
    oa = (tp.sum() / cm.sum())

    metrics = {
        "overall_accuracy": float(oa),
        "mean_accuracy": float(mean_acc),
        "mean_iou": float(mean_iou),
        "macro_f1": float(macro_f1),
        "per_class": {}
    }
    for i, name in enumerate(class_names):
        metrics["per_class"][name] = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "iou": float(iou[i]),
            "f1": float(f1[i]),
        }
    return metrics

CLASS_NAMES = ["Encía"] + [f"Diente {i}" for i in 
                           [11,12,13,14,15,16,17,18,
                            21,22,23,24,25,26,27,28,
                            31,32,33,34,35,36,37,38,
                            41,42,43,44,45,46,47,48]]

# ====== Gráfico de barras F1 por clase ======
def plot_f1_per_class(metrics, out_path):
    names = list(metrics["per_class"].keys())
    f1_values = [metrics["per_class"][n]["f1"] for n in names]
    colors = ["lime" if "21" in n else "#007acc" for n in names]
    fig, ax = plt.subplots(figsize=(10,6), dpi=300)
    bars = ax.barh(names, f1_values, color=colors)
    ax.set_xlabel("F1-score")
    ax.set_ylabel("Clase")
    ax.set_title("F1 por clase (Diente 21 resaltado)")
    ax.set_xlim(0, 1)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[OK] Gráfico de F1 guardado en {out_path}")

# ====== Main ======
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--which_model", default="final", choices=["final", "best"])
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_samples_vis", type=int, default=3)
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    data_dir = Path(args.data_dir)
    Xte, Yte, _ = load_npz_splits(data_dir)
    num_classes = int(Yte.max() + 1)
    print(f"[DATA] Test {Xte.shape}, classes={num_classes}")

    model_path = run_dir / ("checkpoints/best" if args.which_model=="best" else "final_model")
    print(f"[MODEL] Loading from {model_path}")
    model = tf.keras.models.load_model(str(model_path), custom_objects=CUSTOM_OBJECTS, compile=False)
    model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    preds = batched_predict(model, Xte, batch_size=args.batch_size)
    yhat = preds.argmax(axis=-1)
    acc = float((yhat == Yte).mean())
    print(f"[TEST] Accuracy global: {acc:.4f}")

    metrics = evaluate_detailed(Yte, yhat, num_classes, CLASS_NAMES[:num_classes])
    detailed_path = run_dir / f"metrics_detailed_{args.which_model}.json"
    detailed_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[OK] Métricas detalladas guardadas en {detailed_path}")

    # === Gráfico F1 ===
    f1_plot_path = run_dir / f"f1_per_class_{args.which_model}.png"
    plot_f1_per_class(metrics, f1_plot_path)

    # === Visualización ===
    vis_dir = run_dir / f"vis_{args.which_model}"
    vis_dir.mkdir(exist_ok=True, parents=True)
    sel = np.linspace(0, Xte.shape[0]-1, num=min(args.num_samples_vis, Xte.shape[0]), dtype=int)
    for j, k in enumerate(sel):
        _save_vis_triplet(vis_dir, Xte[k], Yte[k], yhat[k], j, dpi=args.dpi)
    print(f"[VIS] Figuras guardadas en {vis_dir}")

if __name__ == "__main__":
    main()
