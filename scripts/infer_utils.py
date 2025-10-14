#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


# ==============================
# Carga de splits .npz
# ==============================
def load_npz_splits(data_path: str):
    p = Path(data_path)
    Xtr = np.load(p/"X_train.npz")["X"]
    Ytr = np.load(p/"Y_train.npz")["Y"]
    Xva = np.load(p/"X_val.npz")["X"]
    Yva = np.load(p/"Y_val.npz")["Y"]
    Xte = np.load(p/"X_test.npz")["X"]
    Yte = np.load(p/"Y_test.npz")["Y"]
    num_classes = int(max(Ytr.max(), Yva.max(), Yte.max()) + 1)
    return (Xtr, Ytr), (Xva, Yva), (Xte, Yte), {"num_classes": num_classes}


def make_dataset(X: np.ndarray, Y: np.ndarray, batch_size: int):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ==============================
# Mapping de etiquetas
# ==============================
def load_label_mapping(meta_path: Optional[str]) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Devuelve (label_to_index, index_to_label). Si meta_path es None o no existe,
    infiere un mapping identidad.
    meta.json esperado (si existe): { "label_to_index": {"11":1, "12":2, ...} }
    """
    if meta_path is None:
        return {}, {}  # identidad

    mp = Path(meta_path)
    if not mp.exists():
        return {}, {}

    meta = json.loads(mp.read_text(encoding="utf-8"))
    l2i = meta.get("label_to_index", None)
    if l2i is None:
        return {}, {}

    # keys del json son str → conviértelas a int
    l2i = {int(k): int(v) for k, v in l2i.items()}
    i2l = {v: k for k, v in l2i.items()}
    return l2i, i2l


def convert_labels_to_indices(y: np.ndarray, label_to_index: Dict[int, int]) -> np.ndarray:
    if not label_to_index:
        return y  # identidad
    vv = np.vectorize(lambda t: label_to_index.get(int(t), int(t)))
    return vv(y)


def convert_indices_to_labels(yidx: np.ndarray, index_to_label: Dict[int, int]) -> np.ndarray:
    if not index_to_label:
        return yidx  # identidad
    vv = np.vectorize(lambda t: index_to_label.get(int(t), int(t)))
    return vv(yidx)


# ==============================
# Colores por etiqueta (editable)
# ==============================
DEFAULT_LABEL_COLORS = {
    0: 'red',
    11: 'blue', 12: 'green', 13: 'orange', 14: 'purple', 15: 'cyan', 16: 'magenta', 17: 'yellow', 18: 'brown',
    21: 'lime', 22: 'navy', 23: 'teal', 24: 'violet', 25: 'salmon', 26: 'gold', 27: 'lightblue', 28: 'coral',
    31: 'olive', 32: 'silver', 33: 'gray', 34: 'black', 35: 'darkred', 36: 'darkgreen', 37: 'darkblue',
    38: 'darkviolet', 41: 'peru', 42: 'chocolate', 43: 'mediumvioletred', 44: 'lightskyblue',
    45: 'lightpink', 46: 'plum', 47: 'khaki', 48: 'powderblue',
}


# ==============================
# Evaluación por lotes
# ==============================
def evaluate_model_in_batches(model: keras.Model,
                              X: np.ndarray,
                              Y: np.ndarray,
                              batch_size: int = 32) -> Tuple[float, float]:
    ds = make_dataset(X, Y, batch_size)
    # esto devuelve dict si el modelo se compiló con metrics con nombre
    metrics = model.evaluate(ds, verbose=1, return_dict=True)
    # fallback si el modelo fue cargado sin nombres
    if isinstance(metrics, dict):
        loss = float(metrics.get("loss", 0.0))
        acc  = float(metrics.get("accuracy", metrics.get("acc", 0.0)))
    else:
        # típico: [loss, acc]
        loss = float(metrics[0]) if len(metrics) > 0 else 0.0
        acc  = float(metrics[1]) if len(metrics) > 1 else 0.0
    return loss, acc


# ==============================
# Predicción de un sample
# ==============================
def predict_one_sample(model: keras.Model,
                       point_cloud: np.ndarray) -> np.ndarray:
    """
    point_cloud: (P, 3)
    return: predicted indices (P,)
    """
    pc = np.expand_dims(point_cloud, axis=0)  # (1, P, 3)
    probs = model.predict(pc, verbose=0)
    pred_idx = np.argmax(probs, axis=-1)[0]   # (P,)
    return pred_idx


# ==============================
# Plots
# ==============================
def plot_history(history: dict, out_dir: Path, show: bool = False):
    # history: dict como el de history.history
    out_dir.mkdir(parents=True, exist_ok=True)

    # Loss
    plt.figure()
    if 'loss' in history: plt.plot(history['loss'], label='Training Loss')
    if 'val_loss' in history: plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend()
    plt.tight_layout()
    (out_dir/"loss.png").write_bytes(plt.savefig(fname=None, format='png') or b'')
    if show: plt.show()
    plt.close()

    # Accuracy
    plt.figure()
    if 'accuracy' in history: plt.plot(history['accuracy'], label='Training Acc')
    if 'val_accuracy' in history: plt.plot(history['val_accuracy'], label='Validation Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.legend()
    plt.tight_layout()
    (out_dir/"accuracy.png").write_bytes(plt.savefig(fname=None, format='png') or b'')
    if show: plt.show()
    plt.close()


def plot_true_vs_pred_3d(point_cloud: np.ndarray,
                         true_labels: np.ndarray,
                         pred_labels: np.ndarray,
                         index_to_label: Dict[int, int],
                         out_path: Path,
                         label_colors: Dict[int, str] = DEFAULT_LABEL_COLORS,
                         show: bool = False):
    """
    true_labels y pred_labels deben ser etiquetas "originales" (no índices),
    para poder mapear colores por ID clínico si usas mapping.
    """
    # Mapear colores
    true_colors = np.array([label_colors.get(int(l), 'gray') for l in true_labels])
    pred_colors = np.array([label_colors.get(int(l), 'gray') for l in pred_labels])

    fig = plt.figure(figsize=(9, 9))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(point_cloud[:,0], point_cloud[:,1], point_cloud[:,2], c=true_colors, s=1)
    ax1.set_title('True Segmentation')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(point_cloud[:,0], point_cloud[:,1], point_cloud[:,2], c=pred_colors, s=1)
    ax2.set_title('Predicted Segmentation')

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)
