#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# ====== Métricas custom (para cargar el modelo) ======
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

    def reset_state(self):  # API moderna
        self.cm.assign(tf.zeros_like(self.cm))

    def reset_states(self):  # compat
        self.reset_state()

class PrecisionMacro(_ConfusionMatrixMetric):
    def result(self):
        cm = self.cm
        tp   = tf.linalg.tensor_diag_part(cm)
        pred = tf.reduce_sum(cm, axis=0)
        prec = tf.math.divide_no_nan(tp, pred)
        return tf.reduce_mean(prec)

class RecallMacro(_ConfusionMatrixMetric):
    def result(self):
        cm = self.cm
        tp = tf.linalg.tensor_diag_part(cm)
        gt = tf.reduce_sum(cm, axis=1)
        rec = tf.math.divide_no_nan(tp, gt)
        return tf.reduce_mean(rec)

class F1Macro(_ConfusionMatrixMetric):
    def result(self):
        cm = self.cm
        tp   = tf.linalg.tensor_diag_part(cm)
        gt   = tf.reduce_sum(cm, axis=1)
        pred = tf.reduce_sum(cm, axis=0)
        prec = tf.math.divide_no_nan(tp, pred)
        rec  = tf.math.divide_no_nan(tp, gt)
        f1 = tf.math.divide_no_nan(2.0 * prec * rec, prec + rec)
        return tf.reduce_mean(f1)

class SparseMeanIoU(_ConfusionMatrixMetric):
    def result(self):
        cm = self.cm
        tp   = tf.linalg.tensor_diag_part(cm)
        gt   = tf.reduce_sum(cm, axis=1)
        pred = tf.reduce_sum(cm, axis=0)
        union = gt + pred - tp
        iou = tf.math.divide_no_nan(tp, union)
        return tf.reduce_mean(iou)

# Para cargar el modelo con tus clases personalizadas
CUSTOM_OBJECTS = {

    "PrecisionMacro": PrecisionMacro,
    "RecallMacro": RecallMacro,
    "F1Macro": F1Macro,
    "SparseMeanIoU": SparseMeanIoU,
}

# --- STUBS para deserialización segura (evitar error num_classes) ---
# Durante load_model(..., compile=False) Keras puede intentar instanciar las métricas.
# Damos fábricas 'dummy' que satisfacen la firma; luego recompilamos con num_classes real.
CUSTOM_OBJECTS.setdefault("PrecisionMacro", lambda **kw: PrecisionMacro(num_classes=1, **{k:v for k,v in kw.items() if k!="num_classes"}))
CUSTOM_OBJECTS.setdefault("RecallMacro",    lambda **kw: RecallMacro(num_classes=1, **{k:v for k,v in kw.items() if k!="num_classes"}))
CUSTOM_OBJECTS.setdefault("F1Macro",        lambda **kw: F1Macro(num_classes=1, **{k:v for k,v in kw.items() if k!="num_classes"}))
CUSTOM_OBJECTS.setdefault("SparseMeanIoU",  lambda **kw: SparseMeanIoU(num_classes=1, **{k:v for k,v in kw.items() if k!="num_classes"}))

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
    n = X.shape[0]
    for i in range(0, n, batch_size):
        xb = X[i:i+batch_size]
        pb = model.predict(xb, verbose=0)  # [B,P,C]
        preds.append(pb)
    return np.concatenate(preds, axis=0)

def save_scatter(points, labels, title, out_path, dpi=600, label_colors=None):
    plt.ioff()
    fig = plt.figure(figsize=(10,8), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    if label_colors is None:
        sc = ax.scatter(points[:,0], points[:,1], points[:,2],
                        c=labels, s=1, cmap="tab20")
        plt.colorbar(sc, ax=ax, shrink=0.6)
    else:
        # mapa de color custom dict {label: 'color'}
        # (convertimos a lista de colores por punto)
        import numpy as np
        cols = np.array([label_colors.get(int(l), "gray") for l in labels])
        ax.scatter(points[:,0], points[:,1], points[:,2], c=cols, s=1)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

DEFAULT_LABEL_COLORS = {

    0: 'red', 11: 'blue', 12: 'green', 13: 'orange', 14: 'purple', 15: 'cyan',
    16: 'magenta', 17: 'yellow', 18: 'brown', 21: 'lime', 22: 'navy', 23: 'teal',
    24: 'violet', 25: 'salmon', 26: 'gold', 27: 'lightblue', 28: 'coral',
    31: 'olive', 32: 'silver', 33: 'gray', 34: 'black', 35: 'darkred',
    36: 'darkgreen', 37: 'darkblue', 38: 'darkviolet', 41: 'peru',
    42: 'chocolate', 43: 'mediumvioletred', 44: 'lightskyblue', 45: 'lightpink',
    46: 'plum', 47: 'khaki', 48: 'powderblue',
}


def resolve_model_path(run_dir: Path, which: str) -> Path:
    candidates = []
    rels = ["final_model"] if which == "final" else ["checkpoints/best"]
    for rel in rels:
        candidates.append(run_dir / rel)
        candidates.append(run_dir / "run_single" / rel)
        for sub in run_dir.glob("*/run_single"):
            candidates.append(sub / rel)
    for c in candidates:
        if c.exists():
            return c.resolve()
    sm = list(run_dir.rglob("saved_model.pb"))
    if sm:
        return Path(sm[0]).parent
    raise FileNotFoundError(f"No pude encontrar modelo ('{which}') bajo {run_dir}")

def build_metrics(num_classes: int):
    return [
        'accuracy',
        PrecisionMacro(num_classes=num_classes, name="prec_macro"),
        RecallMacro(num_classes=num_classes,    name="rec_macro"),
        F1Macro(num_classes=num_classes,        name="f1_macro"),
        SparseMeanIoU(num_classes=num_classes,  name="miou"),
    ]

def safe_load_and_compile(model_path: Path, num_classes: int, custom_objects: dict):
    import tensorflow as tf
    # Ir directo a compile=False para evitar deserializar métricas con estado incompleto
    m = tf.keras.models.load_model(str(model_path), custom_objects=custom_objects, compile=False)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    m.compile(optimizer="adam", loss=loss, metrics=build_metrics(num_classes))
    return m


def run_savedmodel_inference(model_dir: Path, X, batch_size=8):
    """
    Carga un SavedModel (TensorFlow puro) y devuelve yhat [N,P] usando la firma serving_default.
    """
    import tensorflow as tf
    sm = tf.saved_model.load(str(model_dir))
    sig = sm.signatures.get("serving_default", None)
    if sig is None:
        # si no tiene nombre serving_default, toma alguna firma disponible
        sigs = list(sm.signatures.keys())
        if not sigs:
            raise RuntimeError("SavedModel no tiene firmas exportadas.")
        sig = sm.signatures[sigs[0]]

    # Detectar el nombre del input (primer TensorSpec)
    in_names = list(sig.structured_input_signature[1].keys())
    if not in_names:
        raise RuntimeError("No se pudo detectar el input del SavedModel.")
    in_key = in_names[0]

    preds = []
    N = X.shape[0]
    for i in range(0, N, batch_size):
        xb = X[i:i+batch_size]
        out = sig(**{in_key: tf.convert_to_tensor(xb)})
        # Obtener primer tensor de salida
        if isinstance(out, dict):
            y = next(iter(out.values()))
        else:
            # EagerTensor directo
            y = out
        preds.append(y.numpy())
    P = preds[0].shape[1]
    import numpy as np
    return np.concatenate(preds, axis=0)  # [N,P,C]



def _save_vis_triplet(out_dir, pts, y_true, y_pred, idx, dpi=300):
    """
    Guarda dos figuras: GT y Pred. pts:[P,3], y_*:[P]
    """
    import numpy as np
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    # submuestreo visual si hay demasiados puntos
    vis_idx = np.arange(pts.shape[0])
    if pts.shape[0] > 5000:
        vis_idx = np.random.default_rng(0).choice(pts.shape[0], 5000, replace=False)
    pv = pts[vis_idx]
    tg = y_true[vis_idx]
    pd = y_pred[vis_idx]

    # GT
    fig = plt.figure(figsize=(8,6), dpi=dpi)
    ax  = fig.add_subplot(111, projection='3d')
    sc  = ax.scatter(pv[:,0], pv[:,1], pv[:,2], c=tg, s=1, cmap='tab20')
    plt.colorbar(sc, ax=ax, shrink=0.6)
    ax.set_title(f'GT sample {idx}')
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    fig.tight_layout()
    (out_dir / f'gt_{idx}.png').write_bytes(fig_to_png_bytes(fig))
    plt.close(fig)

    # Pred
    fig = plt.figure(figsize=(8,6), dpi=dpi)
    ax  = fig.add_subplot(111, projection='3d')
    sc  = ax.scatter(pv[:,0], pv[:,1], pv[:,2], c=pd, s=1, cmap='tab20')
    plt.colorbar(sc, ax=ax, shrink=0.6)
    ax.set_title(f'Pred sample {idx}')
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    fig.tight_layout()
    (out_dir / f'pred_{idx}.png').write_bytes(fig_to_png_bytes(fig))
    plt.close(fig)

def fig_to_png_bytes(fig):
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=fig.dpi, bbox_inches='tight', pad_inches=0.02)
    buf.seek(0)
    return buf.read()


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True,
                    help="Carpeta del experimento: contiene checkpoints/ y final_model/")
    ap.add_argument("--data_dir", required=True,
                    help="Carpeta con X_*.npz / Y_*.npz (el split que usaste)")
    ap.add_argument("--which_model", default="final",
                    choices=["final", "best"], help="Cargar 'final' o 'best' checkpoint")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_samples_vis", type=int, default=3, help="Cuántas figuras guardar")
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    data_dir = Path(args.data_dir)

    # 1) Cargar datos de test
    Xte, Yte, meta = load_npz_splits(data_dir)
    num_classes = int(Yte.max() + 1)
    print(f"[DATA] Test: {Xte.shape}, classes={num_classes}")

    # 2) Cargar modelo
    model_path = resolve_model_path(run_dir, args.which_model)
    print(f"[MODEL] Resolved path: {model_path}")
    # Inferimos num_classes desde Y_test
    _, Ytmp, _ = load_npz_splits(data_dir)
    num_classes = int(Ytmp.max() + 1)
    # Añadimos posibles capas custom de tu modelo (si no las usas, no molesta)
    try:
        from train_models import TNet, DentalPointNet
        CUSTOM_OBJECTS.update({"TNet": TNet, "DentalPointNet": DentalPointNet})
    except Exception:
        pass
    try:
        model = safe_load_and_compile(model_path, num_classes, CUSTOM_OBJECTS)
    except Exception as e:
        print('[FALLBACK] No se pudo cargar con Keras. Usando tf.saved_model.load para inferencia.\n ', e)
        model = None

    # 3) Evaluación en test
    try:
        test_metrics = model.evaluate(Xte, Yte, verbose=1, return_dict=True)
    except Exception:
        print('[FALLBACK] Evaluación Keras no disponible. Calculando métricas básicas en NumPy...')
        import numpy as np
        # Accuracy
        if 'preds' not in locals():
            # si aún no hemos hecho inferencia, hacerla
            if model is not None:
                preds = batched_predict(model, Xte, batch_size=args.batch_size)
            else:
                preds = run_savedmodel_inference(model_path, Xte, batch_size=args.batch_size)
        yhat_tmp = preds.argmax(axis=-1)
        acc = float((yhat_tmp == Yte).mean())
        test_metrics = {'accuracy': acc}
    out_metrics = run_dir / f"infer_test_metrics_{args.which_model}.json"
    out_metrics.write_text(json.dumps({k: float(v) for k, v in test_metrics.items()}, indent=2), encoding="utf-8")
    print("[OK] Test metrics saved:", out_metrics)

    # 4) Predicción por lotes y figuras
    if model is not None:
        preds = batched_predict(model, Xte, batch_size=args.batch_size)  # [N,P,C]
    else:
        preds = run_savedmodel_inference(model_path, Xte, batch_size=args.batch_size)  # [N,P,C]
    yhat = preds.argmax(axis=-1)  # [N,P]
    # ==== VISUALIZACIÓN ====
    # Crear carpeta de destino según which_model
    vis_name = f"vis_{args.which_model}"
    # Si el run_dir dado era el padre, muchos setups guardan dentro de run_single
    real_run_dir = Path(args.run_dir)
    if (real_run_dir / "run_single").is_dir():
        real_run_dir = real_run_dir / "run_single"
    vis_dir = real_run_dir / vis_name
    vis_dir.mkdir(parents=True, exist_ok=True)

    if args.num_samples_vis > 0:
        import numpy as np
        N = Xte.shape[0]
        # indices espaciados para cubrir test
        sel = np.linspace(0, N-1, num=min(args.num_samples_vis, N), dtype=int)
        print(f"[VIS] Guardando {len(sel)} muestras en {vis_dir} (dpi={args.dpi})")
        for j, k in enumerate(sel):
            _save_vis_triplet(vis_dir, Xte[k], Yte[k], yhat[k], j, dpi=args.dpi)
    else:
        print(f"[VIS] args.num_samples_vis == 0 → no se guardan figuras, pero se dejó creada {vis_dir}")

