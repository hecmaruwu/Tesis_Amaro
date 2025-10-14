#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Entrenamiento de segmentación por puntos (PointNet) con el NUEVO esquema de carpetas.

Esperado (ejemplo):
Tesis_final/
└── data/
    └── data_teeth3ds/
        └── splits/
            └── 8192/
               ├── X_train.npz   # (N,P,3) float32
               ├── Y_train.npz   # (N,P)   int
               ├── X_val.npz
               ├── Y_val.npz
               ├── X_test.npz
               └── Y_test.npz

Cómo llamarlo (ejemplo):
python scripts/train_models.py \
  --data_path data/data_teeth3ds/splits/8192 \
  --out_dir runs \
  --tag pointnet_8192_s42 \
  --epochs 60 --batch_size 8 --metrics_macro
"""

import os, json, argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ============================================================
# GPU / Entorno
# ============================================================
def setup_devices(cuda_visible: str | None, multi_gpu: bool):
    if cuda_visible is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[INFO] GPUs visibles: {len(gpus)} -> {gpus}")
        except Exception as e:
            print("[WARN] No pude setear memory_growth:", e)
    else:
        print("[INFO] GPUs visibles: [] (usará CPU)")

    strategy = None
    if multi_gpu:
        g = tf.config.list_physical_devices('GPU')
        if len(g) > 1:
            strategy = tf.distribute.MirroredStrategy()
            print(f"[INFO] Usando MirroredStrategy con {len(g)} GPUs.")
        else:
            print("[WARN] --multi_gpu activado pero no hay múltiples GPUs visibles.")
    return strategy

# ============================================================
# Datos (.npz con X_*, Y_*) + Normalización/Augment
# ============================================================
def load_npz_splits(data_path: Path):
    p = Path(data_path)
    req = ["X_train.npz","Y_train.npz","X_val.npz","Y_val.npz","X_test.npz","Y_test.npz"]
    missing = [r for r in req if not (p/r).exists()]
    if missing:
        raise FileNotFoundError(
            f"Faltan archivos en {p}:\n  " + "\n  ".join(missing) +
            "\nAsegúrate de tener el dataset exportado en el nuevo esquema "
            "(ej: data/data_teeth3ds/splits/8192/)."
        )
    Xtr = np.load(p/"X_train.npz")["X"]     # (N,P,3) float32
    Ytr = np.load(p/"Y_train.npz")["Y"]     # (N,P)   int
    Xva = np.load(p/"X_val.npz")["X"]
    Yva = np.load(p/"Y_val.npz")["Y"]
    Xte = np.load(p/"X_test.npz")["X"]
    Yte = np.load(p/"Y_test.npz")["Y"]
    print(f"[DATA] X_train:{Xtr.shape} X_val:{Xva.shape} X_test:{Xte.shape}  (path={p})")
    num_classes = int(max(Ytr.max(), Yva.max(), Yte.max()) + 1)
    return (Xtr, Ytr), (Xva, Yva), (Xte, Yte), {"num_classes": num_classes}

# ---------- Normalización (siempre aplicada) ----------
@tf.function
def _normalize_cloud(x):
    # x: (P,3) o (B,P,3)
    x = tf.cast(x, tf.float32)
    mean = tf.reduce_mean(x, axis=-2, keepdims=True)  # media por nube
    x = x - mean
    # radio máximo (norma 2) sobre todos los puntos
    r = tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1, keepdims=True))  # (...,P,1)
    r = tf.reduce_max(r, axis=-2, keepdims=True)                      # (...,1,1)
    x = x / (r + 1e-6)
    return x

# ---------- Augmentation suave (solo train) ----------
def _augment_cloud(x, y, rot_deg=10.0, jitter_std=0.005, scale_low=0.9, scale_high=1.1):
    # Rotación pequeña en Z
    theta = tf.random.uniform([], minval=-rot_deg, maxval=rot_deg, dtype=tf.float32) * (tf.constant(np.pi)/180.0)
    c, s = tf.cos(theta), tf.sin(theta)
    R = tf.stack([[c, -s, 0.0],
                  [s,  c, 0.0],
                  [0.0, 0.0, 1.0]], axis=0)  # (3,3)
    x = tf.matmul(x, R)  # (P,3)

    # Escala leve isotrópica
    s = tf.random.uniform([], minval=scale_low, maxval=scale_high, dtype=tf.float32)
    x = x * s

    # Jitter gaussiano suave
    noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=jitter_std, dtype=tf.float32)
    x = x + noise
    return x, y

def make_datasets(data_path: str, batch_size: int, seed: int,
                  do_augment: bool, rot_deg: float, jitter_std: float,
                  scale_low: float, scale_high: float):
    (Xtr,Ytr), (Xva,Yva), (Xte,Yte), info = load_npz_splits(Path(data_path))

    def _ds(X, Y, shuffle=False, augment=False):
        ds = tf.data.Dataset.from_tensor_slices((X, Y))
        if shuffle:
            ds = ds.shuffle(buffer_size=min(4096, X.shape[0]), seed=seed, reshuffle_each_iteration=True)
        # normalización SIEMPRE
        ds = ds.map(lambda x,y: (_normalize_cloud(x), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
        # augmentation SOLO train (muy suave)
        if augment:
            ds = ds.map(lambda x,y: _augment_cloud(x, y, rot_deg, jitter_std, scale_low, scale_high),
                        num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    ds_tr = _ds(Xtr, Ytr, shuffle=True,  augment=do_augment)
    ds_va = _ds(Xva, Yva, shuffle=False, augment=False)
    ds_te = _ds(Xte, Yte, shuffle=False, augment=False)
    return ds_tr, ds_va, ds_te, info

# ============================================================
# Métricas macro y mIoU
# ============================================================
class _ConfusionMatrixMetric(tf.keras.metrics.Metric):
    def __init__(self, num_classes: int, name="cm_metric", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = int(num_classes)
        self.cm = self.add_weight(
            name="cm",
            shape=(self.num_classes, self.num_classes),
            initializer="zeros",
            dtype=tf.float32,
        )
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(tf.cast(y_true, tf.int32), [-1])          # (B*P,)
        y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)     # (B,P)
        y_pred = tf.reshape(y_pred, [-1])
        cm = tf.math.confusion_matrix(y_true, y_pred,
                                      num_classes=self.num_classes, dtype=tf.float32)
        self.cm.assign_add(cm)
    def reset_state(self):
        self.cm.assign(tf.zeros_like(self.cm))
    def reset_states(self):  # compat
        self.reset_state()

class PrecisionMacro(_ConfusionMatrixMetric):
    def result(self):
        cm = self.cm
        tp   = tf.linalg.tensor_diag_part(cm)
        pred = tf.reduce_sum(cm, axis=0)  # TP+FP
        prec = tf.math.divide_no_nan(tp, pred)
        return tf.reduce_mean(prec)

class RecallMacro(_ConfusionMatrixMetric):
    def result(self):
        cm = self.cm
        tp = tf.linalg.tensor_diag_part(cm)
        gt = tf.reduce_sum(cm, axis=1)    # TP+FN
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

# ============================================================
# Pérdidas por punto
# ============================================================
class WeightedSparseCE(tf.keras.losses.Loss):
    def __init__(self, class_weights: dict, num_classes: int, name="w_sce"):
        super().__init__(name=name)
        w = np.ones((num_classes,), dtype=np.float32)
        for k, v in class_weights.items():
            k = int(k)
            if 0 <= k < num_classes:
                w[k] = float(v)
        self.w = tf.constant(w, dtype=tf.float32)
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)  # (B,P)
        base = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)  # (B,P)
        w = tf.gather(self.w, y_true)  # (B,P)
        return tf.reduce_mean(base * w)

class FocalSparseCE(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha_vec=None, num_classes=None, name="focal_sce"):
        super().__init__(name=name)
        self.gamma = float(gamma)
        if alpha_vec is not None:
            if num_classes is None:
                raise ValueError("num_classes requerido si se pasa alpha_vec")
            a = np.ones((num_classes,), dtype=np.float32)
            for k, v in alpha_vec.items():
                k = int(k)
                if 0 <= k < num_classes:
                    a[k] = float(v)
            self.alpha = tf.constant(a, dtype=tf.float32)
        else:
            self.alpha = None
    def call(self, y_true, y_pred):
        eps = tf.keras.backend.epsilon()
        y_true = tf.cast(y_true, tf.int32)                       # (B,P)
        y_true_oh = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])  # (B,P,C)
        p = tf.clip_by_value(y_pred, eps, 1.0)                   # (B,P,C)
        pt = tf.reduce_sum(y_true_oh * p, axis=-1)               # (B,P)
        if self.alpha is not None:
            a = tf.gather(self.alpha, y_true)                    # (B,P)
        else:
            a = 1.0
        loss = - a * tf.pow(1.0 - pt, self.gamma) * tf.math.log(pt)
        return tf.reduce_mean(loss)

# ============================================================
# Modelo PointNet
# ============================================================
class TNet(layers.Layer):
    def __init__(self, K=3, activation="relu"):
        super().__init__()
        act = activation
        self.K = K
        self.conv1 = layers.Conv1D(64,   1, activation=act)
        self.conv2 = layers.Conv1D(128,  1, activation=act)
        self.conv3 = layers.Conv1D(1024, 1, activation=act)
        self.fc1   = layers.Dense(512, activation=act)
        self.fc2   = layers.Dense(256, activation=act)
        self.fc3   = layers.Dense(K*K)
        self.reshape = layers.Reshape((K,K))
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = layers.GlobalMaxPooling1D()(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.reshape(x)
        bs = tf.shape(x)[0]
        I  = tf.eye(self.K, batch_shape=[bs])
        return x + I

class DentalPointNet(keras.Model):
    def __init__(self, num_classes, base_channels=64, dropout=0.5, activation="relu"):
        super().__init__()
        C = int(base_channels)
        act = activation
        self.tnet1 = TNet(K=3, activation=act)
        self.conv1 = layers.Conv1D(C,   1, activation=act)
        self.conv2 = layers.Conv1D(C,   1, activation=act)
        self.tnet2 = TNet(K=C, activation=act)
        self.conv3 = layers.Conv1D(C,   1, activation=act)
        self.conv4 = layers.Conv1D(2*C, 1, activation=act)
        self.conv5 = layers.Conv1D(16*C,1, activation=act)
        self.conv6 = layers.Conv1D(8*C, 1, activation=act)
        self.dp6   = layers.Dropout(dropout)
        self.conv7 = layers.Conv1D(4*C, 1, activation=act)
        self.dp7   = layers.Dropout(dropout)
        self.conv8 = layers.Conv1D(2*C, 1, activation=act)
        self.conv9 = layers.Conv1D(num_classes, 1, activation='softmax')
    def call(self, inputs, training=False):
        t1 = self.tnet1(inputs, training=training)             # (B,3,3)
        x  = tf.matmul(inputs, t1)                             # (B,P,3)
        x  = self.conv1(x)
        x  = self.conv2(x)
        t2 = self.tnet2(x, training=training)                  # (B,C,C)
        x  = tf.matmul(x, t2)
        x  = self.conv3(x)
        x  = self.conv4(x)
        x  = self.conv5(x)
        g  = layers.GlobalMaxPooling1D()(x)
        g  = tf.expand_dims(g, axis=1)
        g  = tf.tile(g, [1, tf.shape(inputs)[1], 1])
        x  = tf.concat([x, g], axis=-1)
        x  = self.conv6(x); x = self.dp6(x, training=training)
        x  = self.conv7(x); x = self.dp7(x, training=training)
        x  = self.conv8(x)
        x  = self.conv9(x)
        return x

# ============================================================
# Optimizador
# ============================================================
def get_optimizer(args):
    if args.optimizer.lower() == "adam":
        try:
            return keras.optimizers.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay)
        except Exception:
            return keras.optimizers.Adam(learning_rate=args.lr)
    elif args.optimizer.lower() == "sgd":
        return keras.optimizers.SGD(learning_rate=args.lr, momentum=0.9, nesterov=True)
    else:
        raise ValueError(f"Optimizer no soportado: {args.optimizer}")

# ============================================================
# Auxiliares guardado
# ============================================================
def save_model_summary(model, run_dir: Path):
    txt = []
    model.summary(print_fn=lambda s: txt.append(s))
    (run_dir/"model_summary.txt").write_text("\n".join(txt), encoding="utf-8")
    try:
        keras.utils.plot_model(model, to_file=str(run_dir/"model.png"),
                               show_shapes=True, expand_nested=True)
    except Exception as e:
        (run_dir/"model_plot_error.txt").write_text(str(e), encoding="utf-8")

def save_history(history, run_dir: Path):
    hist = {k: [float(x) for x in v] for k, v in history.history.items()}
    (run_dir/"history.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")

# ============================================================
# Entrenamiento
# ============================================================
def train_once(args, run_dir: Path, hparams: dict):
    # 🔧 Semillas para determinismo de tf.random en augment + tf.data
    tf.keras.utils.set_random_seed(args.seed)

    ds_tr, ds_va, ds_te, dinfo = make_datasets(
        args.data_path, hparams["batch_size"], args.seed,
        do_augment=args.augment,
        rot_deg=args.rot_deg, jitter_std=args.jitter_std,
        scale_low=args.scale_low, scale_high=args.scale_high
    )
    num_classes = dinfo["num_classes"]

    # Class Weights (opcional)
    cw = None
    try:
        if args.class_weights_json:
            with open(args.class_weights_json, "r") as f:
                cw = json.load(f)
        elif args.infer_class_weights:
            y_path = Path(args.data_path) / "Y_train.npz"
            if y_path.exists():
                ytr = np.load(y_path)["Y"].reshape(-1)
                u, c = np.unique(ytr, return_counts=True)
                w = 1.0 / (c + 1e-6)
                w = w / w.mean()
                cw = {int(i): float(v) for i, v in zip(u, w)}
                (run_dir/"class_weights.json").write_text(json.dumps(cw, indent=2), encoding="utf-8")
            else:
                print(f"[WARN] No existe {y_path}; no se pudo inferir class weights.")
    except Exception as e:
        print("[WARN] Error preparando class weights:", e)
        cw = None

    # Smoke
    if args.smoke:
        ds_tr = ds_tr.take(args.smoke_batches)
        ds_va = ds_va.take(max(1, args.smoke_batches // 2))
        hparams["epochs"] = 1
        print(f"[SMOKE] epochs={hparams['epochs']} train_batches={args.smoke_batches}")

    # Construcción + compile
    def compile_with_metrics(m):
        metrics = ['accuracy']
        if args.metrics_macro:
            metrics += [
                PrecisionMacro(num_classes=num_classes, name="prec_macro"),
                RecallMacro(num_classes=num_classes,    name="rec_macro"),
                F1Macro(num_classes=num_classes,        name="f1_macro"),
                SparseMeanIoU(num_classes=num_classes,  name="miou"),
            ]
        opt  = get_optimizer(args)
        # Pérdida
        if args.focal:
            loss = FocalSparseCE(gamma=2.0,
                                 alpha_vec=cw if cw is not None else None,
                                 num_classes=num_classes)
        elif cw is not None:
            loss = WeightedSparseCE(class_weights=cw, num_classes=num_classes)
        else:
            loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        m.compile(optimizer=opt, loss=loss, metrics=metrics)
        return m

    strategy = setup_devices(args.cuda, args.multi_gpu)
    if strategy:
        with strategy.scope():
            model = DentalPointNet(
                num_classes=num_classes,
                base_channels=hparams["base_channels"],
                dropout=hparams["dropout"],
                activation=args.activation
            )
            model = compile_with_metrics(model)
    else:
        model = DentalPointNet(
            num_classes=num_classes,
            base_channels=hparams["base_channels"],
            dropout=hparams["dropout"],
            activation=args.activation
        )
        model = compile_with_metrics(model)

    # Build antes del summary (para shapes)
    # Tomamos P del dataset
    spec = ds_tr.element_spec[0].shape
    P = int(spec[1]) if spec.rank is not None and spec[1] is not None else None
    if P is None:
        _xb, _yb = next(iter(ds_tr))
        P = int(_xb.shape[1])
    _ = model(tf.zeros([1, P, 3], dtype=tf.float32), training=False)

    # Guardados base
    save_model_summary(model, run_dir)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=args.lr_factor,
                                          patience=args.lr_patience, min_lr=args.min_lr, verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.es_patience,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(filepath=str(ckpt_dir/"best"),
                                        save_best_only=True, monitor='val_loss', save_format='tf', verbose=1),
        keras.callbacks.CSVLogger(str(run_dir/"train_log.csv")),
        keras.callbacks.TensorBoard(log_dir=str(run_dir/"tb")),
    ]
    (run_dir/"config.json").write_text(json.dumps({"hparams": hparams, "args": vars(args)}, indent=2), encoding="utf-8")

    # Fit
    print("[HPARAMS]", hparams)
    history = model.fit(
        ds_tr, validation_data=ds_va,
        epochs=hparams["epochs"], verbose=1, callbacks=callbacks
    )
    save_history(history, run_dir)

    # Evaluación
    print("[EVAL] Test:")
    test_metrics = model.evaluate(ds_te, verbose=1, return_dict=True)
    (run_dir/"test_metrics.json").write_text(
        json.dumps({k: float(v) for k, v in test_metrics.items()}, indent=2), encoding="utf-8"
    )

    # Guardado final
    final_dir = run_dir / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save(final_dir, save_format='tf')
    print(f"[FIN] Modelo guardado en: {final_dir}")

# ============================================================
# CLI
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True,
                    help="Carpeta con X_*.npz / Y_*.npz. Ej: data/data_teeth3ds/splits/8192")
    ap.add_argument("--out_dir",   required=True,
                    help="Carpeta base de salidas (p.ej. runs)")
    ap.add_argument("--tag",       required=True,
                    help="Subcarpeta del experimento (p.ej. pointnet_8192_s42)")

    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--activation", default="relu")
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--base_channels", type=int, default=64)

    ap.add_argument("--optimizer", default="adam", choices=["adam","sgd"])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--lr_patience", type=int, default=5)
    ap.add_argument("--lr_factor", type=float, default=0.5)
    ap.add_argument("--min_lr", type=float, default=1e-5)
    ap.add_argument("--es_patience", type=int, default=10)

    ap.add_argument("--cuda", default=None, help="Ej: 0 o 1; si None, respeta CUDA_VISIBLE_DEVICES")
    ap.add_argument("--multi_gpu", action="store_true")

    ap.add_argument("--metrics_macro", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="1 epoch y pocos batches")
    ap.add_argument("--smoke_batches", type=int, default=2)
    ap.add_argument("--grid", action="store_true", help="(no usado aquí)")

    # Pesos / Focal
    ap.add_argument("--class_weights_json", type=str, default=None,
                    help="Ruta a JSON con pesos por clase {label: weight}.")
    ap.add_argument("--infer_class_weights", action="store_true",
                    help="Inferir pesos desde Y_train.npz si no se pasan por JSON.")
    ap.add_argument("--focal", action="store_true",
                    help="Usar Focal Loss (gamma=2) con alpha=class_weights si hay.")

    # Augment (suave, por defecto ON)
    ap.add_argument("--augment", dest="augment", action="store_true", default=True,
                    help="Aplicar augmentation suave en train.")
    ap.add_argument("--no-augment", dest="augment", action="store_false",
                    help="Desactivar augmentation en train.")
    ap.add_argument("--rot_deg", type=float, default=10.0, help="Rotación ±grados alrededor de Z.")
    ap.add_argument("--jitter_std", type=float, default=0.005, help="Std del jitter gaussiano.")
    ap.add_argument("--scale_low", type=float, default=0.90, help="Escala mínima.")
    ap.add_argument("--scale_high", type=float, default=1.10, help="Escala máxima.")

    args = ap.parse_args()

    out_base = Path(args.out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    run_dir = out_base / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)

    hp = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "dropout": args.dropout,
        "base_channels": args.base_channels,
    }
    train_once(args, run_dir, hp)

if __name__ == "__main__":
    main()
